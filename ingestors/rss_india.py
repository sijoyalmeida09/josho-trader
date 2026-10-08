"""rss_india.py - Layer-4 ingestor for Indian financial-news RSS feeds.

Owner: Agent 3. Spec: SITUATIONAL_SPEC.md sections 1, 3, 4, 10, 12.
Pulls 6 RSS feeds via feedparser, normalizes each item to the Layer-4 raw-event
shape, and attaches a lightweight regex-based entities_hint so the classifier
(Agent 8) has structured signals. FREE sources, failsoft, no DB writes,
no classification beyond entities_hint, no imports from other agents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

try:
    import feedparser  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "feedparser required. pip install 'feedparser>=6.0.0'"
    ) from exc


SOURCE_NAME = "rss_india"
USER_AGENT = "JoshoTrader-LayerFour/0.1 (+https://joshoit.com)"
INTER_FEED_SLEEP_SEC = 0.5
DEFAULT_LOOKBACK_MIN = 90

# (feed_name, feed_url). Order is intentional: highest reliability first.
FEEDS: tuple[tuple[str, str], ...] = (
    ("Moneycontrol", "https://www.moneycontrol.com/rss/MCtopnews.xml"),
    ("EconomicTimes", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("BusinessStandard", "https://www.business-standard.com/rss/markets-106.rss"),
    ("LiveMint", "https://www.livemint.com/rss/markets"),
    ("ReutersIndia", "https://feeds.reuters.com/reuters/INbusinessNews"),
    ("HinduBusinessLine", "https://www.thehindubusinessline.com/feeder/default.rss"),
)

# Inlined NSE ticker watchlist (32 names). Case-insensitive, word-boundary
# protected. Nifty heavyweights + commodity/defense/PSU names recurring in
# geopolitical news.
NSE_TICKERS: tuple[str, ...] = (
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN",
    "ONGC", "BPCL", "HPCL", "IOC", "TATASTEEL", "HINDALCO",
    "JSWSTEEL", "COALINDIA", "UPL", "BHARTIARTL", "HAL", "BEL",
    "BDL", "BAJFINANCE", "AXISBANK", "ITC", "ADANIPOWER",
    "TATAPOWER", "SUZLON", "PNB", "BANKBARODA", "SAIL", "VEDL",
    "WIPRO", "NIFTY", "BANKNIFTY",
)

COMMODITY_KEYWORDS: tuple[str, ...] = (
    "oil", "crude", "gold", "silver", "copper", "steel",
    "aluminium", "aluminum", "wheat", "sugar", "cotton",
)

LOCATION_KEYWORDS: tuple[str, ...] = (
    "India", "Maharashtra", "Gujarat", "Karnataka", "Tamil Nadu",
    "Kerala", "Punjab", "Haryana", "Rajasthan", "Uttar Pradesh",
    "West Bengal", "Madhya Pradesh", "Andhra Pradesh", "Telangana",
    "Odisha", "Bihar", "Mumbai", "Delhi", "Bengaluru", "Chennai",
    "Kolkata", "Iran", "Israel", "China", "Russia", "Saudi Arabia",
    "Pakistan", "Bangladesh", "Sri Lanka",
)

THEME_KEYWORDS: tuple[str, ...] = (
    "rate", "monsoon", "cyclone", "sanction", "defense", "defence",
    "M&A", "merger", "acquisition", "earnings", "budget",
    "inflation", "RBI", "FOMC", "policy", "tariff", "export ban",
)


def _mk(words: Iterable[str]) -> re.Pattern[str]:
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)


_TICKER_REGEX = _mk(NSE_TICKERS)
_COMMODITY_REGEX = _mk(COMMODITY_KEYWORDS)
_LOCATION_REGEX = _mk(LOCATION_KEYWORDS)
_THEME_REGEX = _mk(THEME_KEYWORDS)
_HTML_TAG_REGEX = re.compile(r"<[^>]+>")

_TICKER_CANON = {t.lower(): t.upper() for t in NSE_TICKERS}
_COMMODITY_CANON = {c.lower(): c.lower() for c in COMMODITY_KEYWORDS}
_LOCATION_CANON = {loc.lower(): loc for loc in LOCATION_KEYWORDS}
_THEME_CANON = {th.lower(): th.lower() for th in THEME_KEYWORDS}

logger = logging.getLogger("rss_india")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s rss_india: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class FeedSpec:
    name: str
    url: str


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _strip_html(text: str) -> str:
    if not text:
        return ""
    return _HTML_TAG_REGEX.sub("", text).strip()


def _to_utc_iso(struct_time: Any) -> str | None:
    if not struct_time:
        return None
    try:
        return datetime(*struct_time[:6], tzinfo=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def _make_event_id(link: str, published: str) -> str:
    raw = f"{link or ''}|{published or ''}".encode("utf-8", errors="ignore")
    return f"{SOURCE_NAME}:" + hashlib.sha1(raw).hexdigest()[:16]


def _collect(regex: re.Pattern[str], canon: dict[str, str], text: str) -> list[str]:
    seen: list[str] = []
    for match in regex.finditer(text):
        value = canon.get(match.group(1).lower(), match.group(1))
        if value not in seen:
            seen.append(value)
    return seen


def _extract_entities(text: str) -> dict[str, list[str]]:
    haystack = text or ""
    return {
        "organizations": _collect(_TICKER_REGEX, _TICKER_CANON, haystack),
        "commodities": _collect(_COMMODITY_REGEX, _COMMODITY_CANON, haystack),
        "locations": _collect(_LOCATION_REGEX, _LOCATION_CANON, haystack),
        "themes": _collect(_THEME_REGEX, _THEME_CANON, haystack),
    }


def _has_any_signal(entities: dict[str, list[str]]) -> bool:
    return bool(entities.get("organizations")) or bool(entities.get("themes"))


def _item_timestamp(entry: Any) -> str | None:
    for key in ("published_parsed", "updated_parsed"):
        struct = getattr(entry, key, None)
        if struct is None and isinstance(entry, dict):
            struct = entry.get(key)
        iso = _to_utc_iso(struct)
        if iso:
            return iso
    return None


def _fetch_feed(spec: FeedSpec) -> Any | None:
    feedparser.USER_AGENT = USER_AGENT
    try:
        parsed = feedparser.parse(spec.url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("feed %s exception: %s", spec.name, exc)
        return None

    status = getattr(parsed, "status", None)
    bozo = getattr(parsed, "bozo", 0)
    if bozo and not getattr(parsed, "entries", None):
        logger.warning("feed %s malformed (bozo=%s status=%s)", spec.name, bozo, status)
        return None
    if status and status >= 400:
        logger.warning("feed %s http %s", spec.name, status)
        return None
    return parsed


def _normalize_entry(entry: Any, spec: FeedSpec) -> dict[str, Any] | None:
    title = (getattr(entry, "title", None) or "").strip()
    link = (getattr(entry, "link", None) or "").strip()
    if not title or not link:
        return None

    ts_iso = _item_timestamp(entry)
    summary = _strip_html(getattr(entry, "summary", "") or getattr(entry, "description", "") or "")
    author = getattr(entry, "author", None)
    tags: list[str] = []
    for t in getattr(entry, "tags", None) or []:
        term = getattr(t, "term", None) or (t.get("term") if isinstance(t, dict) else None)
        if term:
            tags.append(str(term))

    raw_payload: dict[str, Any] = {"feed_url": spec.url, "feed_name": spec.name, "summary": summary}
    if author:
        raw_payload["author"] = str(author)
    if tags:
        raw_payload["tags"] = tags

    return {
        "event_id": _make_event_id(link, ts_iso or ""),
        "headline": title,
        "url": link,
        "timestamp": ts_iso,
        "source": SOURCE_NAME,
        "raw_payload": raw_payload,
        "entities_hint": _extract_entities(f"{title}\n{summary}"),
    }


def poll(since: datetime | None = None) -> list[dict]:
    """Poll all 6 RSS feeds and return normalized raw events.

    Returns list of dicts: {event_id, source, headline, url, timestamp,
    raw_payload, entities_hint}. Items older than `since` (default 90min ago)
    are filtered. Items with no organizations AND no themes are filtered.
    """
    cutoff = since if since is not None else _now_utc() - timedelta(minutes=DEFAULT_LOOKBACK_MIN)
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)

    seen_ids: set[str] = set()
    events: list[dict] = []

    for idx, (name, url) in enumerate(FEEDS):
        spec = FeedSpec(name=name, url=url)
        parsed = _fetch_feed(spec)
        if parsed is None:
            if idx < len(FEEDS) - 1:
                time.sleep(INTER_FEED_SLEEP_SEC)
            continue

        kept = 0
        for entry in getattr(parsed, "entries", []) or []:
            try:
                evt = _normalize_entry(entry, spec)
            except Exception as exc:  # noqa: BLE001
                logger.warning("feed %s entry skip: %s", spec.name, exc)
                continue
            if evt is None:
                continue

            ts_iso = evt.get("timestamp")
            if ts_iso:
                try:
                    if datetime.fromisoformat(ts_iso) < cutoff:
                        continue
                except ValueError:
                    pass

            if not _has_any_signal(evt["entities_hint"]):
                continue

            eid = evt["event_id"]
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            events.append(evt)
            kept += 1

        logger.info(
            "feed %s parsed %d entries, kept %d",
            spec.name,
            len(getattr(parsed, "entries", []) or []),
            kept,
        )
        if idx < len(FEEDS) - 1:
            time.sleep(INTER_FEED_SLEEP_SEC)

    return events


def poll_once_cli() -> int:
    """argparse entry: --since=ISO, --dry-run. Writes JSON lines to stdout."""
    parser = argparse.ArgumentParser(
        prog="rss_india",
        description="Poll Indian financial-news RSS feeds (Layer-4 Agent 3).",
    )
    parser.add_argument("--since", type=str, default=None, help="ISO-8601 lower bound.")
    parser.add_argument("--dry-run", action="store_true", help="Stdout only (default).")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging.")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    since: datetime | None = None
    if args.since:
        try:
            since = datetime.fromisoformat(args.since)
        except ValueError:
            logger.error("invalid --since: %r", args.since)
            return 2

    try:
        events = poll(since=since)
    except Exception as exc:  # noqa: BLE001
        logger.error("poll failed: %s", exc)
        return 1

    for evt in events:
        sys.stdout.write(json.dumps(evt, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()
    logger.info("rss_india emitted %d events", len(events))
    return 0


def _smoke_test() -> int:
    """Offline-tolerant smoke: poll once, log a summary, never raise."""
    logger.info("smoke: starting rss_india poll (lookback %dmin)", DEFAULT_LOOKBACK_MIN)
    try:
        events = poll()
    except Exception as exc:  # noqa: BLE001
        logger.error("smoke: poll raised %s", exc)
        return 1

    logger.info("smoke: total events kept = %d", len(events))
    if events:
        sample = events[0]
        logger.info("smoke: sample event_id=%s", sample["event_id"])
        logger.info("smoke: sample headline=%r", sample["headline"][:90])
        logger.info("smoke: sample entities_hint=%s", sample["entities_hint"])
        sys.stdout.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")
    else:
        logger.info("smoke: no events (network down or all filtered out)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        sys.exit(_smoke_test())
    sys.exit(poll_once_cli())

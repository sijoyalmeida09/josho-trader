"""GDELT GKG ingestor (Agent 2, Layer 4).

Polls the GDELT 2.0 DOC API filtered by GKG theme codes that map to event
types in SITUATIONAL_SPEC.md sec.2. Pure ingestion. Failsoft per sec.12.
Endpoint: https://api.gdeltproject.org/api/v2/doc/doc?query=theme:<CODE>&...
Reference: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/ — the
DOC 2.0 API exposes the GKG theme filter via the `theme:` operator. The
raw v2 GKG endpoint (gkg/gkg) only ships 15-min CSV files, so DOC is the
supported JSON path for theme queries without auth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT = "Sijoy-Trader-Layer4/1.0 (+research; contact via repo)"
REQUEST_TIMEOUT = 20
MAX_RECORDS_PER_THEME = 15
INTER_REQUEST_SLEEP = 1.0
DEFAULT_LOOKBACK_MIN = 60
MAX_BACKOFF_ATTEMPTS = 3

# Theme code -> event_type hint (Classifier (Agent 8) makes the final call).
THEME_MAP: dict[str, str] = {
    "ECON_PRICE_OIL": "oil_shock",
    "ENV_CLIMATE_CHANGE": "monsoon_cyclone",
    "NATURAL_DISASTER": "flood_quake",
    "ECON_BANKRUPTCY": "corporate_action",
    "ECON_TRADE": "policy_shift",
    "GENERAL_GOVERNMENT": "policy_shift",
    "MILITARY": "defense_flashpoint",
    "TAX_FNCACT_INTEREST_RATE": "rate_decision",
}

# Indian-relevance tokens (substring match, lowercased).
INDIA_TOKENS = frozenset(
    "india bharat delhi mumbai kolkata chennai bengaluru bangalore hyderabad "
    "pune ahmedabad jaipur lucknow kanpur nagpur indore bhopal patna ranchi "
    "kochi guwahati chandigarh gurugram noida andhra assam bihar gujarat "
    "haryana karnataka kerala maharashtra odisha punjab rajasthan telangana "
    "kashmir ladakh".split() + ["new delhi", "tamil nadu", "uttar pradesh", "west bengal"]
)


def _http_get(params: dict[str, Any]) -> dict[str, Any] | None:
    """GET with exponential backoff. Returns parsed JSON or None on failure."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    delay = 1.0
    for attempt in range(1, MAX_BACKOFF_ATTEMPTS + 1):
        try:
            resp = requests.get(
                GDELT_DOC_URL, params=params, headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            logger.warning("gdelt_gkg request error attempt=%d err=%s", attempt, exc)
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code == 429:
            logger.warning("gdelt_gkg 429 attempt=%d backoff=%.1fs", attempt, delay * 2)
            time.sleep(delay * 2)
            delay *= 2
            continue
        if 500 <= resp.status_code < 600:
            logger.warning("gdelt_gkg 5xx status=%d attempt=%d", resp.status_code, attempt)
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code != 200:
            logger.warning(
                "gdelt_gkg non-200 status=%d body=%s",
                resp.status_code, resp.text[:200],
            )
            return None
        text = resp.text.strip()
        if not text:
            return {"articles": []}
        try:
            return resp.json()
        except json.JSONDecodeError:
            logger.warning("gdelt_gkg JSON decode failure body=%s", text[:200])
            return None
    logger.error("gdelt_gkg gave up after %d attempts", MAX_BACKOFF_ATTEMPTS)
    return None


def _parse_seendate(seendate: str) -> datetime | None:
    """GDELT seendate format: YYYYMMDDTHHMMSSZ."""
    if not seendate:
        return None
    try:
        return datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None

def _format_gdelt_datetime(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")

def _compute_event_id(url: str, seendate: str) -> str:
    digest = hashlib.sha1(f"{url}|{seendate}".encode("utf-8")).hexdigest()[:16]
    return f"gdelt_gkg:{digest}"

def _extract_locations(article: dict[str, Any]) -> list[str]:
    out: list[str] = []
    country = article.get("sourcecountry")
    if isinstance(country, str) and country.strip():
        out.append(country.strip())
    raw = article.get("locations")
    if isinstance(raw, str) and raw:
        for tok in raw.split(";"):
            tok = tok.strip()
            if tok and tok not in out:
                out.append(tok)
    return out

def _extract_organizations(article: dict[str, Any]) -> list[str]:
    raw = article.get("organizations")
    if not isinstance(raw, str) or not raw:
        return []
    return [t.strip() for t in raw.split(";") if t.strip()]

def _is_india_relevant(locations: list[str], domain: str) -> bool:
    if domain and domain.lower().endswith(".in"):
        return True
    for loc in locations:
        ll = loc.lower()
        for tok in INDIA_TOKENS:
            if tok in ll:
                return True
    return False


def _normalize_article(
    article: dict[str, Any], theme_code: str,
) -> dict[str, Any] | None:
    url = article.get("url")
    if not isinstance(url, str) or not url.strip():
        return None
    seendate = article.get("seendate", "") or ""
    headline = article.get("title") or ""
    if not isinstance(headline, str):
        headline = str(headline)
    ts = _parse_seendate(seendate) or datetime.now(timezone.utc)
    domain = article.get("domain") or ""
    if not isinstance(domain, str):
        domain = str(domain)
    locations = _extract_locations(article)
    organizations = _extract_organizations(article)
    tone_raw = article.get("tone")
    try:
        tone: float | None = float(tone_raw) if tone_raw is not None else None
    except (TypeError, ValueError):
        tone = None
    india_relevant = _is_india_relevant(locations, domain)
    raw_payload: dict[str, Any] = {
        "theme": theme_code,
        "tone": tone,
        "sourcecollectionidentifier": article.get("sourcecollectionidentifier"),
        "domain": domain,
        "locations": locations,
        "persons": [],
        "organizations": organizations,
        "language": article.get("language"),
        "socialimage": article.get("socialimage"),
        "india_relevant": india_relevant,
    }
    entities_hint: dict[str, Any] = {
        "themes": [theme_code],
        "locations": locations,
        "organizations": organizations,
    }
    return {
        "event_id": _compute_event_id(url, seendate),
        "source": "gdelt_gkg",
        "headline": headline.strip(),
        "url": url,
        "timestamp": ts.isoformat(),
        "raw_payload": raw_payload,
        "entities_hint": entities_hint,
        "_theme_match": theme_code,
    }


def _poll_theme(theme_code: str, since: datetime) -> list[dict[str, Any]]:
    params = {
        "query": f"theme:{theme_code}",
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(MAX_RECORDS_PER_THEME),
        "sort": "DateDesc",
        "startdatetime": _format_gdelt_datetime(since),
        "enddatetime": _format_gdelt_datetime(datetime.now(timezone.utc)),
    }
    payload = _http_get(params)
    if payload is None:
        return []
    articles = payload.get("articles")
    if not isinstance(articles, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in articles:
        if not isinstance(raw, dict):
            continue
        try:
            ev = _normalize_article(raw, theme_code)
        except Exception as exc:  # noqa: BLE001 — failsoft per sec.12.3
            logger.warning("gdelt_gkg normalize err theme=%s err=%s", theme_code, exc)
            continue
        if ev is None or ev["event_id"] in seen:
            continue
        seen.add(ev["event_id"])
        out.append(ev)
    return out


def poll(since: datetime | None = None) -> list[dict[str, Any]]:
    """Return raw GDELT GKG events filtered by the 8 mapped themes.

    Each dict carries: event_id, source, headline, url, timestamp,
    raw_payload, entities_hint, _theme_match. Pure ingestion; no DB writes.
    """
    if since is None:
        since = datetime.now(timezone.utc) - timedelta(minutes=DEFAULT_LOOKBACK_MIN)
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    aggregated: dict[str, dict[str, Any]] = {}
    themes = list(THEME_MAP.keys())
    for idx, theme_code in enumerate(themes):
        try:
            events = _poll_theme(theme_code, since)
        except Exception as exc:  # noqa: BLE001 — never stall the orchestrator
            logger.error("gdelt_gkg theme=%s fatal err=%s", theme_code, exc)
            events = []
        for ev in events:
            existing = aggregated.get(ev["event_id"])
            if existing is None:
                aggregated[ev["event_id"]] = ev
                continue
            themes_acc = existing["entities_hint"].setdefault("themes", [])
            if theme_code not in themes_acc:
                themes_acc.append(theme_code)
            existing["raw_payload"]["theme"] = ",".join(sorted(set(themes_acc)))
        if idx < len(themes) - 1:
            time.sleep(INTER_REQUEST_SLEEP)
    logger.info(
        "gdelt_gkg poll done themes=%d unique=%d", len(themes), len(aggregated),
    )
    return list(aggregated.values())


def poll_once_cli() -> int:
    parser = argparse.ArgumentParser(
        prog="gdelt_gkg",
        description="Poll GDELT GKG themes; emit events as JSON lines.",
    )
    parser.add_argument("--since", type=str, default=None,
                        help="ISO timestamp lower bound; default = now-60min")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print events only (no DB writes anyway).")
    parser.add_argument("--verbose", action="store_true",
                        help="DEBUG logging to stderr")
    args = parser.parse_args()
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    since: datetime | None = None
    if args.since:
        try:
            since = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
        except ValueError:
            logger.error("invalid --since value; use ISO8601")
            return 2
    try:
        events = poll(since=since)
    except Exception as exc:  # noqa: BLE001
        logger.error("gdelt_gkg poll fatal err=%s", exc)
        return 1
    for ev in events:
        sys.stdout.write(json.dumps(ev, default=str, ensure_ascii=True) + "\n")
    sys.stdout.flush()
    logger.info("gdelt_gkg cli emitted=%d dry=%s", len(events), args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(poll_once_cli())

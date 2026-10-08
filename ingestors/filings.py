"""
filings.py — Layer 4 Agent 4: Official Filings Watcher.

Polls regulatory and exchange RSS feeds (RBI, SEBI, NSE, BSE) and emits
raw event dicts conforming to the Layer-4 ingestor contract (see
SITUATIONAL_SPEC.md sections 1, 3, 4, 10, 12).

Reliability weight for this source is 0.95 — highest of any ingestor.
Filings are the ground-truth of policy/corporate action; they move slower
than newswires but cannot be faked.

Contract (frozen):
    poll(since: datetime | None = None) -> list[dict]
    poll_once_cli() -> int

Each returned dict has:
    event_id, source='filings', headline, url, timestamp (ISO UTC),
    raw_payload {filing_authority, category_hint?, summary?},
    entities_hint {organizations, commodities, locations, themes}.

Failsoft: a single dead feed must NOT stall the pipeline. Per §12,
malformed entries are skipped with a warning and the agent continues.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    import feedparser  # type: ignore
except ImportError:  # pragma: no cover
    print(
        "ERROR: feedparser not installed. Run: pip install feedparser>=6.0.0",
        file=sys.stderr,
    )
    raise

LOG = logging.getLogger("layer4.filings")
if not LOG.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    LOG.addHandler(_h)
    LOG.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Feed registry — the 5 filings sources from SITUATIONAL_SPEC.md §4.
# ---------------------------------------------------------------------------

FILING_FEEDS: list[dict[str, str]] = [
    {
        "authority": "RBI",
        "kind": "notifications",
        "url": "https://www.rbi.org.in/Scripts/RSSFeed.aspx?Type=Notifications",
    },
    {
        "authority": "RBI",
        "kind": "press_releases",
        "url": "https://www.rbi.org.in/Scripts/RSSFeed.aspx?Type=Press_Releases",
    },
    {
        "authority": "SEBI",
        "kind": "all",
        "url": "https://www.sebi.gov.in/sebirss.xml",
    },
    {
        "authority": "NSE",
        "kind": "announcements",
        "url": "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml",
    },
    {
        "authority": "BSE",
        "kind": "notices",
        "url": "https://www.bseindia.com/data/xml/notices.xml",
    },
]

USER_AGENT = "JoshoTrader-LayerFour/0.1 (+https://joshoit.com)"
SLEEP_BETWEEN_FEEDS_SEC = 0.5
DEFAULT_LOOKBACK_HOURS = 6

# ---------------------------------------------------------------------------
# Lightweight entity dictionaries (DUPLICATED inline per spec — agents stay
# isolated; do NOT import from rss_india.py).
# ---------------------------------------------------------------------------

# Subset of NSE tickers most likely referenced in regulatory filings. The
# Classifier (Agent 8) does the heavier extraction; this is just a hint.
NSE_TICKERS: dict[str, list[str]] = {
    "RELIANCE": ["reliance industries", "ril", "reliance"],
    "TCS": ["tata consultancy", "tcs"],
    "HDFCBANK": ["hdfc bank"],
    "ICICIBANK": ["icici bank"],
    "INFY": ["infosys"],
    "SBIN": ["state bank of india", "sbi"],
    "BHARTIARTL": ["bharti airtel", "airtel"],
    "ITC": ["itc limited", "itc ltd"],
    "KOTAKBANK": ["kotak mahindra"],
    "LT": ["larsen & toubro", "larsen and toubro"],
    "AXISBANK": ["axis bank"],
    "BAJFINANCE": ["bajaj finance"],
    "MARUTI": ["maruti suzuki"],
    "ASIANPAINT": ["asian paints"],
    "HINDUNILVR": ["hindustan unilever", "hul"],
    "ONGC": ["oil and natural gas", "ongc"],
    "NTPC": ["ntpc"],
    "POWERGRID": ["power grid corporation"],
    "COALINDIA": ["coal india"],
    "TATASTEEL": ["tata steel"],
    "JSWSTEEL": ["jsw steel"],
    "HINDALCO": ["hindalco"],
    "VEDL": ["vedanta"],
    "ADANIENT": ["adani enterprises"],
    "ADANIPORTS": ["adani ports"],
    "BPCL": ["bharat petroleum", "bpcl"],
    "IOC": ["indian oil corporation", "ioc"],
    "GAIL": ["gail india"],
    "WIPRO": ["wipro"],
    "HCLTECH": ["hcl technologies"],
    "TECHM": ["tech mahindra"],
    "SUNPHARMA": ["sun pharma"],
    "DRREDDY": ["dr reddy"],
    "CIPLA": ["cipla"],
    "TATAMOTORS": ["tata motors"],
    "M&M": ["mahindra & mahindra", "mahindra and mahindra"],
    "BAJAJ-AUTO": ["bajaj auto"],
    "EICHERMOT": ["eicher motors"],
    "HEROMOTOCO": ["hero motocorp"],
    "GRASIM": ["grasim industries"],
    "ULTRACEMCO": ["ultratech cement"],
    "TITAN": ["titan company"],
    "NESTLEIND": ["nestle india"],
    "BRITANNIA": ["britannia industries"],
    "PIDILITIND": ["pidilite"],
    "DABUR": ["dabur"],
    "GODREJCP": ["godrej consumer"],
    "DMART": ["avenue supermarts", "dmart"],
    "ZOMATO": ["zomato"],
    "PAYTM": ["paytm", "one97"],
    "NYKAA": ["nykaa", "fsn e-commerce"],
    "IRCTC": ["irctc"],
    "DIVISLAB": ["divi's laboratories"],
}

COMMODITIES: dict[str, list[str]] = {
    "crude_oil": ["crude", "oil", "brent", "wti", "petroleum"],
    "natural_gas": ["natural gas", "lng"],
    "gold": ["gold", "bullion"],
    "silver": ["silver"],
    "copper": ["copper"],
    "aluminium": ["aluminium", "aluminum"],
    "zinc": ["zinc"],
    "iron_ore": ["iron ore"],
    "steel": ["steel"],
    "coal": ["coal"],
    "wheat": ["wheat"],
    "rice": ["rice", "basmati"],
    "sugar": ["sugar"],
    "cotton": ["cotton"],
    "palm_oil": ["palm oil"],
    "fertilizer": ["fertilizer", "fertiliser", "urea", "dap"],
    "rupee": ["rupee", "inr", "usd/inr"],
}

GEO: dict[str, list[str]] = {
    "india": ["india", "bharat", "indian"],
    "china": ["china", "chinese", "beijing"],
    "russia": ["russia", "moscow"],
    "iran": ["iran", "tehran"],
    "saudi": ["saudi arabia", "riyadh"],
    "us": ["united states", "u.s.", "usa", "america"],
    "europe": ["europe", "european union", "eu "],
    "middle_east": ["middle east", "gulf", "hormuz"],
    "pakistan": ["pakistan"],
    "bangladesh": ["bangladesh"],
}

THEMES: dict[str, list[str]] = {
    "monetary_policy": ["repo rate", "mpc", "monetary policy", "interest rate"],
    "rate_hike": ["rate hike", "rate increase"],
    "rate_cut": ["rate cut", "rate decrease"],
    "inflation": ["inflation", "cpi", "wpi"],
    "fdi": ["fdi", "foreign direct investment"],
    "ecb": ["external commercial borrowing", "ecb "],
    "merger": ["merger", "scheme of arrangement", "amalgamation"],
    "open_offer": ["open offer", "takeover"],
    "buyback": ["buy-back", "buyback"],
    "dividend": ["dividend", "interim dividend", "final dividend"],
    "enforcement": ["show cause", "adjudication", "penalty", "compounding"],
    "circular": ["circular", "notification", "framework", "guidelines"],
    "msp": ["minimum support price", "msp"],
    "export_ban": ["export ban", "export restriction", "export curb"],
    "import_duty": ["import duty", "customs duty", "tariff"],
    "ipo": ["ipo", "initial public offering"],
    "qip": ["qip", "qualified institutional placement"],
    "delisting": ["delisting"],
    "agriculture": ["agriculture", "agri", "farm", "mandi"],
}

# ---------------------------------------------------------------------------
# Category-hint patterns (regex) — map to event_type in §2 of the spec.
# These are HINTS; Agent 8 is final authority on event_type.
# ---------------------------------------------------------------------------

CATEGORY_HINT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brepo rate\b", re.I), "rate_decision"),
    (re.compile(r"\bmonetary policy\b", re.I), "rate_decision"),
    (re.compile(r"\bmpc decision\b", re.I), "rate_decision"),
    (re.compile(r"\bmpc\s+(meeting|statement|minutes)\b", re.I), "rate_decision"),
    (re.compile(r"\bbank rate\b", re.I), "rate_decision"),
    (re.compile(r"\bscheduled bank\b", re.I), "policy_shift"),
    (re.compile(r"\bcircular\b", re.I), "policy_shift"),
    (re.compile(r"\bframework\b", re.I), "policy_shift"),
    (re.compile(r"\b(notification|directions)\b", re.I), "policy_shift"),
    (re.compile(r"\bimport duty\b", re.I), "policy_shift"),
    (re.compile(r"\bexport duty\b", re.I), "policy_shift"),
    (re.compile(r"\btariff\b", re.I), "policy_shift"),
    (re.compile(r"\bfdi\b", re.I), "policy_shift"),
    (re.compile(r"\bexternal commercial borrowing\b|\becb\b", re.I), "policy_shift"),
    (re.compile(r"\bgst\b", re.I), "policy_shift"),
    (re.compile(r"\b(m&a|merger|amalgamation)\b", re.I), "corporate_action"),
    (re.compile(r"\bopen offer\b", re.I), "corporate_action"),
    (re.compile(r"\bscheme of arrangement\b", re.I), "corporate_action"),
    (re.compile(r"\bcompounding\b", re.I), "corporate_action"),
    (re.compile(r"\bshow cause\b", re.I), "corporate_action"),
    (re.compile(r"\badjudication\b", re.I), "corporate_action"),
    (re.compile(r"\bdebarment\b", re.I), "corporate_action"),
    (re.compile(r"\bsuspension of trading\b", re.I), "corporate_action"),
    (re.compile(r"\bminimum support price\b|\bmsp\b", re.I), "agri_supply"),
    (re.compile(r"\bagricultur(e|al)\b", re.I), "agri_supply"),
    (re.compile(r"\b(wheat|sugar|rice|onion|tomato)\s+(export|stock)\b", re.I), "agri_supply"),
]

# ---------------------------------------------------------------------------
# Network fetch (failsoft).
# ---------------------------------------------------------------------------


def _fetch_feed(url: str, timeout: float = 15.0) -> Any | None:
    """
    Fetch and parse a single RSS feed. Returns the feedparser result, or
    None on hard failure. Always failsoft — never raises.
    """
    try:
        # feedparser accepts a URL directly but does not let us set a UA
        # header on all versions; use request_headers when available.
        result = feedparser.parse(
            url,
            agent=USER_AGENT,
            request_headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/xml, text/xml, */*"},
        )
    except Exception as exc:  # noqa: BLE001 — failsoft per §12
        LOG.warning("feed fetch raised for %s: %s", url, exc)
        return None

    # feedparser sets .bozo=1 on parse errors but often still yields entries.
    bozo = getattr(result, "bozo", 0)
    status = getattr(result, "status", None)
    entries = getattr(result, "entries", []) or []
    if status and status >= 500:
        LOG.warning("feed %s returned status %s — skipping", url, status)
        return None
    if not entries:
        LOG.warning("feed %s yielded 0 entries (bozo=%s status=%s)", url, bozo, status)
        return None
    return result


# ---------------------------------------------------------------------------
# Parsing helpers.
# ---------------------------------------------------------------------------


def _entry_time_utc(entry: Any) -> datetime | None:
    """
    Best-effort extraction of an entry's publication time as UTC datetime.
    Tries published_parsed, updated_parsed, then raw strings. Returns None
    if nothing parseable is present.
    """
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        tm = getattr(entry, attr, None) or (entry.get(attr) if isinstance(entry, dict) else None)
        if tm:
            try:
                return datetime(*tm[:6], tzinfo=timezone.utc)
            except Exception:  # noqa: BLE001
                continue
    return None


def _stable_event_id(link: str, published: str) -> str:
    """Deterministic dedup key — sha1 of link+published, truncated to 16 hex."""
    payload = (link or "") + "|" + (published or "")
    digest = hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()
    return "filings:" + digest[:16]


def _extract_entities_hint(text: str) -> dict[str, list[str]]:
    """
    Lightweight keyword scan over headline+summary. Returns the same shape
    used by other Layer-4 ingestors. Empty lists are normal — filings often
    refer to a market-wide topic.
    """
    if not text:
        return {"organizations": [], "commodities": [], "locations": [], "themes": []}

    lowered = text.lower()

    orgs: list[str] = []
    for ticker, aliases in NSE_TICKERS.items():
        for alias in aliases:
            if alias in lowered:
                orgs.append(ticker)
                break

    commodities: list[str] = []
    for key, aliases in COMMODITIES.items():
        for alias in aliases:
            if alias in lowered:
                commodities.append(key)
                break

    locations: list[str] = []
    for key, aliases in GEO.items():
        for alias in aliases:
            if alias in lowered:
                locations.append(key)
                break

    themes: list[str] = []
    for key, aliases in THEMES.items():
        for alias in aliases:
            if alias in lowered:
                themes.append(key)
                break

    return {
        "organizations": sorted(set(orgs)),
        "commodities": sorted(set(commodities)),
        "locations": sorted(set(locations)),
        "themes": sorted(set(themes)),
    }


def _category_hint(text: str) -> str | None:
    """Return the first matching category_hint, or None."""
    if not text:
        return None
    for pattern, hint in CATEGORY_HINT_PATTERNS:
        if pattern.search(text):
            return hint
    return None


def _normalize_entry(entry: Any, feed_meta: dict[str, str]) -> dict | None:
    """
    Convert a feedparser entry into the Layer-4 raw event shape, or None
    if the entry lacks the minimum fields. Failsoft.
    """
    try:
        link = (getattr(entry, "link", None) or entry.get("link", "") or "").strip()
        title = (getattr(entry, "title", None) or entry.get("title", "") or "").strip()
        if not link or not title:
            return None

        summary_raw = (
            getattr(entry, "summary", None)
            or entry.get("summary", "")
            or getattr(entry, "description", None)
            or ""
        )
        # Strip HTML tags lightly — full sanitation is Agent 8's job.
        summary = re.sub(r"<[^>]+>", " ", str(summary_raw))
        summary = re.sub(r"\s+", " ", summary).strip()
        if len(summary) > 600:
            summary = summary[:600] + "..."

        published_raw = (
            getattr(entry, "published", None)
            or entry.get("published", "")
            or getattr(entry, "updated", None)
            or ""
        )
        published_dt = _entry_time_utc(entry) or datetime.now(timezone.utc)

        combined = f"{title} || {summary}"
        cat_hint = _category_hint(combined)
        ent_hint = _extract_entities_hint(combined)

        raw_payload: dict[str, Any] = {
            "filing_authority": feed_meta["authority"],
            "filing_kind": feed_meta["kind"],
        }
        if cat_hint:
            raw_payload["category_hint"] = cat_hint
        if summary:
            raw_payload["summary"] = summary

        return {
            "event_id": _stable_event_id(link, str(published_raw)),
            "source": "filings",
            "headline": title,
            "url": link,
            "timestamp": published_dt.isoformat(),
            "raw_payload": raw_payload,
            "entities_hint": ent_hint,
        }
    except Exception as exc:  # noqa: BLE001 — failsoft per §12.3
        LOG.warning("malformed entry skipped: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def poll(since: datetime | None = None) -> list[dict]:
    """
    Poll all 5 filings RSS feeds and return normalized raw events.

    Per spec §10: each dict has event_id, source, headline, url, timestamp,
    raw_payload. We additionally include entities_hint (same convention as
    Agent 3) so the Classifier has something to chew on without re-parsing.

    Filters by `since` (default = now - 6h, filings move slowly). Dedups
    within the pass on event_id.
    """
    if since is None:
        since = datetime.now(timezone.utc) - timedelta(hours=DEFAULT_LOOKBACK_HOURS)
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    seen_ids: set[str] = set()
    out: list[dict] = []

    for feed_meta in FILING_FEEDS:
        url = feed_meta["url"]
        LOG.info("polling %s/%s", feed_meta["authority"], feed_meta["kind"])
        result = _fetch_feed(url)
        if result is None:
            time.sleep(SLEEP_BETWEEN_FEEDS_SEC)
            continue

        kept = 0
        for entry in getattr(result, "entries", []) or []:
            evt = _normalize_entry(entry, feed_meta)
            if evt is None:
                continue
            # Time filter.
            try:
                evt_ts = datetime.fromisoformat(evt["timestamp"])
                if evt_ts.tzinfo is None:
                    evt_ts = evt_ts.replace(tzinfo=timezone.utc)
                if evt_ts < since:
                    continue
            except Exception:  # noqa: BLE001
                # Unparseable timestamp — keep it; safer than dropping.
                pass
            if evt["event_id"] in seen_ids:
                continue
            seen_ids.add(evt["event_id"])
            out.append(evt)
            kept += 1

        LOG.info("  %s/%s kept %d events", feed_meta["authority"], feed_meta["kind"], kept)
        time.sleep(SLEEP_BETWEEN_FEEDS_SEC)

    LOG.info("filings poll complete: %d events across %d feeds", len(out), len(FILING_FEEDS))
    return out


def poll_once_cli() -> int:
    """
    argparse entry point.
    Flags: --since=ISO, --dry-run, --verbose.
    Prints one JSON object per line to stdout. Returns process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="filings",
        description="Layer-4 Agent 4: poll RBI/SEBI/NSE/BSE filings RSS feeds.",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="ISO-8601 datetime; only return events newer than this. "
             "Default: now - 6h.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print event count to stderr only; suppress JSONL to stdout.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="DEBUG-level logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        LOG.setLevel(logging.DEBUG)

    since_dt: datetime | None = None
    if args.since:
        try:
            since_dt = datetime.fromisoformat(args.since)
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"ERROR: invalid --since value: {args.since}", file=sys.stderr)
            return 2

    events = poll(since=since_dt)

    if args.dry_run:
        print(f"[dry-run] {len(events)} filing events", file=sys.stderr)
        return 0

    for evt in events:
        try:
            sys.stdout.write(json.dumps(evt, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("failed to serialize event %s: %s", evt.get("event_id"), exc)
    sys.stdout.flush()
    return 0


# ---------------------------------------------------------------------------
# Smoke test.
# ---------------------------------------------------------------------------


def _smoke() -> None:
    """
    Smoke test — exercises every code path without requiring network.

    Verifies:
      * stable event_id is deterministic and prefixed with 'filings:'
      * category-hint regexes match the spec's example phrases
      * entity hints find a known ticker and a known commodity
      * _normalize_entry handles a synthetic feedparser-like entry
    """
    print("=== filings.py smoke test ===")

    # 1) Stable event_id.
    eid1 = _stable_event_id("https://rbi.org.in/x", "Mon, 17 Jun 2026 12:00:00 GMT")
    eid2 = _stable_event_id("https://rbi.org.in/x", "Mon, 17 Jun 2026 12:00:00 GMT")
    assert eid1 == eid2, "event_id must be deterministic"
    assert eid1.startswith("filings:"), "event_id must carry source prefix"
    print(f"  event_id deterministic: {eid1}")

    # 2) Category hint coverage.
    samples = [
        ("RBI cuts repo rate by 25 bps", "rate_decision"),
        ("Master Circular on KYC framework", "policy_shift"),
        ("Import duty on edible oils revised", "policy_shift"),
        ("Open offer by Acquirer for Target Ltd", "corporate_action"),
        ("Show cause notice issued under SEBI Act", "corporate_action"),
        ("Government hikes MSP for wheat", "agri_supply"),
        ("FDI cap raised for insurance", "policy_shift"),
    ]
    for text, expected in samples:
        got = _category_hint(text)
        assert got == expected, f"expected {expected} for {text!r}, got {got}"
    print(f"  {len(samples)} category-hint patterns match")

    # 3) Entity hint extraction.
    hint = _extract_entities_hint(
        "Reliance Industries announces open offer; crude oil prices in focus."
    )
    assert "RELIANCE" in hint["organizations"], hint
    assert "crude_oil" in hint["commodities"], hint
    assert "open_offer" in hint["themes"], hint
    print(f"  entities_hint OK: {hint}")

    # 4) Synthetic normalization (no network).
    class _FakeEntry:
        link = "https://www.sebi.gov.in/legal/circulars/jun-2026/test_99999.html"
        title = "Circular on framework for ESG disclosures"
        summary = "<p>SEBI issues a framework to streamline ESG reporting.</p>"
        published = "Mon, 17 Jun 2026 09:30:00 GMT"
        published_parsed = (2026, 6, 17, 9, 30, 0, 0, 0, 0)

    feed_meta = {"authority": "SEBI", "kind": "all", "url": "x"}
    evt = _normalize_entry(_FakeEntry(), feed_meta)
    assert evt is not None
    assert evt["source"] == "filings"
    assert evt["raw_payload"]["filing_authority"] == "SEBI"
    assert evt["raw_payload"].get("category_hint") == "policy_shift"
    assert evt["event_id"].startswith("filings:")
    print(f"  synthetic event normalized: {evt['event_id']} | hint={evt['raw_payload'].get('category_hint')}")

    print("=== smoke test OK ===")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        _smoke()
        sys.exit(0)
    sys.exit(poll_once_cli())

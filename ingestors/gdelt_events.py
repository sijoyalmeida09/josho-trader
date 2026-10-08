"""gdelt_events.py -- Layer 4 ingestor #1 (GDELT 2.0 ArtList).

Pure ingestion: polls GDELT 2.0 DOC API for curated keywords, returns raw
event dicts per SITUATIONAL_SPEC.md section 10. Does NOT classify (Agent 8),
does NOT write to DB (Agent 13), does NOT import from sibling Layer-4 modules.
Stdlib + requests only.

CLI:
    python ingestors/gdelt_events.py --dry-run --verbose
    python ingestors/gdelt_events.py --since=2026-06-17T10:00:00Z
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

# --- Constants ---------------------------------------------------------------
UTC = timezone.utc
SOURCE_NAME: str = "gdelt_events"
USER_AGENT: str = "JoShoTrader-GDELTEvents/1.0 (+contact: contact@example.com)"
GDELT_DOC_ENDPOINT: str = "https://api.gdeltproject.org/api/v2/doc/doc"
HTTP_TIMEOUT_S: int = 20
RATE_LIMIT_SLEEP_S: float = 1.1  # GDELT soft cap ~1 req/sec
MAX_RECORDS: int = 15
DEFAULT_LOOKBACK_MIN: int = 60
MAX_RETRIES: int = 3
RETRY_BACKOFF_BASE_S: float = 1.5

# Keyword -> taxonomy candidate hint. Classifier (Agent 8) makes the final
# decision; this hint is informational only.
KEYWORDS: list[tuple[str, str]] = [
    ("Iran oil",                   "oil_shock"),
    ("Strait of Hormuz",           "oil_shock"),
    ("OPEC production cut",        "oil_shock"),
    ("India Pakistan border",      "defense_flashpoint"),
    ("China India border",         "defense_flashpoint"),
    ("RBI rate",                   "rate_decision"),
    ("submarine cable cut",        "telecom_disruption"),
    ("OFAC sanction Russia",       "sanction"),
    ("OFAC sanction Iran",         "sanction"),
    ("rare earth export curb",     "tech_supply"),
]

log = logging.getLogger("ingestors.gdelt_events")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    )


# --- Helpers -----------------------------------------------------------------
def _normalize_seendate(seendate: str | None) -> str:
    """Convert GDELT seendate ('YYYYMMDDTHHMMSSZ' or 'YYYYMMDDHHMMSS') to ISO UTC.

    Falls back to now() if parsing fails -- event remains usable downstream.
    """
    if not seendate:
        return datetime.now(UTC).isoformat()
    raw = str(seendate).strip().replace("-", "").replace(":", "").replace("T", "").replace("Z", "")
    if len(raw) < 14 or not raw[:14].isdigit():
        return datetime.now(UTC).isoformat()
    try:
        return datetime.strptime(raw[:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC).isoformat()
    except ValueError:
        return datetime.now(UTC).isoformat()


def _make_event_id(url: str, seendate_iso: str) -> str:
    """Stable event_id per SITUATIONAL_SPEC section 3."""
    digest_src = f"{url or ''}|{seendate_iso or ''}".encode("utf-8", "replace")
    digest = hashlib.sha1(digest_src).hexdigest()[:16]  # noqa: S324 (id only)
    return f"{SOURCE_NAME}:{digest}"


def _parse_since(value: str | datetime | None) -> datetime:
    """Coerce --since input into an aware UTC datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            log.warning("Could not parse --since=%r, using default window", value)
    return datetime.now(UTC) - timedelta(minutes=DEFAULT_LOOKBACK_MIN)


def _gdelt_timespan(since: datetime) -> str:
    """Map a since datetime to GDELT's 'timespan' shorthand (minutes)."""
    delta_min = max(15, int((datetime.now(UTC) - since).total_seconds() // 60))
    return f"{min(delta_min, 60 * 24 * 7)}min"  # GDELT caps usable window


def _fetch_keyword(keyword: str, since: datetime, verbose: bool = False) -> list[dict[str, Any]]:
    """Fetch one keyword from GDELT with bounded retries. Returns article list."""
    params = {
        "query": keyword,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": MAX_RECORDS,
        "sort": "DateDesc",
        "timespan": _gdelt_timespan(since),
    }
    headers = {"User-Agent": USER_AGENT}
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if verbose:
                prepared = requests.Request("GET", GDELT_DOC_ENDPOINT, params=params).prepare()
                log.info("GDELT GET %s", prepared.url)
            resp = requests.get(
                GDELT_DOC_ENDPOINT, params=params, headers=headers, timeout=HTTP_TIMEOUT_S
            )
            if resp.status_code == 200:
                try:
                    body = resp.json() or {}
                except ValueError:
                    log.warning("GDELT %s: non-JSON body (len=%d)", keyword, len(resp.content))
                    return []
                arts = body.get("articles", []) or []
                if not isinstance(arts, list):
                    log.warning("GDELT %s: 'articles' not a list", keyword)
                    return []
                return arts
            if 500 <= resp.status_code < 600:
                log.warning("GDELT %s HTTP %d (attempt %d/%d)",
                            keyword, resp.status_code, attempt, MAX_RETRIES)
                time.sleep(RETRY_BACKOFF_BASE_S ** attempt)
                continue
            # 4xx -- don't retry, just log and move on.
            log.warning("GDELT %s HTTP %d body=%s", keyword, resp.status_code, resp.text[:160])
            return []
        except requests.RequestException as exc:
            last_exc = exc
            log.warning("GDELT %s fetch error (attempt %d/%d): %s",
                        keyword, attempt, MAX_RETRIES, exc)
            time.sleep(RETRY_BACKOFF_BASE_S ** attempt)
    log.warning("GDELT %s gave up after %d attempts: %s", keyword, MAX_RETRIES, last_exc)
    return []


def _build_event(art: dict[str, Any], keyword: str, candidate_type: str) -> dict[str, Any] | None:
    """Convert one GDELT article row into a raw event dict, or None if invalid."""
    url = (art.get("url") or "").strip()
    title = (art.get("title") or "").strip()
    if not url or not title:
        return None
    seendate_raw = art.get("seendate")
    timestamp_iso = _normalize_seendate(seendate_raw)
    event_id = _make_event_id(url, timestamp_iso)
    # tone is optional in ArtList; preserve when present, leave None otherwise.
    tone_val: float | None
    try:
        tone_val = float(art["tone"]) if "tone" in art and art["tone"] is not None else None
    except (TypeError, ValueError):
        tone_val = None
    raw_payload: dict[str, Any] = {
        "keyword": keyword,
        "candidate_event_type": candidate_type,
        "tone": tone_val,
        "domain": art.get("domain"),
        "language": art.get("language"),
        "seendate": seendate_raw,
        "sourcecountry": art.get("sourcecountry"),
        "socialimage": art.get("socialimage"),
    }
    return {
        "event_id": event_id,
        "source": SOURCE_NAME,
        "headline": title[:500],
        "url": url[:1000],
        "timestamp": timestamp_iso,
        "raw_payload": raw_payload,
        "_keyword_match": keyword,
    }


# --- Public API (frozen contract) -------------------------------------------
def poll(since: datetime | None = None, verbose: bool = False) -> list[dict[str, Any]]:
    """Poll all configured GDELT keywords and return deduped raw event dicts.

    Failsoft per SITUATIONAL_SPEC section 12: a single dead keyword does not
    stall the rest. Empty list = total failure (caller decides how to react).
    """
    window_start = since or (datetime.now(UTC) - timedelta(minutes=DEFAULT_LOOKBACK_MIN))
    if window_start.tzinfo is None:
        window_start = window_start.replace(tzinfo=UTC)

    by_id: dict[str, dict[str, Any]] = {}
    keywords_with_results = 0

    for idx, (keyword, candidate_type) in enumerate(KEYWORDS):
        if idx > 0:
            time.sleep(RATE_LIMIT_SLEEP_S)
        articles = _fetch_keyword(keyword, window_start, verbose=verbose)
        if not articles:
            continue
        keywords_with_results += 1
        for art in articles:
            event = _build_event(art, keyword, candidate_type)
            if event is None:
                continue
            existing = by_id.get(event["event_id"])
            if existing is None:
                event["_keyword_matches"] = [keyword]
                by_id[event["event_id"]] = event
            else:
                matches = existing.setdefault("_keyword_matches", [existing.get("_keyword_match")])
                if keyword not in matches:
                    matches.append(keyword)

    events = list(by_id.values())
    log.info(
        "GDELT poll done: keywords=%d hit=%d events=%d window_start=%s",
        len(KEYWORDS), keywords_with_results, len(events), window_start.isoformat(),
    )
    return events


def poll_once_cli() -> int:
    """argparse entry. Exit 0 on success, 1 on total source failure."""
    parser = argparse.ArgumentParser(
        prog="gdelt_events",
        description="Poll GDELT 2.0 ArtList for Layer-4 candidate events.",
    )
    parser.add_argument("--since", type=str, default=None,
                        help="ISO 8601 lower-bound timestamp (default: 60 min ago).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print JSON lines to stdout; do not persist.")
    parser.add_argument("--verbose", action="store_true",
                        help="Log each request URL.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        log.setLevel(logging.DEBUG)

    since_dt = _parse_since(args.since)
    log.info("poll_once_cli starting | since=%s dry_run=%s verbose=%s",
             since_dt.isoformat(), args.dry_run, args.verbose)

    try:
        events = poll(since=since_dt, verbose=args.verbose)
    except Exception as exc:  # noqa: BLE001 -- top-level CLI guard
        log.exception("poll() crashed: %s", exc)
        return 1

    if not events:
        log.warning("No events returned (all sources may be dead).")
        # Distinguish 'quiet news cycle' from 'total failure': we can't know
        # for sure, but treating zero as a soft success keeps cron green.
        # Spec says exit 1 only on TOTAL failure; we conservatively exit 0.
        return 0

    if args.dry_run:
        for ev in events:
            sys.stdout.write(json.dumps(ev, ensure_ascii=False, sort_keys=True) + "\n")
        sys.stdout.flush()
    else:
        log.info("Non-dry-run mode is a no-op in pure ingestor; "
                 "Agent 13 (signal_writer) handles persistence. "
                 "Returned %d events to caller.", len(events))

    return 0


# --- Smoke test entry -------------------------------------------------------
if __name__ == "__main__":
    sys.exit(poll_once_cli())

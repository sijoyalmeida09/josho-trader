"""macro.py -- Layer 4 ingestor #7 (FRED macro & commodity context).

Polls FRED CSV endpoint for 5 series (Brent, WTI, USD/INR, US 10y, gold).
Emits a raw event ONLY when latest obs crosses |delta_pct|>=1.0% daily or
|z_score_20d|>=2.0 sigma. Pure ingestor: no classify, no DB, no Layer-4
sibling imports. Stdlib + requests + csv only.

CLI: python ingestors/macro.py --dry-run --verbose
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any

import requests

UTC = timezone.utc
SOURCE_NAME: str = "macro"
USER_AGENT: str = "JoShoTrader-Macro/1.0 (+contact: contact@example.com)"
FRED_CSV_ENDPOINT: str = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_SERIES_URL: str = "https://fred.stlouisfed.org/series/{series_id}"
HTTP_TIMEOUT_S: int = 20
RATE_LIMIT_SLEEP_S: float = 1.0
MAX_RETRIES: int = 3
RETRY_BACKOFF_BASE_S: float = 1.5
OBSERVATION_TAIL: int = 15
ZSCORE_WINDOW: int = 20

# Regime-flag thresholds (frozen per spec).
DELTA_PCT_THRESHOLD: float = 1.0
ZSCORE_THRESHOLD: float = 2.0

SeriesSpec = tuple[str, str, str, dict[str, list[str]]]

# (series_id, friendly_label, unit, entities_hint per SITUATIONAL_SPEC #3)
SERIES: list[SeriesSpec] = [
    ("DCOILBRENTEU", "Brent", "USD/bbl",
     {"themes": ["oil_price", "macro"], "commodities": ["crude oil"]}),
    ("DCOILWTICO", "WTI", "USD/bbl",
     {"themes": ["oil_price", "macro"], "commodities": ["crude oil"]}),
    ("DEXINUS", "USD/INR", "INR per USD",
     {"themes": ["fx_move", "macro"], "commodities": []}),
    ("DGS10", "US 10y", "percent",
     {"themes": ["rates", "macro"], "commodities": []}),
    ("GOLDAMGBD228NLBM", "Gold (London PM)", "USD/oz",
     {"themes": ["safe_haven", "macro"], "commodities": ["gold"]}),
]

log = logging.getLogger("ingestors.macro")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    )


def _parse_since(value: str | datetime | None) -> datetime | None:
    """Coerce --since into aware UTC datetime. Spec ignores it; kept for CLI parity."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            log.warning("Could not parse --since=%r, ignoring", value)
    return None


def _observation_to_iso(date_str: str) -> str:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC).isoformat()
    except ValueError:
        return datetime.now(UTC).isoformat()


def _delta_bucket(delta_pct: float) -> str:
    if not math.isfinite(delta_pct):
        return "+0.0"
    return f"{round(delta_pct * 2.0) / 2.0:+.1f}"


def _format_value(value: float, unit: str) -> str:
    if "USD/bbl" in unit:
        return f"${value:.2f}"
    if "USD/oz" in unit:
        return f"${value:,.2f}"
    if unit == "percent":
        return f"{value:.2f}%"
    if "INR" in unit:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _fetch_series_csv(series_id: str, verbose: bool = False) -> str | None:
    """GET the FRED observation CSV for one series with bounded retries."""
    params = {"id": series_id}
    headers = {"User-Agent": USER_AGENT, "Accept": "text/csv,*/*"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if verbose:
                prepared = requests.Request("GET", FRED_CSV_ENDPOINT, params=params).prepare()
                log.info("FRED GET %s", prepared.url)
            resp = requests.get(FRED_CSV_ENDPOINT, params=params, headers=headers,
                                timeout=HTTP_TIMEOUT_S)
            if resp.status_code == 200:
                text = resp.text or ""
                if text.strip():
                    return text
                log.warning("FRED %s: empty body", series_id)
                return None
            if 500 <= resp.status_code < 600:
                log.warning("FRED %s HTTP %d (attempt %d/%d)",
                            series_id, resp.status_code, attempt, MAX_RETRIES)
                time.sleep(RETRY_BACKOFF_BASE_S ** attempt)
                continue
            log.warning("FRED %s HTTP %d body=%s",
                        series_id, resp.status_code, resp.text[:120])
            return None
        except requests.RequestException as exc:
            log.warning("FRED %s fetch error (attempt %d/%d): %s",
                        series_id, attempt, MAX_RETRIES, exc)
            time.sleep(RETRY_BACKOFF_BASE_S ** attempt)
    log.warning("FRED %s gave up after %d attempts", series_id, MAX_RETRIES)
    return None


def _parse_observations(csv_text: str) -> list[tuple[str, float]]:
    """Parse FRED CSV into [(date_str, value), ...]. Drops '.' / blank rows."""
    rows: list[tuple[str, float]] = []
    try:
        reader = csv.reader(io.StringIO(csv_text))
        if next(reader, None) is None:
            return []
        for row in reader:
            if not row or len(row) < 2:
                continue
            date_str, raw_val = (row[0] or "").strip(), (row[1] or "").strip()
            if not date_str or raw_val in ("", "."):
                continue
            try:
                val = float(raw_val)
            except ValueError:
                continue
            if math.isfinite(val):
                rows.append((date_str, val))
    except (csv.Error, StopIteration) as exc:
        log.warning("CSV parse error: %s", exc)
        return []
    rows.sort(key=lambda r: r[0])
    return rows


def _zscore(values: list[float], current: float) -> float | None:
    """Z-score of `current` against `values`. None if insufficient/zero-stdev."""
    if len(values) < 3:
        return None
    try:
        mu, sd = statistics.fmean(values), statistics.pstdev(values)
    except statistics.StatisticsError:
        return None
    if sd == 0 or not math.isfinite(sd):
        return None
    z = (current - mu) / sd
    return z if math.isfinite(z) else None


def _build_event(series_id: str, label: str, unit: str,
                 entities_hint: dict[str, list[str]],
                 obs: list[tuple[str, float]]) -> dict[str, Any] | None:
    """Compute regime flags for one series; emit event if any flag fires."""
    if len(obs) < 2:
        return None
    tail = obs[-OBSERVATION_TAIL:]
    last_date, current_value = tail[-1]
    _, prev_close = tail[-2]
    if prev_close == 0:
        return None
    delta_pct = ((current_value - prev_close) / prev_close) * 100.0
    z = _zscore([v for _, v in obs[-ZSCORE_WINDOW:]], current_value)
    flag_delta = abs(delta_pct) >= DELTA_PCT_THRESHOLD
    flag_z = (z is not None) and (abs(z) >= ZSCORE_THRESHOLD)
    if not (flag_delta or flag_z):
        return None
    z_disp = "n/a" if z is None else f"{z:+.1f}sigma"
    headline = (f"{label} {delta_pct:+.1f}% to "
                f"{_format_value(current_value, unit)} (z={z_disp})")
    return {
        "event_id": f"{SOURCE_NAME}:{series_id}:{last_date}:{_delta_bucket(delta_pct)}",
        "source": SOURCE_NAME,
        "headline": headline[:500],
        "url": FRED_SERIES_URL.format(series_id=series_id),
        "timestamp": _observation_to_iso(last_date),
        "raw_payload": {
            "series_id": series_id, "label": label, "unit": unit,
            "current_value": current_value, "prev_close": prev_close,
            "delta_pct": round(delta_pct, 4),
            "z_score_20d": None if z is None else round(z, 4),
            "observation_date": last_date,
            "last_15_observations": [{"date": d, "value": v} for d, v in tail],
            "regime_flags": {"delta_pct_breach": flag_delta, "zscore_breach": flag_z},
            "direction_hint": 1 if delta_pct > 0 else (-1 if delta_pct < 0 else 0),
            "entities_hint": entities_hint,
        },
    }


# ---------------------------------------------------------------------------
# Public API (frozen contract)
# ---------------------------------------------------------------------------

def poll(since: datetime | None = None, verbose: bool = False) -> list[dict[str, Any]]:
    """Poll all configured FRED series and return raw event dicts.

    `since` accepted for contract parity but ignored (we always read latest).
    Failsoft per SITUATIONAL_SPEC #12: a single dead series does not stall others.
    """
    _ = _parse_since(since)
    events: list[dict[str, Any]] = []
    series_ok = 0

    for idx, (series_id, label, unit, entities_hint) in enumerate(SERIES):
        if idx > 0:
            time.sleep(RATE_LIMIT_SLEEP_S)
        csv_text = _fetch_series_csv(series_id, verbose=verbose)
        if csv_text is None:
            if verbose:
                log.info("  %-22s status=fetch_failed", label)
            continue
        obs = _parse_observations(csv_text)
        if len(obs) < 2:
            log.warning("FRED %s: insufficient observations (%d)", series_id, len(obs))
            if verbose:
                log.info("  %-22s status=insufficient_data n=%d", label, len(obs))
            continue
        series_ok += 1
        if verbose:
            last_date, current_value = obs[-1]
            _, prev_close = obs[-2]
            d = 0.0 if prev_close == 0 else ((current_value - prev_close) / prev_close) * 100.0
            z = _zscore([v for _, v in obs[-ZSCORE_WINDOW:]], current_value)
            z_disp = "n/a" if z is None else f"{z:+.2f}"
            log.info("  %-22s %s | value=%s delta=%+.2f%% z=%s",
                     label, last_date, _format_value(current_value, unit), d, z_disp)
        event = _build_event(series_id, label, unit, entities_hint, obs)
        if event is not None:
            events.append(event)

    log.info("FRED poll done: series=%d ok=%d events=%d (delta>=%.1f%% or |z|>=%.1f)",
             len(SERIES), series_ok, len(events),
             DELTA_PCT_THRESHOLD, ZSCORE_THRESHOLD)
    return events


def poll_once_cli() -> int:
    """argparse entry. Exit 0 on success, 1 on crash."""
    parser = argparse.ArgumentParser(
        prog="macro",
        description="Poll FRED for macro/commodity regime-flag events.",
    )
    parser.add_argument("--since", type=str, default=None,
                        help="ISO 8601 timestamp (accepted for contract parity; "
                             "macro always reads latest observation).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print JSON lines to stdout; do not persist.")
    parser.add_argument("--verbose", action="store_true",
                        help="Log per-series current value + delta even when "
                             "no signal fires.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        log.setLevel(logging.DEBUG)

    since_dt = _parse_since(args.since)
    log.info("poll_once_cli starting | since=%s dry_run=%s verbose=%s",
             since_dt.isoformat() if since_dt else "none",
             args.dry_run, args.verbose)

    try:
        events = poll(since=since_dt, verbose=args.verbose)
    except Exception as exc:  # noqa: BLE001 -- top-level CLI guard
        log.exception("poll() crashed: %s", exc)
        return 1

    if not events:
        log.info("No regime-flag events emitted (markets quiet or sources dead).")
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


# Smoke test entry
if __name__ == "__main__":
    sys.exit(poll_once_cli())

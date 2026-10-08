"""attribution.py - Layer 4 Agent 14: Signal Outcome Attribution.

For every shadow/active signal that has aged past two half-lives, fetch the
actual price move on each affected NSE instrument and write a row to
``trader_signal_outcomes``. Agent 11 consumes these to recalibrate hit-rates.

Boundaries (spec sec 14):
  * READ-ONLY against trader_signals; INSERT-ONLY into trader_signal_outcomes.
  * No edits to trader_db.py or backtest/harness.py. Yahoo fetcher is
    reimplemented inline to keep modules decoupled.
  * stdlib + ``requests`` only. ASCII only. Type-hinted. Failsoft.

Public contract (SITUATIONAL_SPEC.md sec 10):
    def attribute_outcomes(window_hours: int = 24) -> list[dict]
    def cli() -> int
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from trader_db import TraderDB  # type: ignore

UTC = timezone.utc
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "data" / "attribution_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

YAHOO_CHART = "https://query1.finance.yahoo.com/v7/finance/chart/{ticker}"
HTTP_TIMEOUT_S = 20
HTTP_GAP_S = 0.5
USER_AGENT = "JoShoTrader-Attribution/1.0 (+contact@example.com)"
# IST market open 09:15 = UTC 03:45. Events with UTC clock earlier than this
# are treated as overnight (price_start = prior-day close).
IST_OPEN_UTC_H, IST_OPEN_UTC_M = 3, 45

logger = logging.getLogger("attribution")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    )

# --- helpers --------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(UTC)


def _parse_ts(ts_str: Optional[str]) -> Optional[datetime]:
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def _is_overnight(event_ts: datetime) -> bool:
    return (event_ts.hour, event_ts.minute) < (IST_OPEN_UTC_H, IST_OPEN_UTC_M)


def _at(lst: list[Any], i: int) -> Optional[float]:
    if 0 <= i < len(lst):
        v = lst[i]
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return None


# --- Yahoo fetcher (inline; decoupled from backtest/harness.py) ----------

_LAST_FETCH_TS: float = 0.0


def _yahoo_fetch_bars(
    ticker: str, period1: int, period2: int, cache_date: str
) -> Optional[dict[str, Any]]:
    """Fetch daily bars for ``<ticker>.NS``. Caches to
    data/attribution_cache/<ticker>_<date>.json. Returns None on failure."""
    global _LAST_FETCH_TS
    safe = ticker.replace("/", "_").replace("\\", "_")
    cache_file = CACHE_DIR / f"{safe}_{cache_date}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            ts_list = cached.get("timestamps") or []
            if ts_list and ts_list[0] <= period1 and ts_list[-1] >= period2 - 86400:
                return cached
        except Exception:
            pass
    gap = time.time() - _LAST_FETCH_TS
    if gap < HTTP_GAP_S:
        time.sleep(HTTP_GAP_S - gap)
    url = YAHOO_CHART.format(ticker=f"{ticker}.NS")
    try:
        resp = requests.get(
            url,
            params={"interval": "1d", "period1": period1, "period2": period2},
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT_S,
        )
        _LAST_FETCH_TS = time.time()
        if resp.status_code != 200:
            logger.warning("yahoo_http ticker=%s status=%s", ticker, resp.status_code)
            return None
        raw = resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.warning("yahoo_fetch_failed ticker=%s err=%s", ticker, e)
        return None
    except Exception as e:
        logger.warning("yahoo_fetch_unexpected ticker=%s err=%s", ticker, e)
        return None
    chart = raw.get("chart") or {}
    if chart.get("error") or not (chart.get("result") or []):
        logger.warning("yahoo_chart_empty ticker=%s err=%s", ticker, chart.get("error"))
        return None
    r0 = chart["result"][0]
    quote = ((r0.get("indicators") or {}).get("quote") or [{}])[0]
    parsed = {
        "ticker": ticker,
        "timestamps": r0.get("timestamp") or [],
        "open": quote.get("open") or [],
        "close": quote.get("close") or [],
        "fetched_at": int(time.time()),
    }
    try:
        cache_file.write_text(json.dumps(parsed), encoding="utf-8")
    except Exception as e:
        logger.debug("cache_write_failed ticker=%s err=%s", ticker, e)
    return parsed


# --- Move computation ----------------------------------------------------

def _compute_move(
    ticker: str, event_ts: datetime, window_minutes: int
) -> tuple[Optional[float], list[dict[str, Any]], Optional[datetime]]:
    """Realized pct move for ticker around event_ts.

    Overnight (event_ts UTC < 03:45 = pre IST open):
      price_start = prior-day close;
      price_end   = next-day open (fallback: next-day close).
    Intraday: price_start = prior-day close (daily-bar approximation);
      price_end = close at or before event_ts + window_minutes. The window
      value is preserved on the outcome row as observed_within_minutes.

    Returns (move_pct, window_bars, first_price_ts). LOOK-AHEAD GUARD:
    asserts event_ts < first observed bar timestamp.
    """
    event_unix = int(event_ts.timestamp())
    window_end_unix = event_unix + window_minutes * 60
    period1 = event_unix - 14 * 86400
    period2 = max(window_end_unix, event_unix) + 14 * 86400
    bars = _yahoo_fetch_bars(ticker, period1, period2, event_ts.strftime("%Y-%m-%d"))
    if not bars:
        return None, [], None
    ts_list = bars.get("timestamps") or []
    closes = bars.get("close") or []
    opens = bars.get("open") or []
    if not ts_list or not closes:
        return None, [], None
    idx_after: Optional[int] = None
    for i, t in enumerate(ts_list):
        if t > event_unix:
            idx_after = i
            break
    if idx_after is None or idx_after == 0:
        return None, [], None
    first_price_ts = datetime.fromtimestamp(ts_list[idx_after], tz=UTC)
    # LOOK-AHEAD GUARD (spec): event_ts must strictly precede first price obs.
    assert event_ts < first_price_ts, (
        f"lookahead_guard: event_ts={event_ts} !< first_price_ts={first_price_ts}"
    )
    try:
        if _is_overnight(event_ts):
            price_start = _at(closes, idx_after - 1)
            price_end = _at(opens, idx_after)
            if price_end is None:
                price_end = _at(closes, idx_after)
        else:
            price_start = _at(closes, idx_after - 1)
            idx_end = idx_after
            for j in range(idx_after, len(ts_list)):
                if ts_list[j] <= window_end_unix:
                    idx_end = j
                else:
                    break
            price_end = _at(closes, idx_end)
    except Exception as e:
        logger.warning("move_compute_failed ticker=%s err=%s", ticker, e)
        return None, [], first_price_ts
    if price_start is None or price_end is None or price_start == 0:
        return None, [], first_price_ts
    move = (price_end - price_start) / price_start
    bars_payload: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i in (idx_after - 1, idx_after, min(idx_after + 1, len(ts_list) - 1)):
        if 0 <= i < len(ts_list):
            iso = datetime.fromtimestamp(ts_list[i], tz=UTC).isoformat()
            if iso in seen:
                continue
            seen.add(iso)
            bars_payload.append({"ts": iso, "open": _at(opens, i), "close": _at(closes, i)})
    return float(move), bars_payload[:3], first_price_ts


# --- Signal selection ----------------------------------------------------

def _fetch_candidate_signals(
    db: TraderDB, window_hours: int, since_override: Optional[datetime] = None
) -> list[dict[str, Any]]:
    """Shadow/active signals in window, aged past 2x half_life_min, not yet
    attributed. Two-query strategy because TraderDB lacks a left-anti-join
    wrapper; age filter is client-side (half_life_min is per-row)."""
    client = db._conn()  # noqa: SLF001
    if client is None:
        return []
    since_iso = (
        since_override.astimezone(UTC).isoformat()
        if since_override is not None
        else (_now_utc() - timedelta(hours=window_hours)).isoformat()
    )
    now = _now_utc()
    try:
        res = (
            client.table("trader_signals")
            .select(
                "id,event_id,event_type,instruments,direction,magnitude,"
                "half_life_min,status,created_at,headline,source,confidence,"
                "affected_symbols,category"
            )
            .in_("status", ["shadow", "active"])
            .gte("created_at", since_iso)
            .order("created_at", desc=True)
            .limit(500)
            .execute()
        )
        rows = list(res.data or [])
    except Exception as e:
        logger.warning("fetch_signals_failed err=%s", e)
        return []
    aged: list[dict[str, Any]] = []
    for r in rows:
        created = _parse_ts(r.get("created_at"))
        if created is None:
            continue
        try:
            hl_int = int(r.get("half_life_min") or 240)
        except (TypeError, ValueError):
            hl_int = 240
        if created + timedelta(minutes=hl_int * 2) <= now:
            aged.append(r)
    if not aged:
        return []
    try:
        ids = [r["id"] for r in aged if r.get("id")]
        if not ids:
            return aged
        existing = (
            client.table("trader_signal_outcomes")
            .select("signal_id")
            .in_("signal_id", ids)
            .execute()
        )
        attributed = {row["signal_id"] for row in (existing.data or []) if row.get("signal_id")}
    except Exception as e:
        logger.warning("fetch_outcomes_failed err=%s - treating as none-attributed", e)
        attributed = set()
    fresh = [r for r in aged if r.get("id") not in attributed]
    logger.info("candidate_signals aged=%d attributed=%d fresh=%d",
                len(aged), len(attributed), len(fresh))
    return fresh


def _insert_outcome(db: TraderDB, outcome: dict[str, Any]) -> bool:
    """Direct client insert (no trader_db.py edits per spec boundaries)."""
    client = db._conn()  # noqa: SLF001
    if client is None:
        return False
    try:
        client.table("trader_signal_outcomes").insert(outcome).execute()
        return True
    except Exception as e:
        logger.warning("insert_outcome_failed signal=%s sym=%s err=%s",
                       outcome.get("signal_id"), outcome.get("symbol"), e)
        return False


# --- Public contract -----------------------------------------------------

def attribute_outcomes(
    window_hours: int = 24,
    since_override: Optional[datetime] = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """For each shadow/active signal aged past 2x half-life: compute realized
    price move per instrument and write a trader_signal_outcomes row.
    Returns the outcome dicts produced. Failsoft on every fetch."""
    db = TraderDB()
    if not db.enabled:
        logger.warning("trader_db disabled - returning empty outcome list")
        return []
    signals = _fetch_candidate_signals(db, window_hours, since_override)
    if not signals:
        return []
    outcomes_written: list[dict[str, Any]] = []
    now_iso = _now_utc().isoformat()
    for sig in signals:
        sid = sig.get("id")
        created = _parse_ts(sig.get("created_at"))
        if not sid or not created:
            continue
        instruments = sig.get("instruments") or sig.get("affected_symbols") or []
        direction = sig.get("direction") or {}
        if isinstance(direction, str):
            try:
                direction = json.loads(direction)
            except Exception:
                direction = {}
        try:
            magnitude = float(sig.get("magnitude") or 0.0)
        except (TypeError, ValueError):
            magnitude = 0.0
        try:
            hl_min = int(sig.get("half_life_min") or 240)
        except (TypeError, ValueError):
            hl_min = 240
        window_minutes = hl_min * 2
        for sym in instruments:
            try:
                d_raw = direction.get(sym, 0) if isinstance(direction, dict) else 0
                d = int(d_raw) if d_raw is not None else 0
            except (TypeError, ValueError):
                d = 0
            if d == 0:
                continue  # spec: skip neutral
            try:
                move, bars, first_ts = _compute_move(sym, created, window_minutes)
            except AssertionError as e:
                logger.warning("lookahead_guard signal=%s sym=%s err=%s", sid, sym, e)
                continue
            except Exception as e:
                logger.warning("compute_move_failed signal=%s sym=%s err=%s", sid, sym, e)
                continue
            if move is None:
                continue
            hit = (d == 1 and move > 0) or (d == -1 and move < 0)
            outcome = {
                "signal_id": sid,
                "symbol": sym,
                "predicted_direction": d,
                "predicted_magnitude": round(magnitude, 3),
                "observed_pct_move": round(move, 4),
                "observed_within_minutes": window_minutes,
                "hit": bool(hit),
                "observed_at": now_iso,
                "payload": {
                    "event_type": sig.get("event_type") or sig.get("category"),
                    "source": sig.get("source"),
                    "headline": sig.get("headline"),
                    "first_price_ts": first_ts.isoformat() if first_ts else None,
                    "window_bars": bars,
                },
            }
            if dry_run:
                outcomes_written.append(outcome)
                continue
            if _insert_outcome(db, outcome):
                outcomes_written.append(outcome)
    logger.info("attribution_complete signals=%d outcomes=%d dry_run=%s",
                len(signals), len(outcomes_written), dry_run)
    return outcomes_written


# --- CLI -----------------------------------------------------------------

def cli() -> int:
    p = argparse.ArgumentParser(
        description="Attribution - Layer 4 Agent 14. "
                    "Match aged shadow/active signals to realized price moves.")
    p.add_argument("--window-hours", type=int, default=24,
                   help="Lookback window in hours (default: 24)")
    p.add_argument("--since", default=None,
                   help="ISO-8601 timestamp override; supersedes --window-hours")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute outcomes but skip INSERT")
    p.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    args = p.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    since_override: Optional[datetime] = None
    if args.since:
        since_override = _parse_ts(args.since)
        if since_override is None:
            logger.error("invalid --since value: %r", args.since)
            return 2
    outcomes = attribute_outcomes(
        window_hours=args.window_hours,
        since_override=since_override,
        dry_run=args.dry_run,
    )
    summary = {
        "outcomes_count": len(outcomes),
        "hit_count": sum(1 for o in outcomes if o.get("hit")),
        "dry_run": args.dry_run,
        "window_hours": args.window_hours,
    }
    print(json.dumps(summary, indent=2))
    if args.verbose:
        for o in outcomes[:10]:
            print(json.dumps(o, indent=2, default=str))
    return 0


if __name__ == "__main__":
    # Spec smoke test: empty list expected (no signals yet).
    _result = attribute_outcomes(window_hours=24)
    print(f"smoke_test: outcome_count={len(_result)}")
    sys.exit(0)

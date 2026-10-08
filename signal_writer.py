"""signal_writer.py - Layer 4 Agent 13 (Signal Writer).

Writes scored, deduped, classified signals into ``trader_signals``. Every
signal starts at ``status='shadow'``. Promotion to ``active`` is gated by
the calibration harness (Agent 11) read from ``backtest/calibration.json``.

Contract (frozen, see SITUATIONAL_SPEC.md section 10):

    class SignalWriter:
        def __init__(self, trader_db=None, calibration_path: Path | None = None): ...
        def write(self, signal: dict, force_shadow: bool = True) -> str | None: ...
        def write_batch(self, signals: list[dict], force_shadow: bool = True) -> list[str]: ...

Failsoft: never raises. On DB unavailability buffers to a JSONL outbox.
Idempotent: re-writing same ``event_id`` is a no-op or trivial update,
never downgrades confidence/status.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("signal_writer")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

try:
    from trader_db import TraderDB  # type: ignore
except Exception:  # pragma: no cover
    try:
        from automation.josho_trader.trader_db import TraderDB  # type: ignore
    except Exception:
        TraderDB = None  # type: ignore

# 14 event types from SITUATIONAL_SPEC.md section 2.
_TAXONOMY_FALLBACK: tuple[str, ...] = (
    "oil_shock", "defense_flashpoint", "sanction", "monsoon_cyclone",
    "flood_quake", "rate_decision", "metals_supply", "fx_move",
    "telecom_disruption", "agri_supply", "tech_supply", "corporate_action",
    "policy_shift", "geopolitical_other",
)
try:
    from situational_map import EVENT_TYPE_MAP  # type: ignore
    _ALLOWED_EVENT_TYPES: frozenset[str] = frozenset(EVENT_TYPE_MAP.keys())
except Exception:  # pragma: no cover
    _ALLOWED_EVENT_TYPES = frozenset(_TAXONOMY_FALLBACK)
_HERE = Path(__file__).resolve().parent
_DEFAULT_CALIB_PATH = _HERE / "backtest" / "calibration.json"
_OUTBOX_DIR = _HERE / "data" / "signals_outbox"
_CALIB_RELOAD_SECONDS = 600  # 10 minutes

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_event_id(event_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", event_id)[:80] or uuid.uuid4().hex[:12]


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dominant_direction_sign(direction: dict[str, Any], instruments: list[str]) -> int:
    """Sign of the instrument with the largest absolute direction."""
    if not direction:
        return 0
    best_sym: str | None = None
    best_abs: float = -1.0
    for sym, raw in direction.items():
        if instruments and sym not in instruments:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if abs(val) > best_abs:
            best_abs = abs(val)
            best_sym = sym
    if best_sym is None:
        return 0
    try:
        v = float(direction.get(best_sym, 0))
    except (TypeError, ValueError):
        return 0
    return 1 if v > 0 else (-1 if v < 0 else 0)


def _validate(signal: dict) -> tuple[bool, str]:
    """Return (ok, reason). reason='' on success."""
    if not isinstance(signal, dict):
        return False, "not_a_dict"
    event_id = signal.get("event_id")
    if not event_id or not isinstance(event_id, str):
        return False, "event_id_missing_or_invalid"
    event_type = signal.get("event_type")
    if event_type not in _ALLOWED_EVENT_TYPES:
        return False, f"event_type_not_in_taxonomy:{event_type}"
    confidence = _coerce_int(signal.get("confidence"), -1)
    if not (0 <= confidence <= 100):
        return False, f"confidence_out_of_range:{confidence}"
    magnitude = _coerce_float(signal.get("magnitude"), -1.0)
    if not (0.0 <= magnitude <= 1.0):
        return False, f"magnitude_out_of_range:{magnitude}"
    instruments = signal.get("instruments") or []
    if not isinstance(instruments, list) or not instruments:
        return False, "instruments_empty"
    provenance = signal.get("provenance_urls") or []
    if not isinstance(provenance, list) or not provenance:
        return False, "provenance_urls_empty"
    return True, ""


class SignalWriter:
    """Writes deduped, scored signals to ``trader_signals``."""

    def __init__(
        self,
        trader_db: Any = None,
        calibration_path: Path | None = None,
    ) -> None:
        self._db = trader_db
        if self._db is None and TraderDB is not None:
            try:
                self._db = TraderDB()
            except Exception as e:
                log.warning("signal_writer: TraderDB init failed: %s", e)
                self._db = None
        self._calib_path: Path = Path(calibration_path) if calibration_path else _DEFAULT_CALIB_PATH
        self._calib: dict[str, dict] = {}
        self._calib_loaded_at: float = 0.0
        try:
            _OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover
            log.warning("signal_writer: cannot create outbox dir: %s", e)

    # ---- calibration ----
    def _load_calibration(self) -> dict[str, dict]:
        now = time.time()
        if self._calib and (now - self._calib_loaded_at) < _CALIB_RELOAD_SECONDS:
            return self._calib
        data: dict[str, dict] = {}
        try:
            if self._calib_path.exists():
                raw = json.loads(self._calib_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    # Accept either {event_type: {...}} or {results: {event_type: {...}}}
                    src = raw.get("results") if isinstance(raw.get("results"), dict) else raw
                    data = {k: v for k, v in src.items() if isinstance(v, dict)}
        except Exception as e:
            log.warning("signal_writer: calibration load failed: %s", e)
            data = {}
        self._calib = data
        self._calib_loaded_at = now
        return data

    def _resolve_status(self, event_type: str, force_shadow: bool) -> str:
        if force_shadow:
            return "shadow"
        entry = self._load_calibration().get(event_type) or {}
        return "active" if bool(entry.get("gate_passed")) else "shadow"

    # ---- row builder ----
    @staticmethod
    def _build_row(signal: dict, status: str) -> dict:
        instruments: list[str] = list(signal.get("instruments") or [])
        direction: dict[str, Any] = dict(signal.get("direction") or {})
        magnitude = max(0.0, min(1.0, _coerce_float(signal.get("magnitude"), 0.0)))
        confidence = max(0, min(100, _coerce_int(signal.get("confidence"), 0)))
        half_life_min = max(1, _coerce_int(signal.get("half_life_min"), 240))
        corroboration = max(1, _coerce_int(signal.get("corroboration_count"), 1))
        provenance: list[str] = list(signal.get("provenance_urls") or [])

        severity = max(1, min(10, confidence // 10 or 1))
        sign = _dominant_direction_sign(direction, instruments)
        score_adjustment = int(round(magnitude * 100 * sign))
        legacy_url = provenance[0] if provenance else None

        expires_at = signal.get("expires_at")
        if not expires_at:
            try:
                expires_at = (datetime.now(timezone.utc) + timedelta(minutes=3 * half_life_min)).isoformat()
            except Exception:
                expires_at = None

        row: dict[str, Any] = {
            # new (041) columns
            "event_id": signal["event_id"],
            "event_type": signal["event_type"],
            "entities": signal.get("entities") or {},
            "instruments": instruments,
            "direction": direction,
            "magnitude": magnitude,
            "confidence": confidence,
            "half_life_min": half_life_min,
            "corroboration_count": corroboration,
            "provenance_urls": provenance,
            "status": status,
            # legacy (040) mirror
            "category": signal["event_type"],
            "affected_symbols": instruments,
            "severity": severity,
            "score_adjustment": score_adjustment,
            "headline": signal.get("headline") or signal.get("headline_normalized") or "",
            "url": legacy_url,
            "payload": signal.get("raw_payload") or {},
            "expires_at": expires_at,
        }
        return {k: v for k, v in row.items() if v is not None}

    # ---- outbox buffer ----
    @staticmethod
    def _buffer_to_outbox(signal: dict) -> str | None:
        try:
            _OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            slug = _safe_event_id(str(signal.get("event_id") or "noid"))
            path = _OUTBOX_DIR / f"{ts}_{slug}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(signal, default=str) + "\n")
            log.info(
                "signal_writer: buffered to outbox %s event_id=%s",
                path.name, signal.get("event_id"),
            )
            return f"outbox:{path.name}"
        except Exception as e:
            log.warning("signal_writer: outbox buffer failed: %s", e)
            return None

    # ---- supabase upsert ----
    def _upsert(self, row: dict) -> str | None:
        db = self._db
        if db is None or not getattr(db, "enabled", False):
            return None
        client = db._conn() if hasattr(db, "_conn") else None
        if client is None:
            return None
        try:
            existing: dict | None = None
            try:
                res = (
                    client.table("trader_signals")
                    .select("id,confidence,status,corroboration_count,provenance_urls")
                    .eq("event_id", row["event_id"]).limit(1).execute()
                )
                rows = res.data or []
                existing = rows[0] if rows else None
            except Exception:
                existing = None

            if existing:
                cur_conf = _coerce_int(existing.get("confidence"), 0)
                cur_status = str(existing.get("status") or "shadow")
                # Never downgrade active -> shadow.
                if cur_status == "active" and row.get("status") != "active":
                    row["status"] = "active"
                # Only update if new confidence strictly greater OR status changed.
                if _coerce_int(row.get("confidence"), 0) <= cur_conf and cur_status == row.get("status"):
                    log.info(
                        "signal_writer: no-op (existing confidence=%s status=%s) event_id=%s",
                        cur_conf, cur_status, row["event_id"],
                    )
                    return str(existing.get("id")) if existing.get("id") else None
                # Merge provenance + bump corroboration_count.
                merged_urls = list({*list(existing.get("provenance_urls") or []), *list(row.get("provenance_urls") or [])})
                row["provenance_urls"] = merged_urls
                row["corroboration_count"] = max(
                    _coerce_int(existing.get("corroboration_count"), 1),
                    _coerce_int(row.get("corroboration_count"), 1),
                )

            res = client.table("trader_signals").upsert(row, on_conflict="event_id").execute()
            data = res.data or []
            return str(data[0]["id"]) if data and "id" in data[0] else None
        except Exception as e:
            log.warning("signal_writer: upsert failed event_id=%s err=%s", row.get("event_id"), e)
            return None

    # ---- public API ----
    def write(self, signal: dict, force_shadow: bool = True) -> str | None:
        ok, reason = _validate(signal)
        if not ok:
            log.warning(
                "signal_writer: validation skip reason=%s event_id=%s",
                reason, signal.get("event_id") if isinstance(signal, dict) else None,
            )
            return None
        status = self._resolve_status(signal["event_type"], force_shadow)
        row = self._build_row(signal, status)

        new_id = self._upsert(row)
        if new_id:
            log.info(
                "signal_writer: wrote %s [%s] event_type=%s confidence=%s instruments=%s",
                new_id, row.get("status"), row.get("event_type"),
                row.get("confidence"), row.get("instruments"),
            )
            return new_id
        return self._buffer_to_outbox(signal)

    def write_batch(self, signals: list[dict], force_shadow: bool = True) -> list[str]:
        results: list[str] = []
        for sig in signals or []:
            try:
                rid = self.write(sig, force_shadow=force_shadow)
            except Exception as e:  # pragma: no cover
                log.warning("signal_writer: write_batch single failure: %s", e)
                rid = None
            if rid:
                results.append(rid)
        return results

    # ---- outbox drain ----
    def drain_outbox(self) -> int:
        drained = 0
        if not _OUTBOX_DIR.exists():
            return 0
        for path in sorted(_OUTBOX_DIR.glob("*.jsonl")):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except Exception as e:
                log.warning("signal_writer: drain read failed %s: %s", path.name, e)
                continue
            all_ok = True
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    sig = json.loads(line)
                except Exception:
                    all_ok = False
                    continue
                ok, _ = _validate(sig)
                if not ok:
                    continue
                status = self._resolve_status(sig["event_type"], force_shadow=True)
                if self._upsert(self._build_row(sig, status)):
                    drained += 1
                else:
                    all_ok = False
                    break
            if all_ok:
                try:
                    path.unlink()
                except Exception as e:
                    log.warning("signal_writer: drain unlink failed %s: %s", path.name, e)
        return drained


# ---- smoke test ----
def _smoke_sample(event_id: str, event_type: str, instruments: list[str],
                  direction: dict[str, int], magnitude: float, confidence: int,
                  half_life_min: int, urls: list[str], headline: str) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "event_id": event_id, "source": event_id.split(":", 1)[0],
        "event_type": event_type, "entities": {},
        "instruments": instruments, "direction": direction,
        "magnitude": magnitude, "confidence": confidence,
        "half_life_min": half_life_min, "corroboration_count": 1,
        "provenance_urls": urls, "headline": headline,
        "headline_normalized": headline.lower(), "timestamp": now.isoformat(),
        "expires_at": (now + timedelta(minutes=3 * half_life_min)).isoformat(),
        "raw_payload": {"smoke": True},
    }


if __name__ == "__main__":  # pragma: no cover
    sample_a = _smoke_sample(
        "gdelt_events:smoke-oil-001", "oil_shock", ["ONGC", "BPCL"],
        {"ONGC": 1, "BPCL": -1}, 0.62, 73, 240,
        ["https://example.com/a", "https://example.com/b"],
        "Tanker attack near Strait of Hormuz",
    )
    sample_b = _smoke_sample(
        "rss_india:smoke-cyclone-002", "monsoon_cyclone", ["UPL", "COROMANDEL"],
        {"UPL": -1, "COROMANDEL": -1}, 0.45, 58, 720,
        ["https://example.com/c"],
        "Severe cyclonic storm headed for Andhra coast",
    )
    writer = SignalWriter()
    db_enabled = bool(getattr(writer._db, "enabled", False)) if writer._db else False
    print(f"signal_writer smoke: db_enabled={db_enabled} calib_path={writer._calib_path}")
    r1 = writer.write(sample_a)
    r2 = writer.write(sample_b)
    print(f"signal_writer smoke: r1={r1}")
    print(f"signal_writer smoke: r2={r2}")
    outbox_files = sorted(p.name for p in _OUTBOX_DIR.glob("*.jsonl")) if _OUTBOX_DIR.exists() else []
    print(f"signal_writer smoke: outbox_dir={_OUTBOX_DIR}")
    print(f"signal_writer smoke: outbox_files={len(outbox_files)} sample={outbox_files[:3]}")

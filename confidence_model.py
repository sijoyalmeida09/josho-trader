"""Confidence scoring model for Layer-4 situational signals.

Pure-math module (stdlib only). Implements SITUATIONAL_SPEC.md sec 5 verbatim:

    confidence = clamp(0, 100, base * reliability * decay * corroboration * hit_rate)

No DB writes, no ingestion, no mapping. Takes a classified event (Agent 8 +
Agent 9 already attached base_magnitude / instruments / direction) and a
corroboration count (Agent 12) and returns a scored, decayed signal envelope.

Public contract (frozen, see spec sec 10):

    class ConfidenceModel:
        def __init__(self, calibration_path: Path | None = None): ...
        def score(self, classified_event: dict, corroboration_count: int = 1) -> dict
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants from SITUATIONAL_SPEC.md sec 4 and sec 5
# ---------------------------------------------------------------------------

SOURCE_RELIABILITY: dict[str, float] = {
    "gdelt_events": 0.55,
    "gdelt_gkg":    0.50,
    "rss_india":    0.70,
    "filings":      0.95,
    "eonet":        0.90,
    "usgs":         0.95,
    "ofac":         0.95,
    "macro":        0.80,
    "manual":       0.90,
}

HALF_LIFE_DEFAULT_MIN: dict[str, int] = {
    "oil_shock":           240,
    "defense_flashpoint":  360,
    "sanction":           1440,
    "monsoon_cyclone":     720,
    "flood_quake":         180,
    "rate_decision":      2880,
    "metals_supply":       480,
    "fx_move":              90,
    "telecom_disruption":  120,
    "agri_supply":        1440,
    "tech_supply":         720,
    "corporate_action":    180,
    "policy_shift":       1440,
    "geopolitical_other":  360,
}

# Defensive fallbacks for unknown values.
_DEFAULT_RELIABILITY = 0.50
_DEFAULT_HALF_LIFE_MIN = 360
_DEFAULT_EVENT_TYPE = "geopolitical_other"
_DEFAULT_BASE_MAGNITUDE = 0.3

# Hit-rate clipping per spec sec 5.
_HIT_RATE_MIN = 0.5
_HIT_RATE_MAX = 1.5

# Corroboration cap per spec sec 5 (caps at 1.6 for 5+ sources).
_CORROBORATION_STEP = 0.15
_CORROBORATION_CAP_N = 4

# Default calibration path (relative to this file's directory).
_DEFAULT_CALIBRATION_REL = Path("backtest") / "calibration.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: Any) -> datetime:
    """Parse ISO-8601 timestamp; fall back to now() on any failure."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        ts = value.strip()
        # Accept trailing 'Z' (Python <3.11 fromisoformat does not).
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(ts)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            logger.debug("unparseable timestamp %r; using now()", value)
    return _utcnow()


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


# ---------------------------------------------------------------------------
# Confidence model
# ---------------------------------------------------------------------------

class ConfidenceModel:
    """Compute confidence/magnitude/expiry for a classified event.

    Calibration file (optional) is a JSON object:
        { "oil_shock": 0.62, "defense_flashpoint": 0.55, ... }
    Missing/unreadable → defaults to 1.0 for every event_type. Values clipped
    to [0.5, 1.5] per spec sec 5.
    """

    def __init__(self, calibration_path: Path | None = None) -> None:
        self.calibration_path = self._resolve_calibration_path(calibration_path)
        self.hit_rates: dict[str, float] = self._load_calibration(self.calibration_path)

    # ----- public API ------------------------------------------------------

    def score(self, classified_event: dict, corroboration_count: int = 1) -> dict:
        """Score a classified event. Returns dict per spec sec 10."""
        event_type = classified_event.get("event_type") or _DEFAULT_EVENT_TYPE
        source = classified_event.get("source") or "manual"

        base_magnitude = self._coerce_float(
            classified_event.get("base_magnitude"),
            _DEFAULT_BASE_MAGNITUDE,
        )
        base_magnitude = _clamp(base_magnitude, 0.0, 1.0)

        reliability_weight = SOURCE_RELIABILITY.get(source, _DEFAULT_RELIABILITY)
        half_life_min = HALF_LIFE_DEFAULT_MIN.get(event_type, _DEFAULT_HALF_LIFE_MIN)

        created_at = _parse_timestamp(classified_event.get("timestamp"))
        now = _utcnow()
        elapsed_min = (now - created_at).total_seconds() / 60.0

        if elapsed_min < 0:
            # Future-stamped event: do not reward, do not penalize beyond 1.0.
            decay = 1.0
        else:
            decay = 0.5 ** (elapsed_min / half_life_min)

        corroboration_count = max(int(corroboration_count or 1), 1)
        corroboration = 1.0 + _CORROBORATION_STEP * min(
            corroboration_count - 1, _CORROBORATION_CAP_N
        )

        raw_hit_rate = self.hit_rates.get(event_type, 1.0)
        hit_rate = _clamp(float(raw_hit_rate), _HIT_RATE_MIN, _HIT_RATE_MAX)

        base = base_magnitude * 100.0
        raw_score = base * reliability_weight * decay * corroboration * hit_rate
        confidence_int = int(round(_clamp(raw_score, 0.0, 100.0)))

        final_magnitude = _clamp(base_magnitude * decay, 0.0, 1.0)
        expires_at = created_at + timedelta(minutes=half_life_min * 3)

        explanation = (
            f"base={base:.0f}, src={source}({reliability_weight:.2f}), "
            f"decay={decay:.2f}(elapsed={elapsed_min:.0f}m), "
            f"corrob={corroboration:.2f}, hit={hit_rate:.2f} -> {confidence_int}"
        )

        return {
            "confidence": confidence_int,
            "magnitude": round(final_magnitude, 3),
            "half_life_min": half_life_min,
            "expires_at": expires_at,
            "breakdown": {
                "base": round(base, 2),
                "reliability_weight": reliability_weight,
                "decay": round(decay, 4),
                "corroboration": round(corroboration, 3),
                "hit_rate": round(hit_rate, 3),
                "raw_score": round(raw_score, 2),
                "explanation": explanation,
            },
        }

    # ----- calibration -----------------------------------------------------

    @staticmethod
    def _resolve_calibration_path(p: Path | None) -> Path:
        if p is not None:
            return Path(p)
        return Path(__file__).resolve().parent / _DEFAULT_CALIBRATION_REL

    @staticmethod
    def _load_calibration(path: Path) -> dict[str, float]:
        """Return event_type -> hit_rate. Missing/unreadable → default 1.0."""
        fallback: dict[str, float] = defaultdict(lambda: 1.0)

        if not path.exists():
            logger.info(
                "confidence_model: calibration file not found at %s; "
                "defaulting all event_types to hit_rate=1.0",
                path,
            )
            return fallback

        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.info(
                "confidence_model: failed to read calibration at %s (%s); "
                "defaulting hit_rate=1.0",
                path,
                exc,
            )
            return fallback

        if not isinstance(data, dict):
            logger.info(
                "confidence_model: calibration JSON is not an object; "
                "defaulting hit_rate=1.0"
            )
            return fallback

        loaded: dict[str, float] = defaultdict(lambda: 1.0)
        for k, v in data.items():
            try:
                clipped = _clamp(float(v), _HIT_RATE_MIN, _HIT_RATE_MAX)
            except (TypeError, ValueError):
                continue
            loaded[str(k)] = clipped
        return loaded

    # ----- internal --------------------------------------------------------

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


# ---------------------------------------------------------------------------
# CLI (optional)
# ---------------------------------------------------------------------------

def score_cli(argv: list[str] | None = None) -> int:
    """Read JSON lines from stdin (or --input), emit enriched lines on stdout."""
    parser = argparse.ArgumentParser(
        description="Score classified events from JSON lines."
    )
    parser.add_argument("--input", type=Path, default=None,
                        help="Path to JSONL file (default: stdin).")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="Path to calibration.json (default: backtest/calibration.json).")
    parser.add_argument("--corroboration", type=int, default=1,
                        help="Default corroboration_count for each event.")
    args = parser.parse_args(argv)

    model = ConfidenceModel(calibration_path=args.calibration)

    if args.input:
        fh = args.input.open("r", encoding="utf-8")
    else:
        fh = sys.stdin

    try:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("skipping malformed line: %s", exc)
                continue
            corrob = int(event.get("corroboration_count") or args.corroboration)
            result = model.score(event, corroboration_count=corrob)
            enriched = {**event, **{
                "confidence": result["confidence"],
                "magnitude": result["magnitude"],
                "half_life_min": result["half_life_min"],
                "expires_at": result["expires_at"].isoformat(),
                "breakdown": result["breakdown"],
            }}
            sys.stdout.write(json.dumps(enriched, default=str) + "\n")
    finally:
        if args.input:
            fh.close()
    return 0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    model = ConfidenceModel()
    now = _utcnow()

    samples = [
        {
            "label": "fresh OFAC sanction (high reliability, fresh)",
            "event": {
                "event_id": "ofac:demo:1",
                "source": "ofac",
                "event_type": "sanction",
                "entities": {"actors": ["Russia"], "commodities": ["energy"]},
                "base_magnitude": 0.7,
                "timestamp": now.isoformat(),
                "headline": "OFAC adds 12 Russian energy entities to SDN.",
            },
            "corroboration": 1,
        },
        {
            "label": "RSS oil shock, 2h old, 3 corroborators",
            "event": {
                "event_id": "rss:demo:2",
                "source": "rss_india",
                "event_type": "oil_shock",
                "entities": {"locations": ["Strait of Hormuz"]},
                "base_magnitude": 0.55,
                "timestamp": (now - timedelta(hours=2)).isoformat(),
                "headline": "Tanker incident in Strait of Hormuz; Brent up 4%.",
            },
            "corroboration": 3,
        },
        {
            "label": "GDELT stale fx_move (12h old, low reliability, solo)",
            "event": {
                "event_id": "gdelt:demo:3",
                "source": "gdelt_events",
                "event_type": "fx_move",
                "entities": {"actors": ["USD", "INR"]},
                "base_magnitude": 0.4,
                "timestamp": (now - timedelta(hours=12)).isoformat(),
                "headline": "USD/INR opens 0.9% weaker on Asia session.",
            },
            "corroboration": 1,
        },
    ]

    print("=" * 78)
    print("ConfidenceModel smoke test")
    print(f"calibration_path: {model.calibration_path}")
    print("=" * 78)
    for s in samples:
        result = model.score(s["event"], corroboration_count=s["corroboration"])
        print(f"\n[{s['label']}]")
        print(f"  headline       : {s['event']['headline']}")
        print(f"  confidence     : {result['confidence']}")
        print(f"  magnitude      : {result['magnitude']}")
        print(f"  half_life_min  : {result['half_life_min']}")
        print(f"  expires_at     : {result['expires_at'].isoformat()}")
        print(f"  breakdown      : {result['breakdown']['explanation']}")
    print("=" * 78)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--cli", "score"):
        sys.exit(score_cli(sys.argv[2:]))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _smoke_test()

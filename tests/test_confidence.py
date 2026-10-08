"""Smoke tests for confidence_model.py — Agent 10.

Run:
    python -m unittest automation/josho-trader/tests/test_confidence.py
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from confidence_model import ConfidenceModel  # noqa: E402


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ConfidenceModelTest(unittest.TestCase):
    """Pure-math properties of ConfidenceModel.score()."""

    def setUp(self) -> None:
        self.model = ConfidenceModel()

    def test_fresh_signal_decay_near_one(self) -> None:
        ev = {
            "event_id": "test:fresh:1",
            "source": "ofac",
            "event_type": "sanction",
            "base_magnitude": 0.7,
            "timestamp": _now().isoformat(),
        }
        result = self.model.score(ev, corroboration_count=1)
        decay = result["breakdown"]["decay"]
        # Fresh = ~1.0; allow tiny clock skew slack
        self.assertGreaterEqual(decay, 0.99)
        self.assertLessEqual(decay, 1.0001)

    def test_one_day_old_oil_shock_decay(self) -> None:
        # oil_shock half_life = 240 min. 1 day = 1440 min. decay = 0.5^(1440/240) = 0.015625.
        ev = {
            "event_id": "test:old:1",
            "source": "rss_india",
            "event_type": "oil_shock",
            "base_magnitude": 0.5,
            "timestamp": (_now() - timedelta(days=1)).isoformat(),
        }
        result = self.model.score(ev, corroboration_count=1)
        decay = result["breakdown"]["decay"]
        self.assertAlmostEqual(decay, 0.5 ** (1440 / 240), places=3)
        # confidence should be near zero with such heavy decay
        self.assertLess(result["confidence"], 5)

    def test_corroboration_three_multiplies_by_about_1_3(self) -> None:
        ts = _now().isoformat()
        ev = {
            "event_id": "test:corr:1",
            "source": "rss_india",
            "event_type": "oil_shock",
            "base_magnitude": 0.5,
            "timestamp": ts,
        }
        solo = self.model.score(ev, corroboration_count=1)
        triple = self.model.score(ev, corroboration_count=3)
        # corroboration factor = 1 + 0.15 * min(3-1, 4) = 1.30
        self.assertAlmostEqual(triple["breakdown"]["corroboration"], 1.30, places=3)
        # raw_score (pre-clamp) ratio should be ~1.30
        ratio = triple["breakdown"]["raw_score"] / max(solo["breakdown"]["raw_score"], 1e-9)
        self.assertAlmostEqual(ratio, 1.30, places=2)

    def test_confidence_clamped_to_0_100(self) -> None:
        # Force a huge base_magnitude (will be clamped to 1.0).
        ev = {
            "event_id": "test:clamp:1",
            "source": "filings",     # reliability 0.95 — highest
            "event_type": "sanction",
            "base_magnitude": 999.0,  # clamped to 1.0
            "timestamp": _now().isoformat(),
        }
        result = self.model.score(ev, corroboration_count=10)  # caps at 1.60
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 100)

    def test_negative_base_clamped_to_zero(self) -> None:
        ev = {
            "event_id": "test:clamp:2",
            "source": "rss_india",
            "event_type": "oil_shock",
            "base_magnitude": -1.0,
            "timestamp": _now().isoformat(),
        }
        result = self.model.score(ev)
        self.assertEqual(result["confidence"], 0)


if __name__ == "__main__":
    unittest.main()

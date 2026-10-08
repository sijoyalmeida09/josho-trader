"""Smoke tests for classifier.py — Agent 8.

Run:
    python -m unittest automation/josho-trader/tests/test_classifier.py
"""

from __future__ import annotations

import os
import sys
import unittest

# Allow `import classifier` to resolve when running from repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import classifier  # noqa: E402


class ClassifierSmokeTest(unittest.TestCase):
    """Three minimal classifier acceptance tests."""

    def test_oil_event_classified_as_oil_shock(self) -> None:
        raw = {
            "event_id": "test:oil:1",
            "source": "gdelt_events",
            "headline": "Iran threatens Strait of Hormuz; Brent crude spikes 5%",
            "url": "https://example.com/oil",
            "timestamp": "2026-06-17T00:00:00Z",
            "raw_payload": {"themes": ["ENERGY_OIL"]},
            "entities_hint": {"actors": ["Iran"], "themes": ["energy"]},
        }
        out = classifier.classify(raw)
        self.assertIsNotNone(out, "oil event must classify, not return None")
        assert out is not None  # for type checker
        self.assertEqual(out["event_type"], "oil_shock")
        # entities should include at least one location or commodity
        ents = out["entities"]
        self.assertTrue(
            ents.get("locations") or ents.get("commodities"),
            "should pick up locations/commodities for oil event",
        )

    def test_usgs_quake_classified_as_flood_quake(self) -> None:
        raw = {
            "event_id": "test:usgs:1",
            "source": "usgs",
            "headline": "Magnitude 6.2 earthquake strikes northern India",
            "url": "https://example.com/quake",
            "timestamp": "2026-06-17T00:00:00Z",
            "raw_payload": {"magnitude": 6.2},
            "entities_hint": {"locations": ["India"], "india_relevant": True},
        }
        out = classifier.classify(raw)
        self.assertIsNotNone(out, "usgs source must trigger flood_quake prior")
        assert out is not None
        self.assertEqual(out["event_type"], "flood_quake")

    def test_no_signal_returns_none(self) -> None:
        raw = {
            "event_id": "test:noise:1",
            "source": "rss_india",
            # Empty stop-word-only headline + no entities → must drop.
            "headline": "and the it",
            "url": "https://example.com/noise",
            "timestamp": "2026-06-17T00:00:00Z",
            "raw_payload": {},
            "entities_hint": {},
        }
        out = classifier.classify(raw)
        self.assertIsNone(out, "empty/noise event must return None")

    def test_missing_headline_returns_none(self) -> None:
        raw = {"event_id": "test:bad:1", "source": "rss_india", "timestamp": "2026-06-17T00:00:00Z"}
        self.assertIsNone(classifier.classify(raw))


if __name__ == "__main__":
    unittest.main()

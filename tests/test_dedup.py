"""Smoke tests for dedup.py — Agent 12.

Run:
    python -m unittest automation/josho-trader/tests/test_dedup.py
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

import dedup  # noqa: E402


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ev(**kw) -> dict:
    base = {
        "entities": {},
        "corroboration_count": 1,
        "provenance_urls": [kw.get("url", "")],
    }
    return {**base, **kw}


class DedupSmokeTest(unittest.TestCase):
    """Three minimal dedup acceptance tests."""

    def test_two_similar_events_within_6h_are_deduped(self) -> None:
        now = _now()
        a = _ev(
            event_id="rss:hormuz:1", source="rss_india", event_type="oil_shock",
            instruments=["ONGC", "RELIANCE", "BPCL"],
            direction={"ONGC": 1, "RELIANCE": 1, "BPCL": -1},
            magnitude=0.55, confidence=62, half_life_min=240,
            headline_normalized="tanker incident strait of hormuz brent up 4 percent",
            headline="Tanker incident in Strait of Hormuz, Brent up 4%",
            url="https://moneycontrol.com/news/hormuz",
            timestamp=now.isoformat(),
        )
        b = _ev(
            event_id="gdelt:hormuz:2", source="gdelt_events", event_type="oil_shock",
            instruments=["ONGC", "RELIANCE", "IOC"],
            direction={"ONGC": 1, "RELIANCE": 1, "IOC": 1},
            magnitude=0.50, confidence=55, half_life_min=240,
            headline_normalized="strait of hormuz tanker incident sends brent crude higher",
            headline="Strait of Hormuz tanker incident sends Brent crude higher",
            url="https://gdelt.example/hormuz",
            timestamp=(now - timedelta(minutes=30)).isoformat(),
        )
        out = dedup.dedup([a, b])
        self.assertEqual(len(out), 1, "the two similar events must collapse to one")
        merged = out[0]
        self.assertGreaterEqual(merged["corroboration_count"], 2)
        self.assertGreaterEqual(len(merged["provenance_urls"]), 2)

    def test_low_confidence_events_are_filtered(self) -> None:
        now = _now()
        keep = _ev(
            event_id="ok:1", source="rss_india", event_type="oil_shock",
            instruments=["ONGC"], direction={"ONGC": 1},
            magnitude=0.5, confidence=60, half_life_min=240,
            headline_normalized="ongc rallies on oil spike",
            headline="ONGC rallies on oil spike",
            url="https://ex.com/ok", timestamp=now.isoformat(),
        )
        drop_low_conf = _ev(
            event_id="low:1", source="rss_india", event_type="corporate_action",
            instruments=["RELIANCE"], direction={"RELIANCE": 1},
            magnitude=0.10, confidence=8, half_life_min=180,
            headline_normalized="reliance update", headline="Reliance update",
            url="https://ex.com/low", timestamp=now.isoformat(),
        )
        drop_noise = _ev(
            event_id="noise:1", source="rss_india", event_type="corporate_action",
            instruments=["RELIANCE"], direction={"RELIANCE": 1},
            magnitude=0.40, confidence=50, half_life_min=180,
            headline_normalized="reliance stock to watch this week",
            headline="Reliance stock to watch this week",
            url="https://ex.com/noise", timestamp=now.isoformat(),
        )
        out = dedup.noise_filter([keep, drop_low_conf, drop_noise])
        ids = {e["event_id"] for e in out}
        self.assertIn("ok:1", ids)
        self.assertNotIn("low:1", ids, "below CONFIDENCE_FLOOR must be dropped")
        self.assertNotIn("noise:1", ids, "'stock to watch' pattern must be dropped")

    def test_transitive_grouping_via_union_find(self) -> None:
        """A↔B match and B↔C match (but A↔C don't directly) → all three merge."""
        now = _now()
        a = _ev(
            event_id="t:a", source="rss_india", event_type="oil_shock",
            instruments=["ONGC", "RELIANCE", "BPCL", "HPCL"],
            direction={"ONGC": 1}, magnitude=0.5, confidence=60, half_life_min=240,
            headline_normalized="oil rally on hormuz tanker incident reports",
            headline="Oil rally on Hormuz tanker incident reports",
            url="https://ex.com/a", timestamp=now.isoformat(),
        )
        b = _ev(
            event_id="t:b", source="gdelt_events", event_type="oil_shock",
            instruments=["ONGC", "RELIANCE", "BPCL", "OIL"],
            direction={"ONGC": 1}, magnitude=0.5, confidence=55, half_life_min=240,
            headline_normalized="oil tanker incident hormuz brent rally crude",
            headline="Oil tanker incident Hormuz Brent rally",
            url="https://ex.com/b", timestamp=(now - timedelta(minutes=30)).isoformat(),
        )
        c = _ev(
            event_id="t:c", source="filings", event_type="oil_shock",
            instruments=["RELIANCE", "BPCL", "OIL", "ONGC"],
            direction={"ONGC": 1}, magnitude=0.5, confidence=58, half_life_min=240,
            headline_normalized="brent rally crude oil higher hormuz incident reports",
            headline="Brent rally crude oil higher Hormuz incident",
            url="https://ex.com/c", timestamp=(now - timedelta(minutes=60)).isoformat(),
        )
        out = dedup.dedup([a, b, c])
        self.assertEqual(len(out), 1, "transitive A-B-C must collapse to one")
        merged = out[0]
        self.assertEqual(merged["corroboration_count"], 3)


if __name__ == "__main__":
    unittest.main()

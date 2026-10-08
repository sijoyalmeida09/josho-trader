"""Dedup & noise filter for Layer-4 situational signals (Agent 12).

Implements SITUATIONAL_SPEC.md sec 1 rule 8 ("dedup before write"), sec 3
schema, sec 10 frozen contract, and sec 12 failure modes (malformed events
never crash).

Public contract:
    def dedup(events: list[dict]) -> list[dict]
    def noise_filter(events: list[dict]) -> list[dict]
    def dedup_and_filter(events: list[dict]) -> list[dict]

Dedup algorithm:
  1. Bucket by (event_type, calendar_day_of_timestamp_UTC).
  2. Within a bucket, two events MATCH iff ALL of:
       jaccard(instruments) >= 0.5
       jaccard(words(headline_normalized)) >= 0.4
       |time_delta_minutes| <= 360 (6h)
  3. Transitive grouping via Union-Find.
  4. Each group collapses to one merged event.
  5. Confidence is recomputed via ConfidenceModel so corroboration boost
     is reflected (closed-form fallback if import fails).

Stdlib only. No DB, no ingestion.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# --- Tunables (spec sec 10 algorithm + sec 12 hygiene) ----------------------

INSTRUMENT_JACCARD_MIN: float = 0.5
HEADLINE_JACCARD_MIN: float = 0.4
TIME_DELTA_MAX_MIN: float = 360.0          # 6 hours

CONFIDENCE_FLOOR: int = 15
MAGNITUDE_FLOOR: float = 0.05
HALF_LIFE_DECAY_MULTIPLIER: int = 4        # >4 half-lives = <6% energy left
HARD_AGE_CAP_DAYS: int = 7

# Low-info headlines from news scrapers (cite spec sec 12 noise hygiene).
NOISE_HEADLINE_PATTERNS: tuple[str, ...] = (
    "weekly market update", "market wrap", "market open", "market close",
    "daily wrap", "technical analysis", "stock to watch", "stocks to watch",
    "expert view", "buy zone", "target price",
)

# Tokenize on word chars; drop a tight stopword list so jaccard is over
# content words. Over-aggressive stopwords would inflate similarity.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "with",
    "was", "were", "after", "amid", "over", "this", "their", "they", "but",
    "if", "into", "than", "then", "so",
})


# --- Time + similarity primitives -------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse ISO-8601 timestamp. None on failure (caller decides)."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value:
        return None
    ts = value.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _word_set(text: str | None) -> frozenset[str]:
    if not text or not isinstance(text, str):
        return frozenset()
    return frozenset(
        t for t in _TOKEN_RE.findall(text.lower())
        if t and t not in _STOPWORDS
    )


def _jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    sa = a if isinstance(a, (set, frozenset)) else set(a)
    sb = b if isinstance(b, (set, frozenset)) else set(b)
    if not sa and not sb:
        # Both empty → 0.0 (empty events should not auto-merge).
        return 0.0
    union = len(sa | sb)
    return (len(sa & sb) / union) if union else 0.0


def _instruments_set(event: dict) -> frozenset[str]:
    raw = event.get("instruments") or []
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(str(x).upper() for x in raw if x)


# --- Union-Find for transitive grouping -------------------------------------

class _UnionFind:
    """Disjoint-set with iterative path compression. Stable: lower index wins."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        # Lower index becomes parent → deterministic group representatives.
        if ri < rj:
            self.parent[rj] = ri
        else:
            self.parent[ri] = rj

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return out


# --- Validation -------------------------------------------------------------

_REQUIRED_KEYS = ("event_id", "source", "event_type", "timestamp")


def _is_well_formed(ev: Any) -> bool:
    if not isinstance(ev, dict):
        return False
    if any(not ev.get(k) for k in _REQUIRED_KEYS):
        return False
    return _parse_timestamp(ev.get("timestamp")) is not None


# --- Confidence recompute (soft dep on confidence_model) --------------------

def _import_confidence_bits() -> tuple[Any, dict[str, float]]:
    """Import ConfidenceModel + SOURCE_RELIABILITY with both flat + relative
    import paths. Returns (ModelClass_or_None, source_reliability_map)."""
    try:
        from confidence_model import ConfidenceModel, SOURCE_RELIABILITY  # type: ignore
        return ConfidenceModel, dict(SOURCE_RELIABILITY)
    except ImportError:
        pass
    try:
        from .confidence_model import ConfidenceModel, SOURCE_RELIABILITY  # type: ignore
        return ConfidenceModel, dict(SOURCE_RELIABILITY)
    except (ImportError, ValueError):
        return None, {}


_ConfidenceModel, _SOURCE_RELIABILITY = _import_confidence_bits()


def _recompute_confidence(merged: dict, corroboration_count: int) -> int:
    """Recompute confidence so the new corroboration count is reflected.
    Uses ConfidenceModel if available; falls back to spec sec 5 closed form."""
    old_conf = int(merged.get("confidence") or 0)
    if _ConfidenceModel is not None:
        try:
            scored = _ConfidenceModel().score(
                merged, corroboration_count=corroboration_count
            )
            return int(scored["confidence"])
        except Exception as exc:  # noqa: BLE001 — never crash dedup on scoring.
            logger.debug("ConfidenceModel.score failed (%s); using fallback", exc)
    # Fallback: corroboration factor from spec sec 5: 1 + 0.15 * min(n-1, 4).
    n = max(1, corroboration_count)
    return int(min(100, round(old_conf * (1.0 + 0.15 * min(n - 1, 4)))))


# --- Group merge ------------------------------------------------------------

def _merge_directions(events: list[dict]) -> dict[str, int]:
    """Per-symbol sign: higher-magnitude entry wins; tie → higher confidence."""
    # symbol -> (sign, magnitude, confidence)
    chosen: dict[str, tuple[int, float, int]] = {}
    for ev in events:
        direction = ev.get("direction") or {}
        if not isinstance(direction, dict):
            continue
        mag = float(ev.get("magnitude") or 0.0)
        conf = int(ev.get("confidence") or 0)
        for sym, sign_raw in direction.items():
            try:
                sign = int(sign_raw)
            except (TypeError, ValueError):
                continue
            key = str(sym).upper()
            cur = chosen.get(key)
            if cur is None or mag > cur[1] or (mag == cur[1] and conf > cur[2]):
                chosen[key] = (sign, mag, conf)
    return {sym: val[0] for sym, val in chosen.items()}


def _merge_entities(events: list[dict]) -> dict[str, list[str]]:
    """Union each entity field, preserving first-seen order."""
    out: dict[str, list[str]] = {}
    for ev in events:
        ents = ev.get("entities") or {}
        if not isinstance(ents, dict):
            continue
        for field, values in ents.items():
            if not isinstance(values, (list, tuple, set, frozenset)):
                continue
            bucket = out.setdefault(field, [])
            seen = set(bucket)
            for v in values:
                if v is not None and v not in seen:
                    bucket.append(v)
                    seen.add(v)
    return out


def _ordered_union_urls(events: list[dict]) -> list[str]:
    """Union provenance_urls + url, first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for ev in events:
        urls = ev.get("provenance_urls") or []
        if isinstance(urls, (list, tuple)):
            for u in urls:
                if u and u not in seen:
                    out.append(u)
                    seen.add(u)
        u = ev.get("url")
        if u and u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _pick_headline(events: list[dict]) -> str:
    """Pick the headline from the highest-reliability source (spec sec 4)."""
    best: tuple[float, str] | None = None
    for ev in events:
        rel = _SOURCE_RELIABILITY.get(ev.get("source") or "", 0.5)
        headline = ev.get("headline_normalized") or ev.get("headline") or ""
        if isinstance(headline, str) and (best is None or rel > best[0]):
            best = (rel, headline)
    return best[1] if best else ""


def _merge_group(events: list[dict]) -> dict:
    """Collapse a similarity group into one merged event."""
    if len(events) == 1:
        solo = dict(events[0])
        solo.setdefault("corroboration_count", 1)
        solo.setdefault("provenance_urls", _ordered_union_urls([solo]))
        return solo

    # Deterministic identity: lexicographically smallest event_id wins.
    base = min(events, key=lambda e: str(e.get("event_id") or ""))
    # Highest-confidence entry supplies fall-through fields.
    highest_conf = max(events, key=lambda e: int(e.get("confidence") or 0))

    distinct_sources = {
        str(ev.get("source") or "") for ev in events if ev.get("source")
    }
    corroboration_count = max(len(distinct_sources), 1)

    # Union of instruments, preserving first-seen order.
    instruments_seen: set[str] = set()
    instruments_ordered: list[str] = []
    for ev in events:
        for inst in (ev.get("instruments") or []):
            sym = str(inst).upper()
            if sym and sym not in instruments_seen:
                instruments_seen.add(sym)
                instruments_ordered.append(sym)

    merged: dict[str, Any] = {**highest_conf}
    merged["event_id"] = base.get("event_id")
    merged["source"] = "merged"
    merged["confidence"] = max(int(ev.get("confidence") or 0) for ev in events)
    merged["magnitude"] = max(float(ev.get("magnitude") or 0.0) for ev in events)
    merged["corroboration_count"] = corroboration_count
    merged["provenance_urls"] = _ordered_union_urls(events)
    merged["instruments"] = instruments_ordered
    merged["direction"] = _merge_directions(events)
    merged["entities"] = _merge_entities(events)
    merged["headline_normalized"] = (
        _pick_headline(events) or merged.get("headline_normalized", "")
    )
    # Anchor decay to the first sighting (base event's timestamp).
    base_ts = _parse_timestamp(base.get("timestamp"))
    if base_ts is not None:
        merged["timestamp"] = base_ts.isoformat()
    merged["merged_sources"] = sorted(distinct_sources)

    # Recompute confidence so the new corroboration count is reflected.
    merged["confidence"] = _recompute_confidence(merged, corroboration_count)
    return merged


# --- Public API -------------------------------------------------------------

def dedup(events: list[dict]) -> list[dict]:
    """Collapse same real-world event from multiple sources into one signal.

    Merges provenance_urls, sums distinct sources into corroboration_count,
    picks max confidence/magnitude, recomputes confidence with new
    corroboration. Returns a deduped list (possibly fewer entries)."""
    if not events:
        return []

    # Defensive: skip malformed entries; never crash dedup on bad data.
    clean: list[dict] = []
    for ev in events:
        if _is_well_formed(ev):
            clean.append(ev)
        else:
            logger.warning("dedup: skipping malformed event: %r", ev)
    if not clean:
        return []

    # Bucket by (event_type, day); events from different days/types never merge.
    buckets: dict[tuple[str, str], list[int]] = {}
    parsed_ts: list[datetime] = []
    for idx, ev in enumerate(clean):
        ts = _parse_timestamp(ev["timestamp"]) or _utcnow()
        parsed_ts.append(ts)
        key = (str(ev.get("event_type") or ""), ts.date().isoformat())
        buckets.setdefault(key, []).append(idx)

    merged_groups: list[list[dict]] = []
    for indices in buckets.values():
        if len(indices) == 1:
            merged_groups.append([clean[indices[0]]])
            continue

        uf = _UnionFind(len(indices))

        # Precompute per-bucket features once (cheap pairwise compare).
        feats: list[tuple[frozenset[str], frozenset[str], datetime]] = []
        for global_i in indices:
            ev = clean[global_i]
            feats.append((
                _instruments_set(ev),
                _word_set(ev.get("headline_normalized") or ev.get("headline")),
                parsed_ts[global_i],
            ))

        # O(n^2) inside the bucket — buckets are small (same type, same day).
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                inst_a, words_a, ts_a = feats[i]
                inst_b, words_b, ts_b = feats[j]
                if _jaccard(inst_a, inst_b) < INSTRUMENT_JACCARD_MIN:
                    continue
                if _jaccard(words_a, words_b) < HEADLINE_JACCARD_MIN:
                    continue
                if abs((ts_a - ts_b).total_seconds()) / 60.0 > TIME_DELTA_MAX_MIN:
                    continue
                uf.union(i, j)

        for locals_in_group in uf.groups().values():
            merged_groups.append([clean[indices[li]] for li in locals_in_group])

    merged_events = [_merge_group(g) for g in merged_groups]
    # Deterministic ordering — same input always produces same output.
    merged_events.sort(key=lambda e: str(e.get("event_id") or ""))
    return merged_events


def noise_filter(events: list[dict]) -> list[dict]:
    """Drop low-signal items per spec sec 12 hygiene.

    Rules:
      1. confidence < CONFIDENCE_FLOOR              → drop
      2. zero instruments                           → drop (no target)
      3. magnitude < MAGNITUDE_FLOOR                → drop (too small)
      4. age > half_life_min * HALF_LIFE_DECAY_MULT → drop (decayed)
      5. headline matches NOISE_HEADLINE_PATTERNS   → drop (low-info)
      6. age > HARD_AGE_CAP_DAYS                    → drop (data hygiene)
    """
    if not events:
        return []

    now = _utcnow()
    out: list[dict] = []
    for ev in events:
        if not isinstance(ev, dict):
            logger.warning("noise_filter: skipping non-dict: %r", ev)
            continue

        confidence = int(ev.get("confidence") or 0)
        if confidence < CONFIDENCE_FLOOR:
            continue

        instruments = ev.get("instruments") or []
        if not isinstance(instruments, (list, tuple)) or len(instruments) == 0:
            continue

        magnitude = float(ev.get("magnitude") or 0.0)
        if magnitude < MAGNITUDE_FLOOR:
            continue

        ts = _parse_timestamp(ev.get("timestamp"))
        if ts is not None:
            age_min = (now - ts).total_seconds() / 60.0
            half_life = int(ev.get("half_life_min") or 360)
            if age_min > half_life * HALF_LIFE_DECAY_MULTIPLIER:
                continue
            if age_min > HARD_AGE_CAP_DAYS * 24 * 60:
                continue

        headline = (ev.get("headline_normalized") or ev.get("headline") or "")
        if isinstance(headline, str):
            hl_lower = headline.lower()
            if any(p in hl_lower for p in NOISE_HEADLINE_PATTERNS):
                continue

        out.append(ev)
    return out


def dedup_and_filter(events: list[dict]) -> list[dict]:
    """Convenience: noise_filter then dedup. Used by the orchestrator."""
    return dedup(noise_filter(events))


# --- CLI --------------------------------------------------------------------

def dedup_cli(argv: list[str] | None = None) -> int:
    """Read JSON lines from stdin (or --input), emit deduped lines on stdout."""
    parser = argparse.ArgumentParser(
        description="Dedup and noise-filter scored events from JSON lines."
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Path to JSONL file (default: stdin).")
    parser.add_argument("--filter-only", action="store_true",
                        help="Run noise_filter only (skip dedup).")
    parser.add_argument("--dedup-only", action="store_true",
                        help="Run dedup only (skip noise_filter).")
    args = parser.parse_args(argv)

    fh = open(args.input, "r", encoding="utf-8") if args.input else sys.stdin
    try:
        events: list[dict] = []
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("dedup_cli: skipping malformed line: %s", exc)
    finally:
        if args.input:
            fh.close()

    if args.filter_only:
        result = noise_filter(events)
    elif args.dedup_only:
        result = dedup(events)
    else:
        result = dedup_and_filter(events)

    for ev in result:
        sys.stdout.write(json.dumps(ev, default=str) + "\n")
    return 0


# --- Smoke test -------------------------------------------------------------

def _smoke_test() -> None:
    now = _utcnow()

    def _ev(**kw: Any) -> dict:
        """Build a sample event with sensible defaults."""
        base = {
            "entities": {}, "corroboration_count": 1,
            "provenance_urls": [kw.get("url", "")],
        }
        return {**base, **kw}

    samples = [
        _ev(  # A — Hormuz oil shock from RSS
            event_id="rss:moneycontrol:abc123", source="rss_india",
            event_type="oil_shock",
            instruments=["ONGC", "RELIANCE", "BPCL"],
            direction={"ONGC": 1, "RELIANCE": 1, "BPCL": -1},
            magnitude=0.55, confidence=62, half_life_min=240,
            headline_normalized="Tanker incident in Strait of Hormuz Brent up 4 percent",
            headline="Tanker incident in Strait of Hormuz, Brent up 4%",
            url="https://moneycontrol.com/news/hormuz",
            timestamp=now.isoformat(),
        ),
        _ev(  # B — same Hormuz event from GDELT, ~30 min later
            event_id="gdelt:7892341", source="gdelt_events",
            event_type="oil_shock",
            instruments=["ONGC", "RELIANCE", "IOC"],
            direction={"ONGC": 1, "RELIANCE": 1, "IOC": 1},
            magnitude=0.50, confidence=55, half_life_min=240,
            headline_normalized="Strait of Hormuz tanker incident sends Brent crude higher",
            headline="Strait of Hormuz tanker incident sends Brent crude higher",
            url="https://gdelt.example/7892341",
            timestamp=(now - timedelta(minutes=30)).isoformat(),
        ),
        _ev(  # C — legit standalone fx_move (should survive)
            event_id="macro:fred:usdinr:1", source="macro",
            event_type="fx_move",
            instruments=["INFY", "TCS", "HCLTECH"],
            direction={"INFY": 1, "TCS": 1, "HCLTECH": 1},
            magnitude=0.30, confidence=48, half_life_min=90,
            headline_normalized="USD INR weakens 0.9 percent on Asia open IT exporters benefit",
            headline="USD/INR weakens 0.9% on Asia open; IT exporters benefit",
            url="https://fred.example/usdinr",
            timestamp=now.isoformat(),
        ),
        _ev(  # D — noise (low confidence + "stock to watch" pattern)
            event_id="rss:noise:1", source="rss_india",
            event_type="corporate_action",
            instruments=["RELIANCE"], direction={"RELIANCE": 1},
            magnitude=0.10, confidence=8, half_life_min=180,
            headline_normalized="Reliance stock to watch this week",
            headline="Reliance stock to watch this week",
            url="https://example.com/noise",
            timestamp=now.isoformat(),
        ),
        _ev(  # E — too old (10 days; past 4 half-lives + 7-day hard cap)
            event_id="rss:stale:1", source="rss_india",
            event_type="fx_move",
            instruments=["INFY"], direction={"INFY": -1},
            magnitude=0.20, confidence=40, half_life_min=90,
            headline_normalized="Euro weakness pressures IT services",
            headline="Euro weakness pressures IT services",
            url="https://example.com/stale",
            timestamp=(now - timedelta(days=10)).isoformat(),
        ),
    ]

    before = len(samples)
    after_filter = noise_filter(samples)
    after_both = dedup_and_filter(samples)

    print("=" * 78)
    print("dedup.py smoke test")
    print("=" * 78)
    print(f"input events       : {before}")
    print(f"after noise_filter : {len(after_filter)}")
    print(f"after dedup+filter : {len(after_both)}")

    merged = next(
        (e for e in after_both if int(e.get("corroboration_count") or 0) > 1),
        None,
    )
    if merged is None:
        print("WARNING: no merged signal found — thresholds may be off.")
    else:
        print("\nMerged signal metadata:")
        print(f"  event_id            : {merged['event_id']}")
        print(f"  source              : {merged['source']}")
        print(f"  corroboration_count : {merged['corroboration_count']}")
        print(f"  merged_sources      : {merged.get('merged_sources')}")
        print(f"  provenance_urls     : {merged['provenance_urls']}")
        print(f"  instruments         : {merged['instruments']}")
        print(f"  direction           : {merged['direction']}")
        print(f"  confidence          : {merged['confidence']}")
        print(f"  magnitude           : {merged['magnitude']}")
        print(f"  headline_normalized : {merged['headline_normalized']}")
    print("=" * 78)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--cli", "dedup"):
        sys.exit(dedup_cli(sys.argv[2:]))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _smoke_test()

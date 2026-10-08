"""
classifier.py — Agent 8 of Layer-4 (Situational Trading)

Pure type-tagging brain. Transforms a raw ingestor event into a typed event:
    raw_event (dict)  ─►  classified_event (dict with event_type + entities)

Inputs come from any of the §4 sources (gdelt_events, gdelt_gkg, rss_india,
filings, eonet, usgs, ofac, macro). Output uses the §2 taxonomy enum.

This module is FROZEN at the contract level — see §10 of SITUATIONAL_SPEC.md.
No DB writes, no ingestion, no instrument-mapping (Agent 9), no scoring
(Agent 10), no dedup (Agent 12). Just classify.

Defensive design (§12 failure modes):
    - Missing headline           → return None
    - Empty entity surface       → return None
    - India-only types w/ hint   → return None when india_relevant=False
    - Classifier confused (low   → return None
       max score)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Iterable

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFIER_VERSION = "v1"

# All allowed event_type values — must match §2 taxonomy table exactly.
# 14 entries — any deviation breaks the downstream mapping engine.
TAXONOMY: tuple[str, ...] = (
    "oil_shock",
    "defense_flashpoint",
    "sanction",
    "monsoon_cyclone",
    "flood_quake",
    "rate_decision",
    "metals_supply",
    "fx_move",
    "telecom_disruption",
    "agri_supply",
    "tech_supply",
    "corporate_action",
    "policy_shift",
    "geopolitical_other",
)

# Keyword dictionaries — one per event_type that we score via text matching.
# Lower-cased on lookup. Order doesn't matter; matches are summed.
# NB: keep these conservative (high-precision) so we don't over-classify.
# 13 dicts (geopolitical_other handled via tie-breaker, not direct scoring).
KEYWORDS: dict[str, tuple[str, ...]] = {
    "oil_shock": (
        "oil", "crude", "brent", "wti", "opec", "iran", "hormuz",
        "strait", "refinery", "saudi", "barrel", "petroleum",
    ),
    "defense_flashpoint": (
        "border", "clash", "soldier", "terror", "attack", "loc", "lac",
        "escalation", "strike", "militant", "ceasefire", "drone",
    ),
    "sanction": (
        "sanction", "ofac", "sdn", "embargo", "blacklist", "asset freeze",
    ),
    "monsoon_cyclone": (
        "cyclone", "monsoon", "storm", "hurricane", "typhoon", "rainfall",
        "imd", "depression", "landfall",
    ),
    "flood_quake": (
        "flood", "earthquake", "quake", "tremor", "magnitude", "epicentre",
        "epicenter", "aftershock", "tsunami",
    ),
    "rate_decision": (
        "rate", "mpc", "rbi", "fomc", "ecb", "monetary policy",
        "repo", "reverse repo", "basis points", "bps", "hike", "cut",
    ),
    "metals_supply": (
        "steel", "aluminium", "aluminum", "copper", "nickel", "zinc",
        "mine", "lme", "smelter", "iron ore", "lead",
    ),
    "fx_move": (
        "rupee", "usd/inr", "dollar", "currency", "forex", "fx",
        "exchange rate", "depreciation", "appreciation",
    ),
    "telecom_disruption": (
        "cable", "outage", "fibre", "fiber", "network", "submarine",
        "jio", "airtel", "vi ", "vodafone idea", "bsnl",
    ),
    "agri_supply": (
        "wheat", "sugar", "rice", "msp", "agriculture", "farmer", "crop",
        "harvest", "fertiliser", "fertilizer", "onion", "pulses",
    ),
    "tech_supply": (
        "semi", "chip", "rare earth", "tsmc", "export curb", "wafer",
        "lithography", "foundry", "asml", "node",
    ),
    "corporate_action": (
        "merger", "acquisition", "open offer", "demerger", "board",
        "ceo", "earnings", "results", "buyback", "dividend", "qip",
    ),
    "policy_shift": (
        "budget", "gst", "fdi", "tariff", "duty", "circular", "framework",
        "regulation", "notification", "policy",
    ),
}

# Types that REQUIRE Indian relevance to be tradeable on NSE.
# If entities_hint explicitly flags india_relevant=False, drop these.
INDIA_REQUIRED_TYPES = frozenset({
    "monsoon_cyclone",
    "corporate_action",
    "policy_shift",
    "agri_supply",
})

# Words to strip during entity extraction (low signal).
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "into", "after", "before",
    "says", "said", "amid", "near", "over", "under", "this", "that",
    "will", "would", "could", "should", "have", "been", "being",
})

# Regex helpers
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s\-/]")
_CAPS_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b")  # Title Case spans
_TICKER_RE = re.compile(r"\b([A-Z]{2,8})\b")  # Probable ticker-like upper case

# Commodity vocabulary recognised in text (used to populate entities.commodities)
_COMMODITY_VOCAB = frozenset({
    "oil", "crude", "brent", "wti", "gold", "silver", "copper", "nickel",
    "zinc", "aluminium", "aluminum", "steel", "iron ore", "wheat", "sugar",
    "rice", "lithium", "uranium", "cobalt", "rupee", "dollar",
})

# Known location strings worth picking out (small, India-focused).
_LOCATION_VOCAB = frozenset({
    "india", "indian", "delhi", "mumbai", "chennai", "kolkata", "bengaluru",
    "bangalore", "hyderabad", "pune", "gujarat", "maharashtra", "kerala",
    "tamil nadu", "punjab", "haryana", "rajasthan", "uttar pradesh",
    "china", "chinese", "russia", "russian", "iran", "iranian", "saudi",
    "saudi arabia", "iraq", "ukraine", "israel", "gaza", "hormuz",
    "pakistan", "afghan", "usa", "us", "united states",
})


# ─────────────────────────────────────────────────────────────────────────────
# Entity extraction
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_headline(headline: str) -> str:
    """Lower, collapse whitespace, strip punctuation, cap at 200 chars."""
    txt = headline.lower()
    txt = _PUNCT_RE.sub(" ", txt)
    txt = _WS_RE.sub(" ", txt).strip()
    return txt[:200]


def _extract_entities(headline: str, raw_payload: dict, hint: dict) -> dict:
    """Merge entities_hint (authoritative when present) with regex-derived ones.

    Returns dict with keys: actors, locations, commodities, themes, organizations.
    Each is a list of unique strings preserving first-seen order.
    """
    actors: list[str] = []
    locations: list[str] = []
    commodities: list[str] = []
    themes: list[str] = []
    organizations: list[str] = []

    # 1) Seed from hint (ingestor knows best for its source).
    if isinstance(hint, dict):
        for k, dest in (
            ("actors", actors),
            ("locations", locations),
            ("commodities", commodities),
            ("themes", themes),
            ("organizations", organizations),
        ):
            seeded = hint.get(k) or []
            if isinstance(seeded, str):
                seeded = [seeded]
            for v in seeded:
                if v and v not in dest:
                    dest.append(str(v))

    # 2) Regex-derived fillers on the raw headline text.
    title_caps = _CAPS_RE.findall(headline)
    for span in title_caps:
        low = span.lower()
        if low in _STOPWORDS:
            continue
        # Heuristic: if it looks like a place we know, route to locations.
        if low in _LOCATION_VOCAB:
            if span not in locations:
                locations.append(span)
        else:
            # Otherwise treat as a candidate organization (Title Case noun).
            if span not in organizations:
                organizations.append(span)

    # 3) Commodity scan on lowercased text.
    lowered = headline.lower()
    for c in _COMMODITY_VOCAB:
        if c in lowered and c not in commodities:
            commodities.append(c)

    # 4) Location scan (multi-word, on lowercased text).
    for loc in _LOCATION_VOCAB:
        if loc in lowered and loc not in [x.lower() for x in locations]:
            locations.append(loc)

    # 5) Payload hooks — many ingestors stuff useful tags in raw_payload.
    if isinstance(raw_payload, dict):
        for fld in ("themes", "v2themes", "categories", "tags"):
            val = raw_payload.get(fld)
            if not val:
                continue
            if isinstance(val, str):
                val = [val]
            for v in val:
                if v and v not in themes:
                    themes.append(str(v))
        for fld in ("organizations", "orgs", "companies"):
            val = raw_payload.get(fld)
            if not val:
                continue
            if isinstance(val, str):
                val = [val]
            for v in val:
                if v and v not in organizations:
                    organizations.append(str(v))

    return {
        "actors": actors,
        "locations": locations,
        "commodities": commodities,
        "themes": themes,
        "organizations": organizations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Type scoring
# ─────────────────────────────────────────────────────────────────────────────

def _score_keyword_hits(normalized_text: str, entities: dict) -> dict[str, int]:
    """Score each candidate event_type by counting keyword hits in
    headline + entity surface. Returns {event_type: score}.
    """
    # Build a search blob: normalized headline + flattened entity strings.
    entity_blob_parts: list[str] = []
    for k in ("actors", "locations", "commodities", "organizations"):
        entity_blob_parts.extend(str(x).lower() for x in entities.get(k, []))
    themes_lower = [str(t).lower() for t in entities.get("themes", [])]
    full_blob = normalized_text + " " + " ".join(entity_blob_parts) + " " + " ".join(themes_lower)

    scores: dict[str, int] = {}
    for etype, kws in KEYWORDS.items():
        score = 0
        for kw in kws:
            if kw in full_blob:
                score += 1
            # Themes get a 2× boost when keyword appears as a tagged theme.
            if any(kw in t for t in themes_lower):
                score += 1  # additive on top of the +1 above when applicable
        scores[etype] = score
    return scores


def _source_prior(source: str, raw_payload: dict) -> str | None:
    """Strong source-based prior — when set, this OVERRIDES keyword scoring."""
    if source == "ofac":
        return "sanction"
    if source == "usgs":
        return "flood_quake"
    if source == "eonet":
        cat = ""
        if isinstance(raw_payload, dict):
            cats = raw_payload.get("categories") or raw_payload.get("category") or ""
            if isinstance(cats, list):
                cat = " ".join(str(c) for c in cats).lower()
            else:
                cat = str(cats).lower()
        if "cyclone" in cat or "storm" in cat or "hurricane" in cat:
            return "monsoon_cyclone"
        if "wildfire" in cat or "flood" in cat or "earthquake" in cat:
            return "flood_quake"
    if source == "macro" and isinstance(raw_payload, dict):
        sid = str(raw_payload.get("series_id", "")).upper()
        delta_pct = abs(float(raw_payload.get("delta_pct", 0) or 0))
        if sid == "DEXINUS":
            return "fx_move"
        if sid in ("DCOILBRENTEU", "DCOILWTICO") and delta_pct > 2.0:
            return "oil_shock"
        if sid == "DGS10" and delta_pct > 2.0:
            return "geopolitical_other"
        if sid in ("FEDFUNDS", "DFEDTAR", "DFEDTARU"):
            return "rate_decision"
    return None


def _pick_type(scores: dict[str, int]) -> str | None:
    """Pick the highest scoring event_type; needs ≥2 to count.
    Tie-breaker: prefer a more-specific type over `geopolitical_other` when
    scores are within 1 of each other.
    """
    if not scores:
        return None
    # Find top score
    best_type, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score < 2:
        return None
    # Tie-breaker — if geopolitical_other (which we don't score directly) ever
    # leaks in via future logic, prefer the runner-up if within 1.
    runners = [
        (t, s) for t, s in scores.items()
        if t != best_type and best_score - s <= 1 and t != "geopolitical_other"
    ]
    if best_type == "geopolitical_other" and runners:
        runner = max(runners, key=lambda kv: kv[1])
        return runner[0]
    return best_type


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def classify(raw_event: dict) -> dict | None:
    """Tag a single raw event with event_type + entities.

    Returns the input dict enriched with:
        event_type, entities, headline_normalized, classifier_version

    Returns None when the event cannot be confidently classified or lacks
    any usable signal surface (no entities at all).
    """
    if not isinstance(raw_event, dict):
        return None
    headline = raw_event.get("headline")
    if not headline or not isinstance(headline, str):
        return None  # §12.3 malformed payload

    source = str(raw_event.get("source", "")).strip()
    raw_payload = raw_event.get("raw_payload") or {}
    hint = raw_event.get("entities_hint") or {}

    # Extract entities first — needed for both India-relevance gating and scoring.
    entities = _extract_entities(headline, raw_payload, hint)
    normalized = _normalize_headline(headline)

    # 1) Source-based prior wins when applicable.
    event_type = _source_prior(source, raw_payload)

    # 2) Otherwise score by keywords across all event_types.
    if event_type is None:
        scores = _score_keyword_hits(normalized, entities)
        event_type = _pick_type(scores)
        if event_type is None:
            # Last resort: if hint flagged it as high-severity geo, allow catch-all.
            if isinstance(hint, dict) and hint.get("geo_severity", 0) >= 3:
                event_type = "geopolitical_other"
            else:
                return None  # §12.5 classifier confused → drop

    # Validate against taxonomy (defensive).
    if event_type not in TAXONOMY:
        return None

    # 3) India-relevance gate for types that only matter when Indian.
    india_relevant = hint.get("india_relevant") if isinstance(hint, dict) else None
    if india_relevant is False and event_type in INDIA_REQUIRED_TYPES:
        return None

    # 4) Empty-signal guard — if we have NO orgs, NO commodities, NO themes,
    # we have nothing for Agent 9 to map to instruments. Drop.
    if (
        not entities["organizations"]
        and not entities["commodities"]
        and not entities["themes"]
        and not entities["locations"]  # locations also count as signal surface
    ):
        return None

    out = dict(raw_event)  # immutable: don't mutate caller's dict
    out["event_type"] = event_type
    out["entities"] = entities
    out["headline_normalized"] = normalized
    out["classifier_version"] = CLASSIFIER_VERSION
    return out


def classify_batch(events: Iterable[dict]) -> list[dict]:
    """Convenience: classify a list, drop Nones."""
    out: list[dict] = []
    for ev in events:
        c = classify(ev)
        if c is not None:
            out.append(c)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def classify_cli() -> int:
    """argparse entry — reads JSON lines from stdin (or --in=path),
    writes classified JSON lines to stdout. Skipped events go to stderr.
    """
    parser = argparse.ArgumentParser(
        description="Classify raw events into Layer-4 taxonomy."
    )
    parser.add_argument("--in", dest="in_path", default=None,
                        help="Input file (JSON lines). Default: stdin.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress skipped-event reasons on stderr.")
    args = parser.parse_args()

    if args.in_path:
        try:
            stream: Any = open(args.in_path, "r", encoding="utf-8")
        except OSError as e:
            print(f"input open failed: {e}", file=sys.stderr)
            return 2
    else:
        stream = sys.stdin

    n_in = n_out = 0
    try:
        for ln, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                ev = json.loads(line)
            except json.JSONDecodeError as e:
                if not args.quiet:
                    print(f"line {ln}: bad JSON: {e}", file=sys.stderr)
                continue
            classified = classify(ev)
            if classified is None:
                if not args.quiet:
                    h = (ev.get("headline") if isinstance(ev, dict) else "")
                    print(f"line {ln}: SKIP ({(h or '')[:60]!r})", file=sys.stderr)
                continue
            sys.stdout.write(json.dumps(classified, ensure_ascii=False) + "\n")
            n_out += 1
    finally:
        if args.in_path:
            stream.close()

    if not args.quiet:
        print(f"in={n_in} out={n_out} skipped={n_in - n_out}", file=sys.stderr)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test — 5 canned events across sources
# ─────────────────────────────────────────────────────────────────────────────

_SMOKE_EVENTS = [
    {
        "event_id": "gdelt:1",
        "source": "gdelt_events",
        "headline": "Iran threatens to close Strait of Hormuz amid Brent crude spike",
        "url": "https://example.com/1",
        "timestamp": "2026-06-17T00:00:00Z",
        "raw_payload": {"themes": ["ENERGY_OIL"]},
        "entities_hint": {"actors": ["Iran"], "themes": ["energy"]},
    },
    {
        "event_id": "ofac:2",
        "source": "ofac",
        "headline": "OFAC adds 12 Russian energy entities to SDN list",
        "url": "https://example.com/2",
        "timestamp": "2026-06-17T00:00:00Z",
        "raw_payload": {"sdn_delta": 12},
        "entities_hint": {"organizations": ["Rosneft"], "themes": ["sanction"]},
    },
    {
        "event_id": "eonet:3",
        "source": "eonet",
        "headline": "Cyclone Biparjoy intensifies near Gujarat coast",
        "url": "https://example.com/3",
        "timestamp": "2026-06-17T00:00:00Z",
        "raw_payload": {"categories": ["Severe Storms"]},
        "entities_hint": {"locations": ["Gujarat"], "india_relevant": True},
    },
    {
        "event_id": "rss:4",
        "source": "rss_india",
        "headline": "RBI MPC hikes repo rate by 25 bps amid inflation concerns",
        "url": "https://example.com/4",
        "timestamp": "2026-06-17T00:00:00Z",
        "raw_payload": {"tags": ["monetary policy"]},
        "entities_hint": {"organizations": ["RBI"], "themes": ["rates"]},
    },
    {
        "event_id": "macro:5",
        "source": "macro",
        "headline": "USD/INR jumps 0.8% intraday as dollar strengthens",
        "url": "https://example.com/5",
        "timestamp": "2026-06-17T00:00:00Z",
        "raw_payload": {"series_id": "DEXINUS", "delta_pct": 0.8},
        "entities_hint": {"commodities": ["rupee"], "themes": ["fx"]},
    },
]


def _smoke() -> None:
    print(f"[smoke] taxonomy count = {len(TAXONOMY)} (expected 14)")
    print(f"[smoke] keyword dicts  = {len(KEYWORDS)}")
    for ev in _SMOKE_EVENTS:
        c = classify(ev)
        if c is None:
            print(f"[smoke] {ev['event_id']:10s} SKIP -- {ev['headline']}")
        else:
            print(
                f"[smoke] {ev['event_id']:10s} -> {c['event_type']:20s}"
                f" entities={{orgs:{len(c['entities']['organizations'])},"
                f"locs:{len(c['entities']['locations'])},"
                f"cmd:{len(c['entities']['commodities'])}}}"
                f" -- {c['headline_normalized'][:60]}"
            )


if __name__ == "__main__":
    # If stdin is a TTY (interactive), run smoke test instead of CLI.
    if sys.stdin.isatty() and len(sys.argv) == 1:
        _smoke()
    else:
        sys.exit(classify_cli())

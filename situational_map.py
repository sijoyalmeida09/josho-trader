"""
situational_map.py — JoSho Trader Layer 4 (Situational Awareness)

Pure data + tiny helper. NO network calls, NO side effects.
Maps geopolitical / macro event categories to Indian-instrument score
adjustments. Consumed by:

  - signal_worker.py  — normalizes external events into trader_signals rows
                        using these boost/block tables.
  - groww_bot.py      — its scan() functions call compute_score_adjustments()
                        to bias scores with currently-active signals.

The category enum here MUST stay in sync with the `trader_signals.category`
text column (see supabase/migrations/040_trader_tables.sql).
"""

from __future__ import annotations

from typing import Iterable

# ---------------------------------------------------------------------
# SITUATIONAL_MAP — category -> {description, boosts, blocks, severity, ttl}
#
# Semantics:
#   indian_symbols_boost  -> positive score_adjustment (longs favored)
#   indian_symbols_block  -> negative score_adjustment (avoid / short bias)
#   default_severity      -> 1..10 (used if upstream feed didn't supply one)
#   ttl_hours             -> how long the signal stays "active" by default
# ---------------------------------------------------------------------
SITUATIONAL_MAP: dict[str, dict] = {
    "oil_shock": {
        "description": "Oil supply shock from Middle East tension (Iran/Israel/Strait of Hormuz). Crude up -> upstream producers benefit, OMCs squeezed.",
        "indian_symbols_boost": {"ONGC": +30, "RELIANCE": +25, "OIL": +20},
        "indian_symbols_block": {"BPCL": -20, "IOC": -20, "HPCL": -20},
        "default_severity": 6,
        "ttl_hours": 12,
    },
    "defense_flashpoint": {
        "description": "Indo-Pak LoC, Indo-China LAC, or major terror event. Defense PSUs spike, broad index volatility expands.",
        "indian_symbols_boost": {"HAL": +35, "BEL": +30, "BDL": +30, "MAZDOCK": +25, "COCHINSHIP": +20},
        "indian_symbols_block": {"NIFTY": -10, "BANKNIFTY": -10},  # vol up -> long-vega bias
        "default_severity": 7,
        "ttl_hours": 24,
    },
    "cable_cut": {
        "description": "Submarine cable incident (Red Sea / Indian Ocean). Domestic telcos with diverse routes gain pricing power.",
        "indian_symbols_boost": {"BHARTIARTL": +20, "RELIANCE": +15, "IDEA": +10, "TATACOMM": +25},
        "indian_symbols_block": {},
        "default_severity": 5,
        "ttl_hours": 18,
    },
    "cyclone": {
        "description": "Cyclone / severe weather hitting Indian coast. Agri commodities (cotton, sugar) tighten; insurers slightly pressured.",
        "indian_symbols_boost": {"DCMSHRIRAM": +15, "BAJAJHIND": +15, "EIDPARRY": +15, "BALRAMCHIN": +15},
        "indian_symbols_block": {"ICICIGI": -10, "HDFCLIFE": -5, "SBILIFE": -5},
        "default_severity": 6,
        "ttl_hours": 36,
    },
    "sanction": {
        "description": "OFAC SDN delta — new sanctions on Russia/Iran shift fertilizer + LNG flows toward India. Fertilizer co-ops + natgas tape benefit.",
        "indian_symbols_boost": {"UPL": +20, "COROMANDEL": +20, "RCF": +25, "GNFC": +20, "GAIL": +15, "PETRONET": +15},
        "indian_symbols_block": {},
        "default_severity": 5,
        "ttl_hours": 48,
    },
    "ai_compute_buildout": {
        "description": "Aschenbrenner-thesis-aligned: US power grid stress / hyperscaler capex revision / NVDA-cycle event. INFORMATIONAL ONLY — Indian tape rarely tradeable directly; surfaced as context for IT/power names. No automatic score boost.",
        "indian_symbols_boost": {},   # intentionally empty — see description
        "indian_symbols_block": {},
        "default_severity": 4,
        "ttl_hours": 24,
    },
    "india_policy_event": {
        "description": "RBI rate move, Union Budget surprise, or major SEBI/MoF announcement. Whole-index volatility regime shift; bias toward NIFTY straddles, not directional bets.",
        "indian_symbols_boost": {},  # vol play, not directional
        "indian_symbols_block": {"NIFTY": -8, "BANKNIFTY": -8},  # damp directional confidence
        "default_severity": 6,
        "ttl_hours": 8,
    },
}


# ---------------------------------------------------------------------
# Helper — sum boost/block across all active signals for a given symbol.
# Used by groww_bot scan() functions to bias raw scores.
# ---------------------------------------------------------------------
def compute_score_adjustments(active_signals: Iterable[dict], symbol: str) -> int:
    """
    Sum the +/- score_adjustment that applies to `symbol` across all
    currently-active signals.

    Each `active_signal` is a dict as returned by `trader_db.fetch_active_signals()`
    — minimally containing keys: `category`, `affected_symbols`, `score_adjustment`.

    Resolution order per signal:
      1. If the signal carries an explicit `score_adjustment` AND `symbol` is in
         its `affected_symbols`, use that value (the worker pre-computed it).
      2. Otherwise fall back to the SITUATIONAL_MAP boost/block table.

    Returns the integer net adjustment (can be 0).
    """
    if not symbol:
        return 0
    sym = symbol.upper().strip()
    total = 0
    for sig in active_signals or []:
        category = (sig.get("category") or "").strip()
        affected = [s.upper() for s in (sig.get("affected_symbols") or [])]
        if sym not in affected:
            continue
        # Prefer the explicitly-stamped per-signal adjustment when present.
        explicit = sig.get("score_adjustment")
        if explicit is not None:
            try:
                # If the worker stored a single number, it applies uniformly to
                # all affected symbols — including blocks (negative values).
                total += int(explicit)
                continue
            except (TypeError, ValueError):
                pass
        # Fallback: derive from the static map.
        spec = SITUATIONAL_MAP.get(category)
        if not spec:
            continue
        if sym in spec.get("indian_symbols_boost", {}):
            total += int(spec["indian_symbols_boost"][sym])
        if sym in spec.get("indian_symbols_block", {}):
            total += int(spec["indian_symbols_block"][sym])  # already negative
    return total


def affected_symbols_for(category: str) -> list[str]:
    """Return the union of boost+block symbol keys for a category. Used by the
    worker when populating `trader_signals.affected_symbols`."""
    spec = SITUATIONAL_MAP.get(category)
    if not spec:
        return []
    return sorted(set(spec.get("indian_symbols_boost", {}).keys())
                  | set(spec.get("indian_symbols_block", {}).keys()))


__all__ = [
    "SITUATIONAL_MAP",
    "compute_score_adjustments",
    "affected_symbols_for",
    "EVENT_TYPE_MAP",
    "instruments_and_direction_for",
]


# ---------------------------------------------------------------------
# EVENT_TYPE_MAP — keyed by §2 event_type taxonomy (14 event types).
#
# Each entry shape:
#   bullish:     {symbol: weight 0..1}  -> direction +1
#   bearish:     {symbol: weight 0..1}  -> direction -1
#   base_magnitude: float 0..1          -> pre-amplifier starting magnitude
#   amplifiers:  {locations: {...}, themes: {...}, commodities: {...},
#                 organizations: {...}} -> entity-driven adds
#   max_magnitude: float 0..1           -> hard clamp
#   fx_aware / ticker_driven / instruments_override_on_surprise — opt-in special handling
#
# Consumed by `instruments_and_direction_for()` only. The original
# SITUATIONAL_MAP above (keyed by 7 broad categories) is kept untouched
# for back-compat with signal_worker.py / groww_bot.py.
# ---------------------------------------------------------------------
EVENT_TYPE_MAP: dict[str, dict] = {
    "oil_shock": {
        "bullish": {"ONGC": 0.7, "OIL": 0.6, "RELIANCE": 0.5},
        "bearish": {"BPCL": 0.6, "HPCL": 0.6, "IOC": 0.5},
        "base_magnitude": 0.5,
        "amplifiers": {
            "locations": {"iran": 0.25, "hormuz": 0.30, "israel": 0.15, "saudi": 0.20},
            "themes": {"production cut": 0.20, "embargo": 0.25, "opec": 0.10},
            "commodities": {"crude": 0.10, "brent": 0.10},
        },
        "max_magnitude": 1.0,
    },
    "defense_flashpoint": {
        "bullish": {"HAL": 0.6, "BEL": 0.5, "BDL": 0.5, "MAZDOCK": 0.4, "PARAS": 0.3},
        "bearish": {},
        "base_magnitude": 0.4,
        "amplifiers": {
            "locations": {"pakistan": 0.20, "china": 0.20, "loc": 0.25, "lac": 0.25},
            "themes": {"airstrike": 0.20, "terror attack": 0.20},
        },
        "max_magnitude": 0.9,
    },
    "sanction": {
        "bullish": {"UPL": 0.4, "COROMANDEL": 0.3, "RCF": 0.3, "NATIONALUM": 0.2},
        "bearish": {},
        "base_magnitude": 0.3,
        "amplifiers": {
            "themes": {"sanction_russia": 0.25, "sanction_iran": 0.20, "ofac": 0.15},
            "locations": {"russia": 0.15, "iran": 0.15},
        },
        "max_magnitude": 0.7,
    },
    "monsoon_cyclone": {
        "bullish": {"UPL": 0.3, "COROMANDEL": 0.3, "DHANUKA": 0.2},
        "bearish": {"ICICIPRULI": 0.3, "HDFCLIFE": 0.2, "SBILIFE": 0.2},
        "base_magnitude": 0.3,
        "amplifiers": {
            "locations": {
                "maharashtra": 0.2,
                "gujarat": 0.2,
                "odisha": 0.2,
                "andhra pradesh": 0.2,
            },
            "themes": {"landfall": 0.15, "imd warning": 0.10},
        },
        "max_magnitude": 0.7,
    },
    "flood_quake": {
        "bullish": {},
        "bearish": {"ICICIPRULI": 0.2, "HDFCLIFE": 0.2, "ULTRACEMCO": 0.2},
        "base_magnitude": 0.2,
        "amplifiers": {
            "themes": {"earthquake": 0.15, "flood": 0.15},
        },
        "max_magnitude": 0.5,
    },
    "rate_decision": {
        "bullish": {},
        "bearish": {},
        "base_magnitude": 0.4,
        "amplifiers": {
            "themes": {"rate hike surprise": 0.3, "rate cut surprise": 0.3, "rbi": 0.10},
        },
        "instruments_override_on_surprise": {
            "rate cut surprise": {
                "bullish": {"BAJFINANCE": 0.4, "HDFC": 0.4, "DLF": 0.4},
                "bearish": {},
            },
            "rate hike surprise": {
                "bullish": {"SBIN": 0.3, "HDFCBANK": 0.2},
                "bearish": {"DLF": 0.4, "BAJFINANCE": 0.3},
            },
        },
        "max_magnitude": 0.8,
    },
    "metals_supply": {
        "bullish": {
            "TATASTEEL": 0.5,
            "HINDALCO": 0.5,
            "JSWSTEEL": 0.4,
            "VEDL": 0.4,
            "NATIONALUM": 0.3,
        },
        "bearish": {},
        "base_magnitude": 0.4,
        "amplifiers": {
            "themes": {"china export curb": 0.25, "mine accident": 0.2},
            "commodities": {"aluminium": 0.10, "copper": 0.10, "steel": 0.10},
        },
        "max_magnitude": 0.8,
    },
    "fx_move": {
        "bullish": {},
        "bearish": {},
        "base_magnitude": 0.3,
        "amplifiers": {
            "themes": {"rupee weak": 0.20, "rupee strong": 0.20, "inr": 0.10},
        },
        "fx_aware": True,
        "max_magnitude": 0.6,
    },
    "telecom_disruption": {
        "bullish": {"BHARTIARTL": 0.4, "RELIANCE": 0.2},
        "bearish": {},
        "base_magnitude": 0.3,
        "amplifiers": {
            "themes": {"submarine cable": 0.25, "outage": 0.10},
            "locations": {"red sea": 0.15},
        },
        "max_magnitude": 0.6,
    },
    "agri_supply": {
        "bullish": {"UPL": 0.4, "COROMANDEL": 0.3, "RALLIS": 0.3, "DHANUKA": 0.3},
        "bearish": {"HUL": 0.2, "BRITANNIA": 0.2, "NESTLEIND": 0.2},
        "base_magnitude": 0.3,
        "amplifiers": {
            "themes": {"export ban": 0.3, "msp hike": 0.2, "drought": 0.15},
        },
        "max_magnitude": 0.7,
    },
    "tech_supply": {
        "bullish": {"DIXON": 0.3, "TATAELXSI": 0.3, "AMBER": 0.2},
        "bearish": {"TCS": 0.2, "INFY": 0.2, "WIPRO": 0.2, "HCLTECH": 0.2},
        "base_magnitude": 0.3,
        "amplifiers": {
            "themes": {"chip shortage": 0.25, "rare earth": 0.2, "taiwan": 0.15},
        },
        "max_magnitude": 0.7,
    },
    "corporate_action": {
        "bullish": {},
        "bearish": {},
        "base_magnitude": 0.5,
        "amplifiers": {
            "themes": {"earnings beat": 0.20, "buyback": 0.15, "merger": 0.15},
        },
        "ticker_driven": True,
        "max_magnitude": 0.9,
    },
    "policy_shift": {
        "bullish": {},
        "bearish": {},
        "base_magnitude": 0.4,
        "amplifiers": {
            "themes": {"fdi relax": 0.3, "tariff hike": 0.2, "pli scheme": 0.20},
        },
        "max_magnitude": 0.7,
    },
    "geopolitical_other": {
        "bullish": {},
        "bearish": {},
        "base_magnitude": 0.2,
        "amplifiers": {},
        "max_magnitude": 0.5,
    },
}


def _iter_entity_values(entities: dict, key: str) -> list[str]:
    """Pull a list of entity values (locations/themes/etc) as lower-cased strings."""
    raw = entities.get(key) if entities else None
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw.lower().strip()]
    out: list[str] = []
    for item in raw:
        if item is None:
            continue
        try:
            out.append(str(item).lower().strip())
        except Exception:
            continue
    return [v for v in out if v]


def _apply_amplifiers(
    magnitude: float,
    amplifiers: dict,
    entities: dict,
) -> float:
    """Add amplifier weights for any entity that matches the amplifier table."""
    if not amplifiers or not entities:
        return magnitude
    for entity_kind, weight_table in amplifiers.items():
        if not isinstance(weight_table, dict):
            continue
        values = _iter_entity_values(entities, entity_kind)
        if not values:
            continue
        lowered_table = {k.lower(): v for k, v in weight_table.items()}
        for val in values:
            # exact key match
            if val in lowered_table:
                magnitude += float(lowered_table[val])
                continue
            # substring fallback — e.g. "strait of hormuz" should match "hormuz"
            for key, w in lowered_table.items():
                if key and key in val:
                    magnitude += float(w)
                    break
    return magnitude


def _fx_move_resolve(entities: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Resolve fx_move instrument lists from entity hints. INR-weak -> IT bullish,
    importers bearish. INR-strong -> reverse."""
    themes = set(_iter_entity_values(entities, "themes"))
    text_blob = " ".join(_iter_entity_values(entities, "themes")
                         + _iter_entity_values(entities, "headline"))
    weak_signals = {"rupee weak", "inr fall", "usd up", "rupee depreciation"}
    strong_signals = {"rupee strong", "inr strong", "usd down", "rupee appreciation"}
    is_weak = any(s in themes for s in weak_signals) or any(s in text_blob for s in weak_signals)
    is_strong = any(s in themes for s in strong_signals) or any(s in text_blob for s in strong_signals)
    if is_weak and not is_strong:
        bullish = {"TCS": 0.4, "INFY": 0.4, "WIPRO": 0.3, "HCLTECH": 0.3}
        bearish = {"MARUTI": 0.3, "IOC": 0.3, "BPCL": 0.2}
        return bullish, bearish
    if is_strong and not is_weak:
        bullish = {"MARUTI": 0.3, "IOC": 0.2}
        bearish = {"TCS": 0.3, "INFY": 0.3, "WIPRO": 0.2, "HCLTECH": 0.2}
        return bullish, bearish
    # Ambiguous — return neutral empty set.
    return {}, {}


def _corporate_action_resolve(entities: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Use entities.organizations as the instrument list; theme sets the direction."""
    orgs = _iter_entity_values(entities, "organizations")
    if not orgs:
        return {}, {}
    themes = set(_iter_entity_values(entities, "themes"))
    bullish_themes = {"earnings beat", "buyback", "dividend hike", "guidance raise"}
    bearish_themes = {"fraud", "show cause", "earnings miss", "downgrade", "sebi probe"}
    direction = 0
    if themes & bullish_themes:
        direction = +1
    elif themes & bearish_themes:
        direction = -1
    # Upper-case the org tokens (treat as tickers / company labels).
    tickers = {o.upper(): 0.5 for o in orgs}
    if direction > 0:
        return tickers, {}
    if direction < 0:
        return {}, tickers
    # Neutral corporate news — still surface as instruments with direction 0,
    # but bullish/bearish dicts stay empty so direction_map gets 0 below.
    return tickers, {}  # treat as bullish-leaning default; direction_map will be +1


def instruments_and_direction_for(
    event_type: str,
    entities: dict | None = None,
) -> tuple[list[str], dict[str, int], float]:
    """
    Returns (instruments, direction_map, base_magnitude).

    instruments      — NSE trading symbols affected (e.g. ['ONGC','RELIANCE','BPCL'])
    direction_map    — {symbol: +1|-1|0}; +1 bullish, -1 bearish, 0 neutral
    base_magnitude   — 0.0-1.0 (pre-scoring expected percent-of-move proxy)

    Uses event_type + (optional) entities to refine the mapping. For instance,
    event_type='oil_shock' with entities containing 'Iran' or 'Hormuz' produces
    a higher base_magnitude than a generic OPEC chatter.

    Returns ([], {}, 0.0) for unmapped event_types.
    """
    if not event_type:
        return [], {}, 0.0
    spec = EVENT_TYPE_MAP.get(event_type)
    if not spec:
        return [], {}, 0.0

    ents: dict = entities if isinstance(entities, dict) else {}

    bullish: dict[str, float] = dict(spec.get("bullish", {}))
    bearish: dict[str, float] = dict(spec.get("bearish", {}))

    # ---- Special case: rate_decision surprise overrides -----------------
    if "instruments_override_on_surprise" in spec:
        themes = set(_iter_entity_values(ents, "themes"))
        for surprise_key, override in spec["instruments_override_on_surprise"].items():
            if surprise_key.lower() in themes:
                bullish = dict(override.get("bullish", {}))
                bearish = dict(override.get("bearish", {}))
                break

    # ---- Special case: fx_move direction inference ----------------------
    if spec.get("fx_aware"):
        fx_b, fx_be = _fx_move_resolve(ents)
        bullish = fx_b
        bearish = fx_be

    # ---- Special case: corporate_action — entity-driven tickers ---------
    if spec.get("ticker_driven"):
        ca_b, ca_be = _corporate_action_resolve(ents)
        bullish = ca_b
        bearish = ca_be

    # ---- Magnitude: base + amplifiers, clamped --------------------------
    magnitude = float(spec.get("base_magnitude", 0.0))
    magnitude = _apply_amplifiers(magnitude, spec.get("amplifiers", {}), ents)
    max_mag = float(spec.get("max_magnitude", 1.0))
    if magnitude > max_mag:
        magnitude = max_mag
    if magnitude < 0.0:
        magnitude = 0.0

    # ---- Assemble outputs ----------------------------------------------
    instruments: list[str] = []
    direction_map: dict[str, int] = {}
    for sym in bullish.keys():
        s = sym.upper()
        instruments.append(s)
        direction_map[s] = +1
    for sym in bearish.keys():
        s = sym.upper()
        if s not in direction_map:
            instruments.append(s)
        # Bearish wins if a symbol was in both (rare; explicit conflict).
        direction_map[s] = -1

    return instruments, direction_map, round(magnitude, 4)


# ---------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        ("oil_shock", {"locations": ["Iran"]}),
        ("oil_shock", {"locations": ["Strait of Hormuz"], "themes": ["embargo"]}),
        ("defense_flashpoint", {"locations": ["LoC"]}),
        ("sanction", {"themes": ["sanction_russia"]}),
        ("monsoon_cyclone", {"locations": ["Odisha"]}),
        ("flood_quake", {"themes": ["earthquake"]}),
        ("rate_decision", {"themes": ["rate cut surprise"]}),
        ("rate_decision", {"themes": ["rate hike surprise"]}),
        ("metals_supply", {"themes": ["china export curb"]}),
        ("fx_move", {"themes": ["rupee weak"]}),
        ("fx_move", {"themes": ["rupee strong"]}),
        ("telecom_disruption", {"themes": ["submarine cable"]}),
        ("agri_supply", {"themes": ["export ban"]}),
        ("tech_supply", {"themes": ["chip shortage"]}),
        ("corporate_action", {"organizations": ["RELIANCE"], "themes": ["earnings beat"]}),
        ("corporate_action", {"organizations": ["ADANIENT"], "themes": ["fraud"]}),
        ("policy_shift", {"themes": ["fdi relax"]}),
        ("geopolitical_other", {}),
        ("unmapped_event", {}),
    ]
    print(f"EVENT_TYPE_MAP size: {len(EVENT_TYPE_MAP)}")
    for et, ents in cases:
        instr, direction, mag = instruments_and_direction_for(et, ents)
        print(f"  {et:22s} ents={ents} -> instruments={instr} direction={direction} mag={mag}")

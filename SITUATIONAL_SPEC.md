# SITUATIONAL_SPEC.md — Layer 4 Single Source of Truth

> **Mission.** Convert world events into scored, directional, time-decaying
> signals for NSE instruments. **Advisory only.** Writes to `trader_signals`,
> can boost or suppress `trader_pending_orders.ai_confidence`, but
> **never auto-fires a trade**.
>
> **Defaults.** Every new signal is `status='shadow'`. An `event_type` becomes
> `active`-eligible **only** after the backtest harness shows a documented
> edge passing the gate (§7).
>
> **Authority.** This doc is the contract every Layer-4 agent reads.
> Do not edit `_CONTRACT.md`, `supabase/migrations/040_trader_tables.sql`,
> `situational_map.py`, or `signal_worker.py` — only extend.

---

## 1. Hard Rules (every agent obeys)

1. **Additive only.** No edits to existing strategy / auth / execution paths
   (`groww_bot.py` execution paths, `bot_server.py`, `_cockpit-client.tsx`).
2. **Free sources only.** No paid APIs, no auth tokens beyond keys already
   in `vault.master.env`. If a source requires payment → drop it.
3. **Provenance + expiry on every signal.** No row without `provenance_urls`
   (≥1) and `expires_at`.
4. **Shadow by default.** Inserts go to `status='shadow'`. Only the calibration
   harness (agent 11) may promote an `event_type` to `active`-eligible.
5. **Never auto-fires.** No code path in this layer calls `groww.place_order`.
   The only consumer outside Layer 4 is `ai_scorer.score()` reading active
   signals as advisory inputs.
6. **Failsoft.** Every ingestor logs and continues on source outage.
   A dead source must not stall the orchestrator.
7. **Idempotent.** Same source event → same `event_id` → at most one row.
8. **Dedup before write.** Collapse the same real-world event arriving via
   multiple sources into one signal with merged `provenance_urls`.

---

## 2. Event Taxonomy (`event_type` enum)

These are the only allowed values for `event_type`. Add new types only via
amending this section + corresponding mapping in `situational_map.py`.

| event_type | Description | Typical sources |
|---|---|---|
| `oil_shock` | Mid-East tension, OPEC cut, Iran/Saudi escalation, Hormuz disruption | GDELT, RSS, OFAC |
| `defense_flashpoint` | Indo-Pak / Indo-China border, terror attack, NCG escalation | GDELT, RSS, filings |
| `sanction` | OFAC SDN delta — Russia/Iran/China energy/metals/fertilizer | OFAC delta |
| `monsoon_cyclone` | IMD warning / EONET cyclone near Indian coast or breadbasket | EONET, RSS |
| `flood_quake` | USGS quake in supply region, major flood | USGS, EONET, RSS |
| `rate_decision` | RBI / FOMC / ECB rate change or surprise | Filings, RSS |
| `metals_supply` | Mine accident, China export curb, LME squeeze | GDELT, RSS, OFAC |
| `fx_move` | USD/INR sharp move (>0.5% intraday or 1% week) | FRED, RSS |
| `telecom_disruption` | Submarine cable cut, major outage | GDELT, RSS |
| `agri_supply` | Wheat/sugar export ban, pest outbreak, MSP change | RSS, filings |
| `tech_supply` | Semi shortage, China rare-earth export curb, hyperscaler capex revision | GDELT, RSS, macro |
| `corporate_action` | Earnings surprise, M&A, board change, regulatory action on Nifty50 | Filings, RSS |
| `policy_shift` | Budget surprise, GST change, FDI rule | Filings, RSS |
| `geopolitical_other` | Catch-all for high-severity geo events not above | GDELT |

---

## 3. Signal Schema (the row written to `trader_signals`)

After migration 041 (§9), every signal carries:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | uuid | ✅ | Surrogate primary key (auto) |
| `event_id` | text | ✅ | Stable dedup key per real-world event (e.g. `gdelt:7892341`, `eonet:EONET_12345`, `rss:moneycontrol:hash`) |
| `source` | text | ✅ | One of: `gdelt_events`, `gdelt_gkg`, `rss_india`, `filings`, `eonet`, `usgs`, `ofac`, `macro`, `manual` |
| `event_type` | text | ✅ | One of the enum in §2. Maps from `trader_signals.category` (legacy column kept). |
| `entities` | jsonb | ⚠️ | `{actors:[], locations:[], commodities:[], themes:[]}` extracted from source |
| `instruments` | text[] | ✅ | NSE trading symbols affected (e.g. `["ONGC","RELIANCE","BPCL"]`). Maps to `affected_symbols` (legacy alias). |
| `direction` | jsonb | ✅ | `{"ONGC":+1,"BPCL":-1}` — per-instrument directional bias, +1 bullish, −1 bearish, 0 neutral |
| `magnitude` | numeric(4,3) | ✅ | 0.000-1.000 expected percent-of-move proxy (NOT a price target) |
| `confidence` | smallint | ✅ | 0-100, output of the scoring model (§5) |
| `half_life_min` | integer | ✅ | Time after which the signal's `magnitude` halves; controls decay |
| `corroboration_count` | smallint | ✅ | How many distinct sources confirm. Default 1; bumped by dedup agent |
| `provenance_urls` | text[] | ✅ | All source URLs that produced this signal (post-dedup) |
| `status` | text | ✅ | `candidate` (raw ingest) \| `shadow` (default after classify) \| `active` (post-gate) \| `retired` (expired) |
| `created_at` | timestamptz | ✅ | First seen time |
| `expires_at` | timestamptz | ✅ | When this signal stops influencing scoring |
| `score_adjustment` | smallint | (legacy) | Compatibility with 040's `situational_map.py`; will mirror `magnitude * 100 * sign(direction)` |
| `severity` | smallint | (legacy) | 040 column; mirrors `confidence / 10` clamped 1-10 |
| `headline` | text | ✅ | One-line human summary |
| `url` | text | (legacy) | First entry of `provenance_urls` for back-compat |
| `payload` | jsonb | optional | Full normalized event payload for debugging |

**Legacy compatibility.** Columns from 040 (`category`, `affected_symbols`,
`score_adjustment`, `severity`, `url`) remain written for back-compat with
`situational_map.py` and the cockpit UI. The migration 041 ADDS the new
columns. Writers populate both.

---

## 4. Source Registry

All free. No new auth required. Add `reliability_weight` for the scoring
model.

| Source | URL pattern | Free | Auth | Reliability | Owner Agent |
|---|---|---|---|---|---|
| `gdelt_events` | `https://api.gdeltproject.org/api/v2/doc/doc?...` | ✅ | none | 0.55 | Agent 1 |
| `gdelt_gkg` | `https://api.gdeltproject.org/api/v2/gkg/gkg?...` | ✅ | none | 0.50 | Agent 2 |
| `rss_india` | RSS endpoints below | ✅ | none | 0.70 | Agent 3 |
| `filings` | RBI/SEBI/NSE/BSE RSS + page scrape | ✅ | none | 0.95 | Agent 4 |
| `eonet` | `https://eonet.gsfc.nasa.gov/api/v3/events?...` | ✅ | none | 0.90 | Agent 5 |
| `usgs` | `https://earthquake.usgs.gov/fdsnws/event/1/query?...` | ✅ | none | 0.95 | Agent 5 |
| `ofac` | `https://www.treasury.gov/ofac/downloads/sdn.csv` | ✅ | none | 0.95 | Agent 6 |
| `macro` (FRED, Yahoo unofficial) | various free endpoints | ✅ | none | 0.80 | Agent 7 |

**RSS sources for Agent 3** (`rss_india.py`):
```
https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms
https://www.moneycontrol.com/rss/MCtopnews.xml
https://www.business-standard.com/rss/markets-106.rss
https://www.livemint.com/rss/markets
https://feeds.reuters.com/reuters/INbusinessNews   (legacy; fall back to RSSHub if 404)
https://www.thehindubusinessline.com/feeder/default.rss
```

**Filings sources for Agent 4** (`filings.py`):
```
https://www.rbi.org.in/Scripts/RSSFeed.aspx?Type=Notifications
https://www.rbi.org.in/Scripts/RSSFeed.aspx?Type=Press_Releases
https://www.sebi.gov.in/sebirss.xml
https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml
https://www.bseindia.com/data/xml/notices.xml
```

---

## 5. Scoring Formula (Agent 10 implements; freeze the shape)

```
confidence = clamp(0, 100, base * reliability_weight * decay * corroboration * hit_rate)

where:
  base                 = magnitude * 100                      (0..100)
  reliability_weight   = SOURCE_RELIABILITY[source]            (0.50..0.95)
  decay                = 0.5 ** ((now - created_at) / half_life_min)   (1.0 → 0.0)
  corroboration        = 1 + 0.15 * min(corroboration_count - 1, 4)    (1.0..1.6)
  hit_rate             = HISTORICAL_HIT_RATE[event_type]               (0.0..1.5)
                          - default 1.0 until backtest fills it
                          - clipped to [0.5, 1.5]
```

`half_life_min` defaults per `event_type`:
| event_type | half_life_min |
|---|---|
| oil_shock | 240 (4h) |
| defense_flashpoint | 360 |
| sanction | 1440 (24h) |
| monsoon_cyclone | 720 |
| flood_quake | 180 |
| rate_decision | 2880 (48h — markets re-price slowly) |
| metals_supply | 480 |
| fx_move | 90 |
| telecom_disruption | 120 |
| agri_supply | 1440 |
| tech_supply | 720 |
| corporate_action | 180 |
| policy_shift | 1440 |
| geopolitical_other | 360 |

`expires_at = created_at + (half_life_min × 3)` — three half-lives = ~12% of
original magnitude remaining.

---

## 6. Status Lifecycle

```
ingestor → status='candidate' (raw, classifier hasn't tagged yet)
classifier → status='shadow'  (typed, ready to score, NOT influencing live)
backtest gate → eligible for status='active' for that event_type IF passes §7
signal_writer (when gate passed) → status='active' (advisory-only influence)
expiry (now > expires_at) → status='retired'
```

**Promotion to `active` is GLOBAL per `event_type`, not per signal.** Once
calibration shows `oil_shock` has documented edge, all future `oil_shock`
signals enter `active` automatically. Backtest re-runs nightly.

---

## 7. Backtest Gate (Agent 11 — THE GATE)

An `event_type` becomes `active`-eligible **iff** the calibration harness
shows, on a window of ≥6 months of historical data:

1. **N ≥ 25** historical events of this type, AND
2. **Hit-rate ≥ 0.55** on directional move (was the move in the predicted direction
   within `half_life_min × 2`?), AND
3. **Mean |move|** in the predicted direction ≥ **0.4 σ** of the
   instrument's 20-day return distribution, AND
4. **Edge over random** ≥ **+5pp** (binomial test, p<0.10), AND
5. **No look-ahead bias** (timestamps strictly precede price observations).

If any of (1)-(5) fails → `event_type` stays at SHADOW. The harness writes
the calibration result to `backtest/calibration.json`. The `signal_writer`
reads this file at startup; missing/failing event_types stay shadow.

Re-evaluation cadence: **nightly cron** (Layer 4 schedule).

---

## 8. File Layout & Agent Ownership

```
automation/josho-trader/
├── SITUATIONAL_SPEC.md                      (this doc; mine)
├── situational_map.py                       (existing; Agent 9 EXTENDS via additive function)
├── signal_worker.py                         (existing; Agent 15 orchestrates, no replacement)
├── classifier.py                            (Agent 8 — NEW)
├── confidence_model.py                      (Agent 10 — NEW)
├── dedup.py                                 (Agent 12 — NEW)
├── signal_writer.py                         (Agent 13 — NEW)
├── attribution.py                           (Agent 14 — NEW)
├── ingestors/
│   ├── __init__.py
│   ├── gdelt_events.py                      (Agent 1 — NEW)
│   ├── gdelt_gkg.py                         (Agent 2 — NEW)
│   ├── rss_india.py                         (Agent 3 — NEW)
│   ├── filings.py                           (Agent 4 — NEW)
│   ├── natural_events.py                    (Agent 5 — NEW; EONET + USGS)
│   ├── ofac_delta.py                        (Agent 6 — NEW)
│   └── macro.py                             (Agent 7 — NEW)
├── backtest/
│   ├── __init__.py
│   ├── harness.py                           (Agent 11 — NEW)
│   ├── calibration.json                     (output)
│   └── README.md                            (Agent 11 — NEW)
└── tests/                                   (Agent 15)
    ├── test_classifier.py
    ├── test_confidence.py
    ├── test_dedup.py
    └── test_signal_writer.py

supabase/migrations/
├── 040_trader_tables.sql                    (existing; DO NOT EDIT)
└── 041_trader_signals_extended.sql          (NEW; additive columns for §3 fields)
```

---

## 9. Migration 041 — Additive Columns

Required to satisfy the §3 schema. New columns added to `trader_signals`;
existing columns from 040 stay (legacy compatibility).

```sql
-- 041_trader_signals_extended.sql (FILE EXISTS; agents reference, do not edit)
alter table public.trader_signals
  add column if not exists event_id text,
  add column if not exists event_type text,
  add column if not exists entities jsonb default '{}'::jsonb,
  add column if not exists instruments text[] default array[]::text[],
  add column if not exists direction jsonb default '{}'::jsonb,
  add column if not exists magnitude numeric(4,3) default 0.0,
  add column if not exists confidence smallint default 0,
  add column if not exists half_life_min integer default 240,
  add column if not exists corroboration_count smallint default 1,
  add column if not exists provenance_urls text[] default array[]::text[],
  add column if not exists status text default 'shadow'
    check (status in ('candidate','shadow','active','retired'));

create unique index if not exists ux_trader_signals_event_id
  on public.trader_signals(event_id) where event_id is not null;
create index if not exists idx_trader_signals_status_eventtype
  on public.trader_signals(status, event_type) where status in ('shadow','active');
create index if not exists idx_trader_signals_instruments_gin
  on public.trader_signals using gin(instruments);
```

---

## 10. Public Python Interfaces (frozen)

Each agent's module must expose exactly these functions/classes. Internal
helpers may vary.

### Ingestors (Agents 1-7)

Every ingestor exposes:
```python
def poll(since: datetime | None = None) -> list[dict]:
    """Return list of raw event dicts. Each dict includes:
       {event_id, source, headline, url, timestamp, raw_payload}
    """

def poll_once_cli() -> int:
    """argparse entry — --since=ISO, --dry-run; writes to stdout JSON lines"""
```

### Classifier (Agent 8)
```python
def classify(raw_event: dict) -> dict | None:
    """Input: raw event from any ingestor.
       Output: classified event dict with added fields:
       {event_type, entities, headline_normalized}
       Returns None if event doesn't map to taxonomy.
    """
```

### Mapping engine (Agent 9 — EXTEND `situational_map.py`)

Add to existing module:
```python
def instruments_and_direction_for(event_type: str, entities: dict) -> tuple[list[str], dict[str, int], float]:
    """Returns (instruments, direction_map, base_magnitude).
       base_magnitude is 0.0-1.0 *before* scoring.
    """
```

### Confidence model (Agent 10)
```python
class ConfidenceModel:
    def __init__(self, calibration_path: Path | None = None): ...
    def score(self, classified_event: dict, corroboration_count: int = 1) -> dict:
        """Returns {confidence:int, magnitude:float, half_life_min:int, expires_at:datetime}."""
```

### Backtest (Agent 11)
```python
def run_backtest(event_type: str, window_days: int = 180) -> dict:
    """Returns {hit_rate, mean_abs_move, n_events, edge_pp, pass_gate: bool, sample: [...]}.
       Writes/updates backtest/calibration.json."""
```

### Dedup (Agent 12)
```python
def dedup(events: list[dict]) -> list[dict]:
    """Collapse same real-world event from multiple sources.
       Merges provenance_urls, sums corroboration_count, picks max confidence."""
```

### Signal writer (Agent 13)
```python
class SignalWriter:
    def __init__(self, trader_db=None): ...
    def write(self, signal: dict, force_shadow: bool = True) -> str | None:
        """Returns inserted signal id, or None on failure.
           Always inserts as 'shadow' unless calibration grants 'active' for event_type."""
```

### Attribution (Agent 14)
```python
def attribute_outcomes(window_hours: int = 24) -> list[dict]:
    """For each active/shadow signal in window: look up actual price move,
       compute realized vs predicted, write to trader_signal_outcomes table."""
```

### Orchestrator (Agent 15) — extends `signal_worker.py`
```python
# Inside extended signal_worker.py:
def orchestrate_cycle(dry_run: bool = False) -> dict:
    """Run all ingestors → classifier → mapping → confidence → dedup → writer.
       Returns {ingested: N, classified: N, written: N, errors: [...]}.
    """
```

---

## 11. Cockpit Hook (advisory boost into `ai_scorer.score`)

Already in `ai_scorer.py` rule-fallback: `-20 if active adverse situational`.
After Layer 4 lands, the wiring is:

```
ai_scorer.score(signal_for_trade) reads active signals via TraderDB
  → looks up signal_for_trade.symbol in trader_signals.instruments
  → adjusts confidence by:
      + sum(direction[symbol] * magnitude * confidence/100 * 0.3) for active signals only
```

The `0.3` is the **advisory weight cap** — geopolitical signals can move the
AI confidence by at most ±30 points, never more. This is the only place
Layer 4 touches live scoring.

Status=`shadow` signals are written, attributed, and shown in the cockpit
chip area, but NOT used by `ai_scorer`.

---

## 12. Failure Modes (every agent handles these)

1. **Source 5xx** — exponential backoff, give up after 3, log, return empty list.
2. **Source rate-limit** — record `rate_limited_until`, skip until elapsed.
3. **Malformed payload** — skip the single event, log, continue.
4. **DB unreachable** — buffer to `data/signals_outbox/*.jsonl`, drain next cycle.
5. **Classifier confused** — return None, drop the candidate.
6. **Backtest insufficient data** — keep event_type in SHADOW, log reason.

---

## 13. Smoke-Test Spec (deliverable check)

After all agents land, smoke test runs:

```
python -m automation.josho-trader.signal_worker --once --dry-run --verbose
```

Must produce:
- (a) At least one live batch from each of: gdelt_events, rss_india, eonet, ofac
- (b) ≥5 signals printed with all §3 fields populated
- (c) `python -m automation.josho-trader.backtest.harness --event-type=oil_shock`
  prints hit-rate + sample
- (d) When `--dry-run=False`, signals written to `trader_signals` with `status='shadow'`

---

## 14. Out of Scope (DO NOT BUILD)

- No paid sources. No Bloomberg, no Refinitiv.
- No social-media scraping (Twitter/X requires auth; Reddit OK only via free RSSHub if zero-config).
- No trade execution. No order placement. No position sizing.
- No new auth flows.
- No live retraining of an ML model — calibration is rule-based + backtest.
- No edits to `040_trader_tables.sql`, `_CONTRACT.md`, `groww_bot.py`,
  `bot_server.py`, `_cockpit-client.tsx`, or any auth/execution code.

---

End of spec. Agents read sections 1-13 verbatim before writing any code.

# JoSho Trader — System Contract (single source of truth)

> Every parallel agent reads this. Schema lives in
> `supabase/migrations/040_trader_tables.sql` — treat that as authoritative.

## Layout

```
automation/josho-trader/
├── groww_bot.py        (existing — extended by Integration agent only)
├── bot_server.py       (NEW — Flask HTTP layer; agent: Bot Server)
├── trader_db.py        (NEW — Supabase writes/reads; agent: Supabase Writer)
├── ai_scorer.py        (NEW — Groq confidence; agent: AI Scorer)
├── ecosystem.config.js (NEW — PM2 supervision; agent: Process Manager)
├── HOSTING.md          (NEW; agent: Hosting Doc)
├── RUNBOOK.md          (NEW; agent: Runbook)
└── requirements.txt    (extend: flask, supabase, groq are added)

apps/admin/src/app/
├── trader/page.tsx              (existing — extended by Integration agent only)
├── trader/_cockpit-client.tsx   (NEW — iPad cockpit; agent: UI Cockpit)
└── api/trader/
    ├── status/route.ts          (existing — preserved)
    ├── approve/route.ts         (NEW; agent: API Routes)
    ├── reject/route.ts          (NEW; agent: API Routes)
    ├── kill/route.ts            (NEW; agent: API Routes)
    ├── settings/route.ts        (NEW; agent: API Routes)
    └── command/route.ts         (NEW; agent: API Routes — generic command POST)

apps/admin/public/
├── manifest.webmanifest         (NEW; agent: PWA)
└── sw.js                        (NEW; agent: PWA)
```

## Environment

All free, hosted local for now:

| Service | URL | Free | Used by |
|---|---|---|---|
| Groww API | api.groww.in | paid (user already paid) | bot |
| Supabase | already in vault (SUPABASE_*) | ✅ free tier | bot + UI |
| Groq | already in vault (`SHARED__GROQ_API_KEY`) | ✅ free | ai_scorer |
| Telegram | already in vault | ✅ free | bot notifier |
| Vercel | admin.joshoit.com | ✅ hobby | UI |
| Bot process | local Windows PC, PM2 | ✅ free | bot + Flask |

## Python env-var mapping (already wired in groww_bot.py)

The bot's `_load_vault_env()` populates these from `vault.master.env`:

| Vault key | Env var | Purpose |
|---|---|---|
| `SOHAYA__GROWW_TOTP_TOKEN` | `GROWW_API_KEY` / `GROWW_TOTP_TOKEN` | Groww API key (long JWT) |
| `SOHAYA__GROWW_TOTP_QRCODE` | `GROWW_TOTP_SECRET` | Base32 TOTP secret |
| `SOHAYA__GROWW_ACCESS_TOKEN` | `GROWW_ACCESS_TOKEN` | Daily-expiring direct token |
| `SHARED__TELEGRAM_BOT_TOKEN` | `TELEGRAM_BOT_TOKEN` | Notifier |
| `SHARED__TELEGRAM_SIJOY_CHAT_ID` | `TELEGRAM_CHAT_ID` | Notifier chat |
| `SHARED__SUPABASE_URL` (use `SOHAYA__NEXT_PUBLIC_SUPABASE_URL` if SHARED missing) | `SUPABASE_URL` | Cloud DB |
| `SHARED__SUPABASE_SERVICE_ROLE_KEY` | `SUPABASE_SERVICE_ROLE_KEY` | Bot writes |
| `SHARED__GROQ_API_KEY` | `GROQ_API_KEY` | AI scorer |

Integration agent will add the new mappings to `_VAULT_MAP` in `groww_bot.py`.

## Contract: Bot Flask HTTP (`bot_server.py`)

Exposed at `http://127.0.0.1:8080`. Used by existing
`apps/admin/src/app/api/ops/trader/route.ts` which expects:

```json
GET /api/status →
{
  "pnl_today": 245.50,
  "wallet_inr": 10100.00,
  "positions": [
    {
      "symbol": "TATASTEEL-EQ",
      "qty": 28,
      "entry_price": 144.20,
      "current_price": 146.10,
      "pnl": 53.20
    }
  ],
  "trades_today": 3,
  "strategies": [
    { "name": "equity", "win_rate": 0.66, "trades": 3 }
  ],
  "capital_utilization": 0.42,
  "mode": "live"
}

GET / → minimal HTML cockpit (for the existing iframe embed)
```

Also serves health probe (200 OK on any 200-class endpoint).

## Contract: AI scorer (`ai_scorer.py`)

```python
score(signal: dict) -> {confidence: int 0..100, reason: str}

# Input signal example:
{
  "strategy": "FNO_DEEP_OTM",
  "symbol": "COALINDIA26JUN420CE",
  "underlying": "COALINDIA",
  "premium": 2.15,
  "lot": 1350,
  "cost_inr": 2903,
  "otm_pct": 4.2,
  "volume": 18540,
  "oi": 92100,
  "underlying_change_pct": -1.8,
  "active_situational": []  # from trader_signals table
}
# Returns: {"confidence": 72, "reason": "OI rising 12% on -1.8% intraday dip; deep OTM affordable; no adverse situational; bias bullish bounce."}
```

Uses Groq's free `llama-3.3-70b-versatile` model. Cache key:
`(trading_symbol, floor(now_ts / 300))` — same key returns same answer for 5 minutes.

## Contract: Supabase writer (`trader_db.py`)

Public functions:

```python
class TraderDB:
    def __init__(self):
        # reads SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY from env
        ...

    # singletons
    def update_state(**fields) -> None
    def get_settings() -> dict
    def get_kill_switch() -> bool

    # scans
    def insert_scan(equity, fno_list, meanrev, active_situational, wallet) -> int

    # pending orders
    def create_pending_order(order_dict) -> str  # returns id
    def mark_pending(order_id, status: str, **fields) -> None

    # positions
    def upsert_position(position_dict) -> str  # by trading_symbol
    def close_position(trading_symbol) -> dict

    # trades (immutable)
    def insert_trade(trade_dict) -> str

    # commands queue (bot polls)
    def fetch_pending_commands() -> list[dict]
    def mark_command_consumed(command_id, result: dict) -> None

    # signals
    def fetch_active_signals_for_symbol(symbol: str) -> list[dict]
    def fetch_active_signals() -> list[dict]
```

All methods log on error, never raise (fail-soft — bot continues).
Uses `supabase` Python SDK with REST + Realtime.

## Contract: Next.js API routes

```
POST /api/trader/approve  body: { pending_order_id, decided_by? }
POST /api/trader/reject   body: { pending_order_id, reason? }
POST /api/trader/kill     body: { reason?, enable: true|false }  (toggle)
POST /api/trader/settings body: { ...partial settings update }
POST /api/trader/command  body: { command, payload? }            (generic)
```

All routes:
- Use existing admin auth (cookie-based; check `OPS_SECRET` header OR existing session).
- Write to `trader_settings` or `trader_commands` via Supabase service-role client.
- Return `{ success: true, data: {...} }` matching admin's existing envelope pattern.

## Contract: UI (`_cockpit-client.tsx`)

A SINGLE 'use client' React component exported as default. Imported by the
existing `trader/page.tsx` and rendered BELOW the existing probe + iframe
sections (additive, NOT replacement).

Subscribes via Supabase JS client (anon key) to realtime channels:
- `trader_state` (singleton)
- `trader_pending_orders` (filter status=pending)
- `trader_positions`
- `trader_trades` (last 20)
- `trader_settings`

Sections in order:
1. **Header strip** — wallet (big), day P&L (color), mode badge, market open/closed, kill button
2. **Pending approvals** — full-width cards. Each card: symbol, strategy, cost, AI confidence bar (color-coded), reason text, [APPROVE] [REJECT] buttons, "expires in Ns" countdown
3. **Open positions** — touch-friendly rows: symbol, qty, entry → ltp, P&L (color), strategy badge, exit-recommended highlight, [EXIT NOW] button
4. **Latest scan** — top-5 F&O opportunities + equity best + meanrev (compact table)
5. **Today's trades** — closed P&L row by row (collapsible)
6. **Active signals** — chips: 🇮🇱 oil_shock +30 COALINDIA,ONGC (Layer 4; show empty placeholder for now)
7. **Settings drawer** — slide-up panel: approval mode (manual/auto/hybrid), min confidence slider, daily loss cap, kill switch toggle

Responsive breakpoints:
- iPhone 375 → single column, large tap targets (48px+)
- iPad 768/1024 → 2-column grid for positions + scan
- Desktop 1280+ → full layout with side-by-side panels

Visual language: matches existing /trader page (dark, red accents, mono numbers).

## Contract: PWA shell

- `apps/admin/public/manifest.webmanifest` — name "JoSho Trader", short_name "Trader", display "standalone", theme_color black, start_url "/trader", icons (use existing admin favicon — no new image generation needed for now; reference `/favicon.ico` 192/512 as best-effort)
- `apps/admin/public/sw.js` — minimal: cache /trader shell on install, network-first for everything else, skip auth routes
- Registration: small client component `<RegisterSW />` mounted in `trader/page.tsx`
- iPad install: add meta tags in `app/layout.tsx` only on the `/trader` route via metadata export

## Boundaries (DO NOT CROSS — applies to every agent)

1. **Never replace existing files**, only extend / add new. Specifically:
   - `apps/admin/src/app/trader/page.tsx` — only Integration agent touches this, adds 2 lines
   - `automation/josho-trader/groww_bot.py` — only Integration agent touches this
   - `apps/admin/src/app/api/trader/status/route.ts` — preserved as-is
   - `apps/admin/src/app/api/ops/trader/route.ts` — preserved as-is
2. **Never write secret values** into any file. Reference env vars only.
3. **Vault file is owner-managed** — agents must not edit `vault.master.env`.
4. **Stay within own deliverable file(s)**. If you need a function from another
   agent's file, IMPORT it; do not duplicate.
5. **Free-only services**. No paid SDKs.
6. **Bot must keep working in `--scan-once` mode without Supabase reachable** —
   `trader_db.py` calls must fail-soft.
7. **Groww rate limits**: 10 orders/sec, 250 orders/min. 300 quotes/min. The bot
   already has throttling; do not bypass.

# JoSho Trader — Manual Test Plan (v1)

> Run this checklist ONCE end-to-end after all agents have wired their files.
> Total time: ~60 min in dry-run, +10 min if you run Phase 8 (real money).
> You don't need to write any code — every command is copy-pasteable.

**Hard rule:** Do NOT skip phases. Do NOT run Phase 8 until Phases 0–7 all pass.

---

## Pre-flight — terminal setup (2 min)

Open 3 PowerShell windows. Keep them open for the whole run.

```powershell
# Window A — the bot
cd C:\Sijoy_2.0\automation\josho-trader

# Window B — curl probes
cd C:\Sijoy_2.0\automation\josho-trader

# Window C — admin dev server
cd C:\Sijoy_2.0\apps\admin
```

Verify Python deps installed:

```powershell
pip install -r C:\Sijoy_2.0\automation\josho-trader\requirements.txt
```

Expected: `Successfully installed ...` or `Requirement already satisfied` for every package. No red errors.

---

## Phase 0 — Migration applied [5 min]

- [ ] Open Supabase dashboard → SQL Editor → New query.
- [ ] Paste the entire contents of `C:\Sijoy_2.0\supabase\migrations\040_trader_tables.sql`.
- [ ] Click **Run**. Expected: `Success. No rows returned`.
- [ ] In the same SQL Editor, run the verification query:

```sql
select table_name
from information_schema.tables
where table_schema = 'public'
  and table_name like 'trader_%'
order by 1;
```

- [ ] **Expected output: exactly 8 rows** in this order:
  ```
  trader_commands
  trader_pending_orders
  trader_positions
  trader_scans
  trader_settings
  trader_signals
  trader_state
  trader_trades
  ```

- [ ] Verify the singletons were seeded:

```sql
select id, approval_mode, kill_switch from public.trader_settings;
select id, mode, online from public.trader_state;
```

- [ ] **Acceptance:** Both queries return exactly 1 row with `id = 1`.

**If failed:** Check the SQL error message. Most likely cause: `gen_random_uuid()` not available → run `create extension if not exists pgcrypto;` first.

---

## Phase 1 — Bot can scan (no orders) [10 min]

**In Window A:**

```powershell
python groww_bot.py --scan-once
```

- [ ] Process exits with code 0 (run `echo $LASTEXITCODE` immediately after — should print `0`).
- [ ] Console prints the startup banner with `auth ready: True` and an `auth mode` (one of `direct_access_token`, `totp_exchange`, `api_key_secret`).
- [ ] Console prints `Wallet: Rs.XX,XXX` and the number matches the Groww app's available margin.
- [ ] Console prints `[F&O] top 5 option opportunities:` followed by 0–5 rows for COALINDIA/HINDALCO options.

- [ ] A scan file was written. Verify:

```powershell
Get-ChildItem C:\Sijoy_2.0\automation\josho-trader\data\scan_*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

- [ ] Open that file. Confirm `wallet`, `timestamp`, and `signals.fno_top5` keys exist. `fno_top5` should be a non-empty array on a market day during market hours.

**Acceptance:** Wallet matches Groww app, scan file written, no Python tracebacks in the log.

**If failed:** See Failure remediation table at the bottom — Phase 1 owner = `groww_bot.py` (auth wiring) + vault.

---

## Phase 2 — Bot writes to Supabase [10 min]

**Verify env vars exposed to the bot:**

```powershell
# In Window A
python -c "import os; print('URL=', bool(os.environ.get('SUPABASE_URL'))); print('KEY=', bool(os.environ.get('SUPABASE_SERVICE_ROLE_KEY')))"
```

- [ ] Both print `True`. If `False`, check `automation/secrets-os/vault.master.env` has `SHARED__SUPABASE_URL` and `SHARED__SUPABASE_SERVICE_ROLE_KEY`, and that `_VAULT_MAP` in `groww_bot.py` was extended by the Integration agent.

**Re-run the scan now that Supabase creds are populated:**

```powershell
python groww_bot.py --scan-once
```

- [ ] Console line `Scan saved to ... data\scan_<ts>.json` appears (same as Phase 1).
- [ ] No errors of the shape `supabase. *. failed` or `403 Forbidden`. Warnings (fail-soft) are OK; outright crashes are not.

**Verify the write in Supabase:**

```sql
select id, ts, wallet_inr, jsonb_array_length(coalesce(fno_opportunities, '[]'::jsonb)) as fno_count
from public.trader_scans
order by ts desc
limit 3;
```

- [ ] **Expected:** at least 1 row with `ts` = today's date (UTC), `wallet_inr` matches the Groww wallet, `fno_count` ≥ 0.

```sql
select id, mode, wallet_inr, online, updated_at
from public.trader_state
where id = 1;
```

- [ ] **Expected:** `mode = 'scan-once'`, `wallet_inr` matches scan, `updated_at` is within the last 5 minutes.

**Acceptance:** Rows visible in dashboard. JSON columns are parseable (Supabase shows them as expandable trees, not raw text).

**If failed:** Owner = `trader_db.py`. Check `SUPABASE_SERVICE_ROLE_KEY` is the **service_role** key, not the anon key (RLS would block writes with anon).

---

## Phase 3 — Flask HTTP layer up [5 min]

**In Window A:**

```powershell
python groww_bot.py --server --dry-run
```

(The `--server` flag is added by the Integration agent. If the flag name differs, check `groww_bot.py --help`.)

- [ ] Console prints `Flask running on http://127.0.0.1:8080` (or similar) within 30 seconds.
- [ ] Console also prints the normal startup banner and `Wallet: Rs.XX,XXX`.

**In Window B:**

```powershell
curl http://127.0.0.1:8080/healthz
```

- [ ] Response: `{"ok":true}` (or `{"ok": true, ...}` with extra keys).
- [ ] HTTP 200.

```powershell
curl http://127.0.0.1:8080/api/status
```

- [ ] Response is JSON with these keys (at minimum): `pnl_today`, `wallet_inr`, `positions`, `trades_today`, `strategies`, `capital_utilization`, `mode`.
- [ ] `mode` = `"dry-run"`. `wallet_inr` is a number, not null.
- [ ] HTTP 200, response time visibly < 200ms (curl returns instantly).

**In a browser:**

- [ ] Open `http://127.0.0.1:8080/`. Minimal HTML cockpit renders (table or simple page — no React, just plain HTML).

**Acceptance:** All 3 endpoints respond under 200ms. No 5xx errors in the bot console.

**If failed:** Owner = `bot_server.py`. Check port 8080 isn't already taken (`netstat -ano | findstr :8080`).

---

## Phase 4 — Existing /trader page lights up [3 min]

**Leave Window A running (`--server --dry-run`). In Window C:**

```powershell
npm run dev
```

- [ ] Output ends with `Ready in Xs` and a URL like `http://localhost:3000`.

**In a browser:**

- [ ] Open `http://localhost:3000/trader`.
- [ ] Log in if prompted (admin password is in `credentials_all_env`).
- [ ] **Probe section** at the top of the page shows:
  - **Local: ONLINE** (green dot) — because Window A's Flask is up.
  - **Tunnel:** ONLINE green if Cloudflare tunnel is configured, otherwise OFFLINE red (acceptable for this test).
- [ ] **Iframe embed** below the probe shows the same minimal HTML cockpit from Phase 3.

**Acceptance:** Status badge reads `"Online (local + tunnel)"` OR `"Running locally, tunnel down"`. Iframe is not blank.

**If failed:** Owner = `apps/admin/src/app/api/ops/trader/route.ts` (existing) + `trader/page.tsx`. Most likely: CORS / mixed-content (the existing route is server-side, so this is rare). Check browser dev-tools network tab for the failing fetch.

---

## Phase 5 — Realtime cockpit cards render [10 min]

**Same `/trader` page open. Scroll past the existing probe + iframe sections.**

- [ ] A new section `"Live cockpit"` (or similar heading) renders **below** the existing content.
- [ ] **Header strip** visible: wallet (large number), day P&L (0.00 in dry-run), mode badge `dry-run`, market open/closed dot, kill button.
- [ ] **Pending Approvals** section visible but empty (placeholder text like "No pending orders").
- [ ] **Open Positions** section visible but empty.
- [ ] **Latest scan** table visible with 1 row (the scan from Phase 2) showing top-5 F&O opportunities.

**Realtime check — insert a test pending order:**

In Supabase SQL Editor, paste and run:

```sql
insert into trader_pending_orders(
  segment, symbol, trading_symbol, qty, entry_price, cost_inr,
  strategy, ai_confidence, ai_reason
)
values (
  'FNO', 'COALINDIA', 'TEST26JUN420CE', 1350, 2.15, 2903,
  'FNO_DEEP_OTM', 72, 'Test card for UI verification'
);
```

- [ ] **Within 2 seconds, without refreshing the page**, a new card appears in the Pending Approvals section showing:
  - `TEST26JUN420CE` symbol
  - `FNO_DEEP_OTM` strategy badge
  - `₹2,903` cost
  - `72%` confidence (colored bar)
  - `"Test card for UI verification"` reason text
  - **[APPROVE]** and **[REJECT]** buttons

- [ ] Click **REJECT**. Within 2 seconds:
  - The card disappears from the UI.
  - In Supabase, re-run `select id, status from public.trader_pending_orders order by created_at desc limit 1;` → `status = 'rejected'`.
  - A row appears in `trader_commands` with `command = 'reject'`.

**Acceptance:** Realtime insert visible without reload. Reject button changes DB state.

**If failed:** Owner = `_cockpit-client.tsx`. Check browser console for `supabase.channel` errors. Most likely cause: anon key missing from `apps/admin/.env.local`.

---

## Phase 6 — Kill switch end-to-end [3 min]

**Window A still running `--server --dry-run`.**

- [ ] In the cockpit UI header strip, click the **kill switch toggle** (off → on).
- [ ] UI shows a confirm dialog: confirm.
- [ ] In Supabase: `select kill_switch from trader_settings where id=1;` → `true`.
- [ ] A row appears in `trader_commands` with `command = 'kill'`.

**Watch Window A:**

- [ ] **Within 30 seconds**, the bot prints: `KILL switch detected — shutting down` (or `KILL file detected`, or `KILL command consumed`).
- [ ] Process exits with code 0.
- [ ] If running under PM2 (`pm2 list`), entry shows status `stopped` (not `errored`, not `online`).

**Re-arm for next phase:**

```sql
update trader_settings set kill_switch = false where id = 1;
```

```powershell
# Window A — restart
python groww_bot.py --server --dry-run
```

- [ ] Bot starts again normally.

**Acceptance:** Bot honors the kill switch within 30 seconds; UI reflects the state change.

**If failed:** Owner = bot polling loop in `groww_bot.py` (Integration agent's hook into `trader_db.fetch_pending_commands()`).

---

## Phase 7 — AI scorer responding [5 min]

**Bot still running in Window A (`--server --dry-run`). Trigger a fresh scan:**

The continuous-loop bot scans automatically every 30s. Wait 60s, then check:

```sql
select id, created_at, trading_symbol, ai_confidence, ai_reason, status
from public.trader_pending_orders
where created_at > now() - interval '5 minutes'
order by created_at desc
limit 5;
```

- [ ] At least 1 row exists with `created_at` in the last 5 minutes.
- [ ] `ai_confidence` is a non-null integer between 0 and 100.
- [ ] `ai_reason` is a non-empty string (at least 10 characters).

**Test fallback path** — temporarily unset `GROQ_API_KEY`:

```powershell
# Window A — stop the bot (Ctrl+C)
$env:GROQ_API_KEY = ""
python groww_bot.py --server --dry-run
```

Wait 60s, then re-run the SELECT above on the newest row:

- [ ] `ai_reason` starts with `"rule-based fallback"` (or similar — the contract says it must be obvious that Groq was bypassed).

**Restore env and restart:**

```powershell
# Ctrl+C, then re-source vault
python -c "from groww_bot import _load_vault_env; _load_vault_env()"
python groww_bot.py --server --dry-run
```

**Acceptance:** AI scorer fills both fields. Fallback string is recognizable when Groq is unreachable.

**If failed:** Owner = `ai_scorer.py`. Check Groq API key validity at `https://console.groq.com/keys`.

---

## Phase 8 — Approve → execute round-trip [10 min] ⚠️ REAL MONEY

> # 🛑 STOP. READ THIS BOX.
>
> **This is the ONLY phase that places real orders with real INR.**
>
> **Hard cost cap for the test:** Use a single very-cheap OTM call where
> `premium ≤ ₹0.50` and `lot_cost ≤ ₹1,000`.
> If no such option exists in the live scan, **abort this phase** and try again
> tomorrow. Do not raise the cap.
>
> **Maximum possible loss for this test: ₹1,000** (option expires worthless).
> Most likely loss: ~₹100–300 round-trip fees + bid/ask spread.

**Pre-conditions:**
- Phases 0–7 all green.
- Market is open (09:15–15:30 IST, weekday).
- Wallet has ≥ ₹2,000 free margin.
- Groww has F&O segment enabled on your account.

**Procedure:**

```powershell
# Window A — stop dry-run, start live
# Ctrl+C
python groww_bot.py --server --live
```

- [ ] Banner shows `mode=live` and `dry_run=False`.
- [ ] Telegram notification arrives: `JOSHO TRADER ONLINE ... dry_run=False`.

**Wait for a scan to produce a pending order with `confidence ≥ 70` and `cost_inr ≤ 1000`.** Check Supabase or the cockpit UI's Pending Approvals.

- [ ] Open the cockpit on your iPad (or laptop) at `http://localhost:3000/trader`.
- [ ] Tap **APPROVE** on the cheap card.
- [ ] In Supabase:

```sql
select id, command, status, payload from trader_commands
where command = 'approve' order by created_at desc limit 1;
```
  → row exists, `status = 'pending'` then `'consumed'` within 30s.

- [ ] Within 30s, a row appears in `trader_positions`:

```sql
select trading_symbol, qty, entry_price, groww_order_id, opened_at
from trader_positions order by opened_at desc limit 1;
```
  → `groww_order_id` is **not** `'DRY-RUN'`.

- [ ] Open the Groww app on your phone → Portfolio → Positions: the position is visible there too with matching qty + entry price.
- [ ] In the cockpit UI, the Open Positions section shows the live row updating its `current_price` and `pnl_inr` every few seconds.
- [ ] Tap **EXIT NOW** on the position. Within 30s:
  - Position row disappears from `trader_positions`.
  - Row appears in `trader_trades` with `exit_price`, `pnl_inr`, `closed_at`.
  - Groww app shows the position is closed.

**Acceptance:** Round-trip completed. Position visible in Groww at every step. Total realized P&L (likely negative) is within ₹500 of expected (cost + 2x fees).

**If failed:** Stop. Do not retry until you understand why. Check `trader_pending_orders.failure_reason`. Most likely causes: insufficient margin, segment not enabled, symbol expired, Groww rate-limited.

---

## Phase 9 — iPad PWA install [3 min]

> Skip if you don't have an iPad. The rest of the system works without PWA.

- [ ] On iPad Safari, open `https://admin.joshoit.com/trader` (the deployed URL, not localhost).
- [ ] Tap the **Share** button (square with up arrow).
- [ ] Scroll → **Add to Home Screen** → keep the name `Trader` → **Add**.
- [ ] App icon appears on the home screen.
- [ ] Tap the icon. The app opens **fullscreen** with no Safari address bar, no tab bar, no chrome.
- [ ] Realtime still works: insert a test pending order via Supabase SQL (same query as Phase 5) — card appears in the PWA within 2 seconds.

**Acceptance:** App behaves like a native app. No Safari UI visible.

**If failed:** Owner = `apps/admin/public/manifest.webmanifest` + `sw.js`. Check `display: "standalone"` is set.

---

## Phase 10 — 24h soak [optional]

> Run this overnight before flipping to `--live` for production trading.

- [ ] Start the bot under PM2:

```powershell
cd C:\Sijoy_2.0\automation\josho-trader
pm2 start ecosystem.config.js
pm2 save
```

- [ ] Let it run for 24 hours in `--server --dry-run` mode.

**Checks the morning after:**

```powershell
pm2 list
```

- [ ] Process shows `online`, `restarts = 0` (or ≤ 1 — one restart from the 6 AM token refresh is OK).

```sql
select count(*) as scans_24h from trader_scans
where ts > now() - interval '24 hours';
```

- [ ] `scans_24h` is between **600 and 2900**. (Theoretical max = 2,880 = 30s × 60min × 24h; actual lower because market closed ~17.5h/day.)

```sql
select count(*) as errors_24h, max(last_error) as last
from trader_state where id = 1;
```

- [ ] `last_error` is null or stale (older than 1h).

**Check Cloudflare tunnel logs:** No 30-min bans.

**Acceptance:** No PM2 restarts beyond the daily token refresh, no rate-limit bans, no DB errors.

---

## Failure remediation cheatsheet

If a phase fails, identify the owner file from the table below and check the
first thing listed. Cross-reference with `RUNBOOK.md` (same folder) for full
debug procedure per component.

| Phase | First thing failed | Likely owner file | First thing to check |
|---|---|---|---|
| 0 | SQL error on migration | `supabase/migrations/040_trader_tables.sql` | `pgcrypto` extension; run `create extension if not exists pgcrypto;` |
| 1 | `auth ready: False` | `automation/secrets-os/vault.master.env` | One of GROWW_ACCESS_TOKEN / GROWW_TOTP_TOKEN+GROWW_TOTP_SECRET present |
| 1 | Crashes / no scan file | `groww_bot.py` | Run with `python -X tracemalloc` for full traceback; check `logs/groww_bot.log` |
| 2 | No rows in trader_scans | `trader_db.py` | SERVICE_ROLE_KEY is the service-role key, not anon |
| 3 | curl gets connection refused | `bot_server.py` | Port 8080 not taken: `netstat -ano \| findstr :8080` |
| 3 | /api/status missing keys | `bot_server.py` | Compare against contract in `_CONTRACT.md` lines 70–92 |
| 4 | Probe shows OFFLINE | `apps/admin/src/app/api/ops/trader/route.ts` | Existing route — should not have been modified |
| 4 | Iframe blank | `bot_server.py` GET / | View source: should be plain HTML, not 404 |
| 5 | Card doesn't appear | `_cockpit-client.tsx` | Browser console for `supabase.channel` subscription error; anon key in `apps/admin/.env.local` |
| 5 | REJECT does nothing | `apps/admin/src/app/api/trader/reject/route.ts` | OPS_SECRET header or session cookie present |
| 6 | Bot ignores kill | `groww_bot.py` polling hook | Integration agent must call `trader_db.fetch_pending_commands()` each tick |
| 7 | `ai_confidence` null | `ai_scorer.py` | `GROQ_API_KEY` populated; model name `llama-3.3-70b-versatile` still active |
| 8 | Order rejected by Groww | Groww account | Margin / F&O segment / symbol expiry |
| 8 | Position not visible in cockpit | `trader_db.upsert_position` | Realtime subscription on `trader_positions` |
| 9 | App not standalone | `manifest.webmanifest` | `display: "standalone"`, served with correct MIME |
| 10 | PM2 restart loop | `ecosystem.config.js` | `max_restarts` set; `autorestart` is `true`; cwd is correct |

---

## Sign-off

- [ ] All Phases 0–7 passed. **Safe to leave running in dry-run.**
- [ ] Phase 8 passed. **Safe to flip to `--live` for the trading day.**
- [ ] Phase 9 passed. **iPad PWA ready.**
- [ ] Phase 10 passed. **System ready for daily unattended operation.**

Tester: ______________________  Date: ______________________

Notes / regressions found:

_______________________________________________________________

_______________________________________________________________

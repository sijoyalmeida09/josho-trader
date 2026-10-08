# JoSho Trader — Groww F&O + Equity Bot

A multi-strategy intraday trading bot for the Indian market (NSE F&O + equity) via the Groww API. Built with safety-first principles: dry-run mode, a kill switch, a daily loss cap, and an AI signal scorer that filters noise before any order is placed.

> **This is experimental software. Trading involves real financial risk. Never run in live mode without understanding the code fully.**

---

## Features

- **Multi-strategy engine**: F&O (futures & options) + equity intraday
- **Signal intelligence**: GDELT news events, macro indicators, RSS feeds (Economic Times, etc.), OFAC delta, natural events — all scored before entry
- **AI scorer**: Confidence model that gates entries on signal strength
- **Safe-by-default**: dry-run mode logs every decision without placing orders; KILL file stops the bot cleanly; daily loss cap cuts all new positions at threshold
- **Telegram alerts**: real-time trade notifications and P&L updates
- **14+ tests**: approve/reject race, atomic claim, kill latch, fail-closed, entry gate, exit invariants

---

## Quick start

```bash
git clone https://github.com/sijoyalmeida09/josho-trader.git
cd josho-trader

# Install dependencies
pip install -r requirements.txt

# Copy env template and fill in your Groww credentials
cp .env.example .env
# Edit .env with your GROWW_API_KEY and GROWW_TOTP_SECRET

# Run in scan-once mode first (no orders placed, just shows what the bot sees)
python groww_bot.py --scan-once
```

---

## Modes (increasing risk level)

```bash
# 1. SCAN ONCE — connect, fetch balance, scan all stocks + option chains, then exit.
#    Zero orders. Use this to verify connectivity and see market data.
python groww_bot.py --scan-once

# 2. DRY RUN — full trading loop, logs every "would buy / would sell" but NEVER places orders.
python groww_bot.py --dry-run

# 3. LIVE — places real orders. Start here only after dry-run looks correct.
python groww_bot.py --live

# Strategy filter (works with any mode)
python groww_bot.py --scan-once --strategy fno
python groww_bot.py --live --strategy equity

# Status — print last persisted state and exit
python groww_bot.py --status
```

---

## Kill switch

```bash
# Stop a running live bot cleanly (saves state before exiting)
touch KILL           # Linux/Mac
New-Item KILL        # PowerShell/Windows
```

The bot polls for the `KILL` file at every scan tick.

---

## Configuration (`.env`)

```
GROWW_API_KEY=<from Groww app → Profile → API settings>
GROWW_TOTP_SECRET=<base32 TOTP secret from Groww>

# Telegram alerts (optional)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

**Risk controls (env vars)**:

| Variable | Default | Description |
|---|---|---|
| `DAILY_LOSS_CAP_INR` | 2000 | Stop new positions if day P&L drops below `-this` (₹) |
| `KILL_FILE_NAME` | `KILL` | Override kill file name |
| `JOSHO_TRADER_PORT` | 9000 | Control plane HTTP port |

---

## Architecture

```
groww_bot.py          ← main entry, orchestrates the loop
  ├── ingestors/      ← signal data sources
  │   ├── gdelt_events.py    GDELT news event stream
  │   ├── macro.py           macro indicators
  │   ├── rss_india.py       Indian financial RSS feeds
  │   ├── ofac_delta.py      OFAC sanctions delta
  │   └── natural_events.py  natural event monitoring
  ├── ai_scorer.py    ← signal confidence model
  ├── classifier.py   ← strategy classifier
  ├── signal_worker.py ← aggregates + scores signals
  ├── sequencer.ts    ← order sequencing
  ├── trader_db.py    ← SQLite persistence
  ├── control_plane.py ← HTTP control/status server
  └── bot_server.py   ← WebSocket state broadcaster
```

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Disclaimer

This software is provided for educational purposes. Use at your own risk. The authors take no responsibility for financial losses. Always verify trades and risk parameters before enabling live mode.

## Files

- `groww_bot.py` — the bot.
- `requirements.txt` — Python deps.
- `logs/groww_bot.log` — rolling log file.
- `data/state.json` — last persisted state (positions, day P&L).
- `data/scan_YYYYMMDD_HHMMSS.json` — every `--scan-once` writes one of these.

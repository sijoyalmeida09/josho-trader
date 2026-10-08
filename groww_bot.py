"""
JOSHO TRADER — Multi-Strategy Bot (Groww F&O + Equity)

Strategies running simultaneously:
  1. EQUITY INTRADAY (MIS)    — oversold bounce + momentum on 20 stocks
  2. F&O OPTIONS (deep OTM)   — lottery plays on COALINDIA / HINDALCO
  3. MEAN REVERSION           — buy stocks down >3% (SBIN-pattern, ~94% backtest)

Modes:
  --scan-once     One scan pass, print signals, exit. NO orders.        (default analysis mode)
  --dry-run       Continuous loop, scan + log signals, NO orders.       (sandbox)
  --live          Continuous loop, REAL orders placed.                  (requires explicit flag)
  --status        Show last persisted state file and exit.

Strategy filter:
  --strategy {all,equity,fno,meanrev}   (default: all)

Risk knobs (env vars):
  DAILY_LOSS_CAP_INR     Stop entering new positions after day P&L < -this (default 2000)
  KILL_FILE_NAME         Filename in this folder that, if present, halts the bot (default KILL)

Secrets (loaded from SOSOS vault automatically — automation/secrets-os/vault.master.env):
  GROWW_API_KEY        -> GROWW_API_KEY / GROWW_TOTP_TOKEN
  GROWW_TOTP_SECRET    -> GROWW_TOTP_SECRET
  TELEGRAM_BOT_TOKEN   -> TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID -> TELEGRAM_CHAT_ID
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from copy import deepcopy

# ─────────────────────────────────────────────────────────────
# PATHS (repo-relative)
# ─────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"
DATA_DIR = HERE / "data"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = DATA_DIR / "state.json"
KILL_FILE = HERE / os.environ.get("KILL_FILE_NAME", "KILL")
VAULT_FILE = HERE.parent.parent / "automation" / "secrets-os" / "vault.master.env"

# Repo root → vault path: <repo>/automation/josho-trader/groww_bot.py
# HERE = <repo>/automation/josho-trader  → parents[1] = <repo>
REPO_ROOT = HERE.parent.parent  # automation/josho-trader -> automation -> <repo>
VAULT_FILE = REPO_ROOT / "automation" / "secrets-os" / "vault.master.env"


# ─────────────────────────────────────────────────────────────
# ENV LOADER: vault → env vars (idempotent)
# ─────────────────────────────────────────────────────────────
_VAULT_MAP = {
    # User pastes using Groww's UI labels (TOTP_TOKEN = API key, TOTP_QRCODE = base32 secret).
    # Either set of names works; both populate the same downstream env vars.
    "SOHAYA__GROWW_TOTP_TOKEN":     ["GROWW_TOTP_TOKEN", "GROWW_API_KEY"],
    "SOHAYA__GROWW_TOTP_QRCODE":    ["GROWW_TOTP_SECRET"],
    "GROWW_API_KEY":        ["GROWW_API_KEY", "GROWW_TOTP_TOKEN"],
    "GROWW_TOTP_SECRET":    ["GROWW_TOTP_SECRET"],
    "SOHAYA__GROWW_ACCESS_TOKEN":   ["GROWW_ACCESS_TOKEN"],
    "TELEGRAM_BOT_TOKEN":   ["TELEGRAM_BOT_TOKEN"],
    "TELEGRAM_CHAT_ID": ["TELEGRAM_CHAT_ID"],
}

def _load_vault_env() -> dict:
    """Read SOSOS vault, set unprefixed env vars the bot expects. Report by downstream env var
    (not by vault key) so multiple synonymous vault keys don't double-count as missing."""
    report = {
        "vault_path": str(VAULT_FILE),
        "exists": VAULT_FILE.exists(),
        "downstream_set": [],
        "downstream_missing": [],
        "auth_ready": False,
        "auth_mode_available": None,
    }
    if not VAULT_FILE.exists():
        return report
    text = VAULT_FILE.read_text(encoding="utf-8", errors="replace")
    found = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k in _VAULT_MAP:
            found[k] = v
    # First pass: populate env vars from vault values
    for vault_key, env_names in _VAULT_MAP.items():
        val = found.get(vault_key, "")
        if val:
            for name in env_names:
                if not os.environ.get(name):
                    os.environ[name] = val
    # Second pass: report on downstream env var state
    downstream = sorted(set(name for names in _VAULT_MAP.values() for name in names))
    for name in downstream:
        if os.environ.get(name):
            report["downstream_set"].append(name)
        else:
            report["downstream_missing"].append(name)
    # Determine which auth mode is available
    if os.environ.get("GROWW_ACCESS_TOKEN"):
        report["auth_mode_available"] = "direct_access_token"
        report["auth_ready"] = True
    elif os.environ.get("GROWW_TOTP_SECRET") and (os.environ.get("GROWW_TOTP_TOKEN") or os.environ.get("GROWW_API_KEY")):
        report["auth_mode_available"] = "totp_exchange"
        report["auth_ready"] = True
    elif os.environ.get("GROWW_API_KEY") and os.environ.get("GROWW_SECRET_KEY"):
        report["auth_mode_available"] = "api_key_secret"
        report["auth_ready"] = True
    return report


# Load vault BEFORE third-party imports that may read env at import time
_vault_report = _load_vault_env()

# Also load a local .env if present (overrides vault)
try:
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env", override=True)
except ImportError:
    pass

# Third-party (will fail-fast if missing)
try:
    import pyotp
    import requests
    from growwapi import GrowwAPI
except ImportError as e:
    sys.stderr.write(
        f"\n[FATAL] Missing dependency: {e}\n"
        f"Install with:\n"
        f"    pip install -r {HERE / 'requirements.txt'}\n\n"
    )
    sys.exit(1)


IST = timezone(timedelta(hours=5, minutes=30))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "groww_bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("josho")

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")


def tg(msg: str) -> None:
    if TG_TOKEN and TG_CHAT:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json={"chat_id": TG_CHAT, "text": msg},
                timeout=10,
            )
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# CONTROL PLANE (additive, fail-soft) — kill latch, per-tick settings,
# exactly-once command consumption. Bot runs unchanged if these imports fail.
# ─────────────────────────────────────────────────────────────
try:
    from trader_db import TraderDB  # type: ignore
except Exception as _e:  # pragma: no cover - fail-soft
    TraderDB = None  # type: ignore
    log_msg = f"trader_db unavailable ({_e}); DB-backed guardrails disabled"
    logging.getLogger("josho").info(log_msg)

try:
    from control_plane import ControlPlane  # type: ignore
except Exception as _e:  # pragma: no cover - fail-soft
    ControlPlane = None  # type: ignore
    logging.getLogger("josho").info("control_plane unavailable (%s)", _e)

try:
    from bot_server import BotServer  # type: ignore
except Exception as _e:  # pragma: no cover - fail-soft
    BotServer = None  # type: ignore


def _build_control_plane(wallet_provider):
    """Construct a ControlPlane backed by a fail-soft TraderDB, or None."""
    if TraderDB is None or ControlPlane is None:
        return None
    try:
        return ControlPlane(
            db=TraderDB(),
            kill_file=KILL_FILE,
            wallet_provider=wallet_provider,
            notifier=tg,
        )
    except Exception as e:  # pragma: no cover - fail-soft
        log.warning("control plane init failed (%s) — running file-only guardrails", e)
        return None


# ═════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════
LOT_SIZES = {
    "HINDALCO": 700, "ADANIPOWER": 1250, "VEDL": 1150, "PNB": 8000,
    "SAIL": 4700, "TATASTEEL": 5500, "JSWSTEEL": 675, "COALINDIA": 1350,
    "BPCL": 1975, "ONGC": 3850, "TATAPOWER": 1450, "SUZLON": 9025,
    "HFCL": 7000, "YESBANK": 31100, "NBCC": 6500, "NHPC": 6400, "IRFC": 4250,
    "NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 40,
}

EQUITY_STOCKS = [
    "TATASTEEL", "SUZLON", "TATAPOWER", "VEDL", "COALINDIA",
    "BPCL", "ONGC", "SAIL", "PNB", "BANKBARODA",
    "SBIN", "ICICIBANK", "HDFCBANK", "RELIANCE", "INFY",
    "BAJFINANCE", "AXISBANK", "ITC", "BHARTIARTL", "ADANIPOWER",
]

FNO_STOCKS = ["COALINDIA", "HINDALCO"]

EQUITY_CONFIG = {
    "target_pct": 1.5, "stop_pct": -1.0,
    "max_per_trade_pct": 40, "max_trades_per_day": 8,
    "min_price": 15, "cycle_profit_minutes": 30, "redeploy_minutes": 60,
}

FNO_CONFIG = {
    "target_pct": 25, "deep_target_pct": 100, "stop_pct": -40,
    "max_per_trade": 3000, "min_score": 70,
    "trail_start_pct": 15, "trail_pct": 8,
}

DAILY_LOSS_CAP = float(os.environ.get("DAILY_LOSS_CAP_INR", "2000"))


# ═════════════════════════════════════════════════════════════
# FEES
# ═════════════════════════════════════════════════════════════
def calc_equity_fees(value: float) -> float:
    brokerage = min(20, value * 0.001) * 2
    stt = value * 0.00025
    exchange_txn = value * 0.0000297 * 2
    sebi = value * 0.000001 * 2
    gst = (brokerage + exchange_txn + sebi) * 0.18
    stamp = value * 0.00003
    return brokerage + stt + exchange_txn + sebi + gst + stamp


def calc_fno_fees(buy_premium: float, lot: int, sell_premium: float = 0) -> float:
    if sell_premium == 0:
        sell_premium = buy_premium
    buy_val = buy_premium * lot
    sell_val = sell_premium * lot
    brokerage = min(20, buy_val * 0.001) + min(20, sell_val * 0.001)
    stt = sell_val * 0.0015
    exchange_txn = (buy_val + sell_val) * 0.0003503
    sebi = (buy_val + sell_val) * 0.000001
    gst = (brokerage + exchange_txn + sebi) * 0.18
    stamp = buy_val * 0.00003
    return brokerage + stt + exchange_txn + sebi + gst + stamp


# ═════════════════════════════════════════════════════════════
# CONNECTION
# ═════════════════════════════════════════════════════════════
class SmartConnection:
    """Groww API wrapper with TOTP refresh, rate-limit handling, dry-run gating."""

    def __init__(self, dry_run: bool = False):
        self.api = None
        self.connected = False
        self.dry_run = dry_run
        self.auth_mode = None
        self.last_call = 0
        self.min_gap = 2.0
        self.calls_this_min = 0
        self.minute_start = time.time()
        self.max_calls_per_min = 8
        self.rate_limited_until = 0
        # Optional callable returning True when NEW entries must be blocked.
        # Default None preserves original behaviour; wired by the control plane.
        self._kill_check = None

    def connect(self) -> bool:
        if time.time() < self.rate_limited_until:
            return False
        try:
            # Priority 1: direct access token (skip TOTP exchange; expires daily 6am IST)
            access_token = os.environ.get("GROWW_ACCESS_TOKEN", "").strip()
            if access_token:
                self.api = GrowwAPI(access_token)
                self.connected = True
                self.auth_mode = "direct_access_token"
                log.info("Connected to Groww via direct access token (dry_run=%s)", self.dry_run)
                return True

            # Priority 2: TOTP exchange (mints a fresh token, valid till next 6am IST)
            totp_secret = os.environ.get("GROWW_TOTP_SECRET", "").strip()
            totp_token = (os.environ.get("GROWW_TOTP_TOKEN", "") or os.environ.get("GROWW_API_KEY", "")).strip()
            if totp_secret and totp_token:
                totp_code = pyotp.TOTP(totp_secret).now()
                token = GrowwAPI.get_access_token(api_key=totp_token, totp=totp_code)
                self.api = GrowwAPI(token)
                self.connected = True
                self.auth_mode = "totp_exchange"
                log.info("Connected to Groww via TOTP exchange (dry_run=%s)", self.dry_run)
                return True

            # Priority 3: legacy api_key + secret (Mode 1)
            api_key = os.environ.get("GROWW_API_KEY", "").strip()
            secret = os.environ.get("GROWW_SECRET_KEY", "").strip()
            if api_key and secret:
                token = GrowwAPI.get_access_token(api_key=api_key, secret=secret)
                self.api = GrowwAPI(token)
                self.connected = True
                self.auth_mode = "api_key_secret"
                log.info("Connected to Groww via api_key+secret (dry_run=%s)", self.dry_run)
                return True

            log.critical("No Groww credentials found. Need one of: GROWW_ACCESS_TOKEN | GROWW_TOTP_TOKEN+GROWW_TOTP_SECRET | GROWW_API_KEY+GROWW_SECRET_KEY.")
            return False
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg:
                self.rate_limited_until = time.time() + 600
                log.warning("Rate limited. Cooldown 10 min. %s", e)
            else:
                log.error("Connect failed: %s", e)
            return False

    def _throttle(self) -> None:
        if time.time() - self.minute_start > 60:
            self.calls_this_min = 0
            self.minute_start = time.time()
        if self.calls_this_min >= self.max_calls_per_min:
            wait = 60 - (time.time() - self.minute_start)
            if wait > 0:
                time.sleep(wait)
            self.calls_this_min = 0
            self.minute_start = time.time()
        elapsed = time.time() - self.last_call
        if elapsed < self.min_gap:
            time.sleep(self.min_gap - elapsed)
        self.last_call = time.time()
        self.calls_this_min += 1

    def call(self, fn, *args, **kwargs):
        if not self.connected or time.time() < self.rate_limited_until:
            return None
        self._throttle()
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                self.rate_limited_until = time.time() + 600
            log.error("API error: %s", e)
            return None

    def get_balance(self) -> float:
        margin = self.call(self.api.get_available_margin_details)
        if not margin:
            return 0
        return margin.get(
            "clear_cash",
            margin.get("fno_margin_details", {}).get("option_buy_balance_available", 0),
        )

    def get_quote(self, symbol, segment="CASH"):
        return self.call(self.api.get_quote, trading_symbol=symbol, exchange="NSE", segment=segment)

    def get_positions(self, segment=None):
        return self.call(self.api.get_positions_for_user, segment=segment)

    def buy(self, symbol, qty, product="MIS", segment="CASH"):
        # KILL CHECK — POINT 2: immediately before any NEW-ENTRY order fires.
        # Fail-closed: if the gate raises, refuse the buy. Sells are never gated
        # (exits / explicit flatten_all must always be able to run).
        if self._kill_check is not None:
            try:
                blocked = self._kill_check()
            except Exception as e:
                log.critical("kill_check raised (%s) — fail-closed, refusing BUY %s", e, symbol)
                return None
            if blocked:
                log.critical("ENTRY BLOCKED (kill/pause/uncertain) — refusing BUY %s qty=%s", symbol, qty)
                return None
        if self.dry_run:
            log.info("[DRY-RUN] would BUY %s qty=%s product=%s segment=%s", symbol, qty, product, segment)
            return {"groww_order_id": "DRY-RUN", "order_status": "DRY"}
        return self.call(
            self.api.place_order,
            trading_symbol=symbol, quantity=qty, validity="DAY",
            exchange="NSE", segment=segment, product=product,
            order_type="MARKET", transaction_type="BUY",
        )

    def sell(self, symbol, qty, product="MIS", segment="CASH"):
        if self.dry_run:
            log.info("[DRY-RUN] would SELL %s qty=%s product=%s segment=%s", symbol, qty, product, segment)
            return {"groww_order_id": "DRY-RUN", "order_status": "DRY"}
        return self.call(
            self.api.place_order,
            trading_symbol=symbol, quantity=qty, validity="DAY",
            exchange="NSE", segment=segment, product=product,
            order_type="MARKET", transaction_type="SELL",
        )

    def get_option_chain(self, underlying, expiry):
        return self.call(self.api.get_option_chain, exchange="NSE", underlying=underlying, expiry_date=expiry)

    def get_expiries(self, underlying):
        return self.call(self.api.get_expiries, exchange="NSE", underlying_symbol=underlying)


# ═════════════════════════════════════════════════════════════
# STRATEGY 1: EQUITY INTRADAY
# ═════════════════════════════════════════════════════════════
class EquityStrategy:
    def __init__(self, conn: SmartConnection):
        self.conn = conn
        self.position = None
        self.trades_today = 0
        self.day_pnl = 0.0

    def scan(self) -> dict:
        best = None
        best_score = 0
        for sym in EQUITY_STOCKS:
            q = self.conn.get_quote(f"{sym}-EQ")
            if not q or not q.get("last_price"):
                continue
            ltp = q["last_price"]
            prev = q.get("ohlc", {}).get("close", 0)
            high = q.get("ohlc", {}).get("high", ltp)
            low = q.get("ohlc", {}).get("low", ltp)
            if prev == 0 or ltp < EQUITY_CONFIG["min_price"]:
                continue
            change = ((ltp - prev) / prev) * 100
            pos_in_range = (ltp - low) / (high - low) if high > low else 0.5

            score, strategy = 0, ""
            if change < -1.5 and pos_in_range < 0.3:
                score = 50 + min(abs(change) * 5, 30); strategy = "OVERSOLD"
            elif change > 2.0 and pos_in_range > 0.8:
                score = 45 + min(change * 5, 25); strategy = "MOMENTUM"
            elif abs(change) < 0.3 and pos_in_range > 0.6:
                score = 30; strategy = "RANGE"

            if score > best_score:
                best_score = score
                best = {"symbol": sym, "ltp": ltp, "change": change, "score": score, "strategy": strategy}
        return best

    def enter(self, wallet: float) -> bool:
        if self.position or self.trades_today >= EQUITY_CONFIG["max_trades_per_day"]:
            return False
        signal = self.scan()
        if not signal or signal["score"] < 40:
            return False
        sym, ltp = signal["symbol"], signal["ltp"]
        qty = int((wallet * EQUITY_CONFIG["max_per_trade_pct"] / 100) / ltp)
        if qty == 0:
            return False
        value = ltp * qty
        fees = calc_equity_fees(value)
        expected = value * (EQUITY_CONFIG["target_pct"] / 100)
        if expected < fees * 3:
            return False
        result = self.conn.buy(f"{sym}-EQ", qty)
        if not result:
            return False
        self.position = {
            "symbol": sym, "qty": qty, "entry": ltp,
            "time": datetime.now(IST), "fees": fees, "strategy": signal["strategy"],
        }
        self.trades_today += 1
        log.info("EQ BUY: %sx %s @ Rs.%.2f (%s)", qty, sym, ltp, signal["strategy"])
        tg(f"EQ BUY {qty}x {sym} @ Rs.{ltp:.2f}\n{signal['strategy']} | score={signal['score']}")
        return True

    def check_exit(self) -> float:
        if not self.position:
            return 0
        sym = self.position["symbol"]
        q = self.conn.get_quote(f"{sym}-EQ")
        if not q or not q.get("last_price"):
            return 0
        ltp = q["last_price"]
        entry = self.position["entry"]
        pnl_pct = ((ltp - entry) / entry) * 100
        net_pnl = (ltp - entry) * self.position["qty"] - self.position["fees"]
        held = (datetime.now(IST) - self.position["time"]).total_seconds() / 60
        now = datetime.now(IST)

        reason = None
        if pnl_pct >= EQUITY_CONFIG["target_pct"]:
            reason = "TARGET"
        elif pnl_pct <= EQUITY_CONFIG["stop_pct"]:
            reason = "STOP"
        elif net_pnl > 0 and held > EQUITY_CONFIG["cycle_profit_minutes"]:
            reason = "PROFIT_CYCLE"
        elif pnl_pct < -0.3 and held > EQUITY_CONFIG["redeploy_minutes"]:
            reason = "REDEPLOY"
        elif now.hour == 15 and now.minute >= 10:
            reason = "MIS_CLOSE"

        if reason:
            self.conn.sell(f"{sym}-EQ", self.position["qty"])
            self.day_pnl += net_pnl
            log.info(f"EQ EXIT ({reason}): {sym} P&L=Rs.{net_pnl:+,.0f} ({pnl_pct:+.1f}%)")
            tg(f"EQ EXIT {sym} ({reason})\nP&L: Rs.{net_pnl:+,.0f} ({pnl_pct:+.1f}%)\nDay: Rs.{self.day_pnl:+,.0f}")
            self.position = None
            return net_pnl
        log.info(f"EQ HOLD: {sym} {pnl_pct:+.1f}% Rs.{net_pnl:+,.0f} held={held:.0f}min")
        return 0


# ═════════════════════════════════════════════════════════════
# STRATEGY 2: F&O OPTIONS (Deep OTM)
# ═════════════════════════════════════════════════════════════
class FnOStrategy:
    def __init__(self, conn: SmartConnection):
        self.conn = conn
        self.positions = []
        self.closed = []
        self.peaks = {}

    def scan_options(self, budget: float) -> list:
        opportunities = []
        today_iso = datetime.now(IST).strftime("%Y-%m-%d")
        for stock in FNO_STOCKS:
            expiries = self.conn.get_expiries(stock)
            if not expiries:
                continue
            valid = [e for e in expiries.get("expiries", []) if e >= today_iso]
            for exp in valid[:2]:
                chain = self.conn.get_option_chain(stock, exp)
                if not chain:
                    continue
                strikes = chain.get("strikes", {})
                ultp = chain.get("underlying_ltp", 0)
                lot = LOT_SIZES.get(stock, 0)
                if not lot:
                    continue
                for sp, sides in strikes.items():
                    ce = sides.get("CE", {})
                    sym = ce.get("trading_symbol", "")
                    ltp = ce.get("ltp", 0)
                    vol = ce.get("volume", 0)
                    oi = ce.get("open_interest", 0)
                    delta = ce.get("greeks", {}).get("delta", 0)
                    if not sym or ltp <= 0:
                        continue
                    strike = float(sp)
                    otm = ((strike - ultp) / ultp * 100) if ultp > 0 else 0
                    cost = ltp * lot
                    if cost > budget or cost < 50 or otm < 3:
                        continue
                    fees = calc_fno_fees(ltp, lot, ltp * 1.25)
                    if cost * 0.25 < fees * 2:
                        continue
                    opportunities.append({
                        "symbol": sym, "stock": stock, "strike": strike, "expiry": exp,
                        "premium": ltp, "lot": lot, "cost": cost, "otm_pct": otm,
                        "volume": vol, "oi": oi, "delta": delta, "fees": fees,
                    })
        opportunities.sort(key=lambda x: (x["volume"] + x["oi"], -x["otm_pct"]), reverse=True)
        return opportunities[:5]

    def enter(self, budget: float) -> bool:
        if sum(p.get("cost", 0) for p in self.positions) > budget * 0.8:
            return False
        opps = self.scan_options(budget)
        entered = False
        for opp in opps:
            sym = opp["symbol"]
            if any(p["symbol"] == sym for p in self.positions):
                continue
            result = self.conn.buy(sym, opp["lot"], product="NRML", segment="FNO")
            if not result or result.get("order_status") == "FAILED":
                log.warning("FNO BUY failed: %s - %s", sym, result)
                continue
            is_cheap = opp["premium"] < 1.0
            target = FNO_CONFIG["deep_target_pct"] if is_cheap else FNO_CONFIG["target_pct"]
            self.positions.append({
                "symbol": sym, "stock": opp["stock"], "lot": opp["lot"],
                "entry": opp["premium"], "cost": opp["cost"], "fees": opp["fees"],
                "target_pct": target, "stop_pct": FNO_CONFIG["stop_pct"],
                "expiry": opp["expiry"], "entry_time": datetime.now(IST).isoformat(),
                "order": result,
            })
            self.peaks[sym] = opp["premium"]
            entered = True
            log.info(f"FNO BUY: {sym} @ Rs.{opp['premium']} x {opp['lot']} = Rs.{opp['cost']:,.0f}")
            tg(f"FNO BUY {sym}\nRs.{opp['premium']} x {opp['lot']} = Rs.{opp['cost']:,.0f}\nOTM +{opp['otm_pct']:.1f}% | Target +{target}%")
        return entered

    def check_exits(self) -> float:
        booked = 0.0
        for pos in list(self.positions):
            sym = pos["symbol"]
            q = self.conn.get_quote(sym, segment="FNO")
            if not q or not q.get("last_price"):
                continue
            ltp = q["last_price"]
            entry = pos["entry"]
            pnl_pct = ((ltp - entry) / entry * 100) if entry > 0 else 0
            if ltp > self.peaks.get(sym, 0):
                self.peaks[sym] = ltp
            reason = None
            if pnl_pct >= pos["target_pct"]:
                reason = "TARGET"
            elif pnl_pct <= pos["stop_pct"]:
                reason = "STOP"
            elif pnl_pct >= FNO_CONFIG["trail_start_pct"]:
                trail_stop = self.peaks[sym] * (1 - FNO_CONFIG["trail_pct"] / 100)
                if ltp <= trail_stop:
                    reason = f"TRAIL_STOP (peak Rs.{self.peaks[sym]:.2f})"
            if reason:
                self.conn.sell(sym, pos["lot"], product="NRML", segment="FNO")
                net_pnl = (ltp - entry) * pos["lot"] - pos["fees"]
                pos["exit"] = ltp
                pos["net_pnl"] = round(net_pnl, 2)
                pos["reason"] = reason
                self.closed.append(deepcopy(pos))
                self.positions.remove(pos)
                self.peaks.pop(sym, None)
                booked += net_pnl
                win = "WIN" if net_pnl > 0 else "LOSS"
                log.info(f"FNO EXIT ({reason}): {sym} P&L=Rs.{net_pnl:+,.0f} ({pnl_pct:+.1f}%)")
                tg(f"FNO EXIT {win} ({reason})\n{sym}\nRs.{entry} -> Rs.{ltp}\nP&L: Rs.{net_pnl:+,.0f}")
            else:
                log.info(f"FNO HOLD: {sym} Rs.{ltp} pnl={pnl_pct:+.1f}%")
        return booked


# ═════════════════════════════════════════════════════════════
# STRATEGY 3: MEAN REVERSION
# ═════════════════════════════════════════════════════════════
class MeanReversionStrategy:
    WATCHLIST = ["SBIN", "HDFCBANK", "ICICIBANK", "BAJFINANCE", "AXISBANK", "COALINDIA"]

    def __init__(self, conn: SmartConnection):
        self.conn = conn
        self.position = None

    def scan(self) -> dict:
        for sym in self.WATCHLIST:
            q = self.conn.get_quote(f"{sym}-EQ")
            if not q or not q.get("last_price"):
                continue
            ltp = q["last_price"]
            prev = q.get("ohlc", {}).get("close", 0)
            if prev == 0:
                continue
            change = ((ltp - prev) / prev) * 100
            if change <= -3.0:
                return {"symbol": sym, "ltp": ltp, "drop_pct": change}
        return None

    def enter(self, wallet: float) -> bool:
        if self.position:
            return False
        signal = self.scan()
        if not signal:
            return False
        sym, ltp = signal["symbol"], signal["ltp"]
        qty = int((wallet * 0.3) / ltp)
        if qty == 0:
            return False
        if not self.conn.buy(f"{sym}-EQ", qty):
            return False
        self.position = {
            "symbol": sym, "qty": qty, "entry": ltp,
            "time": datetime.now(IST), "drop": signal["drop_pct"],
        }
        log.info(f"MEAN REV BUY: {qty}x {sym} @ Rs.{ltp:.2f} (drop {signal['drop_pct']:+.1f}%)")
        tg(f"MEAN REV BUY {qty}x {sym}\nDrop: {signal['drop_pct']:.1f}%\nExpecting bounce")
        return True

    def check_exit(self) -> float:
        if not self.position:
            return 0
        sym = self.position["symbol"]
        q = self.conn.get_quote(f"{sym}-EQ")
        if not q or not q.get("last_price"):
            return 0
        ltp = q["last_price"]
        entry = self.position["entry"]
        pnl_pct = ((ltp - entry) / entry) * 100
        held = (datetime.now(IST) - self.position["time"]).total_seconds() / 60
        target = abs(self.position["drop"]) * 0.4
        if pnl_pct >= target or pnl_pct >= 1.5:
            net_pnl = (ltp - entry) * self.position["qty"]
            self.conn.sell(f"{sym}-EQ", self.position["qty"])
            self.position = None
            log.info(f"MEAN REV EXIT: {sym} +{pnl_pct:.1f}% Rs.{net_pnl:+,.0f}")
            tg(f"MEAN REV EXIT {sym}\nBounce: +{pnl_pct:.1f}%\nP&L: Rs.{net_pnl:+,.0f}")
            return net_pnl
        if pnl_pct <= -2.0 or held > 120:
            net_pnl = (ltp - entry) * self.position["qty"]
            self.conn.sell(f"{sym}-EQ", self.position["qty"])
            self.position = None
            return net_pnl
        return 0


# ═════════════════════════════════════════════════════════════
# STATE PERSISTENCE
# ═════════════════════════════════════════════════════════════
def save_state(equity, fno, mean_rev, wallet: float) -> None:
    """Persist enough state to resume after a crash."""
    state = {
        "ts": datetime.now(IST).isoformat(),
        "wallet": wallet,
        "equity": {
            "position": _serialize(equity.position) if equity and equity.position else None,
            "trades_today": equity.trades_today if equity else 0,
            "day_pnl": equity.day_pnl if equity else 0,
        } if equity else None,
        "fno": {
            "positions": [_serialize(p) for p in fno.positions],
            "closed": fno.closed,
            "peaks": fno.peaks,
        } if fno else None,
        "mean_rev": {
            "position": _serialize(mean_rev.position) if mean_rev and mean_rev.position else None,
        } if mean_rev else None,
    }
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _serialize(p):
    if not p:
        return None
    out = dict(p)
    for k, v in list(out.items()):
        if isinstance(v, datetime):
            out[k] = v.isoformat()
    return out


# ═════════════════════════════════════════════════════════════
# RUN MODES
# ═════════════════════════════════════════════════════════════
def _market_is_open(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 15 <= minutes <= 15 * 60 + 30


def _print_startup_banner(args, dry_run: bool):
    now_ist = datetime.now(IST)
    print("\n" + "═" * 64)
    print(f"  JOSHO TRADER — mode={args.mode}  strategy={args.strategy}  dry_run={dry_run}")
    print(f"  IST: {now_ist.strftime('%Y-%m-%d %H:%M:%S %a')}  market_open={_market_is_open(now_ist)}")
    print(f"  daily_loss_cap: Rs.{DAILY_LOSS_CAP:,.0f}   kill_file: {KILL_FILE.name}")
    print(f"  env populated:   {sorted(_vault_report.get('downstream_set', []))}")
    if _vault_report.get("downstream_missing"):
        print(f"  env missing:     {sorted(_vault_report['downstream_missing'])}")
    print(f"  auth ready:      {_vault_report.get('auth_ready')}   mode={_vault_report.get('auth_mode_available')}")
    print("═" * 64 + "\n")


# ─────────────────────────────────────────────────────────────
# CONTROL-PLANE LOOP HELPERS (additive)
# ─────────────────────────────────────────────────────────────
def _state_snapshot(wallet: float, equity, fno, mean_rev, dry_run: bool) -> dict:
    """Build the bot_server /api/status payload from live strategy objects."""
    positions = []
    day_pnl = 0.0
    trades_today = 0
    if equity and equity.position:
        p = equity.position
        positions.append({"symbol": f"{p['symbol']}-EQ", "qty": p["qty"], "entry_price": p["entry"]})
    if equity:
        day_pnl += equity.day_pnl
        trades_today += equity.trades_today
    if fno:
        for p in fno.positions:
            positions.append({"symbol": p["symbol"], "qty": p.get("lot", 0), "entry_price": p.get("entry", 0)})
        day_pnl += sum(p.get("net_pnl", 0) for p in fno.closed)
        trades_today += len(fno.closed)
    if mean_rev and mean_rev.position:
        p = mean_rev.position
        positions.append({"symbol": f"{p['symbol']}-EQ", "qty": p["qty"], "entry_price": p["entry"]})
    return {
        "pnl_today": day_pnl,
        "wallet_inr": wallet,
        "positions": positions,
        "trades_today": trades_today,
        "strategies": [],
        "capital_utilization": 0.0,
        "mode": "live" if not dry_run else "paper",
    }


def _start_server(state_provider) -> None:
    """Start the Flask cockpit on a daemon thread (best-effort)."""
    if BotServer is None:
        log.warning("--server requested but bot_server unavailable")
        return
    try:
        BotServer(state_provider, logger=log).start_in_background()
    except Exception as e:  # pragma: no cover - port-in-use etc.
        log.warning("bot_server failed to start: %s", e)


def _force_exit(conn, equity, fno, mean_rev, trading_symbol=None, flatten=False) -> dict:
    """Explicit liquidation path used by manual_exit / flatten_all. Issues SELLs
    only (never gated by kill). Exits a single trading_symbol, or all if flatten."""
    exited = []
    if fno:
        for p in list(fno.positions):
            if flatten or p.get("symbol") == trading_symbol:
                conn.sell(p["symbol"], p.get("lot", 0), product="NRML", segment="FNO")
                fno.positions.remove(p)
                fno.peaks.pop(p.get("symbol"), None)
                exited.append(p.get("symbol"))
    for strat in (equity, mean_rev):
        if strat and strat.position:
            p = strat.position
            tsym = f"{p['symbol']}-EQ"
            if flatten or tsym == trading_symbol or p["symbol"] == trading_symbol:
                conn.sell(tsym, p["qty"])
                strat.position = None
                exited.append(tsym)
    return {"ok": True, "exited": exited}


def _build_command_handlers(cp, conn, equity, fno, mean_rev) -> dict:
    """Map DB command names to handlers. Kill overrides everything; manual
    entries are not executed here (approve is acked, not auto-filled)."""

    def h_kill(row):
        payload = row.get("payload") or {}
        enable = payload.get("enable", True)
        if enable is False:
            ok = cp.clear_kill(confirm=bool(payload.get("confirm")))
            return {"ok": ok, "action": "kill_cleared" if ok else "clear_refused"}
        cp.engage_kill(reason=str(payload.get("reason") or "command"))
        return {"ok": True, "action": "kill_latched"}

    def h_pause(_row):
        cp.set_pause(True)
        return {"ok": True, "action": "paused"}

    def h_resume(_row):
        cp.set_pause(False)
        return {"ok": True, "action": "resumed"}

    def h_set_setting(row):
        # toggle_strategy: payload {strategy, enabled}
        p = row.get("payload") or {}
        strat = p.get("strategy")
        if strat in ("equity", "fno", "meanrev") and "enabled" in p:
            cp.set_strategy_disabled(strat, not bool(p["enabled"]))
            return {"ok": True, "action": "strategy_toggled", "strategy": strat, "enabled": bool(p["enabled"])}
        return {"ok": False, "error": "set_setting needs payload {strategy, enabled}"}

    def h_exit(row):
        # manual_exit (payload.trading_symbol) OR flatten_all (payload.all=true)
        p = row.get("payload") or {}
        flatten = bool(p.get("all"))
        tsym = p.get("trading_symbol")
        if not flatten and not tsym:
            return {"ok": False, "error": "exit_position needs payload.trading_symbol or all=true"}
        return _force_exit(conn, equity, fno, mean_rev, trading_symbol=tsym, flatten=flatten)

    def h_approve_ack(_row):
        # Approval execution is handled by the strategy layer, not here. Kill
        # overrides: a manual approval is refused while kill is engaged.
        if cp.kill_engaged():
            return {"ok": False, "error": "kill engaged — approval not executed"}
        return {"ok": True, "action": "approval_acknowledged"}

    def h_reject_ack(_row):
        return {"ok": True, "action": "reject_acknowledged"}

    return {
        "kill": h_kill,
        "pause": h_pause,
        "resume": h_resume,
        "set_setting": h_set_setting,
        "exit_position": h_exit,
        "approve": h_approve_ack,
        "reject": h_reject_ack,
    }


def scan_once(strategy_filter: str, with_server: bool = False) -> int:
    """Single scan pass — connect, balance, scan all strategies, print, exit. NO orders."""
    # ---- CONTROL-PLANE PROOF TICK (runs before Groww connect so it works even
    #      when the broker/network is unavailable). Reads settings + commands +
    #      kill at the same three points the live loop uses. Read-only here:
    #      pending commands are peeked, not claimed, so scan-once has no side
    #      effects on the queue. ----
    cp = _build_control_plane(lambda: 0.0)
    if cp is not None:
        snap = cp.snapshot()
        print("\n[CONTROL PLANE] guardrails snapshot (read-only):")
        print(f"  kill: engaged={snap['kill']['engaged']} latched={snap['kill']['latched']} "
              f"local_file={snap['kill']['local_file']} db_kill_switch={snap['kill']['db_kill_switch']} "
              f"uncertain={snap['kill']['uncertain']} db_enabled={snap['kill']['db_enabled']} "
              f"sources={snap['kill']['sources']}")
        print(f"  new_entries_blocked: {snap['new_entries_blocked']}")
        print(f"  settings(effective): {snap['settings']}")
        print(f"  pending_commands: {snap['pending_commands']}")
    else:
        print("\n[CONTROL PLANE] unavailable (trader_db/control_plane import failed) — "
              "file-only kill guardrail active.")

    if with_server:
        _start_server(lambda: _state_snapshot(0.0, None, None, None, True))

    conn = SmartConnection(dry_run=True)
    if cp is not None:
        conn._kill_check = cp.new_entries_blocked  # POINT 2 wiring (no buys in scan-once)
    if not conn.connect():
        log.critical("Could not connect to Groww. Check credentials.")
        log.info("Control-plane proof tick already ran above; exiting after broker-connect failure.")
        return 2
    wallet = conn.get_balance()
    log.info(f"Wallet: Rs.{wallet:,.0f}")

    out = {"wallet": wallet, "timestamp": datetime.now(IST).isoformat(), "signals": {}}

    if strategy_filter in ("all", "equity"):
        eq = EquityStrategy(conn)
        sig = eq.scan()
        out["signals"]["equity_best"] = sig
        print(f"\n[EQUITY] best signal: {sig}")

    if strategy_filter in ("all", "fno"):
        fno = FnOStrategy(conn)
        opps = fno.scan_options(min(wallet, FNO_CONFIG["max_per_trade"]))
        out["signals"]["fno_top5"] = opps
        print(f"\n[F&O] top {len(opps)} option opportunities:")
        for o in opps:
            print(f"  {o['symbol']:30s} prem=Rs.{o['premium']:>7.2f} lot={o['lot']:>5} cost=Rs.{o['cost']:>8,.0f} OTM={o['otm_pct']:+.1f}% vol={o['volume']} oi={o['oi']}")

    if strategy_filter in ("all", "meanrev"):
        mr = MeanReversionStrategy(conn)
        sig = mr.scan()
        out["signals"]["meanrev"] = sig
        print(f"\n[MEAN REV] signal: {sig}")

    scan_file = DATA_DIR / f"scan_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.json"
    scan_file.write_text(json.dumps(out, indent=2, default=str))
    log.info("Scan saved to %s", scan_file)
    return 0


def run_loop(strategy_filter: str, dry_run: bool, with_server: bool = False) -> int:
    conn = SmartConnection(dry_run=dry_run)
    if not conn.connect():
        log.critical("Cannot connect to Groww. Waiting 5 min...")
        time.sleep(300)
        if not conn.connect():
            log.critical("Still can't connect. Exiting.")
            return 2

    wallet = conn.get_balance()
    log.info(f"Wallet: Rs.{wallet:,.0f}  (dry_run={dry_run})")
    tg(f"JOSHO TRADER ONLINE\nWallet: Rs.{wallet:,.0f}\nStrategies: {strategy_filter}  dry_run={dry_run}")

    equity = EquityStrategy(conn) if strategy_filter in ("all", "equity") else None
    fno = FnOStrategy(conn) if strategy_filter in ("all", "fno") else None
    mean_rev = MeanReversionStrategy(conn) if strategy_filter in ("all", "meanrev") else None

    # ---- CONTROL PLANE wiring (additive) ----
    wallet_box = [wallet]  # mutable so the control plane sees the latest balance
    cp = _build_control_plane(lambda: wallet_box[0])
    if cp is not None:
        conn._kill_check = cp.new_entries_blocked  # POINT 2: gates every BUY
        handlers = _build_command_handlers(cp, conn, equity, fno, mean_rev)
        log.info("control plane ACTIVE (db_enabled=%s)", getattr(cp, "_db_enabled", False))
    else:
        handlers = {}
        log.info("control plane INACTIVE — file-only KILL guardrail")
    if with_server:
        _start_server(lambda: _state_snapshot(wallet_box[0], equity, fno, mean_rev, dry_run))

    scan = 0
    while True:
        now = datetime.now(IST)
        scan += 1

        # ---- CONTROL PLANE: consume queued commands exactly once (kill, pause,
        #      resume, toggle_strategy, manual_exit, flatten_all). Runs every
        #      pass, including off-hours, so kill/flatten act without delay. ----
        if cp is not None:
            cp.consume_commands(handlers)

        # KILL CHECK — POINT 1 (loop top) + POINT 3 (synchronous local KILL-file
        # read happens inside new_entries_blocked -> kill_engaged -> refresh_kill).
        # LATCHING + FAIL-CLOSED. Kill halts NEW ENTRIES only; exits below still
        # run and open positions are NEVER auto-liquidated.
        entries_blocked = cp.new_entries_blocked() if cp is not None else KILL_FILE.exists()
        if entries_blocked and scan % 10 == 1:
            log.critical("KILL/PAUSE engaged — new entries halted; managing exits only (no auto-liquidation).")

        if not conn.connected:
            conn.connect()
            continue

        if not _market_is_open(now):
            if now.hour == 15 and now.minute >= 31 and scan == 1:
                day_pnl = (equity.day_pnl if equity else 0)
                tg(f"MARKET CLOSED\nDay P&L: Rs.{day_pnl:+,.0f}\nF&O positions held: {len(fno.positions) if fno else 0}")
            time.sleep(60)
            continue

        if scan % 10 == 1:
            new_bal = conn.get_balance()
            if new_bal > 0:
                wallet = new_bal

        # Exits
        if equity and equity.position:
            pnl = equity.check_exit()
            if pnl:
                wallet += pnl
        if fno and fno.positions:
            wallet += fno.check_exits()
        if mean_rev:
            mean_rev.check_exit()

        # Daily loss cap — read EFFECTIVE caps from settings each tick (no
        # startup caching) so UI changes apply with no restart. Falls back to
        # the module defaults when the control plane is inactive.
        wallet_box[0] = wallet
        eff = cp.read_settings() if cp is not None else None
        daily_cap = eff.daily_loss_cap_inr if eff else DAILY_LOSS_CAP
        per_trade_cap = eff.per_trade_cap_inr if eff else FNO_CONFIG["max_per_trade"]
        max_fno_pos = eff.max_concurrent_fno_positions if eff else FNO_CONFIG.get("max_positions", 2)

        day_pnl = (equity.day_pnl if equity else 0) + sum(p.get("net_pnl", 0) for p in (fno.closed if fno else []))
        cap_breached = day_pnl <= -daily_cap

        fno_invested = sum(p.get("cost", 0) for p in fno.positions) if fno else 0
        eq_invested = (equity.position["entry"] * equity.position["qty"]) if equity and equity.position else 0
        available = wallet - fno_invested - eq_invested

        def _strat_on(name: str) -> bool:
            return cp.strategy_enabled(name) if cp is not None else True

        if cap_breached or entries_blocked:
            if scan % 10 == 0:
                why = f"cap Rs.{daily_cap:,.0f} (day_pnl=Rs.{day_pnl:+,.0f})" if cap_breached else "kill/pause"
                log.warning(f"NEW ENTRIES HALTED [{why}] — holding/managing exits only.")
        else:
            if (fno and _strat_on("fno") and scan % 5 == 0 and available > 500
                    and len(fno.positions) < max_fno_pos):
                fno.enter(min(available * 0.6, per_trade_cap))
            if equity and _strat_on("equity") and not equity.position and available > 1000 and now.hour < 14:
                equity.enter(available)
            if mean_rev and _strat_on("meanrev") and not mean_rev.position and available > 1000:
                mean_rev.enter(available)

        if scan % 10 == 0:
            save_state(equity, fno, mean_rev, wallet)
            eq_state = "OPEN" if equity and equity.position else "none"
            mr_state = "OPEN" if mean_rev and mean_rev.position else "none"
            fno_count = len(fno.positions) if fno else 0
            cap_tag = "  [CAP-LOCKED]" if cap_breached else ("  [KILL/PAUSE]" if entries_blocked else "")
            log.info(f"[SCAN {scan}] wallet=Rs.{wallet:,.0f} | eq={eq_state} | fno={fno_count} | meanrev={mr_state} | day_pnl=Rs.{day_pnl:+,.0f}{cap_tag}")

        time.sleep(30)


def show_status() -> int:
    if STATE_FILE.exists():
        print(STATE_FILE.read_text())
    else:
        print(f"No state file yet at {STATE_FILE}")
    return 0


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════
def main() -> int:
    parser = argparse.ArgumentParser(description="JoSho Trader — Groww multi-strategy bot")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scan-once", dest="scan_once", action="store_true",
                      help="One scan pass, print signals, exit. NO orders. [default analysis mode]")
    mode.add_argument("--dry-run", dest="dry_run", action="store_true",
                      help="Continuous loop, scan + log, NO orders.")
    mode.add_argument("--live", action="store_true",
                      help="Continuous loop, REAL orders placed. Requires explicit flag.")
    mode.add_argument("--status", action="store_true",
                      help="Print last persisted state and exit.")
    parser.add_argument("--strategy", default="all", choices=["all", "equity", "fno", "meanrev"])
    parser.add_argument("--server", action="store_true",
                        help="Start the Flask cockpit (bot_server) on 127.0.0.1:8080.")
    args = parser.parse_args()

    # Resolve mode
    if args.status:
        args.mode = "status"
    elif args.live:
        args.mode = "live"
    elif args.dry_run:
        args.mode = "dry-run"
    else:
        args.mode = "scan-once"

    _print_startup_banner(args, dry_run=(args.mode != "live"))

    if args.mode == "status":
        return show_status()
    if args.mode == "scan-once":
        return scan_once(args.strategy, with_server=args.server)
    if args.mode == "dry-run":
        return run_loop(args.strategy, dry_run=True, with_server=args.server)
    if args.mode == "live":
        if KILL_FILE.exists():
            log.critical("KILL file exists at %s — refusing to start live. Delete and retry.", KILL_FILE)
            return 3
        return run_loop(args.strategy, dry_run=False, with_server=args.server)
    return 1


if __name__ == "__main__":
    sys.exit(main())

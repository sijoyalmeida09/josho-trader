"""
sijoy_strategy.py — Sijoy's Personal Trading Strategy
======================================================
Built from real losses and wins. Every line exists because something broke before.

Lessons baked in:
- Token caching (Cloudflare bans after 60 auth calls)
- Fee-first thinking (never trade if fees > expected profit)
- Fast cycles (take profit, redeploy, don't hold losers)
- Smart throttle (max 8 API calls/min, 2s gap)
- 24/7 operation (F&O overnight, equity intraday)

Usage:
  python sijoy_strategy.py                # Run live
  python sijoy_strategy.py --paper        # Paper mode (safe)
  python sijoy_strategy.py --status       # Show positions
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

import requests
import pyotp
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

IST = timezone(timedelta(hours=5, minutes=30))
DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
LOGS = Path(__file__).parent / "logs"
LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS / "sijoy_strategy.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("sijoy")

# ── Config ──────────────────────────────────────────────────────

CAPITAL = 7500  # Updated when wallet is read
MIN_TRADE_VALUE = 800  # Don't trade less than this (fees eat it)
MIN_STOCK_PRICE = 15  # Skip penny stocks
MAX_TRADES_PER_DAY = 12
MAX_OPEN_POSITIONS = 4
TARGET_PCT_EQUITY = 1.5  # Exit equity at +1.5%
STOP_PCT_EQUITY = -1.0  # Stop equity at -1%
TARGET_PCT_FNO = 25.0  # Exit options at +25%
STOP_PCT_FNO = -50.0  # Stop options at -50%
FAST_EXIT_MINUTES = 30  # Take any equity profit after 30 min
API_CALLS_PER_MIN = 8  # Groww Cloudflare limit
API_CALL_GAP = 2.0  # Min seconds between calls

# Telegram
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

# F&O lot sizes (Apr 2026)
LOT_SIZES = {
    "ADANIPOWER": 1250, "HINDALCO": 550, "VEDL": 1500, "PNB": 4000,
    "SAIL": 4000, "TATASTEEL": 1500, "TATAPOWER": 1350, "COALINDIA": 1500,
    "BPCL": 1100, "ONGC": 3850, "SUZLON": 4000, "NBCC": 7000,
}

# Equity universe — quality F&O stocks only (no penny stocks)
EQUITY_UNIVERSE = [
    "TATASTEEL", "SUZLON", "TATAPOWER", "VEDL", "COALINDIA",
    "BPCL", "ONGC", "SAIL", "PNB", "BANKBARODA",
    "SBIN", "ICICIBANK", "HDFCBANK", "RELIANCE", "INFY",
    "ADANIPOWER", "BAJFINANCE", "AXISBANK", "ITC", "BHARTIARTL",
]


def tg(msg: str):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": f"📈 *SIJOY TRADER*\n\n{msg}", "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception:
        pass


# ── Token Manager (NEVER get rate limited again) ───────────────

class TokenManager:
    """Manages Groww access tokens with disk caching.

    KEY INSIGHT: Cloudflare rate-limits the auth endpoint at ~60 calls/30min.
    Each GrowwAPI.get_access_token() = 1 call. If we cache the token to disk,
    restarts cost ZERO API calls. Token valid for 4+ hours.
    """
    CACHE_FILE = DATA / ".groww_token_cache"

    def __init__(self):
        self.token = None
        self.expiry = 0
        self._load_cache()

    def _load_cache(self):
        try:
            if self.CACHE_FILE.exists():
                data = json.loads(self.CACHE_FILE.read_text())
                if data.get("expiry", 0) > time.time() + 300:
                    self.token = data["token"]
                    self.expiry = data["expiry"]
                    remaining = (self.expiry - time.time()) / 60
                    log.info(f"Cached token loaded ({remaining:.0f} min remaining)")
        except Exception:
            pass

    def _save_cache(self):
        try:
            self.CACHE_FILE.write_text(json.dumps({
                "token": self.token, "expiry": self.expiry
            }))
        except Exception:
            pass

    def get_token(self) -> str:
        """Get valid access token. Uses cache first, auth only if expired."""
        if self.token and time.time() < self.expiry:
            return self.token

        log.info("Token expired/missing. Authenticating...")
        from growwapi import GrowwAPI

        totp_secret = os.environ.get("GROWW_TOTP_SECRET", "")
        totp_token = os.environ.get("GROWW_TOTP_TOKEN", "")
        api_key = os.environ.get("GROWW_API_KEY", "")
        secret = os.environ.get("GROWW_SECRET_KEY", "")

        # Method 1: TOTP (preferred — fully automated)
        if totp_secret and totp_token:
            try:
                code = pyotp.TOTP(totp_secret).now()
                log.info(f"TOTP auth (code: {code})")
                self.token = GrowwAPI.get_access_token(api_key=totp_token, totp=code)
                self.expiry = time.time() + 3600 * 4
                self._save_cache()
                log.info("Authenticated via TOTP — token cached to disk")
                return self.token
            except Exception as e:
                log.warning(f"TOTP failed: {e}")

        # Method 2: API Key + Secret
        if api_key and secret:
            try:
                self.token = GrowwAPI.get_access_token(api_key=api_key, secret=secret)
                self.expiry = time.time() + 3600 * 4
                self._save_cache()
                log.info("Authenticated via API key — token cached")
                return self.token
            except Exception as e:
                log.error(f"API key auth failed: {e}")

        raise RuntimeError("All auth methods failed")


# ── Smart API Client (rate limit aware) ────────────────────────

class SmartAPI:
    """Groww API wrapper that NEVER hits rate limits."""

    def __init__(self, paper: bool = False):
        self.paper = paper
        self.token_mgr = TokenManager()
        self.api = None
        self.last_call = 0
        self.calls_this_minute = 0
        self.minute_start = time.time()

    def connect(self) -> bool:
        try:
            from growwapi import GrowwAPI
            token = self.token_mgr.get_token()
            self.api = GrowwAPI(token)
            log.info("Groww API ready")
            return True
        except Exception as e:
            log.error(f"Connect failed: {e}")
            return False

    def _throttle(self):
        """Enforce rate limits locally so Cloudflare never blocks us."""
        now = time.time()
        if now - self.minute_start > 60:
            self.calls_this_minute = 0
            self.minute_start = now

        if self.calls_this_minute >= API_CALLS_PER_MIN:
            wait = 60 - (now - self.minute_start) + 1
            log.info(f"Throttle: waiting {wait:.0f}s (hit {API_CALLS_PER_MIN} calls/min)")
            time.sleep(wait)
            self.calls_this_minute = 0
            self.minute_start = time.time()

        elapsed = time.time() - self.last_call
        if elapsed < API_CALL_GAP:
            time.sleep(API_CALL_GAP - elapsed)

        self.last_call = time.time()
        self.calls_this_minute += 1

    def call(self, fn, *args, **kwargs):
        """Safe API call with throttle."""
        if not self.api:
            return None
        self._throttle()
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                log.critical(f"RATE LIMITED despite throttle: {e}")
                time.sleep(300)  # 5 min hard wait
            else:
                log.error(f"API error: {e}")
            return None

    def get_quote(self, symbol: str, segment: str = "CASH") -> dict:
        return self.call(self.api.get_quote, trading_symbol=symbol, exchange="NSE", segment=segment) or {}

    def get_margin(self) -> dict:
        return self.call(self.api.get_available_margin_details) or {}

    def buy(self, symbol: str, qty: int, product: str = "MIS", segment: str = "CASH") -> dict:
        if self.paper:
            log.info(f"[PAPER] BUY {qty}x {symbol} ({product})")
            return {"groww_order_id": f"PAPER_{int(time.time())}", "order_status": "PAPER"}

        result = self.call(
            self.api.place_order,
            trading_symbol=symbol, quantity=qty, validity="DAY",
            exchange="NSE", segment=segment, product=product,
            order_type="MARKET", transaction_type="BUY",
        )
        if result:
            log.info(f"BUY ORDER: {qty}x {symbol} -> {result.get('groww_order_id', 'unknown')}")
        return result or {}

    def sell(self, symbol: str, qty: int, product: str = "MIS", segment: str = "CASH") -> dict:
        if self.paper:
            log.info(f"[PAPER] SELL {qty}x {symbol} ({product})")
            return {"groww_order_id": f"PAPER_{int(time.time())}", "order_status": "PAPER"}

        result = self.call(
            self.api.place_order,
            trading_symbol=symbol, quantity=qty, validity="DAY",
            exchange="NSE", segment=segment, product=product,
            order_type="MARKET", transaction_type="SELL",
        )
        if result:
            log.info(f"SELL ORDER: {qty}x {symbol} -> {result.get('groww_order_id', 'unknown')}")
        return result or {}


# ── Fee Calculator ──────────────────────────────────────────────

def estimate_fees(value: float, is_fno: bool = False) -> float:
    """Exact fee calculation. If trade profit < this, DON'T TRADE."""
    if is_fno:
        brokerage = 20 * 2  # flat Rs.20 per order, buy+sell
        stt = value * 0.000625  # STT on options sell
        exchange_txn = value * 0.00053
    else:
        brokerage = min(20, value * 0.0005) * 2
        stt = value * 0.00025
        exchange_txn = value * 0.0000345
    gst = (brokerage + exchange_txn) * 0.18
    stamp = value * 0.00003
    return brokerage + stt + exchange_txn + gst + stamp


# ── Prediction Engine ──────────────────────────────────────────

def predict(symbol: str, ltp: float, prev: float, high: float, low: float) -> dict:
    """Score a stock: predicted return, confidence, strategy."""
    if prev == 0 or high == low or ltp < MIN_STOCK_PRICE:
        return {"score": 0, "return": 0, "strategy": "SKIP"}

    change = ((ltp - prev) / prev) * 100
    rng = ((high - low) / prev) * 100
    pos = (ltp - low) / (high - low) if high > low else 0.5

    score, ret, strat, conf = 0, 0, "NONE", 0

    # Oversold bounce: down big, near day low
    if change < -1.5 and pos < 0.3:
        score = 40 + min(abs(change) * 5, 30) + (1 - pos) * 20
        ret = min(abs(change) * 0.4, 3.0)
        strat, conf = "OVERSOLD", 0.65

    # Momentum: up big, at day high
    elif change > 2.0 and pos > 0.8:
        score = 35 + min(change * 5, 25) + pos * 20
        ret = min(change * 0.3, 2.5)
        strat, conf = "MOMENTUM", 0.60

    # Range breakout
    elif rng > 3.0 and pos > 0.9:
        score = 30 + min(rng * 3, 20)
        ret = min(rng * 0.2, 2.0)
        strat, conf = "BREAKOUT", 0.55

    return {"score": min(score, 100), "return": round(ret, 2), "strategy": strat, "confidence": conf, "change": round(change, 2)}


# ── State Management ───────────────────────────────────────────

STATE_FILE = DATA / "sijoy_state.json"


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"positions": [], "closed": [], "day_pnl": 0, "day_fees": 0, "trades_today": 0}


def save_state(state: dict):
    state["updated"] = datetime.now(IST).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Main Strategy Loop ─────────────────────────────────────────

def run(paper: bool = False):
    api = SmartAPI(paper=paper)

    if not api.connect():
        log.critical("Cannot connect. Waiting 5 min then retrying...")
        time.sleep(300)
        if not api.connect():
            log.critical("Still can't connect. Check token cache or .env")
            return

    state = load_state()
    wallet = CAPITAL

    # Read wallet balance
    margin = api.get_margin()
    if margin:
        for key in ["availableMargin", "available_margin", "net", "cash"]:
            if key in margin:
                wallet = float(margin[key])
                log.info(f"Wallet: Rs.{wallet:,.0f}")
                break

    mode = "PAPER" if paper else "LIVE"
    log.info("=" * 60)
    log.info(f"SIJOY STRATEGY — {mode} MODE")
    log.info(f"Wallet: Rs.{wallet:,.0f}")
    log.info(f"Positions: {len(state['positions'])}")
    log.info(f"Day P&L: Rs.{state['day_pnl']:+,.0f}")
    log.info(f"Rules: profit > 3x fees | fast cycle 30min | max {MAX_TRADES_PER_DAY} trades")
    log.info("=" * 60)

    tg(f"Strategy started ({mode})\nWallet: Rs.{wallet:,.0f}\nPositions: {len(state['positions'])}")

    scan = 0
    while True:
        now = datetime.now(IST)
        scan += 1

        # Reset daily counters at 9:00 AM
        if now.hour == 9 and now.minute < 2:
            state["closed"] = []
            state["day_pnl"] = 0
            state["day_fees"] = 0
            state["trades_today"] = 0
            save_state(state)

        # Market hours
        is_market = (now.weekday() < 5 and
                     (now.hour > 9 or (now.hour == 9 and now.minute >= 15)) and
                     (now.hour < 15 or (now.hour == 15 and now.minute <= 30)))

        if not is_market:
            if now.hour == 0 and now.minute < 2 and state["closed"]:
                summary = f"Day End: {len(state['closed'])} trades | P&L: Rs.{state['day_pnl']:+,.0f} | Fees: Rs.{state['day_fees']:,.0f}"
                log.info(summary)
                tg(summary)
            if scan % 60 == 0:
                fno = [p for p in state["positions"] if p.get("type") == "FNO"]
                if fno:
                    log.info(f"After hours: {len(fno)} F&O positions held")
            time.sleep(60)
            continue

        # ── REFRESH WALLET ──
        if scan % 10 == 1:
            margin = api.get_margin()
            if margin:
                for key in ["availableMargin", "available_margin", "net", "cash"]:
                    if key in margin:
                        wallet = float(margin[key])
                        break

        invested = sum(p.get("cost", 0) for p in state["positions"] if p["status"] == "OPEN")
        available = wallet - invested

        log.info(f"[SCAN {scan}] wallet=Rs.{wallet:,.0f} | invested=Rs.{invested:,.0f} | available=Rs.{available:,.0f} | positions={len(state['positions'])} | day_pnl=Rs.{state['day_pnl']:+,.0f}")

        # ── CHECK EXITS ──
        for pos in state["positions"]:
            if pos["status"] != "OPEN":
                continue

            sym = pos["symbol"]
            is_fno = pos.get("type") == "FNO"
            q = api.get_quote(sym, segment="FNO" if is_fno else "CASH")
            if not q or not q.get("last_price"):
                continue

            ltp = q["last_price"]
            entry = pos["entry_price"]
            pnl_pct = ((ltp - entry) / entry) * 100 if entry > 0 else 0
            gross_pnl = (ltp - entry) * pos["qty"]
            net_pnl = gross_pnl - pos["fees"]

            pos["current"] = ltp
            pos["pnl_pct"] = round(pnl_pct, 2)
            pos["net_pnl"] = round(net_pnl, 2)

            target = TARGET_PCT_FNO if is_fno else TARGET_PCT_EQUITY
            stop = STOP_PCT_FNO if is_fno else STOP_PCT_EQUITY
            entry_time = datetime.fromisoformat(pos["entry_time"])
            held_min = (now - entry_time).total_seconds() / 60

            exit_reason = None

            if pnl_pct >= target:
                exit_reason = "TARGET"
            elif pnl_pct <= stop:
                exit_reason = "STOP"
            elif not is_fno and net_pnl > 0 and held_min > FAST_EXIT_MINUTES:
                exit_reason = "FAST_CYCLE"
            elif not is_fno and pnl_pct < -0.3 and held_min > 60:
                exit_reason = "REDEPLOY"
            elif not is_fno and now.hour == 15 and now.minute >= 10:
                exit_reason = "MIS_CLOSE"
            elif is_fno and pnl_pct >= 15 and held_min < 360:
                exit_reason = "FAST_FNO"

            if exit_reason:
                product = "NRML" if is_fno else "MIS"
                segment = "FNO" if is_fno else "CASH"
                api.sell(sym, pos["qty"], product=product, segment=segment)

                pos["status"] = "CLOSED"
                pos["exit_price"] = ltp
                pos["exit_time"] = now.isoformat()
                pos["exit_reason"] = exit_reason
                state["day_pnl"] += net_pnl
                state["day_fees"] += pos["fees"]
                state["closed"].append(deepcopy(pos))

                win = "WIN" if net_pnl > 0 else "LOSS"
                msg = f"EXIT {win} ({exit_reason})\n{sym}: Rs.{net_pnl:+,.0f} ({pnl_pct:+.1f}%)\nHeld: {held_min:.0f}min | Day: Rs.{state['day_pnl']:+,.0f}"
                log.info(msg.replace("\n", " | "))
                tg(msg)
            else:
                rpm = net_pnl / max(held_min, 1)
                log.info(f"  HOLD: {sym} {pnl_pct:+.1f}% net=Rs.{net_pnl:+,.0f} held={held_min:.0f}min Rs.{rpm:+.1f}/min")

        # Clean closed positions
        state["positions"] = [p for p in state["positions"] if p["status"] == "OPEN"]

        # ── ENTER F&O OPTIONS ──
        if available > 1000 and state["trades_today"] < MAX_TRADES_PER_DAY:
            virtual_file = DATA / "virtual_trades.json"
            if virtual_file.exists():
                try:
                    vdata = json.loads(virtual_file.read_text())
                    signals = [t for t in vdata.get("trades", [])
                               if t.get("status") == "OPEN" and t.get("score", 0) >= 75]
                    for sig in sorted(signals, key=lambda s: s.get("score", 0), reverse=True):
                        sym = sig["option_symbol"]
                        stock = sig.get("stock_symbol", "")

                        # Already holding?
                        if any(p["symbol"] == sym and p["status"] == "OPEN" for p in state["positions"]):
                            continue

                        # Correct lot size
                        lot = LOT_SIZES.get(stock, sig.get("lot_size", 0))
                        premium = sig.get("entry_premium", 0)
                        cost = premium * lot
                        fees = estimate_fees(cost, is_fno=True)

                        if cost > available or cost < MIN_TRADE_VALUE:
                            continue
                        if cost * 0.25 < fees * 3:  # target profit must be 3x fees
                            continue

                        api.buy(sym, lot, product="NRML", segment="FNO")
                        state["positions"].append({
                            "symbol": sym, "type": "FNO", "qty": lot,
                            "entry_price": premium, "cost": cost, "fees": round(fees, 2),
                            "entry_time": now.isoformat(), "status": "OPEN",
                        })
                        state["trades_today"] += 1
                        available -= cost
                        log.info(f"FNO BUY: {sym} premium={premium} x {lot} = Rs.{cost:,.0f} fees=Rs.{fees:.0f}")
                        tg(f"BUY {sym}\nRs.{premium} x {lot} = Rs.{cost:,.0f}\nTarget: +25% | Stop: -50%")
                except Exception as e:
                    log.error(f"F&O signals error: {e}")

        # ── ENTER EQUITY ──
        open_equity = len([p for p in state["positions"] if p.get("type") != "FNO" and p["status"] == "OPEN"])
        if available > MIN_TRADE_VALUE and open_equity < 3 and state["trades_today"] < MAX_TRADES_PER_DAY and now.hour < 14:
            best_signal = None
            best_score = 0

            for sym in EQUITY_UNIVERSE:
                q = api.get_quote(f"{sym}-EQ")
                if not q or not q.get("last_price"):
                    continue

                ltp = q["last_price"]
                prev = q.get("ohlc", {}).get("close", 0)
                high = q.get("ohlc", {}).get("high", ltp)
                low = q.get("ohlc", {}).get("low", ltp)

                pred = predict(sym, ltp, prev, high, low)
                if pred["score"] < 45 or pred["return"] < 0.5:
                    continue

                qty = int(min(available * 0.5, available) / ltp)
                if qty == 0:
                    continue

                value = ltp * qty
                fees = estimate_fees(value)
                expected_profit = value * (pred["return"] / 100)

                if expected_profit < fees * 3:
                    continue

                if pred["score"] > best_score:
                    best_score = pred["score"]
                    best_signal = {
                        "symbol": sym, "ltp": ltp, "qty": qty, "value": value,
                        "fees": fees, "pred": pred,
                    }

            if best_signal:
                s = best_signal
                sym = s["symbol"]
                # Check not already holding
                if not any(p["symbol"] == sym and p["status"] == "OPEN" for p in state["positions"]):
                    api.buy(f"{sym}-EQ", s["qty"], product="MIS", segment="CASH")
                    state["positions"].append({
                        "symbol": sym, "type": "EQUITY", "qty": s["qty"],
                        "entry_price": s["ltp"], "cost": s["value"], "fees": round(s["fees"], 2),
                        "entry_time": now.isoformat(), "status": "OPEN",
                        "strategy": s["pred"]["strategy"], "predicted_return": s["pred"]["return"],
                    })
                    state["trades_today"] += 1
                    available -= s["value"]
                    log.info(f"EQUITY BUY: {s['qty']}x {sym} @ Rs.{s['ltp']:.2f} | {s['pred']['strategy']} | predicted +{s['pred']['return']}% | fees=Rs.{s['fees']:.0f}")
                    tg(f"BUY {s['qty']}x {sym} @ Rs.{s['ltp']:,.2f}\n{s['pred']['strategy']} | +{s['pred']['return']}%\nFees: Rs.{s['fees']:.0f}")

        save_state(state)
        time.sleep(30)  # scan every 30 seconds


# ── Entry Point ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true", help="Paper trading mode")
    parser.add_argument("--status", action="store_true", help="Show current positions")
    args = parser.parse_args()

    if args.status:
        state = load_state()
        print(json.dumps(state, indent=2))
    else:
        run(paper=args.paper)

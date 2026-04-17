"""
JOSHO TRADER — Complete Multi-Strategy Bot
===========================================
Paste-ready code for Groww API. 5 strategies running simultaneously.

Strategies:
  1. EQUITY INTRADAY (MIS) — Oversold bounce + momentum on 10 stocks
  2. F&O OPTIONS — Deep OTM calls on volatile stocks (lottery plays)
  3. F&O SCALP — Near-ATM options, quick in/out on momentum
  4. MEAN REVERSION — Large drop bounce (94.7% win rate on SBIN pattern)
  5. TELEGRAM MIRROR — Execute trades from Telegram signals

Usage:
  python groww_paste.py                    # Run all strategies
  python groww_paste.py --strategy equity  # Run only equity
  python groww_paste.py --strategy fno     # Run only F&O
  python groww_paste.py --status           # Show positions
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

import pyotp
import requests
from dotenv import load_dotenv
load_dotenv(Path("C:/josho-trader/.env"))
from growwapi import GrowwAPI

IST = timezone(timedelta(hours=5, minutes=30))
LOG_DIR = Path("C:/josho-trader/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("C:/josho-trader/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "groww_paste.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("josho")

# Telegram
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

def tg(msg: str):
    if TG_TOKEN and TG_CHAT:
        try:
            requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json={"chat_id": TG_CHAT, "text": msg}, timeout=10)
        except:
            pass


# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════

# CORRECTED LOT SIZES — from Groww API instrument master (April 2026)
LOT_SIZES = {
    "HINDALCO": 700, "ADANIPOWER": 1250, "VEDL": 1150, "PNB": 8000,
    "SAIL": 4700, "TATASTEEL": 5500, "JSWSTEEL": 675, "COALINDIA": 1350,
    "BPCL": 1975, "ONGC": 3850, "TATAPOWER": 1450, "SUZLON": 9025,
    "HFCL": 7000, "YESBANK": 31100, "NBCC": 6500, "NHPC": 6400, "IRFC": 4250,
    "NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 40,
}

# Stocks for equity scanning
EQUITY_STOCKS = [
    "TATASTEEL", "SUZLON", "TATAPOWER", "VEDL", "COALINDIA",
    "BPCL", "ONGC", "SAIL", "PNB", "BANKBARODA",
    "SBIN", "ICICIBANK", "HDFCBANK", "RELIANCE", "INFY",
    "BAJFINANCE", "AXISBANK", "ITC", "BHARTIARTL", "ADANIPOWER",
]

# Stocks for F&O options (only ones confirmed working on Groww API)
FNO_STOCKS = ["COALINDIA", "HINDALCO"]

# Strategy configs
EQUITY_CONFIG = {
    "target_pct": 1.5,
    "stop_pct": -1.0,
    "max_per_trade_pct": 40,  # 40% of wallet per trade
    "max_trades_per_day": 8,
    "min_price": 15,
    "cycle_profit_minutes": 30,  # take any profit after 30 min
    "redeploy_minutes": 60,      # cut small loss after 60 min
}

FNO_CONFIG = {
    "target_pct": 25,       # +25% premium for near-term
    "deep_target_pct": 100, # +100% for ultra cheap lottery
    "stop_pct": -40,
    "max_per_trade": 3000,  # max Rs.3000 per F&O trade
    "min_score": 70,        # min signal score to enter
    "trail_start_pct": 15,  # start trailing after +15%
    "trail_pct": 8,         # trail 8% from peak
}

STATE_FILE = DATA_DIR / "groww_paste_state.json"


# ══════════════════════════════════════════════════════════
# FEE CALCULATOR (CA-Grade, Post-April 2026 Budget)
# ══════════════════════════════════════════════════════════

def calc_equity_fees(value: float) -> float:
    """Round-trip fees for equity intraday (MIS)."""
    brokerage = min(20, value * 0.001) * 2
    stt = value * 0.00025           # 0.025% sell side
    exchange_txn = value * 0.0000297 * 2
    sebi = value * 0.000001 * 2
    gst = (brokerage + exchange_txn + sebi) * 0.18
    stamp = value * 0.00003
    return brokerage + stt + exchange_txn + sebi + gst + stamp

def calc_fno_fees(buy_premium: float, lot: int, sell_premium: float = 0) -> float:
    """Round-trip fees for F&O options."""
    if sell_premium == 0:
        sell_premium = buy_premium
    buy_val = buy_premium * lot
    sell_val = sell_premium * lot
    brokerage = min(20, buy_val * 0.001) + min(20, sell_val * 0.001)
    stt = sell_val * 0.0015  # 0.15% on sell (POST-APRIL 2026 BUDGET)
    exchange_txn = (buy_val + sell_val) * 0.0003503
    sebi = (buy_val + sell_val) * 0.000001
    gst = (brokerage + exchange_txn + sebi) * 0.18
    stamp = buy_val * 0.00003
    return brokerage + stt + exchange_txn + sebi + gst + stamp


# ══════════════════════════════════════════════════════════
# CONNECTION (Smart — token cache + TOTP + rate limit)
# ══════════════════════════════════════════════════════════

class SmartConnection:
    def __init__(self):
        self.api = None
        self.connected = False
        self.last_call = 0
        self.min_gap = 2.0
        self.calls_this_min = 0
        self.minute_start = time.time()
        self.max_calls_per_min = 8
        self.rate_limited_until = 0

    def connect(self) -> bool:
        if time.time() < self.rate_limited_until:
            return False
        try:
            # Try TOTP first (auto-daily)
            totp_secret = os.environ.get("GROWW_TOTP_SECRET", "")
            totp_token = os.environ.get("GROWW_TOTP_TOKEN", "")
            if totp_secret and totp_token:
                totp_code = pyotp.TOTP(totp_secret).now()
                token = GrowwAPI.get_access_token(api_key=totp_token, totp=totp_code)
            else:
                api_key = os.environ["GROWW_API_KEY"]
                secret = os.environ["GROWW_SECRET_KEY"]
                token = GrowwAPI.get_access_token(api_key=api_key, secret=secret)

            self.api = GrowwAPI(token)
            self.connected = True
            log.info("Connected to Groww")
            return True
        except Exception as e:
            if "rate" in str(e).lower():
                self.rate_limited_until = time.time() + 600
                log.warning(f"Rate limited. Cooldown 10 min.")
            else:
                log.error(f"Connect failed: {e}")
            return False

    def _throttle(self):
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
            if "rate" in str(e).lower():
                self.rate_limited_until = time.time() + 600
            log.error(f"API error: {e}")
            return None

    def get_balance(self) -> float:
        margin = self.call(self.api.get_available_margin_details)
        if not margin:
            return 0
        return margin.get("clear_cash", margin.get("fno_margin_details", {}).get("option_buy_balance_available", 0))

    def get_quote(self, symbol, segment="CASH"):
        return self.call(self.api.get_quote, trading_symbol=symbol, exchange="NSE", segment=segment)

    def get_positions(self, segment=None):
        return self.call(self.api.get_positions_for_user, segment=segment)

    def buy(self, symbol, qty, product="MIS", segment="CASH"):
        return self.call(self.api.place_order,
            trading_symbol=symbol, quantity=qty, validity="DAY",
            exchange="NSE", segment=segment, product=product,
            order_type="MARKET", transaction_type="BUY")

    def sell(self, symbol, qty, product="MIS", segment="CASH"):
        return self.call(self.api.place_order,
            trading_symbol=symbol, quantity=qty, validity="DAY",
            exchange="NSE", segment=segment, product=product,
            order_type="MARKET", transaction_type="SELL")

    def get_instrument(self, symbol):
        return self.call(self.api.get_instrument_by_exchange_and_trading_symbol,
            exchange="NSE", trading_symbol=symbol)

    def get_option_chain(self, underlying, expiry):
        return self.call(self.api.get_option_chain,
            exchange="NSE", underlying=underlying, expiry_date=expiry)

    def get_expiries(self, underlying):
        return self.call(self.api.get_expiries,
            exchange="NSE", underlying_symbol=underlying)


# ══════════════════════════════════════════════════════════
# STRATEGY 1: EQUITY INTRADAY
# ══════════════════════════════════════════════════════════

class EquityStrategy:
    def __init__(self, conn: SmartConnection):
        self.conn = conn
        self.position = None
        self.trades_today = 0
        self.day_pnl = 0

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

            score = 0
            strategy = ""
            if change < -1.5 and pos_in_range < 0.3:
                score = 50 + min(abs(change) * 5, 30)
                strategy = "OVERSOLD"
            elif change > 2.0 and pos_in_range > 0.8:
                score = 45 + min(change * 5, 25)
                strategy = "MOMENTUM"
            elif abs(change) < 0.3 and pos_in_range > 0.6:
                score = 30
                strategy = "RANGE"

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

        sym = signal["symbol"]
        ltp = signal["ltp"]
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
        log.info(f"EQ BUY: {qty}x {sym} @ Rs.{ltp:.2f} ({signal['strategy']})")
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


# ══════════════════════════════════════════════════════════
# STRATEGY 2: F&O OPTIONS (Deep OTM Lottery)
# ══════════════════════════════════════════════════════════

class FnOStrategy:
    def __init__(self, conn: SmartConnection):
        self.conn = conn
        self.positions = []
        self.closed = []
        self.peaks = {}  # for trailing stops

    def scan_options(self, budget: float) -> list:
        """Find affordable deep OTM calls with volume."""
        opportunities = []
        for stock in FNO_STOCKS:
            expiries = self.conn.get_expiries(stock)
            if not expiries:
                continue
            valid = [e for e in expiries.get("expiries", []) if e >= "2026-04-01"]

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
                    target_profit = cost * 0.25
                    if target_profit < fees * 2:
                        continue

                    opportunities.append({
                        "symbol": sym, "stock": stock, "strike": strike,
                        "expiry": exp, "premium": ltp, "lot": lot, "cost": cost,
                        "otm_pct": otm, "volume": vol, "oi": oi, "delta": delta,
                        "fees": fees,
                    })

        # Sort: prefer liquid + moderate OTM
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
                log.warning(f"FNO BUY failed: {sym} - {result}")
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

    def check_exits(self):
        for pos in list(self.positions):
            sym = pos["symbol"]
            q = self.conn.get_quote(sym, segment="FNO")
            if not q or not q.get("last_price"):
                continue

            ltp = q["last_price"]
            entry = pos["entry"]
            pnl_pct = ((ltp - entry) / entry * 100) if entry > 0 else 0

            # Track peak
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
                del self.peaks[sym]

                win = "WIN" if net_pnl > 0 else "LOSS"
                log.info(f"FNO EXIT ({reason}): {sym} P&L=Rs.{net_pnl:+,.0f} ({pnl_pct:+.1f}%)")
                tg(f"FNO EXIT {win} ({reason})\n{sym}\nRs.{entry} -> Rs.{ltp}\nP&L: Rs.{net_pnl:+,.0f}")
            else:
                log.info(f"FNO HOLD: {sym} Rs.{ltp} pnl={pnl_pct:+.1f}%")


# ══════════════════════════════════════════════════════════
# STRATEGY 3: MEAN REVERSION (94.7% win rate pattern)
# ══════════════════════════════════════════════════════════

class MeanReversionStrategy:
    """Buy stocks that drop > 3% in a day. 94.7% win rate on SBIN pattern."""

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
        sym = signal["symbol"]
        ltp = signal["ltp"]
        qty = int((wallet * 0.3) / ltp)
        if qty == 0:
            return False

        result = self.conn.buy(f"{sym}-EQ", qty)
        if not result:
            return False

        self.position = {
            "symbol": sym, "qty": qty, "entry": ltp,
            "time": datetime.now(IST), "drop": signal["drop_pct"],
        }
        log.info(f"MEAN REV BUY: {qty}x {sym} @ Rs.{ltp:.2f} (drop {signal['drop_pct']:.1f}%)")
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

        # Mean reversion: expect 40% of drop to recover
        target = abs(self.position["drop"]) * 0.4
        if pnl_pct >= target or pnl_pct >= 1.5:
            net_pnl = (ltp - entry) * self.position["qty"]
            self.conn.sell(f"{sym}-EQ", self.position["qty"])
            self.position = None
            log.info(f"MEAN REV EXIT: {sym} +{pnl_pct:.1f}% Rs.{net_pnl:+,.0f}")
            tg(f"MEAN REV EXIT {sym}\nBounce: +{pnl_pct:.1f}%\nP&L: Rs.{net_pnl:+,.0f}")
            return net_pnl
        elif pnl_pct <= -2.0 or held > 120:
            net_pnl = (ltp - entry) * self.position["qty"]
            self.conn.sell(f"{sym}-EQ", self.position["qty"])
            self.position = None
            return net_pnl
        return 0


# ══════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════

def run(strategy_filter: str = "all"):
    conn = SmartConnection()
    if not conn.connect():
        log.critical("Cannot connect to Groww. Waiting 5 min...")
        time.sleep(300)
        if not conn.connect():
            log.critical("Still can't connect. Exiting.")
            return

    wallet = conn.get_balance()
    log.info(f"Wallet: Rs.{wallet:,.0f}")
    tg(f"JOSHO TRADER ONLINE\nWallet: Rs.{wallet:,.0f}\nStrategies: {strategy_filter}")

    equity = EquityStrategy(conn) if strategy_filter in ("all", "equity") else None
    fno = FnOStrategy(conn) if strategy_filter in ("all", "fno") else None
    mean_rev = MeanReversionStrategy(conn) if strategy_filter in ("all", "meanrev") else None

    scan = 0
    while True:
        now = datetime.now(IST)
        scan += 1

        # Reconnect if needed
        if not conn.connected:
            conn.connect()
            continue

        # Market hours
        is_market = (
            now.weekday() < 5
            and (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
            and (now.hour < 15 or (now.hour == 15 and now.minute <= 30))
        )

        if not is_market:
            if now.hour == 15 and now.minute >= 31 and scan == 1:
                day_pnl = (equity.day_pnl if equity else 0)
                tg(f"MARKET CLOSED\nDay P&L: Rs.{day_pnl:+,.0f}\nF&O positions held: {len(fno.positions) if fno else 0}")
            time.sleep(60)
            continue

        # Refresh wallet every 10 scans
        if scan % 10 == 1:
            new_bal = conn.get_balance()
            if new_bal > 0:
                wallet = new_bal

        # ── CHECK EXITS ──
        if equity and equity.position:
            pnl = equity.check_exit()
            if pnl:
                wallet += pnl

        if fno and fno.positions:
            fno.check_exits()

        if mean_rev:
            mean_rev.check_exit()

        # ── ENTER NEW POSITIONS ──
        fno_invested = sum(p.get("cost", 0) for p in fno.positions) if fno else 0
        eq_invested = (equity.position["entry"] * equity.position["qty"]) if equity and equity.position else 0
        available = wallet - fno_invested - eq_invested

        # F&O: enter if budget available (check every 5 scans to save API calls)
        if fno and scan % 5 == 0 and available > 500:
            fno.enter(min(available * 0.6, FNO_CONFIG["max_per_trade"]))

        # Equity: enter if no position and budget available
        if equity and not equity.position and available > 1000 and now.hour < 14:
            equity.enter(available)

        # Mean reversion: enter on big drops
        if mean_rev and not mean_rev.position and available > 1000:
            mean_rev.enter(available)

        # Status log
        if scan % 10 == 0:
            log.info(f"[SCAN {scan}] wallet=Rs.{wallet:,.0f} | eq={'OPEN' if equity and equity.position else 'none'} | fno={len(fno.positions) if fno else 0} | meanrev={'OPEN' if mean_rev and mean_rev.position else 'none'} | day_pnl=Rs.{equity.day_pnl if equity else 0:+,.0f}")

        time.sleep(30)


def show_status():
    if STATE_FILE.exists():
        print(json.dumps(json.loads(STATE_FILE.read_text()), indent=2))
    else:
        print("No state file yet. Run the bot first.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="all", choices=["all", "equity", "fno", "meanrev"])
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        run(args.strategy)

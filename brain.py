"""
brain.py — 24/7 Trading Brain
==============================
Single unified process that manages ALL trading:
- F&O options (NRML, hold overnight)
- Equity intraday (MIS, auto-close 3:15 PM)
- Wallet management (always fetch real balance, never idle money)
- Smart API rate limiting (never burn rate limit tokens)

Runs 24/7. During market hours: actively trade.
After hours: monitor positions, plan next day.

Usage:
  python brain.py              # Start the brain
  python brain.py --status     # Show all positions
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
from dotenv import load_dotenv
load_dotenv(Path("C:/josho-trader/.env"))

IST = timezone(timedelta(hours=5, minutes=30))
LOG_DIR = Path("C:/josho-trader/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("C:/josho-trader/data")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "brain.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("brain")

# Telegram alerts
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

def tg(msg: str):
    if not TG_TOKEN or not TG_CHAT: return
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": f"🧠 *BRAIN*\n\n{msg}", "parse_mode": "Markdown"}, timeout=10)
    except: pass

STATE_FILE = DATA_DIR / "brain_state.json"
VIRTUAL_FILE = DATA_DIR / "virtual_trades.json"

# F&O lot sizes (Apr 2026)
LOT_SIZES = {
    "HINDALCO": 550, "ADANIPOWER": 1250, "VEDL": 1500, "PNB": 4000,
    "SAIL": 4000, "TATASTEEL": 1500, "JSWSTEEL": 500, "COALINDIA": 1500,
    "BPCL": 1100, "ONGC": 3850, "TATAPOWER": 1350, "SUZLON": 4000,
    "HFCL": 7000, "YESBANK": 5000, "NBCC": 7000, "NHPC": 7000, "IRFC": 5000,
}

# ── Groww Connection (with smart rate limiting) ─────────────

class SmartGroww:
    """Groww API with built-in rate limit awareness."""

    def __init__(self):
        self.client = None
        self.connected = False
        self.last_api_call = 0
        self.min_call_gap = 2.0  # minimum 2 seconds between API calls
        self.rate_limited_until = 0
        self.api_calls_this_minute = 0
        self.minute_start = time.time()
        self.MAX_CALLS_PER_MINUTE = 8  # Groww allows ~10, we stay at 8

    def connect(self) -> bool:
        if time.time() < self.rate_limited_until:
            remaining = int(self.rate_limited_until - time.time())
            log.info(f"Rate limit cooldown: {remaining}s remaining")
            return False
        try:
            sys.path.insert(0, "C:/josho-trader/src")
            from client import GrowwClient
            self.client = GrowwClient()
            self.client.connect()
            self.connected = True
            log.info("Groww connected")
            return True
        except Exception as e:
            if "rate limit" in str(e).lower():
                self.rate_limited_until = time.time() + 600  # 10 min cooldown
                log.warning(f"Rate limited. Cooldown until {datetime.fromtimestamp(self.rate_limited_until, IST).strftime('%H:%M')}")
            else:
                log.error(f"Connect failed: {e}")
            self.connected = False
            return False

    def _throttle(self):
        """Smart throttle — respect rate limits."""
        # Reset minute counter
        if time.time() - self.minute_start > 60:
            self.api_calls_this_minute = 0
            self.minute_start = time.time()

        # Don't exceed calls per minute
        if self.api_calls_this_minute >= self.MAX_CALLS_PER_MINUTE:
            wait = 60 - (time.time() - self.minute_start)
            if wait > 0:
                log.info(f"Throttle: {self.api_calls_this_minute} calls this minute, waiting {wait:.0f}s")
                time.sleep(wait)
                self.api_calls_this_minute = 0
                self.minute_start = time.time()

        # Minimum gap between calls
        elapsed = time.time() - self.last_api_call
        if elapsed < self.min_call_gap:
            time.sleep(self.min_call_gap - elapsed)

        self.last_api_call = time.time()
        self.api_calls_this_minute += 1

    def _safe_call(self, fn, *args, **kwargs):
        """Wrap any API call with throttle + error handling."""
        if not self.connected:
            return None
        if time.time() < self.rate_limited_until:
            return None
        self._throttle()
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as e:
            if "rate limit" in str(e).lower():
                self.rate_limited_until = time.time() + 600
                log.warning(f"Rate limited on API call. Cooling down 10 min.")
            else:
                log.error(f"API call failed: {e}")
            return None

    def get_balance(self) -> float:
        margin = self._safe_call(self.client.get_margin)
        if not margin or not isinstance(margin, dict):
            return -1
        for key in ['availableMargin', 'available_margin', 'net', 'cash', 'availableBalance']:
            if key in margin: return float(margin[key])
        if 'data' in margin:
            for key in ['availableMargin', 'net', 'cash']:
                if key in margin.get('data', {}): return float(margin['data'][key])
        log.info(f"Margin keys: {list(margin.keys())}")
        return -1

    def get_quote(self, symbol, exchange="NSE", segment="CASH"):
        return self._safe_call(self.client.get_quote, symbol, exchange=exchange, segment=segment)

    def get_positions(self):
        return self._safe_call(self.client.get_positions)

    def buy(self, symbol, qty, product="MIS", segment="CASH"):
        result = self._safe_call(self.client.place_order,
            symbol=symbol, qty=qty, side="BUY",
            product=product, segment=segment, exchange="NSE")
        if result:
            log.info(f"BUY: {qty}x {symbol} ({product}) -> {result}")
        return result

    def sell(self, symbol, qty, product="MIS", segment="CASH"):
        result = self._safe_call(self.client.place_order,
            symbol=symbol, qty=qty, side="SELL",
            product=product, segment=segment, exchange="NSE")
        if result:
            log.info(f"SELL: {qty}x {symbol} ({product}) -> {result}")
        return result


# ── Trading Brain ───────────────────────────────────────────

class TradingBrain:
    def __init__(self):
        self.api = SmartGroww()
        self.wallet = 0
        self.positions = []  # our tracked positions
        self.closed_today = []
        self.day_pnl = 0
        self.day_fees = 0
        self.scan_count = 0
        self._load_state()

    def _load_state(self):
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                self.positions = data.get("positions", [])
                self.closed_today = data.get("closed_today", [])
                self.day_pnl = data.get("day_pnl", 0)
                self.day_fees = data.get("day_fees", 0)
                log.info(f"Loaded state: {len(self.positions)} positions, day P&L Rs.{self.day_pnl:+,.0f}")
            except: pass

    def _save_state(self):
        STATE_FILE.write_text(json.dumps({
            "positions": self.positions,
            "closed_today": self.closed_today,
            "day_pnl": round(self.day_pnl, 2),
            "day_fees": round(self.day_fees, 2),
            "wallet": self.wallet,
            "last_updated": datetime.now(IST).isoformat(),
        }, indent=2))

    def get_lot_size(self, symbol: str) -> int:
        for name, lot in LOT_SIZES.items():
            if symbol.startswith(name) or name in symbol:
                return lot
        return 0

    def estimate_fees(self, value: float, is_fno: bool = False) -> float:
        """Estimate total transaction fees."""
        if is_fno:
            brokerage = 20 * 2
            stt = value * 0.000625
            exchange_txn = value * 0.00053
        else:
            brokerage = min(20, value * 0.0005) * 2
            stt = value * 0.00025
            exchange_txn = value * 0.0000345
        gst = (brokerage + exchange_txn) * 0.18
        stamp = value * 0.00003
        return brokerage + stt + exchange_txn + gst + stamp

    def invested_amount(self) -> float:
        return sum(p.get("cost", 0) for p in self.positions if p.get("status") == "OPEN")

    def available(self) -> float:
        return max(self.wallet - self.invested_amount(), 0)

    # ── Options Signals ─────────────────────────────────

    def get_option_signals(self) -> list:
        """Read from options predictor."""
        if not VIRTUAL_FILE.exists(): return []
        try:
            data = json.loads(VIRTUAL_FILE.read_text())
            return [t for t in data.get("trades", [])
                    if t.get("status") == "OPEN" and t.get("score", 0) >= 75]
        except: return []

    def enter_option(self, signal: dict):
        """Enter an F&O position."""
        sym = signal["option_symbol"]
        stock = signal.get("stock_symbol", "")
        lot = self.get_lot_size(sym)
        if lot == 0:
            lot = signal.get("lot_size", 0)

        premium = signal.get("entry_premium", 0)
        cost = premium * lot
        fees = self.estimate_fees(cost, is_fno=True)

        # RULE: profit at 25% target must be > 2x fees
        target_profit = cost * 0.25
        if target_profit < fees * 2:
            log.info(f"SKIP {sym}: target profit Rs.{target_profit:.0f} < 2x fees Rs.{fees*2:.0f}")
            return

        if cost > self.available():
            log.info(f"SKIP {sym}: cost Rs.{cost:.0f} > available Rs.{self.available():.0f}")
            return

        # Already holding?
        if any(p.get("symbol") == sym and p.get("status") == "OPEN" for p in self.positions):
            return

        log.info(f"ENTERING: {sym} | premium={premium} | lot={lot} | cost=Rs.{cost:.0f} | fees=Rs.{fees:.0f} | target_profit=Rs.{target_profit:.0f}")

        result = self.api.buy(sym, lot, product="NRML", segment="FNO")

        self.positions.append({
            "symbol": sym, "stock": stock, "type": "FNO",
            "qty": lot, "entry_price": premium, "cost": cost,
            "fees": round(fees, 2),
            "entry_time": datetime.now(IST).isoformat(),
            "target_pct": 25, "stop_pct": -50,
            "status": "OPEN",
            "order": result,
        })
        self._save_state()
        tg(f"BUY {sym}\nPremium: Rs.{premium} x {lot} = Rs.{cost:,.0f}\nTarget: +25% | Stop: -50%")

    # ── Prediction Engine ─────────────────────────────────

    def predict_return(self, symbol: str, ltp: float, prev: float, high: float, low: float, volume: int = 0) -> dict:
        """Score a trade: predicted return %, confidence, risk-reward ratio.
        Uses momentum, mean-reversion, volatility, and volume analysis."""

        if prev == 0 or high == low:
            return {"score": 0, "predicted_return": 0, "confidence": 0, "strategy": "SKIP"}

        change_pct = ((ltp - prev) / prev) * 100
        day_range = ((high - low) / prev) * 100
        pos_in_range = (ltp - low) / (high - low) if (high - low) > 0 else 0.5

        score = 0
        strategy = ""
        predicted_return = 0
        confidence = 0

        # ── OVERSOLD BOUNCE (mean reversion) ──
        if change_pct < -1.5 and pos_in_range < 0.3:
            # Stock down big, near day low = likely bounce
            score += 40
            score += min(abs(change_pct) * 5, 30)  # bigger drop = stronger bounce
            score += (1 - pos_in_range) * 20  # closer to low = better entry
            predicted_return = min(abs(change_pct) * 0.4, 3.0)  # expect 40% of drop to recover
            confidence = 0.65
            strategy = "OVERSOLD_BOUNCE"

        # ── MOMENTUM BREAKOUT ──
        elif change_pct > 2.0 and pos_in_range > 0.8:
            # Stock up big, at day high = momentum continuation
            score += 35
            score += min(change_pct * 5, 25)
            score += pos_in_range * 20
            predicted_return = min(change_pct * 0.3, 2.5)  # expect 30% more of move
            confidence = 0.60
            strategy = "MOMENTUM"

        # ── RANGE BREAKOUT ──
        elif day_range > 3.0 and pos_in_range > 0.9:
            # Wide range day, breaking out
            score += 30
            score += min(day_range * 3, 20)
            predicted_return = min(day_range * 0.2, 2.0)
            confidence = 0.55
            strategy = "RANGE_BREAK"

        # ── VOLUME SPIKE (if data available) ──
        # High volume confirms the move
        if volume > 0:
            score += 10  # bonus for having volume data

        return {
            "score": min(score, 100),
            "predicted_return": round(predicted_return, 2),
            "confidence": confidence,
            "strategy": strategy,
            "change_pct": round(change_pct, 2),
            "day_range": round(day_range, 2),
            "pos_in_range": round(pos_in_range, 2),
        }

    # ── Equity Signals ──────────────────────────────────

    def scan_equity(self):
        """Scan stocks, predict returns, only take trades where predicted profit > 3x fees."""
        STOCKS = [
            "TATASTEEL", "SUZLON", "TATAPOWER", "VEDL", "COALINDIA",
            "BPCL", "ONGC", "SAIL", "PNB", "BANKBARODA",
            "SBIN", "ICICIBANK", "HDFCBANK", "RELIANCE", "INFY",
            "BAJFINANCE", "AXISBANK", "KOTAKBANK", "ITC", "BHARTIARTL",
            "ADANIPOWER", "NHPC", "IRFC", "NBCC", "YESBANK",
        ]

        signals = []
        for sym in STOCKS:
            q = self.api.get_quote(f"{sym}-EQ")
            if not q or not q.get("last_price"): continue

            ltp = q["last_price"]
            prev = q.get("ohlc", {}).get("close", 0)
            high = q.get("ohlc", {}).get("high", ltp)
            low = q.get("ohlc", {}).get("low", ltp)
            volume = q.get("volume", 0)

            if prev == 0 or ltp < 15: continue

            # Predict return
            pred = self.predict_return(sym, ltp, prev, high, low, volume)
            if pred["score"] < 40 or pred["predicted_return"] < 0.5:
                continue  # skip weak signals

            # Position sizing
            avail = self.available()
            max_per_trade = avail * 0.5  # max 50% per trade
            qty = int(max_per_trade / ltp)
            if qty == 0: continue

            value = ltp * qty
            fees = self.estimate_fees(value)
            expected_profit = value * (pred["predicted_return"] / 100)

            # STRICT: predicted profit must be > 3x fees
            if expected_profit < fees * 3:
                log.info(f"  SKIP {sym}: predicted profit Rs.{expected_profit:.0f} < 3x fees Rs.{fees*3:.0f}")
                continue

            net_return_pct = ((expected_profit - fees) / value) * 100

            signals.append({
                "symbol": sym, "ltp": ltp, "qty": qty,
                "change_pct": pred["change_pct"],
                "fees": fees,
                "expected_profit": expected_profit,
                "predicted_return": pred["predicted_return"],
                "net_return_pct": net_return_pct,
                "score": pred["score"],
                "strategy": pred["strategy"],
                "confidence": pred["confidence"],
            })

            log.info(f"  SIGNAL: {sym} @ Rs.{ltp:.2f} | {pred['strategy']} | predicted +{pred['predicted_return']}% | score={pred['score']} | net_return={net_return_pct:.2f}%")

        # Sort by score, take best
        signals.sort(key=lambda s: s["score"], reverse=True)
        return signals[:2]  # Max 2 equity trades at a time
                expected_profit = value * 0.01

                if expected_profit > fees * 2:
                    signals.append({"symbol": sym, "ltp": ltp, "qty": qty,
                                    "change_pct": change_pct, "fees": fees,
                                    "expected_profit": expected_profit})

        return signals[:3]  # Max 3 signals per scan

    def enter_equity(self, signal: dict):
        """Enter equity MIS position."""
        sym = signal["symbol"]
        qty = signal["qty"]
        ltp = signal["ltp"]
        cost = ltp * qty
        fees = signal["fees"]

        if any(p.get("symbol") == sym and p.get("status") == "OPEN" for p in self.positions):
            return

        result = self.api.buy(f"{sym}-EQ", qty, product="MIS", segment="CASH")

        self.positions.append({
            "symbol": sym, "type": "EQUITY_MIS",
            "qty": qty, "entry_price": ltp, "cost": cost,
            "fees": round(fees, 2),
            "entry_time": datetime.now(IST).isoformat(),
            "target_pct": 1.5, "stop_pct": -1.0,
            "status": "OPEN",
            "order": result,
        })
        self._save_state()
        pred_ret = signal.get("predicted_return", "?")
        strat = signal.get("strategy", "?")
        net_ret = signal.get("net_return_pct", 0)
        log.info(f"EQUITY BUY: {qty}x {sym} @ Rs.{ltp} = Rs.{cost:.0f} | {strat} | predicted +{pred_ret}% | net +{net_ret:.2f}% | fees=Rs.{fees:.0f}")
        tg(f"BUY {qty}x {sym} @ Rs.{ltp:,.2f}\n{strat} | Predicted: +{pred_ret}%\nValue: Rs.{cost:,.0f} | Fees: Rs.{fees:.0f}\nNet return: +{net_ret:.2f}%")

    # ── Exit Management ─────────────────────────────────

    def check_exits(self):
        """Check all open positions for exit signals."""
        for pos in self.positions:
            if pos["status"] != "OPEN": continue

            sym = pos["symbol"]
            is_fno = pos["type"] == "FNO"

            if is_fno:
                q = self.api.get_quote(sym, segment="FNO")
            else:
                q = self.api.get_quote(f"{sym}-EQ")

            if not q or not q.get("last_price"): continue

            ltp = q["last_price"]
            entry = pos["entry_price"]
            pnl_pct = ((ltp - entry) / entry) * 100 if entry > 0 else 0
            gross_pnl = (ltp - entry) * pos["qty"]
            net_pnl = gross_pnl - pos["fees"]

            pos["current_price"] = ltp
            pos["pnl_pct"] = round(pnl_pct, 2)
            pos["net_pnl"] = round(net_pnl, 2)

            target = pos.get("target_pct", 25 if is_fno else 1.5)
            stop = pos.get("stop_pct", -50 if is_fno else -1.0)

            # EXIT: Target hit
            if pnl_pct >= target:
                self._exit(pos, "TARGET", ltp, net_pnl)
                continue

            # EXIT: Stop loss
            if pnl_pct <= stop:
                self._exit(pos, "STOP", ltp, net_pnl)
                continue

            # EXIT: MIS equity before 3:15 PM
            if not is_fno:
                now = datetime.now(IST)
                if now.hour == 15 and now.minute >= 10:
                    self._exit(pos, "MIS_CLOSE", ltp, net_pnl)
                    continue

            log.info(f"HOLD: {sym} | {pnl_pct:+.1f}% | net=Rs.{net_pnl:+,.0f} | target=+{target}%")

        self._save_state()

    def _exit(self, pos: dict, reason: str, exit_price: float, net_pnl: float):
        sym = pos["symbol"]
        is_fno = pos["type"] == "FNO"

        if is_fno:
            self.api.sell(sym, pos["qty"], product="NRML", segment="FNO")
        else:
            self.api.sell(f"{sym}-EQ", pos["qty"], product="MIS", segment="CASH")

        pos["status"] = "CLOSED"
        pos["exit_price"] = exit_price
        pos["exit_time"] = datetime.now(IST).isoformat()
        pos["exit_reason"] = reason
        pos["net_pnl"] = round(net_pnl, 2)

        self.day_pnl += net_pnl
        self.day_fees += pos["fees"]
        self.closed_today.append(deepcopy(pos))
        self.positions = [p for p in self.positions if p["status"] == "OPEN"]
        self._save_state()

        win = "WIN" if net_pnl > 0 else "LOSS"
        msg = f"EXIT {win} ({reason})\n{sym}\nP&L: Rs.{net_pnl:+,.0f} (after Rs.{pos['fees']:.0f} fees)\nDay total: Rs.{self.day_pnl:+,.0f}"
        log.info(msg.replace('\n', ' | '))
        tg(msg)

    # ── Main Loop ───────────────────────────────────────

    def run(self):
        log.info("=" * 60)
        log.info("TRADING BRAIN — 24/7 MODE")
        log.info("=" * 60)

        if not self.api.connect():
            log.warning("Initial connect failed. Will retry in loop.")

        while True:
            now = datetime.now(IST)
            self.scan_count += 1

            # ── RECONNECT if needed ──
            if not self.api.connected:
                if self.api.connect():
                    log.info("Reconnected!")
                else:
                    time.sleep(60)
                    continue

            # ── REFRESH WALLET every 5 scans ──
            if self.scan_count % 5 == 1:
                bal = self.api.get_balance()
                if bal > 0:
                    self.wallet = bal

            # ── MARKET HOURS: 9:15 AM - 3:30 PM Mon-Fri ──
            is_market = (now.weekday() < 5 and
                        (now.hour > 9 or (now.hour == 9 and now.minute >= 15)) and
                        (now.hour < 15 or (now.hour == 15 and now.minute <= 30)))

            if is_market:
                log.info(f"[SCAN {self.scan_count}] Market OPEN | wallet=Rs.{self.wallet:,.0f} | invested=Rs.{self.invested_amount():,.0f} | available=Rs.{self.available():,.0f} | day_pnl=Rs.{self.day_pnl:+,.0f} | day_fees=Rs.{self.day_fees:,.0f}")

                # 1. Check exits on all positions
                if self.positions:
                    self.check_exits()

                # 2. F&O: enter option signals if capital available
                if self.available() > 500:
                    for sig in self.get_option_signals():
                        self.enter_option(sig)

                # 3. Equity: scan and enter if capital available
                if self.available() > 1000 and now.hour < 14:  # no new equity after 2 PM
                    for sig in self.scan_equity():
                        self.enter_equity(sig)

                # Scan every 30s during market
                time.sleep(30)

            else:
                # ── AFTER HOURS ──
                # Reset daily counters at midnight
                if now.hour == 0 and now.minute < 5:
                    if self.closed_today:
                        log.info(f"Day summary: {len(self.closed_today)} trades, P&L=Rs.{self.day_pnl:+,.0f}, fees=Rs.{self.day_fees:,.0f}")
                        tg(f"📊 Day End\nTrades: {len(self.closed_today)}\nP&L: Rs.{self.day_pnl:+,.0f}\nFees: Rs.{self.day_fees:,.0f}")
                    self.closed_today = []
                    self.day_pnl = 0
                    self.day_fees = 0
                    self._save_state()

                # F&O positions hold overnight — just log
                fno_pos = [p for p in self.positions if p["type"] == "FNO" and p["status"] == "OPEN"]
                if fno_pos and self.scan_count % 60 == 0:  # log every 30 min after hours
                    log.info(f"After hours: {len(fno_pos)} F&O positions held overnight")

                time.sleep(60)  # check every minute after hours


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text())
            print(json.dumps(data, indent=2))
        else:
            print("No state file yet.")
    else:
        brain = TradingBrain()
        brain.run()

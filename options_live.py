"""
options_live.py — LIVE F&O Options Trader
==========================================
Takes signals from options_predictor's virtual trades and executes them LIVE.
Holds until 25% return, then exits. Runs on loop during market hours.

Strategy: Buy deep OTM calls on volatile stocks, hold days-to-weeks.
Capital: Allocated per trade (premium * lot_size must fit budget).
Exit: Only when +25% premium gain OR stop loss at -50%.

Usage:
  python options_live.py                    # Run live
  python options_live.py --capital 3200     # Set capital
  python options_live.py --status           # Show positions
  python options_live.py --min-return 25    # Min % return to exit (default 25)
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

# IST
IST = timezone(timedelta(hours=5, minutes=30))

# Logging
LOG_DIR = Path("C:/josho-trader/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "options_live.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("options_live")

# Telegram
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

def tg_alert(msg: str):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": f"*OPTIONS LIVE*\n\n{msg}", "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception:
        pass

# Data
DATA_DIR = Path("C:/josho-trader/data")
VIRTUAL_FILE = DATA_DIR / "virtual_trades.json"
LIVE_OPTIONS_FILE = DATA_DIR / "live_options.json"

# ── Groww API Client ─────────────────────────────────────────────

class GrowwFnO:
    """Minimal Groww F&O client for options trading."""

    def __init__(self):
        self.base_url = "https://growwapi.devapi.in"
        self.api_key = os.environ.get("GROWW_API_KEY", "")
        self.totp_token = os.environ.get("GROWW_TOTP_TOKEN", "")
        self.access_token = os.environ.get("GROWW_ACCESS_TOKEN", "")
        self.session = requests.Session()
        self.connected = False

    def connect(self):
        """Initialize API connection."""
        try:
            sys.path.insert(0, "C:/josho-trader/src")
            from client import GrowwClient
            self.client = GrowwClient()
            self.client.connect()
            self.connected = True
            log.info("Groww F&O API connected")
            return True
        except Exception as e:
            log.error(f"Groww connect failed: {e}")
            return False

    def get_balance(self) -> float:
        """Get REAL available balance from Groww wallet."""
        try:
            margin = self.client.get_margin()
            # Groww returns margin details - extract available cash
            if isinstance(margin, dict):
                # Try common keys
                for key in ['availableMargin', 'available_margin', 'net', 'cash', 'availableBalance']:
                    if key in margin:
                        bal = float(margin[key])
                        log.info(f"Wallet balance (from {key}): Rs.{bal:,.2f}")
                        return bal
                # If nested
                if 'data' in margin:
                    for key in ['availableMargin', 'net', 'cash']:
                        if key in margin['data']:
                            bal = float(margin['data'][key])
                            log.info(f"Wallet balance (from data.{key}): Rs.{bal:,.2f}")
                            return bal
                log.info(f"Margin response keys: {list(margin.keys())}")
            return 0
        except Exception as e:
            log.warning(f"Balance check failed: {e}")
            return 0

    def get_live_positions(self) -> list:
        """Get actual F&O positions from Groww to reconcile."""
        try:
            pos = self.client.get_positions(segment="FNO")
            if isinstance(pos, dict):
                return pos.get('positions', pos.get('data', []))
            return pos if isinstance(pos, list) else []
        except Exception as e:
            log.warning(f"Positions check failed: {e}")
            return []

    def get_ltp(self, symbol: str) -> float:
        """Get last traded price for an option symbol."""
        try:
            q = self.client.get_quote(symbol, exchange="NSE", segment="FNO")
            return q.get("last_price", 0)
        except Exception as e:
            log.error(f"LTP failed for {symbol}: {e}")
            return 0

    def buy_option(self, symbol: str, lot_size: int) -> dict:
        """Buy an option (NRML, market order)."""
        try:
            result = self.client.place_order(
                symbol=symbol,
                qty=lot_size,
                side="BUY",
                order_type="MARKET",
                product="NRML",
                segment="FNO",
                exchange="NSE",
            )
            log.info(f"BUY ORDER: {lot_size}x {symbol} -> {result}")
            return result
        except Exception as e:
            log.error(f"BUY failed for {symbol}: {e}")
            return {"error": str(e)}

    def sell_option(self, symbol: str, lot_size: int) -> dict:
        """Sell/exit an option position."""
        try:
            result = self.client.place_order(
                symbol=symbol,
                qty=lot_size,
                side="SELL",
                order_type="MARKET",
                product="NRML",
                segment="FNO",
                exchange="NSE",
            )
            log.info(f"SELL ORDER: {lot_size}x {symbol} -> {result}")
            return result
        except Exception as e:
            log.error(f"SELL failed for {symbol}: {e}")
            return {"error": str(e)}


# ── Live Options Manager ─────────────────────────────────────────

class LiveOptionsTrader:
    def __init__(self, capital: float = 3200, min_return_pct: float = 25.0, stop_loss_pct: float = 50.0):
        self.capital = capital
        self.min_return_pct = min_return_pct  # Don't exit until +25%
        self.stop_loss_pct = stop_loss_pct    # Cut at -50%
        self.api = GrowwFnO()
        self.live_positions = []
        self.closed_trades = []
        self._load_state()

    def _load_state(self):
        if LIVE_OPTIONS_FILE.exists():
            try:
                data = json.loads(LIVE_OPTIONS_FILE.read_text())
                self.live_positions = data.get("positions", [])
                self.closed_trades = data.get("closed", [])
                log.info(f"Loaded {len(self.live_positions)} live positions, {len(self.closed_trades)} closed")
            except Exception:
                pass

    def _save_state(self):
        LIVE_OPTIONS_FILE.write_text(json.dumps({
            "positions": self.live_positions,
            "closed": self.closed_trades,
            "last_updated": datetime.now(IST).isoformat(),
        }, indent=2))

    def get_virtual_signals(self) -> list:
        """Read signals from options_predictor's virtual trades."""
        if not VIRTUAL_FILE.exists():
            return []
        try:
            data = json.loads(VIRTUAL_FILE.read_text())
            trades = data.get("trades", [])
            # All open virtual positions with score >= 75 (aggressive)
            return [t for t in trades if t.get("status") == "OPEN" and t.get("score", 0) >= 75]
        except Exception:
            return []

    def is_already_live(self, option_symbol: str) -> bool:
        return any(p["option_symbol"] == option_symbol for p in self.live_positions)

    def can_afford(self, premium: float, lot_size: int, option_symbol: str = "") -> bool:
        invested = sum(p.get("entry_cost", 0) for p in self.live_positions)
        remaining = self.capital - invested
        # Use corrected lot size
        actual_lot = self.get_lot_size(option_symbol, lot_size) if option_symbol else lot_size
        cost = premium * actual_lot
        return cost <= remaining and cost > 0

    def get_lot_size(self, option_symbol: str, fallback_lot: int) -> int:
        """Get actual lot size from Groww, with common lot sizes as fallback."""
        # NSE lot sizes as of Apr 2026 (update periodically)
        KNOWN_LOTS = {
            # CORRECTED from Groww API instrument master (April 2026)
            "HINDALCO": 700, "ADANIPOWER": 1250, "VEDL": 1150, "PNB": 8000,
            "SAIL": 4700, "TATASTEEL": 5500, "JSWSTEEL": 675, "COALINDIA": 1350,
            "BPCL": 1975, "ONGC": 3850, "TATAPOWER": 1450, "SUZLON": 9025,
            "HFCL": 7000, "YESBANK": 31100, "NBCC": 6500, "NHPC": 6400, "IRFC": 4250,
        }
        # Extract stock name from option symbol (e.g. HINDALCO26JUN1200CE -> HINDALCO)
        stock = ""
        for name in KNOWN_LOTS:
            if option_symbol.startswith(name):
                stock = name
                break
        if stock in KNOWN_LOTS:
            return KNOWN_LOTS[stock]
        return fallback_lot

    def enter_position(self, signal: dict):
        """Go live on a virtual signal."""
        option_symbol = signal["option_symbol"]
        lot_size = self.get_lot_size(option_symbol, signal["lot_size"])

        if self.is_already_live(option_symbol):
            log.info(f"Already live on {option_symbol}, skipping")
            return

        # Get fresh premium
        ltp = self.api.get_ltp(option_symbol)
        if ltp <= 0:
            log.warning(f"No LTP for {option_symbol}, using virtual premium")
            ltp = signal.get("entry_premium", 0)

        cost = ltp * lot_size
        if not self.can_afford(ltp, lot_size):
            log.warning(f"Can't afford {option_symbol}: cost Rs.{cost:.0f} > remaining capital")
            return

        # Estimate fees — CA-grade post-April 2026 Budget rates
        try:
            sys.path.insert(0, "C:/josho-trader/src")
            from charges import ChargeCalculator
            calc = ChargeCalculator()
            # Use actual premium for accurate calculation
            buy_charges = calc.options_buy(ltp, lot_size)
            # Estimate sell at same premium (worst case for breakeven calc)
            sell_charges = calc.options_sell(ltp, lot_size)
            total_fees = buy_charges.total + sell_charges.total
        except Exception:
            # Fallback with CORRECTED rates
            brokerage = 20 * 2
            stt = cost * 0.0015  # 0.15% on sell (CORRECTED from 0.000625)
            exchange_txn = cost * 0.0003503 * 2
            total_fees = brokerage + stt + exchange_txn + (brokerage + exchange_txn) * 0.18
        breakeven_pct = (total_fees / cost) * 100

        # STRICT RULE: Expected profit at target MUST be > 2x total fees
        expected_profit_at_target = cost * (self.min_return_pct / 100)  # e.g. 25% of cost
        if expected_profit_at_target < total_fees * 2:
            log.warning(f"SKIP {option_symbol}: target profit Rs.{expected_profit_at_target:.0f} < 2x fees Rs.{total_fees*2:.0f} — NOT WORTH IT")
            return

        log.info(f"ENTERING LIVE: {option_symbol} | premium={ltp} | lot={lot_size} | cost=Rs.{cost:.0f} | fees=Rs.{total_fees:.0f} ({breakeven_pct:.1f}%) | profit@target=Rs.{expected_profit_at_target:.0f}")

        # Place live order
        result = self.api.buy_option(option_symbol, lot_size)
        if "error" in result:
            log.error(f"Order failed: {result['error']}")
            tg_alert(f"ORDER FAILED: {option_symbol}\n{result['error']}")
            return

        position = {
            "option_symbol": option_symbol,
            "stock_symbol": signal.get("stock_symbol", ""),
            "strike": signal.get("strike", 0),
            "expiry": signal.get("expiry", ""),
            "lot_size": lot_size,
            "entry_premium": ltp,
            "entry_cost": cost,
            "entry_time": datetime.now(IST).isoformat(),
            "current_premium": ltp,
            "score": signal.get("score", 0),
            "est_fees": round(total_fees, 2),
            "target_pct": self.min_return_pct,
            "stop_pct": self.stop_loss_pct,
            "order_result": result,
            "status": "OPEN",
        }
        self.live_positions.append(position)
        self._save_state()

        msg = (
            f"BUY {option_symbol}\n"
            f"Premium: Rs.{ltp} x {lot_size} = Rs.{cost:,.0f}\n"
            f"Score: {signal.get('score', 0)}\n"
            f"Target: +{self.min_return_pct}% (Rs.{cost * (1 + self.min_return_pct/100):,.0f})\n"
            f"Stop: -{self.stop_loss_pct}% (Rs.{cost * (1 - self.stop_loss_pct/100):,.0f})"
        )
        tg_alert(msg)
        log.info(msg.replace('\n', ' | '))

    def check_exits(self):
        """Check all live positions for exit conditions."""
        for pos in self.live_positions:
            if pos["status"] != "OPEN":
                continue

            symbol = pos["option_symbol"]
            ltp = self.api.get_ltp(symbol)
            if ltp <= 0:
                continue

            pos["current_premium"] = ltp
            entry = pos["entry_premium"]
            pnl_pct = ((ltp - entry) / entry) * 100 if entry > 0 else 0

            # EXIT: Target hit (+25% or whatever min_return_pct is)
            if pnl_pct >= self.min_return_pct:
                log.info(f"TARGET HIT: {symbol} +{pnl_pct:.1f}% (target was +{self.min_return_pct}%)")
                self._exit_position(pos, "TARGET_HIT", ltp, pnl_pct)
                continue

            # EXIT: Stop loss (-50%)
            if pnl_pct <= -self.stop_loss_pct:
                log.info(f"STOP LOSS: {symbol} {pnl_pct:.1f}% (stop was -{self.stop_loss_pct}%)")
                self._exit_position(pos, "STOP_LOSS", ltp, pnl_pct)
                continue

            # Log status
            log.info(f"HOLDING: {symbol} premium={ltp} pnl={pnl_pct:+.1f}% (waiting for +{self.min_return_pct}%)")

        self._save_state()

    def _exit_position(self, pos: dict, reason: str, exit_premium: float, pnl_pct: float):
        """Exit a live position."""
        symbol = pos["option_symbol"]
        lot_size = pos["lot_size"]

        result = self.api.sell_option(symbol, lot_size)

        gross_pnl = (exit_premium - pos["entry_premium"]) * lot_size
        net_pnl = gross_pnl - pos.get("est_fees", 0)

        pos["exit_premium"] = exit_premium
        pos["exit_time"] = datetime.now(IST).isoformat()
        pos["exit_reason"] = reason
        pos["gross_pnl"] = round(gross_pnl, 2)
        pos["net_pnl"] = round(net_pnl, 2)
        pos["pnl_pct"] = round(pnl_pct, 2)
        pos["status"] = "CLOSED"
        pos["sell_result"] = result

        self.closed_trades.append(deepcopy(pos))
        self.live_positions = [p for p in self.live_positions if p["status"] == "OPEN"]
        self._save_state()

        win = "WIN" if net_pnl > 0 else "LOSS"
        msg = (
            f"EXIT — {win} ({reason})\n"
            f"{symbol}\n"
            f"Entry: Rs.{pos['entry_premium']} -> Exit: Rs.{exit_premium}\n"
            f"Gross P&L: Rs.{gross_pnl:+,.0f} ({pnl_pct:+.1f}%)\n"
            f"Net P&L: Rs.{net_pnl:+,.0f} (after Rs.{pos.get('est_fees', 0):.0f} fees)"
        )
        tg_alert(msg)
        log.info(msg.replace('\n', ' | '))

    def run_loop(self):
        """Main loop: enter signals, check exits, repeat."""
        if not self.api.connect():
            log.critical("Cannot connect to Groww. Exiting.")
            return

        wallet_balance = self.api.get_balance()
        live_positions = self.api.get_live_positions()

        # Use real wallet balance as capital if available
        if wallet_balance > 0:
            self.capital = wallet_balance
            log.info(f"Using REAL wallet balance as capital: Rs.{wallet_balance:,.0f}")

        invested = sum(p.get("entry_cost", 0) for p in self.live_positions)
        available = self.capital - invested

        log.info(f"Wallet balance: Rs.{wallet_balance:,.0f}")
        log.info(f"Capital allocated: Rs.{self.capital:,.0f}")
        log.info(f"Already invested: Rs.{invested:,.0f}")
        log.info(f"Available to invest: Rs.{available:,.0f}")
        log.info(f"Live Groww positions: {len(live_positions)}")
        log.info(f"Min return to exit: +{self.min_return_pct}%")
        log.info(f"Stop loss: -{self.stop_loss_pct}%")
        log.info(f"RULE: Every trade profit > 2x fees — no fee-burning trades")
        log.info("=" * 60)

        scan_count = 0
        while True:
            now = datetime.now(IST)

            # Market hours: 9:15 AM - 3:30 PM IST (Mon-Fri)
            if now.weekday() >= 5:
                log.info("Weekend. Sleeping until Monday...")
                time.sleep(3600)
                continue

            hour_min = now.hour * 100 + now.minute
            if hour_min < 915:
                log.info(f"Pre-market. Waiting for 9:15 AM IST...")
                time.sleep(60)
                continue
            if hour_min > 1530:
                log.info("Market closed. Options positions held overnight (NRML). Sleeping...")
                time.sleep(3600)
                continue

            scan_count += 1
            log.info(f"[SCAN {scan_count}] Checking positions and signals...")

            # 1. Check exits on existing positions
            if self.live_positions:
                self.check_exits()

            # 2. Look for new signals from virtual predictor
            signals = self.get_virtual_signals()
            for sig in sorted(signals, key=lambda s: s.get("score", 0), reverse=True):
                sym = sig["option_symbol"]
                if not self.is_already_live(sym):
                    if self.can_afford(sig.get("entry_premium", 0), sig.get("lot_size", 0), sym):
                        log.info(f"New signal: {sym} score={sig.get('score', 0)}")
                        self.enter_position(sig)
                    else:
                        actual_lot = self.get_lot_size(sym, sig.get("lot_size", 0))
                        cost = sig.get("entry_premium", 0) * actual_lot
                        invested = sum(p.get("entry_cost", 0) for p in self.live_positions)
                        log.info(f"Can't afford {sym}: cost Rs.{cost:.0f}, remaining Rs.{self.capital - invested:.0f}")

            # 3. Refresh wallet balance every 10 scans
            if scan_count % 10 == 0:
                fresh_balance = self.api.get_balance()
                if fresh_balance > 0:
                    self.capital = fresh_balance

            # 4. Status summary
            total_invested = sum(p.get("entry_cost", 0) for p in self.live_positions)
            total_current = sum(p.get("current_premium", 0) * p.get("lot_size", 0) for p in self.live_positions)
            unrealized = total_current - total_invested
            available = self.capital - total_invested
            log.info(f"[STATUS] {len(self.live_positions)} pos | wallet=Rs.{self.capital:,.0f} | invested=Rs.{total_invested:,.0f} | available=Rs.{available:,.0f} | unrealized={unrealized:+,.0f}")

            # Scan every 30 seconds during market hours
            time.sleep(30)


def show_status():
    if not LIVE_OPTIONS_FILE.exists():
        print("No live options data yet.")
        return
    data = json.loads(LIVE_OPTIONS_FILE.read_text())
    positions = data.get("positions", [])
    closed = data.get("closed", [])
    print(f"\nLive positions: {len(positions)}")
    for p in positions:
        pnl = ((p.get('current_premium', 0) - p['entry_premium']) / p['entry_premium'] * 100) if p['entry_premium'] > 0 else 0
        print(f"  {p['option_symbol']} | entry={p['entry_premium']} | current={p.get('current_premium', '?')} | pnl={pnl:+.1f}% | target=+{p.get('target_pct', 25)}%")
    print(f"\nClosed trades: {len(closed)}")
    total_pnl = sum(t.get("net_pnl", 0) for t in closed)
    for t in closed[-5:]:
        print(f"  {t['option_symbol']} | {t.get('exit_reason', '?')} | net_pnl=Rs.{t.get('net_pnl', 0):+,.0f}")
    print(f"\nTotal realized P&L: Rs.{total_pnl:+,.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=3200)
    parser.add_argument("--min-return", type=float, default=25.0)
    parser.add_argument("--stop-loss", type=float, default=50.0)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        trader = LiveOptionsTrader(
            capital=args.capital,
            min_return_pct=args.min_return,
            stop_loss_pct=args.stop_loss,
        )
        trader.run_loop()

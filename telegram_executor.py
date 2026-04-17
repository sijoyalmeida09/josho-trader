"""
telegram_executor.py — Execute Trades from Telegram Signals
============================================================
Monitors Telegram for trade signals (from channels, groups, or your own bot)
and auto-executes them on Groww.

Signal formats supported:
  1. "BUY ADANIPOWER26JUN210CE @ 2.56"
  2. "ADANIPOWER 210 CE JUN BUY"
  3. "Entry: TATASTEEL26JUN1800CE premium 5.50 target 8.00 SL 3.00"
  4. JSON: {"symbol": "ADANIPOWER26JUN210CE", "action": "BUY", "premium": 2.56}

Also reads profit signals and mirrors them:
  "Booked profit on ADANIPOWER26JUN210CE @ 7.68"
  "EXIT ADANIPOWER26JUN210CE +200%"

Usage:
  python telegram_executor.py                    # Start monitoring
  python telegram_executor.py --dry-run          # Parse signals but don't execute
  python telegram_executor.py --status           # Show active positions
  python telegram_executor.py --execute ADANIPOWER26JUN210CE  # Execute specific trade NOW
"""

import os
import sys
import re
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
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "telegram_executor.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("tg_executor")

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")
SIGNALS_FILE = DATA_DIR / "telegram_signals.json"
POSITIONS_FILE = DATA_DIR / "tg_positions.json"
VIRTUAL_FILE = DATA_DIR / "virtual_trades.json"

# F&O lot sizes (Apr 2026)
LOT_SIZES = {
    # CORRECTED from Groww API instrument master (April 2026)
    "HINDALCO": 700, "ADANIPOWER": 1250, "VEDL": 1150, "PNB": 8000,
    "SAIL": 4700, "TATASTEEL": 5500, "JSWSTEEL": 675, "COALINDIA": 1350,
    "BPCL": 1975, "ONGC": 3850, "TATAPOWER": 1450, "SUZLON": 9025,
    "HFCL": 7000, "YESBANK": 31100, "NBCC": 6500, "NHPC": 6400, "IRFC": 4250,
    "NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 40,
}


def tg_send(msg: str):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": f"*TG EXECUTOR*\n\n{msg}", "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception:
        pass


def get_lot_size(symbol: str) -> int:
    """Extract lot size from symbol name."""
    for name, lot in LOT_SIZES.items():
        if symbol.upper().startswith(name):
            return lot
    return 0


# ── Signal Parser ────────────────────────────────────────

class SignalParser:
    """Parse trade signals from various Telegram message formats."""

    # Pattern: ADANIPOWER26JUN210CE or TATASTEEL26JUN1800CE
    OPTION_SYMBOL_RE = re.compile(
        r'([A-Z]+\d{2}[A-Z]{3}\d+(?:CE|PE))', re.IGNORECASE
    )

    # Pattern: "BUY SYMBOL @ price" or "SELL SYMBOL @ price"
    ACTION_PRICE_RE = re.compile(
        r'(BUY|SELL|ENTRY|EXIT|BOOK|BOOKED)\s+(\S+)\s*(?:@|at|premium|price)?\s*([\d.]+)?',
        re.IGNORECASE,
    )

    # Pattern: "target 8.00 SL 3.00" or "TP: 8 SL: 3"
    TARGET_SL_RE = re.compile(
        r'(?:target|tp|tgt)[:\s]*([\d.]+).*?(?:sl|stop|stoploss)[:\s]*([\d.]+)',
        re.IGNORECASE,
    )

    # Pattern: profit booking "profit +200%" or "booked +Rs.5000"
    PROFIT_RE = re.compile(
        r'(?:profit|booked|exit|closed?).*?([+-]?\d+(?:\.\d+)?)\s*%?',
        re.IGNORECASE,
    )

    @staticmethod
    def parse(text: str) -> dict:
        """Parse a Telegram message into a trade signal."""
        signal = {
            "raw": text,
            "symbol": "",
            "action": "",  # BUY, SELL, EXIT
            "premium": 0.0,
            "target": 0.0,
            "stop_loss": 0.0,
            "lot_size": 0,
            "confidence": 0,
            "parsed": False,
        }

        # Try JSON first
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "symbol" in data:
                signal["symbol"] = data["symbol"].upper()
                signal["action"] = data.get("action", "BUY").upper()
                signal["premium"] = float(data.get("premium", 0))
                signal["target"] = float(data.get("target", 0))
                signal["stop_loss"] = float(data.get("sl", data.get("stop_loss", 0)))
                signal["lot_size"] = get_lot_size(signal["symbol"])
                signal["parsed"] = True
                signal["confidence"] = 90
                return signal
        except (json.JSONDecodeError, ValueError):
            pass

        # Extract option symbol
        sym_match = SignalParser.OPTION_SYMBOL_RE.search(text)
        if sym_match:
            signal["symbol"] = sym_match.group(1).upper()
            signal["lot_size"] = get_lot_size(signal["symbol"])

        # Extract action and price
        action_match = SignalParser.ACTION_PRICE_RE.search(text)
        if action_match:
            action = action_match.group(1).upper()
            if action in ("EXIT", "BOOK", "BOOKED"):
                signal["action"] = "SELL"
            elif action in ("BUY", "ENTRY"):
                signal["action"] = "BUY"
            else:
                signal["action"] = action

            if not signal["symbol"] and action_match.group(2):
                candidate = action_match.group(2).upper()
                if SignalParser.OPTION_SYMBOL_RE.match(candidate):
                    signal["symbol"] = candidate
                    signal["lot_size"] = get_lot_size(candidate)

            if action_match.group(3):
                signal["premium"] = float(action_match.group(3))

        # Extract target and stop loss
        tgt_match = SignalParser.TARGET_SL_RE.search(text)
        if tgt_match:
            signal["target"] = float(tgt_match.group(1))
            signal["stop_loss"] = float(tgt_match.group(2))

        # Determine if this is an exit signal
        exit_keywords = ["exit", "sell", "book", "booked", "profit", "close", "closed", "square off"]
        text_lower = text.lower()
        if any(kw in text_lower for kw in exit_keywords) and signal["action"] != "BUY":
            signal["action"] = "SELL"

        # Default to BUY if symbol found but no action
        if signal["symbol"] and not signal["action"]:
            signal["action"] = "BUY"

        signal["parsed"] = bool(signal["symbol"] and signal["action"])
        signal["confidence"] = 80 if signal["parsed"] and signal["premium"] > 0 else 50 if signal["parsed"] else 0

        return signal


# ── Groww Executor ───────────────────────────────────────

class GrowwExecutor:
    """Execute trades on Groww with rate limiting and fee awareness."""

    def __init__(self):
        self.client = None
        self.connected = False
        self.last_call = 0
        self.min_gap = 2.0

    def connect(self) -> bool:
        try:
            sys.path.insert(0, "C:/josho-trader/src")
            from client import GrowwClient
            self.client = GrowwClient()
            self.client.connect()
            self.connected = True
            log.info("Groww connected")
            return True
        except Exception as e:
            log.error(f"Connect failed: {e}")
            return False

    def _throttle(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.min_gap:
            time.sleep(self.min_gap - elapsed)
        self.last_call = time.time()

    def get_balance(self) -> float:
        if not self.connected:
            return 0
        self._throttle()
        try:
            margin = self.client.get_margin()
            if isinstance(margin, dict):
                for key in ['availableMargin', 'available_margin', 'net', 'cash', 'availableBalance']:
                    if key in margin:
                        return float(margin[key])
                if 'data' in margin:
                    for key in ['availableMargin', 'net', 'cash']:
                        if key in margin.get('data', {}):
                            return float(margin['data'][key])
        except Exception as e:
            log.error(f"Balance check failed: {e}")
        return 0

    def get_ltp(self, symbol: str) -> float:
        if not self.connected:
            return 0
        self._throttle()
        try:
            q = self.client.get_quote(symbol, exchange="NSE", segment="FNO")
            return q.get("last_price", 0)
        except Exception as e:
            log.error(f"LTP failed for {symbol}: {e}")
            return 0

    def buy(self, symbol: str, qty: int) -> dict:
        if not self.connected:
            return {"error": "not connected"}
        self._throttle()
        try:
            result = self.client.place_order(
                symbol=symbol, qty=qty, side="BUY",
                order_type="MARKET", product="NRML",
                segment="FNO", exchange="NSE",
            )
            log.info(f"BUY ORDER: {qty}x {symbol} -> {result}")
            return result
        except Exception as e:
            log.error(f"BUY failed: {e}")
            return {"error": str(e)}

    def sell(self, symbol: str, qty: int) -> dict:
        if not self.connected:
            return {"error": "not connected"}
        self._throttle()
        try:
            result = self.client.place_order(
                symbol=symbol, qty=qty, side="SELL",
                order_type="MARKET", product="NRML",
                segment="FNO", exchange="NSE",
            )
            log.info(f"SELL ORDER: {qty}x {symbol} -> {result}")
            return result
        except Exception as e:
            log.error(f"SELL failed: {e}")
            return {"error": str(e)}


# ── Position Manager ─────────────────────────────────────

class PositionManager:
    """Track positions opened from Telegram signals."""

    def __init__(self):
        self.positions = []
        self.closed = []
        self._load()

    def _load(self):
        if POSITIONS_FILE.exists():
            try:
                data = json.loads(POSITIONS_FILE.read_text())
                self.positions = data.get("positions", [])
                self.closed = data.get("closed", [])
            except Exception:
                pass

    def _save(self):
        POSITIONS_FILE.write_text(json.dumps({
            "positions": self.positions,
            "closed": self.closed,
            "last_updated": datetime.now(IST).isoformat(),
        }, indent=2))

    def is_open(self, symbol: str) -> bool:
        return any(p["symbol"] == symbol and p["status"] == "OPEN" for p in self.positions)

    def add(self, symbol: str, premium: float, lot_size: int, fees: float, order_result: dict):
        self.positions.append({
            "symbol": symbol,
            "entry_premium": premium,
            "lot_size": lot_size,
            "entry_cost": premium * lot_size,
            "fees": round(fees, 2),
            "entry_time": datetime.now(IST).isoformat(),
            "status": "OPEN",
            "source": "telegram",
            "order_result": order_result,
        })
        self._save()

    def close(self, symbol: str, exit_premium: float, reason: str, order_result: dict) -> dict:
        for pos in self.positions:
            if pos["symbol"] == symbol and pos["status"] == "OPEN":
                pos["status"] = "CLOSED"
                pos["exit_premium"] = exit_premium
                pos["exit_time"] = datetime.now(IST).isoformat()
                pos["exit_reason"] = reason
                gross_pnl = (exit_premium - pos["entry_premium"]) * pos["lot_size"]
                pos["gross_pnl"] = round(gross_pnl, 2)
                pos["net_pnl"] = round(gross_pnl - pos["fees"], 2)
                self.closed.append(deepcopy(pos))
                self.positions = [p for p in self.positions if p["status"] == "OPEN"]
                self._save()
                return pos
        return {}

    def get_open(self) -> list:
        return [p for p in self.positions if p["status"] == "OPEN"]

    def total_invested(self) -> float:
        return sum(p.get("entry_cost", 0) for p in self.positions if p["status"] == "OPEN")


# ── Telegram Monitor ─────────────────────────────────────

class TelegramMonitor:
    """Poll Telegram for new messages and extract trade signals."""

    def __init__(self, source_chat_ids: list = None):
        self.token = TG_TOKEN
        self.last_update_id = 0
        self.source_chat_ids = source_chat_ids or []  # filter to specific chats
        self._load_offset()

    def _load_offset(self):
        offset_file = DATA_DIR / "tg_offset.json"
        if offset_file.exists():
            try:
                data = json.loads(offset_file.read_text())
                self.last_update_id = data.get("offset", 0)
            except Exception:
                pass

    def _save_offset(self):
        offset_file = DATA_DIR / "tg_offset.json"
        offset_file.write_text(json.dumps({"offset": self.last_update_id}))

    def get_updates(self) -> list:
        """Poll for new Telegram messages."""
        if not self.token:
            return []
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{self.token}/getUpdates",
                params={"offset": self.last_update_id + 1, "timeout": 10},
                timeout=15,
            )
            data = resp.json()
            if not data.get("ok"):
                return []

            updates = data.get("result", [])
            if updates:
                self.last_update_id = updates[-1]["update_id"]
                self._save_offset()

            messages = []
            for update in updates:
                msg = update.get("message") or update.get("channel_post", {})
                text = msg.get("text", "")
                chat_id = str(msg.get("chat", {}).get("id", ""))

                # Filter by source if configured
                if self.source_chat_ids and chat_id not in self.source_chat_ids:
                    continue

                if text:
                    messages.append({
                        "text": text,
                        "chat_id": chat_id,
                        "from": msg.get("from", {}).get("first_name", ""),
                        "date": msg.get("date", 0),
                    })
            return messages
        except Exception as e:
            log.error(f"Telegram poll failed: {e}")
            return []


# ── Main Executor ────────────────────────────────────────

class TelegramTradeExecutor:
    """Full pipeline: Telegram -> Parse -> Validate -> Execute -> Track."""

    def __init__(self, dry_run: bool = False, source_chats: list = None):
        self.dry_run = dry_run
        self.parser = SignalParser()
        self.groww = GrowwExecutor()
        self.positions = PositionManager()
        self.monitor = TelegramMonitor(source_chat_ids=source_chats)
        self.signals_log = []

    def _load_charge_calc(self):
        try:
            sys.path.insert(0, "C:/josho-trader/src")
            from charges import ChargeCalculator
            return ChargeCalculator()
        except ImportError:
            return None

    def process_signal(self, signal: dict) -> bool:
        """Validate and execute a parsed signal."""
        if not signal["parsed"] or signal["confidence"] < 50:
            log.info(f"Skipping weak signal: {signal['raw'][:80]}")
            return False

        symbol = signal["symbol"]
        action = signal["action"]
        lot_size = signal["lot_size"]

        if lot_size == 0:
            log.warning(f"Unknown lot size for {symbol}")
            tg_send(f"SKIP {symbol}: unknown lot size")
            return False

        # ── SELL / EXIT ──
        if action == "SELL":
            if not self.positions.is_open(symbol):
                log.info(f"No open position for {symbol}, skipping exit signal")
                return False

            ltp = self.groww.get_ltp(symbol) if not self.dry_run else signal["premium"]
            if ltp <= 0:
                log.warning(f"Can't get LTP for exit: {symbol}")
                return False

            if self.dry_run:
                log.info(f"DRY RUN EXIT: {symbol} @ {ltp}")
                tg_send(f"DRY RUN EXIT: {symbol} @ Rs.{ltp}")
                return True

            result = self.groww.sell(symbol, lot_size)
            closed = self.positions.close(symbol, ltp, "TELEGRAM_SIGNAL", result)

            if closed:
                msg = (
                    f"EXIT {symbol}\n"
                    f"Entry: Rs.{closed['entry_premium']} -> Exit: Rs.{ltp}\n"
                    f"Net P&L: Rs.{closed.get('net_pnl', 0):+,.0f}"
                )
                tg_send(msg)
                log.info(msg.replace('\n', ' | '))
            return True

        # ── BUY ──
        if self.positions.is_open(symbol):
            log.info(f"Already holding {symbol}, skipping")
            return False

        # Get live premium
        ltp = self.groww.get_ltp(symbol) if not self.dry_run else (signal["premium"] or 2.0)
        if ltp <= 0:
            log.warning(f"Can't get LTP for {symbol}")
            return False

        cost = ltp * lot_size

        # Fee check
        calc = self._load_charge_calc()
        if calc:
            buy_charges = calc.options_buy(ltp, lot_size)
            sell_charges = calc.options_sell(ltp, lot_size)
            total_fees = buy_charges.total + sell_charges.total
            breakeven_move = calc.min_profitable_premium_move(ltp, lot_size)
            breakeven_pct = (breakeven_move / ltp) * 100 if ltp > 0 else 0
        else:
            total_fees = 20 * 2 + cost * 0.0015 + cost * 0.0003503 * 2
            breakeven_pct = (total_fees / cost) * 100

        # Affordability check
        if not self.dry_run:
            balance = self.groww.get_balance()
            invested = self.positions.total_invested()
            available = balance - invested
            if cost > available:
                msg = f"SKIP {symbol}: cost Rs.{cost:.0f} > available Rs.{available:.0f}"
                log.warning(msg)
                tg_send(msg)
                return False

        # Profitability check: target profit must be > 2x fees
        target_pct = 25  # default 25% target
        expected_profit = cost * (target_pct / 100)
        if expected_profit < total_fees * 2:
            msg = f"SKIP {symbol}: target profit Rs.{expected_profit:.0f} < 2x fees Rs.{total_fees*2:.0f}"
            log.warning(msg)
            tg_send(msg)
            return False

        if self.dry_run:
            msg = (
                f"DRY RUN BUY: {symbol}\n"
                f"Premium: Rs.{ltp} x {lot_size} = Rs.{cost:,.0f}\n"
                f"Fees: Rs.{total_fees:.0f} | Breakeven: +{breakeven_pct:.2f}%\n"
                f"Confidence: {signal['confidence']}"
            )
            log.info(msg.replace('\n', ' | '))
            tg_send(msg)
            return True

        # EXECUTE
        result = self.groww.buy(symbol, lot_size)
        if "error" in result:
            tg_send(f"ORDER FAILED: {symbol}\n{result['error']}")
            return False

        self.positions.add(symbol, ltp, lot_size, total_fees, result)

        msg = (
            f"BUY {symbol}\n"
            f"Premium: Rs.{ltp} x {lot_size} = Rs.{cost:,.0f}\n"
            f"Fees: Rs.{total_fees:.0f} | Breakeven: +{breakeven_pct:.2f}%\n"
            f"Target: +25% = Rs.{ltp * 1.25:.2f} | Stop: -50% = Rs.{ltp * 0.50:.2f}\n"
            f"Source: Telegram signal"
        )
        tg_send(msg)
        log.info(msg.replace('\n', ' | '))
        return True

    def execute_specific(self, symbol: str):
        """Execute a specific trade immediately (e.g., from virtual_trades.json)."""
        signal = {
            "raw": f"Manual: {symbol}",
            "symbol": symbol.upper(),
            "action": "BUY",
            "premium": 0,
            "target": 0,
            "stop_loss": 0,
            "lot_size": get_lot_size(symbol),
            "confidence": 95,
            "parsed": True,
        }

        # Try to get premium from virtual trades
        if VIRTUAL_FILE.exists():
            try:
                data = json.loads(VIRTUAL_FILE.read_text())
                for t in data.get("trades", []):
                    if t["option_symbol"] == symbol:
                        signal["premium"] = t.get("entry_premium", 0)
                        signal["target"] = t.get("target_premium", 0)
                        signal["stop_loss"] = t.get("stop_premium", 0)
                        break
            except Exception:
                pass

        return self.process_signal(signal)

    def check_exits(self):
        """Monitor open positions for exit signals (target/stop)."""
        for pos in self.positions.get_open():
            symbol = pos["symbol"]
            ltp = self.groww.get_ltp(symbol)
            if ltp <= 0:
                continue

            entry = pos["entry_premium"]
            pnl_pct = ((ltp - entry) / entry) * 100 if entry > 0 else 0

            # Target: +25%
            if pnl_pct >= 25:
                log.info(f"TARGET HIT: {symbol} +{pnl_pct:.1f}%")
                if not self.dry_run:
                    result = self.groww.sell(symbol, pos["lot_size"])
                    closed = self.positions.close(symbol, ltp, "TARGET_HIT", result)
                    if closed:
                        tg_send(f"TARGET HIT {symbol}\nP&L: Rs.{closed['net_pnl']:+,.0f} ({pnl_pct:+.1f}%)")
                continue

            # Stop: -50%
            if pnl_pct <= -50:
                log.info(f"STOP LOSS: {symbol} {pnl_pct:.1f}%")
                if not self.dry_run:
                    result = self.groww.sell(symbol, pos["lot_size"])
                    closed = self.positions.close(symbol, ltp, "STOP_LOSS", result)
                    if closed:
                        tg_send(f"STOP LOSS {symbol}\nP&L: Rs.{closed['net_pnl']:+,.0f} ({pnl_pct:+.1f}%)")
                continue

            log.info(f"HOLD: {symbol} | premium={ltp} | pnl={pnl_pct:+.1f}%")

    def run(self):
        """Main loop: poll Telegram, parse signals, execute."""
        log.info("=" * 60)
        log.info("TELEGRAM TRADE EXECUTOR")
        log.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        log.info("=" * 60)

        if not self.dry_run:
            if not self.groww.connect():
                log.critical("Cannot connect to Groww. Exiting.")
                return

            balance = self.groww.get_balance()
            log.info(f"Wallet balance: Rs.{balance:,.0f}")
            log.info(f"Open positions: {len(self.positions.get_open())}")

        tg_send(f"Executor started ({'DRY RUN' if self.dry_run else 'LIVE'})\nOpen positions: {len(self.positions.get_open())}")

        scan = 0
        while True:
            now = datetime.now(IST)
            scan += 1

            # Market hours check
            is_market = (
                now.weekday() < 5
                and (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
                and (now.hour < 15 or (now.hour == 15 and now.minute <= 30))
            )

            # Always poll Telegram (signals can come anytime)
            messages = self.monitor.get_updates()
            for msg in messages:
                text = msg["text"]
                log.info(f"TG [{msg['from']}]: {text[:100]}")

                signal = SignalParser.parse(text)
                if signal["parsed"]:
                    log.info(f"SIGNAL: {signal['action']} {signal['symbol']} @ {signal['premium']} (confidence: {signal['confidence']})")

                    # Save signal
                    self.signals_log.append({
                        "time": datetime.now(IST).isoformat(),
                        "message": text[:200],
                        "signal": signal,
                        "executed": False,
                    })

                    if is_market or self.dry_run:
                        executed = self.process_signal(signal)
                        self.signals_log[-1]["executed"] = executed
                    else:
                        log.info("Market closed. Queuing signal for next session.")
                        tg_send(f"Signal received (market closed, queued):\n{signal['action']} {signal['symbol']}")

            # Check exits during market hours
            if is_market and self.positions.get_open() and scan % 2 == 0:
                self.check_exits()

            # Save signals log periodically
            if scan % 10 == 0 and self.signals_log:
                SIGNALS_FILE.write_text(json.dumps(self.signals_log[-100:], indent=2))

            # Status every 20 scans
            if scan % 20 == 0 and self.positions.get_open():
                invested = self.positions.total_invested()
                log.info(f"[STATUS] {len(self.positions.get_open())} positions | invested=Rs.{invested:,.0f}")

            time.sleep(15 if is_market else 60)


def show_status():
    if not POSITIONS_FILE.exists():
        print("No positions yet.")
        return
    data = json.loads(POSITIONS_FILE.read_text())
    positions = data.get("positions", [])
    closed = data.get("closed", [])

    print(f"\nOpen positions: {len(positions)}")
    for p in positions:
        print(f"  {p['symbol']} | entry={p['entry_premium']} | cost=Rs.{p.get('entry_cost', 0):,.0f} | fees=Rs.{p.get('fees', 0):.0f}")

    print(f"\nClosed trades: {len(closed)}")
    total_pnl = sum(t.get("net_pnl", 0) for t in closed)
    for t in closed[-5:]:
        print(f"  {t['symbol']} | {t.get('exit_reason', '?')} | net_pnl=Rs.{t.get('net_pnl', 0):+,.0f}")
    print(f"\nTotal realized P&L: Rs.{total_pnl:+,.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate but don't execute")
    parser.add_argument("--status", action="store_true", help="Show current positions")
    parser.add_argument("--execute", type=str, help="Execute specific symbol NOW (e.g., ADANIPOWER26JUN210CE)")
    parser.add_argument("--source-chats", nargs="*", help="Telegram chat IDs to monitor")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.execute:
        executor = TelegramTradeExecutor(dry_run=False)
        if not executor.groww.connect():
            print("Cannot connect to Groww")
            sys.exit(1)
        executor.execute_specific(args.execute)
    else:
        executor = TelegramTradeExecutor(
            dry_run=args.dry_run,
            source_chats=args.source_chats,
        )
        executor.run()

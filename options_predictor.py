"""
options_predictor.py — Virtual F&O Prediction Engine
=====================================================
Tests deep OTM options strategies against live market data.
Tracks virtual P&L to prove the algorithm works before going live.

Sijoy's style: Buy deep OTM calls on volatile metal/energy/PSU stocks,
far-month expiry, cheap premiums, hold days-to-weeks, exit on 2-5% stock moves
when option premium explodes 200-600%.

Usage:
  python options_predictor.py                  # Run predictor (default)
  python options_predictor.py --status         # Show virtual portfolio
  python options_predictor.py --reset          # Clear all virtual trades
  python options_predictor.py --scan           # One-time scan, no loop
"""

import os
import sys
import json
import time
import logging
import argparse
import signal
import math
from copy import deepcopy
from datetime import datetime, timedelta, date
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests

from dotenv import load_dotenv
load_dotenv(Path("C:/josho-trader/.env"))

# ── Logging ────────────────────────────────────────────────────────────
LOG_DIR = Path("C:/josho-trader/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "predictor.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("predictor")

# ── Config ─────────────────────────────────────────────────────────────
DATA_DIR = Path("C:/josho-trader/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRADES_FILE = DATA_DIR / "virtual_trades.json"

SCAN_INTERVAL = 15  # seconds between scans
MAX_VIRTUAL_POSITIONS = 5
VIRTUAL_CAPITAL = 50000  # Rs — virtual capital for paper trading
LOT_QTY = 1  # number of lots per virtual trade (we track per-lot P&L)

# Telegram
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── Target Stocks ──────────────────────────────────────────────────────
# Sijoy's preferred volatile metal/energy/PSU stocks
TARGET_STOCKS = [
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "SAIL",
    "TATAPOWER", "COALINDIA", "BPCL", "ONGC", "ADANIPOWER",
    "BANKBARODA", "PNB", "IRFC", "NHPC",
]

# Lot sizes per stock (NSE F&O standard lots — update if changed)
LOT_SIZES = {
    "TATASTEEL": 1500, "JSWSTEEL": 750, "HINDALCO": 1075,
    "VEDL": 1500, "SAIL": 4000, "TATAPOWER": 1350,
    "COALINDIA": 1200, "BPCL": 1800, "ONGC": 1925,
    "ADANIPOWER": 1250, "BANKBARODA": 2600, "PNB": 4000,
    "IRFC": 5000, "NHPC": 5000,
}


# ── Data Classes ───────────────────────────────────────────────────────
@dataclass
class StockSnapshot:
    symbol: str
    ltp: float
    open_price: float
    high: float
    low: float
    prev_close: float
    day_change_pct: float
    volume: int
    timestamp: str


@dataclass
class OptionCandidate:
    stock_symbol: str
    option_symbol: str
    strike: float
    expiry: str
    option_type: str  # "CE" or "PE"
    premium: float
    stock_ltp: float
    distance_pct: float  # how far OTM (% from stock price)
    lot_size: int
    volume: int
    open_interest: int
    score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)


@dataclass
class VirtualTrade:
    id: str
    stock_symbol: str
    option_symbol: str
    strike: float
    expiry: str
    entry_premium: float
    current_premium: float
    target_premium: float  # 3x entry
    stop_premium: float  # 0.5x entry
    lot_size: int
    entry_stock_price: float
    current_stock_price: float
    entry_time: str
    last_updated: str
    status: str  # "OPEN", "TARGET_HIT", "STOP_HIT", "EXPIRED", "MANUAL_EXIT"
    exit_premium: float = 0.0
    exit_time: str = ""
    pnl: float = 0.0  # per lot P&L in Rs
    pnl_pct: float = 0.0
    score: float = 0.0


# ── Groww API Connection ───────────────────────────────────────────────
class GrowwConnection:
    """Manages Groww API connection with retry logic."""

    def __init__(self):
        self._api = None
        self._token_expiry = 0.0

    def connect(self):
        """Authenticate and return GrowwAPI instance."""
        if self._api and time.time() < self._token_expiry:
            return self._api

        from growwapi import GrowwAPI

        max_retries = 5
        base_delay = 5

        for attempt in range(1, max_retries + 1):
            try:
                return self._attempt_connect(GrowwAPI)
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(
                    kw in error_str
                    for kw in ("rate", "429", "too many", "throttl")
                )
                if attempt == max_retries:
                    log.error(f"Connect failed after {max_retries} attempts: {e}")
                    raise
                delay = min(base_delay * (2 ** (attempt - 1)), 300)
                if is_rate_limit:
                    delay = min(delay * 2, 300)
                log.warning(
                    f"Connect attempt {attempt}/{max_retries} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)

        raise RuntimeError("Failed to connect")

    def _attempt_connect(self, GrowwAPI):
        """Single connection attempt — tries TOTP first, then API key."""
        totp_secret = os.environ.get("GROWW_TOTP_SECRET", "")
        totp_token = os.environ.get("GROWW_TOTP_TOKEN", "")

        if totp_secret and totp_token:
            import pyotp
            totp_code = pyotp.TOTP(totp_secret).now()
            token = GrowwAPI.get_access_token(api_key=totp_token, totp=totp_code)
            self._api = GrowwAPI(token)
            self._token_expiry = time.time() + 3600 * 8
            log.info("Connected via TOTP")
            return self._api

        api_key = os.environ.get("GROWW_API_KEY", "")
        secret = os.environ.get("GROWW_SECRET_KEY", "")
        if api_key and secret:
            token = GrowwAPI.get_access_token(api_key=api_key, secret=secret)
            self._api = GrowwAPI(token)
            self._token_expiry = time.time() + 3600 * 8
            log.info("Connected via API Key")
            return self._api

        # Fallback: direct access token from env
        access_token = os.environ.get("GROWW_ACCESS_TOKEN", "")
        if access_token:
            self._api = GrowwAPI(access_token)
            self._token_expiry = time.time() + 3600 * 2
            log.info("Connected via access token")
            return self._api

        raise RuntimeError(
            "No Groww credentials found. Set GROWW_TOTP_SECRET+GROWW_TOTP_TOKEN "
            "or GROWW_API_KEY+GROWW_SECRET_KEY or GROWW_ACCESS_TOKEN in .env"
        )

    @property
    def api(self):
        return self.connect()


# ── Telegram Alerts ────────────────────────────────────────────────────
def tg_alert(msg: str):
    """Send Telegram alert (non-blocking, best-effort)."""
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={
                "chat_id": TG_CHAT,
                "text": f"*OPTIONS PREDICTOR*\n\n{msg}",
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
    except Exception:
        pass


# ── Market Hours ───────────────────────────────────────────────────────
def is_market_hours() -> bool:
    """Check if within Indian market hours (9:15 - 15:30 IST).
    Also checks for weekdays only."""
    now_utc = datetime.utcnow()
    ist = now_utc + timedelta(hours=5, minutes=30)

    # Weekend check
    if ist.weekday() >= 5:
        return False

    market_open = ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = ist.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= ist <= market_close


def ist_now() -> str:
    """Current time in IST as string."""
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ── Expiry Helpers ─────────────────────────────────────────────────────
def get_far_month_expiries() -> list[str]:
    """Generate far-month expiry labels (2-3 months out).
    Returns month codes like '26MAY', '26JUN', '26JUL'."""
    now_utc = datetime.utcnow()
    ist = now_utc + timedelta(hours=5, minutes=30)
    results = []
    for months_ahead in (2, 3, 4):
        target = ist + timedelta(days=30 * months_ahead)
        yy = target.strftime("%y")
        mon = target.strftime("%b").upper()
        results.append(f"{yy}{mon}")
    return results


def build_option_symbol(stock: str, expiry_code: str, strike: int, opt_type: str) -> str:
    """Build Groww FNO trading symbol.
    Example: TATASTEEL26JUN160CE"""
    return f"{stock}{expiry_code}{strike}{opt_type}"


def estimate_expiry_date(expiry_code: str) -> str:
    """Estimate last Thursday of the month for expiry.
    expiry_code like '26JUN' -> '2026-06-25'"""
    months = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    yy = int(expiry_code[:2])
    mon_str = expiry_code[2:]
    month = months.get(mon_str, 1)
    year = 2000 + yy

    # Last Thursday of the month
    if month == 12:
        next_month_start = date(year + 1, 1, 1)
    else:
        next_month_start = date(year, month + 1, 1)
    last_day = next_month_start - timedelta(days=1)

    # Walk backwards to find Thursday (weekday 3)
    d = last_day
    while d.weekday() != 3:
        d -= timedelta(days=1)
    return d.isoformat()


# ── Strike Generation ──────────────────────────────────────────────────
def generate_otm_strikes(stock_price: float, stock_symbol: str) -> list[int]:
    """Generate deep OTM call strike prices (10-25% above current price).
    Strike step depends on stock price level."""
    if stock_price < 50:
        step = 2.5
    elif stock_price < 200:
        step = 5
    elif stock_price < 500:
        step = 10
    elif stock_price < 1000:
        step = 20
    elif stock_price < 2000:
        step = 50
    else:
        step = 100

    strikes = []
    # 10% to 25% OTM
    low_strike = stock_price * 1.10
    high_strike = stock_price * 1.25

    # Round to nearest step
    s = math.ceil(low_strike / step) * step
    while s <= high_strike:
        strikes.append(int(s))
        s += step

    # Also add a few closer strikes (7-10% OTM) for higher-probability plays
    close_low = stock_price * 1.07
    close_high = stock_price * 1.10
    cs = math.ceil(close_low / step) * step
    while cs < close_high:
        if int(cs) not in strikes:
            strikes.append(int(cs))
        cs += step

    return sorted(strikes)


# ── Scoring Engine ─────────────────────────────────────────────────────
def score_option(
    candidate: OptionCandidate,
    stock_snap: StockSnapshot,
) -> OptionCandidate:
    """Score an option candidate. Higher = better lottery ticket.

    Factors:
    1. Stock momentum (day change positive = bullish signal)
    2. Premium cheapness (lower premium relative to stock = more leverage)
    3. Distance to strike (10-20% OTM sweet spot)
    4. Intraday range (high volatility today = stock is moving)
    5. Volume (need some liquidity, but less important for far-month)
    """
    scored = deepcopy(candidate)
    breakdown = {}

    # 1. Momentum score (0-30 points)
    # Positive day = good, strong positive = better
    day_change = stock_snap.day_change_pct
    if day_change > 2:
        momentum = 30
    elif day_change > 1:
        momentum = 25
    elif day_change > 0:
        momentum = 15 + day_change * 10
    elif day_change > -1:
        momentum = 10  # Slight dip = okay for entry
    else:
        momentum = max(0, 5 + day_change * 2)  # Big drops = lower score
    breakdown["momentum"] = round(momentum, 1)

    # 2. Premium cheapness (0-25 points)
    # Lower premium = more leverage when stock moves
    premium_pct = (candidate.premium / stock_snap.ltp) * 100
    if premium_pct < 0.5:
        cheapness = 25
    elif premium_pct < 1.0:
        cheapness = 20
    elif premium_pct < 2.0:
        cheapness = 15
    elif premium_pct < 3.0:
        cheapness = 10
    else:
        cheapness = 5
    breakdown["cheapness"] = round(cheapness, 1)

    # 3. Distance to strike — sweet spot is 10-20% OTM (0-25 points)
    dist = candidate.distance_pct
    if 10 <= dist <= 15:
        distance_score = 25
    elif 15 < dist <= 20:
        distance_score = 20
    elif 7 <= dist < 10:
        distance_score = 18
    elif 20 < dist <= 25:
        distance_score = 12
    else:
        distance_score = 5
    breakdown["distance"] = round(distance_score, 1)

    # 4. Intraday volatility (0-15 points)
    if stock_snap.high > 0 and stock_snap.low > 0:
        intraday_range = ((stock_snap.high - stock_snap.low) / stock_snap.low) * 100
    else:
        intraday_range = 0
    if intraday_range > 3:
        vol_score = 15
    elif intraday_range > 2:
        vol_score = 12
    elif intraday_range > 1:
        vol_score = 8
    else:
        vol_score = 3
    breakdown["volatility"] = round(vol_score, 1)

    # 5. Option volume/OI (0-5 points)
    # Far-month options are thin; just need SOME activity
    oi = candidate.open_interest
    vol = candidate.volume
    if oi > 10000 or vol > 100:
        liquidity = 5
    elif oi > 1000 or vol > 10:
        liquidity = 3
    elif oi > 0 or vol > 0:
        liquidity = 1
    else:
        liquidity = 0
    breakdown["liquidity"] = round(liquidity, 1)

    scored.score = round(
        momentum + cheapness + distance_score + vol_score + liquidity, 1
    )
    scored.score_breakdown = breakdown
    return scored


# ── Virtual Portfolio ──────────────────────────────────────────────────
class VirtualPortfolio:
    """Track virtual (paper) option trades against live data."""

    def __init__(self):
        self.trades: list[dict] = []
        self._load()

    def _load(self):
        """Load trades from disk."""
        if TRADES_FILE.exists():
            try:
                data = json.loads(TRADES_FILE.read_text(encoding="utf-8"))
                self.trades = data.get("trades", [])
                log.info(f"Loaded {len(self.trades)} virtual trades from disk")
            except (json.JSONDecodeError, KeyError) as e:
                log.warning(f"Failed to load trades file: {e}")
                self.trades = []
        else:
            self.trades = []

    def _save(self):
        """Persist trades to disk."""
        summary = self._compute_summary()
        data = {
            "last_updated": ist_now(),
            "summary": summary,
            "trades": self.trades,
        }
        TRADES_FILE.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

    def _compute_summary(self) -> dict:
        """Compute portfolio summary stats."""
        open_trades = [t for t in self.trades if t["status"] == "OPEN"]
        closed_trades = [t for t in self.trades if t["status"] != "OPEN"]

        total_pnl = sum(t.get("pnl", 0) for t in closed_trades)
        open_pnl = sum(t.get("pnl", 0) for t in open_trades)
        winners = [t for t in closed_trades if t.get("pnl", 0) > 0]
        losers = [t for t in closed_trades if t.get("pnl", 0) < 0]

        return {
            "total_trades": len(self.trades),
            "open_positions": len(open_trades),
            "closed_trades": len(closed_trades),
            "realized_pnl": round(total_pnl, 2),
            "unrealized_pnl": round(open_pnl, 2),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": (
                round(len(winners) / len(closed_trades) * 100, 1)
                if closed_trades
                else 0
            ),
            "best_trade": (
                round(max((t.get("pnl_pct", 0) for t in closed_trades), default=0), 1)
            ),
            "worst_trade": (
                round(min((t.get("pnl_pct", 0) for t in closed_trades), default=0), 1)
            ),
        }

    @property
    def open_positions(self) -> list[dict]:
        return [t for t in self.trades if t["status"] == "OPEN"]

    @property
    def open_count(self) -> int:
        return len(self.open_positions)

    def has_position(self, option_symbol: str) -> bool:
        """Check if we already hold this option."""
        return any(
            t["option_symbol"] == option_symbol and t["status"] == "OPEN"
            for t in self.trades
        )

    def has_stock_position(self, stock_symbol: str) -> bool:
        """Check if we already have an open trade on this stock."""
        return any(
            t["stock_symbol"] == stock_symbol and t["status"] == "OPEN"
            for t in self.trades
        )

    def enter_trade(self, candidate: OptionCandidate) -> Optional[dict]:
        """Enter a new virtual trade."""
        if self.open_count >= MAX_VIRTUAL_POSITIONS:
            log.info(
                f"Max positions ({MAX_VIRTUAL_POSITIONS}) reached. Skipping {candidate.option_symbol}"
            )
            return None

        if self.has_stock_position(candidate.stock_symbol):
            log.info(
                f"Already have position in {candidate.stock_symbol}. Skipping."
            )
            return None

        trade_id = f"VT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{candidate.stock_symbol}"
        entry_cost = candidate.premium * candidate.lot_size

        trade = {
            "id": trade_id,
            "stock_symbol": candidate.stock_symbol,
            "option_symbol": candidate.option_symbol,
            "strike": candidate.strike,
            "expiry": candidate.expiry,
            "entry_premium": candidate.premium,
            "current_premium": candidate.premium,
            "target_premium": round(candidate.premium * 3, 2),  # 3x target (200% gain)
            "stop_premium": round(candidate.premium * 0.5, 2),  # 50% stop loss
            "lot_size": candidate.lot_size,
            "entry_stock_price": candidate.stock_ltp,
            "current_stock_price": candidate.stock_ltp,
            "entry_time": ist_now(),
            "last_updated": ist_now(),
            "status": "OPEN",
            "exit_premium": 0.0,
            "exit_time": "",
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "score": candidate.score,
            "score_breakdown": candidate.score_breakdown,
            "entry_cost": round(entry_cost, 2),
        }

        self.trades.append(trade)
        self._save()

        msg = (
            f"VIRTUAL ENTRY\n"
            f"Stock: {candidate.stock_symbol} @ Rs.{candidate.stock_ltp:.2f}\n"
            f"Option: {candidate.option_symbol}\n"
            f"Premium: Rs.{candidate.premium:.2f}\n"
            f"Target: Rs.{trade['target_premium']:.2f} (3x)\n"
            f"Stop: Rs.{trade['stop_premium']:.2f} (0.5x)\n"
            f"Score: {candidate.score}/100\n"
            f"Cost (1 lot): Rs.{entry_cost:,.0f}"
        )
        log.info(msg)
        tg_alert(msg)
        return trade

    def update_trade(self, trade: dict, current_premium: float, current_stock: float):
        """Update a trade with latest prices and check exits."""
        old_status = trade["status"]
        trade["current_premium"] = current_premium
        trade["current_stock_price"] = current_stock
        trade["last_updated"] = ist_now()

        # Calculate P&L
        pnl_per_unit = current_premium - trade["entry_premium"]
        trade["pnl"] = round(pnl_per_unit * trade["lot_size"], 2)
        entry = trade["entry_premium"]
        trade["pnl_pct"] = round((pnl_per_unit / entry) * 100, 2) if entry > 0 else 0

        # Check target hit
        if current_premium >= trade["target_premium"]:
            trade["status"] = "TARGET_HIT"
            trade["exit_premium"] = current_premium
            trade["exit_time"] = ist_now()
            msg = (
                f"TARGET HIT\n"
                f"{trade['option_symbol']}\n"
                f"Entry: Rs.{entry:.2f} -> Exit: Rs.{current_premium:.2f}\n"
                f"P&L: Rs.{trade['pnl']:+,.0f} ({trade['pnl_pct']:+.1f}%)\n"
                f"Stock moved: {trade['entry_stock_price']:.2f} -> {current_stock:.2f}"
            )
            log.info(msg)
            tg_alert(msg)

        # Check stop loss hit
        elif current_premium <= trade["stop_premium"] and current_premium > 0:
            trade["status"] = "STOP_HIT"
            trade["exit_premium"] = current_premium
            trade["exit_time"] = ist_now()
            msg = (
                f"STOP HIT\n"
                f"{trade['option_symbol']}\n"
                f"Entry: Rs.{entry:.2f} -> Exit: Rs.{current_premium:.2f}\n"
                f"P&L: Rs.{trade['pnl']:+,.0f} ({trade['pnl_pct']:+.1f}%)"
            )
            log.info(msg)
            tg_alert(msg)

        # Check expiry
        expiry_date = trade.get("expiry", "")
        if expiry_date:
            try:
                exp = datetime.fromisoformat(expiry_date).date()
                today_ist = (
                    datetime.utcnow() + timedelta(hours=5, minutes=30)
                ).date()
                if today_ist >= exp and trade["status"] == "OPEN":
                    trade["status"] = "EXPIRED"
                    trade["exit_premium"] = current_premium
                    trade["exit_time"] = ist_now()
                    log.info(
                        f"EXPIRED: {trade['option_symbol']} "
                        f"P&L: Rs.{trade['pnl']:+,.0f}"
                    )
            except ValueError:
                pass

        if trade["status"] != old_status:
            self._save()

    def save_if_dirty(self):
        """Save after batch updates."""
        self._save()

    def reset(self):
        """Clear all trades."""
        self.trades = []
        self._save()
        log.info("Virtual portfolio reset")

    def print_status(self):
        """Print current portfolio status."""
        summary = self._compute_summary()
        print("\n" + "=" * 70)
        print("  OPTIONS PREDICTOR — Virtual Portfolio")
        print("=" * 70)
        print(f"  Total trades: {summary['total_trades']}")
        print(f"  Open positions: {summary['open_positions']}")
        print(f"  Closed trades: {summary['closed_trades']}")
        print(f"  Win rate: {summary['win_rate']}%")
        print(f"  Realized P&L: Rs.{summary['realized_pnl']:+,.2f}")
        print(f"  Unrealized P&L: Rs.{summary['unrealized_pnl']:+,.2f}")
        print(f"  Best trade: {summary['best_trade']:+.1f}%")
        print(f"  Worst trade: {summary['worst_trade']:+.1f}%")
        print("-" * 70)

        open_trades = self.open_positions
        if open_trades:
            print("\n  OPEN POSITIONS:")
            for t in open_trades:
                pnl_str = f"Rs.{t['pnl']:+,.0f} ({t['pnl_pct']:+.1f}%)"
                print(
                    f"    {t['option_symbol']:30s} "
                    f"Entry: Rs.{t['entry_premium']:.2f}  "
                    f"Now: Rs.{t['current_premium']:.2f}  "
                    f"P&L: {pnl_str}"
                )
        else:
            print("\n  No open positions.")

        closed = [t for t in self.trades if t["status"] != "OPEN"]
        if closed:
            print(f"\n  RECENT CLOSED (last 10):")
            for t in closed[-10:]:
                pnl_str = f"Rs.{t['pnl']:+,.0f} ({t['pnl_pct']:+.1f}%)"
                print(
                    f"    {t['option_symbol']:30s} "
                    f"{t['status']:12s} "
                    f"P&L: {pnl_str}"
                )

        print("=" * 70 + "\n")


# ── Main Engine ────────────────────────────────────────────────────────
class OptionPredictor:
    """Deep OTM options prediction engine with virtual trading."""

    def __init__(self):
        self.conn = GrowwConnection()
        self.portfolio = VirtualPortfolio()
        self.running = False
        self._rate_limit_backoff = 1  # seconds between API calls

    def _api_call_with_backoff(self, fn, *args, **kwargs):
        """Make an API call with rate limit protection."""
        time.sleep(self._rate_limit_backoff)
        try:
            result = fn(*args, **kwargs)
            # Successful call — reduce backoff
            self._rate_limit_backoff = max(0.5, self._rate_limit_backoff * 0.9)
            return result
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "429" in error_str or "throttl" in error_str:
                self._rate_limit_backoff = min(
                    self._rate_limit_backoff * 2, 30
                )
                log.warning(
                    f"Rate limited. Backoff now {self._rate_limit_backoff:.1f}s"
                )
            raise

    # ── Stock Data ─────────────────────────────────────────────────
    def get_stock_snapshot(self, symbol: str) -> Optional[StockSnapshot]:
        """Fetch live stock data from Groww."""
        try:
            quote = self._api_call_with_backoff(
                self.conn.api.get_quote,
                trading_symbol=f"{symbol}-EQ",
                exchange="NSE",
                segment="CASH",
            )
            if not quote or not quote.get("last_price"):
                return None

            ohlc = quote.get("ohlc", {})
            return StockSnapshot(
                symbol=symbol,
                ltp=float(quote["last_price"]),
                open_price=float(ohlc.get("open", 0)),
                high=float(ohlc.get("high", 0)),
                low=float(ohlc.get("low", 0)),
                prev_close=float(ohlc.get("close", 0)),
                day_change_pct=float(quote.get("day_change_perc", 0)),
                volume=int(quote.get("volume", 0)),
                timestamp=ist_now(),
            )
        except Exception as e:
            log.warning(f"Failed to get snapshot for {symbol}: {e}")
            return None

    def get_option_quote(self, option_symbol: str) -> Optional[dict]:
        """Fetch live option quote."""
        try:
            quote = self._api_call_with_backoff(
                self.conn.api.get_quote,
                trading_symbol=option_symbol,
                exchange="NSE",
                segment="FNO",
            )
            if quote and quote.get("last_price"):
                return quote
            return None
        except Exception as e:
            log.debug(f"Option quote failed for {option_symbol}: {e}")
            return None

    # ── Scan Opportunities ─────────────────────────────────────────
    def scan_opportunities(self) -> list[OptionCandidate]:
        """Scan all target stocks for deep OTM call opportunities.

        For each stock:
        1. Get live price
        2. Generate OTM strike prices (10-25% above)
        3. For each far-month expiry, try to get option quotes
        4. Score each valid option
        """
        log.info("Scanning for deep OTM opportunities...")
        candidates = []
        expiry_codes = get_far_month_expiries()

        for stock in TARGET_STOCKS:
            snap = self.get_stock_snapshot(stock)
            if not snap:
                continue

            lot_size = LOT_SIZES.get(stock, 1000)
            strikes = generate_otm_strikes(snap.ltp, stock)

            if not strikes:
                continue

            log.info(
                f"{stock}: Rs.{snap.ltp:.2f} ({snap.day_change_pct:+.2f}%) "
                f"— checking {len(strikes)} strikes x {len(expiry_codes)} expiries"
            )

            for expiry_code in expiry_codes:
                expiry_date = estimate_expiry_date(expiry_code)

                for strike in strikes:
                    option_symbol = build_option_symbol(
                        stock, expiry_code, strike, "CE"
                    )

                    # Skip if we already hold this
                    if self.portfolio.has_position(option_symbol):
                        continue

                    quote = self.get_option_quote(option_symbol)
                    if not quote:
                        continue

                    premium = float(quote.get("last_price", 0))
                    if premium <= 0:
                        continue

                    # Skip if premium is too expensive (>5% of stock price)
                    if premium / snap.ltp > 0.05:
                        continue

                    # Skip if premium is too cheap (likely no real market)
                    if premium < 0.5:
                        continue

                    distance_pct = ((strike - snap.ltp) / snap.ltp) * 100

                    candidate = OptionCandidate(
                        stock_symbol=stock,
                        option_symbol=option_symbol,
                        strike=float(strike),
                        expiry=expiry_date,
                        option_type="CE",
                        premium=premium,
                        stock_ltp=snap.ltp,
                        distance_pct=round(distance_pct, 2),
                        lot_size=lot_size,
                        volume=int(quote.get("volume", 0)),
                        open_interest=int(quote.get("open_interest", 0)),
                    )

                    scored = score_option(candidate, snap)
                    candidates.append(scored)

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)

        if candidates:
            log.info(
                f"Found {len(candidates)} candidates. "
                f"Top: {candidates[0].option_symbol} (score={candidates[0].score})"
            )
        else:
            log.info("No valid candidates found this scan")

        return candidates

    # ── Predict & Enter Trades ─────────────────────────────────────
    def predict_and_enter(self, candidates: list[OptionCandidate]):
        """For top-scored candidates, decide whether to enter virtual trades.

        Entry criteria:
        - Score >= 50 (out of 100)
        - Not already holding the same stock
        - Under max position limit
        - Reasonable premium (Rs.1 to Rs.50 sweet spot for cheap lottos)
        """
        if not candidates:
            return

        slots_available = MAX_VIRTUAL_POSITIONS - self.portfolio.open_count
        if slots_available <= 0:
            log.info("Portfolio full. Waiting for exits.")
            return

        entered = 0
        for candidate in candidates:
            if entered >= slots_available:
                break

            # Minimum score threshold
            if candidate.score < 50:
                log.debug(
                    f"Skipping {candidate.option_symbol} — score {candidate.score} < 50"
                )
                continue

            # Premium sweet spot (Rs.1 to Rs.50 per unit)
            if candidate.premium > 50:
                log.debug(
                    f"Skipping {candidate.option_symbol} — premium Rs.{candidate.premium} too high"
                )
                continue

            # Skip if already holding this stock
            if self.portfolio.has_stock_position(candidate.stock_symbol):
                continue

            trade = self.portfolio.enter_trade(candidate)
            if trade:
                entered += 1

        if entered:
            log.info(f"Entered {entered} new virtual trades")

    # ── Update Open Positions ──────────────────────────────────────
    def update_positions(self):
        """Update all open virtual positions with live prices."""
        open_trades = self.portfolio.open_positions
        if not open_trades:
            return

        log.info(f"Updating {len(open_trades)} open positions...")
        for trade in open_trades:
            # Get current option premium
            opt_quote = self.get_option_quote(trade["option_symbol"])
            current_premium = (
                float(opt_quote["last_price"])
                if opt_quote and opt_quote.get("last_price")
                else trade["current_premium"]
            )

            # Get current stock price
            snap = self.get_stock_snapshot(trade["stock_symbol"])
            current_stock = snap.ltp if snap else trade["current_stock_price"]

            self.portfolio.update_trade(trade, current_premium, current_stock)

        self.portfolio.save_if_dirty()

    # ── One Scan Cycle ─────────────────────────────────────────────
    def run_one_cycle(self):
        """Execute one full scan-predict-update cycle."""
        try:
            # 1. Update existing positions first
            self.update_positions()

            # 2. Scan for new opportunities if we have slots
            if self.portfolio.open_count < MAX_VIRTUAL_POSITIONS:
                candidates = self.scan_opportunities()
                self.predict_and_enter(candidates)

            # 3. Log summary
            summary = self.portfolio._compute_summary()
            log.info(
                f"Cycle done | "
                f"Open: {summary['open_positions']} | "
                f"Closed: {summary['closed_trades']} | "
                f"Win rate: {summary['win_rate']}% | "
                f"Realized: Rs.{summary['realized_pnl']:+,.0f} | "
                f"Unrealized: Rs.{summary['unrealized_pnl']:+,.0f}"
            )

        except Exception as e:
            log.error(f"Cycle error: {e}", exc_info=True)

    # ── Main Loop ──────────────────────────────────────────────────
    def run(self):
        """Main prediction loop — runs during market hours."""
        self.running = True
        log.info("=" * 60)
        log.info("OPTIONS PREDICTOR STARTING")
        log.info(f"Target stocks: {', '.join(TARGET_STOCKS)}")
        log.info(f"Max positions: {MAX_VIRTUAL_POSITIONS}")
        log.info(f"Scan interval: {SCAN_INTERVAL}s")
        log.info(f"Current open: {self.portfolio.open_count}")
        log.info("=" * 60)

        tg_alert(
            f"Predictor started\n"
            f"Stocks: {len(TARGET_STOCKS)}\n"
            f"Open positions: {self.portfolio.open_count}"
        )

        # Graceful shutdown
        def handle_signal(signum, frame):
            log.info("Shutdown signal received")
            self.running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        consecutive_errors = 0
        max_consecutive_errors = 10

        while self.running:
            try:
                if is_market_hours():
                    self.run_one_cycle()
                    consecutive_errors = 0
                    time.sleep(SCAN_INTERVAL)
                else:
                    # Outside market hours — slow poll (check every 2 minutes)
                    now_ist = ist_now()
                    log.info(f"Market closed. IST: {now_ist}. Sleeping 2 min...")
                    time.sleep(120)

            except KeyboardInterrupt:
                log.info("Keyboard interrupt — shutting down")
                break
            except Exception as e:
                consecutive_errors += 1
                log.error(f"Loop error ({consecutive_errors}): {e}", exc_info=True)

                if consecutive_errors >= max_consecutive_errors:
                    log.critical(
                        f"Too many consecutive errors ({consecutive_errors}). "
                        f"Stopping."
                    )
                    tg_alert(
                        f"STOPPED — {consecutive_errors} consecutive errors\n"
                        f"Last: {str(e)[:200]}"
                    )
                    break

                # Exponential backoff on errors
                sleep_time = min(30 * (2 ** (consecutive_errors - 1)), 300)
                log.info(f"Backing off {sleep_time}s...")
                time.sleep(sleep_time)

        log.info("Predictor stopped")
        tg_alert("Predictor stopped")


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Options Predictor — Virtual F&O prediction engine"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show virtual portfolio"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Clear all virtual trades"
    )
    parser.add_argument(
        "--scan", action="store_true", help="One-time scan (no loop)"
    )
    args = parser.parse_args()

    if args.status:
        portfolio = VirtualPortfolio()
        portfolio.print_status()
        return

    if args.reset:
        portfolio = VirtualPortfolio()
        portfolio.reset()
        print("Virtual portfolio cleared.")
        return

    predictor = OptionPredictor()

    if args.scan:
        log.info("Running single scan...")
        predictor.run_one_cycle()
        predictor.portfolio.print_status()
        return

    # Default: run continuous loop
    predictor.run()


if __name__ == "__main__":
    main()

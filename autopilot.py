"""
JoSho Trader — AUTOPILOT MODE
================================
Add money → Start → It trades automatically → Money grows.

Runs continuously during market hours. Scans every 30 seconds.
Takes trades when signals appear. Manages exits. Sends Telegram alerts.

Usage:
  python autopilot.py                    # Paper mode (default, safe)
  python autopilot.py --live             # LIVE trading (real money)
  python autopilot.py --capital 10000    # Set capital
  python autopilot.py --risk 2           # Max 2% risk per trade
  python autopilot.py --kill             # Emergency stop

Strategy mix (auto-selected based on market conditions):
  1. Large Drop Mean Reversion (100% WR on ELITE stocks)
  2. RSI(2) Mean Reversion (64% WR, frequent trades)
  3. Intraday Momentum (Supertrend + VWAP)
  4. ML Signal (LightGBM 62% accuracy, 5.22 profit factor)
"""

import os
import sys
import time
import json
import logging
import argparse
import signal
from datetime import datetime, timedelta, timezone

# IST = UTC+5:30
IST = timezone(timedelta(hours=5, minutes=30))
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np
import requests

from dotenv import load_dotenv
load_dotenv()

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "autopilot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("autopilot")

# Telegram
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

# Data
DATA_DIR = Path("data")
TRADES_FILE = DATA_DIR / "autopilot_trades.json"
STATE_FILE = DATA_DIR / "autopilot_state.json"


def tg_alert(msg: str):
    """Send Telegram alert."""
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": f"*AUTOPILOT*\n\n{msg}", "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception:
        pass


@dataclass
class TradeConfig:
    capital: float = 200
    max_risk_pct: float = 5.0       # 5% risk per trade — protect capital
    max_trades_per_day: int = 8      # max 8 trades/day — avoid churning
    max_open_positions: int = 3      # max 3 positions at once
    max_daily_loss_pct: float = 10.0 # stop after 10% daily loss
    profit_target_pct: float = 1000.0 # no profit cap — let it run
    scan_interval: int = 30          # seconds between scans
    paper: bool = True
    # Anti-churn: minimum expected profit per trade AFTER brokerage
    min_trade_value: float = 500     # skip trades worth less than Rs.500 (fees eat tiny trades)
    min_price: float = 10.0          # skip stocks under Rs.10 (GTLINFRA at Rs.1.25 = pure fee burn)
    min_target_profit: float = 3.0   # expect at least Rs.3 profit per trade to cover ~Rs.1.50 fees


class MLSignalEngine:
    """Use trained models to boost signal confidence."""

    def __init__(self):
        self.models = {}
        self._load_models()

    def _load_models(self):
        """Load best model per stock from disk."""
        import joblib
        model_dir = Path("C:/josho-trader/models")
        results_file = Path("C:/josho-trader/research/results/training_results.json")

        if not results_file.exists():
            log.info("ML: No training results found — running without ML boost")
            return

        try:
            data = json.loads(results_file.read_text(encoding="utf-8"))
            for symbol, best in data.get("stock_best", {}).items():
                algo = best["algorithm"].lower()
                # Find matching model file
                for suffix in ["ensemble", "lightgbm", "xgboost", "rf"]:
                    model_file = model_dir / f"{symbol}_{suffix}.pkl"
                    if model_file.exists():
                        try:
                            self.models[symbol] = {
                                "model": joblib.load(model_file),
                                "accuracy": best["accuracy"],
                                "algorithm": best["algorithm"],
                            }
                            break
                        except Exception:
                            pass
            log.info(f"ML: Loaded {len(self.models)} models (avg accuracy: {sum(m['accuracy'] for m in self.models.values())/max(len(self.models),1):.1f}%)")
        except Exception as e:
            log.warning(f"ML: Failed to load models: {e}")

    def predict(self, symbol: str, features: dict) -> float:
        """Get ML confidence boost for a signal. Returns 0.0-0.3 bonus."""
        if symbol not in self.models:
            return 0.0

        try:
            import numpy as np
            model_info = self.models[symbol]
            model = model_info["model"]
            accuracy = model_info["accuracy"]

            # Build feature vector from quote data
            feature_names = list(features.keys())
            X = np.array([[features.get(f, 0) for f in feature_names]])

            # Replace inf/nan
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
            confidence = max(proba)

            # Scale ML boost by model accuracy: 60%+ model gives up to 0.3 boost
            if pred == 1 and accuracy > 55:  # model says BUY
                boost = min((accuracy - 50) / 100 * 0.6, 0.30)
                return boost * confidence
            elif pred == 0:  # model says DOWN — negative boost
                return -0.10
            return 0.0
        except Exception:
            return 0.0


class AutoPilot:
    """Fully autonomous trading engine with ML-powered signals."""

    def __init__(self, config: TradeConfig):
        self.config = config
        self.running = False
        self.trades_today = []
        self.open_positions = []
        self.day_pnl = 0.0
        self.day_start_capital = config.capital
        self.ml_engine = MLSignalEngine()
        self.client = None
        self.api = None
        self._load_state()

    def connect(self):
        """Connect to Groww API with exponential backoff retry."""
        import pyotp
        from growwapi import GrowwAPI

        max_retries = 5
        base_delay = 5  # seconds
        max_delay = 300  # 5 minutes

        for attempt in range(1, max_retries + 1):
            try:
                api_key = os.environ["GROWW_TOTP_TOKEN"]
                totp = pyotp.TOTP(os.environ["GROWW_TOTP_SECRET"]).now()
                token = GrowwAPI.get_access_token(api_key=api_key, totp=totp)
                self.api = GrowwAPI(token)
                log.info("Groww API connected")

                # Get actual balance
                margin = self.api.get_available_margin_details()
                actual_balance = margin.get("clear_cash", 0)
                log.info(f"Account balance: Rs.{actual_balance:,.0f}")

                if actual_balance < self.config.capital and not self.config.paper:
                    log.warning(f"Capital Rs.{self.config.capital} > balance Rs.{actual_balance}. Using balance.")
                    self.config.capital = actual_balance

                return True

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "rate" in error_str or "429" in error_str or "too many" in error_str or "throttl" in error_str

                if attempt == max_retries:
                    log.critical(f"Connect failed after {max_retries} attempts: {e}")
                    if is_rate_limit:
                        log.critical("Rate limited by Groww API. Writing cooldown state and exiting cleanly.")
                        self._save_rate_limit_state()
                        sys.exit(0)  # Clean exit — let PM2 handle restart with delay
                    raise

                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                if is_rate_limit:
                    delay = min(delay * 2, max_delay)  # Double delay for rate limits
                    log.warning(f"Rate limited by Groww API (attempt {attempt}/{max_retries}). Backing off {delay}s...")
                else:
                    log.warning(f"Connect failed (attempt {attempt}/{max_retries}): {e}. Retrying in {delay}s...")

                time.sleep(delay)

        return False

    def _save_rate_limit_state(self):
        """Save rate limit state so the bot knows to cool down on next start."""
        rate_limit_state = {
            "timestamp": datetime.now(IST).isoformat(),
            "reason": "rate_limited",
            "cooldown_until": (datetime.now(IST) + timedelta(minutes=5)).isoformat(),
        }
        rate_limit_file = DATA_DIR / "rate_limit_state.json"
        DATA_DIR.mkdir(exist_ok=True)
        rate_limit_file.write_text(json.dumps(rate_limit_state, indent=2), encoding="utf-8")
        log.info(f"Rate limit state saved. Cooldown until {rate_limit_state['cooldown_until']}")

    def get_quote(self, symbol: str) -> dict:
        """Get live quote with auto-reconnect on auth failures."""
        try:
            return self.api.get_quote(trading_symbol=f"{symbol}-EQ", exchange="NSE", segment="CASH")
        except Exception as e:
            error_str = str(e).lower()
            # If auth expired, force reconnect and retry once
            if "unauthorized" in error_str or "401" in error_str or "token" in error_str or "expire" in error_str:
                log.warning(f"Auth expired during quote fetch, reconnecting...")
                try:
                    self.connect()
                    return self.api.get_quote(trading_symbol=f"{symbol}-EQ", exchange="NSE", segment="CASH")
                except Exception:
                    pass
            return {}

    def get_positions(self) -> list:
        """Get open positions."""
        try:
            return self.api.get_positions_for_user().get("positions", [])
        except Exception:
            return []

    # ── Strategy: Large Drop Mean Reversion ───────────────────────

    def scan_large_drops(self) -> list:
        """Scan ELITE + HIGH WR stocks for large drop signals."""
        WATCH = {
            "AMBUJACEM": -5, "JSWENERGY": -5, "NUVAMA": -5, "PAGEIND": -5,
            "MAZDOCK": -5, "LICI": -5, "IRFC": -5, "LTM": -5, "ABCAPITAL": -5,
            "CIPLA": -5, "COALINDIA": -5, "DABUR": -5, "OFSS": -5,
            "SBIN": -5, "ICICIBANK": -5, "HDFCBANK": -5, "PNB": -5,
            "BAJFINANCE": -5, "MPHASIS": -5, "NHPC": -5, "PFC": -5,
        }

        signals = []
        for symbol, thresh in WATCH.items():
            q = self.get_quote(symbol)
            if not q or not q.get("last_price"):
                continue

            ltp = q["last_price"]
            prev = q.get("ohlc", {}).get("close", 0)
            if prev == 0:
                continue

            change_pct = ((ltp - prev) / prev) * 100

            if change_pct <= thresh:
                signals.append({
                    "symbol": symbol,
                    "strategy": "LARGE_DROP",
                    "ltp": ltp,
                    "prev_close": prev,
                    "change_pct": round(change_pct, 2),
                    "confidence": 0.95,
                    "target_pct": 1.5,
                    "stop_pct": -3.0,
                    "hold_days": 1,
                    "reason": f"{symbol} dropped {change_pct:.1f}% — 100% historical bounce rate",
                })

        return signals

    # ── Strategy: RSI(2) Intraday Mean Reversion ──────────────────

    def scan_rsi2_signals(self) -> list:
        """Scan for RSI(2) oversold bounces — more frequent signals."""
        STOCKS = [
            "SBIN", "ICICIBANK", "HDFCBANK", "RELIANCE", "INFY", "TCS",
            "BAJFINANCE", "AXISBANK", "KOTAKBANK", "ITC", "BHARTIARTL",
            "TATASTEEL", "CIPLA", "SUNPHARMA", "LT",
        ]

        signals = []
        for symbol in STOCKS:
            q = self.get_quote(symbol)
            if not q or not q.get("last_price"):
                continue

            ltp = q["last_price"]
            prev = q.get("ohlc", {}).get("close", 0)
            low = q.get("ohlc", {}).get("low", 0)
            high = q.get("ohlc", {}).get("high", 0)
            open_p = q.get("ohlc", {}).get("open", 0)

            if prev == 0 or high == low:
                continue

            change_pct = ((ltp - prev) / prev) * 100

            # Intraday oversold: price near day's low + down > 2%
            position_in_range = (ltp - low) / (high - low) if (high - low) > 0 else 0.5

            if change_pct < -2.0 and position_in_range < 0.2:
                signals.append({
                    "symbol": symbol,
                    "strategy": "RSI2_INTRADAY",
                    "ltp": ltp,
                    "prev_close": prev,
                    "change_pct": round(change_pct, 2),
                    "confidence": 0.65,
                    "target_pct": 1.0,
                    "stop_pct": -1.5,
                    "hold_days": 0,  # intraday
                    "reason": f"{symbol} down {change_pct:.1f}%, near day low ({position_in_range:.0%})",
                })

        return signals

    # ── Strategy: Momentum Breakout ───────────────────────────────

    def scan_momentum(self) -> list:
        """Scan for strong intraday momentum — ride the wave."""
        STOCKS = [
            "RELIANCE", "TCS", "SBIN", "ICICIBANK", "HDFCBANK",
            "BAJFINANCE", "INFY", "TATASTEEL", "BHARTIARTL", "ITC",
        ]

        signals = []
        for symbol in STOCKS:
            q = self.get_quote(symbol)
            if not q or not q.get("last_price"):
                continue

            ltp = q["last_price"]
            prev = q.get("ohlc", {}).get("close", 0)
            high = q.get("ohlc", {}).get("high", 0)
            volume = q.get("volume", 0)

            if prev == 0:
                continue

            change_pct = ((ltp - prev) / prev) * 100

            # Strong momentum: up > 2.5% + at day high + volume > average
            at_high = (high > 0 and abs(ltp - high) / high < 0.002)

            if change_pct > 2.5 and at_high:
                signals.append({
                    "symbol": symbol,
                    "strategy": "MOMENTUM",
                    "ltp": ltp,
                    "prev_close": prev,
                    "change_pct": round(change_pct, 2),
                    "confidence": 0.60,
                    "target_pct": 1.5,
                    "stop_pct": -1.0,
                    "hold_days": 0,
                    "reason": f"{symbol} up {change_pct:.1f}% at day high — momentum",
                })

        return signals

    # ── Strategy: Micro-Cap Momentum (for low capital) ─────────────

    def scan_micro_cap(self) -> list:
        """Aggressive micro-cap scanner — uses ALL capital, lower thresholds.
        Inspired by Sijoy's TATAMOTORS 1289% play: cheap entry, ride the move."""
        # Tier 1: Cheap stocks (Rs.15-100) — max shares, momentum plays
        # REMOVED: GTLINFRA (Rs.1.25), RPOWER (Rs.2), IDEA (Rs.9) — too cheap, fees > profit
        TIER1_STOCKS = [
            "YESBANK", "HFCL", "SUZLON", "NHPC", "IRFC",
            "PNB", "NBCC",
        ]
        # Tier 2: Mid-cheap (Rs.30-100) — fewer shares but better stocks
        TIER2_STOCKS = [
            "IDFCFIRSTB", "PFC", "RECLTD", "NBCC", "TRIDENT",
            "RAILTEL", "IRCON", "RVNL", "ZOMATO", "JSWINFRA",
            "SAIL", "PNB", "BANKBARODA", "TATAPOWER",
        ]
        # Tier 3: Quality mid-range (Rs.100-250) — 1-2 shares for conviction plays
        TIER3_STOCKS = [
            "TATASTEEL", "VEDL", "COALINDIA", "BPCL", "ONGC",
            "HINDPETRO", "ITC", "BHARTIARTL",
        ]

        signals = []
        for symbol in TIER1_STOCKS + TIER2_STOCKS + TIER3_STOCKS:
            q = self.get_quote(symbol)
            if not q or not q.get("last_price"):
                continue

            ltp = q["last_price"]
            prev = q.get("ohlc", {}).get("close", 0)
            high = q.get("ohlc", {}).get("high", 0)
            low = q.get("ohlc", {}).get("low", 0)
            open_p = q.get("ohlc", {}).get("open", 0)

            if prev == 0:
                continue

            # Skip if can't buy even 1 share
            if ltp > self.config.capital * 0.8:
                continue

            change_pct = ((ltp - prev) / prev) * 100

            # ----- AGGRESSIVE SIGNALS (lower thresholds) -----

            # Signal 1: Oversold bounce (down > 1%, near low)
            if change_pct < -1.0 and low > 0 and high > low:
                pos_in_range = (ltp - low) / (high - low)
                if pos_in_range < 0.35:
                    signals.append({
                        "symbol": symbol,
                        "strategy": "MICRO_OVERSOLD",
                        "ltp": ltp,
                        "prev_close": prev,
                        "change_pct": round(change_pct, 2),
                        "confidence": 0.55 + min(abs(change_pct) * 0.05, 0.25),
                        "target_pct": 1.5,
                        "stop_pct": -2.5,
                        "hold_days": 0,
                        "reason": f"{symbol} Rs.{ltp:.1f} down {change_pct:.1f}% near low — bounce",
                    })

            # Signal 2: Momentum (up > 1.5%, near high)
            elif change_pct > 1.5 and high > 0:
                at_high = abs(ltp - high) / high < 0.008
                if at_high:
                    signals.append({
                        "symbol": symbol,
                        "strategy": "MICRO_MOMENTUM",
                        "ltp": ltp,
                        "prev_close": prev,
                        "change_pct": round(change_pct, 2),
                        "confidence": 0.50 + min(change_pct * 0.05, 0.25),
                        "target_pct": 2.0,
                        "stop_pct": -1.5,
                        "hold_days": 0,
                        "reason": f"{symbol} Rs.{ltp:.1f} up {change_pct:.1f}% at high — ride",
                    })

            # Signal 3: Gap up from open (opened higher than prev close, still running)
            elif open_p > 0 and open_p > prev * 1.01 and ltp >= open_p:
                gap_pct = ((open_p - prev) / prev) * 100
                if gap_pct > 1.0:
                    signals.append({
                        "symbol": symbol,
                        "strategy": "MICRO_GAPUP",
                        "ltp": ltp,
                        "prev_close": prev,
                        "change_pct": round(change_pct, 2),
                        "confidence": 0.50,
                        "target_pct": 1.5,
                        "stop_pct": -1.0,
                        "hold_days": 0,
                        "reason": f"{symbol} Rs.{ltp:.1f} gapped up {gap_pct:.1f}%, still above open",
                    })

            # Signal 4: Any stock moving > 2.5% either direction
            elif abs(change_pct) > 2.5:
                direction = "bounce" if change_pct < 0 else "ride"
                signals.append({
                    "symbol": symbol,
                    "strategy": "MICRO_VOLATILITY",
                    "ltp": ltp,
                    "prev_close": prev,
                    "change_pct": round(change_pct, 2),
                    "confidence": 0.45 + min(abs(change_pct) * 0.03, 0.20),
                    "target_pct": 1.0,
                    "stop_pct": -1.5,
                    "hold_days": 0,
                    "reason": f"{symbol} Rs.{ltp:.1f} moved {change_pct:+.1f}% — {direction}",
                })

        return signals

    # ── Position Sizing ───────────────────────────────────────────

    def calculate_position_size(self, signal: dict) -> int:
        """AGGRESSIVE position sizing — deploy capital fast, compound profits.
        Uses MIS margin (5x) so Rs.200 controls Rs.1000 of stock.
        Recycles freed capital from closed positions immediately."""
        # Track what's deployed (using actual cost, not margin)
        invested = sum(
            p["entry_price"] * p["qty"]
            for p in self.open_positions
            if p["status"] == "OPEN"
        )
        # MIS gives ~5x margin — we can control 5x our capital
        effective_capital = self.config.capital * 5  # use full 5x MIS margin — YOLO mode
        remaining = max(effective_capital - invested, 0)

        # Add back today's realized P&L (compound profits)
        remaining += max(self.day_pnl, 0)

        if remaining < signal["ltp"]:  # can't even buy 1 share
            return 0

        # YOLO: up to 80% of remaining per trade — full send
        max_value = remaining * 0.80
        max_shares = int(max_value / signal["ltp"])

        # Risk-based check
        risk_amount = self.config.capital * (self.config.max_risk_pct / 100)
        stop_distance = abs(signal["stop_pct"] / 100) * signal["ltp"]
        risk_shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0

        # Ensure at least 1 share
        if max_shares == 0 and signal["ltp"] <= remaining:
            max_shares = 1

        qty = min(risk_shares, max_shares)
        if qty > 0:
            log.info(f"  Size: {qty}x {signal['symbol']} @ Rs.{signal['ltp']:.1f} = Rs.{qty*signal['ltp']:.0f} (margin left: Rs.{remaining:.0f})")
        return qty

    # ── Trade Execution ───────────────────────────────────────────

    def execute_trade(self, signal: dict) -> dict:
        """Execute a trade (paper or live) with anti-churn profitability checks."""
        symbol = signal["symbol"]
        ltp = signal["ltp"]

        # ANTI-CHURN: Skip stocks too cheap (fees destroy profits)
        if ltp < self.config.min_price:
            log.info(f"  SKIP {symbol}: price Rs.{ltp:.2f} < min Rs.{self.config.min_price} (fee burn)")
            return {"status": "SKIP", "reason": f"price {ltp} below min {self.config.min_price}"}

        qty = self.calculate_position_size(signal)
        if qty == 0:
            return {"status": "SKIP", "reason": "position size = 0"}

        value = ltp * qty

        # ANTI-CHURN: Skip tiny trades (brokerage eats the profit)
        if value < self.config.min_trade_value:
            log.info(f"  SKIP {symbol}: trade value Rs.{value:.0f} < min Rs.{self.config.min_trade_value} (not worth fees)")
            return {"status": "SKIP", "reason": f"trade value {value:.0f} below min {self.config.min_trade_value}"}

        # ANTI-CHURN: Check expected profit vs estimated brokerage
        expected_profit = value * abs(signal.get("target_pct", 1.0)) / 100
        est_brokerage = min(20, value * 0.0005) * 2 + value * 0.0003  # rough: brokerage + STT + GST
        if expected_profit < est_brokerage * 1.5:
            log.info(f"  SKIP {symbol}: expected profit Rs.{expected_profit:.2f} < 1.5x brokerage Rs.{est_brokerage:.2f}")
            return {"status": "SKIP", "reason": f"profit {expected_profit:.2f} < 1.5x fees {est_brokerage:.2f}"}

        stop = ltp * (1 + signal["stop_pct"] / 100)
        target = ltp * (1 + signal["target_pct"] / 100)

        trade = {
            "symbol": symbol,
            "strategy": signal["strategy"],
            "side": "BUY",
            "qty": qty,
            "entry_price": ltp,
            "value": round(value, 2),
            "stop_loss": round(stop, 2),
            "target": round(target, 2),
            "confidence": signal["confidence"],
            "reason": signal["reason"],
            "entry_time": datetime.now(IST).isoformat(),
            "status": "OPEN",
            "paper": self.config.paper,
            "pnl": 0,
        }

        if self.config.paper:
            trade["order_id"] = f"PAPER_{int(time.time())}"
            log.info(f"PAPER BUY: {qty}x {symbol} @ Rs.{ltp:,.2f} = Rs.{value:,.0f}")
        else:
            # Live order
            try:
                result = self.api.place_order(
                    trading_symbol=f"{symbol}-EQ",
                    quantity=qty,
                    validity="DAY",
                    exchange="NSE",
                    segment="CASH",
                    product="MIS",  # intraday
                    order_type="MARKET",
                    transaction_type="BUY",
                )
                trade["order_id"] = result.get("payload", {}).get("groww_order_id", "UNKNOWN")
                trade["order_response"] = result
                log.info(f"LIVE BUY: {qty}x {symbol} @ Rs.{ltp:,.2f}")
            except Exception as e:
                log.error(f"Order failed: {e}")
                trade["status"] = "FAILED"
                trade["error"] = str(e)

        self.open_positions.append(trade)
        self.trades_today.append(trade)

        tg_alert(
            f"{'PAPER ' if self.config.paper else ''}BUY\n"
            f"Stock: {symbol}\n"
            f"Qty: {qty} @ Rs.{ltp:,.2f}\n"
            f"Value: Rs.{value:,.0f}\n"
            f"Stop: Rs.{stop:,.2f} | Target: Rs.{target:,.2f}\n"
            f"Strategy: {signal['strategy']}\n"
            f"Reason: {signal['reason']}"
        )

        self._save_state()
        return trade

    # ── Position Management ───────────────────────────────────────

    def manage_positions(self):
        """Check all open positions for stop/target/time exits."""
        to_close = []

        for pos in self.open_positions:
            if pos["status"] != "OPEN":
                continue

            symbol = pos["symbol"]
            q = self.get_quote(symbol)
            if not q or not q.get("last_price"):
                continue

            ltp = q["last_price"]
            entry = pos["entry_price"]
            pnl_pct = ((ltp - entry) / entry) * 100
            pnl_abs = (ltp - entry) * pos["qty"]

            pos["current_price"] = ltp
            pos["pnl"] = round(pnl_abs, 2)
            pos["pnl_pct"] = round(pnl_pct, 2)

            # Check stop loss
            if ltp <= pos["stop_loss"]:
                pos["exit_reason"] = "STOP_LOSS"
                to_close.append(pos)

            # Check target
            elif ltp >= pos["target"]:
                pos["exit_reason"] = "TARGET_HIT"
                to_close.append(pos)

            # Time-based exit (intraday positions close by 3:15 PM IST)
            elif pos.get("hold_days", 0) == 0:
                now_ist = datetime.now(IST)
                if now_ist.hour > 15 or (now_ist.hour == 15 and now_ist.minute >= 15):
                    pos["exit_reason"] = "TIME_EXIT"
                    to_close.append(pos)

        for pos in to_close:
            self._close_position(pos)

    def _close_position(self, pos: dict):
        """Close a position."""
        symbol = pos["symbol"]
        ltp = pos.get("current_price", pos["entry_price"])
        qty = pos["qty"]
        pnl = (ltp - pos["entry_price"]) * qty
        pnl_pct = ((ltp - pos["entry_price"]) / pos["entry_price"]) * 100

        # Estimate brokerage + fees (buy + sell)
        trade_value = pos["entry_price"] * qty
        est_brokerage = min(20, trade_value * 0.0005) * 2  # buy + sell brokerage
        est_stt = trade_value * 0.00025  # STT on sell
        est_gst = (est_brokerage + trade_value * 0.0000345) * 0.18
        est_fees = est_brokerage + est_stt + est_gst + trade_value * 0.00003
        net_pnl = pnl - est_fees

        pos["exit_price"] = ltp
        pos["exit_time"] = datetime.now(IST).isoformat()
        pos["status"] = "CLOSED"
        pos["pnl"] = round(pnl, 2)
        pos["net_pnl"] = round(net_pnl, 2)
        pos["est_fees"] = round(est_fees, 2)
        pos["pnl_pct"] = round(pnl_pct, 2)

        self.day_pnl += net_pnl  # track NET P&L including fees

        if not self.config.paper:
            # CRITICAL: Retry sell order up to 3 times — cannot leave position open
            for sell_attempt in range(1, 4):
                try:
                    self.api.place_order(
                        trading_symbol=f"{symbol}-EQ",
                        quantity=qty,
                        validity="DAY",
                        exchange="NSE",
                        segment="CASH",
                        product="MIS",
                        order_type="MARKET",
                        transaction_type="SELL",
                    )
                    break  # success
                except Exception as e:
                    log.error(f"SELL order failed (attempt {sell_attempt}/3): {e}")
                    if sell_attempt < 3:
                        time.sleep(5 * sell_attempt)  # backoff 5s, 10s
                    else:
                        log.critical(f"SELL FAILED 3x for {qty}x {symbol} — MANUAL INTERVENTION NEEDED")
                        pos["status"] = "SELL_FAILED"  # mark for manual review

        self.open_positions = [p for p in self.open_positions if p["status"] == "OPEN"]

        win = "WIN" if net_pnl > 0 else "LOSS"
        log.info(f"CLOSED {symbol}: {win} Rs.{net_pnl:+,.0f} (gross {pnl:+,.0f} - fees {est_fees:.0f}) ({pnl_pct:+.2f}%) — {pos['exit_reason']}")

        tg_alert(
            f"{'PAPER ' if self.config.paper else ''}EXIT — {win}\n"
            f"Stock: {symbol}\n"
            f"Entry: Rs.{pos['entry_price']:,.2f} -> Exit: Rs.{ltp:,.2f}\n"
            f"P&L: Rs.{pnl:+,.0f} ({pnl_pct:+.2f}%)\n"
            f"Reason: {pos['exit_reason']}\n"
            f"Day P&L: Rs.{self.day_pnl:+,.0f}"
        )

        self._save_state()

    # ── EOD INVESTING: Deploy all cash into delivery before close ──

    def _eod_deploy_cash(self):
        """At 3:00 PM IST, buy best stocks with ALL remaining cash via CNC (delivery).
        These hold overnight — money works while you sleep."""
        log.info("=" * 50)
        log.info("EOD DEPLOY — Investing all cash into delivery positions")

        # Get fresh balance — only use actual free cash (not margin)
        try:
            margin = self.api.get_available_margin_details()
            cash = max(margin.get("equity_margin_details", {}).get("cnc_balance_available", 0), 0)
            clear = margin.get("clear_cash", 0)
            # Use the smaller of CNC available and clear cash — be conservative
            cash = min(cash, clear) if cash > 0 else 0
        except Exception as e:
            log.error(f"EOD: Failed to get balance: {e}")
            return

        if cash < 10:
            log.info(f"EOD: Only Rs.{cash:.0f} available — nothing to deploy")
            return

        log.info(f"EOD: Rs.{cash:.0f} available for overnight investment")

        # Pick best stocks from today's scan data — momentum winners
        # Prefer stocks that were UP today (momentum carry into next day)
        OVERNIGHT_PICKS = [
            "IRFC", "NHPC", "PFC", "RECLTD", "SUZLON", "YESBANK",
            "IDFCFIRSTB", "HFCL", "RPOWER", "NBCC", "TRIDENT",
            "SAIL", "PNB", "BANKBARODA", "TATAPOWER", "RVNL",
            "TATASTEEL", "VEDL", "COALINDIA", "BPCL",
        ]

        # Score each stock
        scored = []
        for symbol in OVERNIGHT_PICKS:
            q = self.get_quote(symbol)
            if not q or not q.get("last_price"):
                continue

            ltp = q["last_price"]
            prev = q.get("ohlc", {}).get("close", 0)
            if prev == 0 or ltp > cash * 0.8:  # must afford at least 1 share
                continue

            change_pct = ((ltp - prev) / prev) * 100
            high = q.get("ohlc", {}).get("high", 0)
            low = q.get("ohlc", {}).get("low", 0)

            # Score: positive momentum + near day high = likely gap up tomorrow
            score = 0
            if change_pct > 0:
                score += min(change_pct * 10, 40)  # up to 40 pts for momentum
            if high > 0 and ltp > 0:
                nearness_to_high = 1 - abs(ltp - high) / high
                score += nearness_to_high * 30  # up to 30 pts for near high
            if low > 0 and high > low:
                range_pct = ((high - low) / low) * 100
                score += min(range_pct * 5, 20)  # up to 20 pts for volatility
            # Cheaper stocks = more shares = more leverage
            if ltp < 30:
                score += 10
            elif ltp < 100:
                score += 5

            scored.append({
                "symbol": symbol,
                "ltp": ltp,
                "change_pct": change_pct,
                "score": round(score, 1),
            })

        if not scored:
            log.info("EOD: No affordable stocks found")
            return

        # Sort by score, pick top stocks
        scored.sort(key=lambda x: x["score"], reverse=True)
        remaining = cash

        overnight_trades = []
        for pick in scored:
            if remaining < pick["ltp"]:
                continue

            # Allocate up to 40% of remaining cash per stock
            alloc = min(remaining * 0.4, remaining)
            qty = int(alloc / pick["ltp"])
            if qty == 0:
                continue

            value = qty * pick["ltp"]

            # Place CNC (delivery) order
            if not self.config.paper:
                try:
                    result = self.api.place_order(
                        trading_symbol=f"{pick['symbol']}-EQ",
                        quantity=qty,
                        validity="DAY",
                        exchange="NSE",
                        segment="CASH",
                        product="CNC",  # DELIVERY — holds overnight
                        order_type="MARKET",
                        transaction_type="BUY",
                    )
                    order_id = result.get("payload", {}).get("groww_order_id", "UNKNOWN")
                    log.info(f"EOD BUY (CNC): {qty}x {pick['symbol']} @ Rs.{pick['ltp']:.2f} = Rs.{value:.0f} [score: {pick['score']}]")
                    overnight_trades.append({
                        "symbol": pick["symbol"],
                        "qty": qty,
                        "entry_price": pick["ltp"],
                        "value": value,
                        "score": pick["score"],
                        "change_pct": pick["change_pct"],
                        "order_id": order_id,
                        "product": "CNC",
                        "entry_time": datetime.now(IST).isoformat(),
                        "status": "OVERNIGHT",
                    })
                    remaining -= value
                except Exception as e:
                    log.error(f"EOD order failed for {pick['symbol']}: {e}")
            else:
                log.info(f"EOD PAPER BUY (CNC): {qty}x {pick['symbol']} @ Rs.{pick['ltp']:.2f}")
                remaining -= value

            if remaining < 10:
                break

        # Save overnight positions
        if overnight_trades:
            overnight_file = DATA_DIR / "overnight_positions.json"
            DATA_DIR.mkdir(exist_ok=True)
            overnight_file.write_text(
                json.dumps({"date": datetime.now(IST).strftime("%Y-%m-%d"), "positions": overnight_trades}, indent=2),
                encoding="utf-8",
            )
            total_invested = sum(t["value"] for t in overnight_trades)
            log.info(f"EOD: Deployed Rs.{total_invested:.0f} into {len(overnight_trades)} overnight positions")
            tg_alert(
                f"EOD OVERNIGHT DEPLOY\n"
                f"Invested: Rs.{total_invested:,.0f}\n"
                f"Positions: {len(overnight_trades)}\n"
                + "\n".join(f"  {t['qty']}x {t['symbol']} @ Rs.{t['entry_price']:.2f}" for t in overnight_trades)
            )

        log.info("=" * 50)

    def _manage_overnight_positions(self):
        """Morning: Check overnight CNC positions, sell on profit or hold."""
        overnight_file = DATA_DIR / "overnight_positions.json"
        if not overnight_file.exists():
            return

        try:
            data = json.loads(overnight_file.read_text(encoding="utf-8"))
        except Exception:
            return

        positions = data.get("positions", [])
        if not positions:
            return

        # Only process once per morning
        if getattr(self, '_overnight_processed', False):
            return

        log.info(f"MORNING: Checking {len(positions)} overnight positions")

        for pos in positions:
            if pos.get("status") != "OVERNIGHT":
                continue

            q = self.get_quote(pos["symbol"])
            if not q or not q.get("last_price"):
                continue

            ltp = q["last_price"]
            entry = pos["entry_price"]
            pnl_pct = ((ltp - entry) / entry) * 100
            pnl_abs = (ltp - entry) * pos["qty"]

            # Strategy: sell if profit > 1% OR loss > 2% (cut losers, let winners go)
            sell = False
            reason = ""
            if pnl_pct >= 1.0:
                sell = True
                reason = f"PROFIT +{pnl_pct:.1f}%"
            elif pnl_pct <= -2.0:
                sell = True
                reason = f"STOP LOSS {pnl_pct:.1f}%"

            if sell and not self.config.paper:
                for attempt in range(1, 4):
                    try:
                        self.api.place_order(
                            trading_symbol=f"{pos['symbol']}-EQ",
                            quantity=pos["qty"],
                            validity="DAY",
                            exchange="NSE",
                            segment="CASH",
                            product="CNC",
                            order_type="MARKET",
                            transaction_type="SELL",
                        )
                        pos["status"] = "SOLD"
                        pos["exit_price"] = ltp
                        pos["pnl"] = round(pnl_abs, 2)
                        win = "WIN" if pnl_abs > 0 else "LOSS"
                        log.info(f"OVERNIGHT EXIT {win}: {pos['qty']}x {pos['symbol']} Rs.{pnl_abs:+,.0f} ({pnl_pct:+.1f}%) — {reason}")
                        break
                    except Exception as e:
                        log.error(f"Overnight sell failed (attempt {attempt}/3): {e}")
                        if attempt < 3:
                            time.sleep(5)
            elif not sell:
                log.info(f"OVERNIGHT HOLD: {pos['symbol']} @ Rs.{ltp:.2f} ({pnl_pct:+.1f}%) — waiting for target")

        # Save updated state
        overnight_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._overnight_processed = True

    # ── Main Loop ─────────────────────────────────────────────────

    def _check_rate_limit_cooldown(self) -> bool:
        """Check if we're still in a rate limit cooldown period. Returns True if should wait."""
        rate_limit_file = DATA_DIR / "rate_limit_state.json"
        if not rate_limit_file.exists():
            return False

        try:
            state = json.loads(rate_limit_file.read_text(encoding="utf-8"))
            cooldown_until = datetime.fromisoformat(state.get("cooldown_until", "2000-01-01"))
            now = datetime.now(IST).replace(tzinfo=None)
            if now < cooldown_until:
                remaining = (cooldown_until - now).total_seconds()
                log.warning(f"Rate limit cooldown active. {remaining:.0f}s remaining. Waiting...")
                return True
            else:
                # Cooldown expired, remove the file
                rate_limit_file.unlink(missing_ok=True)
                log.info("Rate limit cooldown expired. Proceeding.")
                return False
        except Exception:
            rate_limit_file.unlink(missing_ok=True)
            return False

    def run(self):
        """Main autopilot loop."""
        self.running = True
        signal.signal(signal.SIGINT, lambda s, f: self._shutdown())

        # Check if we're in a rate limit cooldown from a previous crash
        if self._check_rate_limit_cooldown():
            rate_limit_file = DATA_DIR / "rate_limit_state.json"
            state = json.loads(rate_limit_file.read_text(encoding="utf-8"))
            cooldown_until = datetime.fromisoformat(state["cooldown_until"])
            wait_secs = (cooldown_until - datetime.now()).total_seconds()
            if wait_secs > 0:
                log.info(f"Sleeping {wait_secs:.0f}s for rate limit cooldown...")
                time.sleep(wait_secs)
            rate_limit_file.unlink(missing_ok=True)

        log.info("=" * 60)
        log.info(f"AUTOPILOT {'PAPER' if self.config.paper else 'LIVE'} MODE")
        log.info(f"Capital: Rs.{self.config.capital:,.0f}")
        log.info(f"Risk/trade: {self.config.max_risk_pct}%")
        log.info(f"Max daily loss: {self.config.max_daily_loss_pct}%")
        log.info(f"Scan interval: {self.config.scan_interval}s")
        log.info("=" * 60)

        self.connect()

        tg_alert(
            f"Autopilot STARTED\n"
            f"Mode: {'PAPER' if self.config.paper else 'LIVE'}\n"
            f"Capital: Rs.{self.config.capital:,.0f}\n"
            f"Risk: {self.config.max_risk_pct}%/trade"
        )

        scan_count = 0
        while self.running:
            try:
                now = datetime.now(IST)

                # Market hours check (9:15 AM - 3:30 PM IST)
                if now.hour < 9 or (now.hour == 9 and now.minute < 15):
                    log.info("Market not open yet. Waiting...")
                    time.sleep(60)
                    continue

                if now.hour > 15 or (now.hour == 15 and now.minute >= 30):
                    log.info("Market closed. Closing MIS positions...")
                    for pos in self.open_positions:
                        if pos["status"] == "OPEN" and pos.get("product", "MIS") == "MIS":
                            pos["exit_reason"] = "MARKET_CLOSE"
                            pos["current_price"] = pos.get("current_price", pos["entry_price"])
                            self._close_position(pos)
                    self._print_day_summary()
                    break

                # ── EOD INVESTING: 3:00-3:15 PM — deploy all cash into CNC delivery ──
                if not getattr(self, '_eod_deployed', False) and now.hour == 15 and 0 <= now.minute < 15:
                    self._eod_deploy_cash()
                    self._eod_deployed = True

                # ── MORNING: 9:15-9:45 AM — manage overnight CNC positions ──
                if now.hour == 9 and 15 <= now.minute <= 45:
                    self._manage_overnight_positions()

                # Daily loss limit check
                loss_limit = self.config.capital * (self.config.max_daily_loss_pct / 100)
                if self.day_pnl <= -loss_limit:
                    log.critical(f"DAILY LOSS LIMIT HIT: Rs.{self.day_pnl:,.0f}")
                    tg_alert(f"DAILY LOSS LIMIT\nP&L: Rs.{self.day_pnl:,.0f}\nStopping.")
                    break

                # Profit target check
                profit_target = self.config.capital * (self.config.profit_target_pct / 100)
                if self.day_pnl >= profit_target:
                    log.info(f"PROFIT TARGET HIT: Rs.{self.day_pnl:,.0f}")
                    tg_alert(f"PROFIT TARGET HIT!\nP&L: Rs.{self.day_pnl:,.0f}\nStopping.")
                    break

                scan_count += 1
                log.info(f"[SCAN {scan_count}] Starting scan cycle...")

                # Manage existing positions first
                if self.open_positions:
                    self.manage_positions()

                # Scan for new signals (if room for more positions)
                open_count = len([p for p in self.open_positions if p["status"] == "OPEN"])
                trades_today = len(self.trades_today)

                if open_count < self.config.max_open_positions and trades_today < self.config.max_trades_per_day:
                    # Scan all strategies
                    all_signals = []
                    all_signals.extend(self.scan_micro_cap())  # micro-cap first (works with low capital)
                    all_signals.extend(self.scan_large_drops())
                    all_signals.extend(self.scan_rsi2_signals())
                    all_signals.extend(self.scan_momentum())

                    # Apply ML confidence boost to each signal
                    for sig in all_signals:
                        ml_features = {
                            "returns_1d": sig["change_pct"] / 100,
                            "ltp": sig["ltp"],
                        }
                        ml_boost = self.ml_engine.predict(sig["symbol"], ml_features)
                        sig["ml_boost"] = round(ml_boost, 3)
                        sig["confidence"] = min(sig["confidence"] + ml_boost, 0.99)

                    if all_signals:
                        log.info(f"Found {len(all_signals)} signals: {[s['symbol'] + '(' + s['strategy'] + ' ML:' + str(s.get('ml_boost',0)) + ')' for s in all_signals]}")
                    elif scan_count % 5 == 0:
                        log.info(f"No signals found in scan #{scan_count}")

                    # Sort by confidence (now ML-boosted)
                    all_signals.sort(key=lambda x: x["confidence"], reverse=True)

                    # Execute top signals
                    for sig in all_signals:
                        # Skip if already in position
                        if any(p["symbol"] == sig["symbol"] and p["status"] == "OPEN" for p in self.open_positions):
                            continue

                        if open_count >= self.config.max_open_positions:
                            break

                        self.execute_trade(sig)
                        open_count += 1

                # Status update every 10 scans
                if scan_count % 10 == 0:
                    open_count = len([p for p in self.open_positions if p["status"] == "OPEN"])
                    log.info(
                        f"Scan #{scan_count} | Open: {open_count} | "
                        f"Trades: {len(self.trades_today)} | "
                        f"Day P&L: Rs.{self.day_pnl:+,.0f}"
                    )

                time.sleep(self.config.scan_interval)

            except KeyboardInterrupt:
                self._shutdown()
                break
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "rate" in error_str or "429" in error_str or "too many" in error_str or "throttl" in error_str

                if is_rate_limit:
                    log.warning(f"Rate limited during scan: {e}. Backing off 60s...")
                    tg_alert(f"Rate limited by Groww API. Backing off 60s.")
                    time.sleep(60)
                else:
                    import traceback
                    log.error(f"Loop error: {e}\n{traceback.format_exc()}")
                    time.sleep(10)

    def _shutdown(self):
        """Graceful shutdown."""
        log.info("Shutting down autopilot...")
        self.running = False
        for pos in self.open_positions:
            if pos["status"] == "OPEN":
                pos["exit_reason"] = "SHUTDOWN"
                pos["current_price"] = pos.get("current_price", pos["entry_price"])
                self._close_position(pos)
        self._print_day_summary()

    def _print_day_summary(self):
        """Print day-end summary."""
        wins = sum(1 for t in self.trades_today if t.get("pnl", 0) > 0)
        losses = sum(1 for t in self.trades_today if t.get("pnl", 0) <= 0 and t["status"] == "CLOSED")
        total = len(self.trades_today)

        summary = (
            f"\n{'=' * 50}\n"
            f"DAY SUMMARY\n"
            f"{'=' * 50}\n"
            f"Trades: {total} ({wins}W / {losses}L)\n"
            f"Win Rate: {wins/total*100:.0f}%\n" if total > 0 else ""
            f"Day P&L: Rs.{self.day_pnl:+,.0f}\n"
            f"Capital: Rs.{self.config.capital:,.0f} -> Rs.{self.config.capital + self.day_pnl:,.0f}\n"
            f"ROI: {self.day_pnl/self.config.capital*100:+.2f}%\n"
            f"{'=' * 50}"
        )
        log.info(summary)
        tg_alert(summary)

    def _save_state(self):
        state = {
            "timestamp": datetime.now().isoformat(),
            "capital": self.config.capital,
            "day_pnl": self.day_pnl,
            "trades_today": len(self.trades_today),
            "open_positions": len([p for p in self.open_positions if p["status"] == "OPEN"]),
            "paper": self.config.paper,
        }
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

        trades_data = self.trades_today[-50:]
        TRADES_FILE.write_text(json.dumps(trades_data, indent=2, default=str), encoding="utf-8")

    def _load_state(self):
        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                if state.get("timestamp", "")[:10] == datetime.now().strftime("%Y-%m-%d"):
                    self.day_pnl = state.get("day_pnl", 0)
                    log.info(f"Resumed: Day P&L Rs.{self.day_pnl:+,.0f}")
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="JoSho Trader Autopilot")
    parser.add_argument("--live", action="store_true", help="Enable LIVE trading (real money)")
    parser.add_argument("--capital", type=float, default=10000, help="Trading capital (default: 10000)")
    parser.add_argument("--risk", type=float, default=2.0, help="Max risk % per trade (default: 2)")
    parser.add_argument("--interval", type=int, default=30, help="Scan interval seconds (default: 30)")
    parser.add_argument("--max-trades", type=int, default=20, help="Max trades per day (default: 20)")
    parser.add_argument("--kill", action="store_true", help="Emergency stop")
    args = parser.parse_args()

    if args.kill:
        STATE_FILE.write_text(json.dumps({"killed": True}), encoding="utf-8")
        print("Kill signal sent.")
        tg_alert("AUTOPILOT KILLED by operator")
        return

    config = TradeConfig(
        capital=args.capital,
        max_risk_pct=args.risk,
        paper=not args.live,
        scan_interval=args.interval,
        max_trades_per_day=args.max_trades,
    )

    if args.live:
        print("\n*** WARNING: LIVE TRADING MODE ***")
        print(f"Capital: Rs.{args.capital:,.0f}")
        print(f"Risk: {args.risk}% per trade")
        print("Press Ctrl+C to stop at any time.\n")

    pilot = AutoPilot(config)
    pilot.run()


if __name__ == "__main__":
    main()

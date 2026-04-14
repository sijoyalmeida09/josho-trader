"""
MTM (Mark-to-Market) Monitor — Real-time P&L tracking.
Integrated from: algo_trading_strategies_india (EnhancedMTMMonitor).

Features:
  - Real-time position P&L monitoring (every 3 seconds)
  - Combined premium trailing stop loss
  - Auto square-off at configured time
  - Volume freeze handling (splits large orders)
  - Daily profit target with discipline mode
  - Breakeven adjustment when one leg SL hits
"""

import logging
import json
import time
import threading
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Optional, Callable

log = logging.getLogger("josho.mtm")

# NSE volume freeze limits per instrument
FREEZE_LIMITS = {
    "NIFTY": 1800,
    "BANKNIFTY": 900,
    "FINNIFTY": 1800,
    "MIDCPNIFTY": 2800,
    "SENSEX": 600,
}


class MTMMonitor:
    """
    Real-time Mark-to-Market monitoring with auto-actions.
    Runs in a background thread, checks every N seconds.
    """

    def __init__(
        self,
        get_positions_fn: Callable,
        exit_fn: Callable,
        check_interval: int = 3,
        max_loss: float = 5000,
        profit_target: float = 10000,
        trailing_sl_pct: float = 30,
        auto_squareoff_time: dt_time = dt_time(15, 15),
        discipline_mode: bool = True,
    ):
        self.get_positions = get_positions_fn
        self.exit_fn = exit_fn
        self.check_interval = check_interval
        self.max_loss = max_loss
        self.profit_target = profit_target
        self.trailing_sl_pct = trailing_sl_pct
        self.auto_squareoff_time = auto_squareoff_time
        self.discipline_mode = discipline_mode

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._peak_pnl = 0.0
        self._trailing_sl = -max_loss
        self._target_hit = False

        self.state_file = Path(__file__).parent.parent.parent / "data" / "mtm_state.json"

    def start(self):
        """Start MTM monitoring in background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        log.info(
            f"MTM Monitor started | Max loss: Rs.{self.max_loss} | "
            f"Target: Rs.{self.profit_target} | Check: {self.check_interval}s"
        )

    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        log.info("MTM Monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._check()
            except Exception as e:
                log.error(f"MTM check error: {e}")
            time.sleep(self.check_interval)

    def _check(self):
        """Single monitoring check."""
        now = datetime.now().time()

        # Auto square-off time check
        if now >= self.auto_squareoff_time:
            log.warning("Auto square-off time reached — exiting all positions")
            self._exit_all("Auto square-off at market close")
            self._running = False
            return

        # Get current positions P&L
        positions = self.get_positions()
        if not positions:
            return

        total_pnl = sum(p.get("pnl", 0) for p in positions)

        # Update peak P&L and trailing SL
        if total_pnl > self._peak_pnl:
            self._peak_pnl = total_pnl
            # Trail the stop: lock in profits
            self._trailing_sl = self._peak_pnl * (1 - self.trailing_sl_pct / 100)

        # Max loss hit
        if total_pnl <= -self.max_loss:
            log.critical(f"MAX LOSS HIT: Rs.{total_pnl:.0f} — exiting all")
            self._exit_all(f"Max loss Rs.{self.max_loss} breached")
            self._running = False
            return

        # Trailing SL hit (only if we were in profit)
        if self._peak_pnl > 0 and total_pnl <= self._trailing_sl:
            log.warning(
                f"Trailing SL hit: PnL Rs.{total_pnl:.0f} < SL Rs.{self._trailing_sl:.0f} "
                f"(peak was Rs.{self._peak_pnl:.0f})"
            )
            self._exit_all(f"Trailing SL: peak Rs.{self._peak_pnl:.0f}, current Rs.{total_pnl:.0f}")
            self._running = False
            return

        # Profit target hit
        if total_pnl >= self.profit_target and not self._target_hit:
            self._target_hit = True
            log.info(f"PROFIT TARGET HIT: Rs.{total_pnl:.0f}")
            if self.discipline_mode:
                self._exit_all(f"Profit target Rs.{self.profit_target} hit — discipline exit")
                self._running = False
                return
            else:
                # Tighten trailing SL to lock profits
                self._trailing_sl = total_pnl * 0.8
                log.info(f"Target hit, trailing SL tightened to Rs.{self._trailing_sl:.0f}")

        # Save state
        self._save_state(total_pnl, positions)

    def _exit_all(self, reason: str):
        """Exit all positions with volume freeze handling."""
        positions = self.get_positions()
        for pos in positions:
            symbol = pos.get("symbol", "")
            qty = abs(pos.get("quantity", 0))

            if qty == 0:
                continue

            # Volume freeze handling — split large orders
            freeze_limit = FREEZE_LIMITS.get(
                symbol.split("2")[0] if "2" in symbol else symbol,
                1800,
            )

            while qty > 0:
                order_qty = min(qty, freeze_limit)
                side = "SELL" if pos.get("quantity", 0) > 0 else "BUY"
                self.exit_fn(symbol, order_qty, side, reason)
                qty -= order_qty

    def _save_state(self, pnl: float, positions: list):
        """Persist MTM state."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "total_pnl": pnl,
            "peak_pnl": self._peak_pnl,
            "trailing_sl": self._trailing_sl,
            "target_hit": self._target_hit,
            "position_count": len(positions),
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "peak_pnl": self._peak_pnl,
            "trailing_sl": self._trailing_sl,
            "target_hit": self._target_hit,
        }

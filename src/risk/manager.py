"""
Risk Manager — Position sizing, max loss, kill switch.
The most important module. This keeps you alive.
"""

import os
import json
import logging
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, field

log = logging.getLogger("josho.risk")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RiskLimits:
    """Daily risk parameters."""
    max_loss_per_day: float = float(os.environ.get("MAX_LOSS_PER_DAY", 5000))
    max_position_size: float = float(os.environ.get("MAX_POSITION_SIZE", 50000))
    max_open_positions: int = int(os.environ.get("MAX_OPEN_POSITIONS", 5))
    max_loss_per_trade: float = 2000  # hard stop per trade
    paper_trading: bool = os.environ.get("PAPER_TRADING", "true").lower() == "true"


@dataclass
class DayState:
    """Tracks P&L and position count for the day."""
    date: str = ""
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades_taken: int = 0
    open_positions: int = 0
    peak_pnl: float = 0.0
    max_drawdown: float = 0.0
    killed: bool = False
    kill_reason: str = ""


class RiskManager:
    """
    Enforces risk rules before every order.
    If any rule is violated, the order is REJECTED.
    """

    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        self.state = DayState(date=date.today().isoformat())
        self._load_state()
        log.info(
            f"Risk manager active | Max loss: ₹{self.limits.max_loss_per_day} | "
            f"Max position: ₹{self.limits.max_position_size} | "
            f"Paper: {self.limits.paper_trading}"
        )

    def can_trade(self, order_value: float, is_new_position: bool = True) -> tuple[bool, str]:
        """
        Check if a trade is allowed under current risk rules.
        Returns (allowed: bool, reason: str).
        """
        # Reset state if new day
        if self.state.date != date.today().isoformat():
            self._reset_day()

        # Kill switch active
        if self.state.killed:
            return False, f"KILL SWITCH: {self.state.kill_reason}"

        # Max daily loss breached
        if self.state.realized_pnl <= -self.limits.max_loss_per_day:
            self._kill(f"Daily loss limit hit: ₹{self.state.realized_pnl}")
            return False, f"Daily loss limit ₹{self.limits.max_loss_per_day} breached"

        # Max open positions
        if is_new_position and self.state.open_positions >= self.limits.max_open_positions:
            return False, f"Max {self.limits.max_open_positions} open positions reached"

        # Max position size
        if order_value > self.limits.max_position_size:
            return False, f"Order ₹{order_value} exceeds max ₹{self.limits.max_position_size}"

        # Paper trading check
        if self.limits.paper_trading:
            return True, "PAPER MODE — order will be simulated"

        return True, "OK"

    def record_trade(self, pnl: float, is_open: bool = True):
        """Record a trade's P&L."""
        self.state.trades_taken += 1
        if not is_open:
            self.state.realized_pnl += pnl
            self.state.open_positions = max(0, self.state.open_positions - 1)
        else:
            self.state.open_positions += 1

        # Track peak and drawdown
        total = self.state.realized_pnl + self.state.unrealized_pnl
        if total > self.state.peak_pnl:
            self.state.peak_pnl = total
        dd = self.state.peak_pnl - total
        if dd > self.state.max_drawdown:
            self.state.max_drawdown = dd

        self._save_state()

        # Auto-kill on excessive drawdown
        if self.state.realized_pnl <= -self.limits.max_loss_per_day:
            self._kill(f"Daily loss limit: ₹{self.state.realized_pnl:.0f}")

    def update_unrealized(self, unrealized: float):
        """Update unrealized P&L from live positions."""
        self.state.unrealized_pnl = unrealized
        total = self.state.realized_pnl + unrealized
        if total <= -(self.limits.max_loss_per_day * 1.5):
            self._kill(f"Total drawdown critical: ₹{total:.0f}")

    def get_status(self) -> dict:
        """Get current risk status."""
        return {
            "date": self.state.date,
            "realized_pnl": self.state.realized_pnl,
            "unrealized_pnl": self.state.unrealized_pnl,
            "total_pnl": self.state.realized_pnl + self.state.unrealized_pnl,
            "trades": self.state.trades_taken,
            "open_positions": self.state.open_positions,
            "max_drawdown": self.state.max_drawdown,
            "killed": self.state.killed,
            "kill_reason": self.state.kill_reason,
            "paper": self.limits.paper_trading,
            "remaining_loss_budget": self.limits.max_loss_per_day + self.state.realized_pnl,
        }

    def _kill(self, reason: str):
        """Activate kill switch — no more trades today."""
        self.state.killed = True
        self.state.kill_reason = reason
        self._save_state()
        log.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def force_kill(self, reason: str = "Manual kill"):
        """Manually activate kill switch."""
        self._kill(reason)

    def reset_kill(self):
        """Reset kill switch (use with caution)."""
        self.state.killed = False
        self.state.kill_reason = ""
        self._save_state()
        log.warning("Kill switch reset")

    def _reset_day(self):
        """Start a fresh day."""
        old = self.state
        self.state = DayState(date=date.today().isoformat())
        log.info(f"New trading day. Previous: ₹{old.realized_pnl:.0f} P&L, {old.trades_taken} trades")
        self._save_state()

    def _save_state(self):
        path = DATA_DIR / "risk_state.json"
        path.write_text(json.dumps(vars(self.state), indent=2), encoding="utf-8")

    def _load_state(self):
        path = DATA_DIR / "risk_state.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("date") == date.today().isoformat():
                    self.state = DayState(**data)
                    log.info(f"Resumed day state: ₹{self.state.realized_pnl:.0f} P&L")
            except Exception:
                pass

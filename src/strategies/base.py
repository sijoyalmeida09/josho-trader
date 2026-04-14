"""
Strategy Base — Abstract base for all trading strategies.
Every strategy produces Signals that the executor evaluates.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"
    HOLD = "HOLD"
    EXIT = "EXIT"


@dataclass
class Signal:
    """A trading signal produced by a strategy."""
    signal_type: SignalType
    symbol: str
    exchange: str = "NSE"
    segment: str = "FNO"
    product: str = "NRML"
    quantity: int = 0
    price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    confidence: float = 0.0  # 0-1, how confident the strategy is
    reason: str = ""
    strategy_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    @property
    def risk_reward(self) -> float:
        if self.stop_loss == 0 or self.price == 0:
            return 0
        risk = abs(self.price - self.stop_loss)
        reward = abs(self.target - self.price) if self.target else 0
        return reward / risk if risk > 0 else 0


class Strategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.enabled = True
        self.signals_generated = 0

    @abstractmethod
    def analyze(self, market_data: dict) -> list[Signal]:
        """
        Analyze market data and return list of signals.
        Must be implemented by each strategy.
        """
        ...

    @abstractmethod
    def get_required_symbols(self) -> list[str]:
        """Return list of symbols this strategy needs data for."""
        ...

    def __repr__(self):
        return f"<Strategy: {self.name} | enabled={self.enabled}>"

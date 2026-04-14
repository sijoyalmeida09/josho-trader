"""
Backtesting Engine — Simulate strategy execution on historical OHLCV data.

Processes each candle through a Strategy, tracks positions, and computes
performance metrics: total return, CAGR, Sharpe, max drawdown, win rate,
profit factor, avg win/loss, monthly returns, and growth-of-Rs.10,000 series.

F&O defaults: 0.03% commission, 0.05% slippage.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from src.strategies.base import Strategy, Signal, SignalType


# ── Configuration ────────────────────────────────────────────────

@dataclass(frozen=True)
class BacktestConfig:
    """Immutable backtest configuration."""
    initial_capital: float = 10_000.0
    commission_pct: float = 0.03    # 0.03% for F&O
    slippage_pct: float = 0.05      # 0.05% slippage
    position_size_pct: float = 10.0  # % of capital per trade
    max_positions: int = 5
    risk_free_rate: float = 6.5     # India 10Y ~6.5%


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


# ── Internal Position Tracking ───────────────────────────────────

@dataclass
class _Position:
    """Internal mutable position tracker (not exposed outside engine)."""
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: int
    entry_time: datetime
    stop_loss: float = 0.0
    target: float = 0.0


@dataclass
class _Trade:
    """Completed trade record."""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: int
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    commission: float
    reason: str


# ── Result Container ─────────────────────────────────────────────

@dataclass(frozen=True)
class BacktestResult:
    """Immutable backtest result with all metrics and series data."""
    # Core metrics
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int

    # Series data (for plotting)
    equity_curve: pd.Series        # daily portfolio value
    growth_of_10k: pd.Series       # growth of Rs.10,000
    drawdown_series: pd.Series     # running drawdown %
    daily_returns: pd.Series       # daily % returns
    monthly_returns: pd.DataFrame  # year x month pivot

    # Trade log
    trades: list[dict]

    # Config used
    config: BacktestConfig


# ── Backtesting Engine ───────────────────────────────────────────

class BacktestEngine:
    """
    Event-driven backtester.

    Usage:
        engine = BacktestEngine(strategy, df, config)
        result = engine.run()
    """

    def __init__(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        config: Optional[BacktestConfig] = None,
    ):
        _validate_dataframe(data)
        self._strategy = strategy
        self._data = data.copy()
        self._config = config or BacktestConfig()

        # State — reset on each run
        self._capital: float = 0.0
        self._positions: dict[str, _Position] = {}
        self._trades: list[_Trade] = []
        self._equity_history: list[tuple[datetime, float]] = []

    # ── Public API ───────────────────────────────────────────────

    def run(self) -> BacktestResult:
        """Execute the backtest and return immutable results."""
        self._reset()

        for i in range(1, len(self._data)):
            window = self._data.iloc[: i + 1]
            current_bar = self._data.iloc[i]
            bar_time = (
                current_bar.name
                if isinstance(current_bar.name, datetime)
                else datetime.now()
            )

            # Check stop-loss / target hits on current bar
            self._check_exits(current_bar, bar_time)

            # Ask strategy for signals
            market_data = _bar_window_to_market_data(window, self._strategy)
            signals = self._strategy.analyze(market_data)

            for signal in signals:
                self._process_signal(signal, current_bar, bar_time)

            # Record equity
            portfolio_value = self._portfolio_value(current_bar)
            self._equity_history.append((bar_time, portfolio_value))

        # Close any remaining positions at last bar
        last_bar = self._data.iloc[-1]
        last_time = (
            last_bar.name
            if isinstance(last_bar.name, datetime)
            else datetime.now()
        )
        self._close_all_positions(last_bar, last_time, reason="backtest_end")

        return self._compile_results()

    # ── Signal Processing ────────────────────────────────────────

    def _process_signal(
        self, signal: Signal, bar: pd.Series, bar_time: datetime
    ) -> None:
        symbol = signal.symbol

        if signal.signal_type == SignalType.BUY:
            if symbol not in self._positions and len(self._positions) < self._config.max_positions:
                self._open_position(
                    symbol, PositionSide.LONG, bar, bar_time, signal
                )

        elif signal.signal_type == SignalType.SHORT:
            if symbol not in self._positions and len(self._positions) < self._config.max_positions:
                self._open_position(
                    symbol, PositionSide.SHORT, bar, bar_time, signal
                )

        elif signal.signal_type in (SignalType.SELL, SignalType.COVER, SignalType.EXIT):
            if symbol in self._positions:
                self._close_position(symbol, bar, bar_time, reason=signal.reason or signal.signal_type.value)

    # ── Position Management ──────────────────────────────────────

    def _open_position(
        self,
        symbol: str,
        side: PositionSide,
        bar: pd.Series,
        bar_time: datetime,
        signal: Signal,
    ) -> None:
        raw_price = signal.price if signal.price > 0 else bar["close"]
        slippage_mult = (
            1 + self._config.slippage_pct / 100
            if side == PositionSide.LONG
            else 1 - self._config.slippage_pct / 100
        )
        fill_price = raw_price * slippage_mult

        allocation = self._capital * (self._config.position_size_pct / 100)
        quantity = signal.quantity if signal.quantity > 0 else max(1, int(allocation / fill_price))
        cost = fill_price * quantity
        commission = cost * (self._config.commission_pct / 100)

        if cost + commission > self._capital:
            return  # insufficient funds

        self._capital -= cost + commission

        self._positions[symbol] = _Position(
            symbol=symbol,
            side=side,
            entry_price=fill_price,
            quantity=quantity,
            entry_time=bar_time,
            stop_loss=signal.stop_loss,
            target=signal.target,
        )

    def _close_position(
        self, symbol: str, bar: pd.Series, bar_time: datetime, reason: str = ""
    ) -> None:
        pos = self._positions.pop(symbol, None)
        if pos is None:
            return

        raw_price = bar["close"]
        slippage_mult = (
            1 - self._config.slippage_pct / 100
            if pos.side == PositionSide.LONG
            else 1 + self._config.slippage_pct / 100
        )
        fill_price = raw_price * slippage_mult

        proceeds = fill_price * pos.quantity
        commission = proceeds * (self._config.commission_pct / 100)
        self._capital += proceeds - commission

        if pos.side == PositionSide.LONG:
            pnl = (fill_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - fill_price) * pos.quantity

        pnl -= commission + (pos.entry_price * pos.quantity * self._config.commission_pct / 100)
        entry_cost = pos.entry_price * pos.quantity
        pnl_pct = (pnl / entry_cost * 100) if entry_cost > 0 else 0.0

        total_commission = commission + (entry_cost * self._config.commission_pct / 100)

        self._trades.append(
            _Trade(
                symbol=symbol,
                side=pos.side.value,
                entry_price=pos.entry_price,
                exit_price=fill_price,
                quantity=pos.quantity,
                entry_time=pos.entry_time,
                exit_time=bar_time,
                pnl=pnl,
                pnl_pct=pnl_pct,
                commission=total_commission,
                reason=reason,
            )
        )

    def _check_exits(self, bar: pd.Series, bar_time: datetime) -> None:
        """Check stop-loss and target hits for all open positions."""
        symbols_to_close: list[tuple[str, str]] = []

        for symbol, pos in self._positions.items():
            if pos.side == PositionSide.LONG:
                if pos.stop_loss > 0 and bar["low"] <= pos.stop_loss:
                    symbols_to_close.append((symbol, "stop_loss"))
                elif pos.target > 0 and bar["high"] >= pos.target:
                    symbols_to_close.append((symbol, "target_hit"))
            else:  # SHORT
                if pos.stop_loss > 0 and bar["high"] >= pos.stop_loss:
                    symbols_to_close.append((symbol, "stop_loss"))
                elif pos.target > 0 and bar["low"] <= pos.target:
                    symbols_to_close.append((symbol, "target_hit"))

        for symbol, reason in symbols_to_close:
            self._close_position(symbol, bar, bar_time, reason=reason)

    def _close_all_positions(
        self, bar: pd.Series, bar_time: datetime, reason: str = ""
    ) -> None:
        for symbol in list(self._positions.keys()):
            self._close_position(symbol, bar, bar_time, reason=reason)

    # ── Portfolio Valuation ──────────────────────────────────────

    def _portfolio_value(self, bar: pd.Series) -> float:
        """Current capital + mark-to-market value of open positions."""
        mtm = 0.0
        price = bar["close"]
        for pos in self._positions.values():
            if pos.side == PositionSide.LONG:
                mtm += (price - pos.entry_price) * pos.quantity
            else:
                mtm += (pos.entry_price - price) * pos.quantity
            mtm += pos.entry_price * pos.quantity  # return the locked capital
        return self._capital + mtm

    # ── Results Compilation ──────────────────────────────────────

    def _compile_results(self) -> BacktestResult:
        if not self._equity_history:
            return _empty_result(self._config)

        dates, values = zip(*self._equity_history)
        equity = pd.Series(values, index=pd.DatetimeIndex(dates), name="equity")

        # Deduplicate index (keep last value per timestamp)
        equity = equity[~equity.index.duplicated(keep="last")]

        initial = self._config.initial_capital
        final = equity.iloc[-1]

        # Total return
        total_return_pct = ((final - initial) / initial) * 100

        # CAGR
        days = max((equity.index[-1] - equity.index[0]).days, 1)
        years = days / 365.25
        cagr_pct = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0.0

        # Daily returns
        daily_returns = equity.pct_change().dropna()

        # Sharpe ratio (annualised, excess over risk-free)
        daily_rf = self._config.risk_free_rate / 100 / 252
        excess = daily_returns - daily_rf
        sharpe = (
            (excess.mean() / excess.std()) * np.sqrt(252)
            if len(excess) > 1 and excess.std() > 0
            else 0.0
        )

        # Drawdown
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max * 100
        max_drawdown_pct = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        # Growth of Rs.10,000
        growth_of_10k = (equity / initial) * 10_000

        # Trade statistics
        wins = [t for t in self._trades if t.pnl > 0]
        losses = [t for t in self._trades if t.pnl <= 0]
        total_trades = len(self._trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0.0

        # Monthly returns pivot (year x month)
        monthly_returns = _compute_monthly_returns(equity)

        # Trade log as dicts
        trade_dicts = [
            {
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "quantity": t.quantity,
                "entry_time": str(t.entry_time),
                "exit_time": str(t.exit_time),
                "pnl": round(t.pnl, 2),
                "pnl_pct": round(t.pnl_pct, 2),
                "commission": round(t.commission, 2),
                "reason": t.reason,
            }
            for t in self._trades
        ]

        return BacktestResult(
            total_return_pct=round(total_return_pct, 2),
            cagr_pct=round(cagr_pct, 2),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown_pct=round(max_drawdown_pct, 2),
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            avg_win_pct=round(float(avg_win), 2),
            avg_loss_pct=round(float(avg_loss), 2),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            equity_curve=equity,
            growth_of_10k=growth_of_10k,
            drawdown_series=drawdown,
            daily_returns=daily_returns,
            monthly_returns=monthly_returns,
            trades=trade_dicts,
            config=self._config,
        )

    # ── Reset ────────────────────────────────────────────────────

    def _reset(self) -> None:
        self._capital = self._config.initial_capital
        self._positions = {}
        self._trades = []
        self._equity_history = []


# ── Pure Helper Functions ────────────────────────────────────────

def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that the DataFrame has required OHLCV columns."""
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    if len(df) < 2:
        raise ValueError("DataFrame must have at least 2 rows")


def _bar_window_to_market_data(window: pd.DataFrame, strategy: Strategy) -> dict:
    """Convert a DataFrame window into the market_data dict that Strategy.analyze expects."""
    symbols = strategy.get_required_symbols()
    # Provide the same window for all symbols the strategy requires.
    # Real execution would have per-symbol data; backtest uses the single feed.
    result: dict = {}
    for sym in symbols:
        result[sym] = window
    # Also provide under a generic key so strategies that index by any key work
    if symbols:
        result["default"] = window
    return result


def _compute_monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Build a year x month pivot of monthly returns (%)."""
    if len(equity) < 2:
        return pd.DataFrame()

    monthly = equity.resample("ME").last().pct_change().dropna() * 100
    if monthly.empty:
        return pd.DataFrame()

    df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = df.pivot_table(index="year", columns="month", values="return", aggfunc="sum")
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][: len(pivot.columns)]
    return pivot


def _empty_result(config: BacktestConfig) -> BacktestResult:
    """Return a zeroed-out result when there is no equity history."""
    empty_series = pd.Series(dtype=float)
    return BacktestResult(
        total_return_pct=0.0,
        cagr_pct=0.0,
        sharpe_ratio=0.0,
        max_drawdown_pct=0.0,
        win_rate_pct=0.0,
        profit_factor=0.0,
        avg_win_pct=0.0,
        avg_loss_pct=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        equity_curve=empty_series,
        growth_of_10k=empty_series,
        drawdown_series=empty_series,
        daily_returns=empty_series,
        monthly_returns=pd.DataFrame(),
        trades=[],
        config=config,
    )

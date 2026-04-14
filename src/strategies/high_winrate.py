"""
High Win Rate Strategies — Academically documented mean-reversion patterns.

Strategies:
  1. RSI2 Mean Reversion (91% documented win rate)
  2. Connors RSI (75% documented win rate)
  3. Triple Confirmation Bounce (70-80% estimated)
  4. Large Drop Mean Reversion (78% academic)
  5. Iron Condor Active Management (86% win rate) — simulated as delta-neutral

Each strategy extends Strategy from base.py and works with
the BacktestEngine's bar-by-bar analyze() interface.
"""

import numpy as np
import pandas as pd
from typing import Optional

from .base import Strategy, Signal, SignalType
from ..utils.indicators import rsi, sma, bollinger_bands, volume_surge


# ── Helpers ───────────────────────────────────────────────────────


def _rsi_custom(series: pd.Series, period: int) -> pd.Series:
    """Compute RSI on any series (not just df['close'])."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _streak_series(close: pd.Series) -> pd.Series:
    """Count consecutive up/down day streak. Positive = up, negative = down."""
    diff = close.diff()
    streak = pd.Series(0, index=close.index, dtype=float)
    for i in range(1, len(close)):
        if diff.iloc[i] > 0:
            streak.iloc[i] = max(streak.iloc[i - 1], 0) + 1
        elif diff.iloc[i] < 0:
            streak.iloc[i] = min(streak.iloc[i - 1], 0) - 1
        else:
            streak.iloc[i] = 0
    return streak


def _percentile_rank(series: pd.Series, lookback: int = 100) -> pd.Series:
    """Percentile rank of each value within its trailing window."""
    result = pd.Series(np.nan, index=series.index, dtype=float)
    for i in range(lookback, len(series)):
        window = series.iloc[i - lookback : i + 1]
        current = series.iloc[i]
        result.iloc[i] = (window < current).sum() / lookback * 100
    return result


# ── Strategy 1: RSI2 Mean Reversion ──────────────────────────────


class RSI2MeanReversion(Strategy):
    """
    RSI(2) dips below threshold while price is above 200 SMA.
    Exit when price closes above 5-day SMA.

    Documented win rate: ~91% on large-cap US stocks.
    Enhanced variant requires 3 consecutive bars with RSI(2) < 10.
    """

    def __init__(
        self,
        symbol: str = "STOCK",
        rsi_period: int = 2,
        rsi_entry: float = 5.0,
        exit_sma_period: int = 5,
        trend_sma_period: int = 200,
        enhanced: bool = False,
    ):
        super().__init__(
            name="RSI2_MeanReversion",
            description="Buy when RSI(2) < 5 in uptrend, exit above 5-day SMA",
        )
        self.symbol = symbol
        self.rsi_period = rsi_period
        self.rsi_entry = rsi_entry
        self.exit_sma_period = exit_sma_period
        self.trend_sma_period = trend_sma_period
        self.enhanced = enhanced
        self._in_position = False

    def get_required_symbols(self) -> list[str]:
        return [self.symbol]

    def analyze(self, market_data: dict) -> list[Signal]:
        df = market_data.get(self.symbol)
        if df is None:
            df = market_data.get("default")
        if df is None or len(df) < self.trend_sma_period + 5:
            return []

        close = df["close"]
        rsi2 = _rsi_custom(close, self.rsi_period)
        sma200 = close.rolling(window=self.trend_sma_period).mean()
        sma5 = close.rolling(window=self.exit_sma_period).mean()

        current_close = close.iloc[-1]
        current_rsi = rsi2.iloc[-1]
        current_sma200 = sma200.iloc[-1]
        current_sma5 = sma5.iloc[-1]

        if pd.isna(current_rsi) or pd.isna(current_sma200):
            return []

        signals = []

        # Exit condition: price above 5-day SMA
        if self._in_position:
            if current_close > current_sma5:
                self._in_position = False
                signals.append(Signal(
                    signal_type=SignalType.EXIT,
                    symbol=self.symbol,
                    price=current_close,
                    reason="close_above_5sma",
                    strategy_name=self.name,
                ))
            return signals

        # Entry condition: uptrend + RSI(2) oversold
        if current_close > current_sma200:
            if self.enhanced:
                # Require 3 consecutive bars with RSI(2) < 10
                if len(rsi2) >= 3:
                    last3 = rsi2.iloc[-3:]
                    if all(v < 10 for v in last3 if not pd.isna(v)):
                        self._in_position = True
                        signals.append(Signal(
                            signal_type=SignalType.BUY,
                            symbol=self.symbol,
                            price=current_close,
                            confidence=0.9,
                            reason="rsi2_capitulation_3bars",
                            strategy_name=self.name,
                        ))
            else:
                if current_rsi < self.rsi_entry:
                    self._in_position = True
                    signals.append(Signal(
                        signal_type=SignalType.BUY,
                        symbol=self.symbol,
                        price=current_close,
                        confidence=0.85,
                        reason=f"rsi2={current_rsi:.1f}_below_{self.rsi_entry}",
                        strategy_name=self.name,
                    ))

        return signals


# ── Strategy 2: Connors RSI ──────────────────────────────────────


class ConnorsRSI(Strategy):
    """
    Connors RSI = mean of RSI(3), Streak RSI, and Percentile Rank.
    Entry: ConnorsRSI < 10 AND price > 200 DMA.
    Exit: ConnorsRSI > 70 OR close > 5-day SMA.

    Documented win rate: ~75%.
    """

    def __init__(
        self,
        symbol: str = "STOCK",
        rsi_period: int = 3,
        streak_rsi_period: int = 2,
        pct_rank_lookback: int = 100,
        entry_threshold: float = 10.0,
        exit_threshold: float = 70.0,
        trend_sma_period: int = 200,
        exit_sma_period: int = 5,
    ):
        super().__init__(
            name="ConnorsRSI",
            description="Mean of RSI(3) + Streak RSI + Percentile Rank",
        )
        self.symbol = symbol
        self.rsi_period = rsi_period
        self.streak_rsi_period = streak_rsi_period
        self.pct_rank_lookback = pct_rank_lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.trend_sma_period = trend_sma_period
        self.exit_sma_period = exit_sma_period
        self._in_position = False

    def get_required_symbols(self) -> list[str]:
        return [self.symbol]

    def analyze(self, market_data: dict) -> list[Signal]:
        df = market_data.get(self.symbol)
        if df is None:
            df = market_data.get("default")
        if df is None or len(df) < max(self.trend_sma_period, self.pct_rank_lookback) + 10:
            return []

        close = df["close"]

        # Component 1: RSI(3)
        rsi3 = _rsi_custom(close, self.rsi_period)

        # Component 2: Streak RSI
        streak = _streak_series(close)
        streak_rsi = _rsi_custom(streak, self.streak_rsi_period)

        # Component 3: Percentile rank of daily return
        daily_return = close.pct_change()
        pct_rank = _percentile_rank(daily_return, self.pct_rank_lookback)

        # Connors RSI = average of 3 components
        connors = (rsi3 + streak_rsi + pct_rank) / 3

        sma200 = close.rolling(window=self.trend_sma_period).mean()
        sma5 = close.rolling(window=self.exit_sma_period).mean()

        current_close = close.iloc[-1]
        current_connors = connors.iloc[-1]
        current_sma200 = sma200.iloc[-1]
        current_sma5 = sma5.iloc[-1]

        if pd.isna(current_connors) or pd.isna(current_sma200):
            return []

        signals = []

        if self._in_position:
            if current_connors > self.exit_threshold or current_close > current_sma5:
                self._in_position = False
                signals.append(Signal(
                    signal_type=SignalType.EXIT,
                    symbol=self.symbol,
                    price=current_close,
                    reason=f"connors_rsi={current_connors:.1f}_exit",
                    strategy_name=self.name,
                ))
            return signals

        if current_close > current_sma200 and current_connors < self.entry_threshold:
            self._in_position = True
            signals.append(Signal(
                signal_type=SignalType.BUY,
                symbol=self.symbol,
                price=current_close,
                confidence=0.75,
                reason=f"connors_rsi={current_connors:.1f}_below_{self.entry_threshold}",
                strategy_name=self.name,
            ))

        return signals


# ── Strategy 3: Triple Confirmation Bounce ───────────────────────


class TripleConfirmationBounce(Strategy):
    """
    All three must be true simultaneously:
      - RSI(14) < 20
      - Price below lower Bollinger Band (20, 2SD)
      - Volume > 2x 20-day average

    Exit: price crosses above middle BB OR RSI > 50.
    Stop: -5% or 5-day time stop.

    Estimated win rate: 70-80%.
    """

    def __init__(
        self,
        symbol: str = "STOCK",
        rsi_period: int = 14,
        rsi_entry: float = 20.0,
        rsi_exit: float = 50.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        volume_mult: float = 2.0,
        stop_loss_pct: float = 5.0,
        time_stop_bars: int = 5,
    ):
        super().__init__(
            name="TripleConfirmBounce",
            description="RSI<20 + below lower BB + 2x volume",
        )
        self.symbol = symbol
        self.rsi_period = rsi_period
        self.rsi_entry = rsi_entry
        self.rsi_exit = rsi_exit
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.volume_mult = volume_mult
        self.stop_loss_pct = stop_loss_pct
        self.time_stop_bars = time_stop_bars
        self._in_position = False
        self._entry_price = 0.0
        self._bars_held = 0

    def get_required_symbols(self) -> list[str]:
        return [self.symbol]

    def analyze(self, market_data: dict) -> list[Signal]:
        df = market_data.get(self.symbol)
        if df is None:
            df = market_data.get("default")
        if df is None or len(df) < self.bb_period + 5:
            return []

        close = df["close"]
        rsi14 = _rsi_custom(close, self.rsi_period)
        bb = bollinger_bands(df, self.bb_period, self.bb_std)
        vol_ratio = volume_surge(df, self.bb_period, self.volume_mult)

        current_close = close.iloc[-1]
        current_rsi = rsi14.iloc[-1]
        current_lower_bb = bb["lower"].iloc[-1]
        current_middle_bb = bb["middle"].iloc[-1]
        current_vol_ratio = vol_ratio.iloc[-1]

        if any(pd.isna(v) for v in [current_rsi, current_lower_bb, current_vol_ratio]):
            return []

        signals = []

        if self._in_position:
            self._bars_held += 1

            # Stop loss
            loss_pct = (current_close - self._entry_price) / self._entry_price * 100
            if loss_pct < -self.stop_loss_pct:
                self._in_position = False
                signals.append(Signal(
                    signal_type=SignalType.EXIT,
                    symbol=self.symbol,
                    price=current_close,
                    reason="stop_loss",
                    strategy_name=self.name,
                ))
                return signals

            # Time stop
            if self._bars_held >= self.time_stop_bars:
                self._in_position = False
                signals.append(Signal(
                    signal_type=SignalType.EXIT,
                    symbol=self.symbol,
                    price=current_close,
                    reason="time_stop",
                    strategy_name=self.name,
                ))
                return signals

            # Profit exit: above middle BB or RSI > 50
            if current_close > current_middle_bb or current_rsi > self.rsi_exit:
                self._in_position = False
                signals.append(Signal(
                    signal_type=SignalType.EXIT,
                    symbol=self.symbol,
                    price=current_close,
                    reason="profit_target",
                    strategy_name=self.name,
                ))
            return signals

        # Entry: all 3 conditions
        if (
            current_rsi < self.rsi_entry
            and current_close < current_lower_bb
            and current_vol_ratio > self.volume_mult
        ):
            self._in_position = True
            self._entry_price = current_close
            self._bars_held = 0
            signals.append(Signal(
                signal_type=SignalType.BUY,
                symbol=self.symbol,
                price=current_close,
                stop_loss=current_close * (1 - self.stop_loss_pct / 100),
                confidence=0.8,
                reason=f"triple_confirm_rsi={current_rsi:.1f}_vol={current_vol_ratio:.1f}x",
                strategy_name=self.name,
            ))

        return signals


# ── Strategy 4: Large Drop Mean Reversion ────────────────────────


class LargeDropMeanReversion(Strategy):
    """
    Buy when a large-cap stock drops 5%+ in a single day.
    Exit next day at open (overnight hold) or hold until +1-2%.
    Stop: -3%.

    Academic win rate: ~78% on large caps.
    """

    def __init__(
        self,
        symbol: str = "STOCK",
        drop_threshold: float = -5.0,
        target_pct: float = 1.5,
        stop_loss_pct: float = 3.0,
        max_hold_days: int = 3,
    ):
        super().__init__(
            name="LargeDrop_MeanReversion",
            description="Buy on 5%+ single-day drop, exit next day or at target",
        )
        self.symbol = symbol
        self.drop_threshold = drop_threshold
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_days = max_hold_days
        self._in_position = False
        self._entry_price = 0.0
        self._bars_held = 0

    def get_required_symbols(self) -> list[str]:
        return [self.symbol]

    def analyze(self, market_data: dict) -> list[Signal]:
        df = market_data.get(self.symbol)
        if df is None:
            df = market_data.get("default")
        if df is None or len(df) < 3:
            return []

        close = df["close"]
        current_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        daily_return = (current_close - prev_close) / prev_close * 100

        signals = []

        if self._in_position:
            self._bars_held += 1

            # Stop loss
            loss_pct = (current_close - self._entry_price) / self._entry_price * 100
            if loss_pct < -self.stop_loss_pct:
                self._in_position = False
                signals.append(Signal(
                    signal_type=SignalType.EXIT,
                    symbol=self.symbol,
                    price=current_close,
                    reason="stop_loss",
                    strategy_name=self.name,
                ))
                return signals

            # Target hit
            gain_pct = (current_close - self._entry_price) / self._entry_price * 100
            if gain_pct >= self.target_pct:
                self._in_position = False
                signals.append(Signal(
                    signal_type=SignalType.EXIT,
                    symbol=self.symbol,
                    price=current_close,
                    reason="target_hit",
                    strategy_name=self.name,
                ))
                return signals

            # Time stop (max hold)
            if self._bars_held >= self.max_hold_days:
                self._in_position = False
                signals.append(Signal(
                    signal_type=SignalType.EXIT,
                    symbol=self.symbol,
                    price=current_close,
                    reason="time_stop",
                    strategy_name=self.name,
                ))
            return signals

        # Entry: single-day drop exceeds threshold
        if daily_return <= self.drop_threshold:
            self._in_position = True
            self._entry_price = current_close
            self._bars_held = 0
            stop = current_close * (1 - self.stop_loss_pct / 100)
            target = current_close * (1 + self.target_pct / 100)
            signals.append(Signal(
                signal_type=SignalType.BUY,
                symbol=self.symbol,
                price=current_close,
                stop_loss=stop,
                target=target,
                confidence=0.78,
                reason=f"large_drop={daily_return:.1f}%",
                strategy_name=self.name,
            ))

        return signals


# ── Strategy 5: Iron Condor (Simulated) ──────────────────────────


class IronCondorSimulated(Strategy):
    """
    Simulates Iron Condor premium collection on equity data.

    Since we don't have options chain data, we simulate:
    - Entry when implied vol (approximated by ATR%) is above threshold
    - Collect premium = f(ATR%)
    - Win if price stays within +/- strike_width% over holding period
    - Lose if price breaks out beyond the range

    Management:
    - Take profit at 50% of premium
    - Stop at 2x premium
    - Exit 1 day before expiry (hold_days - 1)

    Documented win rate: ~86% with active management.
    """

    def __init__(
        self,
        symbol: str = "STOCK",
        hold_days: int = 10,
        strike_width_pct: float = 3.0,
        min_vol_threshold: float = 1.5,
        take_profit_pct: float = 50.0,
        stop_loss_mult: float = 2.0,
    ):
        super().__init__(
            name="IronCondor_Simulated",
            description="Simulated iron condor premium selling",
        )
        self.symbol = symbol
        self.hold_days = hold_days
        self.strike_width_pct = strike_width_pct
        self.min_vol_threshold = min_vol_threshold
        self.take_profit_pct = take_profit_pct
        self.stop_loss_mult = stop_loss_mult
        self._in_position = False
        self._entry_price = 0.0
        self._premium_collected = 0.0
        self._bars_held = 0
        self._upper_strike = 0.0
        self._lower_strike = 0.0

    def get_required_symbols(self) -> list[str]:
        return [self.symbol]

    def analyze(self, market_data: dict) -> list[Signal]:
        df = market_data.get(self.symbol)
        if df is None:
            df = market_data.get("default")
        if df is None or len(df) < 22:
            return []

        close = df["close"]
        current_close = close.iloc[-1]

        # Approximate implied vol using ATR% (14-day ATR / close * 100)
        high = df["high"]
        low = df["low"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        atr_pct = (atr14.iloc[-1] / current_close) * 100

        if pd.isna(atr_pct):
            return []

        signals = []

        if self._in_position:
            self._bars_held += 1

            # Simulate P&L: premium erodes as time passes (theta decay)
            # but price movement can cause losses
            price_move_pct = abs(current_close - self._entry_price) / self._entry_price * 100

            # Theta decay: earn proportional premium per day
            time_decay_earned = self._premium_collected * (self._bars_held / self.hold_days)

            # If price breached a strike, loss increases
            if current_close > self._upper_strike or current_close < self._lower_strike:
                breach = max(
                    current_close - self._upper_strike,
                    self._lower_strike - current_close,
                    0,
                )
                unrealized_loss = (breach / self._entry_price) * 100

                # Stop loss: 2x premium
                if unrealized_loss > self._premium_collected * self.stop_loss_mult:
                    self._in_position = False
                    signals.append(Signal(
                        signal_type=SignalType.EXIT,
                        symbol=self.symbol,
                        price=current_close,
                        reason="ic_stop_loss_breach",
                        strategy_name=self.name,
                    ))
                    return signals

            # Take profit: 50% of premium earned through decay
            if time_decay_earned >= self._premium_collected * (self.take_profit_pct / 100):
                if price_move_pct < self.strike_width_pct * 0.7:
                    self._in_position = False
                    signals.append(Signal(
                        signal_type=SignalType.EXIT,
                        symbol=self.symbol,
                        price=current_close,
                        reason="ic_take_profit_50pct",
                        strategy_name=self.name,
                    ))
                    return signals

            # Time exit: 1 day before expiry
            if self._bars_held >= self.hold_days - 1:
                self._in_position = False
                signals.append(Signal(
                    signal_type=SignalType.EXIT,
                    symbol=self.symbol,
                    price=current_close,
                    reason="ic_expiry_exit",
                    strategy_name=self.name,
                ))
            return signals

        # Entry: volatility above threshold (enough premium to sell)
        if atr_pct > self.min_vol_threshold:
            self._in_position = True
            self._entry_price = current_close
            self._bars_held = 0
            self._upper_strike = current_close * (1 + self.strike_width_pct / 100)
            self._lower_strike = current_close * (1 - self.strike_width_pct / 100)
            # Premium approximation: ~0.3-0.5% of spot for 10-delta IC
            self._premium_collected = atr_pct * 0.25

            signals.append(Signal(
                signal_type=SignalType.BUY,
                symbol=self.symbol,
                price=current_close,
                confidence=0.86,
                reason=f"ic_entry_atr_pct={atr_pct:.2f}",
                strategy_name=self.name,
                metadata={
                    "upper_strike": self._upper_strike,
                    "lower_strike": self._lower_strike,
                    "premium_pct": self._premium_collected,
                },
            ))

        return signals

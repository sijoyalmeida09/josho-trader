"""
Momentum & Breakout Strategies — Directional F&O plays.
Integrated from: NSE-Stock-Scanner, PKScreener, SwitchGain.

Strategies:
  1. Breakout Momentum — Buy CE/PE on confirmed breakout
  2. Trend Following — Supertrend + VWAP + volume confirmation
  3. Mean Reversion — Bollinger Band squeeze reversal
"""

import logging
import pandas as pd
from typing import Optional

from .base import Strategy, Signal, SignalType
from ..utils.indicators import (
    rsi, macd, supertrend, bollinger_bands, vwap,
    volume_surge, detect_breakout, detect_consolidation,
    detect_engulfing, momentum_score, atr, ema, adx,
)

log = logging.getLogger("josho.strategy.momentum")


class BreakoutMomentum(Strategy):
    """
    Buy options when stock breaks out of consolidation with volume.
    Uses Donchian Channel breakout + volume confirmation.
    """

    def __init__(
        self,
        watchlist: list[str] = None,
        consolidation_period: int = 10,
        breakout_period: int = 20,
        min_volume_surge: float = 1.5,
        min_momentum_score: float = 30,
    ):
        super().__init__(
            name="BreakoutMomentum",
            description="Buy options on confirmed breakouts with volume",
        )
        self.watchlist = watchlist or [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
            "SBIN", "BHARTIARTL", "ITC", "TATAMOTORS", "BAJFINANCE",
        ]
        self.consolidation_period = consolidation_period
        self.breakout_period = breakout_period
        self.min_volume_surge = min_volume_surge
        self.min_momentum_score = min_momentum_score

    def analyze(self, market_data: dict) -> list[Signal]:
        signals = []

        for symbol in self.watchlist:
            df = market_data.get(f"{symbol}_candles")
            if df is None or len(df) < 50:
                continue

            # Check consolidation → breakout
            was_consolidating = detect_consolidation(df.iloc[:-1], self.consolidation_period)
            breakout = detect_breakout(df, self.breakout_period)
            vol_surge = volume_surge(df).iloc[-1]
            m_score = momentum_score(df)

            if not was_consolidating:
                continue

            if breakout["breakout_up"] and vol_surge >= self.min_volume_surge and m_score >= self.min_momentum_score:
                atr_val = atr(df).iloc[-1]
                signals.append(Signal(
                    signal_type=SignalType.BUY,
                    symbol=f"{symbol}",  # will be converted to CE option by executor
                    segment="FNO",
                    price=df["close"].iloc[-1],
                    stop_loss=df["close"].iloc[-1] - (1.5 * atr_val),
                    target=df["close"].iloc[-1] + (3 * atr_val),
                    confidence=min(m_score / 100, 0.9),
                    reason=(
                        f"Breakout UP: consolidation → breakout | "
                        f"Vol surge: {vol_surge:.1f}x | Momentum: {m_score:.0f}"
                    ),
                    strategy_name=self.name,
                    metadata={
                        "direction": "BULLISH",
                        "option_type": "CE",
                        "volume_surge": vol_surge,
                        "momentum_score": m_score,
                    },
                ))

            elif breakout["breakout_down"] and vol_surge >= self.min_volume_surge and m_score <= -self.min_momentum_score:
                atr_val = atr(df).iloc[-1]
                signals.append(Signal(
                    signal_type=SignalType.BUY,
                    symbol=f"{symbol}",
                    segment="FNO",
                    price=df["close"].iloc[-1],
                    stop_loss=df["close"].iloc[-1] + (1.5 * atr_val),
                    target=df["close"].iloc[-1] - (3 * atr_val),
                    confidence=min(abs(m_score) / 100, 0.9),
                    reason=(
                        f"Breakout DOWN: consolidation → breakdown | "
                        f"Vol surge: {vol_surge:.1f}x | Momentum: {m_score:.0f}"
                    ),
                    strategy_name=self.name,
                    metadata={
                        "direction": "BEARISH",
                        "option_type": "PE",
                        "volume_surge": vol_surge,
                        "momentum_score": m_score,
                    },
                ))

        if signals:
            log.info(f"Breakout signals: {len(signals)} found")
        return signals

    def get_required_symbols(self) -> list[str]:
        return self.watchlist


class TrendFollowing(Strategy):
    """
    Follow established trends using Supertrend + EMA crossover + VWAP.
    Buy CE in uptrend, Buy PE in downtrend.
    """

    def __init__(self, watchlist: list[str] = None):
        super().__init__(
            name="TrendFollowing",
            description="Supertrend + EMA + VWAP trend following",
        )
        self.watchlist = watchlist or ["NIFTY", "BANKNIFTY"]

    def analyze(self, market_data: dict) -> list[Signal]:
        signals = []

        for symbol in self.watchlist:
            df = market_data.get(f"{symbol}_candles")
            if df is None or len(df) < 50:
                continue

            st = supertrend(df)
            ema_9 = ema(df, 9).iloc[-1]
            ema_21 = ema(df, 21).iloc[-1]
            vwap_val = vwap(df).iloc[-1]
            adx_val = adx(df).iloc[-1]
            rsi_val = rsi(df).iloc[-1]
            last_close = df["close"].iloc[-1]

            st_dir = st["direction"].iloc[-1]
            prev_st_dir = st["direction"].iloc[-2]

            # Supertrend flip + EMA alignment + above/below VWAP
            if st_dir == 1 and prev_st_dir == -1:  # flip to bullish
                if ema_9 > ema_21 and last_close > vwap_val and adx_val > 20:
                    signals.append(Signal(
                        signal_type=SignalType.BUY,
                        symbol=symbol,
                        segment="FNO",
                        price=last_close,
                        stop_loss=st["supertrend"].iloc[-1],
                        confidence=min(adx_val / 50, 0.85),
                        reason=(
                            f"Trend flip BULLISH: Supertrend + EMA9>21 + above VWAP | "
                            f"ADX: {adx_val:.0f} | RSI: {rsi_val:.0f}"
                        ),
                        strategy_name=self.name,
                        metadata={"direction": "BULLISH", "option_type": "CE", "adx": adx_val},
                    ))

            elif st_dir == -1 and prev_st_dir == 1:  # flip to bearish
                if ema_9 < ema_21 and last_close < vwap_val and adx_val > 20:
                    signals.append(Signal(
                        signal_type=SignalType.BUY,
                        symbol=symbol,
                        segment="FNO",
                        price=last_close,
                        stop_loss=st["supertrend"].iloc[-1],
                        confidence=min(adx_val / 50, 0.85),
                        reason=(
                            f"Trend flip BEARISH: Supertrend + EMA9<21 + below VWAP | "
                            f"ADX: {adx_val:.0f} | RSI: {rsi_val:.0f}"
                        ),
                        strategy_name=self.name,
                        metadata={"direction": "BEARISH", "option_type": "PE", "adx": adx_val},
                    ))

        return signals

    def get_required_symbols(self) -> list[str]:
        return self.watchlist


class MeanReversion(Strategy):
    """
    Mean reversion using Bollinger Bands + RSI.
    Buy when oversold at lower band, sell when overbought at upper band.
    """

    def __init__(self, watchlist: list[str] = None):
        super().__init__(
            name="MeanReversion",
            description="Bollinger Band + RSI mean reversion",
        )
        self.watchlist = watchlist or [
            "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
        ]

    def analyze(self, market_data: dict) -> list[Signal]:
        signals = []

        for symbol in self.watchlist:
            df = market_data.get(f"{symbol}_candles")
            if df is None or len(df) < 30:
                continue

            bb = bollinger_bands(df)
            rsi_val = rsi(df).iloc[-1]
            last_close = df["close"].iloc[-1]
            bb_pct = bb["pct"].iloc[-1]
            engulfing = detect_engulfing(df)

            # Oversold: price near lower band + RSI < 30
            if bb_pct < 0.05 and rsi_val < 35 and engulfing["bullish"]:
                signals.append(Signal(
                    signal_type=SignalType.BUY,
                    symbol=symbol,
                    segment="FNO",
                    price=last_close,
                    stop_loss=bb["lower"].iloc[-1] * 0.99,
                    target=bb["middle"].iloc[-1],
                    confidence=0.7,
                    reason=(
                        f"Mean reversion BUY: BB%={bb_pct:.2f}, RSI={rsi_val:.0f}, "
                        f"bullish engulfing at lower band"
                    ),
                    strategy_name=self.name,
                    metadata={"direction": "BULLISH", "option_type": "CE", "bb_pct": bb_pct},
                ))

            # Overbought: price near upper band + RSI > 70
            elif bb_pct > 0.95 and rsi_val > 65 and engulfing["bearish"]:
                signals.append(Signal(
                    signal_type=SignalType.BUY,
                    symbol=symbol,
                    segment="FNO",
                    price=last_close,
                    stop_loss=bb["upper"].iloc[-1] * 1.01,
                    target=bb["middle"].iloc[-1],
                    confidence=0.7,
                    reason=(
                        f"Mean reversion SELL: BB%={bb_pct:.2f}, RSI={rsi_val:.0f}, "
                        f"bearish engulfing at upper band"
                    ),
                    strategy_name=self.name,
                    metadata={"direction": "BEARISH", "option_type": "PE", "bb_pct": bb_pct},
                ))

        return signals

    def get_required_symbols(self) -> list[str]:
        return self.watchlist

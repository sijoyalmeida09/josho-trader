"""
Technical Indicators Engine — 40+ indicators for signal generation.
Uses pandas + ta library for calculation. Pure functions, no side effects.

Integrated from: stock_bot_langgraph (multi-agent), SwitchGain (momentum/mean-reversion),
NSE-Stock-Scanner (patterns), PKScreener (breakout detection).
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    import ta
    from ta.trend import (
        SMAIndicator, EMAIndicator, MACD, ADXIndicator,
        IchimokuIndicator, AroonIndicator, CCIIndicator,
    )
    from ta.momentum import (
        RSIIndicator, StochasticOscillator, WilliamsRIndicator,
        ROCIndicator, AwesomeOscillatorIndicator,
    )
    from ta.volatility import (
        BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel,
    )
    from ta.volume import (
        OnBalanceVolumeIndicator, VolumeWeightedAveragePrice,
        AccDistIndexIndicator, MFIIndicator,
    )
    HAS_TA = True
except ImportError:
    HAS_TA = False


# ── Trend Indicators ──────────────────────────────────────────────

def sma(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """Simple Moving Average."""
    return df[col].rolling(window=period).mean()


def ema(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """Exponential Moving Average."""
    return df[col].ewm(span=period, adjust=False).mean()


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """MACD — trend direction + momentum."""
    if HAS_TA:
        m = MACD(df["close"], window_fast=fast, window_slow=slow, window_sign=signal)
        return {
            "macd": m.macd(),
            "signal": m.macd_signal(),
            "histogram": m.macd_diff(),
        }
    fast_ema = ema(df, fast)
    slow_ema = ema(df, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return {"macd": macd_line, "signal": signal_line, "histogram": macd_line - signal_line}


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — trend strength (>25 = trending)."""
    if HAS_TA:
        return ADXIndicator(df["high"], df["low"], df["close"], window=period).adx()
    return pd.Series(dtype=float)


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> dict:
    """Supertrend — popular in Indian markets for trend following."""
    atr_val = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2

    upper = hl2 + (multiplier * atr_val)
    lower = hl2 - (multiplier * atr_val)

    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    st.iloc[0] = upper.iloc[0]
    direction.iloc[0] = -1

    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper.iloc[i - 1]:
            st.iloc[i] = lower.iloc[i]
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower.iloc[i - 1]:
            st.iloc[i] = upper.iloc[i]
            direction.iloc[i] = -1
        else:
            if direction.iloc[i - 1] == 1:
                st.iloc[i] = max(lower.iloc[i], st.iloc[i - 1])
            else:
                st.iloc[i] = min(upper.iloc[i], st.iloc[i - 1])
            direction.iloc[i] = direction.iloc[i - 1]

    return {"supertrend": st, "direction": direction}


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price — institutional reference."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    return (typical * df["volume"]).cumsum() / df["volume"].cumsum()


# ── Momentum Indicators ──────────────────────────────────────────

def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI — overbought (>70) / oversold (<30)."""
    if HAS_TA:
        return RSIIndicator(df["close"], window=period).rsi()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> dict:
    """Stochastic Oscillator — %K and %D."""
    if HAS_TA:
        s = StochasticOscillator(df["high"], df["low"], df["close"], window=k, smooth_window=d)
        return {"k": s.stoch(), "d": s.stoch_signal()}
    low_min = df["low"].rolling(window=k).min()
    high_max = df["high"].rolling(window=k).max()
    k_line = 100 * (df["close"] - low_min) / (high_max - low_min)
    d_line = k_line.rolling(window=d).mean()
    return {"k": k_line, "d": d_line}


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R — momentum."""
    if HAS_TA:
        return WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=period).williams_r()
    return pd.Series(dtype=float)


def roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
    """Rate of Change — momentum."""
    return ((df["close"] - df["close"].shift(period)) / df["close"].shift(period)) * 100


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    if HAS_TA:
        return CCIIndicator(df["high"], df["low"], df["close"], window=period).cci()
    return pd.Series(dtype=float)


# ── Volatility Indicators ────────────────────────────────────────

def bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> dict:
    """Bollinger Bands — volatility + mean reversion signals."""
    if HAS_TA:
        bb = BollingerBands(df["close"], window=period, window_dev=std)
        return {
            "upper": bb.bollinger_hband(),
            "middle": bb.bollinger_mavg(),
            "lower": bb.bollinger_lband(),
            "width": bb.bollinger_wband(),
            "pct": bb.bollinger_pband(),
        }
    mid = sma(df, period)
    std_dev = df["close"].rolling(window=period).std()
    return {
        "upper": mid + (std * std_dev),
        "middle": mid,
        "lower": mid - (std * std_dev),
        "width": (4 * std * std_dev) / mid,
        "pct": (df["close"] - (mid - std * std_dev)) / (2 * std * std_dev),
    }


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — volatility measure."""
    if HAS_TA:
        return AverageTrueRange(df["high"], df["low"], df["close"], window=period).average_true_range()
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def keltner_channel(df: pd.DataFrame, period: int = 20, atr_mult: float = 2.0) -> dict:
    """Keltner Channel — volatility bands."""
    mid = ema(df, period)
    atr_val = atr(df, period)
    return {
        "upper": mid + (atr_mult * atr_val),
        "middle": mid,
        "lower": mid - (atr_mult * atr_val),
    }


def donchian_channel(df: pd.DataFrame, period: int = 20) -> dict:
    """Donchian Channel — breakout detection."""
    return {
        "upper": df["high"].rolling(window=period).max(),
        "lower": df["low"].rolling(window=period).min(),
        "middle": (df["high"].rolling(window=period).max() + df["low"].rolling(window=period).min()) / 2,
    }


# ── Volume Indicators ────────────────────────────────────────────

def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — volume confirming price moves."""
    if HAS_TA:
        return OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    direction = np.where(df["close"] > df["close"].shift(1), 1, -1)
    return (df["volume"] * direction).cumsum()


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index — volume-weighted RSI."""
    if HAS_TA:
        return MFIIndicator(df["high"], df["low"], df["close"], df["volume"], window=period).money_flow_index()
    return pd.Series(dtype=float)


def volume_surge(df: pd.DataFrame, period: int = 20, threshold: float = 2.0) -> pd.Series:
    """Detect volume surges (volume > threshold * average)."""
    avg_vol = df["volume"].rolling(window=period).mean()
    return df["volume"] / avg_vol


# ── Pattern Detection (from PKScreener + NSE-Stock-Scanner) ──────

def detect_consolidation(df: pd.DataFrame, period: int = 10, threshold: float = 0.03) -> bool:
    """Detect price consolidation (tight range before breakout)."""
    recent = df.tail(period)
    price_range = (recent["high"].max() - recent["low"].min()) / recent["close"].mean()
    return price_range < threshold


def detect_breakout(df: pd.DataFrame, period: int = 20) -> dict:
    """Detect breakout from consolidation/channel."""
    dc = donchian_channel(df, period)
    last_close = df["close"].iloc[-1]
    prev_upper = dc["upper"].iloc[-2]
    prev_lower = dc["lower"].iloc[-2]

    return {
        "breakout_up": last_close > prev_upper,
        "breakout_down": last_close < prev_lower,
        "near_upper": (prev_upper - last_close) / last_close < 0.01,
        "near_lower": (last_close - prev_lower) / last_close < 0.01,
    }


def detect_engulfing(df: pd.DataFrame) -> dict:
    """Detect bullish/bearish engulfing candle patterns."""
    if len(df) < 2:
        return {"bullish": False, "bearish": False}

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    bullish = (
        prev["close"] < prev["open"]  # prev was red
        and curr["close"] > curr["open"]  # curr is green
        and curr["open"] <= prev["close"]  # curr opens at/below prev close
        and curr["close"] >= prev["open"]  # curr closes at/above prev open
    )

    bearish = (
        prev["close"] > prev["open"]  # prev was green
        and curr["close"] < curr["open"]  # curr is red
        and curr["open"] >= prev["close"]  # curr opens at/above prev close
        and curr["close"] <= prev["open"]  # curr closes at/below prev open
    )

    return {"bullish": bullish, "bearish": bearish}


def detect_doji(df: pd.DataFrame, threshold: float = 0.1) -> bool:
    """Detect doji candle (indecision)."""
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    total_range = last["high"] - last["low"]
    return (body / total_range) < threshold if total_range > 0 else False


def detect_hammer(df: pd.DataFrame) -> bool:
    """Detect hammer/inverted hammer (reversal signal)."""
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    lower_shadow = min(last["open"], last["close"]) - last["low"]
    upper_shadow = last["high"] - max(last["open"], last["close"])
    return lower_shadow > (2 * body) and upper_shadow < body


# ── Composite Signals ────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame) -> dict:
    """Compute all 40+ indicators at once. Returns dict of results."""
    results = {}

    # Trend
    results["sma_20"] = sma(df, 20)
    results["sma_50"] = sma(df, 50)
    results["sma_200"] = sma(df, 200)
    results["ema_9"] = ema(df, 9)
    results["ema_21"] = ema(df, 21)
    results["macd"] = macd(df)
    results["adx"] = adx(df)
    results["supertrend"] = supertrend(df)
    results["vwap"] = vwap(df)

    # Momentum
    results["rsi"] = rsi(df)
    results["stochastic"] = stochastic(df)
    results["williams_r"] = williams_r(df)
    results["roc"] = roc(df)
    results["cci"] = cci(df)

    # Volatility
    results["bollinger"] = bollinger_bands(df)
    results["atr"] = atr(df)
    results["keltner"] = keltner_channel(df)
    results["donchian"] = donchian_channel(df)

    # Volume
    results["obv"] = obv(df)
    results["mfi"] = mfi(df)
    results["volume_surge"] = volume_surge(df)

    # Patterns
    results["consolidation"] = detect_consolidation(df)
    results["breakout"] = detect_breakout(df)
    results["engulfing"] = detect_engulfing(df)
    results["doji"] = detect_doji(df)
    results["hammer"] = detect_hammer(df)

    return results


def momentum_score(df: pd.DataFrame) -> float:
    """
    Composite momentum score from -100 (extreme bearish) to +100 (extreme bullish).
    Combines RSI, MACD, ADX, Supertrend, volume.
    """
    score = 0.0
    weights_sum = 0.0

    # RSI (weight: 20)
    rsi_val = rsi(df).iloc[-1]
    if not np.isnan(rsi_val):
        rsi_score = (rsi_val - 50) * 2  # -100 to +100
        score += rsi_score * 20
        weights_sum += 20

    # MACD histogram (weight: 25)
    m = macd(df)
    hist = m["histogram"].iloc[-1]
    if not np.isnan(hist):
        macd_score = np.clip(hist * 10, -100, 100)
        score += macd_score * 25
        weights_sum += 25

    # ADX + direction (weight: 20)
    adx_val = adx(df).iloc[-1]
    if not np.isnan(adx_val):
        ema_9 = ema(df, 9).iloc[-1]
        ema_21 = ema(df, 21).iloc[-1]
        direction = 1 if ema_9 > ema_21 else -1
        adx_score = min(adx_val, 100) * direction
        score += adx_score * 20
        weights_sum += 20

    # Supertrend (weight: 15)
    st = supertrend(df)
    st_dir = st["direction"].iloc[-1]
    score += (st_dir * 100) * 15
    weights_sum += 15

    # Volume surge (weight: 20)
    vs = volume_surge(df).iloc[-1]
    if not np.isnan(vs):
        vol_score = np.clip((vs - 1) * 50, -100, 100)
        last_change = (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2]
        vol_direction = 1 if last_change > 0 else -1
        score += (vol_score * vol_direction) * 20
        weights_sum += 20

    return score / weights_sum if weights_sum > 0 else 0

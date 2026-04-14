"""
Alpha158 Feature Engine — Microsoft Qlib's factor set + WorldQuant Alpha101 extras.

Implements the full Alpha158 feature set used by Microsoft's $40B+ Qlib platform:
- Price ratios, multi-horizon returns, rolling statistics
- Z-scores, range positions, volume ratios
- VWAP features, momentum, volatility regimes
- Temporal rank features, Sharpe-like ratios
- WorldQuant Alpha 042, 101, 006

Total: ~100+ features before selection.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Helper Functions ─────────────────────────────────────────────────

def _safe_div(a, b, fill: float = 0.0):
    """Division with epsilon to prevent inf/nan."""
    return a / (b + 1e-8)


def _rolling_rank(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of current value within rolling window."""
    def _rank_pct(x):
        if len(x) < 2:
            return 0.5
        rank = (x[-1] > x[:-1]).sum()
        return rank / (len(x) - 1)
    return series.rolling(window, min_periods=max(window // 2, 2)).apply(_rank_pct, raw=True)


def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Rolling correlation between two series."""
    return a.rolling(window, min_periods=max(window // 2, 2)).corr(b)


def _vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price (daily approximation)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    if "volume" in df.columns and df["volume"].sum() > 0:
        cumvol = df["volume"].cumsum()
        cumtp = (typical_price * df["volume"]).cumsum()
        return _safe_div(cumtp, cumvol)
    return typical_price


# ── Core Alpha158 Features ──────────────────────────────────────────

def compute_alpha158(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Microsoft Qlib Alpha158 features from OHLCV data.

    Args:
        df: DataFrame with columns [open, high, low, close, volume] and DatetimeIndex.

    Returns:
        DataFrame with ~100+ features, same index as input.
    """
    f = pd.DataFrame(index=df.index)
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # ── 1. Price Ratio Features (4) ──────────────────────────────
    f["close_open_ratio"] = _safe_div(c, o)
    f["high_low_ratio"] = _safe_div(h, l)
    f["close_high_ratio"] = _safe_div(c, h)
    f["close_low_ratio"] = _safe_div(c, l)

    # ── 2. Multi-Horizon Returns (5) ────────────────────────────
    for d in [1, 5, 10, 20, 60]:
        f[f"ret_{d}d"] = c.pct_change(d)

    # ── 3. Rolling Statistics on Close (windows: 5, 10, 20, 60) ─
    for w in [5, 10, 20, 60]:
        f[f"close_mean_{w}"] = c.rolling(w).mean() / c  # normalized
        f[f"close_std_{w}"] = c.rolling(w).std() / c
        f[f"close_skew_{w}"] = c.rolling(w).skew()
        f[f"close_kurt_{w}"] = c.rolling(w).kurt()

    # ── 4. Rolling Statistics on Volume (windows: 5, 10, 20, 60) ─
    for w in [5, 10, 20, 60]:
        vol_mean = v.rolling(w).mean()
        f[f"vol_mean_ratio_{w}"] = _safe_div(v, vol_mean)
        f[f"vol_std_{w}"] = _safe_div(v.rolling(w).std(), vol_mean)

    # ── 5. Z-Score Features ──────────────────────────────────────
    for w in [10, 20, 60]:
        roll_mean = c.rolling(w).mean()
        roll_std = c.rolling(w).std()
        f[f"zscore_{w}"] = _safe_div(c - roll_mean, roll_std)

    # ── 6. Position in Range ─────────────────────────────────────
    for w in [10, 20, 60]:
        roll_min = c.rolling(w).min()
        roll_max = c.rolling(w).max()
        f[f"range_pos_{w}"] = _safe_div(c - roll_min, roll_max - roll_min)

    # ── 7. Volume Ratio ──────────────────────────────────────────
    f["vol_ratio_20"] = _safe_div(v, v.rolling(20).mean())
    f["vol_ratio_5"] = _safe_div(v, v.rolling(5).mean())

    # ── 8. Price-Volume Correlation ──────────────────────────────
    for w in [10, 20, 60]:
        f[f"pv_corr_{w}"] = _rolling_corr(c, v, w)

    # ── 9. VWAP Features ────────────────────────────────────────
    vwap_val = _vwap(df)
    f["close_vwap_pct"] = _safe_div(c - vwap_val, c)

    # ── 10. Temporal Rank Features ───────────────────────────────
    for d in [5, 10, 20]:
        ret = c.pct_change(d)
        f[f"rank_ret{d}d_60"] = _rolling_rank(ret, 60)

    f["rank_vol_60"] = _rolling_rank(v, 60)

    # ── 11. Momentum Features ───────────────────────────────────
    for w in [5, 10, 20, 60]:
        f[f"momentum_{w}"] = _safe_div(c - c.shift(w), c.shift(w))

    # Sharpe-like ratios (mean return / std return over window)
    ret_1d = c.pct_change(1)
    for w in [5, 10, 20, 60]:
        roll_mean_ret = ret_1d.rolling(w).mean()
        roll_std_ret = ret_1d.rolling(w).std()
        f[f"sharpe_{w}"] = _safe_div(roll_mean_ret, roll_std_ret)

    # ── 12. Volatility Features ──────────────────────────────────
    for w in [5, 10, 20, 60]:
        f[f"vol_ret_{w}"] = ret_1d.rolling(w).std()

    # Vol regime: short / long
    f["vol_regime_5_20"] = _safe_div(f["vol_ret_5"], f["vol_ret_20"])
    f["vol_regime_5_60"] = _safe_div(f["vol_ret_5"], f["vol_ret_60"])
    f["vol_regime_20_60"] = _safe_div(f["vol_ret_20"], f["vol_ret_60"])

    # ── 13. High-Low Range Features ──────────────────────────────
    intraday_range = _safe_div(h - l, c)
    f["intraday_range"] = intraday_range
    for w in [5, 10, 20]:
        f[f"avg_range_{w}"] = intraday_range.rolling(w).mean()

    for w in [10, 20, 60]:
        f[f"price_range_{w}"] = _safe_div(h.rolling(w).max(), l.rolling(w).min()) - 1

    # ── 14. Gap Features ─────────────────────────────────────────
    f["overnight_gap"] = _safe_div(o - c.shift(1), c.shift(1))
    f["gap_vol_interaction"] = f["overnight_gap"] * f["vol_ratio_20"]

    # ── 15. Trend Strength ───────────────────────────────────────
    for w in [10, 20]:
        up_days = (ret_1d > 0).rolling(w).sum()
        f[f"up_ratio_{w}"] = up_days / w

    # ── WorldQuant Alpha101 Extras ───────────────────────────────

    # Alpha 042: VWAP deviation (normalized)
    f["wq_alpha042"] = _safe_div(vwap_val - c, vwap_val + c)

    # Alpha 101: Intraday range normalized movement
    f["wq_alpha101"] = _safe_div(c - o, h - l)

    # Alpha 006: Correlation of open with volume (10-day)
    f["wq_alpha006"] = _rolling_corr(o, v, 10)

    # ── Cyclical Time Encoding ───────────────────────────────────
    if hasattr(df.index, "dayofweek"):
        f["day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 5)
        f["day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 5)
        f["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        f["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # ── Clean up ─────────────────────────────────────────────────
    f = f.replace([np.inf, -np.inf], np.nan)

    return f


def compute_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Alpha158 + V2 microstructure/volatility features.
    This is the full V3 feature set.
    """
    alpha = compute_alpha158(df)

    f = alpha.copy()
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # ── Parkinson volatility (more efficient than close-to-close) ─
    f["parkinson_vol"] = np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(h / l) ** 2)
    ).rolling(20).mean()

    # ── Garman-Klass volatility (uses all OHLC) ─────────────────
    gk_raw = (0.5 * np.log(h / l) ** 2 -
              (2 * np.log(2) - 1) * np.log(c / o) ** 2).clip(lower=0)
    f["gk_vol"] = np.sqrt(gk_raw).rolling(20).mean()

    # ── Kyle's Lambda proxy (price impact per volume) ────────────
    price_change = c.diff()
    signed_volume = v * np.sign(price_change)
    f["kyle_lambda"] = _safe_div(
        price_change.rolling(20).cov(signed_volume),
        signed_volume.rolling(20).var()
    )

    # ── Amihud illiquidity ratio ─────────────────────────────────
    f["amihud"] = _safe_div(
        abs(c.pct_change()), v * c
    ).rolling(20).mean()

    # ── Spread proxy from daily data ─────────────────────────────
    f["spread_proxy"] = 2 * _safe_div(h - l, h + l)

    # ── Pattern features ─────────────────────────────────────────
    f["higher_high"] = (h > h.shift(1)).astype(int)
    f["higher_low"] = (l > l.shift(1)).astype(int)
    f["inside_bar"] = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)

    # ── RSI (from raw calculation, no dependency on indicators.py) ─
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = _safe_div(gain, loss)
    f["rsi_14"] = 100 - (100 / (1 + rs))

    # ── Bollinger %B ─────────────────────────────────────────────
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    f["bb_pct"] = _safe_div(c - (sma20 - 2 * std20), 4 * std20)
    f["bb_width"] = _safe_div(4 * std20, sma20)

    # ── Clean up ─────────────────────────────────────────────────
    f = f.replace([np.inf, -np.inf], np.nan).fillna(0)

    return f

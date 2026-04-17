"""
mega_features.py — Multi-Factor Feature Engineering
=====================================================
Combines EVERYTHING that affects a stock:
  1. Technical indicators (price/volume)
  2. Macro factors (crude, gold, USD/INR, VIX)
  3. Global markets (S&P, Nikkei, Hang Seng)
  4. Weather (monsoon impacts agriculture→FMCG)
  5. Calendar effects (day of week, month, pre-holiday)
  6. Sentiment (Fear & Greed index)
  7. FII/DII flow proxy (India ETF volumes)
  8. Cross-stock correlation (sector momentum)

This creates a 100+ feature dataset per stock for training.

Usage:
    from mega_features import build_mega_features
    df = build_mega_features("COALINDIA", lookback_years=5)
    # df has 100+ columns ready for ML
"""

import math
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("mega_features")
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "historical"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "macro_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
# DATA FETCHERS (with caching)
# ══════════════════════════════════════════════════════════

def _fetch_yahoo(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical data from Yahoo Finance."""
    cache = CACHE_DIR / f"{symbol.replace('^','').replace('=','_')}_{period}.csv"
    if cache.exists():
        age = (datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)).total_seconds() / 3600
        if age < 12:  # cache for 12 hours
            return pd.read_csv(cache, index_col=0, parse_dates=True)

    try:
        resp = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={period}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        if not resp.ok:
            return pd.DataFrame()

        data = resp.json()["chart"]["result"][0]
        timestamps = data["timestamp"]
        quotes = data["indicators"]["quote"][0]

        df = pd.DataFrame({
            "open": quotes.get("open", []),
            "high": quotes.get("high", []),
            "low": quotes.get("low", []),
            "close": quotes.get("close", []),
            "volume": quotes.get("volume", []),
        }, index=pd.to_datetime(timestamps, unit="s"))

        df = df.dropna()
        df.to_csv(cache)
        return df
    except Exception as e:
        log.warning(f"Yahoo fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def fetch_macro_data() -> dict:
    """Fetch all macro data: crude, gold, USD/INR, VIX, global indices."""
    macro = {}

    sources = {
        "crude": "BZ=F",
        "gold": "GC=F",
        "usdinr": "USDINR=X",
        "vix_india": "^INDIAVIX",
        "vix_us": "^VIX",
        "sp500": "^GSPC",
        "nikkei": "^N225",
        "hangseng": "^HSI",
        "ftse": "^FTSE",
        "india_etf": "INDA",
        "dxy": "DX-Y.NYB",
        "us10y": "^TNX",
    }

    for name, symbol in sources.items():
        df = _fetch_yahoo(symbol, "5y")
        if not df.empty:
            macro[name] = df
            log.info(f"  Fetched {name}: {len(df)} rows")

    return macro


# ══════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 40+ technical indicators to a price DataFrame."""
    d = df.copy()
    c = d["close"]
    h = d["high"]
    l = d["low"]
    v = d["volume"]

    # Returns
    d["return_1d"] = c.pct_change(1)
    d["return_2d"] = c.pct_change(2)
    d["return_5d"] = c.pct_change(5)
    d["return_10d"] = c.pct_change(10)
    d["return_20d"] = c.pct_change(20)

    # Moving averages
    for w in [5, 10, 20, 50, 100, 200]:
        d[f"sma_{w}"] = c.rolling(w).mean()
        d[f"ema_{w}"] = c.ewm(span=w).mean()

    # Price vs MAs
    d["price_vs_sma20"] = (c / d["sma_20"] - 1) * 100
    d["price_vs_sma50"] = (c / d["sma_50"] - 1) * 100
    d["price_vs_sma200"] = (c / d["sma_200"] - 1) * 100
    d["sma20_vs_sma50"] = (d["sma_20"] / d["sma_50"] - 1) * 100

    # Volatility
    d["volatility_5d"] = d["return_1d"].rolling(5).std()
    d["volatility_20d"] = d["return_1d"].rolling(20).std()
    d["volatility_ratio"] = d["volatility_5d"] / d["volatility_20d"].replace(0, np.nan)

    # Bollinger Bands
    d["bb_upper"] = d["sma_20"] + 2 * c.rolling(20).std()
    d["bb_lower"] = d["sma_20"] - 2 * c.rolling(20).std()
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / d["sma_20"] * 100
    d["bb_position"] = (c - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"]).replace(0, np.nan)

    # RSI
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    d["macd"] = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    d["macd_signal"] = d["macd"].ewm(span=9).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # ATR
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    d["atr_14"] = tr.rolling(14).mean()
    d["atr_pct"] = d["atr_14"] / c * 100

    # Volume features
    d["volume_sma20"] = v.rolling(20).mean()
    d["volume_ratio"] = v / d["volume_sma20"].replace(0, np.nan)
    d["volume_change"] = v.pct_change()

    # OBV (On Balance Volume)
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    d["obv"] = obv
    d["obv_sma20"] = obv.rolling(20).mean()

    # Stochastic
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    d["stoch_k"] = (c - low14) / (high14 - low14).replace(0, np.nan) * 100
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    # Williams %R
    d["williams_r"] = (high14 - c) / (high14 - low14).replace(0, np.nan) * -100

    # CCI
    tp = (h + l + c) / 3
    d["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x)))))

    # Price patterns
    d["doji"] = ((h - l) > 0) & (abs(c - d["open"]) / (h - l).replace(0, np.nan) < 0.1)
    d["hammer"] = ((c > d["open"]) & ((d["open"] - l) > 2 * (c - d["open"])))
    d["engulfing_bull"] = ((c > d["open"]) & (c.shift(1) < d["open"].shift(1)) & (c > d["open"].shift(1)) & (d["open"] < c.shift(1)))

    # Day range
    d["day_range_pct"] = (h - l) / c * 100
    d["upper_shadow"] = (h - pd.concat([c, d["open"]], axis=1).max(axis=1)) / (h - l).replace(0, np.nan)
    d["lower_shadow"] = (pd.concat([c, d["open"]], axis=1).min(axis=1) - l) / (h - l).replace(0, np.nan)

    # Gaps
    d["gap_pct"] = (d["open"] - c.shift(1)) / c.shift(1) * 100

    return d


# ══════════════════════════════════════════════════════════
# CALENDAR & SEASONAL FEATURES
# ══════════════════════════════════════════════════════════

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day of week, month, quarter, pre/post holiday effects."""
    d = df.copy()
    idx = d.index

    d["day_of_week"] = idx.dayofweek  # 0=Mon, 4=Fri
    d["month"] = idx.month
    d["quarter"] = idx.quarter
    d["is_monday"] = (idx.dayofweek == 0).astype(int)
    d["is_friday"] = (idx.dayofweek == 4).astype(int)
    d["is_month_end"] = (idx.is_month_end).astype(int)
    d["is_month_start"] = (idx.is_month_start).astype(int)
    d["is_quarter_end"] = (idx.is_quarter_end).astype(int)

    # Expiry week effect (F&O expiry = last Thursday of month)
    d["day_in_month"] = idx.day
    d["days_to_month_end"] = (idx + pd.offsets.MonthEnd(0)).day - idx.day

    # Monsoon season (Jun-Sep affects agriculture/FMCG)
    d["is_monsoon"] = ((idx.month >= 6) & (idx.month <= 9)).astype(int)

    # Budget month (Feb — high volatility)
    d["is_budget_month"] = (idx.month == 2).astype(int)

    # Results season (Jan, Apr, Jul, Oct)
    d["is_results_season"] = (idx.month.isin([1, 4, 7, 10])).astype(int)

    return d


# ══════════════════════════════════════════════════════════
# MACRO FACTOR FEATURES
# ══════════════════════════════════════════════════════════

def add_macro_features(df: pd.DataFrame, macro: dict) -> pd.DataFrame:
    """Merge macro data (crude, gold, VIX, global indices) into stock DataFrame."""
    d = df.copy()

    for name, macro_df in macro.items():
        if macro_df.empty:
            continue

        # Align by date
        macro_close = macro_df["close"].reindex(d.index, method="ffill")
        macro_return = macro_close.pct_change()

        d[f"{name}_price"] = macro_close
        d[f"{name}_return_1d"] = macro_return
        d[f"{name}_return_5d"] = macro_close.pct_change(5)

        # Correlation with stock over 20-day rolling window
        d[f"{name}_corr_20d"] = d["return_1d"].rolling(20).corr(macro_return)

    return d


# ══════════════════════════════════════════════════════════
# SECTOR MOMENTUM FEATURES
# ══════════════════════════════════════════════════════════

def add_sector_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add sector peers' momentum as features."""
    d = df.copy()

    # Sector mapping
    SECTORS = {
        "METALS": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "SAIL", "COALINDIA"],
        "BANKING": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "PNB"],
        "IT": ["INFY", "TCS", "WIPRO", "HCLTECH", "TECHM"],
        "OIL": ["ONGC", "BPCL", "IOC", "GAIL"],
        "AUTO": ["MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT"],
        "INFRA": ["ADANIPOWER", "NTPC", "POWERGRID", "TATAPOWER", "NHPC"],
    }

    # Find which sector this stock belongs to
    stock_sector = None
    sector_peers = []
    for sector, stocks in SECTORS.items():
        if symbol.upper() in stocks:
            stock_sector = sector
            sector_peers = [s for s in stocks if s != symbol.upper()]
            break

    if not sector_peers:
        return d

    # Fetch peer returns and create sector momentum
    peer_returns = []
    for peer in sector_peers[:3]:  # max 3 peers to limit API calls
        peer_file = DATA_DIR / "daily_5y" / f"{peer.lower()}_daily_5y.csv"
        if peer_file.exists():
            try:
                pdf = pd.read_csv(peer_file, index_col=0, parse_dates=True)
                if "Close" in pdf.columns:
                    ret = pdf["Close"].pct_change().reindex(d.index, method="ffill")
                    peer_returns.append(ret)
                elif "close" in pdf.columns:
                    ret = pdf["close"].pct_change().reindex(d.index, method="ffill")
                    peer_returns.append(ret)
            except:
                pass

    if peer_returns:
        sector_avg = pd.concat(peer_returns, axis=1).mean(axis=1)
        d["sector_return_1d"] = sector_avg
        d["sector_return_5d"] = sector_avg.rolling(5).sum()
        d["vs_sector"] = d["return_1d"] - sector_avg  # outperformance
        d[f"sector_{stock_sector}_momentum"] = sector_avg.rolling(10).mean()

    return d


# ══════════════════════════════════════════════════════════
# LABEL CREATION
# ══════════════════════════════════════════════════════════

def add_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.5) -> pd.DataFrame:
    """Create prediction labels: will price go up > threshold% in N days?"""
    d = df.copy()
    future_return = d["close"].pct_change(horizon).shift(-horizon) * 100

    d["target_return"] = future_return
    d["target_binary"] = (future_return > threshold).astype(int)  # 1=UP, 0=DOWN
    d["target_3class"] = pd.cut(future_return, bins=[-100, -threshold, threshold, 100], labels=[0, 1, 2])

    return d


# ══════════════════════════════════════════════════════════
# MASTER BUILDER
# ══════════════════════════════════════════════════════════

def build_mega_features(
    symbol: str,
    lookback_years: int = 5,
    prediction_horizon: int = 5,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Build the complete 100+ feature dataset for a stock.

    Returns DataFrame with:
    - 40+ technical indicators
    - 10+ calendar features
    - 20+ macro factors (crude, gold, VIX, currencies, global indices)
    - 5+ sector momentum features
    - Target labels (binary + 3-class)
    """
    log.info(f"Building mega features for {symbol}...")

    # 1. Load stock data
    stock_file = DATA_DIR / "daily_5y" / f"{symbol.lower()}_daily_5y.csv"
    if stock_file.exists():
        df = pd.read_csv(stock_file, index_col=0, parse_dates=True)
        # Normalize column names
        df.columns = [c.lower() for c in df.columns]
    else:
        log.info(f"  No local data, fetching from Yahoo...")
        df = _fetch_yahoo(f"{symbol}.NS", f"{lookback_years}y")

    if df.empty:
        log.error(f"  No data for {symbol}")
        return pd.DataFrame()

    log.info(f"  Stock data: {len(df)} rows")

    # 2. Technical indicators
    df = add_technical_features(df)
    log.info(f"  Technical: {len(df.columns)} features")

    # 3. Calendar features
    df = add_calendar_features(df)
    log.info(f"  Calendar: {len(df.columns)} features")

    # 4. Macro factors
    log.info(f"  Fetching macro data...")
    macro = fetch_macro_data()
    df = add_macro_features(df, macro)
    log.info(f"  Macro: {len(df.columns)} features")

    # 5. Sector features
    df = add_sector_features(df, symbol)
    log.info(f"  Sector: {len(df.columns)} features")

    # 6. Labels
    df = add_labels(df, prediction_horizon, threshold)
    log.info(f"  Labels added. Total features: {len(df.columns)}")

    # 7. Drop NaN rows (from rolling calculations)
    before = len(df)
    df = df.dropna(subset=["target_binary"])
    df = df.fillna(method="ffill").fillna(0)
    log.info(f"  Final: {len(df)} rows ({before - len(df)} dropped)")

    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    symbol = sys.argv[1] if len(sys.argv) > 1 else "COALINDIA"
    df = build_mega_features(symbol)

    if not df.empty:
        print(f"\n{'='*60}")
        print(f"MEGA FEATURES: {symbol}")
        print(f"{'='*60}")
        print(f"Rows: {len(df)}")
        print(f"Features: {len(df.columns)}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"\nFeature categories:")
        tech = [c for c in df.columns if any(c.startswith(p) for p in ["sma_", "ema_", "rsi", "macd", "bb_", "atr", "stoch", "cci", "williams", "obv", "return_", "volatility", "volume_"])]
        macro_cols = [c for c in df.columns if any(c.startswith(p) for p in ["crude", "gold", "usdinr", "vix", "sp500", "nikkei", "hang", "ftse", "india_etf", "dxy", "us10y"])]
        calendar = [c for c in df.columns if any(c.startswith(p) for p in ["day_", "month", "quarter", "is_"])]
        sector = [c for c in df.columns if "sector" in c or "vs_sector" in c]
        print(f"  Technical: {len(tech)}")
        print(f"  Macro: {len(macro_cols)}")
        print(f"  Calendar: {len(calendar)}")
        print(f"  Sector: {len(sector)}")
        print(f"  Other: {len(df.columns) - len(tech) - len(macro_cols) - len(calendar) - len(sector)}")

        # Label distribution
        print(f"\nLabel distribution (5-day horizon, 0.5% threshold):")
        print(df["target_binary"].value_counts())

        # Save
        out = Path(__file__).parent.parent.parent / "data" / f"{symbol}_mega_features.csv"
        df.to_csv(out)
        print(f"\nSaved to {out}")

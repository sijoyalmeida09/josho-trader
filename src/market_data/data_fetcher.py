"""
Data Fetcher — 5 years of daily OHLCV via jugaad-data (NSE direct).

Fetches NIFTY, BANKNIFTY, and top 10 FNO stocks.
Saves to data/historical/{symbol}_5yr.csv (~1250 trading days each).
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger("josho.market_data.data_fetcher")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "historical"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# jugaad-data uses NSE symbols directly
INDEX_SYMBOLS = {
    "nifty": "NIFTY 50",
    "banknifty": "NIFTY BANK",
}

STOCK_SYMBOLS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "BHARTIARTL", "ITC", "TATAMOTORS", "BAJFINANCE",
]

FROM_DATE = date(2021, 1, 1)
TO_DATE = date(2026, 4, 13)


def fetch_index(name: str, symbol: str) -> Optional[pd.DataFrame]:
    """Fetch index data via yfinance (jugaad-data index API is broken as of 2026)."""
    YAHOO_MAP = {
        "NIFTY 50": "^NSEI",
        "NIFTY BANK": "^NSEBANK",
    }
    yahoo_ticker = YAHOO_MAP.get(symbol)
    if yahoo_ticker is None:
        log.error(f"No Yahoo ticker mapping for index {symbol}")
        return None

    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed: pip install yfinance")
        return None

    log.info(f"Fetching index {name} ({yahoo_ticker}) via yfinance from {FROM_DATE} to {TO_DATE}...")
    try:
        df = yf.download(yahoo_ticker, start=str(FROM_DATE), end=str(TO_DATE), progress=False)
    except Exception as e:
        log.error(f"Failed to fetch {name}: {e}")
        return None

    if df is None or df.empty:
        log.warning(f"No data returned for {name}")
        return None

    # yfinance returns columns like Open, High, Low, Close, Volume (or MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.strip().lower() for c in df.columns]
    df.index.name = "date"

    keep = ["open", "high", "low", "close"]
    if "volume" in df.columns:
        keep.append("volume")
    else:
        df["volume"] = 0
        keep.append("volume")

    df = df[keep].dropna(subset=["open", "high", "low", "close"])

    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    return df


def fetch_stock(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch stock data via jugaad-data."""
    try:
        from jugaad_data.nse import stock_df
    except ImportError:
        log.error("jugaad-data not installed: pip install jugaad-data")
        return None

    log.info(f"Fetching stock {symbol} from {FROM_DATE} to {TO_DATE}...")
    try:
        df = stock_df(symbol=symbol, from_date=FROM_DATE, to_date=TO_DATE, series="EQ")
    except Exception as e:
        log.error(f"Failed to fetch {symbol}: {e}")
        return None

    if df is None or df.empty:
        log.warning(f"No data returned for {symbol}")
        return None

    # jugaad-data returns uppercase columns: DATE, OPEN, HIGH, LOW, CLOSE, VOLUME, etc.
    df.columns = [c.strip().upper() for c in df.columns]

    # Map to lowercase standard names
    col_map = {"DATE": "date", "OPEN": "open", "HIGH": "high", "LOW": "low",
               "CLOSE": "close", "VOLUME": "volume"}
    df = df.rename(columns=col_map)

    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        log.warning(f"Columns for {symbol}: {list(df.columns)}")
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    keep = ["open", "high", "low", "close"]
    if "volume" in df.columns:
        keep.append("volume")
    else:
        df["volume"] = 0
        keep.append("volume")

    df = df[keep].dropna(subset=["open", "high", "low", "close"])

    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    return df


def fetch_all() -> dict[str, int]:
    """Fetch 5 years of data for all symbols. Returns {symbol: row_count}."""
    results = {}

    # Indices
    for name, symbol in INDEX_SYMBOLS.items():
        df = fetch_index(name, symbol)
        if df is not None and len(df) > 0:
            out_path = DATA_DIR / f"{name}_5yr.csv"
            df.to_csv(out_path)
            results[name] = len(df)
            log.info(f"Saved {name}: {len(df)} rows -> {out_path}")
        else:
            log.error(f"FAILED: {name}")
            results[name] = 0

    # Stocks
    for symbol in STOCK_SYMBOLS:
        df = fetch_stock(symbol)
        name = symbol.lower()
        if df is not None and len(df) > 0:
            out_path = DATA_DIR / f"{name}_5yr.csv"
            df.to_csv(out_path)
            results[name] = len(df)
            log.info(f"Saved {name}: {len(df)} rows -> {out_path}")
        else:
            log.error(f"FAILED: {symbol}")
            results[name] = 0

    # Summary
    total = sum(results.values())
    log.info(f"\nFetch complete: {total} total rows across {len(results)} symbols")
    for sym, count in results.items():
        status = "OK" if count > 0 else "FAILED"
        log.info(f"  {sym:15} {count:>6} rows [{status}]")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    fetch_all()

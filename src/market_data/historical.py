"""
Historical Data Loader — Fetches 2 years of daily OHLCV data for NIFTY,
BANKNIFTY, and top FNO stocks. Uses Yahoo Finance as the primary free source,
with Groww API as a fallback for authenticated users.

Saves CSV files to data/historical/ for ML training.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger("josho.market_data.historical")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "historical"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Yahoo Finance ticker mapping for Indian instruments
YAHOO_TICKERS = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
}

# Top FNO stocks
FNO_STOCKS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "BHARTIARTL", "ITC", "TATAMOTORS", "BAJFINANCE",
]

ALL_SYMBOLS = ["NIFTY", "BANKNIFTY"] + FNO_STOCKS


def fetch_yahoo(symbol: str, years: int = 2) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV from Yahoo Finance using yf.download() (more reliable)."""
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed: pip install yfinance")
        return None

    yahoo_ticker = YAHOO_TICKERS.get(symbol)
    if yahoo_ticker is None:
        log.warning(f"No Yahoo Finance mapping for {symbol}")
        return None

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    log.info(f"Fetching {symbol} ({yahoo_ticker}) from Yahoo Finance...")
    try:
        df = yf.download(
            yahoo_ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=True,
        )

        if df is None or df.empty:
            log.warning(f"No data returned for {symbol} from Yahoo Finance")
            return None

        # yf.download returns MultiIndex columns like (Price, Ticker).
        # Flatten to simple column names.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower().replace(" ", "_") for c in df.columns]
        else:
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Keep only OHLCV columns
        required = ["open", "high", "low", "close", "volume"]
        available_cols = [c for c in required if c in df.columns]
        if len(available_cols) < 5:
            log.warning(f"Missing columns for {symbol}: have {available_cols}, need {required}")
            return None

        df = df[required].copy()
        df.index.name = "date"

        # Drop rows with zero volume (non-trading days that slipped in)
        df = df[df["volume"] > 0]

        log.info(f"  {symbol}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
        return df

    except Exception as e:
        log.error(f"Yahoo Finance fetch failed for {symbol}: {e}")
        return None


def fetch_groww(symbol: str, years: int = 2) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV from Groww API (requires authentication)."""
    try:
        from ..client import get_client
        client = get_client()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        exchange = "NSE"
        segment = "CASH"
        if symbol in ("NIFTY", "BANKNIFTY"):
            segment = "INDICES"

        data = client.get_historical(
            symbol=symbol,
            exchange=exchange,
            segment=segment,
            interval="1d",
            from_date=start_date.strftime("%Y-%m-%d"),
            to_date=end_date.strftime("%Y-%m-%d"),
        )

        if not data or "candles" not in data:
            log.warning(f"No Groww data for {symbol}")
            return None

        candles = data["candles"]
        df = pd.DataFrame(candles, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[df["volume"] > 0]

        log.info(f"  {symbol} (Groww): {len(df)} rows")
        return df

    except Exception as e:
        log.warning(f"Groww fetch failed for {symbol}: {e}")
        return None


def save_csv(df: pd.DataFrame, symbol: str) -> Path:
    """Save DataFrame to CSV in data/historical/."""
    path = DATA_DIR / f"{symbol.lower()}_daily.csv"
    df.to_csv(path)
    log.info(f"  Saved: {path} ({len(df)} rows)")
    return path


def load_csv(symbol: str) -> Optional[pd.DataFrame]:
    """Load historical CSV for a symbol."""
    path = DATA_DIR / f"{symbol.lower()}_daily.csv"
    if not path.exists():
        log.warning(f"No CSV found for {symbol}: {path}")
        return None

    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


def download_all(years: int = 2, use_groww: bool = False) -> dict:
    """
    Download historical data for all symbols.
    Primary source: Yahoo Finance (free, no auth needed).
    Fallback: Groww API (if use_groww=True and authenticated).

    Returns dict of {symbol: row_count}.
    """
    results = {}

    for symbol in ALL_SYMBOLS:
        # Try Yahoo Finance first
        df = fetch_yahoo(symbol, years)

        # Fallback to Groww if Yahoo fails and use_groww is enabled
        if df is None and use_groww:
            df = fetch_groww(symbol, years)

        if df is not None and not df.empty:
            save_csv(df, symbol)
            results[symbol] = len(df)
        else:
            log.error(f"FAILED to fetch {symbol} from any source")
            results[symbol] = 0

    return results


def get_available_data() -> dict:
    """Check which symbols have data saved locally."""
    available = {}
    for symbol in ALL_SYMBOLS:
        path = DATA_DIR / f"{symbol.lower()}_daily.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            available[symbol] = {
                "rows": len(df),
                "start": str(df.index[0].date()),
                "end": str(df.index[-1].date()),
            }
    return available


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    print("=== Historical Data Download ===")
    results = download_all(years=2)
    print("\n--- Results ---")
    total = 0
    for sym, count in results.items():
        status = f"{count} rows" if count > 0 else "FAILED"
        print(f"  {sym:15s} {status}")
        total += count
    print(f"\nTotal: {total} rows across {sum(1 for c in results.values() if c > 0)} symbols")

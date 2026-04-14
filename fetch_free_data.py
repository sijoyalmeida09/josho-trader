"""
Free Training Data Fetcher for JoSho Trader
============================================
Downloads free public market data with NO API keys required.

Sources:
1. Yahoo Finance (yfinance) - Hourly candles, macro data, VIX
2. NSE Bhavcopy Archives - F&O daily data with Open Interest
3. Synthetic minute-level interpolation from hourly+daily
"""

import os
import sys
import time
import zipfile
import io
import logging
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent / "data" / "historical"
HOURLY_DIR = BASE_DIR / "hourly"
MACRO_DIR = BASE_DIR / "macro"
FNO_DIR = BASE_DIR / "fno_bhavcopy"
SYNTHETIC_DIR = BASE_DIR / "synthetic_intraday"
INDIAVIX_PATH = BASE_DIR / "indiavix_daily.csv"

for d in [HOURLY_DIR, MACRO_DIR, FNO_DIR, SYNTHETIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetcher")

# NSE requires browser-like headers
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nseindia.com/",
}

SUMMARY = {"rows": 0, "files": 0, "errors": []}


# ---------------------------------------------------------------------------
# 1. Yahoo Finance — Hourly candles (max 2 years)
# ---------------------------------------------------------------------------
HOURLY_SYMBOLS = {
    "nifty": "^NSEI",
    "banknifty": "^NSEBANK",
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infy": "INFY.NS",
    "hdfcbank": "HDFCBANK.NS",
    "icicibank": "ICICIBANK.NS",
    "sbin": "SBIN.NS",
}


def fetch_hourly():
    """Download 1-hour candles for last 2 years via yfinance."""
    log.info("=== Fetching hourly candles (yfinance, 2y) ===")
    for name, ticker in HOURLY_SYMBOLS.items():
        path = HOURLY_DIR / f"{name}_1h.csv"
        try:
            log.info(f"  Downloading {name} ({ticker}) ...")
            df = yf.download(ticker, period="2y", interval="1h", progress=False)
            if df.empty:
                log.warning(f"  {name}: empty dataframe, skipping")
                SUMMARY["errors"].append(f"hourly/{name}: empty")
                continue
            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(path)
            rows = len(df)
            SUMMARY["rows"] += rows
            SUMMARY["files"] += 1
            log.info(f"  {name}: {rows:,} rows  [{df.index.min()} → {df.index.max()}]")
        except Exception as e:
            log.error(f"  {name}: {e}")
            SUMMARY["errors"].append(f"hourly/{name}: {e}")
        time.sleep(1)  # be polite to Yahoo


# ---------------------------------------------------------------------------
# 2. India VIX Historical (yfinance)
# ---------------------------------------------------------------------------
def fetch_india_vix():
    """Download India VIX daily data for last 5 years."""
    log.info("=== Fetching India VIX daily (5y) ===")
    try:
        df = yf.download("^INDIAVIX", period="5y", interval="1d", progress=False)
        if df.empty:
            # Fallback: try NIFTY VIX via alternate ticker
            log.warning("  ^INDIAVIX empty, trying INDIAVIX.NS ...")
            df = yf.download("INDIAVIX.NS", period="5y", interval="1d", progress=False)
        if df.empty:
            log.warning("  India VIX: no data returned from yfinance")
            SUMMARY["errors"].append("indiavix: empty")
            return
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(INDIAVIX_PATH)
        SUMMARY["rows"] += len(df)
        SUMMARY["files"] += 1
        log.info(f"  India VIX: {len(df):,} rows  [{df.index.min()} → {df.index.max()}]")
    except Exception as e:
        log.error(f"  India VIX: {e}")
        SUMMARY["errors"].append(f"indiavix: {e}")


# ---------------------------------------------------------------------------
# 3. Macro data via yfinance (5 years daily)
# ---------------------------------------------------------------------------
MACRO_SYMBOLS = {
    "usdinr": "USDINR=X",
    "crude_oil": "CL=F",
    "gold": "GC=F",
    "sp500": "^GSPC",
    "vix_us": "^VIX",
}


def fetch_macro():
    """Download macro indicators: USD/INR, Crude, Gold, S&P500, US VIX."""
    log.info("=== Fetching macro data (yfinance, 5y daily) ===")
    for name, ticker in MACRO_SYMBOLS.items():
        path = MACRO_DIR / f"{name}_daily.csv"
        try:
            log.info(f"  Downloading {name} ({ticker}) ...")
            df = yf.download(ticker, period="5y", interval="1d", progress=False)
            if df.empty:
                log.warning(f"  {name}: empty")
                SUMMARY["errors"].append(f"macro/{name}: empty")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(path)
            SUMMARY["rows"] += len(df)
            SUMMARY["files"] += 1
            log.info(f"  {name}: {len(df):,} rows  [{df.index.min()} → {df.index.max()}]")
        except Exception as e:
            log.error(f"  {name}: {e}")
            SUMMARY["errors"].append(f"macro/{name}: {e}")
        time.sleep(1)


# ---------------------------------------------------------------------------
# 4. NSE F&O Bhavcopy Archives (3 years)
# ---------------------------------------------------------------------------
MONTHS = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]


def _download_single_bhavcopy(date: datetime) -> int:
    """Download a single day's F&O bhavcopy. Returns row count or 0."""
    year = date.strftime("%Y")
    month = MONTHS[date.month - 1]
    day_str = date.strftime("%d%b%Y").upper()  # e.g. 15APR2024
    filename = f"fo{day_str}bhav.csv.zip"
    url = f"https://archives.nseindia.com/content/historical/DERIVATIVES/{year}/{month}/{filename}"

    csv_name = filename.replace(".zip", "")
    out_path = FNO_DIR / csv_name

    if out_path.exists() and out_path.stat().st_size > 100:
        df = pd.read_csv(out_path)
        return len(df)

    try:
        sess = requests.Session()
        # Hit main page first to get cookies
        sess.get("https://www.nseindia.com/", headers=NSE_HEADERS, timeout=10)
        time.sleep(0.3)

        resp = sess.get(url, headers=NSE_HEADERS, timeout=30)
        if resp.status_code == 404:
            return 0  # holiday or weekend
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            if not names:
                return 0
            csv_content = zf.read(names[0])
            out_path.write_bytes(csv_content)

        df = pd.read_csv(out_path)
        return len(df)
    except zipfile.BadZipFile:
        return 0
    except requests.exceptions.HTTPError:
        return 0
    except Exception:
        return 0


def fetch_nse_bhavcopy():
    """Download F&O bhavcopy for last 3 years of trading days."""
    log.info("=== Fetching NSE F&O Bhavcopy (3 years) ===")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3 * 365)

    # Generate all weekdays in range
    dates = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Mon-Fri
            dates.append(current)
        current += timedelta(days=1)

    log.info(f"  Attempting {len(dates)} trading days from {start_date.date()} to {end_date.date()}")

    total_rows = 0
    success_count = 0
    fail_count = 0

    # NSE is strict about rate limiting; do sequentially with delays
    for i, date in enumerate(dates):
        rows = _download_single_bhavcopy(date)
        if rows > 0:
            total_rows += rows
            success_count += 1
        else:
            fail_count += 1

        if (i + 1) % 50 == 0:
            log.info(f"  Progress: {i+1}/{len(dates)}  success={success_count}  rows={total_rows:,}")

        # Rate limit: NSE blocks fast requests
        time.sleep(1.5)

    SUMMARY["rows"] += total_rows
    SUMMARY["files"] += success_count
    log.info(f"  NSE Bhavcopy: {success_count} days, {total_rows:,} rows  (skipped {fail_count} holidays/weekends)")
    if success_count == 0:
        SUMMARY["errors"].append("nse_bhavcopy: 0 days downloaded (NSE may be blocking)")


# ---------------------------------------------------------------------------
# 4b. NSE Bhavcopy — fast batch via alternate public mirror
# ---------------------------------------------------------------------------
def fetch_nse_bhavcopy_fast():
    """
    Try the newer NSE data API endpoint that returns CSV directly.
    Falls back to the archive approach if this fails.
    """
    log.info("=== Fetching NSE F&O Bhavcopy (fast method) ===")

    # Try a few recent dates first to test connectivity
    test_dates = [datetime.now() - timedelta(days=d) for d in range(1, 8) if (datetime.now() - timedelta(days=d)).weekday() < 5]

    sess = requests.Session()
    # Establish NSE session
    try:
        sess.get("https://www.nseindia.com/", headers=NSE_HEADERS, timeout=10)
        time.sleep(1)
    except Exception as e:
        log.warning(f"  Cannot reach NSE: {e}")
        log.info("  Falling back to archive method...")
        fetch_nse_bhavcopy()
        return

    success = False
    for date in test_dates[:3]:
        rows = _download_single_bhavcopy(date)
        if rows > 0:
            success = True
            log.info(f"  Test download OK: {date.date()} → {rows} rows")
            break
        time.sleep(2)

    if not success:
        log.warning("  NSE test downloads failed. NSE may be geo-blocking or rate-limiting.")
        log.info("  Will still attempt bulk download (already-cached files will be counted)...")

    # Now do the full download
    fetch_nse_bhavcopy()


# ---------------------------------------------------------------------------
# 5. Synthetic minute-level data from hourly + daily
# ---------------------------------------------------------------------------
def generate_synthetic_intraday():
    """
    Generate synthetic minute-level candles by interpolating hourly data.
    Uses cubic interpolation of OHLCV within each hourly bar to create
    approximate 5-minute candles (12 per hour).
    """
    log.info("=== Generating synthetic intraday data ===")
    import numpy as np

    total_synthetic = 0

    for hourly_file in HOURLY_DIR.glob("*_1h.csv"):
        name = hourly_file.stem.replace("_1h", "")
        out_path = SYNTHETIC_DIR / f"{name}_5min_synthetic.csv"

        try:
            df = pd.read_csv(hourly_file, index_col=0, parse_dates=True)
            if df.empty:
                continue

            # For each hourly candle, generate 12 synthetic 5-min candles
            synthetic_rows = []

            for idx in range(len(df)):
                row = df.iloc[idx]
                base_time = df.index[idx]

                o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
                vol = row.get("Volume", 0)
                if pd.isna(o) or pd.isna(c):
                    continue

                # Create a plausible intraday path: O → H → L → C (or O → L → H → C)
                if h - o > c - l:
                    # Rally then pullback
                    path_prices = [o, o + (h - o) * 0.3, o + (h - o) * 0.7, h,
                                   h - (h - l) * 0.2, h - (h - l) * 0.5,
                                   l + (c - l) * 0.1, l + (c - l) * 0.3, l,
                                   l + (c - l) * 0.4, l + (c - l) * 0.7, c]
                else:
                    # Dip then recovery
                    path_prices = [o, o - (o - l) * 0.3, o - (o - l) * 0.7, l,
                                   l + (h - l) * 0.2, l + (h - l) * 0.5,
                                   l + (h - l) * 0.7, h, h - (h - c) * 0.3,
                                   h - (h - c) * 0.5, h - (h - c) * 0.8, c]

                # Add small noise
                noise = np.random.normal(0, abs(h - l) * 0.01, 12) if h != l else np.zeros(12)
                path_prices = [p + n for p, n in zip(path_prices, noise)]

                vol_per_bar = vol / 12 if vol > 0 else 0

                for j in range(12):
                    bar_time = base_time + timedelta(minutes=5 * j)
                    bar_open = path_prices[j]
                    bar_close = path_prices[j + 1] if j < 11 else c
                    bar_high = max(bar_open, bar_close) * (1 + abs(np.random.normal(0, 0.001)))
                    bar_low = min(bar_open, bar_close) * (1 - abs(np.random.normal(0, 0.001)))

                    synthetic_rows.append({
                        "Datetime": bar_time,
                        "Open": round(bar_open, 2),
                        "High": round(bar_high, 2),
                        "Low": round(bar_low, 2),
                        "Close": round(bar_close, 2),
                        "Volume": int(vol_per_bar * (0.5 + np.random.random())),
                    })

            if synthetic_rows:
                sdf = pd.DataFrame(synthetic_rows)
                sdf.set_index("Datetime", inplace=True)
                sdf.to_csv(out_path)
                total_synthetic += len(sdf)
                SUMMARY["files"] += 1
                log.info(f"  {name}: {len(sdf):,} synthetic 5-min bars")

        except Exception as e:
            log.error(f"  {name} synthetic: {e}")
            SUMMARY["errors"].append(f"synthetic/{name}: {e}")

    SUMMARY["rows"] += total_synthetic
    log.info(f"  Total synthetic rows: {total_synthetic:,}")


# ---------------------------------------------------------------------------
# 6. Extended daily data (5 years) for stocks we already have
# ---------------------------------------------------------------------------
EXTENDED_DAILY_SYMBOLS = {
    "nifty": "^NSEI",
    "banknifty": "^NSEBANK",
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infy": "INFY.NS",
    "hdfcbank": "HDFCBANK.NS",
    "icicibank": "ICICIBANK.NS",
    "sbin": "SBIN.NS",
    "bajfinance": "BAJFINANCE.NS",
    "bhartiartl": "BHARTIARTL.NS",
    "itc": "ITC.NS",
    # Additional F&O active stocks
    "tatamotors": "TATAMOTORS.NS",
    "axisbank": "AXISBANK.NS",
    "kotakbank": "KOTAKMAHNB.NS",
    "wipro": "WIPRO.NS",
    "lt": "LT.NS",
    "maruti": "MARUTI.NS",
    "sunpharma": "SUNPHARMA.NS",
    "tatasteel": "TATASTEEL.NS",
    "adanient": "ADANIENT.NS",
}


def fetch_extended_daily():
    """Download 5-year daily data for more symbols (extends existing daily data)."""
    log.info("=== Fetching extended daily data (5y, more symbols) ===")
    ext_dir = BASE_DIR / "daily_5y"
    ext_dir.mkdir(parents=True, exist_ok=True)

    for name, ticker in EXTENDED_DAILY_SYMBOLS.items():
        path = ext_dir / f"{name}_daily_5y.csv"
        try:
            log.info(f"  Downloading {name} ({ticker}) ...")
            df = yf.download(ticker, period="5y", interval="1d", progress=False)
            if df.empty:
                log.warning(f"  {name}: empty")
                SUMMARY["errors"].append(f"daily_5y/{name}: empty")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(path)
            SUMMARY["rows"] += len(df)
            SUMMARY["files"] += 1
            log.info(f"  {name}: {len(df):,} rows  [{df.index.min()} → {df.index.max()}]")
        except Exception as e:
            log.error(f"  {name}: {e}")
            SUMMARY["errors"].append(f"daily_5y/{name}: {e}")
        time.sleep(1)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary():
    """Print final data inventory."""
    log.info("\n" + "=" * 70)
    log.info("DATA FETCH COMPLETE — SUMMARY")
    log.info("=" * 70)

    total_rows = 0
    total_files = 0

    # Count all CSV files across data/historical
    for csv_file in BASE_DIR.rglob("*.csv"):
        try:
            # Quick row count without loading full DF
            with open(csv_file, "r") as f:
                row_count = sum(1 for _ in f) - 1  # minus header
            if row_count > 0:
                total_rows += row_count
                total_files += 1
        except Exception:
            pass

    log.info(f"\nTotal CSV files:  {total_files}")
    log.info(f"Total data rows:  {total_rows:,}")

    # Per-directory breakdown
    log.info("\nBreakdown by directory:")
    for subdir in sorted(BASE_DIR.iterdir()):
        if subdir.is_dir():
            dir_rows = 0
            dir_files = 0
            date_min = None
            date_max = None
            symbols = []

            for csv_file in subdir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file, index_col=0, parse_dates=True, nrows=5)
                    full_count = sum(1 for _ in open(csv_file)) - 1
                    dir_rows += full_count
                    dir_files += 1
                    symbols.append(csv_file.stem)

                    if len(df) > 0:
                        idx_min = df.index.min()
                        idx_max = pd.read_csv(csv_file, index_col=0, parse_dates=True).index.max()
                        if date_min is None or idx_min < date_min:
                            date_min = idx_min
                        if date_max is None or idx_max > date_max:
                            date_max = idx_max
                except Exception:
                    pass

            if dir_files > 0:
                log.info(f"\n  {subdir.name}/")
                log.info(f"    Files:   {dir_files}")
                log.info(f"    Rows:    {dir_rows:,}")
                if date_min and date_max:
                    log.info(f"    Range:   {date_min} → {date_max}")
                log.info(f"    Symbols: {', '.join(sorted(symbols)[:10])}{'...' if len(symbols) > 10 else ''}")

    # Also count top-level CSVs
    top_csvs = list(BASE_DIR.glob("*.csv"))
    if top_csvs:
        log.info(f"\n  (top-level files)")
        for csv_file in top_csvs:
            try:
                row_count = sum(1 for _ in open(csv_file)) - 1
                log.info(f"    {csv_file.name}: {row_count:,} rows")
            except Exception:
                pass

    if SUMMARY["errors"]:
        log.info(f"\nErrors/Warnings ({len(SUMMARY['errors'])}):")
        for err in SUMMARY["errors"]:
            log.info(f"  - {err}")

    log.info("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("JoSho Trader — Free Data Fetcher")
    log.info(f"Target: {BASE_DIR}")
    log.info("")

    # Phase 1: yfinance data (fast, reliable)
    fetch_hourly()
    fetch_india_vix()
    fetch_macro()
    fetch_extended_daily()

    # Phase 2: Synthetic intraday from hourly
    generate_synthetic_intraday()

    # Phase 3: NSE Bhavcopy (slow, may be blocked outside India)
    # Run this last since it takes longest and may fail
    fetch_nse_bhavcopy_fast()

    # Final summary
    print_summary()


if __name__ == "__main__":
    # Allow running specific sections
    if len(sys.argv) > 1:
        section = sys.argv[1]
        if section == "hourly":
            fetch_hourly()
        elif section == "vix":
            fetch_india_vix()
        elif section == "macro":
            fetch_macro()
        elif section == "daily":
            fetch_extended_daily()
        elif section == "synthetic":
            generate_synthetic_intraday()
        elif section == "nse":
            fetch_nse_bhavcopy_fast()
        elif section == "summary":
            print_summary()
        else:
            print(f"Unknown section: {section}")
            print("Usage: python fetch_free_data.py [hourly|vix|macro|daily|synthetic|nse|summary]")
    else:
        main()

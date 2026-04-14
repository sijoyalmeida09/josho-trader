import yfinance as yf
import json, os, time
from pathlib import Path

OUTPUT = Path("data/historical/fno_all")
OUTPUT.mkdir(parents=True, exist_ok=True)

stocks = json.loads(Path("data/fno_stocks.json").read_text())

# Some NSE symbols need special handling for yfinance
TICKER_MAP = {
    "M&M": "M%26M.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "NAM-INDIA": "NAM-INDIA.NS",
    "360ONE": "360ONE.NS",
}

failed = []
skipped = []
downloaded = []

for i, symbol in enumerate(stocks):
    outfile = OUTPUT / f"{symbol}.csv"
    if outfile.exists() and os.path.getsize(str(outfile)) > 1000:
        print(f"[{i+1}/{len(stocks)}] SKIP {symbol} (exists)")
        skipped.append(symbol)
        continue

    ticker = TICKER_MAP.get(symbol, f"{symbol}.NS")
    try:
        df = yf.download(ticker, period="max", interval="1d", progress=False)
        if len(df) > 100:
            df.to_csv(outfile)
            print(f"[{i+1}/{len(stocks)}] {symbol}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
            downloaded.append((symbol, len(df)))
        else:
            # Try alternate ticker formats
            alt_tickers = [f"{symbol}.NS", symbol.replace("&", "%26") + ".NS"]
            found = False
            for alt in alt_tickers:
                if alt == ticker:
                    continue
                df2 = yf.download(alt, period="max", interval="1d", progress=False)
                if len(df2) > 100:
                    df2.to_csv(outfile)
                    print(f"[{i+1}/{len(stocks)}] {symbol} (alt {alt}): {len(df2)} rows ({df2.index[0].date()} to {df2.index[-1].date()})")
                    downloaded.append((symbol, len(df2)))
                    found = True
                    break
            if not found:
                print(f"[{i+1}/{len(stocks)}] {symbol}: only {len(df)} rows, FAILED")
                failed.append(symbol)
    except Exception as e:
        print(f"[{i+1}/{len(stocks)}] {symbol}: FAILED - {e}")
        failed.append(symbol)

    time.sleep(0.3)

# Summary
total_files = sum(1 for f in OUTPUT.glob("*.csv"))
total_rows = 0
for f in OUTPUT.glob("*.csv"):
    with open(f) as fh:
        total_rows += len(fh.readlines()) - 1  # subtract header

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Total stocks in list: {len(stocks)}")
print(f"Downloaded this run:  {len(downloaded)}")
print(f"Skipped (existing):   {len(skipped)}")
print(f"Failed:               {len(failed)}")
print(f"Total CSV files:      {total_files}")
print(f"Total data rows:      {total_rows:,}")
if failed:
    print(f"\nFailed symbols: {', '.join(failed)}")

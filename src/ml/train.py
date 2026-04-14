"""
ML Training Pipeline — Load historical CSVs, engineer features, train XGBoost
models for NIFTY and BANKNIFTY, save to data/models/.

Usage:
    python -m src.ml.train
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from ..market_data.historical import load_csv, download_all, get_available_data, ALL_SYMBOLS
from .signal_model import XGBoostSignalModel

log = logging.getLogger("josho.ml.train")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "historical"


def ensure_data() -> bool:
    """Download data if not already present."""
    available = get_available_data()

    # Need at least NIFTY and BANKNIFTY
    missing = [s for s in ["NIFTY", "BANKNIFTY"] if s not in available]
    if missing:
        log.info(f"Missing data for: {missing}. Downloading...")
        results = download_all(years=2)
        failed = [s for s, c in results.items() if c == 0]
        if "NIFTY" in failed or "BANKNIFTY" in failed:
            log.error("Failed to download critical data (NIFTY/BANKNIFTY)")
            return False
    else:
        log.info(f"Data available: {list(available.keys())}")

    return True


def train_model(symbol: str, forward_periods: int = 5) -> dict:
    """Train XGBoost model for a single symbol."""
    df = load_csv(symbol)
    if df is None:
        return {"error": f"No data for {symbol}"}

    if len(df) < 200:
        return {"error": f"Not enough data for {symbol}: {len(df)} rows (need 200+)"}

    log.info(f"\n{'='*60}")
    log.info(f"Training model: {symbol}")
    log.info(f"Data: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
    log.info(f"{'='*60}")

    model = XGBoostSignalModel(model_name=symbol.lower())
    result = model.train(df, forward_periods=forward_periods)

    return result


def print_results(symbol: str, result: dict):
    """Pretty-print training results."""
    if "error" in result:
        print(f"\n  {symbol}: ERROR — {result['error']}")
        return

    print(f"\n  {'='*55}")
    print(f"  Model: {symbol}")
    print(f"  {'='*55}")
    print(f"  Samples:         {result.get('samples', 'N/A')}")
    print(f"  Features:        {result.get('features', 'N/A')}")
    print(f"  Train Accuracy:  {result.get('train_accuracy', 'N/A')}")
    print(f"  Test Accuracy:   {result.get('test_accuracy', 'N/A')}")
    print(f"  Forward Periods: {result.get('forward_periods', 'N/A')}")
    print(f"  Trained At:      {result.get('trained_at', 'N/A')}")

    top = result.get("top_features", {})
    if top:
        print(f"\n  Top 15 Features:")
        for i, (feat, imp) in enumerate(top.items(), 1):
            bar = "#" * int(imp * 200)
            print(f"    {i:2d}. {feat:25s} {imp:.4f}  {bar}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    print("\n" + "=" * 60)
    print("  JoSho Trader — ML Training Pipeline")
    print("=" * 60)

    # Step 1: Ensure data exists
    print("\n[1/3] Checking historical data...")
    if not ensure_data():
        print("FATAL: Cannot proceed without data.")
        sys.exit(1)

    available = get_available_data()
    for sym, info in available.items():
        print(f"  {sym:15s} {info['rows']:>5d} rows  ({info['start']} to {info['end']})")

    # Step 2: Train primary models (NIFTY + BANKNIFTY)
    print("\n[2/3] Training primary models (NIFTY, BANKNIFTY)...")
    primary_results = {}
    for symbol in ["NIFTY", "BANKNIFTY"]:
        result = train_model(symbol, forward_periods=5)
        primary_results[symbol] = result
        print_results(symbol, result)

    # Step 3: Train stock models (optional, for correlation signals)
    print("\n[3/3] Training FNO stock models...")
    stock_results = {}
    for symbol in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                    "SBIN", "BHARTIARTL", "ITC", "TATAMOTORS", "BAJFINANCE"]:
        if symbol in available:
            result = train_model(symbol, forward_periods=5)
            stock_results[symbol] = result
            print_results(symbol, result)
        else:
            print(f"\n  {symbol}: SKIPPED (no data)")

    # Summary
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)

    all_results = {**primary_results, **stock_results}
    for sym, res in all_results.items():
        if "error" in res:
            print(f"  {sym:15s} FAILED: {res['error']}")
        else:
            print(f"  {sym:15s} train={res['train_accuracy']:.4f}  test={res['test_accuracy']:.4f}  samples={res['samples']}")

    # Check model files
    model_dir = Path(__file__).parent.parent.parent / "data" / "models"
    model_files = list(model_dir.glob("xgb_*.pkl"))
    print(f"\n  Models saved: {len(model_files)} files in {model_dir}")
    for f in sorted(model_files):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:30s} {size_kb:.1f} KB")


if __name__ == "__main__":
    main()

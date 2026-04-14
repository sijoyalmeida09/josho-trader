"""
Batch Stock Predictability Scorer — score ALL stocks and rank them.

Loads every CSV from a data directory, runs the universal predictor on each,
and outputs a sorted ranking to CSV.
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd

from .universal_predictor import score_stock_predictability

log = logging.getLogger("josho.ml.batch_scorer")

# Default data directories to scan (in priority order)
DEFAULT_DATA_DIRS = [
    "data/historical/fno_all",
    "data/historical/daily_5y",
    "data/historical",
]

RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "results"


def _find_data_dir(data_dir: str) -> Path:
    """Resolve data directory, checking project-relative and absolute paths."""
    project_root = Path(__file__).parent.parent.parent

    # Try the provided path first
    candidate = Path(data_dir)
    if candidate.is_absolute() and candidate.is_dir():
        return candidate

    candidate = project_root / data_dir
    if candidate.is_dir():
        return candidate

    # Fall through defaults
    for fallback in DEFAULT_DATA_DIRS:
        candidate = project_root / fallback
        if candidate.is_dir() and list(candidate.glob("*.csv")):
            return candidate

    raise FileNotFoundError(
        f"No data directory found. Tried: {data_dir} and defaults {DEFAULT_DATA_DIRS}"
    )


def _extract_symbol(csv_path: Path) -> str:
    """Extract stock symbol from filename, stripping common suffixes."""
    name = csv_path.stem.upper()
    for suffix in ["_DAILY_5Y", "_5YR", "_DAILY", "_5Y"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def score_all_stocks(
    data_dir: str = "data/historical/fno_all",
    output_csv: str = "data/results/stock_predictability_ranking.csv",
) -> pd.DataFrame:
    """
    Score ALL stocks in a directory and return sorted DataFrame.

    Args:
        data_dir: Path to directory containing per-stock CSV files.
        output_csv: Where to save the ranking (relative to project root).

    Returns:
        DataFrame sorted by predictability_score descending.
    """
    resolved_dir = _find_data_dir(data_dir)
    csv_files = sorted(resolved_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {resolved_dir}")
        return pd.DataFrame()

    total = len(csv_files)
    print(f"Found {total} stocks in {resolved_dir}")
    print("=" * 70)

    results = []
    start_all = time.time()

    for idx, csv_path in enumerate(csv_files, 1):
        symbol = _extract_symbol(csv_path)
        start_one = time.time()

        try:
            raw_df = pd.read_csv(csv_path)

            if len(raw_df) < 100:
                print(f"[{idx}/{total}] {symbol}: SKIPPED (only {len(raw_df)} rows)")
                continue

            result = score_stock_predictability(raw_df, symbol)
            elapsed = time.time() - start_one

            score = result["predictability_score"]
            rec = result["recommendation"]
            print(
                f"[{idx}/{total}] {symbol}: "
                f"score={score:.1f} {rec} "
                f"(ML={result['best_ml_accuracy']:.3f}/{result['best_ml_auc']:.3f} "
                f"LD={result['large_drop_winrate']:.3f}x{result['large_drop_trades']} "
                f"RSI2={result['rsi2_winrate']:.3f}x{result['rsi2_trades']}) "
                f"[{elapsed:.1f}s]"
            )

            # Flatten for DataFrame (remove nested detail)
            flat = {k: v for k, v in result.items() if k != "ml_detail"}
            results.append(flat)

        except Exception as exc:
            elapsed = time.time() - start_one
            print(f"[{idx}/{total}] {symbol}: ERROR — {exc} [{elapsed:.1f}s]")
            continue

    total_time = time.time() - start_all
    print("=" * 70)
    print(f"Scored {len(results)}/{total} stocks in {total_time:.1f}s")

    if not results:
        print("No results to save.")
        return pd.DataFrame()

    df_results = pd.DataFrame(results).sort_values(
        "predictability_score", ascending=False,
    ).reset_index(drop=True)

    # Save to CSV
    project_root = Path(__file__).parent.parent.parent
    out_path = project_root / output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"Saved ranking to {out_path}")

    # Print top 10
    print("\n--- TOP 10 MOST PREDICTABLE ---")
    top_cols = [
        "symbol", "predictability_score", "recommendation",
        "best_ml_algo", "best_ml_accuracy", "best_ml_auc",
        "large_drop_winrate", "large_drop_trades",
        "rsi2_winrate", "rsi2_trades",
    ]
    display_cols = [c for c in top_cols if c in df_results.columns]
    print(df_results.head(10)[display_cols].to_string(index=False))

    # Print bottom 5
    print("\n--- BOTTOM 5 LEAST PREDICTABLE ---")
    print(df_results.tail(5)[display_cols].to_string(index=False))

    # Summary stats
    print(f"\nSummary:")
    print(f"  HIGHLY_PREDICTABLE (>75): {(df_results['predictability_score'] >= 75).sum()}")
    print(f"  MODERATE (60-75):         {((df_results['predictability_score'] >= 60) & (df_results['predictability_score'] < 75)).sum()}")
    print(f"  LOW (<60):                {(df_results['predictability_score'] < 60).sum()}")
    print(f"  Average score:            {df_results['predictability_score'].mean():.1f}")
    print(f"  Median score:             {df_results['predictability_score'].median():.1f}")

    return df_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/historical/fno_all"
    score_all_stocks(data_dir=data_dir)

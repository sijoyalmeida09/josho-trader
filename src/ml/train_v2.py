"""Train V2 models on all symbols and compare with V1."""

import logging
import json
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_v2")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "historical"
MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"


def train_all():
    from .signal_model_v2 import XGBoostSignalModelV2

    symbols = [
        "nifty", "banknifty", "reliance", "tcs", "infy",
        "hdfcbank", "icicibank", "sbin", "bajfinance", "itc", "bhartiartl",
    ]

    results = []
    for symbol in symbols:
        csv_path = DATA_DIR / f"{symbol}_daily.csv"
        if not csv_path.exists():
            log.warning(f"No data for {symbol}")
            continue

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if len(df) < 100:
            log.warning(f"Not enough data for {symbol}: {len(df)} rows")
            continue

        model = XGBoostSignalModelV2(model_name=symbol)
        meta = model.train(df, forward_periods=5)

        if "error" in meta:
            log.error(f"{symbol}: {meta['error']}")
            continue

        results.append(meta)
        log.info(
            f"{symbol.upper():12} | "
            f"Acc: {meta['test_accuracy']:.3f} | "
            f"AUC: {meta['auc']:.3f} | "
            f"P: {meta['precision']:.3f} | "
            f"R: {meta['recall']:.3f} | "
            f"F1: {meta['f1']:.3f} | "
            f"Features: {meta['features_selected']}"
        )

    # Compare V1 vs V2
    print("\n" + "=" * 70)
    print("V1 vs V2 COMPARISON")
    print("=" * 70)
    print(f"{'Symbol':12} | {'V1 Acc':>8} | {'V2 Acc':>8} | {'V2 AUC':>8} | {'V2 F1':>8} | {'Improvement':>12}")
    print("-" * 70)

    for meta in results:
        name = meta["model_name"]
        v1_meta_path = MODEL_DIR / f"xgb_{name}_meta.json"
        v1_acc = 0
        if v1_meta_path.exists():
            v1 = json.loads(v1_meta_path.read_text())
            v1_acc = v1.get("test_accuracy", 0)

        v2_acc = meta["test_accuracy"]
        improvement = v2_acc - v1_acc
        print(
            f"{name.upper():12} | "
            f"{v1_acc:>7.3f} | "
            f"{v2_acc:>7.3f} | "
            f"{meta['auc']:>7.3f} | "
            f"{meta['f1']:>7.3f} | "
            f"{improvement:>+10.3f} {'BETTER' if improvement > 0 else 'WORSE'}"
        )

    print("=" * 70)


if __name__ == "__main__":
    train_all()

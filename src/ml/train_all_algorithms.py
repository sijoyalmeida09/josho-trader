"""
Train ALL ML algorithms on historical data and compare results.

Usage:
    python -m src.ml.train_all_algorithms
    python -m src.ml.train_all_algorithms --symbol ICICIBANK
    python -m src.ml.train_all_algorithms --all-symbols
"""

import argparse
import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .algorithms import (
    ALGORITHM_REGISTRY,
    get_algorithm,
    get_ensemble_algorithms,
    get_individual_algorithms,
    try_add_catboost,
)
from .signal_model_v2 import (
    create_binary_labels,
    engineer_features_v2,
    purged_train_test_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("train_all")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "historical"
MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Data Preparation ─────────────────────────────────────────────

def load_and_prepare(
    symbol: str,
    forward_periods: int = 5,
    threshold: float = 0.003,
):
    """
    Load CSV, engineer features, create labels, purged split.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scale_pos_weight, info_dict)
        or None if data is insufficient.
    """
    csv_path = DATA_DIR / f"{symbol.lower()}_daily.csv"
    if not csv_path.exists():
        log.error(f"No data file: {csv_path}")
        return None

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if len(df) < 120:
        log.error(f"Not enough data for {symbol}: {len(df)} rows")
        return None

    features = engineer_features_v2(df)
    labels = create_binary_labels(df, forward_periods, threshold)

    # Warmup (60 bars for indicators) + drop future-less rows
    valid_features = features.iloc[60:-forward_periods].copy()
    valid_labels = labels.iloc[60:-forward_periods].copy()

    # Drop NaN labels (neutral zone)
    mask = valid_features.notna().all(axis=1) & valid_labels.notna()
    X = valid_features[mask]
    y = valid_labels[mask]

    if len(X) < 80:
        log.error(f"Too few valid samples for {symbol}: {len(X)}")
        return None

    X_train, X_test, y_train, y_test = purged_train_test_split(
        X, y, test_ratio=0.2, purge_gap=10,
    )

    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale = neg_count / pos_count if pos_count > 0 else 1.0

    info = {
        "symbol": symbol.upper(),
        "total_rows": len(df),
        "valid_samples": len(X),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "up_count": int((y == 1).sum()),
        "down_count": int((y == 0).sum()),
        "scale_pos_weight": round(float(scale), 4),
        "n_features": X.shape[1],
    }

    log.info(
        f"{symbol.upper()}: {len(X)} samples "
        f"(train={len(X_train)}, test={len(X_test)}, "
        f"UP={info['up_count']}, DOWN={info['down_count']})"
    )

    return X_train, X_test, y_train, y_test, scale, info


# ── Training ─────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """Compute all metrics for a fitted model."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

    # AUC requires predict_proba
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["auc"] = round(float(roc_auc_score(y_test, y_proba)), 4)
    except Exception:
        metrics["auc"] = 0.0

    return metrics


def train_single_algorithm(
    name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scale_pos_weight: float,
):
    """
    Train one algorithm, return metrics dict.
    Returns None on failure.
    """
    start = time.time()

    try:
        model = get_algorithm(name, scale_pos_weight)
    except Exception as exc:
        log.warning(f"  {name}: failed to create — {exc}")
        return None

    try:
        # XGBoost needs eval_set for early stopping
        entry = ALGORITHM_REGISTRY[name]
        if entry.get("needs_early_stop"):
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)
    except Exception as exc:
        log.warning(f"  {name}: training failed — {exc}")
        return None

    elapsed = time.time() - start
    metrics = evaluate_model(model, X_test, y_test)
    metrics["train_time_sec"] = round(elapsed, 2)
    metrics["algorithm"] = name
    metrics["description"] = entry["description"]

    log.info(
        f"  {name:22} | "
        f"Acc: {metrics['accuracy']:.3f} | "
        f"AUC: {metrics['auc']:.3f} | "
        f"F1: {metrics['f1']:.3f} | "
        f"Time: {elapsed:.1f}s"
    )

    return {"model": model, "metrics": metrics}


def train_catboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scale_pos_weight: float,
):
    """Try CatBoost if installed."""
    start = time.time()
    model, available = try_add_catboost(scale_pos_weight)
    if not available:
        log.info("  catboost: not installed, skipping")
        return None

    try:
        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    except Exception as exc:
        log.warning(f"  catboost: training failed — {exc}")
        return None

    elapsed = time.time() - start
    metrics = evaluate_model(model, X_test, y_test)
    metrics["train_time_sec"] = round(elapsed, 2)
    metrics["algorithm"] = "catboost"
    metrics["description"] = "CatBoost"

    log.info(
        f"  {'catboost':22} | "
        f"Acc: {metrics['accuracy']:.3f} | "
        f"AUC: {metrics['auc']:.3f} | "
        f"F1: {metrics['f1']:.3f} | "
        f"Time: {elapsed:.1f}s"
    )

    return {"model": model, "metrics": metrics}


def build_custom_stacking(
    results: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scale_pos_weight: float,
    top_n: int = 4,
):
    """
    Build a Stacking Ensemble from the best N individual models.
    Uses fresh estimator clones (not already-fitted ones).
    """
    # Rank individual results by AUC
    individual = {
        k: v for k, v in results.items()
        if k not in get_ensemble_algorithms() and k != "catboost"
    }
    ranked = sorted(
        individual.items(),
        key=lambda x: x[1]["metrics"]["auc"],
        reverse=True,
    )[:top_n]

    if len(ranked) < 2:
        log.warning("  Not enough individual models for custom stacking")
        return None

    best_names = [name for name, _ in ranked]
    log.info(f"  Stacking top-{top_n}: {best_names}")

    # Build fresh estimators for stacking
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression

    base_estimators = []
    for name in best_names:
        est = get_algorithm(name, scale_pos_weight)
        base_estimators.append((name, est))

    stacker = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )

    start = time.time()
    try:
        stacker.fit(X_train, y_train)
    except Exception as exc:
        log.warning(f"  custom_stacking: training failed — {exc}")
        return None

    elapsed = time.time() - start
    metrics = evaluate_model(stacker, X_test, y_test)
    metrics["train_time_sec"] = round(elapsed, 2)
    metrics["algorithm"] = "stacking_best4"
    metrics["description"] = f"Stacking (top-{top_n}: {', '.join(best_names)})"

    log.info(
        f"  {'stacking_best4':22} | "
        f"Acc: {metrics['accuracy']:.3f} | "
        f"AUC: {metrics['auc']:.3f} | "
        f"F1: {metrics['f1']:.3f} | "
        f"Time: {elapsed:.1f}s"
    )

    return {"model": stacker, "metrics": metrics}


# ── Main Pipeline ────────────────────────────────────────────────

def train_all_for_symbol(symbol: str) -> dict:
    """Train all algorithms for one symbol. Returns results dict."""
    prepared = load_and_prepare(symbol)
    if prepared is None:
        return {}

    X_train, X_test, y_train, y_test, scale, info = prepared

    print(f"\n{'=' * 78}")
    print(f"  TRAINING ALL ALGORITHMS — {info['symbol']}")
    print(f"  Samples: {info['valid_samples']} | "
          f"Train: {info['train_size']} | Test: {info['test_size']} | "
          f"Features: {info['n_features']}")
    print(f"{'=' * 78}\n")

    results = {}

    # 1. Train all individual algorithms
    log.info("Phase 1: Individual algorithms")
    for algo_name in get_individual_algorithms():
        result = train_single_algorithm(
            algo_name, X_train, X_test, y_train, y_test, scale,
        )
        if result is not None:
            results[algo_name] = result

    # 2. Try CatBoost
    log.info("Phase 2: CatBoost (optional)")
    catboost_result = train_catboost(X_train, X_test, y_train, y_test, scale)
    if catboost_result is not None:
        results["catboost"] = catboost_result

    # 3. Default ensembles (stacking + voting with default base estimators)
    log.info("Phase 3: Default ensembles")
    for algo_name in get_ensemble_algorithms():
        result = train_single_algorithm(
            algo_name, X_train, X_test, y_train, y_test, scale,
        )
        if result is not None:
            results[algo_name] = result

    # 4. Custom stacking with best 4 individual models
    log.info("Phase 4: Custom stacking (best 4 individuals)")
    custom_stack = build_custom_stacking(
        results, X_train, X_test, y_train, y_test, scale, top_n=4,
    )
    if custom_stack is not None:
        results["stacking_best4"] = custom_stack

    # 5. Save the best model
    best_name = max(
        results.keys(),
        key=lambda k: results[k]["metrics"]["auc"],
    )
    best_result = results[best_name]
    best_model_path = MODEL_DIR / f"best_{symbol.lower()}.pkl"
    best_meta_path = MODEL_DIR / f"best_{symbol.lower()}_meta.json"

    with open(best_model_path, "wb") as fp:
        pickle.dump({
            "model": best_result["model"],
            "algorithm": best_name,
            "features": list(X_train.columns),
        }, fp)

    meta = {
        **best_result["metrics"],
        **info,
        "saved_at": datetime.now().isoformat(),
    }
    best_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(f"\nBest model saved: {best_name} -> {best_model_path}")

    # 6. Print comparison table
    _print_comparison_table(results, info)

    return results


def _print_comparison_table(results: dict, info: dict):
    """Print formatted comparison table."""
    print(f"\n{'=' * 90}")
    print(f"  ALGORITHM COMPARISON — {info['symbol']}")
    print(f"{'=' * 90}")
    header = (
        f"{'Algorithm':<25} | {'Accuracy':>8} | {'AUC':>7} | "
        f"{'F1':>7} | {'Precision':>9} | {'Recall':>7} | {'Time':>6}"
    )
    print(header)
    print("-" * 90)

    # Sort by AUC descending
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["metrics"]["auc"],
        reverse=True,
    )

    for name, result in sorted_results:
        m = result["metrics"]
        is_best = name == sorted_results[0][0]
        marker = " ***" if is_best else ""
        print(
            f"{m['description']:<25} | "
            f"{m['accuracy']:>7.1%} | "
            f"{m['auc']:>7.4f} | "
            f"{m['f1']:>7.4f} | "
            f"{m['precision']:>9.4f} | "
            f"{m['recall']:>7.4f} | "
            f"{m['train_time_sec']:>5.1f}s{marker}"
        )

    print("-" * 90)
    best_name, best = sorted_results[0]
    print(f"  BEST: {best['metrics']['description']} "
          f"(AUC={best['metrics']['auc']:.4f}, Acc={best['metrics']['accuracy']:.1%})")
    print(f"{'=' * 90}\n")


def train_all_symbols():
    """Train all algorithms on all available symbols."""
    csvs = sorted(DATA_DIR.glob("*_daily.csv"))
    if not csvs:
        log.error(f"No CSV files found in {DATA_DIR}")
        return

    all_results = {}
    for csv_path in csvs:
        symbol = csv_path.stem.replace("_daily", "")
        results = train_all_for_symbol(symbol)
        if results:
            all_results[symbol] = results

    # Summary across all symbols
    if all_results:
        _print_cross_symbol_summary(all_results)


def _print_cross_symbol_summary(all_results: dict):
    """Print which algorithm won for each symbol."""
    print(f"\n{'=' * 70}")
    print("  CROSS-SYMBOL SUMMARY: BEST ALGORITHM PER SYMBOL")
    print(f"{'=' * 70}")
    print(f"{'Symbol':<14} | {'Best Algorithm':<25} | {'Accuracy':>8} | {'AUC':>7}")
    print("-" * 70)

    algo_wins = {}
    for symbol, results in all_results.items():
        best_name = max(results.keys(), key=lambda k: results[k]["metrics"]["auc"])
        m = results[best_name]["metrics"]
        desc = m["description"]
        print(f"{symbol.upper():<14} | {desc:<25} | {m['accuracy']:>7.1%} | {m['auc']:>7.4f}")
        algo_wins[desc] = algo_wins.get(desc, 0) + 1

    print("-" * 70)
    print("\n  WIN COUNT:")
    for algo, count in sorted(algo_wins.items(), key=lambda x: x[1], reverse=True):
        print(f"    {algo:<25}: {count} wins")
    print(f"{'=' * 70}\n")


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train all ML algorithms")
    parser.add_argument(
        "--symbol", type=str, default="icicibank",
        help="Symbol to train on (default: icicibank)",
    )
    parser.add_argument(
        "--all-symbols", action="store_true",
        help="Train on all available symbols",
    )
    args = parser.parse_args()

    if args.all_symbols:
        train_all_symbols()
    else:
        train_all_for_symbol(args.symbol)


if __name__ == "__main__":
    main()

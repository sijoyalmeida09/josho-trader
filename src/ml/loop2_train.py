"""
LOOP-2: Train ALL algorithms on 5yr data with macro features + hybrid models.

Previous results:
  - V1 XGBoost 3-class: 45%
  - V2 XGBoost binary: 67%
  - LOOP-1 LightGBM: 71.1% on 2yr daily

This loop:
  - 5yr daily data (~1320 rows vs ~500 before = 2.6x more)
  - Macro features (USD/INR, VIX, Crude, Gold, S&P500)
  - All algorithms from registry + custom hybrids
  - Multiple prediction targets (3d/5d/10d x 0.2%/0.3%/0.5%/1.0%)
  - Hybrid models: LGB+XGB blend, Top-3 Stack, Weighted Ensemble, Cascading

Usage:
    python -m src.ml.loop2_train
"""

import json
import logging
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .algorithms import ALGORITHM_REGISTRY, get_algorithm, get_individual_algorithms
from .signal_model_v2 import (
    create_binary_labels,
    engineer_features_v2,
    purged_train_test_split,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("loop2")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "historical"
MACRO_DIR = DATA_DIR / "macro"
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "results"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Symbols and targets ──────────────────────────────────────────

BENCHMARK_SYMBOLS = ["icicibank", "reliance", "sbin"]

TARGET_CONFIGS = [
    {"forward_days": 3, "threshold": 0.003, "label": "3d/0.3%"},
    {"forward_days": 5, "threshold": 0.003, "label": "5d/0.3%"},
    {"forward_days": 5, "threshold": 0.005, "label": "5d/0.5%"},
    {"forward_days": 10, "threshold": 0.005, "label": "10d/0.5%"},
    {"forward_days": 3, "threshold": 0.002, "label": "3d/0.2%"},
    {"forward_days": 5, "threshold": 0.002, "label": "5d/0.2%"},
    {"forward_days": 10, "threshold": 0.01, "label": "10d/1.0%"},
]

INDIVIDUAL_ALGOS = [
    "xgboost_v2", "lightgbm", "random_forest", "extra_trees",
    "gradient_boosting", "adaboost",
]

HYBRID_ALGOS = [
    "lgb_xgb_blend",
    "top3_stack",
    "weighted_ensemble",
    "cascading_rf_lgb",
]


# ── Macro Feature Loading ────────────────────────────────────────

def load_macro_features() -> Optional[pd.DataFrame]:
    """Load and merge all macro CSV files into a single DataFrame with daily features."""
    macro_files = {
        "usdinr": MACRO_DIR / "usdinr_daily.csv",
        "vix": MACRO_DIR / "vix_us_daily.csv",
        "crude": MACRO_DIR / "crude_oil_daily.csv",
        "gold": MACRO_DIR / "gold_daily.csv",
        "sp500": MACRO_DIR / "sp500_daily.csv",
    }

    frames = {}
    for name, path in macro_files.items():
        if not path.exists():
            log.warning(f"Macro file missing: {path}")
            continue
        try:
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            df.index = pd.to_datetime(df.index).normalize()
            df = df.sort_index()
            frames[name] = df["Close"].rename(f"macro_{name}")
        except Exception as exc:
            log.warning(f"Failed to load macro {name}: {exc}")

    if not frames:
        return None

    merged = pd.concat(frames.values(), axis=1, join="outer")
    merged = merged.sort_index().ffill().bfill()

    # Engineer macro features: returns + levels
    macro_feat = pd.DataFrame(index=merged.index)
    for col in merged.columns:
        short_name = col.replace("macro_", "")
        macro_feat[f"{short_name}_ret1d"] = merged[col].pct_change(1)
        macro_feat[f"{short_name}_ret5d"] = merged[col].pct_change(5)
        macro_feat[f"{short_name}_zscore20"] = (
            (merged[col] - merged[col].rolling(20).mean())
            / (merged[col].rolling(20).std() + 1e-8)
        )

    # VIX level is directly useful (not just change)
    if "macro_vix" in merged.columns:
        macro_feat["vix_level"] = merged["macro_vix"]
        macro_feat["vix_high"] = (merged["macro_vix"] > 25).astype(int)

    # Cross-asset: gold/crude ratio change, USD strength
    if "macro_gold" in merged.columns and "macro_crude" in merged.columns:
        ratio = merged["macro_gold"] / (merged["macro_crude"] + 1e-8)
        macro_feat["gold_crude_ratio_ret"] = ratio.pct_change(5)

    return macro_feat.replace([np.inf, -np.inf], np.nan).fillna(0)


# ── Data Loading ─────────────────────────────────────────────────

def load_5yr_data(symbol: str) -> Optional[pd.DataFrame]:
    """Load 5yr CSV file for a symbol."""
    csv_path = DATA_DIR / f"{symbol.lower()}_5yr.csv"
    if not csv_path.exists():
        log.error(f"No 5yr data file: {csv_path}")
        return None

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    df = df.sort_index()

    # Ensure standard OHLCV columns
    expected = {"open", "high", "low", "close", "volume"}
    if not expected.issubset(set(df.columns)):
        log.error(f"Missing columns in {symbol}: {expected - set(df.columns)}")
        return None

    log.info(f"Loaded {symbol.upper()}: {len(df)} rows, "
             f"{df.index[0].date()} to {df.index[-1].date()}")
    return df


def prepare_features_with_macro(
    df: pd.DataFrame,
    forward_days: int,
    threshold: float,
) -> Optional[tuple]:
    """
    Engineer V2 features + macro features, create labels, purged split.

    Returns:
        (X_train, X_test, y_train, y_test, scale_pos_weight, info) or None.
    """
    # V2 features (WorldQuant alphas + microstructure)
    features = engineer_features_v2(df)

    # Add macro features if available
    macro = load_macro_features()
    if macro is not None:
        # Align macro to stock dates
        aligned_macro = macro.reindex(features.index, method="ffill").fillna(0)
        features = pd.concat([features, aligned_macro], axis=1)
        n_macro = len(aligned_macro.columns)
    else:
        n_macro = 0

    # Binary labels
    labels = create_binary_labels(df, forward_days, threshold)

    # Warmup (60 bars) + drop future-less rows
    warmup = 60
    valid_features = features.iloc[warmup:-forward_days].copy()
    valid_labels = labels.iloc[warmup:-forward_days].copy()

    # Drop rows with NaN labels (neutral zone) or NaN features
    mask = valid_features.notna().all(axis=1) & valid_labels.notna()
    X = valid_features[mask]
    y = valid_labels[mask]

    if len(X) < 100:
        log.error(f"Too few samples: {len(X)}")
        return None

    X_train, X_test, y_train, y_test = purged_train_test_split(
        X, y, test_ratio=0.2, purge_gap=max(forward_days, 10),
    )

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale = neg / pos if pos > 0 else 1.0

    info = {
        "total_rows": len(df),
        "valid_samples": len(X),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "up_count": int((y == 1).sum()),
        "down_count": int((y == 0).sum()),
        "balance_ratio": round(float(pos / (pos + neg)) if (pos + neg) > 0 else 0, 3),
        "n_features_base": len(engineer_features_v2(df).columns),
        "n_features_macro": n_macro,
        "n_features_total": X.shape[1],
        "scale_pos_weight": round(float(scale), 4),
    }

    return X_train, X_test, y_train, y_test, scale, info


# ── Model Evaluation ─────────────────────────────────────────────

def evaluate(model, X_test, y_test) -> dict:
    """Compute all metrics for a fitted model."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["auc"] = round(float(roc_auc_score(y_test, y_proba)), 4)
    except Exception:
        metrics["auc"] = 0.0

    return metrics


# ── Individual Algorithm Training ────────────────────────────────

def train_individual(
    name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scale: float,
) -> Optional[dict]:
    """Train one algorithm from the registry. Returns {model, metrics} or None."""
    start = time.time()
    try:
        model = get_algorithm(name, scale)
    except Exception as exc:
        log.warning(f"  {name}: create failed — {exc}")
        return None

    try:
        entry = ALGORITHM_REGISTRY.get(name, {})
        if entry.get("needs_early_stop"):
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)
    except Exception as exc:
        log.warning(f"  {name}: train failed — {exc}")
        return None

    elapsed = time.time() - start
    metrics = evaluate(model, X_test, y_test)
    metrics["train_time"] = round(elapsed, 2)

    return {"model": model, "metrics": metrics}


# ── Hybrid Models ────────────────────────────────────────────────

class ProbabilityBlender:
    """Averages predict_proba from two models (soft voting without sklearn overhead)."""

    def __init__(self, model_a, model_b, weight_a=0.5, weight_b=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.weight_a = weight_a
        self.weight_b = weight_b
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        pa = self.model_a.predict_proba(X)
        pb = self.model_b.predict_proba(X)
        return self.weight_a * pa + self.weight_b * pb

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class WeightedEnsemble:
    """Weighted soft voting by each model's AUC score."""

    def __init__(self, models_with_weights: list):
        """models_with_weights: list of (model, weight) tuples."""
        self.models_with_weights = models_with_weights
        total = sum(w for _, w in models_with_weights)
        self.models_with_weights = [
            (m, w / total) for m, w in models_with_weights
        ]
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        proba = np.zeros((len(X), 2))
        for model, weight in self.models_with_weights:
            proba += weight * model.predict_proba(X)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class CascadingClassifier:
    """
    Stage 1: High-recall model (RF) filters candidates.
    Stage 2: High-precision model (LightGBM) confirms signal.
    If Stage 1 says DOWN, we say DOWN. If Stage 1 says UP, we defer to Stage 2.
    """

    def __init__(self, stage1_model, stage2_model, stage1_threshold=0.4):
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        self.stage1_threshold = stage1_threshold
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        s1_proba = self.stage1.predict_proba(X)
        s2_proba = self.stage2.predict_proba(X)

        # Stage 1 passes candidates with UP probability > threshold
        passed = s1_proba[:, 1] >= self.stage1_threshold
        # Final proba: for passed samples use stage2, else use stage1
        result = s1_proba.copy()
        result[passed] = s2_proba[passed]
        return result

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def build_hybrids(
    trained_models: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scale: float,
) -> dict:
    """
    Build 4 hybrid models from the already-trained individual models.

    Returns dict of {name: {model, metrics}}.
    """
    hybrids = {}

    # Sort individual models by AUC
    ranked = sorted(
        trained_models.items(),
        key=lambda x: x[1]["metrics"]["auc"],
        reverse=True,
    )

    # ── 1. LightGBM + XGBoost Blend ─────────────────────────────
    lgb_result = trained_models.get("lightgbm")
    xgb_result = trained_models.get("xgboost_v2")

    if lgb_result and xgb_result:
        blender = ProbabilityBlender(
            lgb_result["model"], xgb_result["model"],
            weight_a=0.55, weight_b=0.45,  # Slight LGB bias (historically better)
        )
        metrics = evaluate(blender, X_test, y_test)
        metrics["train_time"] = 0.0  # Already trained
        hybrids["lgb_xgb_blend"] = {"model": blender, "metrics": metrics}
        log.info(f"  lgb_xgb_blend: Acc={metrics['accuracy']:.4f} AUC={metrics['auc']:.4f}")

    # ── 2. Top-3 Stacking with LogisticRegression meta ───────────
    if len(ranked) >= 3:
        top3_names = [name for name, _ in ranked[:3]]
        log.info(f"  Top-3 Stack base models: {top3_names}")

        from sklearn.ensemble import StackingClassifier

        # Need fresh estimators for stacking (sklearn limitation)
        base_estimators = []
        for name in top3_names:
            fresh = get_algorithm(name, scale)
            base_estimators.append((name, fresh))

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
            elapsed = time.time() - start
            metrics = evaluate(stacker, X_test, y_test)
            metrics["train_time"] = round(elapsed, 2)
            hybrids["top3_stack"] = {"model": stacker, "metrics": metrics}
            log.info(f"  top3_stack: Acc={metrics['accuracy']:.4f} AUC={metrics['auc']:.4f}")
        except Exception as exc:
            log.warning(f"  top3_stack: failed — {exc}")

    # ── 3. Weighted Ensemble (by AUC) ────────────────────────────
    if len(ranked) >= 3:
        top_models = [
            (result["model"], result["metrics"]["auc"])
            for name, result in ranked[:5]  # Top 5
            if result["metrics"]["auc"] > 0.5
        ]
        if len(top_models) >= 2:
            weighted = WeightedEnsemble(top_models)
            metrics = evaluate(weighted, X_test, y_test)
            metrics["train_time"] = 0.0
            hybrids["weighted_ensemble"] = {"model": weighted, "metrics": metrics}
            log.info(f"  weighted_ensemble: Acc={metrics['accuracy']:.4f} AUC={metrics['auc']:.4f}")

    # ── 4. Cascading: RF (high recall) → LightGBM (precision) ───
    rf_result = trained_models.get("random_forest")
    if rf_result and lgb_result:
        cascade = CascadingClassifier(
            rf_result["model"], lgb_result["model"],
            stage1_threshold=0.4,
        )
        metrics = evaluate(cascade, X_test, y_test)
        metrics["train_time"] = 0.0
        hybrids["cascading_rf_lgb"] = {"model": cascade, "metrics": metrics}
        log.info(f"  cascading_rf_lgb: Acc={metrics['accuracy']:.4f} AUC={metrics['auc']:.4f}")

    return hybrids


# ── Full Training Pipeline ───────────────────────────────────────

def train_all_for_target(
    symbol: str,
    forward_days: int,
    threshold: float,
    df: pd.DataFrame,
) -> Optional[dict]:
    """
    Train ALL algorithms + hybrids for one symbol + one target config.

    Returns dict of {algo_name: {model, metrics}} or None.
    """
    prepared = prepare_features_with_macro(df, forward_days, threshold)
    if prepared is None:
        return None

    X_train, X_test, y_train, y_test, scale, info = prepared

    log.info(
        f"  Target {forward_days}d/{threshold*100:.1f}%: "
        f"{info['valid_samples']} samples "
        f"(train={info['train_size']}, test={info['test_size']}, "
        f"UP={info['up_count']}, DOWN={info['down_count']}, "
        f"features={info['n_features_total']})"
    )

    # Phase 1: Individual algorithms
    individual_results = {}
    for algo_name in INDIVIDUAL_ALGOS:
        result = train_individual(algo_name, X_train, X_test, y_train, y_test, scale)
        if result is not None:
            individual_results[algo_name] = result

    # Phase 2: Hybrid models
    hybrid_results = build_hybrids(
        individual_results, X_train, X_test, y_train, y_test, scale,
    )

    # Merge all results
    all_results = {**individual_results, **hybrid_results}

    # Add info to each result
    for name in all_results:
        all_results[name]["info"] = info

    return all_results


def run_full_benchmark():
    """
    Run the complete LOOP-2 benchmark:
    - 3 symbols x 7 target configs x (6 individual + 4 hybrid) algorithms
    - Print comprehensive comparison tables
    - Save the best model
    """
    print("\n" + "=" * 100)
    print("  LOOP-2: 5yr DATA x ALL ALGORITHMS x ALL TARGETS")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 100)

    # Master results: {symbol: {target_label: {algo: metrics}}}
    master = {}
    best_overall = {"accuracy": 0, "auc": 0}

    for symbol in BENCHMARK_SYMBOLS:
        df = load_5yr_data(symbol)
        if df is None:
            continue

        print(f"\n{'-' * 100}")
        print(f"  SYMBOL: {symbol.upper()} | {len(df)} rows | "
              f"{df.index[0].date()} to {df.index[-1].date()}")
        print(f"{'-' * 100}")

        master[symbol] = {}

        for target in TARGET_CONFIGS:
            fwd = target["forward_days"]
            thr = target["threshold"]
            label = target["label"]

            results = train_all_for_target(symbol, fwd, thr, df)
            if results is None:
                continue

            # Extract metrics only
            metrics_only = {}
            for algo_name, result in results.items():
                metrics_only[algo_name] = result["metrics"]

                # Track best overall
                acc = result["metrics"]["accuracy"]
                auc = result["metrics"]["auc"]
                if auc > best_overall.get("auc", 0):
                    best_overall = {
                        "symbol": symbol,
                        "target": label,
                        "algorithm": algo_name,
                        "accuracy": acc,
                        "auc": auc,
                        "precision": result["metrics"]["precision"],
                        "recall": result["metrics"]["recall"],
                        "f1": result["metrics"]["f1"],
                        "model": result["model"],
                        "info": result.get("info", {}),
                    }

            master[symbol][label] = metrics_only

    # ── Print comprehensive tables ───────────────────────────────
    _print_full_results(master)
    _print_best_per_symbol(master)
    _print_algorithm_rankings(master)

    # ── Save best model ──────────────────────────────────────────
    if "model" in best_overall:
        _save_best_model(best_overall)

    # ── Save JSON results ────────────────────────────────────────
    results_path = RESULTS_DIR / "loop2_results.json"
    # Strip non-serializable model objects
    serializable = {}
    for sym in master:
        serializable[sym] = {}
        for target in master[sym]:
            serializable[sym][target] = master[sym][target]

    results_path.write_text(
        json.dumps(serializable, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {results_path}")

    return master, best_overall


def _print_full_results(master: dict):
    """Print the comprehensive algorithm x target matrix for each symbol."""
    # Collect all target labels and algo names across all symbols
    all_targets = []
    all_algos = set()
    for sym in master:
        for target in master[sym]:
            if target not in all_targets:
                all_targets.append(target)
            all_algos.update(master[sym][target].keys())

    # Canonical algo order
    algo_order = INDIVIDUAL_ALGOS + HYBRID_ALGOS
    algo_order = [a for a in algo_order if a in all_algos]
    # Add any extras
    for a in sorted(all_algos):
        if a not in algo_order:
            algo_order.append(a)

    for symbol in master:
        print(f"\n{'=' * 120}")
        print(f"  {symbol.upper()} — ACCURACY (AUC) MATRIX")
        print(f"{'=' * 120}")

        # Header
        col_width = 14
        header = f"{'Algorithm':<24}"
        for target in all_targets:
            header += f" | {target:^{col_width}}"
        print(header)
        print("-" * (24 + (col_width + 3) * len(all_targets)))

        # Find best per target
        best_per_target = {}
        for target in all_targets:
            target_data = master[symbol].get(target, {})
            if target_data:
                best_algo = max(target_data, key=lambda a: target_data[a].get("auc", 0))
                best_per_target[target] = best_algo

        for algo in algo_order:
            row = f"{algo:<24}"
            for target in all_targets:
                target_data = master[symbol].get(target, {})
                if algo in target_data:
                    acc = target_data[algo]["accuracy"]
                    auc = target_data[algo]["auc"]
                    marker = " *" if best_per_target.get(target) == algo else "  "
                    cell = f"{acc:.1%}({auc:.3f}){marker}"
                else:
                    cell = "     ---     "
                row += f" | {cell:^{col_width}}"
            print(row)

        print("-" * (24 + (col_width + 3) * len(all_targets)))
        print("  * = best AUC for that target")


def _print_best_per_symbol(master: dict):
    """Print the single best result per symbol."""
    print(f"\n{'=' * 90}")
    print("  BEST RESULT PER SYMBOL")
    print(f"{'=' * 90}")
    print(f"{'Symbol':<12} | {'Target':<10} | {'Algorithm':<24} | "
          f"{'Accuracy':>8} | {'AUC':>7} | {'F1':>7}")
    print("-" * 90)

    for symbol in master:
        best_acc = 0
        best_info = {}
        for target in master[symbol]:
            for algo, metrics in master[symbol][target].items():
                if metrics["auc"] > best_info.get("auc", 0):
                    best_info = {
                        "target": target,
                        "algo": algo,
                        **metrics,
                    }

        if best_info:
            print(
                f"{symbol.upper():<12} | {best_info['target']:<10} | "
                f"{best_info['algo']:<24} | "
                f"{best_info['accuracy']:>7.1%} | "
                f"{best_info['auc']:>7.4f} | "
                f"{best_info['f1']:>7.4f}"
            )
    print(f"{'=' * 90}")


def _print_algorithm_rankings(master: dict):
    """Print average performance of each algorithm across all symbols and targets."""
    print(f"\n{'=' * 80}")
    print("  ALGORITHM RANKINGS (avg across all symbols x targets)")
    print(f"{'=' * 80}")

    algo_scores = {}  # algo -> list of (acc, auc)
    for symbol in master:
        for target in master[symbol]:
            for algo, metrics in master[symbol][target].items():
                if algo not in algo_scores:
                    algo_scores[algo] = []
                algo_scores[algo].append({
                    "accuracy": metrics["accuracy"],
                    "auc": metrics["auc"],
                    "f1": metrics["f1"],
                })

    print(f"{'Algorithm':<24} | {'Avg Acc':>8} | {'Avg AUC':>8} | "
          f"{'Avg F1':>8} | {'#Runs':>6} | {'Best Acc':>8} | {'Best AUC':>8}")
    print("-" * 80)

    ranked = sorted(
        algo_scores.items(),
        key=lambda x: np.mean([s["auc"] for s in x[1]]),
        reverse=True,
    )

    for algo, scores in ranked:
        accs = [s["accuracy"] for s in scores]
        aucs = [s["auc"] for s in scores]
        f1s = [s["f1"] for s in scores]
        print(
            f"{algo:<24} | {np.mean(accs):>7.1%} | {np.mean(aucs):>8.4f} | "
            f"{np.mean(f1s):>8.4f} | {len(scores):>6} | "
            f"{max(accs):>7.1%} | {max(aucs):>8.4f}"
        )

    print(f"{'=' * 80}")


def _save_best_model(best: dict):
    """Save the overall best model to disk."""
    symbol = best["symbol"]
    algo = best["algorithm"]
    target = best["target"]

    model_path = MODEL_DIR / f"loop2_best_{symbol}.pkl"
    meta_path = MODEL_DIR / f"loop2_best_{symbol}_meta.json"

    with open(model_path, "wb") as fp:
        pickle.dump({
            "model": best["model"],
            "algorithm": algo,
            "target": target,
        }, fp)

    meta = {
        "symbol": symbol.upper(),
        "algorithm": algo,
        "target": target,
        "accuracy": best["accuracy"],
        "auc": best["auc"],
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
        "saved_at": datetime.now().isoformat(),
        "loop": "LOOP-2",
        "data": "5yr daily + macro",
        "info": best.get("info", {}),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n{'*' * 80}")
    print(f"  BEST MODEL SAVED: {algo} on {symbol.upper()} ({target})")
    print(f"  Accuracy: {best['accuracy']:.1%} | AUC: {best['auc']:.4f} | "
          f"F1: {best['f1']:.4f}")
    print(f"  Path: {model_path}")
    print(f"{'*' * 80}")


# ── Entry Point ──────────────────────────────────────────────────

def main():
    master, best = run_full_benchmark()

    print(f"\n{'#' * 80}")
    print("  LOOP-2 COMPLETE")
    if best.get("algorithm"):
        print(f"  Winner: {best['algorithm']} on {best['symbol'].upper()} ({best['target']})")
        print(f"  Accuracy: {best['accuracy']:.1%} | AUC: {best['auc']:.4f}")
    print(f"  Previous best: LightGBM 71.1% on 2yr data (LOOP-1)")
    print(f"{'#' * 80}")


if __name__ == "__main__":
    main()

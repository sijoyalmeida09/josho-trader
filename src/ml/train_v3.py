"""
Train V3 models — Alpha158 features + walk-forward validation + XGBoost vs LightGBM.

Improvements over V2:
  1. Alpha158 feature set (~120 features vs 52)
  2. Walk-forward validation (train yr 1-3, val yr 4, test yr 5)
  3. LightGBM comparison
  4. Top-50 feature selection by importance
  5. Profit factor from simulated trades
  6. V1 vs V2 vs V3 comparison table
"""

import logging
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_v3")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "historical"
MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "nifty", "banknifty", "reliance", "tcs", "infy",
    "hdfcbank", "icicibank", "sbin", "bajfinance", "itc", "bhartiartl",
]

FORWARD_PERIODS = 5
THRESHOLD = 0.003
PURGE_GAP = 10
TOP_K_FEATURES = 50
WARMUP = 65  # Need 60+ rows for longest rolling window


# ── Labels ───────────────────────────────────────────────────────────

def create_binary_labels(close: pd.Series, forward_periods: int = FORWARD_PERIODS,
                         threshold: float = THRESHOLD) -> pd.Series:
    """Binary labels: 1 = UP (> threshold), 0 = DOWN (< -threshold), NaN = neutral."""
    fwd_ret = close.shift(-forward_periods) / close - 1
    labels = pd.Series(np.nan, index=close.index)
    labels[fwd_ret > threshold] = 1
    labels[fwd_ret < -threshold] = 0
    return labels


# ── Walk-Forward Split ───────────────────────────────────────────────

def walk_forward_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    """
    Walk-forward: train on years 1-3, validate on year 4, test on year 5.
    Falls back to 60/20/20 if data < 3 years.
    """
    n = len(X)
    total_days = (X.index[-1] - X.index[0]).days

    if total_days > 3 * 365:
        # True walk-forward by year
        start = X.index[0]
        yr3_end = start + pd.DateOffset(years=3)
        yr4_end = start + pd.DateOffset(years=4)

        train_mask = X.index < (yr3_end - pd.Timedelta(days=PURGE_GAP))
        val_mask = (X.index >= yr3_end) & (X.index < (yr4_end - pd.Timedelta(days=PURGE_GAP)))
        test_mask = X.index >= yr4_end

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
    else:
        # Fallback: 60/20/20 with purge
        s1 = int(n * 0.6)
        s2 = int(n * 0.8)
        X_train = X.iloc[:s1 - PURGE_GAP]
        y_train = y.iloc[:s1 - PURGE_GAP]
        X_val = X.iloc[s1:s2 - PURGE_GAP]
        y_val = y.iloc[s1:s2 - PURGE_GAP]
        X_test = X.iloc[s2:]
        y_test = y.iloc[s2:]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ── Profit Factor Simulation ────────────────────────────────────────

def compute_profit_factor(y_true: pd.Series, y_pred: np.ndarray,
                          close_test: pd.Series) -> dict:
    """
    Simulate trades: go long on UP predictions, short on DOWN.
    Returns profit factor = gross_profit / gross_loss.
    """
    fwd_ret = close_test.pct_change(FORWARD_PERIODS).shift(-FORWARD_PERIODS)

    # Align
    valid_idx = fwd_ret.dropna().index
    pred_series = pd.Series(y_pred, index=y_true.index)
    common = valid_idx.intersection(pred_series.index)

    if len(common) < 10:
        return {"profit_factor": 0, "total_trades": 0, "win_rate": 0}

    pred_aligned = pred_series.loc[common]
    ret_aligned = fwd_ret.loc[common]

    # PnL per trade: long if pred=1, short if pred=0
    pnl = pd.Series(0.0, index=common)
    pnl[pred_aligned == 1] = ret_aligned[pred_aligned == 1]
    pnl[pred_aligned == 0] = -ret_aligned[pred_aligned == 0]

    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())

    profit_factor = gross_profit / (gross_loss + 1e-10)
    win_rate = (pnl > 0).sum() / (len(pnl) + 1e-10)

    return {
        "profit_factor": round(float(profit_factor), 3),
        "total_trades": int(len(pnl)),
        "win_rate": round(float(win_rate), 4),
        "total_pnl_pct": round(float(pnl.sum() * 100), 2),
    }


# ── Train Single Symbol ─────────────────────────────────────────────

def train_symbol(symbol: str, use_5yr: bool = True) -> Optional[dict]:
    """Train V3 model for a single symbol."""
    from .alpha158 import compute_combined_features

    # Try 5yr data first, fall back to daily
    csv_5yr = DATA_DIR / f"{symbol}_5yr.csv"
    csv_daily = DATA_DIR / f"{symbol}_daily.csv"

    if use_5yr and csv_5yr.exists():
        df = pd.read_csv(csv_5yr, index_col=0, parse_dates=True)
        data_source = "5yr"
    elif csv_daily.exists():
        df = pd.read_csv(csv_daily, index_col=0, parse_dates=True)
        data_source = "daily"
    else:
        log.warning(f"No data for {symbol}")
        return None

    if len(df) < 200:
        log.warning(f"Not enough data for {symbol}: {len(df)} rows")
        return None

    log.info(f"Training V3 for {symbol.upper()} ({len(df)} rows, source={data_source})")

    # ── Features + Labels ────────────────────────────────────────
    features = compute_combined_features(df)
    labels = create_binary_labels(df["close"])

    # Remove warmup + future-less rows
    valid = features.iloc[WARMUP:-FORWARD_PERIODS].copy()
    valid_labels = labels.iloc[WARMUP:-FORWARD_PERIODS].copy()

    # Drop NaN labels (neutral zone)
    mask = valid.notna().all(axis=1) & valid_labels.notna()
    X = valid[mask]
    y = valid_labels[mask]

    if len(X) < 100:
        log.warning(f"Not enough valid samples for {symbol}: {len(X)}")
        return None

    log.info(f"  Samples: {len(X)} | UP: {int((y == 1).sum())} | DOWN: {int((y == 0).sum())} | Features: {X.shape[1]}")

    # ── Walk-Forward Split ───────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = walk_forward_split(df, X, y)

    if len(X_train) < 50 or len(X_test) < 20:
        log.warning(f"Splits too small for {symbol}")
        return None

    # Class balance
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale = neg_count / pos_count if pos_count > 0 else 1

    # ── Train XGBoost ────────────────────────────────────────────
    xgb_results = _train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, scale)

    # ── Train LightGBM ───────────────────────────────────────────
    lgb_results = _train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test, scale)

    # Pick best model
    best_name = "xgboost" if xgb_results["test_acc"] >= lgb_results["test_acc"] else "lightgbm"
    best = xgb_results if best_name == "xgboost" else lgb_results
    best_model = best["model"]
    best_features_all = list(X.columns)

    # ── Feature Selection: Top 50 ────────────────────────────────
    importances = dict(zip(best_features_all, best_model.feature_importances_))
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    selected = [f for f, imp in sorted_feats if imp > 0][:TOP_K_FEATURES]

    log.info(f"  Feature selection: {len(best_features_all)} -> {len(selected)}")

    # Retrain on selected features
    if len(selected) < len(best_features_all) and len(selected) >= 10:
        X_train_sel = X_train[selected]
        X_val_sel = X_val[selected]
        X_test_sel = X_test[selected]

        if best_name == "xgboost":
            final = _train_xgboost(X_train_sel, y_train, X_val_sel, y_val, X_test_sel, y_test, scale)
        else:
            final = _train_lightgbm(X_train_sel, y_train, X_val_sel, y_val, X_test_sel, y_test, scale)

        final_model = final["model"]
        final_features = selected
    else:
        final = best
        final_model = best_model
        final_features = best_features_all

    # ── Profit Factor ────────────────────────────────────────────
    X_test_final = X_test[final_features]
    y_pred = final_model.predict(X_test_final)
    close_test = df["close"].loc[X_test.index]
    pf = compute_profit_factor(y_test, y_pred, close_test)

    # ── Metrics ──────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

    y_proba = final_model.predict_proba(X_test_final)[:, 1] if hasattr(final_model, "predict_proba") else np.zeros(len(y_test))
    test_acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, final_model.predict(X_train[final_features]))

    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = 0

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # ── Save Model ───────────────────────────────────────────────
    model_path = MODEL_DIR / f"v3_{symbol}.pkl"
    with open(model_path, "wb") as fp:
        pickle.dump({
            "model": final_model,
            "features": final_features,
            "engine": best_name,
            "version": "v3",
        }, fp)

    top_15 = sorted(
        zip(final_features, final_model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )[:15]

    meta = {
        "model_name": symbol,
        "version": "v3",
        "engine": best_name,
        "data_source": data_source,
        "data_rows": len(df),
        "trained_at": datetime.now().isoformat(),
        "samples": len(X),
        "up_count": int((y == 1).sum()),
        "down_count": int((y == 0).sum()),
        "features_total": len(best_features_all),
        "features_selected": len(final_features),
        "train_accuracy": round(float(train_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "val_accuracy": round(float(final.get("val_acc", 0)), 4),
        "auc": round(float(auc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "profit_factor": pf["profit_factor"],
        "win_rate": pf["win_rate"],
        "total_pnl_pct": pf["total_pnl_pct"],
        "xgb_test_acc": round(float(xgb_results["test_acc"]), 4),
        "lgb_test_acc": round(float(lgb_results["test_acc"]), 4),
        "top_features": {k: round(float(v), 4) for k, v in top_15},
    }

    meta_path = MODEL_DIR / f"v3_{symbol}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info(
        f"  V3 {symbol.upper()}: acc={test_acc:.3f} AUC={auc:.3f} "
        f"F1={f1:.3f} PF={pf['profit_factor']:.2f} engine={best_name} "
        f"features={len(final_features)}"
    )
    return meta


def _train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, scale):
    """Train XGBoost classifier."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        log.error("xgboost not installed")
        return {"test_acc": 0, "val_acc": 0, "model": None}

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.5,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=3.0,
        gamma=0.15,
        scale_pos_weight=scale,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        early_stopping_rounds=25,
    )

    # Use validation set for early stopping
    eval_set = [(X_val, y_val)] if len(X_val) > 0 else [(X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    test_acc = model.score(X_test, y_test)
    val_acc = model.score(X_val, y_val) if len(X_val) > 0 else 0

    return {"test_acc": test_acc, "val_acc": val_acc, "model": model}


def _train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test, scale):
    """Train LightGBM classifier."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        log.warning("lightgbm not installed — skipping")
        return {"test_acc": 0, "val_acc": 0, "model": None}

    model = LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.5,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=3.0,
        scale_pos_weight=scale,
        objective="binary",
        metric="auc",
        random_state=42,
        verbose=-1,
    )

    eval_set = [(X_val, y_val)] if len(X_val) > 0 else [(X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        callbacks=[_lgb_early_stopping(25)],
    )

    test_acc = model.score(X_test, y_test)
    val_acc = model.score(X_val, y_val) if len(X_val) > 0 else 0

    return {"test_acc": test_acc, "val_acc": val_acc, "model": model}


def _lgb_early_stopping(rounds):
    """Get LightGBM early stopping callback."""
    try:
        from lightgbm import early_stopping
        return early_stopping(stopping_rounds=rounds, verbose=False)
    except ImportError:
        return None


# ── Main Training Loop ───────────────────────────────────────────────

def train_all():
    """Train V3 models on all symbols and print comparison."""
    results = []

    for symbol in SYMBOLS:
        meta = train_symbol(symbol, use_5yr=True)
        if meta is not None and "error" not in meta:
            results.append(meta)

    if not results:
        log.error("No models trained!")
        return

    # ── V1 vs V2 vs V3 Comparison ────────────────────────────────
    print("\n" + "=" * 110)
    print("V1 vs V2 vs V3 COMPARISON")
    print("=" * 110)
    print(
        f"{'Symbol':12} | {'V1 Acc':>7} | {'V2 Acc':>7} | {'V3 Acc':>7} | "
        f"{'V3 AUC':>7} | {'V3 F1':>6} | {'V3 PF':>6} | {'Engine':>8} | "
        f"{'Feats':>5} | {'Data':>6} | {'Improve':>8}"
    )
    print("-" * 110)

    accs = []
    for meta in results:
        name = meta["model_name"]

        # Load V1 meta
        v1_meta_path = MODEL_DIR / f"xgb_{name}_meta.json"
        v1_acc = 0
        if v1_meta_path.exists():
            try:
                v1 = json.loads(v1_meta_path.read_text())
                v1_acc = v1.get("test_accuracy", 0)
            except Exception:
                pass

        # Load V2 meta
        v2_meta_path = MODEL_DIR / f"xgb_v2_{name}_meta.json"
        v2_acc = 0
        if v2_meta_path.exists():
            try:
                v2 = json.loads(v2_meta_path.read_text())
                v2_acc = v2.get("test_accuracy", 0)
            except Exception:
                pass

        v3_acc = meta["test_accuracy"]
        improvement = v3_acc - max(v1_acc, v2_acc)
        accs.append(v3_acc)

        print(
            f"{name.upper():12} | "
            f"{v1_acc:>6.3f} | "
            f"{v2_acc:>6.3f} | "
            f"{v3_acc:>6.3f} | "
            f"{meta['auc']:>6.3f} | "
            f"{meta['f1']:>5.3f} | "
            f"{meta['profit_factor']:>5.2f} | "
            f"{meta['engine']:>8} | "
            f"{meta['features_selected']:>5} | "
            f"{meta['data_rows']:>6} | "
            f"{improvement:>+7.3f}"
        )

    print("-" * 110)
    avg_acc = np.mean(accs) if accs else 0
    print(f"{'AVERAGE':12} | {'':>7} | {'':>7} | {avg_acc:>6.3f} |")
    print("=" * 110)

    # XGBoost vs LightGBM summary
    xgb_wins = sum(1 for m in results if m["engine"] == "xgboost")
    lgb_wins = sum(1 for m in results if m["engine"] == "lightgbm")
    print(f"\nEngine comparison: XGBoost won {xgb_wins}, LightGBM won {lgb_wins}")

    # Save summary
    summary_path = MODEL_DIR / "v3_training_summary.json"
    summary_path.write_text(json.dumps({
        "trained_at": datetime.now().isoformat(),
        "symbols_trained": len(results),
        "average_accuracy": round(float(avg_acc), 4),
        "results": results,
    }, indent=2), encoding="utf-8")
    log.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    train_all()

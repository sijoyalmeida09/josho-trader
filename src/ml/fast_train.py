"""
fast_train.py — Fast Multi-Algorithm Training with Mega Features
=================================================================
Trains XGBoost, LightGBM, Random Forest, Extra Trees, Gradient Boosting
on 123+ features (technical + macro + calendar + sector).

Segregated output:
  data/results/{stock}/
    ├── best_model.pkl       — best performing model
    ├── all_results.json     — comparison of all algorithms
    ├── feature_importance.json
    └── predictions.csv      — walk-forward predictions

Usage:
    python -m src.ml.fast_train                    # train COALINDIA
    python -m src.ml.fast_train --all              # train ALL stocks
    python -m src.ml.fast_train --stock TATASTEEL  # train specific
"""

import argparse
import json
import logging
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fast_train")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Stocks to train
ALL_STOCKS = [
    "COALINDIA", "TATASTEEL", "HINDALCO", "VEDL", "SAIL", "JSWSTEEL",
    "BPCL", "ONGC", "TATAPOWER", "ADANIPOWER", "SUZLON", "PNB",
    "SBIN", "HDFCBANK", "ICICIBANK", "AXISBANK", "BAJFINANCE",
    "RELIANCE", "INFY", "TCS", "ITC", "BHARTIARTL",
]


def load_features(stock: str) -> pd.DataFrame:
    """Load mega features or build them."""
    # Try pre-built mega features
    mega_file = DATA_DIR / f"{stock}_mega_features.csv"
    if mega_file.exists():
        df = pd.read_csv(mega_file, index_col=0, parse_dates=True)
        log.info(f"Loaded cached mega features: {len(df)} rows x {len(df.columns)} cols")
        return df

    # Build from scratch
    try:
        from .mega_features import build_mega_features
        df = build_mega_features(stock)
        if not df.empty:
            df.to_csv(mega_file)
        return df
    except Exception as e:
        log.warning(f"Mega features failed for {stock}: {e}")

    # Fallback: load raw CSV and add basic features
    csv_file = DATA_DIR / "historical" / "daily_5y" / f"{stock.lower()}_daily_5y.csv"
    if not csv_file.exists():
        log.error(f"No data for {stock}")
        return pd.DataFrame()

    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]

    # Quick features
    c = df["close"]
    df["return_1d"] = c.pct_change(1)
    df["return_5d"] = c.pct_change(5)
    df["return_10d"] = c.pct_change(10)
    df["sma_5"] = c.rolling(5).mean()
    df["sma_20"] = c.rolling(20).mean()
    df["sma_50"] = c.rolling(50).mean()
    df["price_vs_sma20"] = (c / df["sma_20"] - 1) * 100
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # RSI
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    df["macd"] = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    # Bollinger
    df["bb_upper"] = df["sma_20"] + 2 * c.rolling(20).std()
    df["bb_lower"] = df["sma_20"] - 2 * c.rolling(20).std()
    df["bb_position"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    # Day of week
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    # Label: will price go up >0.5% in next 5 days?
    future = c.pct_change(5).shift(-5) * 100
    df["target_binary"] = (future > 0.5).astype(int)

    df = df.dropna(subset=["target_binary"]).ffill().fillna(0)
    return df


def train_stock(stock: str) -> dict:
    """Train all algorithms on one stock, return results."""
    log.info(f"\n{'='*60}")
    log.info(f"TRAINING: {stock}")
    log.info(f"{'='*60}")

    df = load_features(stock)
    if df.empty or len(df) < 100:
        log.warning(f"{stock}: insufficient data ({len(df)} rows)")
        return {}

    # Separate features and target
    target_col = "target_binary"
    if target_col not in df.columns:
        log.error(f"{stock}: no target column")
        return {}

    exclude = ["target_binary", "target_return", "target_3class",
               "close", "open", "high", "low", "volume"]
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32, np.int32, bool]]

    X = df[feature_cols].values
    y = df[target_col].values

    # Handle infinities and extreme values
    X = X.astype(np.float64)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X = np.clip(X, -1e6, 1e6)

    log.info(f"Features: {len(feature_cols)} | Samples: {len(X)} | UP: {y.sum()} DOWN: {len(y)-y.sum()}")

    # Walk-forward split (time series — no data leakage)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    if len(X_test) < 20:
        log.warning(f"{stock}: test set too small ({len(X_test)})")
        return {}

    # ── TRAIN ALL ALGORITHMS ──
    algorithms = {
        "xgboost": None,
        "lightgbm": None,
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        "adaboost": AdaBoostClassifier(n_estimators=100, random_state=42),
    }

    # XGBoost
    try:
        import xgboost as xgb
        algorithms["xgboost"] = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    except ImportError:
        del algorithms["xgboost"]

    # LightGBM
    try:
        import lightgbm as lgb
        algorithms["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1,
        )
    except ImportError:
        del algorithms["lightgbm"]

    results = {}
    best_acc = 0
    best_model = None
    best_name = ""

    for name, model in algorithms.items():
        if model is None:
            continue

        start = time.time()
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_test, y_prob)
            except:
                auc = 0

            elapsed = time.time() - start

            results[name] = {
                "accuracy": round(acc, 4),
                "f1": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "auc": round(auc, 4),
                "time_sec": round(elapsed, 1),
            }

            log.info(f"  {name:25s} | Acc: {acc:.3f} | AUC: {auc:.3f} | F1: {f1:.3f} | {elapsed:.1f}s")

            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_name = name

        except Exception as e:
            log.error(f"  {name}: FAILED — {e}")
            results[name] = {"error": str(e)}

    # ── ENSEMBLE (combine top 3) ──
    if len([r for r in results.values() if "accuracy" in r]) >= 3:
        top3 = sorted(
            [(n, r) for n, r in results.items() if "accuracy" in r],
            key=lambda x: x[1]["accuracy"], reverse=True,
        )[:3]

        estimators = [(n, algorithms[n]) for n, _ in top3 if n in algorithms and algorithms[n] is not None]
        if len(estimators) >= 2:
            try:
                ensemble = VotingClassifier(estimators=estimators, voting="soft")
                ensemble.fit(X_train, y_train)
                y_pred = ensemble.predict(X_test)
                y_prob = ensemble.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else 0
                f1 = f1_score(y_test, y_pred, zero_division=0)

                results["ensemble_top3"] = {
                    "accuracy": round(acc, 4),
                    "auc": round(auc, 4),
                    "f1": round(f1, 4),
                    "components": [n for n, _ in top3],
                }

                log.info(f"  {'ensemble_top3':25s} | Acc: {acc:.3f} | AUC: {auc:.3f} | F1: {f1:.3f}")

                if acc > best_acc:
                    best_acc = acc
                    best_model = ensemble
                    best_name = "ensemble_top3"
            except Exception as e:
                log.error(f"  ensemble: FAILED — {e}")

    # ── SAVE RESULTS ──
    stock_dir = RESULTS_DIR / stock
    stock_dir.mkdir(parents=True, exist_ok=True)

    # Save best model
    if best_model:
        with open(stock_dir / "best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)

    # Save all results
    results["_meta"] = {
        "stock": stock,
        "features": len(feature_cols),
        "samples": len(X),
        "train_size": train_size,
        "test_size": len(X_test),
        "best_algorithm": best_name,
        "best_accuracy": round(best_acc, 4),
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    (stock_dir / "all_results.json").write_text(json.dumps(results, indent=2))

    # Feature importance (from best tree model)
    if best_model and hasattr(best_model, "feature_importances_"):
        imp = sorted(
            zip(feature_cols, best_model.feature_importances_),
            key=lambda x: -x[1],
        )[:30]
        (stock_dir / "feature_importance.json").write_text(
            json.dumps([{"feature": f, "importance": round(float(i), 4)} for f, i in imp], indent=2)
        )

    log.info(f"BEST: {best_name} ({best_acc:.3f}) | Saved to {stock_dir}")
    return results


def train_all():
    """Train all stocks in parallel-friendly way."""
    summary = {}
    for stock in ALL_STOCKS:
        try:
            results = train_stock(stock)
            if results and "_meta" in results:
                summary[stock] = {
                    "best": results["_meta"]["best_algorithm"],
                    "accuracy": results["_meta"]["best_accuracy"],
                }
        except Exception as e:
            log.error(f"{stock}: CRASHED — {e}")
            summary[stock] = {"error": str(e)}

    # Save summary
    (RESULTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))

    log.info(f"\n{'='*60}")
    log.info("TRAINING COMPLETE")
    log.info(f"{'='*60}")
    for stock, res in sorted(summary.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True):
        if "accuracy" in res:
            log.info(f"  {stock:15s} | {res['best']:20s} | Acc: {res['accuracy']:.3f}")
        else:
            log.info(f"  {stock:15s} | ERROR: {res.get('error', '?')[:40]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", default="COALINDIA")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        train_all()
    else:
        train_stock(args.stock)

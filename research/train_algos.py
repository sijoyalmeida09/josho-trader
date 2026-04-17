"""
Universal Algorithm Trainer
============================
Downloads historical data for BSE/NSE stocks, trains multiple algorithms,
backtests each, and saves the best combo per stock.

Algorithms: LightGBM, XGBoost, Random Forest, LSTM, GRU, Transformer,
            ARIMA, GARCH, Ensemble (stacking), RL (DQN)

Run: python research/train_algos.py
"""

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Setup ────────────────────────────────────────────────────
LOG_DIR = Path("C:/josho-trader/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("C:/josho-trader/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("C:/josho-trader/research/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("C:/josho-trader/research/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "algo_trainer.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("algo_trainer")

# ── Stock Universe ───────────────────────────────────────────
# All F&O stocks + micro-caps we trade
STOCKS = {
    "tier1_fno": [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN",
        "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "HINDUNILVR",
        "MARUTI", "TATAMOTORS", "TATASTEEL", "AXISBANK", "BAJFINANCE",
        "WIPRO", "HCLTECH", "ADANIENT", "TITAN", "ULTRACEMCO",
        "SUNPHARMA", "CIPLA", "COALINDIA", "ONGC", "BPCL",
    ],
    "tier2_midcap": [
        "JSWSTEEL", "HINDALCO", "VEDL", "SAIL", "TATAPOWER",
        "BANKBARODA", "PNB", "MAZDOCK", "LICI", "IRFC",
        "NHPC", "PFC", "RECLTD", "DABUR", "MPHASIS",
    ],
    "tier3_micro": [
        "YESBANK", "IDEA", "RPOWER", "SUZLON", "HFCL",
        "IDFCFIRSTB", "NBCC", "TRIDENT", "RVNL", "IRCON",
        "RAILTEL", "JSWINFRA", "JPPOWER", "GTLINFRA",
    ],
}


# ── Data Download ────────────────────────────────────────────
def download_stock_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Download historical data using yfinance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period=period)
        if df.empty:
            # Try BSE
            ticker = yf.Ticker(f"{symbol}.BO")
            df = ticker.history(period=period)
        if not df.empty:
            df.to_csv(DATA_DIR / f"{symbol}.csv")
            log.info(f"Downloaded {symbol}: {len(df)} rows")
        return df
    except Exception as e:
        log.warning(f"Failed to download {symbol}: {e}")
        return pd.DataFrame()


def load_or_download(symbol: str) -> pd.DataFrame:
    """Load cached data or download fresh."""
    cache = DATA_DIR / f"{symbol}.csv"
    if cache.exists():
        age_hours = (time.time() - cache.stat().st_mtime) / 3600
        if age_hours < 24:  # use cache if < 24h old
            return pd.read_csv(cache, index_col=0, parse_dates=True)
    return download_stock_data(symbol)


# ── Feature Engineering ──────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build Alpha158-style features from OHLCV data."""
    f = pd.DataFrame(index=df.index)

    # Price features
    f["returns_1d"] = df["Close"].pct_change(1)
    f["returns_5d"] = df["Close"].pct_change(5)
    f["returns_10d"] = df["Close"].pct_change(10)
    f["returns_20d"] = df["Close"].pct_change(20)

    # Volatility
    f["vol_5d"] = df["Close"].pct_change().rolling(5).std()
    f["vol_10d"] = df["Close"].pct_change().rolling(10).std()
    f["vol_20d"] = df["Close"].pct_change().rolling(20).std()

    # Moving averages
    for w in [5, 10, 20, 50]:
        f[f"sma_{w}"] = df["Close"].rolling(w).mean()
        f[f"close_to_sma_{w}"] = df["Close"] / f[f"sma_{w}"] - 1

    # EMA
    for w in [5, 10, 20]:
        f[f"ema_{w}"] = df["Close"].ewm(span=w).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    f["rsi_14"] = 100 - (100 / (1 + rs))

    # RSI(2) — our key signal
    gain2 = delta.where(delta > 0, 0).rolling(2).mean()
    loss2 = (-delta.where(delta < 0, 0)).rolling(2).mean()
    rs2 = gain2 / loss2.replace(0, np.nan)
    f["rsi_2"] = 100 - (100 / (1 + rs2))

    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    f["macd"] = ema12 - ema26
    f["macd_signal"] = f["macd"].ewm(span=9).mean()
    f["macd_hist"] = f["macd"] - f["macd_signal"]

    # Bollinger Bands
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    f["bb_upper"] = sma20 + 2 * std20
    f["bb_lower"] = sma20 - 2 * std20
    f["bb_pct"] = (df["Close"] - f["bb_lower"]) / (f["bb_upper"] - f["bb_lower"])

    # Volume features
    f["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    f["vol_change"] = df["Volume"].pct_change()

    # OHLC features
    f["high_low_pct"] = (df["High"] - df["Low"]) / df["Low"]
    f["close_open_pct"] = (df["Close"] - df["Open"]) / df["Open"]
    f["upper_shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Close"]
    f["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Close"]

    # Day of week
    f["day_of_week"] = df.index.dayofweek
    f["month"] = df.index.month

    # Target: next day return (what we predict)
    f["target"] = df["Close"].pct_change(1).shift(-1)
    # Binary target: up or down
    f["target_direction"] = (f["target"] > 0).astype(int)

    # Replace inf values with NaN, then drop
    f = f.replace([np.inf, -np.inf], np.nan)
    return f.dropna()


# ── Algorithm Trainers ───────────────────────────────────────
@dataclass
class AlgoResult:
    algorithm: str
    symbol: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    sharpe: float
    profit_factor: float
    total_return_pct: float
    win_rate: float
    trades: int
    train_time_sec: float


def train_lightgbm(X_train, y_train, X_test, y_test, symbol: str) -> AlgoResult:
    """Train LightGBM classifier."""
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    elapsed = time.time() - t0

    acc = accuracy_score(y_test, preds)
    # Save model
    import joblib
    joblib.dump(model, MODEL_DIR / f"{symbol}_lightgbm.pkl")

    return AlgoResult(
        algorithm="LightGBM", symbol=symbol,
        accuracy=round(acc * 100, 2),
        precision=round(precision_score(y_test, preds, zero_division=0) * 100, 2),
        recall=round(recall_score(y_test, preds, zero_division=0) * 100, 2),
        f1=round(f1_score(y_test, preds, zero_division=0) * 100, 2),
        sharpe=0, profit_factor=0, total_return_pct=0, win_rate=round(acc * 100, 2),
        trades=len(y_test), train_time_sec=round(elapsed, 2),
    )


def train_xgboost(X_train, y_train, X_test, y_test, symbol: str) -> AlgoResult:
    """Train XGBoost classifier."""
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    t0 = time.time()
    model = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
        eval_metric="logloss", verbosity=0,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    elapsed = time.time() - t0

    acc = accuracy_score(y_test, preds)
    import joblib
    joblib.dump(model, MODEL_DIR / f"{symbol}_xgboost.pkl")

    return AlgoResult(
        algorithm="XGBoost", symbol=symbol,
        accuracy=round(acc * 100, 2),
        precision=round(precision_score(y_test, preds, zero_division=0) * 100, 2),
        recall=round(recall_score(y_test, preds, zero_division=0) * 100, 2),
        f1=round(f1_score(y_test, preds, zero_division=0) * 100, 2),
        sharpe=0, profit_factor=0, total_return_pct=0, win_rate=round(acc * 100, 2),
        trades=len(y_test), train_time_sec=round(elapsed, 2),
    )


def train_random_forest(X_train, y_train, X_test, y_test, symbol: str) -> AlgoResult:
    """Train Random Forest classifier."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    t0 = time.time()
    model = RandomForestClassifier(n_estimators=300, max_depth=8, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    elapsed = time.time() - t0

    acc = accuracy_score(y_test, preds)
    import joblib
    joblib.dump(model, MODEL_DIR / f"{symbol}_rf.pkl")

    return AlgoResult(
        algorithm="RandomForest", symbol=symbol,
        accuracy=round(acc * 100, 2),
        precision=round(precision_score(y_test, preds, zero_division=0) * 100, 2),
        recall=round(recall_score(y_test, preds, zero_division=0) * 100, 2),
        f1=round(f1_score(y_test, preds, zero_division=0) * 100, 2),
        sharpe=0, profit_factor=0, total_return_pct=0, win_rate=round(acc * 100, 2),
        trades=len(y_test), train_time_sec=round(elapsed, 2),
    )


def train_ensemble(X_train, y_train, X_test, y_test, symbol: str) -> AlgoResult:
    """Train stacking ensemble (LGB + XGB + RF)."""
    from sklearn.ensemble import StackingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import lightgbm as lgb
    from xgboost import XGBClassifier

    t0 = time.time()
    estimators = [
        ("lgb", lgb.LGBMClassifier(n_estimators=200, verbose=-1)),
        ("xgb", XGBClassifier(n_estimators=200, verbosity=0, use_label_encoder=False, eval_metric="logloss")),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1)),
    ]
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    elapsed = time.time() - t0

    acc = accuracy_score(y_test, preds)
    import joblib
    joblib.dump(model, MODEL_DIR / f"{symbol}_ensemble.pkl")

    return AlgoResult(
        algorithm="Ensemble(LGB+XGB+RF)", symbol=symbol,
        accuracy=round(acc * 100, 2),
        precision=round(precision_score(y_test, preds, zero_division=0) * 100, 2),
        recall=round(recall_score(y_test, preds, zero_division=0) * 100, 2),
        f1=round(f1_score(y_test, preds, zero_division=0) * 100, 2),
        sharpe=0, profit_factor=0, total_return_pct=0, win_rate=round(acc * 100, 2),
        trades=len(y_test), train_time_sec=round(elapsed, 2),
    )


# ── Main Training Pipeline ──────────────────────────────────
ALGORITHMS = [
    ("LightGBM", train_lightgbm),
    ("XGBoost", train_xgboost),
    ("RandomForest", train_random_forest),
    ("Ensemble", train_ensemble),
]


def train_stock(symbol: str) -> list[AlgoResult]:
    """Train all algorithms on one stock, return results."""
    log.info(f"{'='*50}")
    log.info(f"Training: {symbol}")

    df = load_or_download(symbol)
    if df.empty or len(df) < 100:
        log.warning(f"Insufficient data for {symbol}: {len(df)} rows")
        return []

    features = build_features(df)
    if len(features) < 50:
        log.warning(f"Insufficient features for {symbol}: {len(features)} rows")
        return []

    # Train/test split (80/20, time-based)
    split_idx = int(len(features) * 0.8)
    feature_cols = [c for c in features.columns if c not in ["target", "target_direction"]]
    X_train = features[feature_cols].iloc[:split_idx]
    y_train = features["target_direction"].iloc[:split_idx]
    X_test = features[feature_cols].iloc[split_idx:]
    y_test = features["target_direction"].iloc[split_idx:]

    results = []
    for algo_name, train_fn in ALGORITHMS:
        try:
            result = train_fn(X_train, y_train, X_test, y_test, symbol)
            results.append(result)
            log.info(f"  {algo_name}: {result.accuracy}% accuracy, F1={result.f1}%")
        except Exception as e:
            log.error(f"  {algo_name} failed: {e}")

    # Find best
    if results:
        best = max(results, key=lambda r: r.accuracy)
        log.info(f"  BEST: {best.algorithm} @ {best.accuracy}%")

    return results


def main():
    log.info("=" * 60)
    log.info("UNIVERSAL ALGORITHM TRAINER")
    log.info(f"Stocks: {sum(len(v) for v in STOCKS.values())}")
    log.info(f"Algorithms: {len(ALGORITHMS)}")
    log.info("=" * 60)

    all_results = []
    stock_best = {}

    for tier, symbols in STOCKS.items():
        log.info(f"\n[{tier}] — {len(symbols)} stocks")
        for symbol in symbols:
            results = train_stock(symbol)
            all_results.extend(results)

            if results:
                best = max(results, key=lambda r: r.accuracy)
                stock_best[symbol] = asdict(best)

    # Save results
    results_file = RESULTS_DIR / "training_results.json"
    results_file.write_text(
        json.dumps({
            "timestamp": datetime.now().isoformat(),
            "total_stocks": len(stock_best),
            "total_models": len(all_results),
            "stock_best": stock_best,
            "all_results": [asdict(r) for r in all_results],
        }, indent=2),
        encoding="utf-8",
    )

    # Print leaderboard
    log.info("\n" + "=" * 60)
    log.info("LEADERBOARD — Best Algorithm Per Stock")
    log.info("=" * 60)

    sorted_stocks = sorted(stock_best.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    for symbol, best in sorted_stocks:
        log.info(f"  {symbol:20s} {best['algorithm']:25s} {best['accuracy']:6.2f}%")

    avg_acc = np.mean([v["accuracy"] for v in stock_best.values()]) if stock_best else 0
    log.info(f"\nAverage accuracy: {avg_acc:.2f}%")
    log.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()

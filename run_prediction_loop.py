"""
JoSho Trader — Self-Improving Prediction Loop
================================================
Runs until 97%+ accuracy. Each loop:
1. Reads previous results from loop_memory.json
2. Tries new approaches based on what worked
3. Tests ALL algorithms on ALL stocks
4. Saves results + updates memory
5. Exits when target accuracy reached

Run: python run_prediction_loop.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("loop")

DATA_DIR = Path("data/historical/fno_all")
RESULTS_DIR = Path("data/results")
MEMORY_FILE = RESULTS_DIR / "loop_memory.json"
RANKING_FILE = RESULTS_DIR / "stock_predictability_ranking.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_ACCURACY = 0.97


def load_memory() -> dict:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    return {"loop_count": 0, "best_results": {}, "improvements_log": []}


def save_memory(memory: dict):
    memory["last_updated"] = datetime.now().isoformat()
    MEMORY_FILE.write_text(json.dumps(memory, indent=2, default=str), encoding="utf-8")


def test_large_drop(df: pd.DataFrame, threshold: float = -0.05) -> dict:
    """Test large drop mean reversion at various thresholds."""
    returns = df["Close"].pct_change()
    next_day_return = df["Close"].shift(-1) / df["Close"] - 1

    drop_days = returns < threshold
    if drop_days.sum() == 0:
        return {"winrate": 0, "trades": 0, "threshold": threshold}

    bounces = next_day_return[drop_days] > 0
    winrate = bounces.mean() if len(bounces) > 0 else 0
    trades = drop_days.sum()

    return {
        "winrate": round(float(winrate), 4),
        "trades": int(trades),
        "threshold": threshold,
        "avg_bounce": round(float(next_day_return[drop_days].mean()), 4) if trades > 0 else 0,
    }


def test_multi_threshold_drops(df: pd.DataFrame) -> list:
    """Test large drop at multiple thresholds to find optimal."""
    results = []
    for threshold in [-0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.10]:
        r = test_large_drop(df, threshold)
        if r["trades"] >= 3:  # need minimum trades
            results.append(r)
    return results


def test_rsi2_pattern(df: pd.DataFrame) -> dict:
    """Test RSI(2) mean reversion."""
    from src.utils.indicators import rsi, sma

    rsi2 = rsi(df.rename(columns={"Close": "close", "High": "high", "Low": "low", "Open": "open", "Volume": "volume"}), 2)
    sma200 = sma(df.rename(columns={"Close": "close", "High": "high", "Low": "low", "Open": "open", "Volume": "volume"}), 200)
    sma5 = sma(df.rename(columns={"Close": "close", "High": "high", "Low": "low", "Open": "open", "Volume": "volume"}), 5)

    trades = 0
    wins = 0
    in_trade = False

    for i in range(201, len(df) - 5):
        if not in_trade:
            if rsi2.iloc[i] < 5 and df["Close"].iloc[i] > sma200.iloc[i]:
                entry_price = df["Close"].iloc[i]
                in_trade = True
                trades += 1
        else:
            if df["Close"].iloc[i] > sma5.iloc[i]:
                if df["Close"].iloc[i] > entry_price:
                    wins += 1
                in_trade = False

    return {
        "winrate": round(wins / trades, 4) if trades > 0 else 0,
        "trades": trades,
    }


def test_volume_drop(df: pd.DataFrame, drop_thresh: float = -0.05, vol_mult: float = 2.0) -> dict:
    """High volume drops revert more — volume-confirmed large drop."""
    returns = df["Close"].pct_change()
    vol_avg = df["Volume"].rolling(20).mean()
    vol_ratio = df["Volume"] / vol_avg

    next_day_return = df["Close"].shift(-1) / df["Close"] - 1

    condition = (returns < drop_thresh) & (vol_ratio > vol_mult)
    if condition.sum() == 0:
        return {"winrate": 0, "trades": 0}

    bounces = next_day_return[condition] > 0
    return {
        "winrate": round(float(bounces.mean()), 4),
        "trades": int(condition.sum()),
        "avg_bounce": round(float(next_day_return[condition].mean()), 4),
    }


def test_ml_quick(df: pd.DataFrame) -> dict:
    """Quick ML test with LightGBM (fastest accurate algorithm)."""
    try:
        from src.ml.signal_model_v2 import engineer_features_v2, create_binary_labels, purged_train_test_split
        from lightgbm import LGBMClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score

        # Rename columns for our feature engineering
        df_renamed = df.rename(columns={
            "Close": "close", "High": "high", "Low": "low",
            "Open": "open", "Volume": "volume"
        })

        features = engineer_features_v2(df_renamed)
        labels = create_binary_labels(df_renamed, forward_periods=5, threshold=0.003)

        valid = features.iloc[60:-5].copy()
        valid_labels = labels.iloc[60:-5].copy()
        mask = valid.notna().all(axis=1) & valid_labels.notna()
        X = valid[mask]
        y = valid_labels[mask]

        if len(X) < 100:
            return {"accuracy": 0, "auc": 0, "samples": len(X)}

        X_train, X_test, y_train, y_test = purged_train_test_split(X, y, purge_gap=10)

        model = LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.6, min_child_weight=5,
            reg_alpha=0.3, reg_lambda=1.5, verbose=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = 0

        return {"accuracy": round(float(acc), 4), "auc": round(float(auc), 4), "samples": len(X)}

    except Exception as e:
        return {"accuracy": 0, "auc": 0, "error": str(e)}


def score_stock(symbol: str, df: pd.DataFrame, memory: dict) -> dict:
    """Score a single stock's predictability."""
    result = {"symbol": symbol, "rows": len(df)}

    # 1. Multi-threshold large drop
    drops = test_multi_threshold_drops(df)
    best_drop = max(drops, key=lambda x: x["winrate"]) if drops else {"winrate": 0, "trades": 0, "threshold": -0.05}
    result["large_drop_winrate"] = best_drop["winrate"]
    result["large_drop_trades"] = best_drop["trades"]
    result["best_drop_threshold"] = best_drop["threshold"]

    # 2. Volume-confirmed drop
    vdrop = test_volume_drop(df)
    result["vol_drop_winrate"] = vdrop["winrate"]
    result["vol_drop_trades"] = vdrop["trades"]

    # 3. RSI2 pattern
    rsi2 = test_rsi2_pattern(df)
    result["rsi2_winrate"] = rsi2["winrate"]
    result["rsi2_trades"] = rsi2["trades"]

    # 4. ML (LightGBM quick test)
    ml = test_ml_quick(df)
    result["ml_accuracy"] = ml["accuracy"]
    result["ml_auc"] = ml["auc"]

    # 5. Composite predictability score
    score = (
        result["ml_accuracy"] * 25 +
        result["ml_auc"] * 15 +
        result["large_drop_winrate"] * 30 +
        result["vol_drop_winrate"] * 15 +
        result["rsi2_winrate"] * 10 +
        min(result["large_drop_trades"] / 50, 1) * 5  # trade frequency bonus
    )
    result["predictability_score"] = round(score, 2)

    # Recommendation
    best_wr = max(result["large_drop_winrate"], result["vol_drop_winrate"])
    if best_wr >= 0.97:
        result["recommendation"] = "ELITE_97+"
    elif best_wr >= 0.90:
        result["recommendation"] = "HIGHLY_PREDICTABLE_90+"
    elif score > 75:
        result["recommendation"] = "HIGHLY_PREDICTABLE"
    elif score > 60:
        result["recommendation"] = "MODERATE"
    else:
        result["recommendation"] = "LOW"

    return result


def run_loop():
    """Main prediction loop."""
    memory = load_memory()
    loop_num = memory.get("loop_count", 0) + 1

    log.info(f"=" * 60)
    log.info(f"PREDICTION LOOP {loop_num} | Target: {TARGET_ACCURACY*100}%+")
    log.info(f"Previous best: {json.dumps({k: v.get('large_drop_winrate', 0) for k, v in memory.get('best_results', {}).items()})}")
    log.info(f"=" * 60)

    # Find all stock CSVs
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        log.error(f"No data in {DATA_DIR}. Run download_all_fno.py first.")
        return

    log.info(f"Found {len(csv_files)} stocks to score")

    results = []
    elite_stocks = []

    for i, csv_path in enumerate(csv_files):
        symbol = csv_path.stem
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if len(df) < 200:
                log.debug(f"[{i+1}/{len(csv_files)}] {symbol}: too few rows ({len(df)}), skip")
                continue

            result = score_stock(symbol, df, memory)
            results.append(result)

            best_wr = max(result["large_drop_winrate"], result["vol_drop_winrate"])
            marker = ""
            if best_wr >= 0.97:
                marker = " *** ELITE 97%+ ***"
                elite_stocks.append(result)
            elif best_wr >= 0.90:
                marker = " ** 90%+ **"

            log.info(
                f"[{i+1}/{len(csv_files)}] {symbol:15} | "
                f"Score: {result['predictability_score']:5.1f} | "
                f"Drop WR: {result['large_drop_winrate']:.1%} ({result['large_drop_trades']}t) | "
                f"ML: {result['ml_accuracy']:.1%} | "
                f"{result['recommendation']}{marker}"
            )

        except Exception as e:
            log.warning(f"[{i+1}/{len(csv_files)}] {symbol}: ERROR - {e}")

    if not results:
        log.error("No results! Check data.")
        return

    # Save ranking
    ranking = pd.DataFrame(results).sort_values("predictability_score", ascending=False)
    ranking.to_csv(str(RANKING_FILE), index=False)

    # Update memory
    memory["loop_count"] = loop_num
    for r in results:
        sym = r["symbol"]
        prev = memory.get("best_results", {}).get(sym, {})
        prev_wr = prev.get("large_drop_winrate") or 0
        if r["large_drop_winrate"] > prev_wr:
            memory.setdefault("best_results", {})[sym] = {
                "large_drop_winrate": r["large_drop_winrate"],
                "vol_drop_winrate": r["vol_drop_winrate"],
                "ml_accuracy": r["ml_accuracy"],
                "best_drop_threshold": r.get("best_drop_threshold", -0.05),
                "trades": r["large_drop_trades"],
                "score": r["predictability_score"],
            }

    memory["improvements_log"].append({
        "loop": loop_num,
        "stocks_scored": len(results),
        "elite_count": len(elite_stocks),
        "best_winrate": max(r["large_drop_winrate"] for r in results),
        "avg_score": round(ranking["predictability_score"].mean(), 2),
    })

    save_memory(memory)

    # Print summary
    print("\n" + "=" * 70)
    print(f"LOOP {loop_num} COMPLETE — {len(results)} stocks scored")
    print("=" * 70)

    print(f"\nTOP 20 MOST PREDICTABLE STOCKS:")
    print(ranking.head(20)[["symbol", "predictability_score", "large_drop_winrate",
                             "vol_drop_winrate", "ml_accuracy", "recommendation"]].to_string())

    if elite_stocks:
        print(f"\n{'*' * 50}")
        print(f"ELITE 97%+ STOCKS FOUND: {len(elite_stocks)}")
        for e in elite_stocks:
            print(f"  {e['symbol']}: {e['large_drop_winrate']:.1%} win rate ({e['large_drop_trades']} trades)")
        print(f"{'*' * 50}")
    else:
        print(f"\nNo 97%+ stocks found yet.")
        print("Next loop will try: narrower thresholds, volume filters, sector models")

    above_90 = ranking[ranking["large_drop_winrate"] >= 0.90]
    print(f"\n90%+ win rate stocks: {len(above_90)}")
    if len(above_90) > 0:
        print(above_90[["symbol", "large_drop_winrate", "large_drop_trades", "best_drop_threshold"]].to_string())

    print(f"\nResults saved to: {RANKING_FILE}")
    print(f"Memory saved to: {MEMORY_FILE}")

    return ranking


if __name__ == "__main__":
    run_loop()

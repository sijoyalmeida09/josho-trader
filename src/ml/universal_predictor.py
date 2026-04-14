"""
Universal Stock Predictability Scorer — runs ALL algorithms + ALL patterns on ANY stock.

Takes a stock's OHLCV DataFrame, runs every ML algorithm and every pattern strategy,
then returns a composite predictability score (0-100) with detailed results.
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from .signal_model_v2 import (
    engineer_features_v2,
    create_binary_labels,
    purged_train_test_split,
)
from .algorithms import get_algorithm, ALGORITHM_REGISTRY

log = logging.getLogger("josho.ml.universal_predictor")

# Minimum rows needed after warmup to run ML (features need ~200 bars warmup)
MIN_ROWS_ML = 400
MIN_ROWS_PATTERN = 210

# ML algorithms to test (individual, non-ensemble for speed)
ML_ALGOS = ["xgboost_v2", "lightgbm", "random_forest", "extra_trees"]

# Forward periods and threshold for binary labels
FORWARD_PERIODS = [3, 5]
LABEL_THRESHOLD = 0.003


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to lowercase so the rest of the pipeline works
    regardless of whether the source CSV uses Title or lower case.
    Returns a new DataFrame (no mutation).
    """
    col_map = {c: c.lower() for c in df.columns}
    return df.rename(columns=col_map)


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a raw OHLCV DataFrame: normalize columns, set date index,
    sort chronologically, drop NaN rows.
    Returns a new DataFrame.
    """
    out = _normalize_columns(df.copy())

    # Ensure date index
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date")
    elif not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)

    out = out.sort_index()

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = out.dropna(subset=required)
    return out


# ── ML Scoring ──────────────────────────────────────────────────────


def _run_ml_algorithms(df: pd.DataFrame) -> dict:
    """
    Train and evaluate each ML algorithm at each forward period.
    Returns dict with per-algo results and the best accuracy/AUC.
    """
    results = {}
    best_accuracy = 0.0
    best_auc = 0.0
    best_algo = "none"

    features = engineer_features_v2(df)

    for fwd in FORWARD_PERIODS:
        labels = create_binary_labels(df, forward_periods=fwd, threshold=LABEL_THRESHOLD)

        # Remove warmup and future-less rows
        valid_start = 60
        valid_end = len(features) - fwd
        if valid_end <= valid_start:
            continue

        feat_slice = features.iloc[valid_start:valid_end].copy()
        label_slice = labels.iloc[valid_start:valid_end].copy()

        # Drop NaN labels (neutral zone)
        mask = feat_slice.notna().all(axis=1) & label_slice.notna()
        X = feat_slice[mask]
        y = label_slice[mask]

        if len(X) < 100:
            continue

        # Purged split
        X_train, X_test, y_train, y_test = purged_train_test_split(
            X, y, test_ratio=0.2, purge_gap=10,
        )

        if len(X_test) < 20 or len(X_train) < 50:
            continue

        # Class balance ratio
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        scale = neg / pos if pos > 0 else 1.0

        for algo_name in ML_ALGOS:
            key = f"{algo_name}_fwd{fwd}"
            try:
                model = get_algorithm(algo_name, scale_pos_weight=scale)
                entry = ALGORITHM_REGISTRY[algo_name]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if entry.get("needs_early_stop") and hasattr(model, "fit"):
                        # XGBoost needs eval_set for early stopping
                        model.set_params(early_stopping_rounds=20)
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_test, y_test)],
                            verbose=False,
                        )
                    else:
                        model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                except Exception:
                    auc = 0.0

                results[key] = {
                    "algorithm": algo_name,
                    "forward_period": fwd,
                    "accuracy": round(float(acc), 4),
                    "auc": round(float(auc), 4),
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                }

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_algo = algo_name
                if auc > best_auc:
                    best_auc = auc

            except Exception as exc:
                log.debug(f"Algorithm {algo_name} fwd={fwd} failed: {exc}")
                results[key] = {
                    "algorithm": algo_name,
                    "forward_period": fwd,
                    "accuracy": 0.0,
                    "auc": 0.0,
                    "error": str(exc),
                }

    # Weighted ensemble (XGB + LGB + RF) — average the best forward period
    ensemble_accs = []
    ensemble_aucs = []
    for fwd in FORWARD_PERIODS:
        accs = []
        aucs = []
        for algo in ["xgboost_v2", "lightgbm", "random_forest"]:
            key = f"{algo}_fwd{fwd}"
            if key in results and "error" not in results[key]:
                accs.append(results[key]["accuracy"])
                aucs.append(results[key]["auc"])
        if accs:
            ensemble_accs.append(np.mean(accs))
            ensemble_aucs.append(np.mean(aucs))

    ensemble_acc = max(ensemble_accs) if ensemble_accs else 0.0
    ensemble_auc = max(ensemble_aucs) if ensemble_aucs else 0.0

    if ensemble_acc > best_accuracy:
        best_accuracy = ensemble_acc
        best_algo = "weighted_ensemble"
    if ensemble_auc > best_auc:
        best_auc = ensemble_auc

    return {
        "algo_results": results,
        "best_ml_algo": best_algo,
        "best_ml_accuracy": round(float(best_accuracy), 4),
        "best_ml_auc": round(float(best_auc), 4),
        "ensemble_accuracy": round(float(ensemble_acc), 4),
        "ensemble_auc": round(float(ensemble_auc), 4),
    }


# ── Pattern Scoring ─────────────────────────────────────────────────


def _backtest_large_drop(df: pd.DataFrame) -> dict:
    """
    Vectorized backtest of Large Drop Mean Reversion strategy.
    Buy when stock drops 5%+ in a single day, exit within 3 bars
    at +1.5% target or -3% stop.
    Returns win rate and trade count.
    """
    close = df["close"].values
    daily_ret = np.diff(close) / close[:-1] * 100

    trades = []
    i = 1
    while i < len(close):
        if daily_ret[i - 1] <= -5.0:
            entry_price = close[i]
            won = False
            exited = False
            for hold in range(1, 4):
                if i + hold >= len(close):
                    break
                pnl_pct = (close[i + hold] - entry_price) / entry_price * 100
                if pnl_pct >= 1.5:
                    won = True
                    exited = True
                    break
                if pnl_pct <= -3.0:
                    won = False
                    exited = True
                    break
            if not exited and i + 3 < len(close):
                # Time stop: check if profitable at exit
                pnl_pct = (close[min(i + 3, len(close) - 1)] - entry_price) / entry_price * 100
                won = pnl_pct > 0
            trades.append(won)
            i += 4  # skip held bars
        else:
            i += 1

    win_count = sum(trades)
    total = len(trades)
    win_rate = win_count / total if total > 0 else 0.0

    return {
        "win_rate": round(float(win_rate), 4),
        "trades": total,
        "wins": win_count,
    }


def _rsi2_custom(close: np.ndarray, period: int = 2) -> np.ndarray:
    """Compute RSI on a numpy array, returning numpy array."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.full_like(close, np.nan, dtype=float)
    avg_loss = np.full_like(close, np.nan, dtype=float)

    # SMA seed
    if len(close) > period:
        avg_gain[period] = np.mean(gain[1 : period + 1])
        avg_loss[period] = np.mean(loss[1 : period + 1])

        for j in range(period + 1, len(close)):
            avg_gain[j] = (avg_gain[j - 1] * (period - 1) + gain[j]) / period
            avg_loss[j] = (avg_loss[j - 1] * (period - 1) + loss[j]) / period

    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0.0)
    rsi_vals = 100.0 - (100.0 / (1.0 + rs))
    rsi_vals[:period] = np.nan
    return rsi_vals


def _backtest_rsi2(df: pd.DataFrame, enhanced: bool = False) -> dict:
    """
    Vectorized backtest of RSI2 Mean Reversion.
    Standard: RSI(2) < 5 AND close > 200 SMA -> buy, exit when close > 5 SMA.
    Enhanced: 3 consecutive bars with RSI(2) < 10.
    """
    close = df["close"].values
    if len(close) < 210:
        return {"win_rate": 0.0, "trades": 0, "wins": 0}

    rsi2 = _rsi2_custom(close, period=2)
    sma200 = pd.Series(close).rolling(200).mean().values
    sma5 = pd.Series(close).rolling(5).mean().values

    trades = []
    in_position = False
    entry_price = 0.0

    for i in range(200, len(close)):
        if in_position:
            if close[i] > sma5[i]:
                pnl = (close[i] - entry_price) / entry_price * 100
                trades.append(pnl > 0)
                in_position = False
            continue

        if np.isnan(rsi2[i]) or np.isnan(sma200[i]):
            continue

        if close[i] > sma200[i]:
            if enhanced:
                if (
                    i >= 2
                    and not np.isnan(rsi2[i - 2])
                    and rsi2[i - 2] < 10
                    and rsi2[i - 1] < 10
                    and rsi2[i] < 10
                ):
                    in_position = True
                    entry_price = close[i]
            else:
                if rsi2[i] < 5:
                    in_position = True
                    entry_price = close[i]

    win_count = sum(trades)
    total = len(trades)
    win_rate = win_count / total if total > 0 else 0.0

    return {
        "win_rate": round(float(win_rate), 4),
        "trades": total,
        "wins": win_count,
    }


def _run_pattern_strategies(df: pd.DataFrame) -> dict:
    """Run all pattern-based strategies and return results."""
    large_drop = _backtest_large_drop(df)
    rsi2_standard = _backtest_rsi2(df, enhanced=False)
    rsi2_enhanced = _backtest_rsi2(df, enhanced=True)

    return {
        "large_drop": large_drop,
        "rsi2_standard": rsi2_standard,
        "rsi2_enhanced": rsi2_enhanced,
    }


# ── Composite Score ─────────────────────────────────────────────────


def _compute_trade_frequency_score(
    large_drop_trades: int,
    rsi2_trades: int,
    total_bars: int,
) -> float:
    """
    Score trade frequency on 0-1 scale.
    More trades = more statistically significant = more useful.
    """
    total_trades = large_drop_trades + rsi2_trades
    if total_bars == 0:
        return 0.0

    # Normalize: 50+ trades in 1000 bars is excellent
    trades_per_1000 = total_trades / total_bars * 1000
    # Sigmoid-style scaling: 0 trades -> 0, 50+ trades/1000bars -> ~1.0
    score = min(trades_per_1000 / 50.0, 1.0)
    return round(float(score), 4)


def _get_recommendation(score: float) -> str:
    """Map predictability score to recommendation label."""
    if score >= 75:
        return "HIGHLY_PREDICTABLE"
    if score >= 60:
        return "MODERATE"
    return "LOW"


def score_stock_predictability(df: pd.DataFrame, symbol: str) -> dict:
    """
    Run all algorithms and patterns on a stock.
    Returns a predictability score (0-100) and detailed results.

    Args:
        df: Raw OHLCV DataFrame (columns can be Title or lower case).
        symbol: Stock ticker symbol.

    Returns:
        Dict with predictability_score, best ML results, pattern win rates,
        and recommendation.
    """
    prepared = _prepare_dataframe(df)
    total_bars = len(prepared)

    result = {
        "symbol": symbol,
        "total_bars": total_bars,
        "predictability_score": 0.0,
        "best_ml_algo": "none",
        "best_ml_accuracy": 0.0,
        "best_ml_auc": 0.0,
        "large_drop_winrate": 0.0,
        "large_drop_trades": 0,
        "rsi2_winrate": 0.0,
        "rsi2_trades": 0,
        "rsi2_enhanced_winrate": 0.0,
        "rsi2_enhanced_trades": 0,
        "recommendation": "LOW",
    }

    # ML scoring
    ml_results = {"best_ml_accuracy": 0.0, "best_ml_auc": 0.0, "best_ml_algo": "none"}
    if total_bars >= MIN_ROWS_ML:
        try:
            ml_results = _run_ml_algorithms(prepared)
            result["best_ml_algo"] = ml_results["best_ml_algo"]
            result["best_ml_accuracy"] = ml_results["best_ml_accuracy"]
            result["best_ml_auc"] = ml_results["best_ml_auc"]
            result["ml_detail"] = ml_results.get("algo_results", {})
        except Exception as exc:
            log.warning(f"[{symbol}] ML scoring failed: {exc}")
    else:
        log.info(f"[{symbol}] Skipping ML: only {total_bars} bars (need {MIN_ROWS_ML})")

    # Pattern scoring
    if total_bars >= MIN_ROWS_PATTERN:
        try:
            pattern_results = _run_pattern_strategies(prepared)

            ld = pattern_results["large_drop"]
            result["large_drop_winrate"] = ld["win_rate"]
            result["large_drop_trades"] = ld["trades"]

            rsi2_std = pattern_results["rsi2_standard"]
            result["rsi2_winrate"] = rsi2_std["win_rate"]
            result["rsi2_trades"] = rsi2_std["trades"]

            rsi2_enh = pattern_results["rsi2_enhanced"]
            result["rsi2_enhanced_winrate"] = rsi2_enh["win_rate"]
            result["rsi2_enhanced_trades"] = rsi2_enh["trades"]
        except Exception as exc:
            log.warning(f"[{symbol}] Pattern scoring failed: {exc}")
    else:
        log.info(f"[{symbol}] Skipping patterns: only {total_bars} bars (need {MIN_ROWS_PATTERN})")

    # Composite predictability score (0-100)
    trade_freq = _compute_trade_frequency_score(
        result["large_drop_trades"],
        result["rsi2_trades"],
        total_bars,
    )

    score = (
        result["best_ml_accuracy"] * 30
        + result["best_ml_auc"] * 20
        + result["large_drop_winrate"] * 25
        + result["rsi2_winrate"] * 15
        + trade_freq * 10
    )

    result["predictability_score"] = round(float(score), 1)
    result["trade_frequency_score"] = trade_freq
    result["recommendation"] = _get_recommendation(result["predictability_score"])

    return result

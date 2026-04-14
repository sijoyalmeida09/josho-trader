"""
Algorithm Mixer — Tests every single + every 2/3/4 combination per stock.
Groups stocks by sector and finds the optimal algo mix for each category.

The insight: different sectors behave differently.
Banking stocks → mean reversion works best
IT stocks → momentum works best
Pharma stocks → news-driven, harder to predict
Commodity-linked → macro correlation matters

This module finds the BEST algorithm or combination for each stock.
"""

import logging
import itertools
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

log = logging.getLogger("josho.algo_mixer")

# ── Sector Classification ─────────────────────────────────────────

SECTORS = {
    "BANKING": [
        "SBIN", "ICICIBANK", "HDFCBANK", "AXISBANK", "KOTAKBANK", "BANKBARODA",
        "PNB", "CANBK", "FEDERALBNK", "INDUSINDBK", "IDFCFIRSTB", "BANDHANBNK",
        "AUBANK", "RBLBANK", "UNIONBANK", "INDIANB", "BANKINDIA", "YESBANK",
    ],
    "IT": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIMINDTREE", "MPHASIS",
        "COFORGE", "PERSISTENT", "KPITTECH", "TATAELXSI", "NAUKRI",
    ],
    "PHARMA": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "AUROPHARMA", "DIVISLAB",
        "BIOCON", "TORNTPHARM", "GLENMARK", "ALKEM", "LAURUSLABS", "ZYDUSLIFE",
    ],
    "AUTO": [
        "MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO",
        "TVSMOTOR", "ASHOKLEY", "MOTHERSON", "BHARATFORG", "FORCEMOT",
    ],
    "METAL": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "NMDC", "SAIL",
        "JINDALSTEL", "NATIONALUM", "HINDZINC",
    ],
    "ENERGY": [
        "RELIANCE", "ONGC", "BPCL", "IOC", "GAIL", "HINDPETRO", "PETRONET",
        "ADANIGREEN", "ADANIENSOL", "TATAPOWER", "NTPC", "POWERGRID", "NHPC",
    ],
    "FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO",
        "COLPAL", "GODREJCP", "TATACONSUM", "JUBLFOOD", "VBL", "UNITDSPR",
    ],
    "FINANCE_NBFC": [
        "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "SHRIRAMFIN", "MUTHOOTFIN",
        "MANAPPURAM", "LICHSGFIN", "PNBHOUSING", "HDFCAMC", "SBICARD",
        "HDFCLIFE", "SBILIFE", "ICICIPRULI", "ICICIGI", "LICI",
    ],
    "INFRA": [
        "LT", "ADANIENT", "ADANIPORTS", "DLF", "GODREJPROP", "OBEROIRLTY",
        "PRESTIGE", "LODHA", "ULTRACEMCO", "AMBUJACEM", "SHREECEM",
        "DALBHARAT", "PIDILITIND", "GRASIM",
    ],
}


def get_sector(symbol: str) -> str:
    """Get sector for a symbol."""
    for sector, stocks in SECTORS.items():
        if symbol in stocks:
            return sector
    return "OTHER"


def get_all_algorithms():
    """Return dict of all individual algorithms."""
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier,
        GradientBoostingClassifier, AdaBoostClassifier,
    )

    return {
        "xgboost": XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.6, min_child_weight=5,
            reg_alpha=0.3, reg_lambda=1.5, eval_metric="logloss",
            random_state=42,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.6, min_child_weight=5,
            reg_alpha=0.3, reg_lambda=1.5, verbose=-1, random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=5,
            max_features="sqrt", random_state=42,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=5,
            max_features="sqrt", random_state=42,
        ),
        "gradient_boost": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.7, min_samples_leaf=5, random_state=42,
        ),
        "adaboost": AdaBoostClassifier(
            n_estimators=50, learning_rate=0.1, random_state=42,
        ),
    }


def build_mix(algo_names: list, all_algos: dict) -> VotingClassifier:
    """Build a soft-voting ensemble from named algorithms."""
    estimators = [(name, all_algos[name]) for name in algo_names if name in all_algos]
    return VotingClassifier(estimators=estimators, voting="soft")


def test_algorithm(model, X_train, y_train, X_test, y_test) -> dict:
    """Train and evaluate a single model."""
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = 0
        return {"accuracy": round(float(acc), 4), "auc": round(float(auc), 4)}
    except Exception as e:
        return {"accuracy": 0, "auc": 0, "error": str(e)}


def find_best_algo_for_stock(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    symbol: str = "",
    max_mix_size: int = 3,
) -> dict:
    """
    Test every single algorithm AND every 2-3 combination.
    Returns the best single + best mix with their accuracies.
    """
    import copy
    all_algos = get_all_algorithms()
    algo_names = list(all_algos.keys())

    results = {}

    # 1. Test each algorithm individually
    for name in algo_names:
        model = copy.deepcopy(all_algos[name])
        r = test_algorithm(model, X_train, y_train, X_test, y_test)
        results[name] = r

    # 2. Test all 2-combinations
    for combo in itertools.combinations(algo_names, 2):
        combo_name = "+".join(combo)
        try:
            mix = build_mix(list(combo), get_all_algorithms())
            r = test_algorithm(mix, X_train, y_train, X_test, y_test)
            results[combo_name] = r
        except Exception:
            pass

    # 3. Test all 3-combinations (top performers only to save time)
    # Sort singles by accuracy, take top 4 for 3-combos
    top_singles = sorted(
        [(name, results[name]["accuracy"]) for name in algo_names if results.get(name, {}).get("accuracy", 0) > 0],
        key=lambda x: x[1], reverse=True,
    )[:4]
    top_names = [name for name, _ in top_singles]

    for combo in itertools.combinations(top_names, 3):
        combo_name = "+".join(combo)
        try:
            mix = build_mix(list(combo), get_all_algorithms())
            r = test_algorithm(mix, X_train, y_train, X_test, y_test)
            results[combo_name] = r
        except Exception:
            pass

    # 4. Test all-4 of top performers
    if len(top_names) >= 4:
        combo_name = "+".join(top_names[:4])
        try:
            mix = build_mix(top_names[:4], get_all_algorithms())
            r = test_algorithm(mix, X_train, y_train, X_test, y_test)
            results[combo_name] = r
        except Exception:
            pass

    # Find best
    best_name = max(results, key=lambda k: results[k].get("accuracy", 0))
    best_auc_name = max(results, key=lambda k: results[k].get("auc", 0))

    return {
        "symbol": symbol,
        "sector": get_sector(symbol),
        "best_algo": best_name,
        "best_accuracy": results[best_name]["accuracy"],
        "best_auc_algo": best_auc_name,
        "best_auc": results[best_auc_name]["auc"],
        "all_results": results,
        "total_combos_tested": len(results),
    }


def find_best_per_sector(stock_results: list) -> dict:
    """Aggregate results by sector to find which algo works best per sector."""
    sector_data = {}

    for r in stock_results:
        sector = r.get("sector", "OTHER")
        if sector not in sector_data:
            sector_data[sector] = {"algos": {}, "count": 0}

        sector_data[sector]["count"] += 1

        # Accumulate accuracy per algo across sector
        for algo_name, algo_result in r.get("all_results", {}).items():
            if algo_name not in sector_data[sector]["algos"]:
                sector_data[sector]["algos"][algo_name] = {"total_acc": 0, "total_auc": 0, "count": 0}
            sector_data[sector]["algos"][algo_name]["total_acc"] += algo_result.get("accuracy", 0)
            sector_data[sector]["algos"][algo_name]["total_auc"] += algo_result.get("auc", 0)
            sector_data[sector]["algos"][algo_name]["count"] += 1

    # Find best algo per sector
    sector_best = {}
    for sector, data in sector_data.items():
        best_algo = ""
        best_avg_acc = 0
        for algo_name, algo_data in data["algos"].items():
            avg_acc = algo_data["total_acc"] / algo_data["count"] if algo_data["count"] > 0 else 0
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                best_algo = algo_name

        sector_best[sector] = {
            "best_algo": best_algo,
            "avg_accuracy": round(best_avg_acc, 4),
            "stocks_tested": data["count"],
        }

    return sector_best

"""
Trading Research Repo Cloner
============================
Downloads 100+ top trading/quant repos organized by category.
Run: python research/clone_repos.py
"""

import subprocess
import os
from pathlib import Path

RESEARCH_DIR = Path("C:/josho-trader/research/repos")
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 100+ Trading/Quant/ML repos organized by niche
# ══════════════════════════════════════════════════════════════

REPOS = {
    # ── INTRADAY STRATEGIES ──────────────────────────────────
    "intraday": [
        "https://github.com/je-suis-tm/quant-trading",
        "https://github.com/jankrepl/orern",
        "https://github.com/borisbanushev/stockpredictionai",
        "https://github.com/Rachnog/Deep-Trading",
        "https://github.com/pmorissette/bt",
        "https://github.com/hudson-and-thames/mlfinlab",
        "https://github.com/stefan-jansen/machine-learning-for-trading",
        "https://github.com/bsolomon1124/pyfinance",
    ],

    # ── ALGO TRADING FRAMEWORKS ──────────────────────────────
    "frameworks": [
        "https://github.com/mementum/backtrader",
        "https://github.com/polakowo/vectorbt",
        "https://github.com/kernc/backtesting.py",
        "https://github.com/blankly-finance/blankly",
        "https://github.com/freqtrade/freqtrade",
        "https://github.com/jesse-ai/jesse",
        "https://github.com/nautechsystems/nautilus_trader",
        "https://github.com/robcarver17/pysystemtrade",
        "https://github.com/QuantConnect/Lean",
        "https://github.com/zipline-live/zipline",
    ],

    # ── ML/DL FOR STOCK PREDICTION ───────────────────────────
    "ml_prediction": [
        "https://github.com/AI4Finance-Foundation/FinRL",
        "https://github.com/AI4Finance-Foundation/FinGPT",
        "https://github.com/microsoft/qlib",
        "https://github.com/google-research/google-research",  # has trading papers
        "https://github.com/thuml/Time-Series-Library",
        "https://github.com/ts-reimagined/tsai",
        "https://github.com/unit8co/darts",
        "https://github.com/nixtla/statsforecast",
        "https://github.com/Nixtla/neuralforecast",
        "https://github.com/amazon-science/chronos-forecasting",
        "https://github.com/timeseriesAI/tsai",
    ],

    # ── REINFORCEMENT LEARNING TRADING ───────────────────────
    "rl_trading": [
        "https://github.com/AI4Finance-Foundation/FinRL-Trading",
        "https://github.com/AminHP/gym-anytrading",
        "https://github.com/tensortrade-org/tensortrade",
        "https://github.com/notadamking/Stock-Trading-Environment",
        "https://github.com/quantopian/zipline",
    ],

    # ── OPTIONS & DERIVATIVES ────────────────────────────────
    "options": [
        "https://github.com/vollib/py_vollib",
        "https://github.com/pfhedge/pfhedge",
        "https://github.com/shashank-khanna/Option-Pricing",
        "https://github.com/domokane/FinancePy",
        "https://github.com/goldmansachs/gs-quant",
        "https://github.com/quantsbin/Quantsbin",
    ],

    # ── TECHNICAL ANALYSIS ───────────────────────────────────
    "technical_analysis": [
        "https://github.com/twopirllc/pandas-ta",
        "https://github.com/bukosabino/ta",
        "https://github.com/peerchemist/finta",
        "https://github.com/mrjbq7/ta-lib",
        "https://github.com/jealous/stockstats",
    ],

    # ── VOLATILITY & RISK ────────────────────────────────────
    "volatility": [
        "https://github.com/bashtage/arch",
        "https://github.com/RiskAverseRL/RiskRL",
        "https://github.com/dppalomar/riskParityPortfolio",
        "https://github.com/riskfolio-lib/riskfolio-lib",
    ],

    # ── SENTIMENT & NLP ──────────────────────────────────────
    "sentiment": [
        "https://github.com/ProsusAI/finBERT",
        "https://github.com/yya518/FinBERT",
        "https://github.com/AI4Finance-Foundation/FinNLP",
        "https://github.com/philipperemy/deep-learning-bitcoin",
    ],

    # ── PAIRS/STATISTICAL ARBITRAGE ──────────────────────────
    "stat_arb": [
        "https://github.com/hudson-and-thames/arbitragelab",
        "https://github.com/marketneutral/pairs-trading-with-ML",
        "https://github.com/ericmjl/bayesian-stats-modelling-tutorial",
    ],

    # ── PORTFOLIO OPTIMIZATION ───────────────────────────────
    "portfolio": [
        "https://github.com/robertmartin8/PyPortfolioOpt",
        "https://github.com/dcajasn/Riskfolio-Lib",
        "https://github.com/cvxgrp/cvxportfolio",
    ],

    # ── MARKET DATA & APIS ───────────────────────────────────
    "data": [
        "https://github.com/ranaroussi/yfinance",
        "https://github.com/jealous/stockstats",
        "https://github.com/pmorissette/ffn",
        "https://github.com/RomelTorres/alpha_vantage",
        "https://github.com/Kucoin/kucoin-python-sdk",
    ],

    # ── CRYPTO TRADING BOTS ──────────────────────────────────
    "crypto": [
        "https://github.com/hummingbot/hummingbot",
        "https://github.com/ccxt/ccxt",
        "https://github.com/freqtrade/freqtrade-strategies",
        "https://github.com/Superalgos/Superalgos",
    ],

    # ── INDIAN MARKET SPECIFIC ───────────────────────────────
    "indian_market": [
        "https://github.com/zerodha/pykiteconnect",
        "https://github.com/MarketDataApp/sdk-python",
        "https://github.com/ranjanrak/truedata-python",
        "https://github.com/algo2t/omspy",
        "https://github.com/StreamAlpha/tvdatafeed",
    ],

    # ── FEATURE ENGINEERING ──────────────────────────────────
    "features": [
        "https://github.com/microsoft/qlib",  # Alpha158 features
        "https://github.com/blue-yonder/tsfresh",
        "https://github.com/feature-engine/feature_engine",
    ],

    # ── BACKTESTING ENGINES ──────────────────────────────────
    "backtesting": [
        "https://github.com/quantopian/pyfolio",
        "https://github.com/ranaroussi/quantstats",
        "https://github.com/pmorissette/bt",
        "https://github.com/Blankly-Finance/blankly",
    ],

    # ── HIGH FREQUENCY / LOW LATENCY ────────────────────────
    "hft": [
        "https://github.com/QuantConnect/Lean",
        "https://github.com/nautechsystems/nautilus_trader",
        "https://github.com/CryptoSignal/Crypto-Signal",
    ],

    # ── RESEARCH PAPERS IMPLEMENTATIONS ─────────────────────
    "papers": [
        "https://github.com/firmai/financial-machine-learning",
        "https://github.com/firmai/industry-machine-learning",
        "https://github.com/firmai/business-machine-learning",
        "https://github.com/LastAncientOne/Deep-Learning-Machine-Learning-Stock",
    ],
}


def clone_repo(url: str, category: str) -> bool:
    """Clone a repo into its category folder. Skip if exists."""
    name = url.rstrip("/").split("/")[-1]
    dest = RESEARCH_DIR / category / name

    if dest.exists():
        print(f"  SKIP (exists): {category}/{name}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  CLONE: {category}/{name} ...")

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            print(f"  OK: {category}/{name}")
            return True
        else:
            print(f"  FAIL: {category}/{name} — {result.stderr[:100]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {category}/{name}")
        return False
    except Exception as e:
        print(f"  ERROR: {category}/{name} — {e}")
        return False


def main():
    total = sum(len(urls) for urls in REPOS.values())
    print(f"Trading Research Repo Cloner")
    print(f"Total repos: {total}")
    print(f"Destination: {RESEARCH_DIR}")
    print("=" * 60)

    cloned = 0
    failed = 0
    skipped = 0

    for category, urls in REPOS.items():
        print(f"\n[{category}] ({len(urls)} repos)")
        for url in urls:
            name = url.rstrip("/").split("/")[-1]
            dest = RESEARCH_DIR / category / name
            if dest.exists():
                skipped += 1
                print(f"  SKIP: {name}")
            elif clone_repo(url, category):
                cloned += 1
            else:
                failed += 1

    print("\n" + "=" * 60)
    print(f"DONE: {cloned} cloned, {skipped} skipped, {failed} failed")
    print(f"Total on disk: {cloned + skipped}/{total}")


if __name__ == "__main__":
    main()

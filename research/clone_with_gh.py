"""Clone repos using gh CLI (better auth handling than raw git)."""
import subprocess, shutil
from pathlib import Path

RESEARCH_DIR = Path("C:/josho-trader/research/repos")

REPOS = {
    "intraday": [
        "je-suis-tm/quant-trading", "borisbanushev/stockpredictionai",
        "Rachnog/Deep-Trading", "pmorissette/bt",
        "stefan-jansen/machine-learning-for-trading",
    ],
    "frameworks": [
        "mementum/backtrader", "polakowo/vectorbt", "kernc/backtesting.py",
        "freqtrade/freqtrade", "jesse-ai/jesse", "robcarver17/pysystemtrade",
        "QuantConnect/Lean",
    ],
    "ml_prediction": [
        "AI4Finance-Foundation/FinRL", "AI4Finance-Foundation/FinGPT",
        "microsoft/qlib", "thuml/Time-Series-Library",
        "unit8co/darts", "nixtla/statsforecast", "Nixtla/neuralforecast",
        "amazon-science/chronos-forecasting",
    ],
    "rl_trading": [
        "AminHP/gym-anytrading", "tensortrade-org/tensortrade",
        "notadamking/Stock-Trading-Environment",
    ],
    "options": [
        "vollib/py_vollib", "domokane/FinancePy",
        "goldmansachs/gs-quant",
    ],
    "technical_analysis": [
        "twopirllc/pandas-ta", "bukosabino/ta", "peerchemist/finta",
    ],
    "volatility": [
        "bashtage/arch", "riskfolio-lib/riskfolio-lib",
    ],
    "sentiment": [
        "ProsusAI/finBERT", "AI4Finance-Foundation/FinNLP",
    ],
    "stat_arb": [
        "hudson-and-thames/arbitragelab",
    ],
    "portfolio": [
        "robertmartin8/PyPortfolioOpt", "dcajasn/Riskfolio-Lib",
    ],
    "data": [
        "ranaroussi/yfinance", "pmorissette/ffn",
    ],
    "crypto": [
        "hummingbot/hummingbot", "ccxt/ccxt",
        "freqtrade/freqtrade-strategies",
    ],
    "indian_market": [
        "zerodha/pykiteconnect", "algo2t/omspy",
        "StreamAlpha/tvdatafeed",
    ],
    "features": [
        "blue-yonder/tsfresh",
    ],
    "backtesting": [
        "quantopian/pyfolio", "ranaroussi/quantstats",
    ],
    "hft": [
        "nautechsystems/nautilus_trader",
    ],
    "papers": [
        "firmai/financial-machine-learning",
        "LastAncientOne/Deep-Learning-Machine-Learning-Stock",
    ],
}

cloned = 0
failed = 0
skipped = 0
total = sum(len(v) for v in REPOS.values())

print(f"Cloning {total} repos via gh CLI...")

for cat, repos in REPOS.items():
    for repo in repos:
        name = repo.split("/")[-1]
        dest = RESEARCH_DIR / cat / name

        # Skip if already has content
        if dest.exists() and any(dest.iterdir()):
            print(f"  SKIP: {cat}/{name}")
            skipped += 1
            continue

        # Clean empty dirs from failed clones
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = subprocess.run(
                ["gh", "repo", "clone", repo, str(dest), "--", "--depth", "1"],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode == 0:
                print(f"  OK: {cat}/{name}")
                cloned += 1
            else:
                print(f"  FAIL: {cat}/{name} — {r.stderr[:80]}")
                failed += 1
        except Exception as e:
            print(f"  ERR: {cat}/{name} — {e}")
            failed += 1

print(f"\nDONE: {cloned} cloned, {skipped} existed, {failed} failed = {cloned+skipped}/{total}")

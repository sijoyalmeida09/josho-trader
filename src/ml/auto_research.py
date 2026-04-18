"""
auto_research.py — Karpathy AutoResearch Pattern for Trading
=============================================================
AI agent loops through strategy parameter experiments.
Keeps only improvements. Runs overnight. Wakes up smarter.

Inspired by: github.com/karpathy/autoresearch (21K+ stars)

What it does:
  1. Takes a trading strategy (e.g., exit thresholds, entry signals)
  2. Generates N parameter variations using LLM
  3. Backtests each variation on historical data
  4. Keeps only improvements (higher Sharpe, better win rate)
  5. Loops until converged or max iterations

Usage:
    python -m src.ml.auto_research --stock COALINDIA --iterations 100
    python -m src.ml.auto_research --stock COALINDIA --iterations 500 --overnight
"""

import json
import time
import random
import logging
import pickle
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger("auto_research")
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = DATA_DIR / "auto_research"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Default Strategy Parameters (the "genome" we evolve) ──

DEFAULT_PARAMS = {
    # Entry signals
    "rsi_oversold": 30,          # RSI below this = buy signal
    "rsi_overbought": 70,        # RSI above this = sell signal
    "sma_fast": 5,               # fast moving average period
    "sma_slow": 20,              # slow moving average period
    "volume_spike_threshold": 1.5,  # volume must be X times average
    "min_change_pct": -1.5,      # minimum drop for oversold bounce

    # Exit thresholds
    "target_pct": 25,            # take profit at X%
    "stop_pct": -40,             # stop loss at X%
    "trail_start_pct": 15,       # start trailing after X% profit
    "trail_pct": 10,             # trail by X% from peak

    # Position sizing
    "max_risk_pct": 15,          # max % of capital per trade
    "min_profit_to_fee_ratio": 2,  # expected profit must be > Nx fees
}


def load_stock_data(stock: str) -> pd.DataFrame:
    """Load historical data for backtesting."""
    # Try mega features first
    mega = DATA_DIR / f"{stock}_mega_features.csv"
    if mega.exists():
        return pd.read_csv(mega, index_col=0, parse_dates=True)

    # Fallback to raw CSV
    raw = DATA_DIR / "historical" / "daily_5y" / f"{stock.lower()}_daily_5y.csv"
    if raw.exists():
        df = pd.read_csv(raw, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        return df

    return pd.DataFrame()


def backtest_strategy(df: pd.DataFrame, params: dict) -> dict:
    """
    Backtest a strategy with given parameters on historical data.
    Returns performance metrics.
    """
    if df.empty or len(df) < 100:
        return {"sharpe": -999, "win_rate": 0, "total_return": 0}

    close = df["close"].values if "close" in df.columns else df["Close"].values
    n = len(close)

    trades = []
    position = None
    capital = 10000
    peak_capital = capital

    # Calculate indicators
    returns = np.diff(close) / close[:-1]
    sma_fast = pd.Series(close).rolling(params["sma_fast"]).mean().values
    sma_slow = pd.Series(close).rolling(params["sma_slow"]).mean().values

    # RSI
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    period = 14
    avg_gain = pd.Series(gain).rolling(period).mean().values
    avg_loss = pd.Series(loss).rolling(period).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    for i in range(max(params["sma_slow"], period) + 1, n):
        price = close[i]
        change_pct = ((price - close[i-1]) / close[i-1]) * 100

        if position is None:
            # ── ENTRY SIGNALS ──
            entry = False

            # Oversold bounce
            if i < len(rsi) and rsi[i-1] < params["rsi_oversold"] and change_pct < params["min_change_pct"]:
                entry = True

            # MA crossover
            if sma_fast[i] > sma_slow[i] and sma_fast[i-1] <= sma_slow[i-1]:
                entry = True

            if entry:
                qty = int((capital * params["max_risk_pct"] / 100) / price)
                if qty > 0:
                    position = {"entry": price, "qty": qty, "peak": price, "idx": i}

        else:
            # ── EXIT SIGNALS ──
            pnl_pct = ((price - position["entry"]) / position["entry"]) * 100

            if price > position["peak"]:
                position["peak"] = price

            exit_trade = False
            reason = ""

            # Target hit
            if pnl_pct >= params["target_pct"]:
                exit_trade = True
                reason = "target"

            # Stop loss
            elif pnl_pct <= params["stop_pct"]:
                exit_trade = True
                reason = "stop"

            # Trailing stop
            elif pnl_pct >= params["trail_start_pct"]:
                trail_price = position["peak"] * (1 - params["trail_pct"] / 100)
                if price <= trail_price:
                    exit_trade = True
                    reason = "trail"

            if exit_trade:
                pnl = (price - position["entry"]) * position["qty"]
                trades.append({
                    "entry": position["entry"],
                    "exit": price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "reason": reason,
                    "days_held": i - position["idx"],
                })
                capital += pnl
                if capital > peak_capital:
                    peak_capital = capital
                position = None

    # Calculate metrics
    if not trades:
        return {"sharpe": -999, "win_rate": 0, "total_return": 0, "trades": 0}

    returns_list = [t["pnl_pct"] / 100 for t in trades]
    wins = sum(1 for t in trades if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in trades)

    avg_return = np.mean(returns_list)
    std_return = np.std(returns_list) if len(returns_list) > 1 else 1
    sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0

    max_drawdown = 0
    running_capital = 10000
    peak = 10000
    for t in trades:
        running_capital += t["pnl"]
        if running_capital > peak:
            peak = running_capital
        dd = (peak - running_capital) / peak
        if dd > max_drawdown:
            max_drawdown = dd

    return {
        "sharpe": round(sharpe, 4),
        "win_rate": round(wins / len(trades) * 100, 1),
        "total_return": round(total_pnl, 2),
        "trades": len(trades),
        "avg_pnl_pct": round(np.mean([t["pnl_pct"] for t in trades]), 2),
        "max_drawdown": round(max_drawdown * 100, 1),
        "profit_factor": round(
            sum(t["pnl"] for t in trades if t["pnl"] > 0) /
            max(abs(sum(t["pnl"] for t in trades if t["pnl"] < 0)), 1), 2
        ),
    }


def mutate_params(params: dict, mutation_rate: float = 0.3) -> dict:
    """Create a mutated copy of parameters."""
    new_params = deepcopy(params)
    keys = list(new_params.keys())

    # Mutate 2-4 parameters
    n_mutate = random.randint(2, min(4, len(keys)))
    to_mutate = random.sample(keys, n_mutate)

    for key in to_mutate:
        value = new_params[key]
        if isinstance(value, int):
            delta = random.randint(-max(1, abs(value) // 5), max(1, abs(value) // 5))
            new_params[key] = max(1, value + delta)
        elif isinstance(value, float):
            delta = random.uniform(-abs(value) * 0.2, abs(value) * 0.2)
            new_params[key] = round(value + delta, 2)

    # Enforce constraints
    new_params["sma_fast"] = max(2, min(new_params["sma_fast"], new_params["sma_slow"] - 1))
    new_params["target_pct"] = max(5, new_params["target_pct"])
    new_params["stop_pct"] = min(-5, new_params["stop_pct"])
    new_params["trail_pct"] = max(2, min(new_params["trail_pct"], 30))
    new_params["max_risk_pct"] = max(5, min(new_params["max_risk_pct"], 50))

    return new_params


def run_auto_research(
    stock: str = "COALINDIA",
    iterations: int = 100,
    population_size: int = 5,
) -> dict:
    """
    AutoResearch loop: mutate → backtest → keep improvements → repeat.
    Returns the best parameters found.
    """
    log.info(f"AutoResearch: {stock} | {iterations} iterations | pop={population_size}")

    df = load_stock_data(stock)
    if df.empty:
        log.error(f"No data for {stock}")
        return {}

    # Initialize with default params
    best_params = deepcopy(DEFAULT_PARAMS)
    best_score = backtest_strategy(df, best_params)
    best_sharpe = best_score["sharpe"]

    log.info(f"Baseline: Sharpe={best_sharpe:.4f} | WinRate={best_score['win_rate']}% | Trades={best_score['trades']}")

    improvements = 0
    history = [{"iteration": 0, "sharpe": best_sharpe, "params": deepcopy(best_params)}]

    for i in range(1, iterations + 1):
        # Generate mutations
        candidates = [mutate_params(best_params) for _ in range(population_size)]

        for candidate in candidates:
            score = backtest_strategy(df, candidate)

            if score["sharpe"] > best_sharpe and score["trades"] >= 5:
                improvement = score["sharpe"] - best_sharpe
                best_sharpe = score["sharpe"]
                best_params = deepcopy(candidate)
                best_score = score
                improvements += 1
                history.append({"iteration": i, "sharpe": best_sharpe, "params": deepcopy(best_params)})

                log.info(
                    f"[{i}/{iterations}] IMPROVED +{improvement:.4f} → "
                    f"Sharpe={best_sharpe:.4f} | WinRate={best_score['win_rate']}% | "
                    f"Return={best_score['total_return']:.0f} | Trades={best_score['trades']}"
                )

        if i % 50 == 0:
            log.info(f"[{i}/{iterations}] Progress: {improvements} improvements found so far")

    # Save results
    result = {
        "stock": stock,
        "iterations": iterations,
        "improvements": improvements,
        "best_sharpe": best_sharpe,
        "best_score": best_score,
        "best_params": best_params,
        "baseline_sharpe": history[0]["sharpe"],
        "improvement_pct": round((best_sharpe - history[0]["sharpe"]) / max(abs(history[0]["sharpe"]), 0.01) * 100, 1),
        "history": history,
        "timestamp": datetime.now().isoformat(),
    }

    out_file = RESULTS_DIR / f"{stock}_auto_research.json"
    out_file.write_text(json.dumps(result, indent=2, default=str))
    log.info(f"\nResults saved to {out_file}")
    log.info(f"Final: Sharpe={best_sharpe:.4f} ({result['improvement_pct']:+.1f}% vs baseline)")
    log.info(f"Best params: {json.dumps(best_params, indent=2)}")

    return result


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", default="COALINDIA")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--population", type=int, default=5)
    parser.add_argument("--overnight", action="store_true", help="Run 500 iterations")
    args = parser.parse_args()

    iters = 500 if args.overnight else args.iterations
    run_auto_research(stock=args.stock, iterations=iters, population_size=args.population)

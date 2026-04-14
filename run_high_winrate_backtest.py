"""
High Win Rate Strategy Backtest Runner
=======================================
Backtests all 5 documented high-win-rate strategies against Indian stock data.
Prints a comparison table with key metrics for each strategy x stock combination.

Usage:
    python run_high_winrate_backtest.py
"""

import sys
import os
import copy
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.strategies.high_winrate import (
    RSI2MeanReversion,
    ConnorsRSI,
    TripleConfirmationBounce,
    LargeDropMeanReversion,
    IronCondorSimulated,
)
from src.backtest.engine import BacktestEngine, BacktestConfig


# ── Data Loading ─────────────────────────────────────────────────


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV CSV with datetime index."""
    df = pd.read_csv(path, parse_dates=["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    # Ensure lowercase columns
    df.columns = [c.lower().strip() for c in df.columns]
    return df


# ── Strategy Factory ─────────────────────────────────────────────


def make_strategies(symbol: str) -> list:
    """Create fresh instances of all 5 strategies for a given symbol."""
    return [
        RSI2MeanReversion(symbol=symbol, enhanced=False),
        RSI2MeanReversion(symbol=symbol, enhanced=True),
        ConnorsRSI(symbol=symbol),
        TripleConfirmationBounce(symbol=symbol),
        LargeDropMeanReversion(symbol=symbol),
        IronCondorSimulated(symbol=symbol),
    ]


# ── Backtest Config ──────────────────────────────────────────────

CONFIG = BacktestConfig(
    initial_capital=100_000.0,    # Rs 1 lakh
    commission_pct=0.03,          # Zerodha-like
    slippage_pct=0.05,
    position_size_pct=20.0,       # 20% per trade to avoid compounding distortion
    max_positions=1,              # one position at a time
    risk_free_rate=6.5,
)


# ── Main ─────────────────────────────────────────────────────────


def run_all():
    data_dir = Path(__file__).resolve().parent / "data" / "historical"

    # Stocks to test
    stock_files = {
        "ICICIBANK": data_dir / "icicibank_5yr.csv",
        "SBIN": data_dir / "sbin_5yr.csv",
        "RELIANCE": data_dir / "reliance_5yr.csv",
        "HDFCBANK": data_dir / "hdfcbank_5yr.csv",
        "INFY": data_dir / "infy_5yr.csv",
        "TCS": data_dir / "tcs_5yr.csv",
        "ITC": data_dir / "itc_5yr.csv",
        "BAJFINANCE": data_dir / "bajfinance_5yr.csv",
    }

    # Filter to files that actually exist
    available = {}
    for name, path in stock_files.items():
        if path.exists():
            available[name] = path
        else:
            print(f"[SKIP] {name}: {path} not found")

    if not available:
        print("ERROR: No stock data files found in", data_dir)
        return

    print(f"\n{'='*100}")
    print(f"HIGH WIN RATE STRATEGY BACKTEST — {len(available)} stocks")
    print(f"Capital: Rs {CONFIG.initial_capital:,.0f} | Commission: {CONFIG.commission_pct}% | Slippage: {CONFIG.slippage_pct}%")
    print(f"{'='*100}\n")

    # Collect all results for the summary table
    all_results = []

    strategy_names = [
        "RSI2_MeanReversion",
        "RSI2_Enhanced",
        "ConnorsRSI",
        "TripleConfirmBounce",
        "LargeDrop_MeanReversion",
        "IronCondor_Simulated",
    ]

    # Aggregate results per strategy (across all stocks)
    strategy_agg = {name: {"wins": 0, "total": 0, "returns": [], "pf": [], "dd": [], "hold_days": []}
                    for name in strategy_names}

    for stock_name, csv_path in available.items():
        print(f"\n--- {stock_name} ---")
        df = load_csv(str(csv_path))
        print(f"    Data: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} bars)")

        strategies = make_strategies(stock_name)

        for strat in strategies:
            # Reset strategy state
            strat._in_position = False

            # Precompute ConnorsRSI indicators on full dataset
            if hasattr(strat, "precompute"):
                strat.precompute(df)

            try:
                engine = BacktestEngine(strat, df, CONFIG)
                result = engine.run()
            except Exception as e:
                print(f"    {strat.name}: ERROR — {e}")
                continue

            # Compute avg holding period from trades
            hold_days = []
            for t in result.trades:
                try:
                    entry = pd.Timestamp(t["entry_time"])
                    exit_ = pd.Timestamp(t["exit_time"])
                    hold_days.append((exit_ - entry).days)
                except Exception:
                    pass
            avg_hold = np.mean(hold_days) if hold_days else 0

            # Determine display name
            display_name = strat.name
            if strat.name == "RSI2_MeanReversion" and getattr(strat, "enhanced", False):
                display_name = "RSI2_Enhanced"

            row = {
                "Stock": stock_name,
                "Strategy": display_name,
                "Trades": result.total_trades,
                "Win Rate %": result.win_rate_pct,
                "Return %": result.total_return_pct,
                "Profit Factor": result.profit_factor,
                "Max DD %": result.max_drawdown_pct,
                "Avg Win %": result.avg_win_pct,
                "Avg Loss %": result.avg_loss_pct,
                "Avg Hold Days": round(avg_hold, 1),
                "Sharpe": result.sharpe_ratio,
            }
            all_results.append(row)

            # Aggregate
            agg = strategy_agg[display_name]
            agg["wins"] += result.winning_trades
            agg["total"] += result.total_trades
            if result.total_return_pct != 0:
                agg["returns"].append(result.total_return_pct)
            if result.profit_factor != float("inf") and result.profit_factor > 0:
                agg["pf"].append(result.profit_factor)
            agg["dd"].append(result.max_drawdown_pct)
            if hold_days:
                agg["hold_days"].extend(hold_days)

            print(f"    {display_name:30s} | Trades: {result.total_trades:4d} | "
                  f"Win: {result.win_rate_pct:5.1f}% | Return: {result.total_return_pct:7.1f}% | "
                  f"PF: {result.profit_factor:5.2f} | MaxDD: {result.max_drawdown_pct:5.1f}% | "
                  f"Hold: {avg_hold:4.1f}d")

    # ── Summary Table ────────────────────────────────────────────

    print(f"\n\n{'='*100}")
    print("AGGREGATED RESULTS ACROSS ALL STOCKS")
    print(f"{'='*100}")
    print(f"\n{'Strategy':<30s} {'Total Trades':>12s} {'Win Rate %':>10s} {'Avg Return %':>12s} "
          f"{'Avg PF':>8s} {'Avg MaxDD %':>11s} {'Avg Hold':>9s} {'Documented':>12s}")
    print("-" * 110)

    documented_rates = {
        "RSI2_MeanReversion": "91%",
        "RSI2_Enhanced": "91%+",
        "ConnorsRSI": "75%",
        "TripleConfirmBounce": "70-80%",
        "LargeDrop_MeanReversion": "78%",
        "IronCondor_Simulated": "86%",
    }

    for name in strategy_names:
        agg = strategy_agg[name]
        total = agg["total"]
        wr = (agg["wins"] / total * 100) if total > 0 else 0
        avg_ret = np.mean(agg["returns"]) if agg["returns"] else 0
        avg_pf = np.mean(agg["pf"]) if agg["pf"] else 0
        avg_dd = np.mean(agg["dd"]) if agg["dd"] else 0
        avg_hold = np.mean(agg["hold_days"]) if agg["hold_days"] else 0
        doc = documented_rates.get(name, "N/A")

        print(f"{name:<30s} {total:>12d} {wr:>10.1f} {avg_ret:>12.1f} "
              f"{avg_pf:>8.2f} {avg_dd:>11.1f} {avg_hold:>9.1f}d {doc:>12s}")

    print("-" * 110)

    # ── Detailed per-stock table ─────────────────────────────────

    if all_results:
        print(f"\n\n{'='*100}")
        print("DETAILED RESULTS (per stock x strategy)")
        print(f"{'='*100}\n")

        results_df = pd.DataFrame(all_results)
        # Pivot: strategies as rows, stocks as columns, win rate as values
        if len(results_df) > 0:
            pivot = results_df.pivot_table(
                index="Strategy",
                columns="Stock",
                values="Win Rate %",
                aggfunc="first",
            )
            print("Win Rate % by Strategy x Stock:")
            print(pivot.to_string(float_format="%.1f"))

            print("\n\nReturn % by Strategy x Stock:")
            pivot_ret = results_df.pivot_table(
                index="Strategy",
                columns="Stock",
                values="Return %",
                aggfunc="first",
            )
            print(pivot_ret.to_string(float_format="%.1f"))

            print("\n\nTrades by Strategy x Stock:")
            pivot_trades = results_df.pivot_table(
                index="Strategy",
                columns="Stock",
                values="Trades",
                aggfunc="first",
            )
            print(pivot_trades.to_string(float_format="%.0f"))

    print(f"\n{'='*100}")
    print("Backtest complete.")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    run_all()

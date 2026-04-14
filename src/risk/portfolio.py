"""
Portfolio Optimization -- Size positions like the 0.001%.
Goes beyond single-trade risk to portfolio-level optimization.

Implements:
  1. Kelly Criterion -- optimal bet size from win rate + payoff
  2. Risk Parity -- equal risk contribution across positions
  3. Maximum Sharpe -- simplified Markowitz for best risk-adjusted weights
  4. Correlation-aware sizing -- reduce overlap between correlated positions
  5. Portfolio heat -- total risk budget across all positions
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("josho.risk.portfolio")


@dataclass
class Position:
    """A single position in the portfolio."""
    symbol: str
    current_value: float  # current notional value
    entry_price: float
    current_price: float
    quantity: int
    side: str  # LONG or SHORT
    daily_vol: float = 0.0  # daily volatility (std dev of returns)
    stop_loss_pct: float = 2.0  # stop loss as % of entry
    option_type: str = ""  # CE/PE if options position
    delta: float = 1.0  # 1.0 for equity, actual delta for options


@dataclass
class PortfolioState:
    """Current portfolio snapshot."""
    total_capital: float
    positions: list[Position] = field(default_factory=list)
    max_portfolio_heat_pct: float = 6.0  # max total risk as % of capital
    max_single_position_pct: float = 20.0  # max single position as % of capital
    max_correlated_group_pct: float = 35.0  # max correlated positions as % of capital


# -- Kelly Criterion ----------------------------------------------------------

def kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.5,
) -> dict:
    """
    Kelly Criterion -- mathematically optimal bet size.
    Full Kelly is too aggressive for trading, so we use fractional Kelly.

    Args:
        win_rate: Historical win rate (0.0 to 1.0)
        avg_win: Average winning trade amount (positive)
        avg_loss: Average losing trade amount (positive, we negate internally)
        fraction: Kelly fraction (0.25 = quarter, 0.5 = half Kelly)

    Returns:
        Optimal position size as fraction of capital.
    """
    if win_rate <= 0 or win_rate >= 1:
        return {
            "kelly_pct": 0,
            "recommended_pct": 0,
            "edge": 0,
            "interpretation": "Invalid win rate -- must be between 0 and 1",
        }

    if avg_win <= 0 or avg_loss <= 0:
        return {
            "kelly_pct": 0,
            "recommended_pct": 0,
            "edge": 0,
            "interpretation": "Invalid avg win/loss -- must be positive",
        }

    # Kelly formula: f* = (bp - q) / b
    # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
    b = avg_win / avg_loss  # payoff ratio
    p = win_rate
    q = 1 - p

    full_kelly = (b * p - q) / b
    edge = (p * avg_win) - (q * avg_loss)

    if full_kelly <= 0:
        return {
            "kelly_pct": round(full_kelly * 100, 2),
            "recommended_pct": 0,
            "edge": round(edge, 2),
            "payoff_ratio": round(b, 2),
            "interpretation": "Negative edge -- do NOT take this trade.",
        }

    recommended = full_kelly * fraction

    # Safety cap at 25% regardless
    recommended = min(recommended, 0.25)

    if full_kelly > 0.5:
        risk_note = "Very high Kelly -- likely overfitting. Use quarter Kelly."
    elif full_kelly > 0.25:
        risk_note = "Strong edge detected. Half Kelly recommended."
    else:
        risk_note = "Moderate edge. Half Kelly is appropriate."

    return {
        "kelly_pct": round(full_kelly * 100, 2),
        "recommended_pct": round(recommended * 100, 2),
        "fraction_used": fraction,
        "edge": round(edge, 2),
        "payoff_ratio": round(b, 2),
        "risk_note": risk_note,
        "interpretation": (
            f"Full Kelly = {full_kelly*100:.1f}% of capital. "
            f"Using {fraction}x Kelly = {recommended*100:.1f}%. "
            f"Edge per trade: {edge:.0f}"
        ),
    }


def kelly_position_size(
    capital: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.5,
) -> float:
    """
    Returns the recommended position size in absolute terms.
    Convenience function wrapping kelly_criterion.
    """
    result = kelly_criterion(win_rate, avg_win, avg_loss, fraction)
    pct = result["recommended_pct"] / 100
    return round(capital * pct, 2)


# -- Risk Parity --------------------------------------------------------------

def risk_parity_weights(
    daily_volatilities: list[float],
    symbols: list[str] = None,
) -> dict:
    """
    Risk Parity -- equal risk contribution from each position.
    Higher-vol assets get smaller weight. Lower-vol assets get more.

    This is how Bridgewater's All Weather fund works (simplified).

    Args:
        daily_volatilities: Daily standard deviation of returns for each asset
        symbols: Optional labels for readability

    Returns:
        Weights that equalize risk contribution.
    """
    if not daily_volatilities or any(v <= 0 for v in daily_volatilities):
        return {
            "weights": [],
            "interpretation": "Invalid volatilities -- all must be positive",
        }

    n = len(daily_volatilities)
    symbols = symbols or [f"Asset_{i+1}" for i in range(n)]

    vols = np.array(daily_volatilities)

    # Risk parity: weight inversely proportional to volatility
    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()

    # Risk contribution of each asset
    risk_contributions = weights * vols
    total_risk = risk_contributions.sum()
    risk_pct = (risk_contributions / total_risk * 100) if total_risk > 0 else np.zeros(n)

    return {
        "weights": {
            sym: {
                "weight_pct": round(float(w) * 100, 2),
                "daily_vol_pct": round(float(v) * 100, 2),
                "risk_contribution_pct": round(float(rp), 2),
            }
            for sym, w, v, rp in zip(symbols, weights, vols, risk_pct)
        },
        "total_portfolio_vol": round(float(total_risk) * 100, 2),
        "interpretation": (
            "Risk parity: each asset contributes roughly equal risk. "
            "High-vol assets get smaller allocation."
        ),
    }


def risk_parity_allocation(
    capital: float,
    daily_volatilities: list[float],
    symbols: list[str] = None,
) -> dict:
    """Returns capital allocation in absolute terms using risk parity."""
    result = risk_parity_weights(daily_volatilities, symbols)
    weights = result.get("weights", {})

    allocations = {}
    for sym, data in weights.items():
        alloc = capital * (data["weight_pct"] / 100)
        allocations[sym] = {
            **data,
            "allocation": round(alloc, 2),
        }

    return {
        "capital": capital,
        "allocations": allocations,
        "interpretation": result.get("interpretation", ""),
    }


# -- Maximum Sharpe (Simplified Markowitz) ------------------------------------

def max_sharpe_weights(
    expected_returns: list[float],
    daily_volatilities: list[float],
    correlation_matrix: list[list[float]] = None,
    risk_free_rate: float = 0.065 / 252,  # daily risk-free rate (India ~6.5% annual)
    symbols: list[str] = None,
) -> dict:
    """
    Simplified Maximum Sharpe Ratio portfolio (Markowitz).
    Finds weights that maximize return per unit risk.

    For a full-blown optimizer we'd need scipy.optimize, but this
    analytical approach works well for small portfolios (< 10 assets).

    Args:
        expected_returns: Expected daily returns for each asset
        daily_volatilities: Daily std dev of returns
        correlation_matrix: NxN correlation matrix (identity if None)
        risk_free_rate: Daily risk-free rate
        symbols: Optional labels
    """
    n = len(expected_returns)
    symbols = symbols or [f"Asset_{i+1}" for i in range(n)]

    if n != len(daily_volatilities):
        return {"error": "Mismatch between returns and volatilities count"}

    returns_arr = np.array(expected_returns)
    vols = np.array(daily_volatilities)

    # Build covariance matrix
    if correlation_matrix is not None:
        corr = np.array(correlation_matrix)
    else:
        corr = np.eye(n)

    # Covariance = diag(vol) @ corr @ diag(vol)
    cov_matrix = np.diag(vols) @ corr @ np.diag(vols)

    # Excess returns
    excess = returns_arr - risk_free_rate

    try:
        # Analytical solution for max Sharpe: w* = inv(Cov) @ excess_returns
        inv_cov = np.linalg.inv(cov_matrix)
        raw_weights = inv_cov @ excess

        # Normalize to sum to 1 (long-only: clip negatives)
        raw_weights = np.maximum(raw_weights, 0)  # no shorting
        total = raw_weights.sum()

        if total <= 0:
            # All negative expected excess returns
            weights = np.ones(n) / n  # equal weight fallback
            note = "All expected returns below risk-free rate. Using equal weight."
        else:
            weights = raw_weights / total
            note = "Weights maximize Sharpe ratio (long-only constraint applied)."

    except np.linalg.LinAlgError:
        weights = np.ones(n) / n
        note = "Singular covariance matrix. Using equal weight fallback."

    # Portfolio metrics
    port_return = float(weights @ returns_arr)
    port_vol = float(np.sqrt(weights @ cov_matrix @ weights))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0

    return {
        "weights": {
            sym: round(float(w) * 100, 2)
            for sym, w in zip(symbols, weights)
        },
        "portfolio_daily_return_pct": round(port_return * 100, 4),
        "portfolio_daily_vol_pct": round(port_vol * 100, 4),
        "sharpe_ratio_daily": round(sharpe, 4),
        "sharpe_ratio_annual": round(sharpe * math.sqrt(252), 4),
        "note": note,
    }


# -- Correlation-Aware Position Sizing ----------------------------------------

def correlation_adjusted_size(
    base_size: float,
    new_symbol: str,
    existing_positions: list[dict],
    correlation_lookup: dict[tuple[str, str], float] = None,
    max_correlation_penalty: float = 0.5,
) -> dict:
    """
    Reduce position size when adding a correlated position.
    If you're long NIFTY CE and add BANKNIFTY CE, they're ~0.85 correlated.
    Your effective risk is nearly doubled. This adjusts for that.

    Args:
        base_size: Original position size from Kelly or other method
        new_symbol: Symbol being added
        existing_positions: List of dicts with 'symbol' and 'value' keys
        correlation_lookup: Dict mapping (sym1, sym2) -> correlation
        max_correlation_penalty: Max reduction factor (0.5 = halve size at 100% corr)

    Returns:
        Adjusted position size and reasoning.
    """
    if not existing_positions or not correlation_lookup:
        return {
            "original_size": base_size,
            "adjusted_size": base_size,
            "adjustment_factor": 1.0,
            "reason": "No existing positions or no correlation data -- full size.",
        }

    # Default correlations for common Indian F&O pairs
    default_corr = {
        ("NIFTY", "BANKNIFTY"): 0.85,
        ("NIFTY", "FINNIFTY"): 0.80,
        ("BANKNIFTY", "FINNIFTY"): 0.75,
        ("RELIANCE", "NIFTY"): 0.70,
        ("HDFCBANK", "BANKNIFTY"): 0.80,
        ("ICICIBANK", "BANKNIFTY"): 0.75,
        ("TCS", "INFY"): 0.85,
        ("SBIN", "BANKNIFTY"): 0.70,
    }
    corr_data = {**default_corr, **(correlation_lookup or {})}

    # Find max correlation with existing positions
    max_corr = 0.0
    most_correlated = ""

    for pos in existing_positions:
        sym = pos.get("symbol", "")
        pair = tuple(sorted([new_symbol, sym]))
        corr = corr_data.get(pair, corr_data.get((pair[1], pair[0]), 0.0))
        if abs(corr) > abs(max_corr):
            max_corr = corr
            most_correlated = sym

    # Penalty: linear reduction based on max correlation
    # At corr=0 -> factor=1.0, at corr=1.0 -> factor=(1-max_penalty)
    penalty = abs(max_corr) * max_correlation_penalty
    adjustment_factor = max(1.0 - penalty, 0.25)  # never go below 25% of base

    adjusted_size = base_size * adjustment_factor

    if abs(max_corr) > 0.7:
        reason = (
            f"High correlation ({max_corr:.2f}) with {most_correlated}. "
            f"Size reduced by {penalty*100:.0f}% to avoid doubling risk."
        )
    elif abs(max_corr) > 0.4:
        reason = (
            f"Moderate correlation ({max_corr:.2f}) with {most_correlated}. "
            f"Size reduced by {penalty*100:.0f}%."
        )
    else:
        reason = f"Low correlation with existing positions. Near-full size allowed."

    return {
        "original_size": round(base_size, 2),
        "adjusted_size": round(adjusted_size, 2),
        "adjustment_factor": round(adjustment_factor, 4),
        "max_correlation": round(max_corr, 2),
        "most_correlated_with": most_correlated,
        "reason": reason,
    }


# -- Portfolio Heat -----------------------------------------------------------

def portfolio_heat(
    portfolio: PortfolioState,
) -> dict:
    """
    Portfolio Heat = total risk across ALL open positions.
    Each position's risk = distance to stop loss * quantity * delta.
    Total heat must stay below max_portfolio_heat_pct of capital.

    If heat > limit, the oldest/weakest position should be trimmed.
    """
    if not portfolio.positions:
        return {
            "total_heat_pct": 0,
            "max_heat_pct": portfolio.max_portfolio_heat_pct,
            "remaining_budget_pct": portfolio.max_portfolio_heat_pct,
            "can_add_position": True,
            "positions": [],
            "interpretation": "No open positions. Full risk budget available.",
        }

    position_heats = []
    total_heat = 0.0

    for pos in portfolio.positions:
        # Risk per position = stop_loss_pct * notional * abs(delta)
        notional = abs(pos.current_price * pos.quantity)
        risk_amount = notional * (pos.stop_loss_pct / 100) * abs(pos.delta)
        heat_pct = (risk_amount / portfolio.total_capital * 100) if portfolio.total_capital > 0 else 0

        total_heat += heat_pct

        position_heats.append({
            "symbol": pos.symbol,
            "side": pos.side,
            "notional": round(notional, 2),
            "risk_amount": round(risk_amount, 2),
            "heat_pct": round(heat_pct, 2),
            "stop_loss_pct": pos.stop_loss_pct,
            "delta": pos.delta,
        })

    remaining = portfolio.max_portfolio_heat_pct - total_heat
    can_add = remaining > 0.5  # need at least 0.5% risk budget for new position

    if total_heat > portfolio.max_portfolio_heat_pct:
        status = "OVER_LIMIT"
        interpretation = (
            f"Portfolio heat {total_heat:.1f}% EXCEEDS limit {portfolio.max_portfolio_heat_pct}%. "
            "MUST reduce positions immediately."
        )
    elif total_heat > portfolio.max_portfolio_heat_pct * 0.8:
        status = "WARNING"
        interpretation = (
            f"Portfolio heat {total_heat:.1f}% approaching limit. "
            f"Only {remaining:.1f}% risk budget remaining."
        )
    else:
        status = "OK"
        interpretation = (
            f"Portfolio heat {total_heat:.1f}% within limits. "
            f"{remaining:.1f}% risk budget available."
        )

    # Sort by heat contribution (highest first)
    position_heats.sort(key=lambda x: x["heat_pct"], reverse=True)

    return {
        "total_heat_pct": round(total_heat, 2),
        "max_heat_pct": portfolio.max_portfolio_heat_pct,
        "remaining_budget_pct": round(remaining, 2),
        "status": status,
        "can_add_position": can_add,
        "positions": position_heats,
        "interpretation": interpretation,
    }


def max_position_size_from_heat(
    portfolio: PortfolioState,
    stop_loss_pct: float = 2.0,
    delta: float = 1.0,
) -> float:
    """
    Given current portfolio heat, calculate max size for next position.
    Ensures total heat stays within budget.
    """
    heat = portfolio_heat(portfolio)
    remaining_pct = heat["remaining_budget_pct"]

    if remaining_pct <= 0:
        return 0.0

    # remaining_risk = capital * remaining_pct / 100
    # position_risk = size * stop_loss_pct / 100 * delta
    # size = remaining_risk / (stop_loss_pct / 100 * delta)
    effective_stop = (stop_loss_pct / 100) * abs(delta)
    if effective_stop <= 0:
        return 0.0

    remaining_risk = portfolio.total_capital * (remaining_pct / 100)
    max_size = remaining_risk / effective_stop

    # Also cap at single position limit
    single_cap = portfolio.total_capital * (portfolio.max_single_position_pct / 100)
    return round(min(max_size, single_cap), 2)


# -- Integrated Sizing --------------------------------------------------------

def optimal_position_size(
    capital: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    daily_vol: float,
    existing_positions: list[dict] = None,
    new_symbol: str = "",
    correlation_lookup: dict = None,
    current_heat_pct: float = 0.0,
    max_heat_pct: float = 6.0,
    stop_loss_pct: float = 2.0,
    delta: float = 1.0,
    kelly_fraction: float = 0.5,
) -> dict:
    """
    Master position sizing that combines all methods.
    Takes the MINIMUM of all sizing methods (most conservative wins).

    This is the function the trading engine should call.
    """
    # 1. Kelly Criterion
    kelly = kelly_criterion(win_rate, avg_win, avg_loss, kelly_fraction)
    kelly_size = capital * (kelly["recommended_pct"] / 100) if kelly["recommended_pct"] > 0 else 0

    # 2. Volatility-based sizing (risk parity concept for single position)
    # Target: risk no more than 1% of capital per day from this position
    target_daily_risk_pct = 1.0
    if daily_vol > 0:
        vol_size = capital * (target_daily_risk_pct / 100) / daily_vol
    else:
        vol_size = capital * 0.1  # 10% fallback

    # 3. Heat budget remaining
    remaining_heat = max_heat_pct - current_heat_pct
    if remaining_heat > 0 and stop_loss_pct > 0:
        heat_size = capital * (remaining_heat / 100) / (stop_loss_pct / 100 * abs(delta))
    else:
        heat_size = 0

    # 4. Correlation adjustment
    if existing_positions and new_symbol:
        corr_result = correlation_adjusted_size(
            base_size=min(kelly_size, vol_size),
            new_symbol=new_symbol,
            existing_positions=existing_positions,
            correlation_lookup=correlation_lookup,
        )
        corr_size = corr_result["adjusted_size"]
    else:
        corr_size = min(kelly_size, vol_size) if kelly_size > 0 else vol_size
        corr_result = {"adjustment_factor": 1.0, "reason": "No correlation data"}

    # Take the minimum (most conservative)
    candidates = {
        "kelly": kelly_size,
        "volatility": vol_size,
        "heat_budget": heat_size,
        "correlation_adjusted": corr_size,
    }

    # Filter out zero/negative
    valid = {k: v for k, v in candidates.items() if v > 0}
    if not valid:
        return {
            "recommended_size": 0,
            "method": "NONE",
            "reason": "All sizing methods returned zero -- do not trade.",
            "breakdown": candidates,
            "kelly": kelly,
        }

    min_method = min(valid, key=valid.get)
    recommended = valid[min_method]

    # Hard cap: never more than 20% of capital
    hard_cap = capital * 0.20
    if recommended > hard_cap:
        recommended = hard_cap
        min_method = "hard_cap"

    return {
        "recommended_size": round(recommended, 2),
        "recommended_pct": round((recommended / capital) * 100, 2) if capital > 0 else 0,
        "binding_constraint": min_method,
        "breakdown": {k: round(v, 2) for k, v in candidates.items()},
        "kelly": kelly,
        "correlation": corr_result,
        "interpretation": (
            f"Size: {recommended:,.0f} ({recommended/capital*100:.1f}% of capital). "
            f"Binding constraint: {min_method}."
        ),
    }

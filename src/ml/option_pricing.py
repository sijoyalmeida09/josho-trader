"""
Option Pricing Engine — Black-Scholes + Greeks + IV calculation.
Integrated from: vollib patterns, gs-quant concepts.

This is CRITICAL for F&O — every trade decision needs:
  - Fair value (is the option over/underpriced?)
  - Greeks (delta, gamma, theta, vega, rho)
  - Implied Volatility (IV) and IV percentile
  - Probability of profit
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional
from scipy.stats import norm

log = logging.getLogger("josho.ml.pricing")


@dataclass
class OptionPrice:
    """Complete option pricing result."""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    iv: float
    intrinsic: float
    time_value: float
    probability_itm: float
    probability_profit: float


def black_scholes(
    spot: float,
    strike: float,
    time_to_expiry: float,  # in years (e.g., 7 days = 7/365)
    risk_free_rate: float = 0.065,  # India ~6.5%
    volatility: float = 0.15,  # annualized
    option_type: str = "CE",  # CE or PE
) -> OptionPrice:
    """
    Black-Scholes option pricing with full Greeks.
    The foundation of every F&O decision.
    """
    if time_to_expiry <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
        return OptionPrice(0, 0, 0, 0, 0, 0, volatility, 0, 0, 0, 0)

    S, K, T, r, sigma = spot, strike, time_to_expiry, risk_free_rate, volatility

    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "CE":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        prob_itm = norm.cdf(d2)
    else:  # PE
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        prob_itm = norm.cdf(-d2)

    # Greeks
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    theta_daily = -(S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))) / 365
    if option_type == "CE":
        theta_daily -= (r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta_daily += (r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365

    vega = S * norm.pdf(d1) * math.sqrt(T) / 100  # per 1% vol change
    rho = K * T * math.exp(-r * T) * (norm.cdf(d2) if option_type == "CE" else -norm.cdf(-d2)) / 100

    # Intrinsic + time value
    intrinsic = max(0, S - K) if option_type == "CE" else max(0, K - S)
    time_value = price - intrinsic

    # Probability of profit (for selling: price stays below premium)
    prob_profit = 1 - prob_itm if option_type == "CE" else prob_itm

    return OptionPrice(
        price=round(price, 2),
        delta=round(delta, 4),
        gamma=round(gamma, 6),
        theta=round(theta_daily, 4),
        vega=round(vega, 4),
        rho=round(rho, 4),
        iv=round(volatility * 100, 2),
        intrinsic=round(intrinsic, 2),
        time_value=round(time_value, 2),
        probability_itm=round(prob_itm * 100, 2),
        probability_profit=round(prob_profit * 100, 2),
    )


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float = 0.065,
    option_type: str = "CE",
    max_iterations: int = 100,
    tolerance: float = 0.0001,
) -> float:
    """
    Calculate Implied Volatility using Newton-Raphson method.
    IV = what the market thinks volatility will be.
    """
    if market_price <= 0 or time_to_expiry <= 0:
        return 0.0

    sigma = 0.2  # initial guess

    for _ in range(max_iterations):
        bs = black_scholes(spot, strike, time_to_expiry, risk_free_rate, sigma, option_type)
        diff = bs.price - market_price

        if abs(diff) < tolerance:
            return sigma

        # Vega for Newton-Raphson step
        d1 = (math.log(spot / strike) + (risk_free_rate + sigma ** 2 / 2) * time_to_expiry) / (sigma * math.sqrt(time_to_expiry))
        vega_raw = spot * norm.pdf(d1) * math.sqrt(time_to_expiry)

        if vega_raw < 0.001:
            break

        sigma -= diff / vega_raw
        sigma = max(0.001, min(5.0, sigma))  # bound between 0.1% and 500%

    return sigma


def analyze_option(
    spot: float,
    strike: float,
    market_premium: float,
    days_to_expiry: int,
    option_type: str = "CE",
    risk_free_rate: float = 0.065,
) -> dict:
    """
    Full option analysis — pricing, IV, Greeks, edge detection.
    This is what the 0.001% use to find mispriced options.
    """
    T = days_to_expiry / 365

    # Calculate IV from market price
    iv = implied_volatility(market_premium, spot, strike, T, risk_free_rate, option_type)

    # Price with calculated IV
    pricing = black_scholes(spot, strike, T, risk_free_rate, iv, option_type)

    # Compare theoretical vs market (edge detection)
    theoretical_price = pricing.price
    edge = market_premium - theoretical_price
    edge_pct = (edge / market_premium * 100) if market_premium > 0 else 0

    # Breakeven
    if option_type == "CE":
        breakeven = strike + market_premium
    else:
        breakeven = strike - market_premium

    breakeven_pct = ((breakeven - spot) / spot) * 100

    # Max profit/loss for buyer
    max_loss_buyer = market_premium
    max_profit_buyer = float("inf") if option_type == "CE" else strike - market_premium

    # Theta decay per day (how much seller earns daily)
    daily_theta_income = abs(pricing.theta)

    return {
        "spot": spot,
        "strike": strike,
        "option_type": option_type,
        "market_premium": market_premium,
        "theoretical_price": theoretical_price,
        "edge": round(edge, 2),
        "edge_pct": round(edge_pct, 2),
        "mispriced": "OVERPRICED" if edge > 1 else "UNDERPRICED" if edge < -1 else "FAIR",
        "iv": round(iv * 100, 2),
        "days_to_expiry": days_to_expiry,
        "breakeven": round(breakeven, 2),
        "breakeven_distance_pct": round(breakeven_pct, 2),
        "greeks": {
            "delta": pricing.delta,
            "gamma": pricing.gamma,
            "theta": pricing.theta,
            "vega": pricing.vega,
            "rho": pricing.rho,
        },
        "probability_itm": pricing.probability_itm,
        "probability_profit_buyer": pricing.probability_itm,
        "probability_profit_seller": 100 - pricing.probability_itm,
        "daily_theta_income": round(daily_theta_income, 2),
        "intrinsic": pricing.intrinsic,
        "time_value": pricing.time_value,
    }


def scan_mispriced_options(option_chain: list[dict], spot: float, days_to_expiry: int) -> list[dict]:
    """
    Scan an entire option chain for mispriced options.
    The edge that institutional traders exploit.
    """
    mispriced = []

    for option in option_chain:
        for opt_type, premium_key, iv_key in [("CE", "ce_ltp", "ce_iv"), ("PE", "pe_ltp", "pe_iv")]:
            premium = option.get(premium_key, 0)
            if premium <= 0:
                continue

            strike = option.get("strike", 0)
            analysis = analyze_option(spot, strike, premium, days_to_expiry, opt_type)

            if abs(analysis["edge_pct"]) > 5:  # more than 5% mispricing
                analysis["symbol_hint"] = f"{strike}{opt_type}"
                mispriced.append(analysis)

    mispriced.sort(key=lambda x: abs(x["edge_pct"]), reverse=True)
    return mispriced[:10]  # top 10 mispriced

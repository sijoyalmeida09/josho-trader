"""
IV Surface Modeling — Implied Volatility surface across strikes and expiries.

Builds a volatility surface from NSE option chain data to identify:
  - IV smile/skew across strikes for a given expiry
  - IV term structure across expiries for a given strike
  - Moneyness scoring (which strikes are cheap vs expensive)
  - Unusual IV patterns (where smart money is hedging)

Uses the existing NseData class from src/market_data/nse_data.py for data.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .nse_data import NseData

log = logging.getLogger("josho.iv_surface")

# Risk-free rate for India (approximate RBI repo rate)
RISK_FREE_RATE = 0.065


@dataclass(frozen=True)
class IVPoint:
    """Single point on the IV surface."""
    strike: float
    expiry: str
    option_type: str  # CE or PE
    iv: float
    ltp: float
    oi: int
    volume: int
    moneyness: float  # S/K for calls, K/S for puts


@dataclass(frozen=True)
class SmileSlice:
    """IV smile for a single expiry."""
    expiry: str
    spot: float
    atm_iv: float
    skew_25d: float  # 25-delta skew (put IV - call IV)
    smile_curvature: float  # how convex the smile is
    points: tuple  # tuple of IVPoint


@dataclass(frozen=True)
class MoneynessScore:
    """How cheap or expensive a strike is relative to the surface."""
    strike: float
    expiry: str
    option_type: str
    market_iv: float
    fair_iv: float  # interpolated from surface
    iv_deviation: float  # market_iv - fair_iv (positive = expensive)
    percentile: float  # where this IV sits in the distribution
    signal: str  # CHEAP, FAIR, EXPENSIVE


def _calc_iv_newton(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    option_type: str = "CE",
    max_iter: int = 50,
    tol: float = 0.0001,
) -> float:
    """
    Calculate IV using Newton-Raphson.
    Returns annualized IV as a decimal (e.g., 0.15 = 15%).
    """
    if market_price <= 0 or time_to_expiry <= 0 or spot <= 0 or strike <= 0:
        return 0.0

    # Intrinsic value check
    intrinsic = max(0, spot - strike) if option_type == "CE" else max(0, strike - spot)
    if market_price < intrinsic:
        return 0.0

    sigma = 0.20  # initial guess
    r = RISK_FREE_RATE

    for _ in range(max_iter):
        sqrt_t = math.sqrt(time_to_expiry)
        d1 = (math.log(spot / strike) + (r + sigma ** 2 / 2) * time_to_expiry) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        if option_type == "CE":
            price = spot * norm.cdf(d1) - strike * math.exp(-r * time_to_expiry) * norm.cdf(d2)
        else:
            price = strike * math.exp(-r * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)

        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        vega = spot * norm.pdf(d1) * sqrt_t
        if vega < 0.001:
            break

        sigma -= diff / vega
        sigma = max(0.001, min(5.0, sigma))

    return sigma


def _parse_expiry_date(expiry_str: str) -> Optional[datetime]:
    """Parse NSE expiry date string to datetime."""
    formats = ["%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d", "%d/%m/%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(expiry_str, fmt)
        except ValueError:
            continue
    return None


def _days_to_expiry(expiry_str: str) -> float:
    """Calculate trading days to expiry."""
    expiry_dt = _parse_expiry_date(expiry_str)
    if not expiry_dt:
        return 7.0  # default fallback

    delta = (expiry_dt - datetime.now()).days
    return max(delta, 0.5)  # at least half a day


class IVSurface:
    """
    Build and analyze the implied volatility surface from NSE option chains.

    The IV surface reveals where the market is pricing risk. Deviations from
    the smooth surface indicate mispricing or smart money activity.
    """

    def __init__(self, nse: Optional[NseData] = None):
        self.nse = nse or NseData()
        self._surface_data: list[IVPoint] = []
        self._spot: float = 0.0
        self._symbol: str = ""
        self._built_at: Optional[datetime] = None

    def build(self, symbol: str = "NIFTY") -> dict:
        """
        Build the full IV surface from the option chain.

        Args:
            symbol: Index or equity symbol (NIFTY, BANKNIFTY, etc.)

        Returns:
            Summary dict with surface stats.
        """
        chain = self.nse.get_option_chain(symbol)
        if not chain or not chain.get("chains"):
            log.warning(f"No option chain data for {symbol}")
            return {"error": "no_data", "symbol": symbol}

        self._spot = chain["underlying"]
        self._symbol = symbol
        self._built_at = datetime.now()

        points = []
        for row in chain["chains"]:
            strike = row["strike"]
            expiry = row["expiry"]
            dte = _days_to_expiry(expiry)
            t = dte / 365.0

            for opt_type, ltp_key, iv_key, oi_key, vol_key in [
                ("CE", "ce_ltp", "ce_iv", "ce_oi", "ce_volume"),
                ("PE", "pe_ltp", "pe_iv", "pe_oi", "pe_volume"),
            ]:
                ltp = row.get(ltp_key, 0)
                nse_iv = row.get(iv_key, 0)
                oi = row.get(oi_key, 0)
                volume = row.get(vol_key, 0)

                if ltp <= 0:
                    continue

                # Calculate IV ourselves for consistency
                calc_iv = _calc_iv_newton(ltp, self._spot, strike, t, opt_type)
                # Prefer NSE-reported IV if available, else use calculated
                iv = (nse_iv / 100.0) if nse_iv > 0 else calc_iv

                if iv <= 0:
                    continue

                moneyness = self._spot / strike if opt_type == "CE" else strike / self._spot

                points.append(IVPoint(
                    strike=strike,
                    expiry=expiry,
                    option_type=opt_type,
                    iv=round(iv, 4),
                    ltp=ltp,
                    oi=oi,
                    volume=volume,
                    moneyness=round(moneyness, 4),
                ))

        self._surface_data = points

        expiries = sorted(set(p.expiry for p in points))
        strikes = sorted(set(p.strike for p in points))

        log.info(
            f"IV Surface built: {symbol}, {len(points)} points, "
            f"{len(expiries)} expiries, {len(strikes)} strikes"
        )

        return {
            "symbol": symbol,
            "spot": self._spot,
            "points": len(points),
            "expiries": len(expiries),
            "strikes": len(strikes),
            "built_at": self._built_at.isoformat(),
        }

    def get_smile(self, expiry: str = "") -> Optional[SmileSlice]:
        """
        Extract IV smile for a given expiry.
        If no expiry specified, uses the nearest one.

        Returns:
            SmileSlice with ATM IV, skew, curvature, and all data points.
        """
        if not self._surface_data:
            log.warning("Surface not built. Call build() first.")
            return None

        expiries = sorted(set(p.expiry for p in self._surface_data))
        if not expiries:
            return None

        if not expiry:
            expiry = expiries[0]
        elif expiry not in expiries:
            log.warning(f"Expiry {expiry} not found. Available: {expiries}")
            return None

        points = [p for p in self._surface_data if p.expiry == expiry]
        if not points:
            return None

        # ATM IV: average of CE and PE IV at strike closest to spot
        strikes = sorted(set(p.strike for p in points))
        atm_strike = min(strikes, key=lambda s: abs(s - self._spot))

        atm_ce = [p for p in points if p.strike == atm_strike and p.option_type == "CE"]
        atm_pe = [p for p in points if p.strike == atm_strike and p.option_type == "PE"]
        atm_iv = 0.0
        if atm_ce and atm_pe:
            atm_iv = (atm_ce[0].iv + atm_pe[0].iv) / 2
        elif atm_ce:
            atm_iv = atm_ce[0].iv
        elif atm_pe:
            atm_iv = atm_pe[0].iv

        # 25-delta skew: OTM put IV vs OTM call IV
        # Approximate 25-delta as ~5% OTM
        otm_put_strike = self._spot * 0.95
        otm_call_strike = self._spot * 1.05

        otm_put_iv = self._interpolate_iv(points, otm_put_strike, "PE")
        otm_call_iv = self._interpolate_iv(points, otm_call_strike, "CE")
        skew_25d = (otm_put_iv - otm_call_iv) if (otm_put_iv > 0 and otm_call_iv > 0) else 0.0

        # Smile curvature: average OTM IV vs ATM IV
        otm_avg = (otm_put_iv + otm_call_iv) / 2 if (otm_put_iv > 0 and otm_call_iv > 0) else atm_iv
        curvature = (otm_avg - atm_iv) if atm_iv > 0 else 0.0

        return SmileSlice(
            expiry=expiry,
            spot=self._spot,
            atm_iv=round(atm_iv, 4),
            skew_25d=round(skew_25d, 4),
            smile_curvature=round(curvature, 4),
            points=tuple(points),
        )

    def get_term_structure(self) -> list[dict]:
        """
        IV term structure — ATM IV across different expiries.
        Reveals if near-term events are priced in (inverted term structure).

        Returns:
            List of dicts with expiry, dte, and atm_iv, sorted by dte.
        """
        if not self._surface_data:
            log.warning("Surface not built. Call build() first.")
            return []

        expiries = sorted(set(p.expiry for p in self._surface_data))
        term_structure = []

        for expiry in expiries:
            points = [p for p in self._surface_data if p.expiry == expiry]
            strikes = sorted(set(p.strike for p in points))
            if not strikes:
                continue

            atm_strike = min(strikes, key=lambda s: abs(s - self._spot))
            atm_points = [p for p in points if p.strike == atm_strike]

            if not atm_points:
                continue

            atm_iv = sum(p.iv for p in atm_points) / len(atm_points)
            dte = _days_to_expiry(expiry)

            term_structure.append({
                "expiry": expiry,
                "dte": round(dte, 1),
                "atm_iv": round(atm_iv, 4),
                "atm_iv_pct": round(atm_iv * 100, 2),
                "atm_strike": atm_strike,
            })

        term_structure.sort(key=lambda x: x["dte"])

        # Detect inversion: near-term IV > far-term IV
        if len(term_structure) >= 2:
            near_iv = term_structure[0]["atm_iv"]
            far_iv = term_structure[-1]["atm_iv"]
            is_inverted = near_iv > far_iv * 1.05  # 5% threshold

            for entry in term_structure:
                entry["term_structure"] = "INVERTED" if is_inverted else "NORMAL"

        return term_structure

    def score_moneyness(self, expiry: str = "") -> list[MoneynessScore]:
        """
        Score each strike's IV relative to the smooth surface.
        Identifies cheap (underpriced) and expensive (overpriced) options.

        Returns:
            Sorted list of MoneynessScore, most deviated first.
        """
        if not self._surface_data:
            log.warning("Surface not built. Call build() first.")
            return []

        expiries = sorted(set(p.expiry for p in self._surface_data))
        if not expiry:
            expiry = expiries[0] if expiries else ""
        if not expiry:
            return []

        points = [p for p in self._surface_data if p.expiry == expiry]
        if len(points) < 5:
            return []

        # Build smooth IV curve using weighted average of neighbors
        all_ivs = [p.iv for p in points if p.iv > 0]
        if not all_ivs:
            return []

        iv_mean = sum(all_ivs) / len(all_ivs)
        iv_std = (sum((iv - iv_mean) ** 2 for iv in all_ivs) / len(all_ivs)) ** 0.5
        iv_std = max(iv_std, 0.001)  # prevent division by zero

        scores = []
        for point in points:
            if point.iv <= 0:
                continue

            # Fair IV: interpolated from neighbors (simple kernel smoothing)
            fair_iv = self._smooth_iv_at_strike(points, point.strike, point.option_type)
            deviation = point.iv - fair_iv
            percentile = norm.cdf((point.iv - iv_mean) / iv_std) * 100

            if deviation > iv_std * 1.5:
                signal = "EXPENSIVE"
            elif deviation < -iv_std * 1.5:
                signal = "CHEAP"
            else:
                signal = "FAIR"

            scores.append(MoneynessScore(
                strike=point.strike,
                expiry=expiry,
                option_type=point.option_type,
                market_iv=round(point.iv, 4),
                fair_iv=round(fair_iv, 4),
                iv_deviation=round(deviation, 4),
                percentile=round(percentile, 1),
                signal=signal,
            ))

        scores.sort(key=lambda s: abs(s.iv_deviation), reverse=True)
        return scores

    def detect_unusual_iv(self, threshold_std: float = 2.0) -> list[dict]:
        """
        Detect unusual IV patterns — where smart money is likely hedging.

        Looks for:
          - IV spikes at specific strikes (large block trades)
          - Volume/OI anomalies with elevated IV
          - Skew steepening (sudden demand for OTM puts)

        Args:
            threshold_std: Number of standard deviations to flag as unusual.

        Returns:
            List of anomalies with strike, type, and interpretation.
        """
        if not self._surface_data:
            log.warning("Surface not built. Call build() first.")
            return []

        anomalies = []
        expiries = sorted(set(p.expiry for p in self._surface_data))

        for expiry in expiries:
            points = [p for p in self._surface_data if p.expiry == expiry and p.iv > 0]
            if len(points) < 5:
                continue

            ivs = [p.iv for p in points]
            iv_mean = sum(ivs) / len(ivs)
            iv_std = (sum((v - iv_mean) ** 2 for v in ivs) / len(ivs)) ** 0.5
            if iv_std < 0.001:
                continue

            for point in points:
                z_score = (point.iv - iv_mean) / iv_std

                if abs(z_score) < threshold_std:
                    continue

                # Classify the anomaly
                distance_from_atm_pct = ((point.strike - self._spot) / self._spot) * 100

                if z_score > threshold_std:
                    if point.option_type == "PE" and distance_from_atm_pct < -3:
                        interpretation = "SMART_MONEY_HEDGING — elevated OTM put IV suggests institutional downside protection"
                    elif point.option_type == "CE" and distance_from_atm_pct > 3:
                        interpretation = "CALL_BUYING_PRESSURE — elevated OTM call IV suggests upside speculation"
                    else:
                        interpretation = "IV_SPIKE — unusual volatility demand at this strike"
                elif z_score < -threshold_std:
                    interpretation = "IV_CRUSH — abnormally low IV, possible overwriting or post-event deflation"
                else:
                    interpretation = "ANOMALY"

                anomalies.append({
                    "strike": point.strike,
                    "expiry": expiry,
                    "option_type": point.option_type,
                    "iv": round(point.iv * 100, 2),
                    "iv_mean": round(iv_mean * 100, 2),
                    "z_score": round(z_score, 2),
                    "oi": point.oi,
                    "volume": point.volume,
                    "distance_from_atm_pct": round(distance_from_atm_pct, 2),
                    "interpretation": interpretation,
                })

        anomalies.sort(key=lambda a: abs(a["z_score"]), reverse=True)
        log.info(f"IV anomalies found: {len(anomalies)} across {len(expiries)} expiries")
        return anomalies

    def get_surface_dataframe(self) -> pd.DataFrame:
        """
        Export the IV surface as a DataFrame for visualization or further analysis.
        Rows = strikes, columns include expiry, option_type, iv, moneyness.
        """
        if not self._surface_data:
            return pd.DataFrame()

        records = [
            {
                "strike": p.strike,
                "expiry": p.expiry,
                "option_type": p.option_type,
                "iv": p.iv,
                "iv_pct": round(p.iv * 100, 2),
                "ltp": p.ltp,
                "oi": p.oi,
                "volume": p.volume,
                "moneyness": p.moneyness,
            }
            for p in self._surface_data
        ]

        return pd.DataFrame(records)

    def summary(self) -> dict:
        """Quick summary of the current IV surface."""
        if not self._surface_data:
            return {"error": "surface_not_built"}

        smile = self.get_smile()
        term = self.get_term_structure()
        anomalies = self.detect_unusual_iv()

        return {
            "symbol": self._symbol,
            "spot": self._spot,
            "built_at": self._built_at.isoformat() if self._built_at else None,
            "total_points": len(self._surface_data),
            "atm_iv_pct": round(smile.atm_iv * 100, 2) if smile else 0,
            "skew_25d": round(smile.skew_25d * 100, 2) if smile else 0,
            "term_structure": "INVERTED" if (term and term[0].get("term_structure") == "INVERTED") else "NORMAL",
            "near_iv_pct": round(term[0]["atm_iv"] * 100, 2) if term else 0,
            "far_iv_pct": round(term[-1]["atm_iv"] * 100, 2) if len(term) > 1 else 0,
            "anomaly_count": len(anomalies),
            "top_anomalies": anomalies[:3],
        }

    # ── Private helpers ──────────────────────────────────────────────

    def _interpolate_iv(
        self,
        points: list[IVPoint],
        target_strike: float,
        option_type: str,
    ) -> float:
        """Interpolate IV at a target strike using nearest neighbors."""
        typed_points = sorted(
            [p for p in points if p.option_type == option_type and p.iv > 0],
            key=lambda p: p.strike,
        )

        if not typed_points:
            return 0.0

        # Find bracketing strikes
        below = [p for p in typed_points if p.strike <= target_strike]
        above = [p for p in typed_points if p.strike >= target_strike]

        if not below:
            return above[0].iv if above else 0.0
        if not above:
            return below[-1].iv

        p_below = below[-1]
        p_above = above[0]

        if p_below.strike == p_above.strike:
            return p_below.iv

        # Linear interpolation
        weight = (target_strike - p_below.strike) / (p_above.strike - p_below.strike)
        return p_below.iv + weight * (p_above.iv - p_below.iv)

    def _smooth_iv_at_strike(
        self,
        points: list[IVPoint],
        strike: float,
        option_type: str,
        bandwidth: float = 0.03,
    ) -> float:
        """
        Gaussian kernel smoothing of IV at a given strike.
        Produces a 'fair' IV by averaging nearby strikes with distance weighting.
        """
        typed_points = [p for p in points if p.option_type == option_type and p.iv > 0]
        if not typed_points:
            return 0.0

        weights = []
        ivs = []

        for p in typed_points:
            distance = abs(p.strike - strike) / max(self._spot, 1.0)
            w = math.exp(-0.5 * (distance / bandwidth) ** 2)
            weights.append(w)
            ivs.append(p.iv)

        total_weight = sum(weights)
        if total_weight <= 0:
            return sum(ivs) / len(ivs) if ivs else 0.0

        return sum(w * iv for w, iv in zip(weights, ivs)) / total_weight

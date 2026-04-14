"""
Pairs Trading — Kalman Filter + Cointegration.
Integrated from: Pairs_Trading_Kalman (ozencgungor), Indian-Markets pairs.

Pipeline:
  1. Select correlated pairs via cointegration test
  2. Kalman Filter for dynamic hedge ratio
  3. Z-score spread for entry/exit signals
  4. Hurst exponent to confirm mean-reversion
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from .base import Strategy, Signal, SignalType

log = logging.getLogger("josho.strategy.pairs")

# Pre-selected Indian pairs (historically cointegrated)
INDIAN_PAIRS = [
    ("ICICIBANK", "HDFCBANK"),
    ("TCS", "INFY"),
    ("SBIN", "BANKBARODA"),
    ("TATASTEEL", "JSWSTEEL"),
    ("RELIANCE", "ONGC"),
    ("MARUTI", "TATAMOTORS"),
    ("HCLTECH", "WIPRO"),
    ("BAJFINANCE", "BAJAJFINSV"),
    ("AXISBANK", "KOTAKBANK"),
    ("HINDALCO", "VEDL"),
]


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """
    Hurst exponent — determines if series is mean-reverting.
    H < 0.5 → mean-reverting (good for pairs trading)
    H = 0.5 → random walk
    H > 0.5 → trending
    """
    lags = range(2, min(max_lag, len(series) // 2))
    tau = []
    for lag in lags:
        diffs = series.values[lag:] - series.values[:-lag]
        tau.append(np.std(diffs))

    if len(tau) < 2 or any(t == 0 for t in tau):
        return 0.5

    log_lags = np.log(list(lags))
    log_tau = np.log(tau)

    coeffs = np.polyfit(log_lags, log_tau, 1)
    return coeffs[0]


def half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion — how fast the spread reverts."""
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()

    aligned = pd.concat([spread_lag, spread_diff], axis=1).dropna()
    if len(aligned) < 10:
        return float("inf")

    x = aligned.iloc[:, 0].values.reshape(-1, 1)
    y = aligned.iloc[:, 1].values

    try:
        beta = np.linalg.lstsq(x, y, rcond=None)[0][0]
        if beta >= 0:
            return float("inf")
        return -np.log(2) / beta
    except Exception:
        return float("inf")


def cointegration_test(y: pd.Series, x: pd.Series) -> tuple[bool, float]:
    """
    Simple cointegration test using Engle-Granger method.
    Returns (is_cointegrated, p_value_approx).
    """
    if len(y) != len(x) or len(y) < 50:
        return False, 1.0

    # OLS regression: y = beta * x + alpha + residual
    x_arr = x.values.reshape(-1, 1)
    x_with_const = np.column_stack([x_arr, np.ones(len(x_arr))])
    try:
        coeffs = np.linalg.lstsq(x_with_const, y.values, rcond=None)[0]
    except Exception:
        return False, 1.0

    residuals = y.values - x_with_const @ coeffs

    # ADF-like test on residuals (simplified)
    resid_series = pd.Series(residuals)
    resid_lag = resid_series.shift(1).dropna()
    resid_diff = resid_series.diff().dropna()

    aligned = pd.concat([resid_lag, resid_diff], axis=1).dropna()
    if len(aligned) < 10:
        return False, 1.0

    x_test = aligned.iloc[:, 0].values.reshape(-1, 1)
    y_test = aligned.iloc[:, 1].values

    try:
        beta_test = np.linalg.lstsq(x_test, y_test, rcond=None)[0][0]
    except Exception:
        return False, 1.0

    # Critical value approximation for Engle-Granger
    # At 5%: -3.37, at 10%: -3.07
    t_stat = beta_test / (np.std(y_test - beta_test * x_test.flatten()) / np.sqrt(np.sum(x_test ** 2)))

    is_coint = t_stat < -3.07
    p_approx = max(0, min(1, 0.5 + t_stat * 0.1))

    return is_coint, p_approx


@dataclass
class KalmanState:
    """Kalman Filter state for dynamic hedge ratio estimation."""
    theta: np.ndarray  # [hedge_ratio, intercept]
    P: np.ndarray  # covariance matrix
    delta: float = 1e-4  # transition covariance multiplier
    Ve: float = 1e-3  # observation noise


class KalmanHedge:
    """Kalman Filter for dynamic hedge ratio — adapts to changing relationships."""

    def __init__(self, delta: float = 1e-4, ve: float = 1e-3):
        self.state = KalmanState(
            theta=np.zeros(2),
            P=np.eye(2),
            delta=delta,
            Ve=ve,
        )

    def update(self, price_x: float, price_y: float) -> dict:
        """Update Kalman filter with new price pair. Returns spread info."""
        F = np.array([price_x, 1.0])

        # Prediction
        R = self.state.P + self.state.delta * np.eye(2)
        y_hat = F @ self.state.theta
        error = price_y - y_hat

        # Kalman gain
        Qt = F @ R @ F.T + self.state.Ve
        Kt = R @ F.T / Qt

        # Update
        self.state.theta = self.state.theta + Kt * error
        self.state.P = R - np.outer(Kt, F) @ R

        return {
            "hedge_ratio": self.state.theta[0],
            "intercept": self.state.theta[1],
            "spread": error,
            "spread_std": np.sqrt(Qt),
            "z_score": error / np.sqrt(Qt) if Qt > 0 else 0,
        }


class PairsTrading(Strategy):
    """
    Pairs trading with Kalman Filter hedge ratio.
    Entry: Z-score exceeds threshold.
    Exit: Z-score reverts to 0.
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]] = None,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        max_half_life: int = 60,
    ):
        super().__init__(
            name="PairsTrading",
            description="Kalman Filter pairs trading on Indian stocks",
        )
        self.pairs = pairs or INDIAN_PAIRS[:5]
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_half_life = max_half_life
        self.kalman_filters: dict[str, KalmanHedge] = {}

    def analyze(self, market_data: dict) -> list[Signal]:
        signals = []

        for sym_a, sym_b in self.pairs:
            df_a = market_data.get(f"{sym_a}_candles")
            df_b = market_data.get(f"{sym_b}_candles")

            if df_a is None or df_b is None or len(df_a) < 50:
                continue

            # Cointegration check
            is_coint, p_val = cointegration_test(df_a["close"], df_b["close"])
            if not is_coint:
                continue

            # Hurst exponent on spread
            spread = df_a["close"] - df_b["close"]
            h = hurst_exponent(spread)
            if h >= 0.5:
                continue

            # Half-life check
            hl = half_life(spread)
            if hl > self.max_half_life:
                continue

            # Kalman Filter update
            pair_key = f"{sym_a}_{sym_b}"
            if pair_key not in self.kalman_filters:
                self.kalman_filters[pair_key] = KalmanHedge()

            kf = self.kalman_filters[pair_key]
            result = kf.update(df_b["close"].iloc[-1], df_a["close"].iloc[-1])
            z = result["z_score"]

            # Entry signals
            if z < -self.entry_threshold:
                # Spread below lower band → buy A, sell B
                signals.append(Signal(
                    signal_type=SignalType.BUY,
                    symbol=sym_a,
                    segment="CASH",
                    price=df_a["close"].iloc[-1],
                    confidence=min(abs(z) / 3, 0.9),
                    reason=f"Pairs LONG {sym_a}: z={z:.2f}, hedge={result['hedge_ratio']:.3f}, hurst={h:.2f}",
                    strategy_name=self.name,
                    metadata={
                        "pair": (sym_a, sym_b),
                        "z_score": z,
                        "hedge_ratio": result["hedge_ratio"],
                        "hurst": h,
                        "half_life": hl,
                        "direction": "long_spread",
                    },
                ))
                signals.append(Signal(
                    signal_type=SignalType.SELL,
                    symbol=sym_b,
                    segment="CASH",
                    quantity=int(abs(result["hedge_ratio"])),
                    price=df_b["close"].iloc[-1],
                    confidence=min(abs(z) / 3, 0.9),
                    reason=f"Pairs SHORT {sym_b}: hedge leg",
                    strategy_name=self.name,
                    metadata={"pair": (sym_a, sym_b), "leg": "hedge"},
                ))

            elif z > self.entry_threshold:
                # Spread above upper band → sell A, buy B
                signals.append(Signal(
                    signal_type=SignalType.SELL,
                    symbol=sym_a,
                    segment="CASH",
                    price=df_a["close"].iloc[-1],
                    confidence=min(abs(z) / 3, 0.9),
                    reason=f"Pairs SHORT {sym_a}: z={z:.2f}, hedge={result['hedge_ratio']:.3f}",
                    strategy_name=self.name,
                    metadata={
                        "pair": (sym_a, sym_b),
                        "z_score": z,
                        "hedge_ratio": result["hedge_ratio"],
                        "direction": "short_spread",
                    },
                ))
                signals.append(Signal(
                    signal_type=SignalType.BUY,
                    symbol=sym_b,
                    segment="CASH",
                    quantity=int(abs(result["hedge_ratio"])),
                    price=df_b["close"].iloc[-1],
                    confidence=min(abs(z) / 3, 0.9),
                    reason=f"Pairs LONG {sym_b}: hedge leg",
                    strategy_name=self.name,
                    metadata={"pair": (sym_a, sym_b), "leg": "hedge"},
                ))

        return signals

    def get_required_symbols(self) -> list[str]:
        symbols = set()
        for a, b in self.pairs:
            symbols.add(a)
            symbols.add(b)
        return list(symbols)

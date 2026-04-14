"""
Macro Correlation Matrix — Track rolling correlations between key Indian market
instruments and detect regime changes.

Instruments: NIFTY, BANKNIFTY, USD/INR, Crude Oil, Gold, US S&P 500, India VIX.
Data source: yfinance (free).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


# ── Constants ────────────────────────────────────────────────────

INSTRUMENTS: dict[str, str] = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "USDINR": "USDINR=X",
    "CRUDE": "CL=F",
    "GOLD": "GC=F",
    "SP500": "^GSPC",
    "VIX_INDIA": "^INDIAVIX",
}

DEFAULT_SHORT_WINDOW = 20
DEFAULT_LONG_WINDOW = 60
REGIME_CHANGE_THRESHOLD = 0.3  # absolute shift in correlation that flags a regime change


# ── Result Containers ────────────────────────────────────────────

@dataclass(frozen=True)
class CorrelationSnapshot:
    """Immutable snapshot of the correlation matrix at a point in time."""
    date: datetime
    short_window: int
    long_window: int
    short_corr: pd.DataFrame   # 20-day rolling correlation matrix
    long_corr: pd.DataFrame    # 60-day rolling correlation matrix
    regime_changes: list[dict]  # pairs where correlation regime shifted


@dataclass(frozen=True)
class RegimeChange:
    """A detected correlation regime change between two instruments."""
    pair: tuple[str, str]
    short_corr: float
    long_corr: float
    shift: float
    direction: str  # "convergence" or "divergence"
    date: datetime


# ── Data Fetching ────────────────────────────────────────────────

def fetch_macro_data(
    lookback_days: int = 365,
    instruments: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Fetch daily close prices for all macro instruments via yfinance.

    Returns DataFrame with instrument names as columns, dates as index.
    Missing data is forward-filled then back-filled.
    """
    if not HAS_YFINANCE:
        raise ImportError(
            "yfinance is required for correlations. Install: pip install yfinance"
        )

    tickers = instruments or INSTRUMENTS
    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    frames: dict[str, pd.Series] = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty:
                # yfinance may return MultiIndex columns; flatten
                close = df["Close"].squeeze() if isinstance(df["Close"], pd.DataFrame) else df["Close"]
                frames[name] = close
        except Exception:
            # Skip instruments that fail (network issues, delisted, etc.)
            continue

    if not frames:
        raise RuntimeError("Failed to fetch data for any instrument")

    combined = pd.DataFrame(frames)
    combined = combined.ffill().bfill()
    return combined


# ── Correlation Computation ──────────────────────────────────────

def compute_rolling_correlations(
    prices: pd.DataFrame,
    short_window: int = DEFAULT_SHORT_WINDOW,
    long_window: int = DEFAULT_LONG_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute rolling correlation matrices from daily returns.

    Returns (short_corr, long_corr) — each is the latest correlation matrix.
    """
    daily_returns = prices.pct_change().dropna()

    if len(daily_returns) < long_window:
        raise ValueError(
            f"Insufficient data: need {long_window} rows, have {len(daily_returns)}"
        )

    short_corr = daily_returns.tail(short_window).corr()
    long_corr = daily_returns.tail(long_window).corr()
    return short_corr, long_corr


def compute_rolling_correlation_series(
    prices: pd.DataFrame,
    pair: tuple[str, str],
    windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Compute rolling correlation time series for a specific pair.

    Returns DataFrame with one column per window size.
    """
    windows = windows or [DEFAULT_SHORT_WINDOW, DEFAULT_LONG_WINDOW]
    daily_returns = prices.pct_change().dropna()

    a, b = pair
    if a not in daily_returns.columns or b not in daily_returns.columns:
        raise ValueError(f"Pair {pair} not found in data columns: {list(daily_returns.columns)}")

    result = pd.DataFrame(index=daily_returns.index)
    for w in windows:
        result[f"corr_{w}d"] = daily_returns[a].rolling(w).corr(daily_returns[b])

    return result.dropna()


# ── Regime Change Detection ──────────────────────────────────────

def detect_regime_changes(
    short_corr: pd.DataFrame,
    long_corr: pd.DataFrame,
    threshold: float = REGIME_CHANGE_THRESHOLD,
) -> list[dict]:
    """
    Detect pairs where the short-term correlation has diverged significantly
    from the long-term correlation — potential trading opportunities.

    A large positive shift means the pair is becoming more correlated (convergence).
    A large negative shift means the pair is decorrelating (divergence).
    """
    instruments = list(short_corr.columns)
    changes: list[dict] = []

    for i, a in enumerate(instruments):
        for b in instruments[i + 1:]:
            s = short_corr.loc[a, b]
            l = long_corr.loc[a, b]  # noqa: E741

            if np.isnan(s) or np.isnan(l):
                continue

            shift = s - l
            if abs(shift) >= threshold:
                direction = "convergence" if shift > 0 else "divergence"
                changes.append({
                    "pair": (a, b),
                    "short_corr": round(float(s), 4),
                    "long_corr": round(float(l), 4),
                    "shift": round(float(shift), 4),
                    "direction": direction,
                })

    # Sort by absolute shift descending (biggest regime change first)
    changes.sort(key=lambda c: abs(c["shift"]), reverse=True)
    return changes


# ── High-Level API ───────────────────────────────────────────────

def get_correlation_snapshot(
    lookback_days: int = 365,
    short_window: int = DEFAULT_SHORT_WINDOW,
    long_window: int = DEFAULT_LONG_WINDOW,
    threshold: float = REGIME_CHANGE_THRESHOLD,
) -> CorrelationSnapshot:
    """
    Fetch latest data and return a full correlation snapshot with regime changes.

    This is the main entry point for daily updates.
    """
    prices = fetch_macro_data(lookback_days=lookback_days)
    short_corr, long_corr = compute_rolling_correlations(prices, short_window, long_window)
    regime_changes = detect_regime_changes(short_corr, long_corr, threshold)

    return CorrelationSnapshot(
        date=datetime.now(),
        short_window=short_window,
        long_window=long_window,
        short_corr=short_corr,
        long_corr=long_corr,
        regime_changes=regime_changes,
    )


def daily_update() -> dict:
    """
    Run daily correlation update. Returns a summary dict suitable for logging
    or dashboard display.
    """
    snapshot = get_correlation_snapshot()

    summary = {
        "date": snapshot.date.isoformat(),
        "instruments": list(snapshot.short_corr.columns),
        "short_window": snapshot.short_window,
        "long_window": snapshot.long_window,
        "regime_changes_count": len(snapshot.regime_changes),
        "regime_changes": snapshot.regime_changes,
        "short_corr_matrix": snapshot.short_corr.round(4).to_dict(),
        "long_corr_matrix": snapshot.long_corr.round(4).to_dict(),
    }

    return summary


def format_correlation_report(snapshot: CorrelationSnapshot) -> str:
    """Format a human-readable correlation report."""
    lines = [
        f"=== Macro Correlation Report — {snapshot.date.strftime('%Y-%m-%d')} ===",
        "",
        f"Short-term ({snapshot.short_window}-day) Correlation Matrix:",
        snapshot.short_corr.round(2).to_string(),
        "",
        f"Long-term ({snapshot.long_window}-day) Correlation Matrix:",
        snapshot.long_corr.round(2).to_string(),
        "",
    ]

    if snapshot.regime_changes:
        lines.append(f"Regime Changes Detected ({len(snapshot.regime_changes)}):")
        for rc in snapshot.regime_changes:
            a, b = rc["pair"]
            lines.append(
                f"  {a} <-> {b}: {rc['direction']} "
                f"(short={rc['short_corr']:.2f}, long={rc['long_corr']:.2f}, "
                f"shift={rc['shift']:+.2f})"
            )
    else:
        lines.append("No significant regime changes detected.")

    return "\n".join(lines)

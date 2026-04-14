"""
Stock Screener — PKScreener + NSE-Stock-Scanner patterns.
Scans for high-probability setups across F&O stocks.

Screens:
  1. VCP (Volatility Contraction Pattern) — Minervini breakout
  2. NR-7 (Narrow Range 7) — volatility squeeze
  3. Open=High / Open=Low — intraday directional
  4. Volume Surge — unusual activity
  5. 52-Week High/Low proximity
  6. Piped Scanner — chain multiple screens
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass

from ..utils.indicators import (
    atr, rsi, sma, ema, adx, bollinger_bands, volume_surge,
    detect_consolidation, detect_breakout, momentum_score,
)

log = logging.getLogger("josho.screener")


@dataclass
class ScanResult:
    """Result from a stock scan."""
    symbol: str
    scan_type: str
    score: float  # 0-100 confidence
    price: float
    details: dict


def scan_vcp(df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
    """
    Volatility Contraction Pattern (VCP) — Mark Minervini method.
    Detects progressively tighter price contractions before breakout.
    """
    if len(df) < 60:
        return None

    # Check for contracting ranges over 3 periods
    ranges = []
    for period in [20, 10, 5]:
        window = df.tail(period)
        r = (window["high"].max() - window["low"].min()) / window["close"].mean()
        ranges.append(r)

    # VCP: each range should be tighter than the previous
    if not (ranges[0] > ranges[1] > ranges[2]):
        return None

    # Contraction ratio
    contraction = ranges[2] / ranges[0] if ranges[0] > 0 else 1
    if contraction > 0.5:  # needs at least 50% contraction
        return None

    # Volume should be declining (drying up before breakout)
    vol_20 = df["volume"].tail(20).mean()
    vol_5 = df["volume"].tail(5).mean()
    vol_declining = vol_5 < vol_20 * 0.8

    # Price should be above 50 SMA (uptrend)
    sma_50 = sma(df, 50).iloc[-1]
    above_sma = df["close"].iloc[-1] > sma_50

    if vol_declining and above_sma:
        return ScanResult(
            symbol=symbol,
            scan_type="VCP",
            score=min((1 - contraction) * 100, 95),
            price=df["close"].iloc[-1],
            details={
                "ranges": [round(r * 100, 2) for r in ranges],
                "contraction": round(contraction, 3),
                "volume_declining": vol_declining,
                "above_sma50": above_sma,
            },
        )
    return None


def scan_nr7(df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
    """
    NR-7 — Current candle has narrowest range of last 7 days.
    Signals volatility compression → imminent breakout.
    """
    if len(df) < 8:
        return None

    recent = df.tail(7)
    ranges = recent["high"] - recent["low"]

    if ranges.iloc[-1] == ranges.min():
        adx_val = adx(df).iloc[-1] if len(df) > 20 else 0
        return ScanResult(
            symbol=symbol,
            scan_type="NR7",
            score=70 + min(adx_val, 30),  # higher ADX = more likely to break
            price=df["close"].iloc[-1],
            details={
                "current_range": round(ranges.iloc[-1], 2),
                "avg_range_7d": round(ranges.mean(), 2),
                "compression_ratio": round(ranges.iloc[-1] / ranges.mean(), 3),
                "adx": round(adx_val, 1),
            },
        )
    return None


def scan_open_equals(df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
    """
    Open=High (short setup) or Open=Low (long setup).
    High probability intraday directional signal.
    """
    if len(df) < 2:
        return None

    last = df.iloc[-1]
    tolerance = (last["high"] - last["low"]) * 0.02  # 2% tolerance

    if abs(last["open"] - last["high"]) <= tolerance:
        # Open=High → bears dominant → short
        atr_val = atr(df).iloc[-1] if len(df) > 15 else 0
        remaining_move = atr_val - (last["open"] - last["low"])
        return ScanResult(
            symbol=symbol,
            scan_type="OPEN_EQ_HIGH",
            score=75,
            price=last["close"],
            details={
                "direction": "SHORT",
                "atr": round(atr_val, 2),
                "remaining_move": round(remaining_move, 2),
                "day_move_pct": round((last["open"] - last["low"]) / last["open"] * 100, 2),
            },
        )

    elif abs(last["open"] - last["low"]) <= tolerance:
        # Open=Low → bulls dominant → long
        atr_val = atr(df).iloc[-1] if len(df) > 15 else 0
        remaining_move = atr_val - (last["high"] - last["open"])
        return ScanResult(
            symbol=symbol,
            scan_type="OPEN_EQ_LOW",
            score=75,
            price=last["close"],
            details={
                "direction": "LONG",
                "atr": round(atr_val, 2),
                "remaining_move": round(remaining_move, 2),
                "day_move_pct": round((last["high"] - last["open"]) / last["open"] * 100, 2),
            },
        )

    return None


def scan_volume_breakout(df: pd.DataFrame, symbol: str, threshold: float = 2.0) -> Optional[ScanResult]:
    """Volume surge + price breakout — institutional activity."""
    if len(df) < 25:
        return None

    vs = volume_surge(df, period=20).iloc[-1]
    if vs < threshold:
        return None

    breakout = detect_breakout(df, period=20)
    if not breakout["breakout_up"] and not breakout["breakout_down"]:
        return None

    direction = "BULLISH" if breakout["breakout_up"] else "BEARISH"
    m_score = momentum_score(df)

    return ScanResult(
        symbol=symbol,
        scan_type="VOLUME_BREAKOUT",
        score=min(vs * 30 + abs(m_score) * 0.3, 95),
        price=df["close"].iloc[-1],
        details={
            "volume_surge": round(vs, 2),
            "direction": direction,
            "momentum_score": round(m_score, 1),
            "breakout_up": breakout["breakout_up"],
            "breakout_down": breakout["breakout_down"],
        },
    )


def scan_52_week(df: pd.DataFrame, symbol: str, proximity_pct: float = 3.0) -> Optional[ScanResult]:
    """Near 52-week high or low."""
    if len(df) < 200:
        return None

    year_data = df.tail(252)
    high_52 = year_data["high"].max()
    low_52 = year_data["low"].min()
    current = df["close"].iloc[-1]

    near_high = ((high_52 - current) / high_52) * 100 < proximity_pct
    near_low = ((current - low_52) / low_52) * 100 < proximity_pct

    if near_high:
        return ScanResult(
            symbol=symbol,
            scan_type="NEAR_52W_HIGH",
            score=80,
            price=current,
            details={
                "52w_high": round(high_52, 2),
                "distance_pct": round((high_52 - current) / high_52 * 100, 2),
                "direction": "BULLISH",
            },
        )
    elif near_low:
        return ScanResult(
            symbol=symbol,
            scan_type="NEAR_52W_LOW",
            score=70,
            price=current,
            details={
                "52w_low": round(low_52, 2),
                "distance_pct": round((current - low_52) / low_52 * 100, 2),
                "direction": "POTENTIAL_REVERSAL",
            },
        )
    return None


# ── Piped Scanner (chain multiple screens) ────────────────────────

ALL_SCANNERS = {
    "vcp": scan_vcp,
    "nr7": scan_nr7,
    "open_equals": scan_open_equals,
    "volume_breakout": scan_volume_breakout,
    "52_week": scan_52_week,
}


def piped_scan(
    df: pd.DataFrame,
    symbol: str,
    scanners: list[str] = None,
) -> list[ScanResult]:
    """
    Run multiple scanners in a pipe — only return if ALL match.
    Like PKScreener's composite scanner chains.

    Example: piped_scan(df, "RELIANCE", ["volume_breakout", "nr7"])
    → Only returns if both volume breakout AND NR7 detected.
    """
    scanners = scanners or list(ALL_SCANNERS.keys())
    results = []

    for scanner_name in scanners:
        scanner_fn = ALL_SCANNERS.get(scanner_name)
        if not scanner_fn:
            continue
        result = scanner_fn(df, symbol)
        if result:
            results.append(result)
        else:
            return []  # pipe breaks — ALL must match

    return results


def full_scan(
    market_data: dict,
    symbols: list[str],
) -> dict[str, list[ScanResult]]:
    """Run all scanners on all symbols. Returns {symbol: [results]}."""
    all_results = {}

    for symbol in symbols:
        df = market_data.get(f"{symbol}_candles")
        if df is None or len(df) < 20:
            continue

        results = []
        for name, scanner in ALL_SCANNERS.items():
            result = scanner(df, symbol)
            if result:
                results.append(result)

        if results:
            all_results[symbol] = results
            for r in results:
                log.info(f"SCAN: {r.symbol} | {r.scan_type} | Score: {r.score:.0f} | {r.details}")

    return all_results

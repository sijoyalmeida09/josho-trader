"""
Order Flow Analyzer — Level 2 market depth analysis.
Detects bid/ask imbalance, large walls, institutional footprints.
Calculates order flow score, volume delta, cumulative delta.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from ..client import get_client, GrowwClient

log = logging.getLogger("josho.order_flow")


# ── Data Structures (immutable) ───────────────────────────────────

@dataclass(frozen=True)
class DepthLevel:
    """Single bid or ask level from market depth."""
    price: float
    quantity: int
    order_count: int


@dataclass(frozen=True)
class MarketDepth:
    """Full Level 2 depth snapshot (immutable)."""
    symbol: str
    timestamp: float
    bids: tuple[DepthLevel, ...]  # best bid first (highest price)
    asks: tuple[DepthLevel, ...]  # best ask first (lowest price)
    ltp: float = 0.0
    volume: int = 0


@dataclass(frozen=True)
class DepthAnalysis:
    """Analysis result from a single depth snapshot (immutable)."""
    symbol: str
    timestamp: float
    ltp: float

    # Imbalance
    bid_total_qty: int
    ask_total_qty: int
    imbalance_ratio: float  # >1 = bid heavy, <1 = ask heavy
    imbalance_pct: float  # -100 to +100

    # Walls
    largest_bid: DepthLevel
    largest_ask: DepthLevel
    bid_wall_detected: bool  # qty > 3x average
    ask_wall_detected: bool

    # Spread
    spread: float
    spread_pct: float
    best_bid: float
    best_ask: float

    # Order density
    bid_order_count: int
    ask_order_count: int
    bid_avg_order_size: float
    ask_avg_order_size: float

    # Institutional footprint signals
    institutional_bid: bool  # few large orders (low count, high qty)
    institutional_ask: bool


@dataclass(frozen=True)
class OrderFlowScore:
    """Composite order flow assessment (immutable)."""
    symbol: str
    timestamp: float

    score: float  # -100 (extreme selling) to +100 (extreme buying)
    volume_delta: float  # positive = buy pressure, negative = sell
    cumulative_delta: float  # running sum of volume delta

    # Component scores
    imbalance_score: float  # from bid/ask imbalance
    wall_score: float  # from wall detection
    spread_score: float  # from spread analysis
    institutional_score: float  # from institutional footprint

    signal: str  # "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
    confidence: float  # 0.0 to 1.0


# ── Depth Parser ──────────────────────────────────────────────────

def parse_depth(quote: dict, symbol: str) -> Optional[MarketDepth]:
    """Parse Groww get_quote() response into MarketDepth.

    Expected quote structure:
    {
        "payload": {
            "ltp": 24500.0,
            "volume": 1234567,
            "depth": {
                "buy": [{"price": 24499, "quantity": 100, "orderCount": 5}, ...],
                "sell": [{"price": 24501, "quantity": 80, "orderCount": 3}, ...]
            }
        }
    }
    """
    payload = quote.get("payload") or quote
    depth_data = payload.get("depth", {})

    buy_levels = depth_data.get("buy", [])
    sell_levels = depth_data.get("sell", [])

    if not buy_levels and not sell_levels:
        return None

    bids = tuple(
        DepthLevel(
            price=float(lvl.get("price", 0)),
            quantity=int(lvl.get("quantity", 0)),
            order_count=int(lvl.get("orderCount", 0) or lvl.get("order_count", 0)),
        )
        for lvl in buy_levels
        if lvl.get("price", 0) > 0
    )

    asks = tuple(
        DepthLevel(
            price=float(lvl.get("price", 0)),
            quantity=int(lvl.get("quantity", 0)),
            order_count=int(lvl.get("orderCount", 0) or lvl.get("order_count", 0)),
        )
        for lvl in sell_levels
        if lvl.get("price", 0) > 0
    )

    return MarketDepth(
        symbol=symbol,
        timestamp=time.time(),
        bids=bids,
        asks=asks,
        ltp=float(payload.get("ltp", 0) or payload.get("last_price", 0) or 0),
        volume=int(payload.get("volume", 0) or payload.get("totalTradedVolume", 0) or 0),
    )


# ── Depth Analyzer ────────────────────────────────────────────────

def analyze_depth(depth: MarketDepth) -> Optional[DepthAnalysis]:
    """Analyze a single depth snapshot for imbalance, walls, institutional signals."""
    if not depth.bids or not depth.asks:
        return None

    # Totals
    bid_total = sum(b.quantity for b in depth.bids)
    ask_total = sum(a.quantity for a in depth.asks)

    # Imbalance
    total = bid_total + ask_total
    imbalance_ratio = bid_total / ask_total if ask_total > 0 else float("inf")
    imbalance_pct = ((bid_total - ask_total) / total * 100) if total > 0 else 0.0

    # Largest levels
    largest_bid = max(depth.bids, key=lambda b: b.quantity)
    largest_ask = max(depth.asks, key=lambda a: a.quantity)

    # Wall detection: a level with qty > 3x median of OTHER levels
    # Using median-excluding-max avoids the wall itself inflating the baseline
    bid_qtys_excl = sorted(b.quantity for b in depth.bids if b is not largest_bid)
    ask_qtys_excl = sorted(a.quantity for a in depth.asks if a is not largest_ask)

    def _median(vals: list[int]) -> float:
        if not vals:
            return 0.0
        mid = len(vals) // 2
        return float(vals[mid]) if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0

    bid_baseline = _median(bid_qtys_excl) if bid_qtys_excl else (bid_total / len(depth.bids) if depth.bids else 0)
    ask_baseline = _median(ask_qtys_excl) if ask_qtys_excl else (ask_total / len(depth.asks) if depth.asks else 0)
    bid_wall = largest_bid.quantity > (3 * bid_baseline) if bid_baseline > 0 else False
    ask_wall = largest_ask.quantity > (3 * ask_baseline) if ask_baseline > 0 else False

    # Spread
    best_bid = depth.bids[0].price
    best_ask = depth.asks[0].price
    spread = best_ask - best_bid
    mid = (best_bid + best_ask) / 2
    spread_pct = (spread / mid * 100) if mid > 0 else 0.0

    # Order counts
    bid_orders = sum(b.order_count for b in depth.bids)
    ask_orders = sum(a.order_count for a in depth.asks)
    bid_avg_size = bid_total / bid_orders if bid_orders > 0 else 0.0
    ask_avg_size = ask_total / ask_orders if ask_orders > 0 else 0.0

    # Institutional footprint: few orders but large quantity (avg size > 5x median)
    # Heuristic: order count low but qty high relative to other side
    institutional_bid = (
        bid_orders > 0
        and bid_avg_size > (3 * ask_avg_size)
        and bid_orders < ask_orders * 0.5
    ) if ask_avg_size > 0 and ask_orders > 0 else False

    institutional_ask = (
        ask_orders > 0
        and ask_avg_size > (3 * bid_avg_size)
        and ask_orders < bid_orders * 0.5
    ) if bid_avg_size > 0 and bid_orders > 0 else False

    return DepthAnalysis(
        symbol=depth.symbol,
        timestamp=depth.timestamp,
        ltp=depth.ltp,
        bid_total_qty=bid_total,
        ask_total_qty=ask_total,
        imbalance_ratio=round(imbalance_ratio, 3),
        imbalance_pct=round(imbalance_pct, 2),
        largest_bid=largest_bid,
        largest_ask=largest_ask,
        bid_wall_detected=bid_wall,
        ask_wall_detected=ask_wall,
        spread=round(spread, 2),
        spread_pct=round(spread_pct, 4),
        best_bid=best_bid,
        best_ask=best_ask,
        bid_order_count=bid_orders,
        ask_order_count=ask_orders,
        bid_avg_order_size=round(bid_avg_size, 1),
        ask_avg_order_size=round(ask_avg_size, 1),
        institutional_bid=institutional_bid,
        institutional_ask=institutional_ask,
    )


# ── Order Flow Scorer ─────────────────────────────────────────────

class OrderFlowAnalyzer:
    """Continuous order flow analysis with history and cumulative delta.

    Usage:
        client = get_client()
        analyzer = OrderFlowAnalyzer()

        # Single snapshot
        score = analyzer.analyze_symbol("NIFTY 50", client)

        # Continuous polling
        for score in analyzer.poll("NIFTY 50", client, interval=1.0):
            print(score.signal, score.score)
    """

    HISTORY_SIZE = 120  # Keep last 120 snapshots (~2 min at 1/sec)

    def __init__(self):
        # symbol -> deque of DepthAnalysis
        self._history: dict[str, deque[DepthAnalysis]] = {}
        # symbol -> cumulative volume delta
        self._cum_delta: dict[str, float] = {}
        # symbol -> previous volume for delta calculation
        self._prev_volume: dict[str, int] = {}
        # symbol -> previous ltp for delta direction
        self._prev_ltp: dict[str, float] = {}

    def analyze_symbol(
        self,
        symbol: str,
        client: GrowwClient,
        exchange: str = "NSE",
        segment: str = "CASH",
    ) -> Optional[OrderFlowScore]:
        """Fetch quote and compute order flow score for a symbol."""
        quote = client.get_quote(symbol, exchange=exchange, segment=segment)
        if not quote:
            return None

        depth = parse_depth(quote, symbol)
        if depth is None:
            return None

        analysis = analyze_depth(depth)
        if analysis is None:
            return None

        return self._compute_score(analysis, depth)

    def analyze_depth_snapshot(self, depth: MarketDepth) -> Optional[OrderFlowScore]:
        """Compute score from an already-parsed depth snapshot."""
        analysis = analyze_depth(depth)
        if analysis is None:
            return None
        return self._compute_score(analysis, depth)

    def poll(
        self,
        symbol: str,
        client: GrowwClient,
        interval: float = 1.0,
        exchange: str = "NSE",
        segment: str = "CASH",
        max_iterations: int = 0,
    ):
        """Generator: continuously poll and yield OrderFlowScore.

        Args:
            max_iterations: 0 = infinite, >0 = stop after N iterations.
        """
        count = 0
        while max_iterations == 0 or count < max_iterations:
            score = self.analyze_symbol(symbol, client, exchange, segment)
            if score is not None:
                yield score
            count += 1
            if max_iterations == 0 or count < max_iterations:
                time.sleep(interval)

    def _compute_score(self, analysis: DepthAnalysis, depth: MarketDepth) -> OrderFlowScore:
        """Compute composite order flow score from depth analysis."""
        symbol = analysis.symbol

        # Initialize history
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self.HISTORY_SIZE)

        self._history[symbol].append(analysis)

        # ── Volume delta ──────────────────────────────────────
        # Positive delta = buying, negative = selling
        # Heuristic: if LTP moved up, attribute volume change to buyers
        prev_vol = self._prev_volume.get(symbol, depth.volume)
        prev_ltp = self._prev_ltp.get(symbol, analysis.ltp)
        vol_change = max(depth.volume - prev_vol, 0)
        ltp_direction = 1 if analysis.ltp >= prev_ltp else -1
        volume_delta = vol_change * ltp_direction

        cum_delta = self._cum_delta.get(symbol, 0.0) + volume_delta
        self._cum_delta[symbol] = cum_delta
        self._prev_volume[symbol] = depth.volume
        self._prev_ltp[symbol] = analysis.ltp

        # ── Component scores (-100 to +100 each) ─────────────

        # 1. Imbalance score: directly from imbalance_pct
        imbalance_score = max(-100.0, min(100.0, analysis.imbalance_pct))

        # 2. Wall score: bid wall = bullish support, ask wall = bearish resistance
        wall_score = 0.0
        if analysis.bid_wall_detected and not analysis.ask_wall_detected:
            wall_score = 50.0  # Support below
        elif analysis.ask_wall_detected and not analysis.bid_wall_detected:
            wall_score = -50.0  # Resistance above
        elif analysis.bid_wall_detected and analysis.ask_wall_detected:
            # Both walls — lean toward the larger one
            bid_wall_size = analysis.largest_bid.quantity
            ask_wall_size = analysis.largest_ask.quantity
            wall_ratio = (bid_wall_size - ask_wall_size) / max(bid_wall_size + ask_wall_size, 1)
            wall_score = wall_ratio * 50

        # 3. Spread score: tight spread = liquid/confident, wide = uncertain
        # Narrow spread slightly bullish if imbalance is positive
        spread_score = 0.0
        if analysis.spread_pct < 0.05:  # Tight spread
            spread_score = imbalance_score * 0.3  # Amplify imbalance direction
        elif analysis.spread_pct > 0.2:  # Wide spread
            spread_score = -abs(imbalance_score) * 0.2  # Penalize confidence

        # 4. Institutional score
        institutional_score = 0.0
        if analysis.institutional_bid and not analysis.institutional_ask:
            institutional_score = 75.0  # Big players buying
        elif analysis.institutional_ask and not analysis.institutional_bid:
            institutional_score = -75.0  # Big players selling
        elif analysis.institutional_bid and analysis.institutional_ask:
            institutional_score = 0.0  # Standoff

        # ── Weighted composite ────────────────────────────────
        weights = {
            "imbalance": 0.40,
            "wall": 0.20,
            "spread": 0.10,
            "institutional": 0.30,
        }

        raw_score = (
            imbalance_score * weights["imbalance"]
            + wall_score * weights["wall"]
            + spread_score * weights["spread"]
            + institutional_score * weights["institutional"]
        )

        # Smooth with recent history if available
        history = self._history[symbol]
        if len(history) >= 3:
            recent_imbalances = [h.imbalance_pct for h in list(history)[-5:]]
            trend = sum(recent_imbalances) / len(recent_imbalances)
            # Blend: 70% current, 30% trend
            raw_score = raw_score * 0.7 + trend * 0.3

        final_score = max(-100.0, min(100.0, raw_score))

        # ── Signal classification ─────────────────────────────
        if final_score >= 60:
            signal = "STRONG_BUY"
        elif final_score >= 25:
            signal = "BUY"
        elif final_score <= -60:
            signal = "STRONG_SELL"
        elif final_score <= -25:
            signal = "SELL"
        else:
            signal = "NEUTRAL"

        # Confidence: based on depth quality + history consistency
        depth_quality = min(len(depth.bids), len(depth.asks)) / 5  # 5 levels = full depth
        history_len = min(len(history), 10) / 10
        confidence = round((depth_quality * 0.6 + history_len * 0.4), 2)

        return OrderFlowScore(
            symbol=symbol,
            timestamp=analysis.timestamp,
            score=round(final_score, 1),
            volume_delta=volume_delta,
            cumulative_delta=cum_delta,
            imbalance_score=round(imbalance_score, 1),
            wall_score=round(wall_score, 1),
            spread_score=round(spread_score, 1),
            institutional_score=round(institutional_score, 1),
            signal=signal,
            confidence=confidence,
        )

    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset history for a symbol or all symbols."""
        if symbol:
            self._history.pop(symbol, None)
            self._cum_delta.pop(symbol, None)
            self._prev_volume.pop(symbol, None)
            self._prev_ltp.pop(symbol, None)
        else:
            self._history.clear()
            self._cum_delta.clear()
            self._prev_volume.clear()
            self._prev_ltp.clear()


# ── Multi-symbol scanner ──────────────────────────────────────────

def scan_order_flow(
    symbols: list[str],
    client: Optional[GrowwClient] = None,
    exchange: str = "NSE",
    segment: str = "CASH",
) -> list[OrderFlowScore]:
    """Scan multiple symbols and return scores sorted by absolute strength.

    Useful for finding the strongest buying/selling pressure across FNO watchlist.
    """
    if client is None:
        client = get_client()

    analyzer = OrderFlowAnalyzer()
    scores = []

    for symbol in symbols:
        try:
            score = analyzer.analyze_symbol(symbol, client, exchange, segment)
            if score is not None:
                scores.append(score)
        except Exception as e:
            log.debug(f"Order flow scan failed for {symbol}: {e}")
            continue

    scores.sort(key=lambda s: abs(s.score), reverse=True)
    return scores

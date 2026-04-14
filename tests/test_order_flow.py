"""Tests for order_flow depth analysis and scoring (no API needed)."""

from src.market_data.order_flow import (
    DepthLevel, MarketDepth, parse_depth, analyze_depth, OrderFlowAnalyzer,
)


def _make_depth(
    bids: list[tuple[float, int, int]],
    asks: list[tuple[float, int, int]],
    symbol: str = "NIFTY",
    ltp: float = 24500,
    volume: int = 100000,
) -> MarketDepth:
    """Helper to build a MarketDepth from simple tuples (price, qty, orders)."""
    return MarketDepth(
        symbol=symbol,
        timestamp=1000.0,
        bids=tuple(DepthLevel(p, q, o) for p, q, o in bids),
        asks=tuple(DepthLevel(p, q, o) for p, q, o in asks),
        ltp=ltp,
        volume=volume,
    )


def test_parse_depth_from_quote():
    """parse_depth extracts bid/ask levels from Groww quote format."""
    quote = {
        "payload": {
            "ltp": 24500,
            "volume": 500000,
            "depth": {
                "buy": [
                    {"price": 24499, "quantity": 100, "orderCount": 5},
                    {"price": 24498, "quantity": 200, "orderCount": 10},
                ],
                "sell": [
                    {"price": 24501, "quantity": 80, "orderCount": 3},
                    {"price": 24502, "quantity": 150, "orderCount": 7},
                ],
            },
        }
    }
    depth = parse_depth(quote, "NIFTY")
    assert depth is not None
    assert depth.symbol == "NIFTY"
    assert len(depth.bids) == 2
    assert len(depth.asks) == 2
    assert depth.bids[0].price == 24499
    assert depth.asks[0].quantity == 80
    assert depth.ltp == 24500


def test_parse_depth_empty():
    """Empty depth returns None."""
    assert parse_depth({"payload": {"depth": {}}}, "X") is None
    assert parse_depth({}, "X") is None


def test_analyze_depth_balanced():
    """Balanced book has imbalance near 0."""
    depth = _make_depth(
        bids=[(100, 500, 10), (99, 500, 10)],
        asks=[(101, 500, 10), (102, 500, 10)],
    )
    analysis = analyze_depth(depth)
    assert analysis is not None
    assert -5 < analysis.imbalance_pct < 5
    assert analysis.imbalance_ratio == 1.0
    assert not analysis.bid_wall_detected
    assert not analysis.ask_wall_detected


def test_analyze_depth_bid_heavy():
    """Bid-heavy book has positive imbalance."""
    depth = _make_depth(
        bids=[(100, 1000, 20), (99, 800, 15)],
        asks=[(101, 100, 5), (102, 100, 5)],
    )
    analysis = analyze_depth(depth)
    assert analysis is not None
    assert analysis.imbalance_pct > 50
    assert analysis.imbalance_ratio > 5


def test_analyze_depth_wall_detection():
    """Large bid at one level triggers wall detection."""
    depth = _make_depth(
        bids=[(100, 10000, 3), (99, 100, 10), (98, 100, 10)],
        asks=[(101, 200, 10), (102, 200, 10), (103, 200, 10)],
    )
    analysis = analyze_depth(depth)
    assert analysis is not None
    assert analysis.bid_wall_detected
    assert not analysis.ask_wall_detected
    assert analysis.largest_bid.quantity == 10000


def test_analyze_depth_spread():
    """Spread calculation is correct."""
    depth = _make_depth(
        bids=[(24499, 100, 5)],
        asks=[(24501, 100, 5)],
        ltp=24500,
    )
    analysis = analyze_depth(depth)
    assert analysis is not None
    assert analysis.spread == 2.0
    assert analysis.best_bid == 24499
    assert analysis.best_ask == 24501


def test_analyze_depth_institutional():
    """Few large orders on bid side = institutional footprint."""
    depth = _make_depth(
        bids=[(100, 10000, 2), (99, 8000, 1)],  # 3 orders, 18000 qty → avg 6000
        asks=[(101, 100, 50), (102, 100, 40)],  # 90 orders, 200 qty → avg 2.2
    )
    analysis = analyze_depth(depth)
    assert analysis is not None
    assert analysis.institutional_bid
    assert not analysis.institutional_ask


def test_order_flow_score_bullish():
    """Strong bid imbalance produces positive score."""
    analyzer = OrderFlowAnalyzer()
    depth = _make_depth(
        bids=[(100, 2000, 20), (99, 1500, 15), (98, 1000, 10)],
        asks=[(101, 200, 5), (102, 150, 5), (103, 100, 3)],
        volume=100000,
    )
    score = analyzer.analyze_depth_snapshot(depth)
    assert score is not None
    assert score.score > 0
    assert score.signal in ("BUY", "STRONG_BUY")


def test_order_flow_score_bearish():
    """Strong ask imbalance produces negative score."""
    analyzer = OrderFlowAnalyzer()
    depth = _make_depth(
        bids=[(100, 100, 5), (99, 80, 3)],
        asks=[(101, 3000, 20), (102, 2500, 15), (103, 2000, 10)],
        volume=100000,
    )
    score = analyzer.analyze_depth_snapshot(depth)
    assert score is not None
    assert score.score < 0
    assert score.signal in ("SELL", "STRONG_SELL")


def test_order_flow_score_range():
    """Score is always in [-100, +100]."""
    analyzer = OrderFlowAnalyzer()
    # Extreme imbalance
    depth = _make_depth(
        bids=[(100, 99999, 1)],
        asks=[(101, 1, 1)],
        volume=100000,
    )
    score = analyzer.analyze_depth_snapshot(depth)
    assert score is not None
    assert -100 <= score.score <= 100


def test_order_flow_cumulative_delta():
    """Cumulative delta accumulates across snapshots."""
    analyzer = OrderFlowAnalyzer()

    d1 = _make_depth(bids=[(100, 500, 10)], asks=[(101, 500, 10)],
                      volume=1000, ltp=100)
    d2 = _make_depth(bids=[(100, 500, 10)], asks=[(101, 500, 10)],
                      volume=2000, ltp=101)  # price up → positive delta
    d3 = _make_depth(bids=[(100, 500, 10)], asks=[(101, 500, 10)],
                      volume=3500, ltp=99)  # price down → negative delta

    s1 = analyzer.analyze_depth_snapshot(d1)
    s2 = analyzer.analyze_depth_snapshot(d2)
    s3 = analyzer.analyze_depth_snapshot(d3)

    assert s1 is not None and s2 is not None and s3 is not None
    # s2 should have positive delta (price went up, volume increased)
    assert s2.volume_delta > 0
    # s3 should have negative delta (price went down)
    assert s3.volume_delta < 0
    # Cumulative includes all
    assert s3.cumulative_delta == s2.volume_delta + s3.volume_delta


def test_order_flow_reset():
    """Reset clears history and cumulative delta."""
    analyzer = OrderFlowAnalyzer()
    depth = _make_depth(bids=[(100, 500, 10)], asks=[(101, 500, 10)], volume=1000)
    analyzer.analyze_depth_snapshot(depth)

    analyzer.reset("NIFTY")
    assert "NIFTY" not in analyzer._history
    assert "NIFTY" not in analyzer._cum_delta


def test_depth_immutability():
    """Data structures are frozen."""
    d = DepthLevel(price=100, quantity=500, order_count=10)
    try:
        d.price = 200  # type: ignore
        assert False, "Should not allow mutation"
    except AttributeError:
        pass


if __name__ == "__main__":
    test_parse_depth_from_quote()
    test_parse_depth_empty()
    test_analyze_depth_balanced()
    test_analyze_depth_bid_heavy()
    test_analyze_depth_wall_detection()
    test_analyze_depth_spread()
    test_analyze_depth_institutional()
    test_order_flow_score_bullish()
    test_order_flow_score_bearish()
    test_order_flow_score_range()
    test_order_flow_cumulative_delta()
    test_order_flow_reset()
    test_depth_immutability()
    print("All order_flow tests passed!")

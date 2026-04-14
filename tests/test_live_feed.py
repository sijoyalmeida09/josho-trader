"""Tests for live_feed candle aggregation and tick parsing (no WebSocket needed)."""

import time
from src.market_data.live_feed import Tick, Candle, CandleAggregator, LiveFeed


def test_candle_aggregator_basic():
    """Ticks in the same 1-second window accumulate; new window flushes a candle."""
    agg = CandleAggregator(interval_seconds=1)
    base = 1000.0  # arbitrary epoch

    t1 = Tick(symbol="NIFTY", ltp=24500, volume=100, timestamp=base + 0.1)
    t2 = Tick(symbol="NIFTY", ltp=24510, volume=200, timestamp=base + 0.5)
    t3 = Tick(symbol="NIFTY", ltp=24490, volume=150, timestamp=base + 0.9)

    assert agg.add_tick(t1) is None
    assert agg.add_tick(t2) is None
    assert agg.add_tick(t3) is None

    # Next second — triggers flush of previous window
    t4 = Tick(symbol="NIFTY", ltp=24520, volume=300, timestamp=base + 1.2)
    candle = agg.add_tick(t4)

    assert candle is not None
    assert candle.symbol == "NIFTY"
    assert candle.open == 24500
    assert candle.high == 24510
    assert candle.low == 24490
    assert candle.close == 24490
    assert candle.tick_count == 3
    assert candle.interval_seconds == 1


def test_candle_aggregator_multiple_symbols():
    """Different symbols get independent candles."""
    agg = CandleAggregator(interval_seconds=1)
    base = 2000.0

    agg.add_tick(Tick(symbol="NIFTY", ltp=24500, volume=100, timestamp=base + 0.1))
    agg.add_tick(Tick(symbol="SBIN", ltp=800, volume=500, timestamp=base + 0.2))

    # Roll both
    c1 = agg.add_tick(Tick(symbol="NIFTY", ltp=24510, volume=200, timestamp=base + 1.0))
    c2 = agg.add_tick(Tick(symbol="SBIN", ltp=805, volume=600, timestamp=base + 1.1))

    assert c1 is not None and c1.symbol == "NIFTY"
    assert c2 is not None and c2.symbol == "SBIN"
    assert c2.open == 800


def test_candle_aggregator_flush_all():
    """flush_all returns partial candles and resets state."""
    agg = CandleAggregator(interval_seconds=60)
    base = 3000.0

    agg.add_tick(Tick(symbol="RELIANCE", ltp=2900, volume=1000, timestamp=base))
    agg.add_tick(Tick(symbol="RELIANCE", ltp=2910, volume=2000, timestamp=base + 10))

    candles = agg.flush_all()
    assert len(candles) == 1
    assert candles[0].symbol == "RELIANCE"
    assert candles[0].high == 2910
    assert candles[0].tick_count == 2

    # After flush, state is clean
    assert agg.flush_all() == []


def test_tick_parsing_single():
    """LiveFeed._parse_ticks handles single tick dict."""
    feed = LiveFeed.__new__(LiveFeed)  # skip __init__
    ticks = feed._parse_ticks({"symbol": "NIFTY", "ltp": 24500, "volume": 100})
    assert len(ticks) == 1
    assert ticks[0].symbol == "NIFTY"
    assert ticks[0].ltp == 24500


def test_tick_parsing_batch():
    """LiveFeed._parse_ticks handles batch format."""
    feed = LiveFeed.__new__(LiveFeed)
    ticks = feed._parse_ticks({
        "data": [
            {"symbol": "NIFTY", "ltp": 24500, "volume": 100},
            {"symbol": "SBIN", "ltp": 800, "volume": 500},
        ]
    })
    assert len(ticks) == 2


def test_tick_parsing_quote_update():
    """LiveFeed._parse_ticks handles quote update format."""
    feed = LiveFeed.__new__(LiveFeed)
    ticks = feed._parse_ticks({
        "type": "quote",
        "payload": {"trading_symbol": "RELIANCE", "last_price": 2900, "volume": 1000},
    })
    assert len(ticks) == 1
    assert ticks[0].symbol == "RELIANCE"
    assert ticks[0].ltp == 2900


def test_tick_parsing_missing_fields():
    """Missing symbol or ltp returns no ticks."""
    feed = LiveFeed.__new__(LiveFeed)
    assert feed._parse_ticks({"symbol": "NIFTY"}) == []  # no ltp
    assert feed._parse_ticks({"ltp": 100}) == []  # no symbol
    assert feed._parse_ticks({}) == []


def test_candle_immutability():
    """Candle and Tick are frozen dataclasses."""
    t = Tick(symbol="X", ltp=100, volume=1, timestamp=0)
    try:
        t.ltp = 200  # type: ignore
        assert False, "Should not allow mutation"
    except AttributeError:
        pass

    c = Candle(symbol="X", open=100, high=101, low=99, close=100,
               volume=10, tick_count=3, start_time=0, interval_seconds=1)
    try:
        c.high = 200  # type: ignore
        assert False, "Should not allow mutation"
    except (AttributeError, TypeError):
        pass


if __name__ == "__main__":
    test_candle_aggregator_basic()
    test_candle_aggregator_multiple_symbols()
    test_candle_aggregator_flush_all()
    test_tick_parsing_single()
    test_tick_parsing_batch()
    test_tick_parsing_quote_update()
    test_tick_parsing_missing_fields()
    test_candle_immutability()
    print("All live_feed tests passed!")

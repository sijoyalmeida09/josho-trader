"""
Live Tick Feed — WebSocket real-time market data from Groww.
Connects via generate_socket_token(), subscribes to FNO instruments,
buffers ticks, and aggregates into 1-second and 1-minute candles.
"""

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional

log = logging.getLogger("josho.live_feed")

IST = timezone(timedelta(hours=5, minutes=30))

# ── Instruments ────────────────────────────────────────────────────

INDEX_SYMBOLS = ["NIFTY 50", "NIFTY BANK", "NIFTY FIN SERVICE"]

FNO_STOCKS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
    "HINDUNILVR", "MARUTI", "TATAMOTORS", "TATASTEEL",
    "AXISBANK", "BAJFINANCE", "WIPRO", "HCLTECH",
    "ADANIENT", "ADANIPORTS", "TITAN", "ULTRACEMCO",
]

DEFAULT_SUBSCRIBE = INDEX_SYMBOLS + FNO_STOCKS


# ── Data Structures ───────────────────────────────────────────────

@dataclass(frozen=True)
class Tick:
    """Single market tick (immutable)."""
    symbol: str
    ltp: float
    volume: int
    timestamp: float  # epoch seconds
    bid: float = 0.0
    ask: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0  # prev close
    change_pct: float = 0.0


@dataclass(frozen=True)
class Candle:
    """Aggregated candle (immutable)."""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    tick_count: int
    start_time: float  # epoch seconds
    interval_seconds: int  # 1 or 60


class CandleAggregator:
    """Aggregates ticks into candles for a given interval.

    Thread-safe. Does not mutate ticks — builds new Candle objects.
    """

    def __init__(self, interval_seconds: int):
        self.interval = interval_seconds
        self._lock = threading.Lock()
        # symbol -> list of ticks in current window
        self._buffers: dict[str, list[Tick]] = defaultdict(list)
        # symbol -> window start epoch
        self._window_start: dict[str, float] = {}

    def _window_for(self, ts: float) -> float:
        """Return the window-start epoch for a timestamp."""
        return (ts // self.interval) * self.interval

    def add_tick(self, tick: Tick) -> Optional[Candle]:
        """Add a tick. Returns a completed Candle if the window rolled over, else None."""
        window = self._window_for(tick.timestamp)

        with self._lock:
            prev_window = self._window_start.get(tick.symbol)

            # First tick for this symbol
            if prev_window is None:
                self._window_start[tick.symbol] = window
                self._buffers[tick.symbol] = [tick]
                return None

            # Same window — accumulate
            if window == prev_window:
                self._buffers[tick.symbol] = [*self._buffers[tick.symbol], tick]
                return None

            # New window — flush old, start new
            completed = self._flush_symbol(tick.symbol, prev_window)
            self._window_start[tick.symbol] = window
            self._buffers[tick.symbol] = [tick]
            return completed

    def _flush_symbol(self, symbol: str, window_start: float) -> Optional[Candle]:
        """Build a candle from buffered ticks. Does NOT clear the buffer (caller does)."""
        ticks = self._buffers.get(symbol, [])
        if not ticks:
            return None

        prices = [t.ltp for t in ticks]
        total_vol = sum(t.volume for t in ticks)

        return Candle(
            symbol=symbol,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=total_vol,
            tick_count=len(ticks),
            start_time=window_start,
            interval_seconds=self.interval,
        )

    def flush_all(self) -> list[Candle]:
        """Force-flush all symbols. Used on disconnect or market close."""
        candles = []
        with self._lock:
            for symbol in list(self._buffers.keys()):
                ws = self._window_start.get(symbol, 0)
                candle = self._flush_symbol(symbol, ws)
                if candle is not None:
                    candles.append(candle)
            self._buffers.clear()
            self._window_start.clear()
        return candles


# ── Callback type aliases ─────────────────────────────────────────

OnTick = Callable[[Tick], None]
OnCandle = Callable[[Candle], None]


# ── Live Feed ─────────────────────────────────────────────────────

class LiveFeed:
    """WebSocket live tick feed with candle aggregation and auto-reconnect.

    Usage:
        from src.client import get_client
        from src.market_data.live_feed import LiveFeed

        feed = LiveFeed(get_client())
        feed.on_tick(lambda t: print(t))
        feed.on_candle_1s(lambda c: print("1s", c))
        feed.on_candle_1m(lambda c: print("1m", c))
        feed.start()   # blocking; call start_async() for background thread
    """

    MAX_RECONNECT_DELAY = 60
    INITIAL_RECONNECT_DELAY = 1

    def __init__(self, client, symbols: Optional[list[str]] = None):
        self.client = client
        self.symbols = symbols or DEFAULT_SUBSCRIBE

        # Callbacks
        self._tick_callbacks: list[OnTick] = []
        self._candle_1s_callbacks: list[OnCandle] = []
        self._candle_1m_callbacks: list[OnCandle] = []

        # Aggregators
        self._agg_1s = CandleAggregator(interval_seconds=1)
        self._agg_1m = CandleAggregator(interval_seconds=60)

        # State
        self._ws = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._reconnect_delay = self.INITIAL_RECONNECT_DELAY
        self._tick_count = 0
        self._last_tick_time = 0.0

    # ── Callback registration ─────────────────────────────────────

    def on_tick(self, cb: OnTick) -> "LiveFeed":
        self._tick_callbacks = [*self._tick_callbacks, cb]
        return self

    def on_candle_1s(self, cb: OnCandle) -> "LiveFeed":
        self._candle_1s_callbacks = [*self._candle_1s_callbacks, cb]
        return self

    def on_candle_1m(self, cb: OnCandle) -> "LiveFeed":
        self._candle_1m_callbacks = [*self._candle_1m_callbacks, cb]
        return self

    # ── Lifecycle ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start the feed (blocking). Reconnects automatically on failure."""
        self._running = True
        log.info(f"LiveFeed starting — {len(self.symbols)} symbols")

        while self._running:
            try:
                self._connect_and_run()
            except Exception as e:
                if not self._running:
                    break
                log.error(f"WebSocket error: {e} — reconnecting in {self._reconnect_delay}s")
                time.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self.MAX_RECONNECT_DELAY,
                )

        log.info("LiveFeed stopped")

    def start_async(self) -> threading.Thread:
        """Start the feed in a background daemon thread."""
        self._thread = threading.Thread(target=self.start, daemon=True, name="live-feed")
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        """Gracefully stop the feed."""
        log.info("LiveFeed stopping...")
        self._running = False

        # Flush remaining candles
        for candle in self._agg_1s.flush_all():
            self._emit_candle_1s(candle)
        for candle in self._agg_1m.flush_all():
            self._emit_candle_1m(candle)

        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    @property
    def stats(self) -> dict:
        """Current feed statistics."""
        return {
            "running": self._running,
            "tick_count": self._tick_count,
            "last_tick_time": self._last_tick_time,
            "symbols": len(self.symbols),
            "reconnect_delay": self._reconnect_delay,
        }

    # ── WebSocket connection ───────────────────────────────────────

    def _connect_and_run(self) -> None:
        """Obtain a socket token, connect, subscribe, and process messages."""
        try:
            import websocket
        except ImportError:
            log.error("websocket-client not installed. Run: pip install websocket-client")
            raise

        # Get auth token
        token = self.client.get_socket_token()
        if not token:
            raise ConnectionError("Failed to obtain WebSocket token")

        log.info("WebSocket token obtained, connecting...")

        ws_url = f"wss://groww.in/ws?token={token}"

        ws = websocket.WebSocketApp(
            ws_url,
            on_open=self._on_ws_open,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close,
        )
        self._ws = ws
        ws.run_forever(ping_interval=30, ping_timeout=10)

    def _on_ws_open(self, ws) -> None:
        """Subscribe to symbols on connection."""
        log.info("WebSocket connected — subscribing...")
        self._reconnect_delay = self.INITIAL_RECONNECT_DELAY

        subscribe_msg = {
            "action": "subscribe",
            "symbols": self.symbols,
        }
        try:
            ws.send(json.dumps(subscribe_msg))
            log.info(f"Subscribed to {len(self.symbols)} symbols")
        except Exception as e:
            log.error(f"Subscribe failed: {e}")

    def _on_ws_message(self, ws, message: str) -> None:
        """Parse incoming tick and feed into aggregators."""
        try:
            data = json.loads(message) if isinstance(message, str) else message
        except (json.JSONDecodeError, TypeError):
            log.debug(f"Non-JSON message: {message[:100]}")
            return

        # Handle different message formats from Groww
        ticks = self._parse_ticks(data)
        for tick in ticks:
            self._process_tick(tick)

    def _on_ws_error(self, ws, error) -> None:
        log.warning(f"WebSocket error: {error}")

    def _on_ws_close(self, ws, close_status_code, close_msg) -> None:
        log.info(f"WebSocket closed: {close_status_code} — {close_msg}")

    # ── Tick parsing ───────────────────────────────────────────────

    def _parse_ticks(self, data: dict) -> list[Tick]:
        """Parse raw WebSocket message into Tick objects.

        Groww WebSocket formats vary — handle the common shapes:
        1. Single tick: {"symbol": ..., "ltp": ..., ...}
        2. Batch: {"data": [{"symbol": ..., ...}, ...]}
        3. Quote update: {"type": "quote", "payload": {...}}
        """
        ticks = []
        now = time.time()

        # Batch format
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            for item in data["data"]:
                tick = self._dict_to_tick(item, now)
                if tick is not None:
                    ticks.append(tick)
            return ticks

        # Quote update format
        if isinstance(data, dict) and data.get("type") == "quote" and "payload" in data:
            tick = self._dict_to_tick(data["payload"], now)
            if tick is not None:
                ticks.append(tick)
            return ticks

        # Single tick format
        if isinstance(data, dict) and ("symbol" in data or "trading_symbol" in data):
            tick = self._dict_to_tick(data, now)
            if tick is not None:
                ticks.append(tick)

        return ticks

    def _dict_to_tick(self, d: dict, fallback_ts: float) -> Optional[Tick]:
        """Convert a raw dict to a Tick. Returns None if essential fields missing."""
        symbol = d.get("symbol") or d.get("trading_symbol") or d.get("tradingSymbol", "")
        ltp = d.get("ltp") or d.get("last_price") or d.get("lastPrice")

        if not symbol or ltp is None:
            return None

        try:
            ltp = float(ltp)
        except (ValueError, TypeError):
            return None

        ts = d.get("timestamp") or d.get("exchange_timestamp") or fallback_ts
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts).timestamp()
            except ValueError:
                ts = fallback_ts

        return Tick(
            symbol=str(symbol),
            ltp=ltp,
            volume=int(d.get("volume", 0) or d.get("totalTradedVolume", 0) or 0),
            timestamp=float(ts),
            bid=float(d.get("bid", 0) or d.get("bestBid", 0) or 0),
            ask=float(d.get("ask", 0) or d.get("bestAsk", 0) or 0),
            open=float(d.get("open", 0) or 0),
            high=float(d.get("high", 0) or 0),
            low=float(d.get("low", 0) or 0),
            close=float(d.get("close", 0) or d.get("prev_close", 0) or 0),
            change_pct=float(d.get("change_pct", 0) or d.get("percentChange", 0) or 0),
        )

    # ── Tick processing ────────────────────────────────────────────

    def _process_tick(self, tick: Tick) -> None:
        """Emit tick callbacks and feed into candle aggregators."""
        self._tick_count += 1
        self._last_tick_time = tick.timestamp

        # Emit raw tick
        for cb in self._tick_callbacks:
            try:
                cb(tick)
            except Exception as e:
                log.error(f"Tick callback error: {e}")

        # 1-second candle
        candle_1s = self._agg_1s.add_tick(tick)
        if candle_1s is not None:
            self._emit_candle_1s(candle_1s)

        # 1-minute candle
        candle_1m = self._agg_1m.add_tick(tick)
        if candle_1m is not None:
            self._emit_candle_1m(candle_1m)

    def _emit_candle_1s(self, candle: Candle) -> None:
        for cb in self._candle_1s_callbacks:
            try:
                cb(candle)
            except Exception as e:
                log.error(f"1s candle callback error: {e}")

    def _emit_candle_1m(self, candle: Candle) -> None:
        for cb in self._candle_1m_callbacks:
            try:
                cb(candle)
            except Exception as e:
                log.error(f"1m candle callback error: {e}")

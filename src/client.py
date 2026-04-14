"""
Groww API Client — Authenticated wrapper around growwapi SDK.
Handles token generation, session management, and reconnection.
"""

import os
import logging
import time
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from growwapi import GrowwAPI

log = logging.getLogger("josho.client")


class GrowwClient:
    """Authenticated Groww API client with auto-reconnect."""

    def __init__(self):
        self.api_key: str = os.environ["GROWW_API_KEY"]
        self.secret: str = os.environ["GROWW_SECRET_KEY"]
        self._api: Optional[GrowwAPI] = None
        self._token: Optional[str] = None
        self._token_expiry: float = 0

    def connect(self) -> GrowwAPI:
        """Authenticate and return a live GrowwAPI instance."""
        if self._api and time.time() < self._token_expiry:
            return self._api

        log.info("Authenticating with Groww API...")

        # Get access token using API Key + Secret
        try:
            self._token = GrowwAPI.get_access_token(
                api_key=self.api_key,
                secret=self.secret,
            )
            self._api = GrowwAPI(self._token)
            self._token_expiry = time.time() + 3600 * 8  # 8 hour session
            log.info("Connected to Groww API")
            return self._api
        except Exception as e:
            log.error(f"Groww auth failed: {e}")
            raise

    @property
    def api(self) -> GrowwAPI:
        """Get the authenticated API instance, reconnecting if needed."""
        return self.connect()

    # ── Account ───────────────────────────────────────────────────

    def get_profile(self) -> dict:
        """Get user profile."""
        try:
            return self.api.get_user_profile()
        except Exception as e:
            log.error(f"Failed to get profile: {e}")
            return {}

    def get_margin(self) -> dict:
        """Get available margin details."""
        try:
            return self.api.get_available_margin_details()
        except Exception as e:
            log.error(f"Failed to get margin: {e}")
            return {}

    def get_positions(self, segment: Optional[str] = None) -> dict:
        """Get all open positions."""
        try:
            return self.api.get_positions_for_user(segment=segment)
        except Exception as e:
            log.error(f"Failed to get positions: {e}")
            return {}

    def get_holdings(self) -> dict:
        """Get holdings."""
        try:
            return self.api.get_holdings_for_user()
        except Exception as e:
            log.error(f"Failed to get holdings: {e}")
            return {}

    # ── Market Data ───────────────────────────────────────────────

    def get_quote(self, symbol: str, exchange: str = "NSE", segment: str = "CASH") -> dict:
        """Get full quote for a symbol (OHLC, depth, volume, circuit limits)."""
        try:
            return self.api.get_quote(
                trading_symbol=symbol,
                exchange=exchange,
                segment=segment,
            )
        except Exception as e:
            log.error(f"Quote failed for {symbol}: {e}")
            return {}

    def get_ohlc(self, symbol: str, exchange: str = "NSE", segment: str = "CASH") -> dict:
        """Get OHLC data."""
        try:
            return self.api.get_ohlc(
                trading_symbol=symbol,
                exchange=exchange,
                segment=segment,
            )
        except Exception as e:
            log.error(f"OHLC failed for {symbol}: {e}")
            return {}

    def get_ltp_price(self, symbol: str, exchange: str = "NSE", segment: str = "CASH") -> Optional[float]:
        """Get LTP via quote (since get_ltp has tuple format)."""
        quote = self.get_quote(symbol, exchange, segment)
        return quote.get("last_price")

    def get_historical(
        self,
        symbol: str,
        exchange: str,
        segment: str,
        interval: str = "1d",
        from_date: str = "",
        to_date: str = "",
    ) -> dict:
        """Get historical candle data."""
        try:
            return self.api.get_historical_candles(
                trading_symbol=symbol,
                exchange=exchange,
                segment=segment,
                interval=interval,
                from_date=from_date,
                to_date=to_date,
            )
        except Exception as e:
            log.error(f"Historical data failed for {symbol}: {e}")
            return {}

    # ── F&O Specific ──────────────────────────────────────────────

    def get_option_chain(self, underlying: str, expiry_date: str, exchange: str = "NSE") -> dict:
        """Get option chain for an underlying (NIFTY, BANKNIFTY, etc)."""
        try:
            return self.api.get_option_chain(
                exchange=exchange,
                underlying=underlying,
                expiry_date=expiry_date,
            )
        except Exception as e:
            log.error(f"Option chain failed for {underlying}: {e}")
            return {}

    def get_expiries(self, underlying: str, exchange: str = "NSE") -> dict:
        """Get available expiry dates for an underlying."""
        try:
            return self.api.get_expiries(
                exchange=exchange,
                underlying_symbol=underlying,
            )
        except Exception as e:
            log.error(f"Expiries failed for {underlying}: {e}")
            return {}

    def get_greeks(self, symbol: str, exchange: str = "NSE") -> dict:
        """Get option greeks (delta, gamma, theta, vega, IV)."""
        try:
            return self.api.get_greeks(
                trading_symbol=symbol,
                exchange=exchange,
            )
        except Exception as e:
            log.error(f"Greeks failed for {symbol}: {e}")
            return {}

    # ── Orders ────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        qty: int,
        side: str,  # "BUY" or "SELL"
        order_type: str = "MARKET",
        price: float = 0,
        trigger_price: float = None,
        product: str = "NRML",
        segment: str = "FNO",
        exchange: str = "NSE",
        validity: str = "DAY",
    ) -> dict:
        """Place an order (equity or F&O)."""
        try:
            return self.api.place_order(
                trading_symbol=symbol,
                quantity=qty,
                validity=validity,
                exchange=exchange,
                segment=segment,
                product=product,
                order_type=order_type,
                transaction_type=side,
                price=price,
                trigger_price=trigger_price,
            )
        except Exception as e:
            log.error(f"Order failed: {side} {qty}x {symbol} @ {price}: {e}")
            return {"status": "FAILED", "error": str(e)}

    def modify_order(self, order_id: str, **kwargs) -> dict:
        """Modify an existing order."""
        try:
            return self.api.modify_order(groww_order_id=order_id, **kwargs)
        except Exception as e:
            log.error(f"Modify failed for {order_id}: {e}")
            return {"status": "FAILED", "error": str(e)}

    def cancel_order(self, order_id: str, segment: str = "FNO") -> dict:
        """Cancel an order."""
        try:
            return self.api.cancel_order(groww_order_id=order_id, segment=segment)
        except Exception as e:
            log.error(f"Cancel failed for {order_id}: {e}")
            return {"status": "FAILED", "error": str(e)}

    def get_orders(self) -> dict:
        """Get all orders for today."""
        try:
            return self.api.get_order_list()
        except Exception as e:
            log.error(f"Order list failed: {e}")
            return {}

    def get_order_status(self, order_id: str) -> dict:
        """Get status of a specific order."""
        try:
            return self.api.get_order_status(groww_order_id=order_id)
        except Exception as e:
            log.error(f"Order status failed for {order_id}: {e}")
            return {}

    # ── WebSocket ─────────────────────────────────────────────────

    def get_socket_token(self) -> str:
        """Get token for WebSocket live feed connection."""
        try:
            result = self.api.generate_socket_token()
            return result.get("token", "")
        except Exception as e:
            log.error(f"Socket token failed: {e}")
            return ""


# Singleton
_client: Optional[GrowwClient] = None


def get_client() -> GrowwClient:
    global _client
    if _client is None:
        _client = GrowwClient()
    return _client

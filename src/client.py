"""
Groww API Client — Authenticated wrapper around growwapi SDK.
Handles token generation, session management, and reconnection.
"""

import os
import hashlib
import time
import logging
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

        try:
            # Method 1: API Key + Secret (approval-based)
            timestamp = str(int(time.time()))
            checksum = hashlib.sha256(
                f"{self.api_key}{timestamp}{self.secret}".encode()
            ).hexdigest()

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

    # ── Convenience Methods ───────────────────────────────────────

    def get_positions(self) -> dict:
        """Get all open positions."""
        try:
            return self.api.get_positions()
        except Exception as e:
            log.error(f"Failed to get positions: {e}")
            return {}

    def get_holdings(self) -> dict:
        """Get holdings."""
        try:
            return self.api.get_holdings()
        except Exception as e:
            log.error(f"Failed to get holdings: {e}")
            return {}

    def get_ltp(self, symbol: str, exchange: str = "NSE", segment: str = "FNO") -> Optional[float]:
        """Get last traded price for a symbol."""
        try:
            data = self.api.get_ltp(
                trading_symbol=symbol,
                exchange=exchange,
                segment=segment,
            )
            return data.get("payload", {}).get("ltp")
        except Exception as e:
            log.error(f"LTP fetch failed for {symbol}: {e}")
            return None

    def get_quote(self, symbol: str, exchange: str = "NSE", segment: str = "FNO") -> dict:
        """Get full quote for a symbol."""
        try:
            return self.api.get_quote(
                trading_symbol=symbol,
                exchange=exchange,
                segment=segment,
            )
        except Exception as e:
            log.error(f"Quote fetch failed for {symbol}: {e}")
            return {}

    def place_order(
        self,
        symbol: str,
        qty: int,
        side: str,  # "BUY" or "SELL"
        order_type: str = "MARKET",
        price: float = 0,
        product: str = "NRML",
        segment: str = "FNO",
        exchange: str = "NSE",
    ) -> dict:
        """Place an F&O order."""
        try:
            return self.api.place_order(
                trading_symbol=symbol,
                quantity=qty,
                validity="DAY",
                exchange=exchange,
                segment=segment,
                product=product,
                order_type=order_type,
                transaction_type=side,
                price=price,
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


# Singleton
_client: Optional[GrowwClient] = None


def get_client() -> GrowwClient:
    global _client
    if _client is None:
        _client = GrowwClient()
    return _client

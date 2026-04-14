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

        # Try TOTP flow first (the JWT token contains role=auth-totp)
        try:
            import pyotp
            totp_gen = pyotp.TOTP(self.secret)
            totp_code = totp_gen.now()
            log.info(f"Using TOTP auth (code: {totp_code})")

            self._token = GrowwAPI.get_access_token(
                api_key=self.api_key,
                totp=totp_code,
            )
            self._api = GrowwAPI(self._token)
            self._token_expiry = time.time() + 3600 * 8
            log.info("Connected to Groww API via TOTP")
            return self._api
        except Exception as e1:
            log.warning(f"TOTP auth failed: {e1}")

        # Fallback: API Key + Secret (approval-based)
        try:
            self._token = GrowwAPI.get_access_token(
                api_key=self.api_key,
                secret=self.secret,
            )
            self._api = GrowwAPI(self._token)
            self._token_expiry = time.time() + 3600 * 8
            log.info("Connected to Groww API via API Key+Secret")
            return self._api
        except Exception as e2:
            log.error(f"API Key+Secret auth also failed: {e2}")

        # Last resort: use the API key directly as access token
        # (if it's already a valid JWT session token)
        try:
            log.info("Attempting direct token usage...")
            self._api = GrowwAPI(self.api_key)
            self._token = self.api_key
            self._token_expiry = time.time() + 3600 * 8
            # Test with a simple call
            self._api.get_positions_for_user()
            log.info("Connected to Groww API via direct token")
            return self._api
        except Exception as e3:
            log.error(f"Direct token also failed: {e3}")
            raise RuntimeError(
                "All auth methods failed. "
                "Go to groww.in/trade-api/api-keys and approve today's session, "
                "or regenerate your API key."
            )

    @property
    def api(self) -> GrowwAPI:
        """Get the authenticated API instance, reconnecting if needed."""
        return self.connect()

    # ── Convenience Methods ───────────────────────────────────────

    def get_positions(self) -> dict:
        """Get all open positions."""
        try:
            return self.api.get_positions_for_user()
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

    def get_profile(self) -> dict:
        """Get user profile."""
        try:
            return self.api.get_user_profile()
        except Exception as e:
            log.error(f"Failed to get profile: {e}")
            return {}

    def get_option_chain(self, symbol: str, expiry: str) -> dict:
        """Get option chain from Groww API."""
        try:
            return self.api.get_option_chain(
                trading_symbol=symbol,
                expiry_date=expiry,
            )
        except Exception as e:
            log.error(f"Option chain failed for {symbol}: {e}")
            return {}

    def get_expiries(self, symbol: str) -> dict:
        """Get available expiry dates."""
        try:
            return self.api.get_expiries(trading_symbol=symbol)
        except Exception as e:
            log.error(f"Expiries failed for {symbol}: {e}")
            return {}

    def get_greeks(self, symbol: str, exchange: str = "NSE") -> dict:
        """Get option greeks."""
        try:
            return self.api.get_greeks(
                trading_symbol=symbol,
                exchange=exchange,
            )
        except Exception as e:
            log.error(f"Greeks failed for {symbol}: {e}")
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

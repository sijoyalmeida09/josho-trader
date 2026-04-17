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
    """Authenticated Groww API client with token caching + auto-reconnect."""

    TOKEN_CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", ".groww_token_cache")

    def __init__(self):
        self.api_key: str = os.environ["GROWW_API_KEY"]
        self.secret: str = os.environ["GROWW_SECRET_KEY"]
        self.totp_secret: str = os.environ.get("GROWW_TOTP_SECRET", "")
        self.totp_token: str = os.environ.get("GROWW_TOTP_TOKEN", "")
        self._api: Optional[GrowwAPI] = None
        self._token: Optional[str] = None
        self._token_expiry: float = 0
        self._load_cached_token()

    def _load_cached_token(self):
        """Load token from disk cache to avoid re-auth on restart."""
        try:
            import json
            if os.path.exists(self.TOKEN_CACHE_FILE):
                data = json.loads(open(self.TOKEN_CACHE_FILE).read())
                if data.get("expiry", 0) > time.time() + 300:  # valid for at least 5 more min
                    self._token = data["token"]
                    self._token_expiry = data["expiry"]
                    log.info(f"Loaded cached token (expires in {int((data['expiry'] - time.time()) / 60)} min)")
        except Exception:
            pass

    def _save_token_cache(self):
        """Save token to disk so restarts don't need re-auth."""
        try:
            import json
            os.makedirs(os.path.dirname(self.TOKEN_CACHE_FILE), exist_ok=True)
            with open(self.TOKEN_CACHE_FILE, "w") as f:
                json.dump({"token": self._token, "expiry": self._token_expiry}, f)
        except Exception:
            pass

    def connect(self) -> GrowwAPI:
        """Authenticate and return a live GrowwAPI instance.
        Uses cached token if available — ZERO API calls on restart.
        """
        # Reuse cached token — NO API call needed
        if self._token and time.time() < self._token_expiry:
            if not self._api:
                self._api = GrowwAPI(self._token)
                log.info("Connected using cached token (no auth call)")
            return self._api

        max_retries = 5
        base_delay = 10  # seconds (was 5)
        max_delay = 300

        for attempt in range(1, max_retries + 1):
            try:
                return self._attempt_connect()
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "rate" in error_str or "429" in error_str or "too many" in error_str or "throttl" in error_str

                if attempt == max_retries:
                    log.error(f"All connect attempts exhausted after {max_retries} retries: {e}")
                    raise

                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                if is_rate_limit:
                    delay = min(delay * 2, max_delay)
                    log.warning(f"Rate limited (attempt {attempt}/{max_retries}). Backing off {delay}s...")
                else:
                    log.warning(f"Connect failed (attempt {attempt}/{max_retries}): {e}. Retrying in {delay}s...")

                time.sleep(delay)

        raise RuntimeError("Failed to connect after all retries")

    def _attempt_connect(self) -> GrowwAPI:
        """Single connection attempt using available auth methods."""
        log.info("Authenticating with Groww API...")

        # Method 1: TOTP (fully automated, no daily approval needed)
        if self.totp_secret and self.totp_token:
            try:
                import pyotp
                totp_gen = pyotp.TOTP(self.totp_secret)
                totp_code = totp_gen.now()
                log.info(f"TOTP auth (code: {totp_code})")

                self._token = GrowwAPI.get_access_token(
                    api_key=self.totp_token,
                    totp=totp_code,
                )
                self._api = GrowwAPI(self._token)
                self._token_expiry = time.time() + 3600 * 4  # refresh every 4h for safety
                self._save_token_cache()
                log.info("Connected via TOTP (auto-daily) — token cached")
                return self._api
            except Exception as e:
                log.warning(f"TOTP auth failed: {e}")

        # Method 2: API Key + Secret (needs manual daily approval)
        self._token = GrowwAPI.get_access_token(
            api_key=self.api_key,
            secret=self.secret,
        )
        self._api = GrowwAPI(self._token)
        self._token_expiry = time.time() + 3600 * 4  # refresh every 4h for safety
        self._save_token_cache()
        log.info("Connected via API Key + Secret — token cached")
        return self._api

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

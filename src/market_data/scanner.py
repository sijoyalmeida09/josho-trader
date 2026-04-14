"""
Market Data Scanner — Real-time market analysis for F&O opportunities.
Scans NIFTY, BANKNIFTY, top stocks for trading setups.
"""

import logging
from datetime import datetime, time
from typing import Optional

from ..client import get_client

log = logging.getLogger("josho.scanner")


# F&O watchlist — high-volume, liquid stocks
FNO_WATCHLIST = [
    "NIFTY", "BANKNIFTY", "FINNIFTY",
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
    "HINDUNILVR", "MARUTI", "TATAMOTORS", "TATASTEEL",
    "AXISBANK", "BAJFINANCE", "WIPRO", "HCLTECH",
    "ADANIENT", "ADANIPORTS", "TITAN", "ULTRACEMCO",
]

# NIFTY option chain strike range
NIFTY_STRIKE_STEP = 50
BANKNIFTY_STRIKE_STEP = 100


class MarketScanner:
    """Scan market for F&O trading opportunities."""

    def __init__(self):
        self.client = get_client()

    def is_market_open(self) -> bool:
        """Check if Indian market is currently open (9:15 AM - 3:30 PM IST)."""
        now = datetime.utcnow()
        # IST = UTC + 5:30
        ist_hour = (now.hour + 5) % 24
        ist_min = now.minute + 30
        if ist_min >= 60:
            ist_hour += 1
            ist_min -= 60

        market_open = time(9, 15)
        market_close = time(15, 30)
        current = time(ist_hour, ist_min)

        return market_open <= current <= market_close

    def get_nifty_spot(self) -> Optional[float]:
        """Get NIFTY 50 spot price."""
        return self.client.get_ltp("NIFTY 50", exchange="NSE", segment="FNO")

    def get_banknifty_spot(self) -> Optional[float]:
        """Get BANK NIFTY spot price."""
        return self.client.get_ltp("NIFTY BANK", exchange="NSE", segment="FNO")

    def get_option_chain(self, symbol: str, expiry: str) -> dict:
        """
        Get option chain for a symbol.
        symbol: "NIFTY" or "BANKNIFTY"
        expiry: "2026-04-17" format
        """
        try:
            # Get ATM strike
            spot = self.get_nifty_spot() if "NIFTY" in symbol and "BANK" not in symbol else self.get_banknifty_spot()
            if not spot:
                return {}

            step = NIFTY_STRIKE_STEP if "BANK" not in symbol else BANKNIFTY_STRIKE_STEP
            atm = round(spot / step) * step

            # Generate strikes around ATM (5 above, 5 below)
            strikes = [atm + (i * step) for i in range(-5, 6)]

            chain = {"spot": spot, "atm": atm, "calls": {}, "puts": {}}

            for strike in strikes:
                # CE symbol format: NIFTY2641724050CE
                ce_symbol = f"{symbol}{expiry.replace('-', '')[2:]}{strike}CE"
                pe_symbol = f"{symbol}{expiry.replace('-', '')[2:]}{strike}PE"

                ce_ltp = self.client.get_ltp(ce_symbol, segment="FNO")
                pe_ltp = self.client.get_ltp(pe_symbol, segment="FNO")

                if ce_ltp is not None:
                    chain["calls"][strike] = {"ltp": ce_ltp, "symbol": ce_symbol}
                if pe_ltp is not None:
                    chain["puts"][strike] = {"ltp": pe_ltp, "symbol": pe_symbol}

            return chain

        except Exception as e:
            log.error(f"Option chain failed for {symbol}: {e}")
            return {}

    def scan_momentum(self) -> list[dict]:
        """Scan FNO stocks for momentum signals."""
        signals = []
        for symbol in FNO_WATCHLIST[:15]:  # Top 15 to stay within rate limits
            try:
                quote = self.client.get_quote(symbol, segment="CASH")
                payload = quote.get("payload", {})

                ltp = payload.get("ltp", 0)
                open_price = payload.get("open", 0)
                high = payload.get("high", 0)
                low = payload.get("low", 0)
                prev_close = payload.get("close", 0)

                if not all([ltp, open_price, prev_close]):
                    continue

                change_pct = ((ltp - prev_close) / prev_close) * 100 if prev_close else 0
                day_range = ((high - low) / low) * 100 if low else 0
                from_high = ((high - ltp) / high) * 100 if high else 0

                # Momentum signal: strong move (>1.5%) with range expansion
                if abs(change_pct) > 1.5 and day_range > 2:
                    signals.append({
                        "symbol": symbol,
                        "ltp": ltp,
                        "change_pct": round(change_pct, 2),
                        "day_range_pct": round(day_range, 2),
                        "from_high_pct": round(from_high, 2),
                        "direction": "BULLISH" if change_pct > 0 else "BEARISH",
                        "strength": "STRONG" if abs(change_pct) > 3 else "MODERATE",
                    })

            except Exception as e:
                log.debug(f"Scan failed for {symbol}: {e}")
                continue

        signals.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
        return signals

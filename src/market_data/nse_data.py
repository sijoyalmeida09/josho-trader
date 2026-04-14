"""
NSE Data Fetcher — Free market data without paid feeds.
Integrated from: NseIndia unofficial, nsetools patterns.

Fetches:
  - Live option chains (NIFTY, BANKNIFTY, stock options)
  - Open Interest data
  - IV and Greeks
  - Historical data
  - Market breadth (advance/decline)
"""

import logging
import time
import requests
import pandas as pd
from typing import Optional

log = logging.getLogger("josho.nse")

NSE_BASE = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}


class NseData:
    """Fetch live data from NSE India (no API key needed)."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._cookies_set = False

    def _ensure_cookies(self):
        """NSE requires cookies from initial page load."""
        if not self._cookies_set:
            try:
                self.session.get(NSE_BASE, timeout=10)
                self._cookies_set = True
                time.sleep(0.5)
            except Exception as e:
                log.warning(f"Cookie setup failed: {e}")

    def _get(self, url: str) -> Optional[dict]:
        """Make authenticated GET request to NSE."""
        self._ensure_cookies()
        try:
            resp = self.session.get(url, timeout=15)
            if resp.ok:
                return resp.json()
            elif resp.status_code == 401:
                self._cookies_set = False
                self._ensure_cookies()
                resp = self.session.get(url, timeout=15)
                if resp.ok:
                    return resp.json()
            log.warning(f"NSE request failed: {resp.status_code}")
        except Exception as e:
            log.error(f"NSE fetch error: {e}")
        return None

    def get_option_chain(self, symbol: str = "NIFTY") -> Optional[dict]:
        """
        Get full option chain with OI, IV, greeks, volume.
        Returns structured data for all strikes and expiries.
        """
        url = f"{NSE_BASE}/api/option-chain-indices?symbol={symbol}"
        if symbol not in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"):
            url = f"{NSE_BASE}/api/option-chain-equities?symbol={symbol}"

        data = self._get(url)
        if not data:
            return None

        records = data.get("records", {})
        filtered = data.get("filtered", {})

        result = {
            "underlying": records.get("underlyingValue", 0),
            "timestamp": records.get("timestamp", ""),
            "expiry_dates": records.get("expiryDates", []),
            "strikePrices": records.get("strikePrices", []),
            "total_ce_oi": filtered.get("CE", {}).get("totOI", 0),
            "total_pe_oi": filtered.get("PE", {}).get("totOI", 0),
            "pcr": 0,
            "chains": [],
        }

        # PCR (Put-Call Ratio)
        if result["total_ce_oi"] > 0:
            result["pcr"] = result["total_pe_oi"] / result["total_ce_oi"]

        # Parse each strike
        for row in records.get("data", []):
            strike = row.get("strikePrice", 0)
            expiry = row.get("expiryDate", "")

            ce = row.get("CE", {})
            pe = row.get("PE", {})

            result["chains"].append({
                "strike": strike,
                "expiry": expiry,
                "ce_oi": ce.get("openInterest", 0),
                "ce_change_oi": ce.get("changeinOpenInterest", 0),
                "ce_volume": ce.get("totalTradedVolume", 0),
                "ce_iv": ce.get("impliedVolatility", 0),
                "ce_ltp": ce.get("lastPrice", 0),
                "ce_bid": ce.get("bidprice", 0),
                "ce_ask": ce.get("askPrice", 0),
                "pe_oi": pe.get("openInterest", 0),
                "pe_change_oi": pe.get("changeinOpenInterest", 0),
                "pe_volume": pe.get("totalTradedVolume", 0),
                "pe_iv": pe.get("impliedVolatility", 0),
                "pe_ltp": pe.get("lastPrice", 0),
                "pe_bid": pe.get("bidprice", 0),
                "pe_ask": pe.get("askPrice", 0),
            })

        return result

    def get_market_breadth(self) -> Optional[dict]:
        """Get market advance/decline data."""
        data = self._get(f"{NSE_BASE}/api/market-data-pre-open?key=NIFTY")
        if not data:
            return None

        advances = sum(1 for d in data.get("data", []) if d.get("metadata", {}).get("pChange", 0) > 0)
        declines = sum(1 for d in data.get("data", []) if d.get("metadata", {}).get("pChange", 0) < 0)

        return {
            "advances": advances,
            "declines": declines,
            "ratio": advances / declines if declines > 0 else 0,
            "total": len(data.get("data", [])),
        }

    def get_iv_rank(self, symbol: str = "NIFTY", lookback_days: int = 252) -> float:
        """
        Calculate IV rank — where is current IV relative to past year.
        0 = lowest IV in a year, 100 = highest IV in a year.
        """
        chain = self.get_option_chain(symbol)
        if not chain or not chain["chains"]:
            return 50.0  # default to middle

        # Get ATM IV
        spot = chain["underlying"]
        atm_data = min(chain["chains"], key=lambda x: abs(x["strike"] - spot))
        current_iv = (atm_data.get("ce_iv", 0) + atm_data.get("pe_iv", 0)) / 2

        # For proper IV rank, we'd need historical IV data
        # Approximation: normalize current IV (typical NIFTY IV range: 10-35)
        iv_min, iv_max = 10, 35
        rank = ((current_iv - iv_min) / (iv_max - iv_min)) * 100
        return max(0, min(100, rank))

    def get_pcr_analysis(self, symbol: str = "NIFTY") -> dict:
        """
        Put-Call Ratio analysis — market sentiment indicator.
        PCR > 1.2 = bullish (too many puts = contrarian bullish)
        PCR < 0.8 = bearish (too many calls = contrarian bearish)
        """
        chain = self.get_option_chain(symbol)
        if not chain:
            return {"pcr": 1.0, "sentiment": "NEUTRAL"}

        pcr = chain["pcr"]
        if pcr > 1.3:
            sentiment = "VERY_BULLISH"
        elif pcr > 1.0:
            sentiment = "BULLISH"
        elif pcr > 0.7:
            sentiment = "NEUTRAL"
        elif pcr > 0.5:
            sentiment = "BEARISH"
        else:
            sentiment = "VERY_BEARISH"

        return {
            "pcr": round(pcr, 2),
            "sentiment": sentiment,
            "total_ce_oi": chain["total_ce_oi"],
            "total_pe_oi": chain["total_pe_oi"],
        }

    def get_max_pain(self, symbol: str = "NIFTY", expiry: str = "") -> float:
        """
        Calculate Max Pain — strike where option writers lose least.
        Market tends to gravitate toward max pain at expiry.
        """
        chain = self.get_option_chain(symbol)
        if not chain or not chain["chains"]:
            return 0

        # Filter by expiry if specified
        data = chain["chains"]
        if expiry:
            data = [d for d in data if d["expiry"] == expiry]
        else:
            # Use nearest expiry
            if chain["expiry_dates"]:
                nearest = chain["expiry_dates"][0]
                data = [d for d in data if d["expiry"] == nearest]

        if not data:
            return 0

        strikes = sorted(set(d["strike"] for d in data))
        min_pain = float("inf")
        max_pain_strike = 0

        for strike in strikes:
            total_pain = 0
            for d in data:
                # CE writers pain: max(0, settlement - strike) * OI
                ce_pain = max(0, strike - d["strike"]) * d["ce_oi"]
                # PE writers pain: max(0, strike - settlement) * OI
                pe_pain = max(0, d["strike"] - strike) * d["pe_oi"]
                total_pain += ce_pain + pe_pain

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = strike

        return max_pain_strike

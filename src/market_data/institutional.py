"""
FII/DII Flow Data — Institutional money flow tracking from NSE.

Tracks Foreign Institutional Investor (FII) and Domestic Institutional
Investor (DII) net buy/sell activity in cash and F&O segments.

Data source: NSE India website (free, no API key required).

FII/DII flows are the single strongest signal for medium-term market direction:
  - Sustained FII buying → market rally
  - FII selling + DII buying → choppy, range-bound
  - Both selling → correction incoming
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger("josho.institutional")

NSE_BASE = "https://www.nseindia.com"
NSDL_BASE = "https://www.fpi.nsdl.co.in"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}


@dataclass(frozen=True)
class DailyFlow:
    """Single day's FII/DII flow data (immutable)."""
    date: str
    fii_buy: float  # FII gross buy (crores)
    fii_sell: float  # FII gross sell (crores)
    fii_net: float  # FII net (buy - sell)
    dii_buy: float  # DII gross buy (crores)
    dii_sell: float  # DII gross sell (crores)
    dii_net: float  # DII net (buy - sell)
    segment: str  # "CASH" or "FO"


@dataclass(frozen=True)
class FlowSentiment:
    """Aggregated institutional sentiment score."""
    score: float  # -1 (very bearish) to +1 (very bullish)
    label: str  # VERY_BULLISH, BULLISH, NEUTRAL, BEARISH, VERY_BEARISH
    fii_trend: str  # NET_BUYER, NET_SELLER, NEUTRAL
    dii_trend: str  # NET_BUYER, NET_SELLER, NEUTRAL
    fii_net_30d: float  # Total FII net over 30 days (crores)
    dii_net_30d: float  # Total DII net over 30 days (crores)
    fii_avg_daily: float  # Average daily FII net
    dii_avg_daily: float  # Average daily DII net
    days_analyzed: int


class InstitutionalFlows:
    """
    Fetch and analyze FII/DII activity from NSE.

    NSE publishes daily FII/DII participation data at:
    https://www.nseindia.com/api/fiidiiTradeReact
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._cookies_set = False
        self._cache: list[DailyFlow] = []
        self._last_fetch: Optional[datetime] = None

    def _ensure_cookies(self):
        """NSE requires cookies from initial page load."""
        if not self._cookies_set:
            try:
                self.session.get(NSE_BASE, timeout=10)
                self._cookies_set = True
                time.sleep(0.5)
            except Exception as e:
                log.warning(f"Cookie setup failed: {e}")

    def _get(self, url: str) -> Optional[dict | list]:
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
            log.warning(f"NSE request failed: {resp.status_code} for {url}")
        except Exception as e:
            log.error(f"NSE fetch error: {e}")
        return None

    def fetch_daily(self) -> list[DailyFlow]:
        """
        Fetch today's FII/DII data from NSE.
        Returns list of DailyFlow for cash and F&O segments.
        """
        url = f"{NSE_BASE}/api/fiidiiTradeReact"
        data = self._get(url)

        if not data:
            log.warning("No FII/DII data returned from NSE")
            return []

        flows = []

        # NSE returns a list of dicts with category, buyValue, sellValue, netValue
        if isinstance(data, list):
            for entry in data:
                category = entry.get("category", "")
                date_str = entry.get("date", datetime.now().strftime("%d-%b-%Y"))
                buy_val = self._parse_crore(entry.get("buyValue", "0"))
                sell_val = self._parse_crore(entry.get("sellValue", "0"))
                net_val = self._parse_crore(entry.get("netValue", "0"))

                if "FII" in category.upper() or "FPI" in category.upper():
                    flows.append(DailyFlow(
                        date=date_str,
                        fii_buy=buy_val,
                        fii_sell=sell_val,
                        fii_net=net_val,
                        dii_buy=0,
                        dii_sell=0,
                        dii_net=0,
                        segment="CASH",
                    ))
                elif "DII" in category.upper():
                    flows.append(DailyFlow(
                        date=date_str,
                        fii_buy=0,
                        fii_sell=0,
                        fii_net=0,
                        dii_buy=buy_val,
                        dii_sell=sell_val,
                        dii_net=net_val,
                        segment="CASH",
                    ))

        if flows:
            self._last_fetch = datetime.now()

        log.info(f"FII/DII daily: {len(flows)} entries fetched")
        return flows

    def fetch_historical(self, days: int = 30) -> list[DailyFlow]:
        """
        Fetch historical FII/DII data.

        NSE provides historical data at:
        /api/fiidiiTradeReact with date parameters.

        Falls back to NSDL FPI data if NSE historical is unavailable.

        Args:
            days: Number of days of history to fetch.

        Returns:
            List of DailyFlow, sorted by date ascending.
        """
        all_flows = []

        # Try NSE historical endpoint
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # NSE provides a bulk download endpoint
        url = (
            f"{NSE_BASE}/api/fiidiiTradeReact?"
            f"startDate={start_date.strftime('%d-%m-%Y')}&"
            f"endDate={end_date.strftime('%d-%m-%Y')}"
        )
        data = self._get(url)

        if data and isinstance(data, list) and len(data) > 1:
            all_flows = self._parse_nse_historical(data)
        else:
            # Fallback: fetch day-by-day from participant-wise OI data
            log.info("NSE historical unavailable, trying participant-wise data")
            all_flows = self._fetch_participant_data()

        # If still empty, try the basic endpoint (returns today only, but cache it)
        if not all_flows:
            today_flows = self.fetch_daily()
            all_flows = today_flows

        self._cache = sorted(all_flows, key=lambda f: f.date)
        return self._cache

    def get_sentiment(self, days: int = 30) -> FlowSentiment:
        """
        Aggregate FII/DII flows into a sentiment score.

        Scoring logic:
          - FII net positive → bullish signal (weight: 0.6)
          - DII net positive → mildly bullish signal (weight: 0.4)
          - Both positive → strong bullish
          - FII selling, DII buying → neutral (institutional rotation)
          - Both selling → strong bearish

        Args:
            days: Lookback period in days.

        Returns:
            FlowSentiment with score, label, and detailed breakdown.
        """
        flows = self._cache if self._cache else self.fetch_historical(days)

        if not flows:
            return FlowSentiment(
                score=0.0,
                label="NEUTRAL",
                fii_trend="NEUTRAL",
                dii_trend="NEUTRAL",
                fii_net_30d=0.0,
                dii_net_30d=0.0,
                fii_avg_daily=0.0,
                dii_avg_daily=0.0,
                days_analyzed=0,
            )

        # Aggregate FII and DII nets
        fii_total = sum(f.fii_net for f in flows)
        dii_total = sum(f.dii_net for f in flows)
        days_count = max(len(set(f.date for f in flows)), 1)

        fii_avg = fii_total / days_count
        dii_avg = dii_total / days_count

        # Normalize to -1..+1 score
        # Typical daily FII range: -5000 to +5000 crores
        fii_norm = max(-1.0, min(1.0, fii_avg / 3000))
        dii_norm = max(-1.0, min(1.0, dii_avg / 3000))

        # Weighted composite: FII matters more for direction
        score = (fii_norm * 0.6) + (dii_norm * 0.4)

        # Classify trends
        fii_trend = (
            "NET_BUYER" if fii_total > 500
            else "NET_SELLER" if fii_total < -500
            else "NEUTRAL"
        )
        dii_trend = (
            "NET_BUYER" if dii_total > 500
            else "NET_SELLER" if dii_total < -500
            else "NEUTRAL"
        )

        # Label
        if score > 0.3:
            label = "VERY_BULLISH"
        elif score > 0.1:
            label = "BULLISH"
        elif score > -0.1:
            label = "NEUTRAL"
        elif score > -0.3:
            label = "BEARISH"
        else:
            label = "VERY_BEARISH"

        return FlowSentiment(
            score=round(score, 4),
            label=label,
            fii_trend=fii_trend,
            dii_trend=dii_trend,
            fii_net_30d=round(fii_total, 2),
            dii_net_30d=round(dii_total, 2),
            fii_avg_daily=round(fii_avg, 2),
            dii_avg_daily=round(dii_avg, 2),
            days_analyzed=days_count,
        )

    def get_recent_flows_df(self, days: int = 30) -> pd.DataFrame:
        """
        Get FII/DII flows as a DataFrame for analysis/charting.

        Returns:
            DataFrame with columns: date, fii_net, dii_net, total_net, cumulative_fii, cumulative_dii.
        """
        flows = self._cache if self._cache else self.fetch_historical(days)

        if not flows:
            return pd.DataFrame()

        # Merge FII and DII entries for the same date
        merged = {}
        for f in flows:
            if f.date not in merged:
                merged[f.date] = {
                    "date": f.date,
                    "fii_buy": 0, "fii_sell": 0, "fii_net": 0,
                    "dii_buy": 0, "dii_sell": 0, "dii_net": 0,
                }
            entry = merged[f.date]
            record = {
                **entry,
                "fii_buy": entry["fii_buy"] + f.fii_buy,
                "fii_sell": entry["fii_sell"] + f.fii_sell,
                "fii_net": entry["fii_net"] + f.fii_net,
                "dii_buy": entry["dii_buy"] + f.dii_buy,
                "dii_sell": entry["dii_sell"] + f.dii_sell,
                "dii_net": entry["dii_net"] + f.dii_net,
            }
            merged[f.date] = record

        rows = sorted(merged.values(), key=lambda r: r["date"])
        df = pd.DataFrame(rows)

        if not df.empty:
            df["total_net"] = df["fii_net"] + df["dii_net"]
            df["cumulative_fii"] = df["fii_net"].cumsum()
            df["cumulative_dii"] = df["dii_net"].cumsum()
            df["cumulative_total"] = df["total_net"].cumsum()

        return df

    def detect_flow_shift(self, lookback: int = 10) -> dict:
        """
        Detect regime shifts in institutional flows.

        Looks for:
          - FII turning from buyer to seller (or vice versa)
          - DII stepping in to absorb FII selling
          - Acceleration/deceleration in flow magnitude

        Args:
            lookback: Number of recent days to analyze.

        Returns:
            Dict with shift detection results.
        """
        df = self.get_recent_flows_df()

        if df.empty or len(df) < lookback:
            return {
                "shift_detected": False,
                "message": "Insufficient data for shift detection",
            }

        recent = df.tail(lookback)
        older = df.tail(lookback * 2).head(lookback) if len(df) >= lookback * 2 else df.head(lookback)

        recent_fii_avg = recent["fii_net"].mean() if "fii_net" in recent.columns else 0
        older_fii_avg = older["fii_net"].mean() if "fii_net" in older.columns else 0
        recent_dii_avg = recent["dii_net"].mean() if "dii_net" in recent.columns else 0
        older_dii_avg = older["dii_net"].mean() if "dii_net" in older.columns else 0

        fii_direction_change = (recent_fii_avg > 0) != (older_fii_avg > 0)
        dii_direction_change = (recent_dii_avg > 0) != (older_dii_avg > 0)

        # Magnitude change
        fii_magnitude_ratio = abs(recent_fii_avg) / max(abs(older_fii_avg), 1)
        dii_magnitude_ratio = abs(recent_dii_avg) / max(abs(older_dii_avg), 1)

        shifts = []

        if fii_direction_change:
            direction = "BUYING" if recent_fii_avg > 0 else "SELLING"
            shifts.append(f"FII shifted to {direction} (avg: {recent_fii_avg:.0f} Cr/day)")

        if dii_direction_change:
            direction = "BUYING" if recent_dii_avg > 0 else "SELLING"
            shifts.append(f"DII shifted to {direction} (avg: {recent_dii_avg:.0f} Cr/day)")

        if fii_magnitude_ratio > 2.0:
            shifts.append(f"FII flow accelerated {fii_magnitude_ratio:.1f}x")

        # DII absorbing FII selling pattern
        if recent_fii_avg < -500 and recent_dii_avg > 500:
            shifts.append("DII absorbing FII selling — market support in place")

        return {
            "shift_detected": len(shifts) > 0,
            "shifts": shifts,
            "recent_fii_avg": round(recent_fii_avg, 2),
            "recent_dii_avg": round(recent_dii_avg, 2),
            "older_fii_avg": round(older_fii_avg, 2),
            "older_dii_avg": round(older_dii_avg, 2),
            "fii_magnitude_ratio": round(fii_magnitude_ratio, 2),
            "dii_magnitude_ratio": round(dii_magnitude_ratio, 2),
        }

    def summary(self) -> dict:
        """Quick summary of institutional flow state."""
        sentiment = self.get_sentiment()
        shift = self.detect_flow_shift()

        return {
            "sentiment": {
                "score": sentiment.score,
                "label": sentiment.label,
                "fii_trend": sentiment.fii_trend,
                "dii_trend": sentiment.dii_trend,
            },
            "totals": {
                "fii_net_30d": sentiment.fii_net_30d,
                "dii_net_30d": sentiment.dii_net_30d,
                "fii_avg_daily": sentiment.fii_avg_daily,
                "dii_avg_daily": sentiment.dii_avg_daily,
            },
            "shift": shift,
            "days_analyzed": sentiment.days_analyzed,
            "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None,
        }

    # ── Private helpers ──────────────────────────────────────────────

    @staticmethod
    def _parse_crore(value) -> float:
        """Parse a crore value from NSE response (could be string or number)."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.replace(",", "").replace(" ", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                return 0.0
        return 0.0

    def _parse_nse_historical(self, data: list) -> list[DailyFlow]:
        """Parse NSE historical FII/DII response into DailyFlow list."""
        flows = []

        for entry in data:
            category = entry.get("category", "")
            date_str = entry.get("date", "")
            buy_val = self._parse_crore(entry.get("buyValue", 0))
            sell_val = self._parse_crore(entry.get("sellValue", 0))
            net_val = self._parse_crore(entry.get("netValue", 0))

            is_fii = "FII" in category.upper() or "FPI" in category.upper()
            is_dii = "DII" in category.upper()

            if is_fii:
                flows.append(DailyFlow(
                    date=date_str,
                    fii_buy=buy_val,
                    fii_sell=sell_val,
                    fii_net=net_val,
                    dii_buy=0,
                    dii_sell=0,
                    dii_net=0,
                    segment="CASH",
                ))
            elif is_dii:
                flows.append(DailyFlow(
                    date=date_str,
                    fii_buy=0,
                    fii_sell=0,
                    fii_net=0,
                    dii_buy=buy_val,
                    dii_sell=sell_val,
                    dii_net=net_val,
                    segment="CASH",
                ))

        return flows

    def _fetch_participant_data(self) -> list[DailyFlow]:
        """
        Fetch participant-wise trading data from NSE.
        This gives FII/DII activity in the F&O segment.
        URL: /api/reports?archives=[{"name":"FO - Pair Contracts",
              "type":"archives","category":"derivatives-market"}]
        """
        url = f"{NSE_BASE}/api/reports?archives=%5B%7B%22name%22%3A%22FO%20-%20Participant%20wise%20Open%20Interest%22%2C%22type%22%3A%22archives%22%2C%22category%22%3A%22derivatives-market%22%7D%5D"
        data = self._get(url)

        if not data:
            return []

        # This endpoint returns different format; parse what we can
        flows = []
        if isinstance(data, list):
            for entry in data:
                date_str = entry.get("date", datetime.now().strftime("%d-%b-%Y"))
                client_type = entry.get("clientType", "")
                fut_long = self._parse_crore(entry.get("futIdxLong", 0))
                fut_short = self._parse_crore(entry.get("futIdxShort", 0))
                opt_long = self._parse_crore(entry.get("optIdxCallLong", 0))
                opt_short = self._parse_crore(entry.get("optIdxPutLong", 0))

                net_fo = (fut_long - fut_short) + (opt_long - opt_short)

                if "FII" in client_type.upper() or "FPI" in client_type.upper():
                    flows.append(DailyFlow(
                        date=date_str,
                        fii_buy=fut_long + opt_long,
                        fii_sell=fut_short + opt_short,
                        fii_net=net_fo,
                        dii_buy=0,
                        dii_sell=0,
                        dii_net=0,
                        segment="FO",
                    ))
                elif "DII" in client_type.upper():
                    flows.append(DailyFlow(
                        date=date_str,
                        fii_buy=0,
                        fii_sell=0,
                        fii_net=0,
                        dii_buy=fut_long + opt_long,
                        dii_sell=fut_short + opt_short,
                        dii_net=net_fo,
                        segment="FO",
                    ))

        return flows

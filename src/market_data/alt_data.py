"""
Alternative Data Sources -- Signals the crowd doesn't watch.
Non-traditional data that gives edge over pure technical/fundamental traders.

Sources:
  1. Google Trends -- retail search sentiment (contrarian indicator)
  2. India VIX -- fear gauge from NSE
  3. Global markets summary -- US futures, Asia, Europe via Yahoo Finance
  4. Put/Call ratio trends -- historical PCR from NSE option chains
"""

import logging
import time
import json
from datetime import datetime, timedelta, date
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import requests
import pandas as pd

log = logging.getLogger("josho.alt_data")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "alt_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NSE_BASE = "https://www.nseindia.com"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

# Google Trends keywords we track (contrarian indicators)
TREND_KEYWORDS = [
    "NIFTY crash",
    "stock market",
    "SBI share price",
    "Reliance share",
    "market crash",
    "bull market",
]

# Yahoo Finance tickers for global markets
GLOBAL_TICKERS = {
    # US futures
    "ES=F": {"name": "S&P 500 Futures", "region": "US"},
    "NQ=F": {"name": "Nasdaq Futures", "region": "US"},
    "YM=F": {"name": "Dow Futures", "region": "US"},
    # Asian markets
    "^N225": {"name": "Nikkei 225", "region": "Asia"},
    "^HSI": {"name": "Hang Seng", "region": "Asia"},
    "000001.SS": {"name": "Shanghai Composite", "region": "Asia"},
    "^STI": {"name": "Straits Times", "region": "Asia"},
    # European markets
    "^FTSE": {"name": "FTSE 100", "region": "Europe"},
    "^GDAXI": {"name": "DAX", "region": "Europe"},
    "^FCHI": {"name": "CAC 40", "region": "Europe"},
}


@dataclass
class TrendSignal:
    """Google Trends signal for a keyword."""
    keyword: str
    current_interest: int  # 0-100 relative search interest
    avg_interest_4w: float  # 4-week average
    spike_ratio: float  # current / avg (>2 = spike)
    is_spike: bool
    fetched_at: str = ""


@dataclass
class VixData:
    """India VIX fear gauge data."""
    value: float
    change: float
    change_pct: float
    level: str  # LOW / MODERATE / HIGH / EXTREME
    signal: str  # Interpretation for trading
    timestamp: str = ""


@dataclass
class GlobalMarket:
    """Single global market index snapshot."""
    ticker: str
    name: str
    region: str
    price: float
    change: float
    change_pct: float
    signal: str  # BULLISH / BEARISH / NEUTRAL


@dataclass
class PcrTrend:
    """Put/Call ratio trend data point."""
    date: str
    pcr: float
    total_ce_oi: int
    total_pe_oi: int
    sentiment: str


# -- Google Trends (pytrends) ------------------------------------------------

def fetch_google_trends(
    keywords: list[str] = None,
    timeframe: str = "today 3-m",
    geo: str = "IN",
) -> list[TrendSignal]:
    """
    Fetch Google Trends data for stock-related keywords.
    Spikes in "NIFTY crash" or "market crash" = retail panic = contrarian buy.
    Spikes in "bull market" = retail euphoria = contrarian sell.

    Falls back to cached data if pytrends is unavailable or rate-limited.
    """
    keywords = keywords or TREND_KEYWORDS
    signals = []

    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="en-IN", tz=330)  # IST offset

        # pytrends allows max 5 keywords per request
        for batch_start in range(0, len(keywords), 5):
            batch = keywords[batch_start:batch_start + 5]
            try:
                pytrends.build_payload(batch, timeframe=timeframe, geo=geo)
                df = pytrends.interest_over_time()

                if df.empty:
                    log.warning(f"Empty trends data for: {batch}")
                    continue

                # Drop the 'isPartial' column if present
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])

                for kw in batch:
                    if kw not in df.columns:
                        continue

                    series = df[kw]
                    current = int(series.iloc[-1]) if len(series) > 0 else 0
                    avg_4w = float(series.tail(28).mean()) if len(series) >= 28 else float(series.mean())
                    spike_ratio = current / avg_4w if avg_4w > 0 else 1.0

                    signals.append(TrendSignal(
                        keyword=kw,
                        current_interest=current,
                        avg_interest_4w=round(avg_4w, 1),
                        spike_ratio=round(spike_ratio, 2),
                        is_spike=spike_ratio >= 2.0,
                        fetched_at=datetime.now().isoformat(),
                    ))

                time.sleep(1)  # rate limit between batches

            except Exception as e:
                log.warning(f"Trends batch failed for {batch}: {e}")
                continue

    except ImportError:
        log.warning("pytrends not installed -- using fallback URL scraping")
        signals = _fetch_trends_fallback(keywords, geo)
    except Exception as e:
        log.error(f"Google Trends fetch failed: {e}")
        signals = _load_cached_trends()

    # Cache results for fallback
    if signals:
        _cache_trends(signals)

    return signals


def _fetch_trends_fallback(keywords: list[str], geo: str = "IN") -> list[TrendSignal]:
    """
    Fallback: scrape Google Trends explore page for relative interest.
    Less reliable than pytrends but works without the library.
    """
    signals = []
    for kw in keywords:
        try:
            url = (
                "https://trends.google.com/trends/api/dailytrends"
                f"?hl=en-IN&tz=-330&geo={geo}&ns=15"
            )
            resp = requests.get(url, timeout=10, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            })
            # Google Trends API returns data with )]}' prefix
            if resp.ok:
                text = resp.text
                if text.startswith(")]}'"):
                    text = text[5:]
                data = json.loads(text)

                # Check if keyword appears in trending searches
                trending = data.get("default", {}).get("trendingSearchesDays", [])
                found_traffic = 0
                for day in trending[:7]:
                    for search in day.get("trendingSearches", []):
                        title = search.get("title", {}).get("query", "").lower()
                        if any(word in title for word in kw.lower().split()):
                            traffic_str = search.get("formattedTraffic", "0")
                            found_traffic = _parse_traffic(traffic_str)
                            break

                signals.append(TrendSignal(
                    keyword=kw,
                    current_interest=min(100, found_traffic),
                    avg_interest_4w=30.0,  # estimated baseline
                    spike_ratio=round(found_traffic / 30.0, 2) if found_traffic > 0 else 0.5,
                    is_spike=found_traffic > 60,
                    fetched_at=datetime.now().isoformat(),
                ))

            time.sleep(0.5)

        except Exception as e:
            log.debug(f"Trends fallback failed for '{kw}': {e}")

    return signals


def _parse_traffic(traffic_str: str) -> int:
    """Parse Google Trends traffic string like '50K+' or '200K+'."""
    traffic_str = traffic_str.replace("+", "").replace(",", "").strip()
    if traffic_str.endswith("K"):
        return int(float(traffic_str[:-1]) * 1000)
    elif traffic_str.endswith("M"):
        return int(float(traffic_str[:-1]) * 1_000_000)
    try:
        return int(traffic_str)
    except ValueError:
        return 0


def _cache_trends(signals: list[TrendSignal]):
    """Cache trends data for fallback use."""
    path = DATA_DIR / "trends_cache.json"
    data = [vars(s) for s in signals]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_cached_trends() -> list[TrendSignal]:
    """Load cached trends if fresh enough (< 6 hours)."""
    path = DATA_DIR / "trends_cache.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        signals = []
        for item in data:
            fetched = item.get("fetched_at", "")
            if fetched:
                age = datetime.now() - datetime.fromisoformat(fetched)
                if age > timedelta(hours=6):
                    continue
            signals.append(TrendSignal(**item))
        return signals
    except Exception:
        return []


def analyze_trend_sentiment(signals: list[TrendSignal] = None) -> dict:
    """
    Interpret Google Trends signals for trading.
    Retail panic = contrarian BUY. Retail euphoria = contrarian SELL.
    """
    if signals is None:
        signals = fetch_google_trends()

    if not signals:
        return {
            "signal": "NO_DATA",
            "fear_keywords": [],
            "greed_keywords": [],
            "interpretation": "Unable to fetch trends data",
        }

    fear_keywords = ["NIFTY crash", "market crash"]
    greed_keywords = ["bull market"]

    fear_spikes = [s for s in signals if s.keyword in fear_keywords and s.is_spike]
    greed_spikes = [s for s in signals if s.keyword in greed_keywords and s.is_spike]

    if fear_spikes:
        signal = "CONTRARIAN_BUY"
        interpretation = (
            f"Retail panic detected: {[s.keyword for s in fear_spikes]} spiking. "
            "Markets often bottom when retail searches for 'crash'."
        )
    elif greed_spikes:
        signal = "CONTRARIAN_SELL"
        interpretation = (
            f"Retail euphoria detected: {[s.keyword for s in greed_spikes]} spiking. "
            "Markets often top when retail searches for 'bull market'."
        )
    else:
        signal = "NEUTRAL"
        interpretation = "No extreme sentiment detected in search trends."

    return {
        "signal": signal,
        "fear_keywords": [
            {"keyword": s.keyword, "spike_ratio": s.spike_ratio}
            for s in signals if s.keyword in fear_keywords
        ],
        "greed_keywords": [
            {"keyword": s.keyword, "spike_ratio": s.spike_ratio}
            for s in signals if s.keyword in greed_keywords
        ],
        "all_trends": [
            {"keyword": s.keyword, "interest": s.current_interest, "spike": s.is_spike}
            for s in signals
        ],
        "interpretation": interpretation,
    }


# -- India VIX (Fear Gauge) --------------------------------------------------

def fetch_india_vix() -> Optional[VixData]:
    """
    Fetch India VIX from NSE.
    VIX < 13 = complacent (sell strangles)
    VIX 13-20 = normal
    VIX 20-30 = elevated fear (reduce size)
    VIX > 30 = extreme fear (contrarian buy / close shorts)
    """
    session = requests.Session()
    session.headers.update(NSE_HEADERS)

    try:
        # Set cookies first
        session.get(NSE_BASE, timeout=10)
        time.sleep(0.5)

        resp = session.get(f"{NSE_BASE}/api/allIndices", timeout=15)
        if not resp.ok:
            log.warning(f"VIX fetch failed: {resp.status_code}")
            return _load_cached_vix()

        data = resp.json()
        vix_entry = None
        for idx in data.get("data", []):
            if idx.get("index") == "INDIA VIX":
                vix_entry = idx
                break

        if not vix_entry:
            log.warning("INDIA VIX not found in indices data")
            return _load_cached_vix()

        vix_val = float(vix_entry.get("last", 0))
        change = float(vix_entry.get("percentChange", 0))
        abs_change = float(vix_entry.get("change", 0))

        # Classify VIX level
        if vix_val < 13:
            level = "LOW"
            signal = "Complacency -- good for selling options / strangles"
        elif vix_val < 20:
            level = "MODERATE"
            signal = "Normal volatility -- standard position sizing"
        elif vix_val < 30:
            level = "HIGH"
            signal = "Elevated fear -- reduce position sizes, widen stops"
        else:
            level = "EXTREME"
            signal = "Extreme fear -- contrarian buy signal, close short positions"

        result = VixData(
            value=round(vix_val, 2),
            change=round(abs_change, 2),
            change_pct=round(change, 2),
            level=level,
            signal=signal,
            timestamp=datetime.now().isoformat(),
        )

        # Cache for fallback
        _cache_vix(result)
        return result

    except Exception as e:
        log.error(f"India VIX fetch error: {e}")
        return _load_cached_vix()


def _cache_vix(vix: VixData):
    path = DATA_DIR / "vix_cache.json"
    path.write_text(json.dumps(vars(vix), indent=2), encoding="utf-8")


def _load_cached_vix() -> Optional[VixData]:
    path = DATA_DIR / "vix_cache.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ts = data.get("timestamp", "")
        if ts:
            age = datetime.now() - datetime.fromisoformat(ts)
            if age > timedelta(hours=2):
                return None
        return VixData(**data)
    except Exception:
        return None


# -- Global Markets Summary ---------------------------------------------------

def fetch_global_markets() -> list[GlobalMarket]:
    """
    Fetch global market data via Yahoo Finance.
    Pre-market: check US futures + Asian markets for NIFTY direction.
    """
    markets = []

    for ticker, info in GLOBAL_TICKERS.items():
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1d"
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                timeout=10,
            )
            if not resp.ok:
                continue

            data = resp.json()
            result = data.get("chart", {}).get("result", [])
            if not result:
                continue

            meta = result[0].get("meta", {})
            price = meta.get("regularMarketPrice", 0)
            prev_close = meta.get("chartPreviousClose", 0) or meta.get("previousClose", 0)

            if price <= 0 or prev_close <= 0:
                continue

            change = price - prev_close
            change_pct = (change / prev_close) * 100

            if change_pct > 0.5:
                signal = "BULLISH"
            elif change_pct < -0.5:
                signal = "BEARISH"
            else:
                signal = "NEUTRAL"

            markets.append(GlobalMarket(
                ticker=ticker,
                name=info["name"],
                region=info["region"],
                price=round(price, 2),
                change=round(change, 2),
                change_pct=round(change_pct, 2),
                signal=signal,
            ))

        except Exception as e:
            log.debug(f"Failed to fetch {ticker}: {e}")
            continue

    return markets


def global_market_summary() -> dict:
    """
    Aggregated global market view for pre-market analysis.
    If US + Asia are red, NIFTY likely gaps down.
    """
    markets = fetch_global_markets()

    if not markets:
        return {
            "signal": "NO_DATA",
            "us": [],
            "asia": [],
            "europe": [],
            "interpretation": "Unable to fetch global market data",
        }

    us = [m for m in markets if m.region == "US"]
    asia = [m for m in markets if m.region == "Asia"]
    europe = [m for m in markets if m.region == "Europe"]

    def region_signal(region_markets: list[GlobalMarket]) -> str:
        if not region_markets:
            return "NO_DATA"
        avg_change = sum(m.change_pct for m in region_markets) / len(region_markets)
        if avg_change > 0.5:
            return "BULLISH"
        elif avg_change < -0.5:
            return "BEARISH"
        return "NEUTRAL"

    us_signal = region_signal(us)
    asia_signal = region_signal(asia)
    europe_signal = region_signal(europe)

    # Overall: if 2/3 regions bearish = bearish, etc.
    signals = [us_signal, asia_signal, europe_signal]
    bullish_count = signals.count("BULLISH")
    bearish_count = signals.count("BEARISH")

    if bearish_count >= 2:
        overall = "BEARISH"
        interpretation = "Global risk-off -- expect NIFTY to open weak. Reduce long exposure."
    elif bullish_count >= 2:
        overall = "BULLISH"
        interpretation = "Global risk-on -- expect NIFTY to open strong. Can add longs."
    else:
        overall = "MIXED"
        interpretation = "Mixed global signals -- wait for Indian market to set its own tone."

    def _serialize_markets(mlist: list[GlobalMarket]) -> list[dict]:
        return [
            {
                "name": m.name,
                "ticker": m.ticker,
                "price": m.price,
                "change_pct": m.change_pct,
                "signal": m.signal,
            }
            for m in mlist
        ]

    return {
        "signal": overall,
        "us": _serialize_markets(us),
        "us_signal": us_signal,
        "asia": _serialize_markets(asia),
        "asia_signal": asia_signal,
        "europe": _serialize_markets(europe),
        "europe_signal": europe_signal,
        "interpretation": interpretation,
        "fetched_at": datetime.now().isoformat(),
    }


# -- Put/Call Ratio Trends ----------------------------------------------------

def track_pcr(symbol: str = "NIFTY", nse_session: Optional[requests.Session] = None) -> Optional[PcrTrend]:
    """
    Fetch current PCR and append to historical trend file.
    Tracks PCR over time for trend analysis.
    """
    session = nse_session or requests.Session()
    if not nse_session:
        session.headers.update(NSE_HEADERS)
        try:
            session.get(NSE_BASE, timeout=10)
            time.sleep(0.5)
        except Exception:
            pass

    try:
        url = f"{NSE_BASE}/api/option-chain-indices?symbol={symbol}"
        resp = session.get(url, timeout=15)
        if not resp.ok:
            return None

        data = resp.json()
        filtered = data.get("filtered", {})
        ce_oi = filtered.get("CE", {}).get("totOI", 0)
        pe_oi = filtered.get("PE", {}).get("totOI", 0)

        if ce_oi <= 0:
            return None

        pcr_val = pe_oi / ce_oi

        if pcr_val > 1.3:
            sentiment = "VERY_BULLISH"
        elif pcr_val > 1.0:
            sentiment = "BULLISH"
        elif pcr_val > 0.7:
            sentiment = "NEUTRAL"
        elif pcr_val > 0.5:
            sentiment = "BEARISH"
        else:
            sentiment = "VERY_BEARISH"

        trend_point = PcrTrend(
            date=datetime.now().isoformat(),
            pcr=round(pcr_val, 3),
            total_ce_oi=ce_oi,
            total_pe_oi=pe_oi,
            sentiment=sentiment,
        )

        # Append to history
        _append_pcr_history(symbol, trend_point)
        return trend_point

    except Exception as e:
        log.error(f"PCR tracking error for {symbol}: {e}")
        return None


def _append_pcr_history(symbol: str, point: PcrTrend):
    """Append PCR data point to history file."""
    path = DATA_DIR / f"pcr_history_{symbol.lower()}.json"
    history = []
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            history = []

    history.append(vars(point))

    # Keep last 90 days of data (assuming ~1 read per 15 min during market hours)
    max_entries = 90 * 26  # ~26 reads per day (6.5 hours * 4 per hour)
    if len(history) > max_entries:
        history = history[-max_entries:]

    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def get_pcr_trend(symbol: str = "NIFTY", lookback_days: int = 30) -> dict:
    """
    Analyze PCR trend over time.
    Rising PCR = more puts being bought = bearish hedging = contrarian bullish.
    Falling PCR = more calls being bought = bullish retail = contrarian bearish.
    """
    path = DATA_DIR / f"pcr_history_{symbol.lower()}.json"
    if not path.exists():
        return {
            "symbol": symbol,
            "signal": "NO_DATA",
            "data_points": 0,
            "interpretation": "No historical PCR data yet. Will build over time.",
        }

    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"symbol": symbol, "signal": "NO_DATA", "data_points": 0}

    if len(history) < 2:
        return {
            "symbol": symbol,
            "signal": "INSUFFICIENT_DATA",
            "data_points": len(history),
            "interpretation": "Need more data points for trend analysis.",
        }

    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    recent = [h for h in history if h.get("date", "") >= cutoff]

    if len(recent) < 2:
        recent = history[-50:]  # use whatever we have

    pcr_values = [h["pcr"] for h in recent]
    current_pcr = pcr_values[-1]
    avg_pcr = sum(pcr_values) / len(pcr_values)
    min_pcr = min(pcr_values)
    max_pcr = max(pcr_values)

    # Trend: compare last 25% vs first 25%
    quarter = max(1, len(pcr_values) // 4)
    recent_avg = sum(pcr_values[-quarter:]) / quarter
    older_avg = sum(pcr_values[:quarter]) / quarter

    if recent_avg > older_avg * 1.1:
        trend = "RISING"
        signal = "CONTRARIAN_BULLISH"
        interpretation = (
            f"PCR rising ({older_avg:.2f} -> {recent_avg:.2f}). "
            "More puts being bought = hedging increasing = contrarian bullish."
        )
    elif recent_avg < older_avg * 0.9:
        trend = "FALLING"
        signal = "CONTRARIAN_BEARISH"
        interpretation = (
            f"PCR falling ({older_avg:.2f} -> {recent_avg:.2f}). "
            "More calls being bought = retail bullish = contrarian bearish."
        )
    else:
        trend = "FLAT"
        signal = "NEUTRAL"
        interpretation = f"PCR stable around {avg_pcr:.2f}. No extreme in either direction."

    return {
        "symbol": symbol,
        "current_pcr": current_pcr,
        "avg_pcr": round(avg_pcr, 3),
        "min_pcr": round(min_pcr, 3),
        "max_pcr": round(max_pcr, 3),
        "trend": trend,
        "signal": signal,
        "data_points": len(recent),
        "interpretation": interpretation,
    }


# -- Master Alternative Data Dashboard ----------------------------------------

def get_alt_data_dashboard() -> dict:
    """
    Complete alternative data snapshot for pre-market / intraday decisions.
    Combines: Google Trends + VIX + Global Markets + PCR Trends.
    """
    log.info("Fetching alternative data dashboard...")

    trends = analyze_trend_sentiment()
    vix = fetch_india_vix()
    global_mkt = global_market_summary()
    pcr = get_pcr_trend("NIFTY")

    # Aggregate signals
    signals = {
        "trends": trends.get("signal", "NO_DATA"),
        "vix": vix.level if vix else "NO_DATA",
        "global": global_mkt.get("signal", "NO_DATA"),
        "pcr": pcr.get("signal", "NO_DATA"),
    }

    # Score: -2 (very bearish) to +2 (very bullish)
    score_map = {
        "CONTRARIAN_BUY": 1.5,
        "CONTRARIAN_BULLISH": 1.0,
        "BULLISH": 1.0,
        "NEUTRAL": 0,
        "MIXED": 0,
        "CONTRARIAN_SELL": -1.5,
        "CONTRARIAN_BEARISH": -1.0,
        "BEARISH": -1.0,
        "NO_DATA": 0,
        "INSUFFICIENT_DATA": 0,
    }

    # VIX scoring is inverted (high VIX = fear = contrarian buy)
    vix_score_map = {
        "LOW": -0.5,  # complacency = slightly bearish
        "MODERATE": 0,
        "HIGH": 0.5,  # fear = slightly bullish (contrarian)
        "EXTREME": 1.5,  # extreme fear = strong contrarian buy
    }

    alt_score = (
        score_map.get(signals["trends"], 0)
        + vix_score_map.get(signals["vix"], 0)
        + score_map.get(signals["global"], 0)
        + score_map.get(signals["pcr"], 0)
    )

    if alt_score >= 2:
        overall = "STRONG_BULLISH"
    elif alt_score >= 0.5:
        overall = "BULLISH"
    elif alt_score <= -2:
        overall = "STRONG_BEARISH"
    elif alt_score <= -0.5:
        overall = "BEARISH"
    else:
        overall = "NEUTRAL"

    return {
        "overall_signal": overall,
        "alt_score": round(alt_score, 1),
        "components": {
            "trends": trends,
            "vix": vars(vix) if vix else None,
            "global_markets": global_mkt,
            "pcr_trend": pcr,
        },
        "signals_summary": signals,
        "timestamp": datetime.now().isoformat(),
    }

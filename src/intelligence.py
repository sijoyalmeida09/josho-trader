"""
intelligence.py — Real-Time Market Intelligence Engine
========================================================
Monitors world leaders, news, social media, commodities, and global markets.
Maps events to Indian stock impact. Generates trade signals BEFORE market reacts.

Pipeline: Event → Parse (2s) → Map to stocks → Score → Signal → Execute

Sources monitored:
  - Trump (Truth Social, Twitter/X) — tariffs, trade war, sanctions
  - Modi / Indian PM — policy, reform, schemes
  - RBI Governor — rate decisions, inflation comments
  - SEBI — regulatory changes
  - Crude oil — sudden moves
  - USD/INR — currency swings
  - US markets — S&P futures, Dow
  - FII/DII flows — institutional money
  - Breaking news — Reuters, Bloomberg, ET, Moneycontrol

Usage:
    from intelligence import MarketBrain
    brain = MarketBrain()
    brain.start()  # runs continuous loop
    # or
    signals = brain.scan_once()  # single scan
"""

import os
import sys
import json
import time
import re
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
load_dotenv(Path("C:/josho-trader/.env"))

IST = timezone(timedelta(hours=5, minutes=30))
log = logging.getLogger("intelligence")

# ── IMPACT MAP: Global events → Indian stocks ────────────

# Which Indian stocks are affected by what global factor
IMPACT_MAP = {
    # TRUMP / US POLICY
    "tariff": {
        "bullish": [],  # tariffs hurt Indian IT
        "bearish": ["INFY", "TCS", "WIPRO", "HCLTECH", "TECHM", "LTI"],
        "sector": "IT",
        "magnitude": "HIGH",
    },
    "trade war": {
        "bullish": ["HINDALCO", "TATASTEEL", "JSWSTEEL"],  # anti-dumping helps domestic
        "bearish": ["INFY", "TCS"],
        "sector": "METALS/IT",
        "magnitude": "HIGH",
    },
    "china tariff": {
        "bullish": ["HINDALCO", "VEDL", "TATASTEEL", "COALINDIA"],  # India alternate supplier
        "bearish": [],
        "sector": "METALS",
        "magnitude": "MEDIUM",
    },
    "india tariff": {
        "bullish": [],
        "bearish": ["RELIANCE", "INFY", "TCS", "WIPRO", "BHARTIARTL"],
        "sector": "BROAD",
        "magnitude": "CRITICAL",
    },
    "sanction": {
        "bullish": [],
        "bearish": ["ONGC", "BPCL", "IOC"],  # oil supply risk
        "sector": "OIL",
        "magnitude": "HIGH",
    },
    "fed rate": {
        "bullish_if_cut": ["HDFCBANK", "ICICIBANK", "SBIN", "BAJFINANCE", "AXISBANK"],
        "bearish_if_hike": ["HDFCBANK", "ICICIBANK", "SBIN", "BAJFINANCE"],
        "sector": "BANKING",
        "magnitude": "HIGH",
    },
    "oil price": {
        "bullish_if_drop": ["BPCL", "IOC", "HINDPETRO", "MARUTI", "BAJAJ-AUTO"],
        "bearish_if_rise": ["BPCL", "IOC", "HINDPETRO", "INDIAVIX"],
        "bullish_if_rise": ["ONGC", "OIL", "COALINDIA", "VEDL"],
        "sector": "OIL/AUTO",
        "magnitude": "HIGH",
    },
    "dollar strength": {
        "bullish": ["INFY", "TCS", "WIPRO"],  # earn in USD
        "bearish": ["ONGC", "BPCL", "IOC"],   # import bill rises
        "sector": "IT/OIL",
        "magnitude": "MEDIUM",
    },
    "gold price": {
        "bullish_if_rise": ["TITAN"],
        "bearish_if_rise": [],
        "sector": "JEWELRY",
        "magnitude": "LOW",
    },

    # INDIA SPECIFIC
    "rbi rate cut": {
        "bullish": ["HDFCBANK", "ICICIBANK", "SBIN", "BAJFINANCE", "AXISBANK", "HDFC", "LIC"],
        "bearish": [],
        "sector": "BANKING",
        "magnitude": "HIGH",
    },
    "rbi rate hike": {
        "bullish": [],
        "bearish": ["HDFCBANK", "ICICIBANK", "SBIN", "BAJFINANCE"],
        "sector": "BANKING",
        "magnitude": "HIGH",
    },
    "gst": {
        "bullish": [],
        "bearish": ["ITC", "HINDUNILVR", "MARUTI"],
        "sector": "FMCG/AUTO",
        "magnitude": "MEDIUM",
    },
    "infrastructure": {
        "bullish": ["LTIM", "ADANIPOWER", "NTPC", "POWERGRID", "NHPC", "IRFC", "SAIL", "JSWSTEEL"],
        "bearish": [],
        "sector": "INFRA",
        "magnitude": "MEDIUM",
    },
    "defence": {
        "bullish": ["HAL", "BEL", "BHEL"],
        "bearish": [],
        "sector": "DEFENCE",
        "magnitude": "MEDIUM",
    },
    "renewable energy": {
        "bullish": ["TATAPOWER", "ADANIGREEN", "SUZLON", "NHPC"],
        "bearish": ["COALINDIA", "NTPC"],
        "sector": "ENERGY",
        "magnitude": "MEDIUM",
    },
    "telecom": {
        "bullish": ["BHARTIARTL", "RELIANCE"],
        "bearish": ["IDEA"],
        "sector": "TELECOM",
        "magnitude": "MEDIUM",
    },

    # GEOPOLITICAL
    "war": {
        "bullish": ["HAL", "BEL"],
        "bearish": ["NIFTY", "BANKNIFTY", "HDFCBANK"],
        "sector": "DEFENCE/BROAD",
        "magnitude": "CRITICAL",
    },
    "pakistan": {
        "bullish": ["HAL", "BEL"],
        "bearish": ["NIFTY"],
        "sector": "DEFENCE",
        "magnitude": "HIGH",
    },
    "recession": {
        "bullish": [],
        "bearish": ["INFY", "TCS", "WIPRO", "RELIANCE", "HDFCBANK"],
        "sector": "BROAD",
        "magnitude": "CRITICAL",
    },
}

# Key people whose words move markets
KEY_PEOPLE = {
    "trump": {"platform": "truth_social", "impact": "CRITICAL", "affects": "global"},
    "modi": {"platform": "twitter", "impact": "HIGH", "affects": "india"},
    "rbi_governor": {"platform": "news", "impact": "HIGH", "affects": "banking"},
    "sebi": {"platform": "news", "impact": "MEDIUM", "affects": "markets"},
    "powell": {"platform": "news", "impact": "HIGH", "affects": "global"},
    "xi_jinping": {"platform": "news", "impact": "MEDIUM", "affects": "trade"},
}


# ── NEWS SCRAPERS ────────────────────────────────────────

class NewsFetcher:
    """Fetch breaking news from multiple free sources."""

    FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY", "")
    NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")

    @staticmethod
    def fetch_finnhub_news(category: str = "general") -> list:
        """Finnhub — 60 calls/min free. Real-time market news."""
        if not NewsFetcher.FINNHUB_KEY:
            return []
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/news",
                params={"category": category, "token": NewsFetcher.FINNHUB_KEY},
                timeout=10,
            )
            return resp.json() if resp.ok else []
        except Exception:
            return []

    @staticmethod
    def fetch_indian_news() -> list:
        """Scrape Indian financial news from free RSS feeds."""
        headlines = []
        feeds = [
            ("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "ET Markets"),
            ("https://www.moneycontrol.com/rss/latestnews.xml", "MoneyControl"),
            ("https://www.livemint.com/rss/markets", "LiveMint"),
        ]
        for url, source in feeds:
            try:
                resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                if resp.ok:
                    # Simple XML title extraction
                    titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', resp.text)
                    if not titles:
                        titles = re.findall(r'<title>(.*?)</title>', resp.text)
                    for t in titles[:10]:
                        headlines.append({"title": t, "source": source, "time": datetime.now(IST).isoformat()})
            except Exception:
                pass
        return headlines

    @staticmethod
    def fetch_trump_social() -> list:
        """Monitor Trump's public posts via news aggregation.
        Truth Social doesn't have a public API, so we monitor via news mentions."""
        posts = []
        try:
            # Method 1: Finnhub news filtered for Trump
            if NewsFetcher.FINNHUB_KEY:
                resp = requests.get(
                    "https://finnhub.io/api/v1/news",
                    params={"category": "general", "token": NewsFetcher.FINNHUB_KEY},
                    timeout=10,
                )
                if resp.ok:
                    for article in resp.json():
                        title = article.get("headline", "").lower()
                        if any(kw in title for kw in ["trump", "tariff", "trade war", "sanction", "china"]):
                            posts.append({
                                "text": article.get("headline", ""),
                                "source": "finnhub_news",
                                "url": article.get("url", ""),
                                "time": article.get("datetime", 0),
                            })
        except Exception:
            pass

        # Method 2: RSS feeds for Trump mentions
        try:
            resp = requests.get(
                "https://news.google.com/rss/search?q=trump+tariff+india&hl=en-IN&gl=IN",
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if resp.ok:
                titles = re.findall(r'<title>(.*?)</title>', resp.text)
                for t in titles[:5]:
                    if any(kw in t.lower() for kw in ["trump", "tariff", "trade", "sanction"]):
                        posts.append({"text": t, "source": "google_news", "time": datetime.now(IST).isoformat()})
        except Exception:
            pass

        return posts


# ── COMMODITY & MACRO DATA ───────────────────────────────

class MacroFetcher:
    """Fetch commodity prices, currencies, and global market data."""

    @staticmethod
    def get_crude_oil() -> dict:
        """Get crude oil price (Brent). Major driver for Indian markets."""
        try:
            resp = requests.get("https://api.frankfurter.dev/v1/latest?base=USD&symbols=INR", timeout=5)
            # Frankfurter doesn't have crude, use alternative
            resp2 = requests.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": "OANDA:BCO_USD", "token": os.environ.get("FINNHUB_API_KEY", "")},
                timeout=5,
            )
            if resp2.ok:
                data = resp2.json()
                return {"price": data.get("c", 0), "change_pct": data.get("dp", 0), "source": "finnhub"}
        except Exception:
            pass
        return {}

    @staticmethod
    def get_usd_inr() -> dict:
        """Get USD/INR exchange rate."""
        try:
            resp = requests.get("https://api.frankfurter.dev/v1/latest?base=USD&symbols=INR", timeout=5)
            if resp.ok:
                data = resp.json()
                return {"rate": data.get("rates", {}).get("INR", 0), "source": "frankfurter"}
        except Exception:
            pass
        return {}

    @staticmethod
    def get_us_futures() -> dict:
        """Get S&P 500 futures — predicts Nifty opening direction."""
        try:
            if os.environ.get("FINNHUB_API_KEY"):
                resp = requests.get(
                    "https://finnhub.io/api/v1/quote",
                    params={"symbol": "^GSPC", "token": os.environ["FINNHUB_API_KEY"]},
                    timeout=5,
                )
                if resp.ok:
                    data = resp.json()
                    return {"price": data.get("c", 0), "change_pct": data.get("dp", 0)}
        except Exception:
            pass
        return {}

    @staticmethod
    def get_vix() -> dict:
        """Get India VIX — fear gauge. VIX < 12 = complacency, VIX > 20 = fear."""
        try:
            resp = requests.get(
                "https://www.nseindia.com/api/allIndices",
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                timeout=10,
            )
            if resp.ok:
                for idx in resp.json().get("data", []):
                    if "VIX" in idx.get("index", ""):
                        return {"value": idx.get("last", 0), "change": idx.get("percentChange", 0)}
        except Exception:
            pass
        return {}


# ── SENTIMENT ANALYZER ───────────────────────────────────

class SentimentAnalyzer:
    """Analyze news/social media text for market sentiment using Groq LLM."""

    GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

    @staticmethod
    def analyze(text: str) -> dict:
        """Fast sentiment analysis using Groq (free tier: 14.4K tokens/min)."""
        if not SentimentAnalyzer.GROQ_KEY or not text:
            return SentimentAnalyzer._keyword_analysis(text)

        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {SentimentAnalyzer.GROQ_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": (
                            "You are a financial market analyst. Analyze this news headline for impact on Indian stock market. "
                            "Respond ONLY with valid JSON: "
                            '{"sentiment": "bullish|bearish|neutral", "confidence": 0-100, '
                            '"affected_sectors": ["IT","BANKING","OIL","METALS","AUTO","PHARMA","FMCG","INFRA","DEFENCE","TELECOM"], '
                            '"affected_stocks": ["SYMBOL1","SYMBOL2"], '
                            '"magnitude": "LOW|MEDIUM|HIGH|CRITICAL", '
                            '"action": "BUY|SELL|HOLD", '
                            '"reason": "one line why"}'
                        )},
                        {"role": "user", "content": text},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 200,
                },
                timeout=15,
            )
            if resp.ok:
                content = resp.json()["choices"][0]["message"]["content"]
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        except Exception as e:
            log.warning(f"Groq sentiment failed: {e}")

        return SentimentAnalyzer._keyword_analysis(text)

    @staticmethod
    def _keyword_analysis(text: str) -> dict:
        """Fallback: keyword-based sentiment when LLM unavailable."""
        text_lower = text.lower()
        result = {
            "sentiment": "neutral",
            "confidence": 30,
            "affected_sectors": [],
            "affected_stocks": [],
            "magnitude": "LOW",
            "action": "HOLD",
            "reason": "keyword analysis",
        }

        # Check impact map
        for keyword, impact in IMPACT_MAP.items():
            if keyword in text_lower:
                result["magnitude"] = impact.get("magnitude", "MEDIUM")
                result["affected_sectors"].append(impact.get("sector", ""))

                # Determine direction
                bullish_words = ["cut", "boost", "support", "positive", "growth", "reform", "invest"]
                bearish_words = ["tariff", "sanction", "war", "crash", "recession", "hike", "ban", "restrict"]

                if any(w in text_lower for w in bearish_words):
                    result["sentiment"] = "bearish"
                    result["affected_stocks"] = impact.get("bearish", [])[:5]
                    result["action"] = "SELL" if result["magnitude"] in ("HIGH", "CRITICAL") else "HOLD"
                elif any(w in text_lower for w in bullish_words):
                    result["sentiment"] = "bullish"
                    result["affected_stocks"] = impact.get("bullish", [])[:5]
                    result["action"] = "BUY" if result["magnitude"] in ("HIGH", "CRITICAL") else "HOLD"

                result["confidence"] = 60
                break

        return result


# ── MARKET BRAIN ─────────────────────────────────────────

class MarketBrain:
    """
    The unified intelligence engine.
    Scans all sources, analyzes sentiment, maps to stocks, generates signals.
    """

    def __init__(self):
        self.news = NewsFetcher()
        self.macro = MacroFetcher()
        self.sentiment = SentimentAnalyzer()
        self.seen_headlines = set()  # dedup
        self.signals = []
        self.last_scan = 0

    def scan_once(self) -> list:
        """Run one full intelligence scan. Returns list of trade signals."""
        signals = []
        now = datetime.now(IST)
        log.info(f"Intelligence scan at {now.strftime('%H:%M:%S IST')}")

        # 1. Trump / geopolitical monitoring
        trump_posts = self.news.fetch_trump_social()
        for post in trump_posts:
            text = post.get("text", "")
            if text in self.seen_headlines:
                continue
            self.seen_headlines.add(text)

            analysis = self.sentiment.analyze(text)
            if analysis.get("magnitude") in ("HIGH", "CRITICAL"):
                signals.append({
                    "source": "trump",
                    "headline": text,
                    "analysis": analysis,
                    "time": now.isoformat(),
                    "priority": 1,
                })
                log.info(f"TRUMP SIGNAL: {text[:80]} -> {analysis['sentiment']} {analysis['affected_stocks']}")

        # 2. Indian news
        indian_news = self.news.fetch_indian_news()
        for article in indian_news[:15]:
            title = article.get("title", "")
            if title in self.seen_headlines or len(title) < 20:
                continue
            self.seen_headlines.add(title)

            # Only analyze potentially market-moving headlines
            market_keywords = [
                "nifty", "sensex", "rbi", "sebi", "rate", "gdp", "inflation",
                "tariff", "crude", "oil", "fii", "dii", "trump", "modi",
                "result", "quarter", "profit", "loss", "merger", "acquisition",
                "ban", "sanction", "reform", "budget", "tax", "gst",
                "adani", "reliance", "tata", "infosys", "hdfc", "sbi",
            ]
            if not any(kw in title.lower() for kw in market_keywords):
                continue

            analysis = self.sentiment.analyze(title)
            if analysis.get("confidence", 0) >= 50 and analysis.get("action") != "HOLD":
                signals.append({
                    "source": article.get("source", "news"),
                    "headline": title,
                    "analysis": analysis,
                    "time": now.isoformat(),
                    "priority": 2,
                })

        # 3. Macro data check
        crude = self.macro.get_crude_oil()
        if crude and abs(crude.get("change_pct", 0)) > 2:
            direction = "up" if crude["change_pct"] > 0 else "down"
            signals.append({
                "source": "crude_oil",
                "headline": f"Crude oil {direction} {abs(crude['change_pct']):.1f}%",
                "analysis": {
                    "sentiment": "bearish" if direction == "up" else "bullish",
                    "affected_stocks": ["ONGC", "BPCL", "IOC", "HINDPETRO"],
                    "magnitude": "HIGH" if abs(crude["change_pct"]) > 3 else "MEDIUM",
                    "action": "BUY" if direction == "down" else "SELL",
                },
                "time": now.isoformat(),
                "priority": 1,
            })

        # Sort by priority
        signals.sort(key=lambda s: s.get("priority", 99))

        # Keep last 50 headlines in dedup set
        if len(self.seen_headlines) > 200:
            self.seen_headlines = set(list(self.seen_headlines)[-50:])

        self.signals = signals
        self.last_scan = time.time()
        return signals

    def format_telegram(self) -> str:
        """Format signals for Telegram alert."""
        if not self.signals:
            return ""

        lines = ["MARKET INTELLIGENCE"]
        for sig in self.signals[:5]:
            analysis = sig["analysis"]
            sentiment = analysis.get("sentiment", "neutral").upper()
            stocks = ", ".join(analysis.get("affected_stocks", [])[:3])
            magnitude = analysis.get("magnitude", "?")
            action = analysis.get("action", "HOLD")

            lines.append(f"\n[{sig['source'].upper()}] {magnitude}")
            lines.append(f"  {sig['headline'][:100]}")
            lines.append(f"  {sentiment} | {action} | Stocks: {stocks}")

        return "\n".join(lines)


# ── STANDALONE TEST ──────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    brain = MarketBrain()
    print("Scanning for market intelligence...")
    signals = brain.scan_once()

    print(f"\nFound {len(signals)} signals:")
    for s in signals:
        print(f"\n[{s['source']}] {s['headline'][:100]}")
        print(f"  Sentiment: {s['analysis'].get('sentiment')} | Magnitude: {s['analysis'].get('magnitude')}")
        print(f"  Stocks: {s['analysis'].get('affected_stocks', [])}")
        print(f"  Action: {s['analysis'].get('action')}")

    if signals:
        print(f"\n{'='*60}")
        print(brain.format_telegram())

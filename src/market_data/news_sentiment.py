"""
News & Sentiment Engine — Real-time financial news + AI sentiment.
Integrated from: Realtime-NewsAPI, Stocker, StockNews, NewsFeel, Alpaca News.

Sources (free, no API key needed unless noted):
  1. Google News RSS — real-time, free, no key
  2. Yahoo Finance — stock-specific news scraping
  3. Alpaca News API — real-time financial news (free tier)
  4. NewsAPI.org — 10,000+ sources (free: 100 req/day)
  5. RSS feeds — Reuters, Bloomberg, Investing.com

Sentiment analysis:
  - VADER (fast, no API needed)
  - Groq AI (deeper analysis, free tier)
  - Keyword scoring (fallback)
"""

import os
import re
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field

import requests

log = logging.getLogger("josho.news")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


@dataclass
class NewsItem:
    """A single news article with sentiment."""
    title: str
    source: str
    url: str = ""
    published: str = ""
    summary: str = ""
    sentiment: float = 0.0  # -1 (bearish) to +1 (bullish)
    sentiment_label: str = "NEUTRAL"
    relevance: float = 0.0  # 0-1 how relevant to the query
    symbols: list[str] = field(default_factory=list)


# ── VADER-like Sentiment (no external dependency) ─────────────────

POSITIVE_WORDS = {
    "bullish", "surge", "rally", "soar", "jump", "gain", "profit", "beat",
    "upgrade", "breakout", "momentum", "strong", "growth", "record", "high",
    "buy", "accumulate", "outperform", "positive", "optimistic", "boom",
    "recover", "rebound", "expand", "upside", "dividend", "bonus",
}

NEGATIVE_WORDS = {
    "bearish", "crash", "plunge", "fall", "drop", "loss", "miss", "downgrade",
    "breakdown", "weak", "decline", "sell", "underperform", "negative",
    "pessimistic", "recession", "debt", "default", "fraud", "scam",
    "investigate", "warning", "risk", "volatile", "correction", "slump",
}

AMPLIFIERS = {"very", "extremely", "significantly", "sharply", "massive", "huge"}
NEGATORS = {"not", "no", "never", "neither", "nor", "hardly", "barely"}


def simple_sentiment(text: str) -> float:
    """Quick keyword-based sentiment scoring (-1 to +1)."""
    words = re.findall(r'\w+', text.lower())
    score = 0
    negate = False

    for i, word in enumerate(words):
        if word in NEGATORS:
            negate = True
            continue

        multiplier = 1.5 if (i > 0 and words[i - 1] in AMPLIFIERS) else 1.0

        if word in POSITIVE_WORDS:
            score += multiplier * (-1 if negate else 1)
            negate = False
        elif word in NEGATIVE_WORDS:
            score += multiplier * (1 if negate else -1)
            negate = False

    # Normalize to -1..+1
    if score == 0:
        return 0
    return max(-1, min(1, score / max(abs(score), 5)))


def ai_sentiment(text: str, symbol: str = "") -> dict:
    """Use Groq AI for deeper sentiment analysis."""
    if not GROQ_API_KEY:
        return {"sentiment": simple_sentiment(text), "analysis": ""}

    prompt = f"""Analyze this financial news for market sentiment.
Symbol context: {symbol or 'general market'}
News: "{text[:500]}"

Return JSON only:
{{"sentiment": <float -1 to 1, negative=bearish positive=bullish>,
  "confidence": <float 0-1>,
  "key_factor": "<one sentence: what drives the sentiment>",
  "impact": "<HIGH/MEDIUM/LOW>"}}"""

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
            timeout=15,
        )
        if resp.ok:
            raw = resp.json()["choices"][0]["message"]["content"]
            return json.loads(raw)
    except Exception as e:
        log.debug(f"AI sentiment failed: {e}")

    return {"sentiment": simple_sentiment(text), "analysis": "fallback"}


# ── News Sources ──────────────────────────────────────────────────

def fetch_google_news(query: str, count: int = 10) -> list[NewsItem]:
    """Fetch news from Google News RSS (free, no key)."""
    try:
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
        resp = requests.get(url, timeout=10)
        if not resp.ok:
            return []

        # Simple XML parsing (avoid lxml dependency)
        items = []
        for match in re.finditer(r'<item>(.*?)</item>', resp.text, re.DOTALL):
            item_xml = match.group(1)

            title_match = re.search(r'<title>(.*?)</title>', item_xml)
            link_match = re.search(r'<link>(.*?)</link>', item_xml)
            pub_match = re.search(r'<pubDate>(.*?)</pubDate>', item_xml)
            source_match = re.search(r'<source.*?>(.*?)</source>', item_xml)

            title = title_match.group(1) if title_match else ""
            if not title:
                continue

            sent = simple_sentiment(title)
            items.append(NewsItem(
                title=title.replace("&amp;", "&").replace("&quot;", '"'),
                source=source_match.group(1) if source_match else "Google News",
                url=link_match.group(1) if link_match else "",
                published=pub_match.group(1) if pub_match else "",
                sentiment=sent,
                sentiment_label="BULLISH" if sent > 0.2 else "BEARISH" if sent < -0.2 else "NEUTRAL",
            ))

            if len(items) >= count:
                break

        return items

    except Exception as e:
        log.error(f"Google News fetch failed: {e}")
        return []


def fetch_yahoo_news(symbol: str, count: int = 5) -> list[NewsItem]:
    """Fetch stock-specific news from Yahoo Finance."""
    try:
        # Yahoo Finance API for news
        url = f"https://query1.finance.yahoo.com/v8/finance/search?q={symbol}&newsCount={count}"
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        if not resp.ok:
            return []

        data = resp.json()
        items = []
        for article in data.get("news", []):
            title = article.get("title", "")
            sent = simple_sentiment(title)
            items.append(NewsItem(
                title=title,
                source=article.get("publisher", "Yahoo Finance"),
                url=article.get("link", ""),
                published=datetime.fromtimestamp(article.get("providerPublishTime", 0)).isoformat(),
                sentiment=sent,
                sentiment_label="BULLISH" if sent > 0.2 else "BEARISH" if sent < -0.2 else "NEUTRAL",
                symbols=[symbol],
            ))

        return items
    except Exception as e:
        log.debug(f"Yahoo news failed for {symbol}: {e}")
        return []


def fetch_market_news(count: int = 15) -> list[NewsItem]:
    """Aggregate news from multiple sources for Indian market."""
    all_news = []

    # Indian market news
    queries = [
        "NSE NIFTY stock market India",
        "BANKNIFTY options futures India",
        "Indian stock market today",
    ]

    for q in queries:
        news = fetch_google_news(q, count=5)
        all_news.extend(news)
        time.sleep(0.5)  # rate limit

    # Deduplicate by title similarity
    seen_titles = set()
    unique = []
    for item in all_news:
        title_key = item.title[:50].lower()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique.append(item)

    return unique[:count]


# ── Sentiment Aggregator ─────────────────────────────────────────

def get_market_sentiment() -> dict:
    """
    Aggregate sentiment across all news sources.
    Returns overall market sentiment score and breakdown.
    """
    news = fetch_market_news(count=15)

    if not news:
        return {
            "overall": 0,
            "label": "NEUTRAL",
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "news_count": 0,
            "top_headlines": [],
        }

    sentiments = [n.sentiment for n in news]
    overall = sum(sentiments) / len(sentiments) if sentiments else 0

    bullish = sum(1 for s in sentiments if s > 0.2)
    bearish = sum(1 for s in sentiments if s < -0.2)
    neutral = len(sentiments) - bullish - bearish

    if overall > 0.2:
        label = "BULLISH"
    elif overall < -0.2:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    return {
        "overall": round(overall, 3),
        "label": label,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "news_count": len(news),
        "top_headlines": [
            {"title": n.title, "sentiment": n.sentiment_label, "source": n.source}
            for n in sorted(news, key=lambda x: abs(x.sentiment), reverse=True)[:5]
        ],
    }


def get_symbol_sentiment(symbol: str) -> dict:
    """Get sentiment for a specific stock/index."""
    # Indian symbol mapping for news search
    symbol_names = {
        "NIFTY": "NIFTY 50 India",
        "BANKNIFTY": "Bank NIFTY India",
        "RELIANCE": "Reliance Industries",
        "TCS": "TCS Tata Consultancy",
        "INFY": "Infosys",
        "HDFCBANK": "HDFC Bank",
        "ICICIBANK": "ICICI Bank",
        "SBIN": "State Bank India SBI",
    }

    query = symbol_names.get(symbol, f"{symbol} stock India NSE")
    news = fetch_google_news(query, count=8)
    yahoo = fetch_yahoo_news(f"{symbol}.NS", count=5)
    all_news = news + yahoo

    if not all_news:
        return {"symbol": symbol, "sentiment": 0, "label": "NEUTRAL", "news_count": 0}

    # AI-enhanced sentiment for top headlines
    if GROQ_API_KEY and all_news:
        top = all_news[0]
        ai = ai_sentiment(top.title, symbol)
        ai_score = ai.get("sentiment", top.sentiment)
    else:
        ai_score = 0

    sentiments = [n.sentiment for n in all_news]
    avg = sum(sentiments) / len(sentiments)

    # Blend simple + AI sentiment
    final = (avg * 0.6 + ai_score * 0.4) if ai_score else avg

    return {
        "symbol": symbol,
        "sentiment": round(final, 3),
        "label": "BULLISH" if final > 0.2 else "BEARISH" if final < -0.2 else "NEUTRAL",
        "news_count": len(all_news),
        "headlines": [{"title": n.title, "sent": n.sentiment_label} for n in all_news[:5]],
    }

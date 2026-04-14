"""
JoSho Trader Dashboard — Live trading dashboard at trader.joshoit.com
FastAPI backend serving real-time data from Groww + NSE + News.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.client import get_client
from src.risk.manager import RiskManager
from src.market_data.news_sentiment import get_market_sentiment, get_symbol_sentiment

log = logging.getLogger("josho.dashboard")

app = FastAPI(title="JoSho Trader", docs_url="/api/docs")

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = Path(__file__).parent.parent / "data"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Cache layer (avoid hitting APIs on every page load) ───────────

_cache = {}
_cache_ttl = {}
CACHE_SECONDS = 30


def cached(key: str, fn, ttl: int = CACHE_SECONDS):
    now = datetime.now().timestamp()
    if key in _cache and now - _cache_ttl.get(key, 0) < ttl:
        return _cache[key]
    try:
        result = fn()
        _cache[key] = result
        _cache_ttl[key] = now
        return result
    except Exception as e:
        log.error(f"Cache miss for {key}: {e}")
        return _cache.get(key, {})


# ── Pages ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse(request=request, name="dashboard.html")


# ── API Endpoints (consumed by frontend JS) ───────────────────────

@app.get("/api/account")
async def api_account():
    """Account info: profile + margin + holdings."""
    client = get_client()
    client.connect()

    margin = cached("margin", lambda: client.get_margin())
    holdings = cached("holdings", lambda: client.get_holdings(), ttl=60)
    profile = cached("profile", lambda: client.get_profile(), ttl=300)

    holdings_list = []
    total_holdings_value = 0
    for h in holdings.get("holdings", []):
        val = h.get("quantity", 0) * h.get("average_price", 0)
        total_holdings_value += val
        holdings_list.append({
            "symbol": h.get("trading_symbol", "N/A"),
            "qty": int(h.get("quantity", 0)),
            "avg_price": h.get("average_price", 0),
            "value": round(val, 2),
        })

    return {
        "balance": margin.get("clear_cash", 0),
        "fno_available": margin.get("fno_margin_details", {}).get("option_buy_balance_available", 0),
        "equity_available": margin.get("equity_margin_details", {}).get("cnc_balance_available", 0),
        "margin_used": margin.get("net_margin_used", 0),
        "ucc": profile.get("ucc", ""),
        "segments": profile.get("active_segments", []),
        "holdings": holdings_list,
        "holdings_value": round(total_holdings_value, 2),
        "total_value": round(margin.get("clear_cash", 0) + total_holdings_value, 2),
    }


@app.get("/api/positions")
async def api_positions():
    """Open positions."""
    client = get_client()
    client.connect()
    return cached("positions", lambda: client.get_positions())


@app.get("/api/orders")
async def api_orders():
    """Today's orders."""
    client = get_client()
    client.connect()
    return cached("orders", lambda: client.get_orders())


@app.get("/api/quotes")
async def api_quotes():
    """Live quotes for watchlist."""
    client = get_client()
    client.connect()

    watchlist = [
        ("NIFTY 50", "NSE", "FNO", "NIFTY"),
        ("RELIANCE-EQ", "NSE", "CASH", "RELIANCE"),
        ("TCS-EQ", "NSE", "CASH", "TCS"),
        ("HDFCBANK-EQ", "NSE", "CASH", "HDFCBANK"),
        ("INFY-EQ", "NSE", "CASH", "INFY"),
        ("SBIN-EQ", "NSE", "CASH", "SBI"),
        ("ICICIBANK-EQ", "NSE", "CASH", "ICICI"),
        ("BAJFINANCE-EQ", "NSE", "CASH", "BAJFIN"),
        ("TATAMOTORS-EQ", "NSE", "CASH", "TATAMTR"),
        ("ITC-EQ", "NSE", "CASH", "ITC"),
    ]

    quotes = []
    for symbol, exchange, segment, label in watchlist:
        def fetch(s=symbol, e=exchange, seg=segment):
            return client.get_quote(s, exchange=e, segment=seg)

        q = cached(f"quote_{symbol}", fetch, ttl=15)
        if q:
            quotes.append({
                "symbol": label,
                "ltp": q.get("last_price", 0),
                "change": round(q.get("day_change", 0), 2),
                "change_pct": round(q.get("day_change_perc", 0), 2),
                "volume": q.get("volume", 0),
                "high": q.get("ohlc", {}).get("high", 0),
                "low": q.get("ohlc", {}).get("low", 0),
                "open": q.get("ohlc", {}).get("open", 0),
                "close": q.get("ohlc", {}).get("close", 0),
            })

    return {"quotes": quotes, "timestamp": datetime.now().isoformat()}


@app.get("/api/risk")
async def api_risk():
    """Risk manager status."""
    rm = RiskManager()
    return rm.get_status()


@app.get("/api/sentiment")
async def api_sentiment():
    """Market + index sentiment."""
    market = cached("sentiment_market", get_market_sentiment, ttl=300)

    return {
        "market": market,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/sentiment/{symbol}")
async def api_symbol_sentiment(symbol: str):
    """Symbol-specific sentiment."""
    return cached(
        f"sentiment_{symbol}",
        lambda: get_symbol_sentiment(symbol.upper()),
        ttl=300,
    )


@app.get("/api/trades")
async def api_trades():
    """Trade log."""
    trades_file = DATA_DIR / "trades.json"
    if trades_file.exists():
        return json.loads(trades_file.read_text(encoding="utf-8"))
    return []


@app.get("/api/health")
async def health():
    """Health check."""
    try:
        client = get_client()
        client.connect()
        return {"status": "ok", "api": "connected", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

"""
quick_start.py — Emergency standalone trader
=============================================
Paste this in ANY Python environment to start trading.
No dependencies on brain.py, autopilot.py, or sijoy_strategy.py.
Just growwapi + pyotp + requests.

If EVERYTHING else fails, run this:
  python quick_start.py

It will:
1. Connect using cached token (or generate new one)
2. Check wallet balance
3. Buy the best F&O options from predictor signals
4. Monitor and auto-exit at +25% profit or -50% stop
5. Scan equity for intraday momentum plays
6. Loop until market close
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── SETUP ───────────────────────────────────────────────────────

# Change this to your josho-trader directory
TRADER_DIR = Path("C:/josho-trader")
DATA_DIR = TRADER_DIR / "data"
ENV_FILE = TRADER_DIR / ".env"
TOKEN_CACHE = DATA_DIR / ".groww_token_cache"
STATE_FILE = DATA_DIR / "quick_state.json"
VIRTUAL_FILE = DATA_DIR / "virtual_trades.json"

IST = timezone(timedelta(hours=5, minutes=30))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("quick")

# Load .env manually (no dotenv dependency needed)
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

# Telegram
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

def tg(msg):
    if not TG_TOKEN or not TG_CHAT: return
    try:
        import requests
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": f"*QUICK TRADER*\n\n{msg}", "parse_mode": "Markdown"}, timeout=10)
    except: pass

# F&O lot sizes
LOTS = {
    "ADANIPOWER": 1250, "HINDALCO": 550, "VEDL": 1500, "PNB": 4000,
    "SAIL": 4000, "TATASTEEL": 1500, "TATAPOWER": 1350, "COALINDIA": 1500,
    "BPCL": 1100, "ONGC": 3850, "SUZLON": 4000, "NBCC": 7000,
}

# ── CONNECT ─────────────────────────────────────────────────────

from growwapi import GrowwAPI

def get_api():
    """Connect to Groww. Uses cached token first (ZERO API calls)."""

    # Try cached token
    if TOKEN_CACHE.exists():
        try:
            cache = json.loads(TOKEN_CACHE.read_text())
            if cache.get("expiry", 0) > time.time() + 300:
                api = GrowwAPI(cache["token"])
                log.info(f"Connected via cached token ({(cache['expiry']-time.time())/60:.0f} min left)")
                return api
        except: pass

    # Generate new token
    log.info("Generating fresh token...")

    totp_secret = os.environ.get("GROWW_TOTP_SECRET", "")
    totp_token = os.environ.get("GROWW_TOTP_TOKEN", "")
    api_key = os.environ.get("GROWW_API_KEY", "")
    secret = os.environ.get("GROWW_SECRET_KEY", "")

    token = None

    # Method 1: TOTP
    if totp_secret and totp_token:
        try:
            import pyotp
            code = pyotp.TOTP(totp_secret).now()
            token = GrowwAPI.get_access_token(api_key=totp_token, totp=code)
            log.info("Auth via TOTP")
        except Exception as e:
            log.warning(f"TOTP failed: {e}")

    # Method 2: API key + secret
    if not token and api_key and secret:
        try:
            token = GrowwAPI.get_access_token(api_key=api_key, secret=secret)
            log.info("Auth via API key")
        except Exception as e:
            log.error(f"API key failed: {e}")

    if not token:
        log.critical("ALL AUTH FAILED. Paste access token manually:")
        token = input("Access token: ").strip()
        if not token:
            sys.exit(1)

    # Cache it
    DATA_DIR.mkdir(exist_ok=True)
    TOKEN_CACHE.write_text(json.dumps({"token": token, "expiry": time.time() + 3600*4}))
    log.info("Token cached to disk")

    return GrowwAPI(token)


# ── FEE CALCULATOR ──────────────────────────────────────────────

def fees(value, fno=False):
    if fno:
        b = 20 * 2
        stt = value * 0.000625
        exc = value * 0.00053
    else:
        b = min(20, value * 0.0005) * 2
        stt = value * 0.00025
        exc = value * 0.0000345
    gst = (b + exc) * 0.18
    return b + stt + exc + gst + value * 0.00003


# ── STATE ───────────────────────────────────────────────────────

def load():
    if STATE_FILE.exists():
        try: return json.loads(STATE_FILE.read_text())
        except: pass
    return {"positions": [], "closed": [], "pnl": 0}

def save(state):
    state["updated"] = datetime.now(IST).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── MAIN LOOP ──────────────────────────────────────────────────

def run():
    api = get_api()
    state = load()
    last_call = 0
    scan = 0

    # Get balance
    try:
        margin = api.get_available_margin_details()
        wallet = margin.get("clear_cash", 0)
        fno_avail = margin.get("fno_margin_details", {}).get("option_buy_balance_available", 0)
        log.info(f"Wallet: Rs.{wallet:,.0f} | F&O available: Rs.{fno_avail:,.0f}")
    except Exception as e:
        log.error(f"Balance check failed: {e}")
        wallet = 7500
        fno_avail = 7500

    log.info("=" * 50)
    log.info("QUICK TRADER — LIVE")
    log.info(f"Positions: {len(state['positions'])}")
    log.info("=" * 50)
    tg(f"Quick trader started\nWallet: Rs.{wallet:,.0f}")

    while True:
        now = datetime.now(IST)
        scan += 1

        # Market hours only
        is_mkt = (now.weekday() < 5 and
                  (now.hour > 9 or (now.hour == 9 and now.minute >= 15)) and
                  (now.hour < 15 or (now.hour == 15 and now.minute <= 30)))

        if not is_mkt:
            if scan % 60 == 0:
                log.info(f"Market closed. {len([p for p in state['positions'] if p.get('type')=='FNO'])} F&O held overnight")
            time.sleep(60)
            continue

        # Throttle: 2s between calls
        def throttled_call(fn, *args, **kw):
            nonlocal last_call
            elapsed = time.time() - last_call
            if elapsed < 2: time.sleep(2 - elapsed)
            last_call = time.time()
            try: return fn(*args, **kw)
            except Exception as e:
                log.error(f"API: {e}")
                return None

        invested = sum(p.get("cost", 0) for p in state["positions"] if p.get("status") == "OPEN")
        avail = max(wallet - invested, 0)

        log.info(f"[{scan}] wallet=Rs.{wallet:,.0f} invested=Rs.{invested:,.0f} avail=Rs.{avail:,.0f} pos={len(state['positions'])} pnl=Rs.{state['pnl']:+,.0f}")

        # ── CHECK EXITS ──
        for pos in state["positions"]:
            if pos.get("status") != "OPEN": continue

            sym = pos["symbol"]
            is_fno = pos.get("type") == "FNO"
            seg = "FNO" if is_fno else "CASH"

            q = throttled_call(api.get_quote, trading_symbol=sym, exchange="NSE", segment=seg)
            if not q or not q.get("last_price"): continue

            ltp = q["last_price"]
            entry = pos["entry_price"]
            pnl_pct = ((ltp - entry) / entry * 100) if entry > 0 else 0
            gross = (ltp - entry) * pos["qty"]
            net = gross - pos.get("fees", 0)

            entry_t = datetime.fromisoformat(pos["entry_time"])
            held = (now - entry_t).total_seconds() / 60

            target = 25 if is_fno else 1.5
            stop = -50 if is_fno else -1.0
            reason = None

            if pnl_pct >= target: reason = "TARGET"
            elif pnl_pct <= stop: reason = "STOP"
            elif not is_fno and net > 0 and held > 30: reason = "FAST_CYCLE"
            elif not is_fno and pnl_pct < -0.3 and held > 60: reason = "REDEPLOY"
            elif not is_fno and now.hour == 15 and now.minute >= 10: reason = "MIS_CLOSE"
            elif is_fno and pnl_pct >= 15 and held < 360: reason = "FAST_FNO"

            if reason:
                product = "NRML" if is_fno else "MIS"
                throttled_call(api.place_order,
                    trading_symbol=sym, quantity=pos["qty"], validity="DAY",
                    exchange="NSE", segment=seg, product=product,
                    order_type="MARKET", transaction_type="SELL")

                pos["status"] = "CLOSED"
                pos["exit"] = ltp
                pos["exit_time"] = now.isoformat()
                pos["reason"] = reason
                pos["net_pnl"] = round(net, 2)
                state["pnl"] += net
                state["closed"].append(pos.copy())
                wallet += pos["cost"] + net  # freed capital

                win = "WIN" if net > 0 else "LOSS"
                msg = f"EXIT {win} ({reason})\n{sym}: Rs.{net:+,.0f} ({pnl_pct:+.1f}%)\nDay: Rs.{state['pnl']:+,.0f}"
                log.info(msg.replace("\n", " | "))
                tg(msg)
            else:
                log.info(f"  HOLD {sym} {pnl_pct:+.1f}% net=Rs.{net:+,.0f} {held:.0f}min")

        state["positions"] = [p for p in state["positions"] if p.get("status") == "OPEN"]

        # ── ENTER F&O ──
        open_count = len(state["positions"])
        if avail > 1000 and open_count < 4:
            if VIRTUAL_FILE.exists():
                try:
                    vdata = json.loads(VIRTUAL_FILE.read_text())
                    for sig in sorted(vdata.get("trades", []), key=lambda s: s.get("score", 0), reverse=True):
                        if sig.get("status") != "OPEN" or sig.get("score", 0) < 75: continue
                        sym = sig["option_symbol"]
                        if any(p["symbol"] == sym for p in state["positions"]): continue

                        stock = sig.get("stock_symbol", "")
                        lot = LOTS.get(stock, sig.get("lot_size", 0))
                        prem = sig.get("entry_premium", 0)
                        cost = prem * lot
                        f = fees(cost, fno=True)

                        if cost > avail or cost < 500: continue
                        if cost * 0.25 < f * 3: continue  # profit must > 3x fees

                        throttled_call(api.place_order,
                            trading_symbol=sym, quantity=lot, validity="DAY",
                            exchange="NSE", segment="FNO", product="NRML",
                            order_type="MARKET", transaction_type="BUY")

                        state["positions"].append({
                            "symbol": sym, "type": "FNO", "qty": lot,
                            "entry_price": prem, "cost": cost, "fees": round(f, 2),
                            "entry_time": now.isoformat(), "status": "OPEN",
                        })
                        avail -= cost
                        log.info(f"FNO BUY: {sym} {prem}x{lot}=Rs.{cost:,.0f} fees=Rs.{f:.0f}")
                        tg(f"BUY {sym}\nRs.{prem}x{lot}=Rs.{cost:,.0f}\nTarget +25% | Stop -50%")
                        break  # one F&O per scan cycle
                except Exception as e:
                    log.error(f"F&O signal error: {e}")

        # ── ENTER EQUITY ──
        eq_count = len([p for p in state["positions"] if p.get("type") != "FNO"])
        if avail > 800 and eq_count < 2 and now.hour < 14:
            best = None
            for sym in ["TATASTEEL", "SUZLON", "TATAPOWER", "ADANIPOWER", "VEDL",
                        "SBIN", "ICICIBANK", "RELIANCE", "BAJFINANCE", "PNB"]:
                q = throttled_call(api.get_quote, trading_symbol=f"{sym}-EQ", exchange="NSE", segment="CASH")
                if not q or not q.get("last_price"): continue

                ltp = q["last_price"]
                prev = q.get("ohlc", {}).get("close", 0)
                high = q.get("ohlc", {}).get("high", ltp)
                low = q.get("ohlc", {}).get("low", ltp)
                if prev == 0 or ltp < 15: continue

                chg = ((ltp - prev) / prev) * 100
                pos_r = (ltp - low) / (high - low) if high > low else 0.5

                score = 0
                if chg < -1.5 and pos_r < 0.3: score = 50 + min(abs(chg)*5, 30)  # oversold
                elif chg > 2.0 and pos_r > 0.8: score = 45 + min(chg*5, 25)  # momentum

                if score < 40: continue

                qty = int(min(avail * 0.5, avail) / ltp)
                if qty == 0: continue
                val = ltp * qty
                f = fees(val)
                exp_profit = val * 0.015  # 1.5% target
                if exp_profit < f * 3: continue

                if not best or score > best["score"]:
                    best = {"sym": sym, "ltp": ltp, "qty": qty, "val": val, "f": f, "score": score, "chg": chg}

            if best and not any(p["symbol"] == best["sym"] for p in state["positions"]):
                s = best
                throttled_call(api.place_order,
                    trading_symbol=f"{s['sym']}-EQ", quantity=s["qty"], validity="DAY",
                    exchange="NSE", segment="CASH", product="MIS",
                    order_type="MARKET", transaction_type="BUY")

                state["positions"].append({
                    "symbol": s["sym"], "type": "EQUITY", "qty": s["qty"],
                    "entry_price": s["ltp"], "cost": s["val"], "fees": round(s["f"], 2),
                    "entry_time": now.isoformat(), "status": "OPEN",
                })
                avail -= s["val"]
                log.info(f"EQ BUY: {s['qty']}x {s['sym']} @{s['ltp']:.2f} score={s['score']} chg={s['chg']:+.1f}%")
                tg(f"BUY {s['qty']}x {s['sym']} @Rs.{s['ltp']:,.2f}\nScore: {s['score']} | Change: {s['chg']:+.1f}%")

        save(state)
        time.sleep(30)


# ── ENTRY ───────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--status" in sys.argv:
        s = load()
        print(json.dumps(s, indent=2))
    else:
        run()

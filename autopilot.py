"""
autopilot.py — Continuous F&O Position Monitor + Auto-Exit
Runs forever. Checks positions every 60s during market hours.
Auto-sells at target/stop. Sends Telegram alerts.
"""
import sys, os, json, time
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timezone, timedelta
load_dotenv(Path("C:/josho-trader/.env"))

sys.path.insert(0, "C:/josho-trader/src")
from client import GrowwClient
from charges import ChargeCalculator
import requests

IST = timezone(timedelta(hours=5, minutes=30))
TG = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TC = os.environ.get("TELEGRAM_CHAT_ID", "")
LOG = Path("C:/josho-trader/logs/autopilot.log")

def tg(msg):
    if TG and TC:
        try:
            requests.post(f"https://api.telegram.org/bot{TG}/sendMessage",
                json={"chat_id": TC, "text": msg}, timeout=10)
        except: pass

def log(msg):
    ts = datetime.now(IST).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")

# Exit rules per position
EXIT_RULES = {
    "COALINDIA26APR460CE": {"target_pct": 25, "stop_pct": -40, "trail_start": 15, "trail_pct": 8},
    "COALINDIA26MAY500CE": {"target_pct": 50, "stop_pct": -40, "trail_start": 25, "trail_pct": 10},
    "COALINDIA26APR480CE": {"target_pct": 100, "stop_pct": -60, "trail_start": 50, "trail_pct": 15},
    "COALINDIA26APR500CE": {"target_pct": 150, "stop_pct": -60, "trail_start": 80, "trail_pct": 20},
}
DEFAULT_RULES = {"target_pct": 25, "stop_pct": -40, "trail_start": 15, "trail_pct": 8}

peaks = {}
scan = 0

log("AUTOPILOT STARTED")
tg("AUTOPILOT ONLINE\nMonitoring F&O positions with auto-exit")

client = GrowwClient()
client.connect()
log("Groww connected")

while True:
    now = datetime.now(IST)
    scan += 1

    # Reconnect if needed
    if not hasattr(client, '_api') or client._api is None:
        try:
            client.connect()
            log("Reconnected")
        except:
            time.sleep(60)
            continue

    # Market hours check
    is_market = (
        now.weekday() < 5
        and (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
        and (now.hour < 15 or (now.hour == 15 and now.minute <= 30))
    )

    if not is_market:
        if now.hour == 0 and now.minute < 2:
            log("Midnight reset")
            peaks.clear()
        time.sleep(120)
        continue

    # Check positions
    try:
        positions = client.get_positions(segment="FNO")
        pos_list = positions.get("positions", [])
    except:
        time.sleep(30)
        continue

    if not pos_list:
        if scan % 30 == 0:
            log("No F&O positions open")
        time.sleep(60)
        continue

    total_pnl = 0
    for p in pos_list:
        sym = p["trading_symbol"]
        entry = p["net_price"]
        qty = p["quantity"]

        time.sleep(2)
        try:
            q = client.get_quote(sym, exchange="NSE", segment="FNO")
            ltp = q.get("last_price", 0)
        except:
            continue
        if ltp <= 0:
            continue

        pnl = (ltp - entry) * qty
        pnl_pct = ((ltp - entry) / entry * 100) if entry > 0 else 0
        total_pnl += pnl

        # Track peak
        if sym not in peaks or ltp > peaks[sym]:
            peaks[sym] = ltp

        rules = EXIT_RULES.get(sym, DEFAULT_RULES)
        reason = None

        # Target hit
        if pnl_pct >= rules["target_pct"]:
            reason = f"TARGET +{pnl_pct:.1f}%"
        # Stop loss
        elif pnl_pct <= rules["stop_pct"]:
            reason = f"STOP {pnl_pct:.1f}%"
        # Trailing stop
        elif pnl_pct >= rules["trail_start"]:
            trail_stop = peaks[sym] * (1 - rules["trail_pct"] / 100)
            if ltp <= trail_stop:
                reason = f"TRAIL (peak Rs.{peaks[sym]:.2f})"

        if reason:
            log(f"SELLING {sym}: {reason}")
            time.sleep(2)
            try:
                result = client.place_order(
                    symbol=sym, qty=qty, side="SELL",
                    order_type="MARKET", product="NRML",
                    segment="FNO", exchange="NSE",
                )
                status = result.get("order_status", "UNKNOWN")
                log(f"  Order: {status}")
                net_pnl = pnl  # approximate
                win = "WIN" if net_pnl > 0 else "LOSS"
                tg(f"EXIT {win} ({reason})\n{sym}\nEntry Rs.{entry} -> Rs.{ltp}\nP&L: Rs.{net_pnl:+,.0f} ({pnl_pct:+.1f}%)")
            except Exception as e:
                log(f"  SELL FAILED: {e}")
        else:
            if scan % 5 == 0:
                log(f"HOLD {sym}: Rs.{ltp} ({pnl_pct:+.1f}%)")

    # Status every 10 scans (~10 min)
    if scan % 10 == 0:
        tg(f"AUTOPILOT [{now.strftime('%H:%M')}]\n{len(pos_list)} positions\nTotal P&L: Rs.{total_pnl:+,.0f}")

    time.sleep(60)

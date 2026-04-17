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

# Supabase — shared hub for ALL Sijoy systems
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://ylfagpbsmbhnmomeosyx.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

def tg(msg):
    if TG and TC:
        try:
            requests.post(f"https://api.telegram.org/bot{TG}/sendMessage",
                json={"chat_id": TC, "text": msg}, timeout=10)
        except: pass

def heartbeat(status: str, message: str, data: dict):
    """Write trader status to Supabase system_status table — the unified hub."""
    if not SUPABASE_KEY:
        return
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/system_status",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates",
            },
            json={
                "system_name": "trader",
                "status": status,
                "message": message,
                "data": data,
                "updated_at": datetime.now(IST).isoformat(),
            },
            timeout=10,
        )
    except Exception:
        pass

def log(msg):
    ts = datetime.now(IST).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
        f.flush()

# Exit rules per position
EXIT_RULES = {
    "COALINDIA26APR460CE": {"target_pct": 30, "stop_pct": -40, "trail_start": 20, "trail_pct": 10},
    "COALINDIA26MAY500CE": {"target_pct": 60, "stop_pct": -40, "trail_start": 30, "trail_pct": 12},
    "COALINDIA26APR480CE": {"target_pct": 120, "stop_pct": -60, "trail_start": 60, "trail_pct": 15},
    "COALINDIA26APR500CE": {"target_pct": 200, "stop_pct": -60, "trail_start": 100, "trail_pct": 20},
}
DEFAULT_RULES = {"target_pct": 30, "stop_pct": -40, "trail_start": 20, "trail_pct": 10}

# ── SAFE EXIT LOGIC ──────────────────────────────────────
# Philosophy: "One peak but two same lows. Exit on the second low."
# Don't sell at first peak. Wait for:
#   Peak → Dip (first low) → Recovery → Second Dip (second low, higher than first)
# Sell during the recovery AFTER second low confirms support.
# This avoids selling too early during a single pullback.

peaks = {}       # highest price seen per symbol
lows = {}        # list of dip prices per symbol [{price, time}]
price_history = {} # last N prices per symbol for pattern detection
scan = 0
HISTORY_LEN = 20  # track last 20 price points (~20 min at 1 min scans)

log("AUTOPILOT STARTED")
tg("AUTOPILOT ONLINE\nMonitoring F&O positions with auto-exit")
heartbeat("online", "Autopilot starting", {"event": "startup"})

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

        # Track price history for pattern detection
        if sym not in price_history:
            price_history[sym] = []
        price_history[sym].append(ltp)
        if len(price_history[sym]) > HISTORY_LEN:
            price_history[sym] = price_history[sym][-HISTORY_LEN:]

        # Track lows (dips from peak)
        if sym not in lows:
            lows[sym] = []
        hist = price_history[sym]
        # Detect a low: price dropped from peak then started rising again
        if len(hist) >= 3 and hist[-2] < hist[-1] and hist[-2] < hist[-3]:
            # hist[-2] was a local low
            low_price = hist[-2]
            # Only record if it's a meaningful dip from peak (>3%)
            if peaks.get(sym, 0) > 0:
                dip_pct = ((peaks[sym] - low_price) / peaks[sym]) * 100
                if dip_pct > 3:
                    lows[sym].append(low_price)
                    if len(lows[sym]) > 5:
                        lows[sym] = lows[sym][-5:]

        rules = EXIT_RULES.get(sym, DEFAULT_RULES)
        reason = None

        # ── SAFE EXIT LOGIC ──
        # Priority 1: Hard stop loss (always respect)
        if pnl_pct <= rules["stop_pct"]:
            reason = f"STOP {pnl_pct:.1f}%"

        # Priority 2: If trailing started AND we have 2+ confirmed lows
        elif pnl_pct >= rules["trail_start"]:
            sym_lows = lows.get(sym, [])
            trail_stop = peaks[sym] * (1 - rules["trail_pct"] / 100)

            if len(sym_lows) >= 2:
                # Second low confirmed — safe to exit on next dip
                second_low = sym_lows[-1]
                first_low = sym_lows[-2]
                # Second low should be higher than or equal to first (healthy trend)
                if second_low >= first_low * 0.97:  # within 3% tolerance
                    if ltp <= trail_stop:
                        reason = f"SAFE EXIT (2 lows: Rs.{first_low:.2f}, Rs.{second_low:.2f}, peak Rs.{peaks[sym]:.2f})"
                else:
                    # Second low LOWER than first = trend breaking, exit immediately
                    if ltp <= trail_stop:
                        reason = f"TREND BREAK (low2 Rs.{second_low:.2f} < low1 Rs.{first_low:.2f})"
            elif ltp <= trail_stop * 0.95:
                # No second low yet but dropped 5% below trail — force exit for safety
                reason = f"DEEP TRAIL (peak Rs.{peaks[sym]:.2f}, no 2nd low yet)"

        # Priority 3: Target hit — but DON'T sell immediately at first peak
        # Wait for confirmation (price holds near target for 2+ scans)
        elif pnl_pct >= rules["target_pct"]:
            # Check if we've been at/near target for at least 2 scans
            near_target_count = sum(1 for p in hist[-3:] if ((p - entry) / entry * 100) >= rules["target_pct"] * 0.9)
            if near_target_count >= 2:
                reason = f"TARGET CONFIRMED +{pnl_pct:.1f}% (held {near_target_count} scans)"
            else:
                log(f"TARGET ZONE {sym}: +{pnl_pct:.1f}% but waiting for confirmation...")

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

    # Heartbeat to Supabase — every 5 scans (~5 min)
    if scan % 5 == 0:
        pos_summary = []
        for p in pos_list:
            pos_summary.append({
                "symbol": p["trading_symbol"],
                "entry": p["net_price"],
                "qty": p["quantity"],
            })
        heartbeat("online", f"{len(pos_list)} positions | P&L Rs.{total_pnl:+,.0f}", {
            "positions": pos_summary,
            "total_pnl": round(total_pnl, 2),
            "position_count": len(pos_list),
            "scan": scan,
            "market_open": is_market,
        })

    time.sleep(60)

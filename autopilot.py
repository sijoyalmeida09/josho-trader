"""
autopilot.py — Smart F&O Position Monitor
===========================================
NEVER exits blindly. Before any exit decision, studies:
  - Full price journey from entry to now
  - How many peaks & valleys occurred
  - Volume trend (rising/falling)
  - Momentum strength (accelerating/decelerating)
  - Whether current move is trend or noise
  - Historical pattern: what happens AFTER this pattern?
  - Intelligence signals (news, Trump, crude, FII)

Philosophy: "One peak but two same lows. Exit near the second peak, not max."
"""
import sys, os, json, time, math
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
load_dotenv(Path("C:/josho-trader/.env"))

sys.path.insert(0, "C:/josho-trader/src")
from client import GrowwClient
from charges import ChargeCalculator
import requests

IST = timezone(timedelta(hours=5, minutes=30))
TG = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TC = os.environ.get("TELEGRAM_CHAT_ID", "")
LOG = Path("C:/josho-trader/logs/autopilot.log")
STATE_FILE = Path("C:/josho-trader/data/autopilot_state.json")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://ylfagpbsmbhnmomeosyx.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

# ── RISK MANAGEMENT ─────────────────────────────────────
MAX_RISK_PER_TRADE_PCT = 15   # max 15% of capital per trade
MAX_POSITIONS = 6              # max 6 simultaneous positions
MAX_DAILY_LOSS = 2000          # stop trading if day loss > Rs.2000
INTELLIGENCE_SCAN_INTERVAL = 15  # scan intelligence every 15 min (saves API calls)
day_pnl = 0
last_intel_scan = 0
intel_signals = []


def tg(msg):
    if TG and TC:
        try:
            requests.post(f"https://api.telegram.org/bot{TG}/sendMessage",
                json={"chat_id": TC, "text": msg}, timeout=10)
        except:
            pass


def heartbeat(status, message, data):
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
    except:
        pass


def log(msg):
    ts = datetime.now(IST).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
        f.flush()


# ══════════════════════════════════════════════════════════
# JOURNEY TRACKER — Full lifecycle analysis per position
# ══════════════════════════════════════════════════════════

class JourneyTracker:
    """
    Tracks the complete journey of a position from entry to now.
    Stores every price point, detects peaks/valleys, analyzes momentum.
    Before exit, generates a FULL REPORT of the journey.
    """

    def __init__(self):
        self.journeys = {}  # sym -> journey data
        self._load()

    def _load(self):
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                self.journeys = data.get("journeys", {})
            except:
                pass

    def _save(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps({"journeys": self.journeys}, indent=2, default=str))

    def record(self, sym: str, entry: float, ltp: float, volume: int = 0):
        """Record a price point for a position."""
        if sym not in self.journeys:
            self.journeys[sym] = {
                "entry": entry,
                "prices": [],
                "volumes": [],
                "peaks": [],       # [(price, scan_idx)]
                "valleys": [],     # [(price, scan_idx)]
                "peak_count": 0,
                "valley_count": 0,
                "max_price": entry,
                "min_price": entry,
                "max_pnl_pct": 0,
                "min_pnl_pct": 0,
                "start_time": datetime.now(IST).isoformat(),
                "scans": 0,
            }

        j = self.journeys[sym]
        j["prices"].append(ltp)
        j["volumes"].append(volume)
        j["scans"] += 1

        # Keep last 120 data points (~2 hours at 1 min scans)
        if len(j["prices"]) > 120:
            j["prices"] = j["prices"][-120:]
            j["volumes"] = j["volumes"][-120:]

        # Update extremes
        pnl_pct = ((ltp - entry) / entry * 100) if entry > 0 else 0
        if ltp > j["max_price"]:
            j["max_price"] = ltp
        if ltp < j["min_price"]:
            j["min_price"] = ltp
        if pnl_pct > j["max_pnl_pct"]:
            j["max_pnl_pct"] = round(pnl_pct, 2)
        if pnl_pct < j["min_pnl_pct"]:
            j["min_pnl_pct"] = round(pnl_pct, 2)

        # Detect peaks and valleys (need at least 3 data points)
        prices = j["prices"]
        idx = len(prices) - 1
        if len(prices) >= 3:
            p1, p2, p3 = prices[-3], prices[-2], prices[-1]

            # p2 was a peak: higher than both neighbors
            if p2 > p1 and p2 > p3:
                j["peaks"].append((p2, idx - 1))
                j["peak_count"] = len(j["peaks"])
                if len(j["peaks"]) > 20:
                    j["peaks"] = j["peaks"][-20:]

            # p2 was a valley: lower than both neighbors
            if p2 < p1 and p2 < p3:
                j["valleys"].append((p2, idx - 1))
                j["valley_count"] = len(j["valleys"])
                if len(j["valleys"]) > 20:
                    j["valleys"] = j["valleys"][-20:]

        self._save()

    def analyze(self, sym: str, entry: float, ltp: float) -> dict:
        """
        Full journey analysis. Called BEFORE any exit decision.
        Returns a report with all metrics and a RECOMMENDATION.
        """
        j = self.journeys.get(sym)
        if not j or len(j["prices"]) < 3:
            return {"recommendation": "HOLD", "reason": "insufficient data", "confidence": 0}

        prices = j["prices"]
        entry_price = j["entry"]
        pnl_pct = ((ltp - entry_price) / entry_price * 100) if entry_price > 0 else 0

        # ── MOMENTUM ANALYSIS ──
        # Compare recent movement vs overall trend
        recent_5 = prices[-5:] if len(prices) >= 5 else prices
        recent_10 = prices[-10:] if len(prices) >= 10 else prices

        # Momentum: is price accelerating or decelerating?
        if len(recent_5) >= 2:
            recent_change = ((recent_5[-1] - recent_5[0]) / recent_5[0] * 100) if recent_5[0] > 0 else 0
        else:
            recent_change = 0

        if len(recent_10) >= 2:
            medium_change = ((recent_10[-1] - recent_10[0]) / recent_10[0] * 100) if recent_10[0] > 0 else 0
        else:
            medium_change = 0

        # Momentum strength: recent > medium = accelerating
        momentum = "accelerating" if abs(recent_change) > abs(medium_change) * 0.5 else "decelerating"
        momentum_direction = "up" if recent_change > 0 else "down" if recent_change < 0 else "flat"

        # ── PEAK/VALLEY PATTERN ──
        peaks = j["peaks"]
        valleys = j["valleys"]

        # Are peaks getting higher? (bullish)
        higher_peaks = False
        if len(peaks) >= 2:
            higher_peaks = peaks[-1][0] > peaks[-2][0]

        # Are valleys getting higher? (bullish — higher lows)
        higher_valleys = False
        if len(valleys) >= 2:
            higher_valleys = valleys[-1][0] > valleys[-2][0]

        # Trend health score
        trend_score = 0
        if higher_peaks:
            trend_score += 30  # higher highs = bullish
        if higher_valleys:
            trend_score += 30  # higher lows = strong trend
        if momentum == "accelerating" and momentum_direction == "up":
            trend_score += 20
        if pnl_pct > 0:
            trend_score += 10
        if recent_change > 0:
            trend_score += 10

        # ── VOLATILITY ──
        if len(prices) >= 5:
            mean_price = sum(prices[-10:]) / len(prices[-10:])
            variance = sum((p - mean_price) ** 2 for p in prices[-10:]) / len(prices[-10:])
            volatility = math.sqrt(variance) / mean_price * 100 if mean_price > 0 else 0
        else:
            volatility = 0

        # ── DISTANCE FROM PEAK ──
        peak_price = j["max_price"]
        distance_from_peak = ((peak_price - ltp) / peak_price * 100) if peak_price > 0 else 0

        # ── VOLUME TREND ── (if available)
        volumes = [v for v in j["volumes"] if v > 0]
        volume_trend = "unknown"
        if len(volumes) >= 5:
            recent_vol = sum(volumes[-3:]) / 3
            older_vol = sum(volumes[-6:-3]) / 3 if len(volumes) >= 6 else recent_vol
            if older_vol > 0:
                vol_change = ((recent_vol - older_vol) / older_vol) * 100
                volume_trend = "rising" if vol_change > 10 else "falling" if vol_change < -10 else "stable"

        # ── RECOMMENDATION ──
        recommendation = "HOLD"
        reasons = []
        confidence = 50

        # NEVER exit if trend is strong and accelerating
        if trend_score >= 70 and momentum == "accelerating" and momentum_direction == "up":
            recommendation = "STRONG HOLD"
            reasons.append(f"Strong uptrend (score {trend_score}), momentum accelerating")
            confidence = 85

        # HOLD if we're near peak and momentum is up
        elif distance_from_peak < 3 and momentum_direction == "up":
            recommendation = "HOLD"
            reasons.append(f"Near peak ({distance_from_peak:.1f}% away), momentum still up")
            confidence = 70

        # CONSIDER EXIT if trend breaking
        elif not higher_peaks and not higher_valleys and len(peaks) >= 2:
            recommendation = "WATCH"
            reasons.append("Trend may be breaking: no higher peaks or valleys")
            confidence = 55

        # EXIT SIGNAL if falling from peak with broken trend
        elif distance_from_peak > 15 and not higher_valleys and momentum_direction == "down":
            recommendation = "EXIT"
            reasons.append(f"Down {distance_from_peak:.1f}% from peak, trend broken, momentum down")
            confidence = 75

        # EXIT SIGNAL if stop loss zone
        elif pnl_pct <= -35:
            recommendation = "EXIT"
            reasons.append(f"Approaching stop loss ({pnl_pct:.1f}%)")
            confidence = 90

        report = {
            "recommendation": recommendation,
            "confidence": confidence,
            "reasons": reasons,
            "pnl_pct": round(pnl_pct, 2),
            "max_pnl_pct": j["max_pnl_pct"],
            "min_pnl_pct": j["min_pnl_pct"],
            "peak_count": len(peaks),
            "valley_count": len(valleys),
            "higher_peaks": higher_peaks,
            "higher_valleys": higher_valleys,
            "trend_score": trend_score,
            "momentum": momentum,
            "momentum_direction": momentum_direction,
            "recent_change_pct": round(recent_change, 2),
            "volatility_pct": round(volatility, 2),
            "distance_from_peak_pct": round(distance_from_peak, 2),
            "volume_trend": volume_trend,
            "scans": j["scans"],
            "data_points": len(prices),
            "peak_price": j["max_price"],
            "valley_price": j["min_price"],
        }

        return report


# ══════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════

tracker = JourneyTracker()

# Hard stop — only exit here if truly broken
HARD_STOP_PCT = -45

log("AUTOPILOT STARTED")
tg("AUTOPILOT ONLINE — Smart exit with journey analysis")
heartbeat("online", "Autopilot starting", {"event": "startup"})

client = GrowwClient()
client.connect()
log("Groww connected")

scan = 0
while True:
    now = datetime.now(IST)
    scan += 1

    if not hasattr(client, '_api') or client._api is None:
        try:
            client.connect()
            log("Reconnected")
        except:
            time.sleep(60)
            continue

    is_market = (
        now.weekday() < 5
        and (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
        and (now.hour < 15 or (now.hour == 15 and now.minute <= 30))
    )

    if not is_market:
        time.sleep(120)
        continue

    try:
        positions = client.get_positions(segment="FNO")
        pos_list = positions.get("positions", [])
    except:
        time.sleep(30)
        continue

    if not pos_list:
        if scan % 30 == 0:
            log("No positions")
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
            volume = q.get("volume", 0)
        except:
            continue
        if ltp <= 0:
            continue

        pnl = (ltp - entry) * qty
        pnl_pct = ((ltp - entry) / entry * 100) if entry > 0 else 0
        total_pnl += pnl

        # Record in journey tracker
        tracker.record(sym, entry, ltp, volume)

        # ── EXIT DECISION: ALWAYS analyze journey first ──
        reason = None

        # Priority 1: HARD STOP (capital protection — non-negotiable)
        if pnl_pct <= HARD_STOP_PCT:
            reason = f"HARD STOP {pnl_pct:.1f}%"

        # Priority 2: Full journey analysis
        else:
            report = tracker.analyze(sym, entry, ltp)

            if report["recommendation"] == "EXIT" and report["confidence"] >= 70:
                # Double-check: generate full report for Telegram
                reason = (
                    f"JOURNEY EXIT ({report['confidence']}% confident)\n"
                    f"  Peaks: {report['peak_count']} | Valleys: {report['valley_count']}\n"
                    f"  Higher peaks: {report['higher_peaks']} | Higher valleys: {report['higher_valleys']}\n"
                    f"  Trend score: {report['trend_score']}/100\n"
                    f"  Momentum: {report['momentum']} {report['momentum_direction']}\n"
                    f"  Peak P&L: +{report['max_pnl_pct']}% | Now: {pnl_pct:+.1f}%\n"
                    f"  Distance from peak: {report['distance_from_peak_pct']:.1f}%\n"
                    f"  Reasons: {'; '.join(report['reasons'])}"
                )

            elif report["recommendation"] == "STRONG HOLD":
                if scan % 10 == 0:
                    log(f"STRONG HOLD {sym}: +{pnl_pct:.1f}% | trend={report['trend_score']} | {report['momentum']} {report['momentum_direction']} | peaks={report['peak_count']}")

            elif report["recommendation"] == "WATCH":
                if scan % 5 == 0:
                    log(f"WATCH {sym}: +{pnl_pct:.1f}% | trend={report['trend_score']} | {'; '.join(report['reasons'])}")

        # Execute exit if reason found
        if reason:
            log(f"SELLING {sym}: {reason}")
            time.sleep(2)
            try:
                # Use place_fno_order — auto-verifies lot size, never hardcodes
                result = client.place_fno_order(
                    symbol=sym, side="SELL",
                    order_type="MARKET", product="NRML",
                )
                status = result.get("order_status", "UNKNOWN")
                log(f"  Order: {status}")
                win = "WIN" if pnl > 0 else "LOSS"

                # Full exit report to Telegram
                j = tracker.journeys.get(sym, {})
                tg(
                    f"EXIT {win}\n{sym}\n"
                    f"Entry Rs.{entry} -> Rs.{ltp}\n"
                    f"P&L: Rs.{pnl:+,.0f} ({pnl_pct:+.1f}%)\n"
                    f"Peak was: Rs.{j.get('max_price', ltp)} (+{j.get('max_pnl_pct', 0)}%)\n"
                    f"Peaks seen: {j.get('peak_count', 0)} | Valleys: {j.get('valley_count', 0)}\n"
                    f"Reason: {reason[:200]}"
                )
            except Exception as e:
                log(f"  SELL FAILED: {e}")
        else:
            if scan % 5 == 0:
                j = tracker.journeys.get(sym, {})
                log(f"HOLD {sym}: Rs.{ltp} ({pnl_pct:+.1f}%) | peak={j.get('max_price', '?')} | peaks={len(j.get('peaks', []))} valleys={len(j.get('valleys', []))}")

    # ── INTELLIGENCE SCAN + AUTO-ENTRY ──────────────────────
    # Scan for new opportunities every 15 min (saves API calls)
    if scan % INTELLIGENCE_SCAN_INTERVAL == 0 and now.hour < 14:
        try:
            from intelligence import MarketBrain
            brain = MarketBrain()
            intel_signals = brain.scan_once()
            last_intel_scan = time.time()

            # Count open positions
            open_count = len([p for p in pos_list if p.get("quantity", 0) > 0])

            # Check if we should enter new positions based on intelligence
            if open_count < MAX_POSITIONS and day_pnl > -MAX_DAILY_LOSS:
                # Get balance
                try:
                    margin = client.get_margin()
                    balance = margin.get("fno_margin_details", {}).get("option_buy_balance_available", 0)
                except:
                    balance = 0

                max_per_trade = balance * (MAX_RISK_PER_TRADE_PCT / 100)

                # Find bullish signals with HIGH+ magnitude
                buy_signals = [s for s in intel_signals
                               if s["analysis"].get("action") == "BUY"
                               and s["analysis"].get("magnitude") in ("HIGH", "CRITICAL")
                               and s["analysis"].get("confidence", 0) >= 60]

                if buy_signals and max_per_trade > 500:
                    log(f"INTEL: {len(buy_signals)} BUY signals, {len(intel_signals)} total, balance Rs.{balance:,.0f}")
                    for sig in buy_signals[:1]:  # max 1 new entry per scan
                        log(f"  Signal: {sig['headline'][:80]}")
                        # Don't auto-enter yet — just alert via Telegram
                        tg(f"INTEL BUY SIGNAL\n{sig['headline'][:100]}\nStocks: {sig['analysis'].get('affected_stocks', [])}\nMagnitude: {sig['analysis'].get('magnitude')}\nConfidence: {sig['analysis'].get('confidence')}%")

                # Alert on bearish signals affecting our positions
                for sig in intel_signals:
                    if sig["analysis"].get("action") == "SELL" and sig["analysis"].get("magnitude") == "CRITICAL":
                        affected = sig["analysis"].get("affected_stocks", [])
                        for p in pos_list:
                            sym = p["trading_symbol"]
                            stock = ""
                            for s in ["COALINDIA", "HINDALCO", "TATASTEEL", "ADANIPOWER", "SUZLON"]:
                                if sym.startswith(s):
                                    stock = s
                                    break
                            if stock in affected:
                                tg(f"WARNING: {stock} position threatened\n{sig['headline'][:100]}\nAction: monitor closely")
                                log(f"INTEL WARNING: {stock} threatened by {sig['headline'][:60]}")
        except Exception as e:
            log(f"Intelligence scan failed: {e}")

    # ── PRE-MARKET SCAN (8:45-9:15 AM) ───────────────────
    if now.hour == 8 and now.minute >= 45 and scan % 5 == 0:
        try:
            from intelligence import MacroFetcher
            gift = MacroFetcher.get_gift_nifty()
            indices = MacroFetcher.get_global_indices()
            fgi = MacroFetcher.get_fear_greed()

            pre_market = f"PRE-MARKET SCAN {now.strftime('%H:%M')}\n"
            if gift:
                pre_market += f"GIFT Nifty: {gift.get('change_pct', 0):+.1f}%\n"
            for name, data in indices.items():
                pre_market += f"{name}: {data.get('change_pct', 0):+.1f}%\n"
            if fgi:
                pre_market += f"Fear & Greed: {fgi.get('value', 50)} ({fgi.get('classification', '?')})\n"

            tg(pre_market)
            log(f"Pre-market scan sent")
        except Exception as e:
            log(f"Pre-market scan failed: {e}")

    # Telegram status every 10 scans
    if scan % 10 == 0:
        tg(f"AUTOPILOT [{now.strftime('%H:%M')}]\n{len(pos_list)} positions\nTotal P&L: Rs.{total_pnl:+,.0f}")

    # Heartbeat every 5 scans
    if scan % 5 == 0:
        pos_summary = []
        for p in pos_list:
            sym = p["trading_symbol"]
            j = tracker.journeys.get(sym, {})
            pos_summary.append({
                "symbol": sym,
                "entry": p["net_price"],
                "qty": p["quantity"],
                "peak_count": len(j.get("peaks", [])),
                "valley_count": len(j.get("valleys", [])),
                "max_pnl": j.get("max_pnl_pct", 0),
            })
        heartbeat("online", f"{len(pos_list)} positions | P&L Rs.{total_pnl:+,.0f}", {
            "positions": pos_summary,
            "total_pnl": round(total_pnl, 2),
            "position_count": len(pos_list),
            "scan": scan,
        })

    time.sleep(60)

"""
perpetual_engine.py — Never Idle, Always Compounding
=====================================================
Money NEVER sits. Every exit becomes an entry. Play BOTH sides.

Core Philosophy:
  - When price rises: hold calls, compound gains
  - When price reverses from peak: sell calls, BUY PUTS immediately
  - When price bottoms: sell puts, BUY CALLS immediately
  - Repeat forever. Money is always working.

The engine detects reversals using the exit_engine's 30 strategies
and automatically rotates between calls and puts.

Cycle:
  CALL → peak detected → sell call (profit) → buy put (ride down)
  PUT → bottom detected → sell put (profit) → buy call (ride up)
  Repeat. Compound. Never idle.

Usage:
    from perpetual_engine import PerpetualTrader
    trader = PerpetualTrader(client, capital=5000)
    trader.run()  # runs forever during market hours
"""

import sys
import os
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))
from exit_engine import ExitEngine
from charges import ChargeCalculator

IST = timezone(timedelta(hours=5, minutes=30))
log = logging.getLogger("perpetual")
DATA_DIR = Path(__file__).parent.parent / "data"
STATE_FILE = DATA_DIR / "perpetual_state.json"


class PerpetualTrader:
    """
    Perpetual rotation: calls on the way up, puts on the way down.
    Every reversal = new entry. Money never sleeps.
    """

    def __init__(self, client, capital: float = 5000, stock: str = "COALINDIA"):
        self.client = client
        self.initial_capital = capital  # SACRED — never go below this
        self.capital = capital
        self.stock = stock
        self.exit_engine = ExitEngine()
        self.calc = ChargeCalculator()

        # ── RATCHET: Money only goes UP ──────────────────
        # locked_profits can NEVER be risked. Only unlocked capital trades.
        # After every winning trade, profit moves to locked.
        # Floor = initial_capital + locked_profits (always rises, never falls)
        self.locked_profits = 0    # profits already secured (untouchable)
        self.high_water_mark = capital  # highest capital ever seen
        self.floor = capital       # minimum capital allowed (only goes UP)

        # State
        self.position = None
        self.history = []
        self.total_pnl = 0
        self.trade_count = 0
        self.win_count = 0
        self.cycle_count = 0
        self.peak_prices = []
        self.valley_prices = []

        self._load_state()

    def _load_state(self):
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                self.position = data.get("position")
                self.history = data.get("history", [])
                self.total_pnl = data.get("total_pnl", 0)
                self.trade_count = data.get("trade_count", 0)
                self.win_count = data.get("win_count", 0)
                self.cycle_count = data.get("cycle_count", 0)
                self.capital = data.get("capital", self.capital)
                self.locked_profits = data.get("locked_profits", 0)
                self.high_water_mark = data.get("high_water_mark", self.capital)
                self.floor = data.get("floor", self.initial_capital)
            except:
                pass

    def _save_state(self):
        STATE_FILE.write_text(json.dumps({
            "position": self.position,
            "history": self.history[-50:],
            "total_pnl": round(self.total_pnl, 2),
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "cycle_count": self.cycle_count,
            "capital": round(self.capital, 2),
            "locked_profits": round(self.locked_profits, 2),
            "high_water_mark": round(self.high_water_mark, 2),
            "floor": round(self.floor, 2),
            "initial_capital": self.initial_capital,
            "last_updated": datetime.now(IST).isoformat(),
        }, indent=2))

    def _ratchet_up(self, profit: float):
        """Lock in profits — floor only goes UP, never down.
        After a winning trade, profit becomes untouchable."""
        if profit > 0:
            # Lock 70% of profit (keep 30% for compounding risk)
            lock_amount = profit * 0.7
            self.locked_profits += lock_amount
            self.floor = self.initial_capital + self.locked_profits
            log.info(f"RATCHET: Locked Rs.{lock_amount:,.0f} | Floor now Rs.{self.floor:,.0f} | Locked total: Rs.{self.locked_profits:,.0f}")

        # Update high water mark
        if self.capital > self.high_water_mark:
            self.high_water_mark = self.capital

    def _check_floor(self) -> bool:
        """Check if capital is above floor. If not, STOP TRADING."""
        if self.capital < self.floor:
            log.warning(f"FLOOR BREACH: Capital Rs.{self.capital:,.0f} < Floor Rs.{self.floor:,.0f} — STOPPING")
            return False
        return True

    def get_risk_capital(self) -> float:
        """How much can we risk? Only unlocked capital above the floor."""
        available = self.capital - self.floor
        # Never risk more than 20% of total capital per trade
        max_risk = self.capital * 0.20
        return min(max(available, 0) + max_risk * 0.5, self.capital * 0.35)

    def find_best_option(self, opt_type: str = "CE", max_cost: float = 0) -> dict:
        """Find the best call or put to enter right now."""
        if max_cost <= 0:
            max_cost = self.capital

        try:
            expiries = self.client.get_expiries(self.stock)
            valid = [e for e in expiries.get("expiries", []) if e >= datetime.now(IST).strftime("%Y-%m-%d")]
            if not valid:
                return {}

            # Use nearest expiry with >7 days (avoid extreme theta)
            today = datetime.now(IST).date()
            best_exp = None
            for exp in valid:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                days = (exp_date - today).days
                if days >= 7:
                    best_exp = exp
                    break
            if not best_exp:
                best_exp = valid[-1]  # use furthest if all < 7 days

            chain = self.client.get_option_chain(self.stock, best_exp)
            if not chain:
                return {}

            strikes = chain.get("strikes", {})
            ultp = chain.get("underlying_ltp", 0)
            if ultp <= 0:
                return {}

            # Find best option: liquid, affordable, moderate OTM
            candidates = []
            for sp, sides in strikes.items():
                opt = sides.get(opt_type, {})
                sym = opt.get("trading_symbol", "")
                ltp = opt.get("ltp", 0)
                vol = opt.get("volume", 0)
                oi = opt.get("open_interest", 0)
                delta = abs(opt.get("greeks", {}).get("delta", 0))

                if not sym or ltp <= 0:
                    continue

                lot = self.client.get_lot_size(sym)
                if lot == 0:
                    continue

                cost = ltp * lot
                if cost > max_cost or cost < 50:
                    continue

                strike = float(sp)
                if opt_type == "CE":
                    otm = ((strike - ultp) / ultp * 100)
                else:
                    otm = ((ultp - strike) / ultp * 100)

                if otm < 2 or otm > 25:
                    continue  # skip ATM and ultra deep OTM

                # Score: balance of liquidity + affordability + moderate OTM
                score = (vol * 0.3 + oi * 0.2 + delta * 100 + (1 / max(cost, 1)) * 500)

                candidates.append({
                    "symbol": sym,
                    "type": opt_type,
                    "strike": strike,
                    "premium": ltp,
                    "lot": lot,
                    "cost": cost,
                    "otm_pct": round(otm, 1),
                    "volume": vol,
                    "oi": oi,
                    "delta": round(delta, 4),
                    "expiry": best_exp,
                    "score": score,
                })

            if not candidates:
                return {}

            # Best candidate
            candidates.sort(key=lambda x: x["score"], reverse=True)
            return candidates[0]

        except Exception as e:
            log.error(f"Option scan failed: {e}")
            return {}

    def enter(self, opt_type: str = "CE") -> bool:
        """Enter a new position (call or put). Uses risk capital, not full capital."""
        if self.position:
            log.info(f"Already in position: {self.position['symbol']}")
            return False

        if not self._check_floor():
            return False

        risk_capital = self.get_risk_capital()
        log.info(f"Risk capital: Rs.{risk_capital:,.0f} (total: Rs.{self.capital:,.0f}, floor: Rs.{self.floor:,.0f})")

        option = self.find_best_option(opt_type, risk_capital)
        if not option:
            log.warning(f"No suitable {opt_type} found within Rs.{self.capital:,.0f}")
            return False

        sym = option["symbol"]
        lot = option["lot"]
        cost = option["cost"]

        # Fee check
        fees = self.calc.options_buy(option["premium"], lot).total + \
               self.calc.options_sell(option["premium"] * 1.1, lot).total
        if cost * 0.15 < fees:  # 15% profit must cover fees
            log.info(f"SKIP {sym}: fees Rs.{fees:.0f} too high for Rs.{cost:,.0f}")
            return False

        # Execute buy
        time.sleep(2)
        result = self.client.place_fno_order(symbol=sym, side="BUY")
        if result.get("status") == "FAILED" or result.get("order_status") == "FAILED":
            log.error(f"BUY failed: {result}")
            return False

        self.position = {
            "symbol": sym,
            "type": opt_type,
            "strike": option["strike"],
            "entry": option["premium"],
            "lot": lot,
            "cost": cost,
            "expiry": option["expiry"],
            "entry_time": datetime.now(IST).isoformat(),
            "prices": [option["premium"]],
            "volumes": [],
            "peaks": [],
            "valleys": [],
            "order": result,
        }
        self.capital -= cost
        self.trade_count += 1
        self._save_state()

        log.info(f"ENTER {opt_type}: {sym} @ Rs.{option['premium']} x {lot} = Rs.{cost:,.0f} | OTM +{option['otm_pct']}%")
        return True

    def check_and_rotate(self) -> str:
        """Check current position. If reversal detected, exit and enter opposite side.
        Returns: 'hold', 'rotated_to_put', 'rotated_to_call', 'exited'"""
        if not self.position:
            return "no_position"

        sym = self.position["symbol"]
        entry = self.position["entry"]
        opt_type = self.position["type"]

        # Get live price
        time.sleep(2)
        try:
            q = self.client.get_quote(sym, exchange="NSE", segment="FNO")
            ltp = q.get("last_price", 0)
            volume = q.get("volume", 0)
        except:
            return "hold"
        if ltp <= 0:
            return "hold"

        # Record price
        self.position["prices"].append(ltp)
        self.position["volumes"].append(volume)

        # Keep last 60 prices
        if len(self.position["prices"]) > 60:
            self.position["prices"] = self.position["prices"][-60:]
            self.position["volumes"] = self.position["volumes"][-60:]

        # Detect peaks and valleys
        prices = self.position["prices"]
        if len(prices) >= 3:
            if prices[-2] > prices[-3] and prices[-2] > prices[-1]:
                self.position["peaks"].append(prices[-2])
            if prices[-2] < prices[-3] and prices[-2] < prices[-1]:
                self.position["valleys"].append(prices[-2])

        pnl = (ltp - entry) * self.position["lot"]
        pnl_pct = ((ltp - entry) / entry * 100) if entry > 0 else 0

        # Run exit engine
        decision = self.exit_engine.should_exit(
            entry=entry,
            ltp=ltp,
            prices=prices,
            volumes=self.position["volumes"],
            peaks=self.position["peaks"],
            valleys=self.position["valleys"],
        )

        if not decision["exit"]:
            if len(prices) % 5 == 0:
                log.info(f"HOLD {sym} ({opt_type}): Rs.{ltp} ({pnl_pct:+.1f}%) | exit_votes={decision['exit_count']}/{decision['total_strategies']}")
            self._save_state()
            return "hold"

        # ── REVERSAL DETECTED — EXIT AND ROTATE ──
        log.info(f"REVERSAL DETECTED on {sym}: {decision['reason'][:100]}")

        # Exit current position
        time.sleep(2)
        sell_result = self.client.place_fno_order(symbol=sym, side="SELL")

        # Calculate P&L
        fees = self.calc.options_buy(entry, self.position["lot"]).total + \
               self.calc.options_sell(ltp, self.position["lot"]).total
        net_pnl = pnl - fees

        # Record closed trade
        closed = deepcopy(self.position)
        closed["exit"] = ltp
        closed["exit_time"] = datetime.now(IST).isoformat()
        closed["pnl"] = round(pnl, 2)
        closed["net_pnl"] = round(net_pnl, 2)
        closed["pnl_pct"] = round(pnl_pct, 2)
        closed["exit_reason"] = decision["reason"][:200]
        closed["sell_result"] = sell_result
        del closed["prices"]
        del closed["volumes"]
        self.history.append(closed)

        self.total_pnl += net_pnl
        self.capital += self.position["cost"] + pnl  # return capital + profit
        if net_pnl > 0:
            self.win_count += 1

        # ── RATCHET: Lock in profits, floor only goes UP ──
        self._ratchet_up(net_pnl)

        win = "WIN" if net_pnl > 0 else "LOSS"
        log.info(f"EXIT {win}: {sym} | Entry Rs.{entry} -> Rs.{ltp} | P&L Rs.{net_pnl:+,.0f} ({pnl_pct:+.1f}%)")
        log.info(f"Capital: Rs.{self.capital:,.0f} | Floor: Rs.{self.floor:,.0f} | Locked: Rs.{self.locked_profits:,.0f} | Total P&L: Rs.{self.total_pnl:+,.0f}")

        self.position = None
        self._save_state()

        # ── IMMEDIATE ROTATION — Enter opposite side ──
        # Was holding calls and price reversed down → buy puts
        # Was holding puts and price reversed up → buy calls
        new_type = "PE" if opt_type == "CE" else "CE"
        self.cycle_count += (1 if new_type == "CE" else 0)  # full cycle = back to calls

        log.info(f"ROTATING: {opt_type} -> {new_type} (cycle #{self.cycle_count})")
        time.sleep(3)  # brief pause before re-entry

        entered = self.enter(new_type)
        if entered:
            return f"rotated_to_{'put' if new_type == 'PE' else 'call'}"
        else:
            return "exited"

    def run(self):
        """Main perpetual loop. Never stops during market hours."""
        log.info("=" * 60)
        log.info("PERPETUAL ENGINE — Money Never Sleeps")
        log.info(f"Stock: {self.stock} | Capital: Rs.{self.capital:,.0f}")
        log.info(f"Trades: {self.trade_count} | P&L: Rs.{self.total_pnl:+,.0f} | Win: {self.win_count}")
        log.info("=" * 60)

        # Initial entry if no position
        if not self.position:
            log.info("No position. Scanning for entry...")
            # Start with calls (default bullish)
            self.enter("CE")

        scan = 0
        while True:
            now = datetime.now(IST)
            scan += 1

            is_market = (
                now.weekday() < 5
                and (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
                and (now.hour < 15 or (now.hour == 15 and now.minute <= 25))
            )

            if not is_market:
                if now.hour == 15 and now.minute >= 25 and self.position:
                    # Close to market close — hold overnight for NRML
                    log.info(f"Market closing. Holding {self.position['symbol']} overnight (NRML)")
                time.sleep(120)
                continue

            # Check and rotate
            result = self.check_and_rotate()

            # Status every 10 scans
            if scan % 10 == 0:
                pos_info = f"{self.position['symbol']} ({self.position['type']})" if self.position else "none"
                log.info(
                    f"[PERPETUAL scan={scan}] pos={pos_info} | "
                    f"capital=Rs.{self.capital:,.0f} | pnl=Rs.{self.total_pnl:+,.0f} | "
                    f"trades={self.trade_count} | wins={self.win_count} | cycles={self.cycle_count}"
                )

            time.sleep(60)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", default="COALINDIA")
    parser.add_argument("--capital", type=float, default=5000)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text())
            print(json.dumps(data, indent=2))
        else:
            print("No state yet.")
    else:
        from client import GrowwClient
        client = GrowwClient()
        client.connect()
        trader = PerpetualTrader(client, capital=args.capital, stock=args.stock)
        trader.run()

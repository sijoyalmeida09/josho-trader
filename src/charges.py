"""
charges.py — CA-Grade Groww Charge Calculator
==============================================
Accurate to the paisa. Updated for post-April 2026 Budget rates.

STT hiked from 0.0625% to 0.15% on F&O options (Budget 2026).
Exchange transaction charges updated to NSE latest schedule.

Usage:
    from charges import ChargeCalculator
    calc = ChargeCalculator()

    # Options round-trip
    result = calc.options_round_trip(buy_premium=2.56, sell_premium=7.68, lot_size=1250)
    print(result)

    # Single leg
    buy = calc.options_buy(premium=2.56, lot_size=1250)
    sell = calc.options_sell(premium=7.68, lot_size=1250)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ChargeBreakdown:
    """Immutable breakdown of charges for one leg of a trade."""
    turnover: float
    brokerage: float
    stt: float
    exchange_txn: float
    sebi: float
    ipft: float
    stamp_duty: float
    gst: float
    total: float

    def __str__(self):
        return (
            f"Turnover: Rs.{self.turnover:,.2f}\n"
            f"  Brokerage:    Rs.{self.brokerage:,.2f}\n"
            f"  STT:          Rs.{self.stt:,.2f}\n"
            f"  Exchange Txn: Rs.{self.exchange_txn:,.2f}\n"
            f"  SEBI:         Rs.{self.sebi:,.2f}\n"
            f"  IPFT:         Rs.{self.ipft:,.2f}\n"
            f"  Stamp Duty:   Rs.{self.stamp_duty:,.2f}\n"
            f"  GST (18%):    Rs.{self.gst:,.2f}\n"
            f"  TOTAL:        Rs.{self.total:,.2f}"
        )


@dataclass(frozen=True)
class RoundTripResult:
    """Immutable result of a buy+sell round trip."""
    buy_turnover: float
    sell_turnover: float
    buy_charges: ChargeBreakdown
    sell_charges: ChargeBreakdown
    gross_profit: float
    total_charges: float
    net_profit: float
    charges_pct_of_profit: float
    charges_pct_of_turnover: float
    breakeven_move_pct: float

    def __str__(self):
        return (
            f"=== ROUND TRIP ===\n"
            f"Buy turnover:  Rs.{self.buy_turnover:,.2f}\n"
            f"Sell turnover: Rs.{self.sell_turnover:,.2f}\n"
            f"Gross profit:  Rs.{self.gross_profit:+,.2f}\n"
            f"Total charges: Rs.{self.total_charges:,.2f}\n"
            f"Net profit:    Rs.{self.net_profit:+,.2f}\n"
            f"Charges/Profit: {self.charges_pct_of_profit:.2f}%\n"
            f"Charges/Turnover: {self.charges_pct_of_turnover:.2f}%\n"
            f"Breakeven move: +{self.breakeven_move_pct:.2f}%\n"
            f"\n--- BUY LEG ---\n{self.buy_charges}\n"
            f"\n--- SELL LEG ---\n{self.sell_charges}"
        )


class ChargeCalculator:
    """
    Groww brokerage charge calculator — post-April 2026 Budget rates.

    All rates are per NSE. BSE rates differ slightly but we trade on NSE.
    """

    # Brokerage
    BROKERAGE_FLAT = 20.0        # Rs.20 per executed order (F&O)
    BROKERAGE_PCT = 0.001        # 0.1% for equity (whichever is lower)

    # STT (Securities Transaction Tax) — POST APRIL 2026 BUDGET
    STT_OPTIONS_SELL = 0.0015    # 0.15% on premium (sell side only)
    STT_OPTIONS_BUY = 0.0        # NIL on buy
    STT_OPTIONS_EXERCISE = 0.0015  # 0.15% on intrinsic value if exercised ITM
    STT_FUTURES_SELL = 0.0005    # 0.05% on sell side
    STT_EQUITY_DELIVERY = 0.001  # 0.1% both sides
    STT_EQUITY_INTRADAY = 0.00025  # 0.025% sell side only

    # Exchange Transaction Charges (NSE)
    EXCHANGE_TXN_OPTIONS = 0.0003503   # 0.03503% on premium
    EXCHANGE_TXN_FUTURES = 0.0000173   # 0.00173% on notional
    EXCHANGE_TXN_EQUITY = 0.0000297    # 0.00297% on turnover

    # Regulatory
    SEBI_FEE = 0.000001          # 0.0001% (Rs.10 per crore)
    IPFT_FEE = 0.000001          # 0.0001% (NSE only)

    # Stamp Duty (buy side only)
    STAMP_OPTIONS = 0.00003      # 0.003%
    STAMP_FUTURES = 0.00002      # 0.002%
    STAMP_EQUITY_DELIVERY = 0.00015  # 0.015%
    STAMP_EQUITY_INTRADAY = 0.00003  # 0.003%

    # GST
    GST_RATE = 0.18              # 18% on (brokerage + exchange txn + SEBI + IPFT)

    # DP Charges (delivery sell only)
    DP_GROWW = 16.50
    DP_CDSL_MALE = 3.50
    DP_CDSL_FEMALE = 3.25

    # ── Options ──────────────────────────────────────────

    def options_buy(self, premium: float, lot_size: int) -> ChargeBreakdown:
        """Calculate charges for BUYING an option."""
        turnover = premium * lot_size
        brokerage = min(self.BROKERAGE_FLAT, turnover * self.BROKERAGE_PCT)
        stt = 0.0  # NIL on buy
        exchange_txn = turnover * self.EXCHANGE_TXN_OPTIONS
        sebi = turnover * self.SEBI_FEE
        ipft = turnover * self.IPFT_FEE
        stamp_duty = turnover * self.STAMP_OPTIONS
        gst = (brokerage + exchange_txn + sebi + ipft) * self.GST_RATE
        total = brokerage + stt + exchange_txn + sebi + ipft + stamp_duty + gst

        return ChargeBreakdown(
            turnover=turnover, brokerage=round(brokerage, 2),
            stt=round(stt, 2), exchange_txn=round(exchange_txn, 2),
            sebi=round(sebi, 2), ipft=round(ipft, 2),
            stamp_duty=round(stamp_duty, 2), gst=round(gst, 2),
            total=round(total, 2),
        )

    def options_sell(self, premium: float, lot_size: int) -> ChargeBreakdown:
        """Calculate charges for SELLING an option."""
        turnover = premium * lot_size
        brokerage = min(self.BROKERAGE_FLAT, turnover * self.BROKERAGE_PCT)
        stt = turnover * self.STT_OPTIONS_SELL  # 0.15% — the big one
        exchange_txn = turnover * self.EXCHANGE_TXN_OPTIONS
        sebi = turnover * self.SEBI_FEE
        ipft = turnover * self.IPFT_FEE
        stamp_duty = 0.0  # NIL on sell side
        gst = (brokerage + exchange_txn + sebi + ipft) * self.GST_RATE
        total = brokerage + stt + exchange_txn + sebi + ipft + stamp_duty + gst

        return ChargeBreakdown(
            turnover=turnover, brokerage=round(brokerage, 2),
            stt=round(stt, 2), exchange_txn=round(exchange_txn, 2),
            sebi=round(sebi, 2), ipft=round(ipft, 2),
            stamp_duty=round(stamp_duty, 2), gst=round(gst, 2),
            total=round(total, 2),
        )

    def options_round_trip(
        self, buy_premium: float, sell_premium: float, lot_size: int
    ) -> RoundTripResult:
        """Full buy + sell round trip for options."""
        buy = self.options_buy(buy_premium, lot_size)
        sell = self.options_sell(sell_premium, lot_size)
        gross_profit = (sell_premium - buy_premium) * lot_size
        total_charges = buy.total + sell.total
        net_profit = gross_profit - total_charges

        # Breakeven: how much premium must move to cover charges
        breakeven_move = total_charges / lot_size
        breakeven_pct = (breakeven_move / buy_premium) * 100 if buy_premium > 0 else 0

        charges_pct_profit = (total_charges / gross_profit * 100) if gross_profit > 0 else float('inf')
        charges_pct_turnover = (total_charges / buy.turnover * 100) if buy.turnover > 0 else 0

        return RoundTripResult(
            buy_turnover=buy.turnover, sell_turnover=sell.turnover,
            buy_charges=buy, sell_charges=sell,
            gross_profit=round(gross_profit, 2),
            total_charges=round(total_charges, 2),
            net_profit=round(net_profit, 2),
            charges_pct_of_profit=round(charges_pct_profit, 2),
            charges_pct_of_turnover=round(charges_pct_turnover, 2),
            breakeven_move_pct=round(breakeven_pct, 2),
        )

    def options_exercise_stt(self, intrinsic_value: float, lot_size: int) -> float:
        """STT if option is exercised ITM at expiry. ALWAYS sell before expiry to avoid this."""
        return round(intrinsic_value * lot_size * self.STT_OPTIONS_EXERCISE, 2)

    # ── Equity ───────────────────────────────────────────

    def equity_intraday_buy(self, price: float, qty: int) -> ChargeBreakdown:
        turnover = price * qty
        brokerage = min(self.BROKERAGE_FLAT, turnover * self.BROKERAGE_PCT)
        stt = 0.0  # NIL on buy for intraday
        exchange_txn = turnover * self.EXCHANGE_TXN_EQUITY
        sebi = turnover * self.SEBI_FEE
        ipft = turnover * self.IPFT_FEE
        stamp_duty = turnover * self.STAMP_EQUITY_INTRADAY
        gst = (brokerage + exchange_txn + sebi + ipft) * self.GST_RATE
        total = brokerage + stt + exchange_txn + sebi + ipft + stamp_duty + gst
        return ChargeBreakdown(
            turnover=turnover, brokerage=round(brokerage, 2),
            stt=round(stt, 2), exchange_txn=round(exchange_txn, 2),
            sebi=round(sebi, 2), ipft=round(ipft, 2),
            stamp_duty=round(stamp_duty, 2), gst=round(gst, 2),
            total=round(total, 2),
        )

    def equity_intraday_sell(self, price: float, qty: int) -> ChargeBreakdown:
        turnover = price * qty
        brokerage = min(self.BROKERAGE_FLAT, turnover * self.BROKERAGE_PCT)
        stt = turnover * self.STT_EQUITY_INTRADAY
        exchange_txn = turnover * self.EXCHANGE_TXN_EQUITY
        sebi = turnover * self.SEBI_FEE
        ipft = turnover * self.IPFT_FEE
        stamp_duty = 0.0
        gst = (brokerage + exchange_txn + sebi + ipft) * self.GST_RATE
        total = brokerage + stt + exchange_txn + sebi + ipft + stamp_duty + gst
        return ChargeBreakdown(
            turnover=turnover, brokerage=round(brokerage, 2),
            stt=round(stt, 2), exchange_txn=round(exchange_txn, 2),
            sebi=round(sebi, 2), ipft=round(ipft, 2),
            stamp_duty=round(stamp_duty, 2), gst=round(gst, 2),
            total=round(total, 2),
        )

    def equity_intraday_round_trip(
        self, buy_price: float, sell_price: float, qty: int
    ) -> RoundTripResult:
        buy = self.equity_intraday_buy(buy_price, qty)
        sell = self.equity_intraday_sell(sell_price, qty)
        gross_profit = (sell_price - buy_price) * qty
        total_charges = buy.total + sell.total
        net_profit = gross_profit - total_charges
        breakeven_move = total_charges / qty
        breakeven_pct = (breakeven_move / buy_price) * 100 if buy_price > 0 else 0
        charges_pct_profit = (total_charges / gross_profit * 100) if gross_profit > 0 else float('inf')
        charges_pct_turnover = (total_charges / buy.turnover * 100) if buy.turnover > 0 else 0

        return RoundTripResult(
            buy_turnover=buy.turnover, sell_turnover=sell.turnover,
            buy_charges=buy, sell_charges=sell,
            gross_profit=round(gross_profit, 2),
            total_charges=round(total_charges, 2),
            net_profit=round(net_profit, 2),
            charges_pct_of_profit=round(charges_pct_profit, 2),
            charges_pct_of_turnover=round(charges_pct_turnover, 2),
            breakeven_move_pct=round(breakeven_pct, 2),
        )

    # ── Quick Estimates (for brain.py compatibility) ─────

    def estimate_options_fees(self, buy_turnover: float) -> float:
        """Quick estimate of total round-trip fees for options.
        Assumes sell at same premium (worst case for fee %).
        Used by brain.py for trade filtering."""
        buy = self.options_buy(1.0, 1)  # per-rupee rates
        sell = self.options_sell(1.0, 1)
        # Scale: buy charges proportional to buy turnover
        # Sell charges proportional to sell turnover (assume same as buy for estimate)
        return round((buy.total + sell.total) * buy_turnover, 2)

    def estimate_equity_fees(self, turnover: float) -> float:
        """Quick estimate of total round-trip fees for equity intraday."""
        buy = self.equity_intraday_buy(1.0, 1)
        sell = self.equity_intraday_sell(1.0, 1)
        return round((buy.total + sell.total) * turnover, 2)

    def min_profitable_premium_move(self, buy_premium: float, lot_size: int) -> float:
        """Minimum premium increase needed to break even after all charges.
        Critical for deciding if a trade is worth taking."""
        # Binary search for breakeven sell premium
        low, high = buy_premium, buy_premium * 3
        for _ in range(50):
            mid = (low + high) / 2
            result = self.options_round_trip(buy_premium, mid, lot_size)
            if result.net_profit > 0:
                high = mid
            else:
                low = mid
        return round(high - buy_premium, 4)


# ── Convenience ──────────────────────────────────────────

_calc = ChargeCalculator()

def estimate_fno_fees(buy_premium: float, lot_size: int, sell_premium: Optional[float] = None) -> float:
    """Quick estimate for brain.py — returns total round-trip charges."""
    if sell_premium is None:
        sell_premium = buy_premium  # worst case
    result = _calc.options_round_trip(buy_premium, sell_premium, lot_size)
    return result.total_charges

def estimate_equity_fees(buy_price: float, qty: int) -> float:
    """Quick estimate for brain.py — returns total round-trip charges."""
    result = _calc.equity_intraday_round_trip(buy_price, buy_price, qty)
    return result.total_charges

def breakeven_pct(buy_premium: float, lot_size: int) -> float:
    """What % must premium move to break even?"""
    move = _calc.min_profitable_premium_move(buy_premium, lot_size)
    return round((move / buy_premium) * 100, 2) if buy_premium > 0 else 0


if __name__ == "__main__":
    calc = ChargeCalculator()

    print("=" * 60)
    print("ADANIPOWER26JUN210CE — Round Trip Analysis")
    print("=" * 60)

    # Current virtual trade
    result = calc.options_round_trip(
        buy_premium=2.56, sell_premium=7.68, lot_size=1250
    )
    print(result)

    print(f"\nMinimum premium move to break even: Rs.{calc.min_profitable_premium_move(2.56, 1250)}")
    print(f"Breakeven move: +{breakeven_pct(2.56, 1250)}%")

    print("\n" + "=" * 60)
    print("FEE IMPACT BY PREMIUM LEVEL")
    print("=" * 60)
    for prem in [0.50, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]:
        be = breakeven_pct(prem, 1250)
        print(f"  Premium Rs.{prem:5.1f} | Breakeven: +{be:5.2f}% | Min move: Rs.{calc.min_profitable_premium_move(prem, 1250):.4f}")

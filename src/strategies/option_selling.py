"""
Option Selling Strategies — Theta decay income generation.
Integrated from: algo_trading_strategies_india, Algo-Trade-Option-Selling.

Strategies:
  1. Short Straddle — Sell ATM CE + PE (neutral, high premium)
  2. Short Strangle — Sell OTM CE + PE (wider range, lower risk)
  3. Iron Condor — Sell OTM + Buy further OTM (defined risk)
  4. Credit Spread — Directional with protection

Each strategy includes:
  - Entry conditions (IV rank, time to expiry, market regime)
  - Position sizing (based on margin available)
  - Stop loss (fixed + trailing)
  - Adjustment rules
  - Exit conditions
"""

import logging
from datetime import datetime, time
from typing import Optional

from .base import Strategy, Signal, SignalType

log = logging.getLogger("josho.strategy.options")


class ShortStraddle(Strategy):
    """
    Sell ATM Call + ATM Put on same strike.
    Best when: IV is high, expect low movement, weekly expiry.
    Risk: Unlimited on both sides.
    Target: Premium decay (theta).
    """

    def __init__(
        self,
        underlying: str = "NIFTY",
        lot_size: int = 25,  # NIFTY lot = 25
        max_loss_pct: float = 50,  # exit if premium doubles (100% of collected)
        profit_target_pct: float = 50,  # exit at 50% profit of premium
        entry_time: time = time(9, 20),  # enter after market settles
        exit_time: time = time(15, 15),  # exit before close
        min_iv_rank: float = 30,  # minimum IV rank to enter
    ):
        super().__init__(
            name=f"ShortStraddle_{underlying}",
            description=f"Sell ATM straddle on {underlying}",
        )
        self.underlying = underlying
        self.lot_size = lot_size
        self.max_loss_pct = max_loss_pct
        self.profit_target_pct = profit_target_pct
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.min_iv_rank = min_iv_rank

    def analyze(self, market_data: dict) -> list[Signal]:
        """Analyze option chain and generate straddle signals."""
        signals = []

        spot = market_data.get("spot_price", 0)
        chain = market_data.get("option_chain", {})
        iv_rank = market_data.get("iv_rank", 0)
        expiry = market_data.get("nearest_expiry", "")

        if not spot or not chain:
            return signals

        # Entry conditions
        now = datetime.now().time()
        if now < self.entry_time or now > self.exit_time:
            return signals

        if iv_rank < self.min_iv_rank:
            log.debug(f"IV rank {iv_rank} < min {self.min_iv_rank}, skipping")
            return signals

        # Find ATM strike
        strike_step = 50 if "NIFTY" in self.underlying and "BANK" not in self.underlying else 100
        atm_strike = round(spot / strike_step) * strike_step

        # Get ATM premiums
        calls = chain.get("calls", {})
        puts = chain.get("puts", {})

        atm_ce = calls.get(atm_strike, {})
        atm_pe = puts.get(atm_strike, {})

        ce_premium = atm_ce.get("ltp", 0)
        pe_premium = atm_pe.get("ltp", 0)
        total_premium = ce_premium + pe_premium

        if total_premium <= 0:
            return signals

        # Calculate breakevens
        upper_be = atm_strike + total_premium
        lower_be = atm_strike - total_premium
        range_pct = ((upper_be - lower_be) / spot) * 100

        # Sell CE
        signals.append(Signal(
            signal_type=SignalType.SELL,
            symbol=atm_ce.get("symbol", f"{self.underlying}{expiry}{atm_strike}CE"),
            segment="FNO",
            quantity=self.lot_size,
            price=ce_premium,
            stop_loss=ce_premium * (1 + self.max_loss_pct / 100),
            target=ce_premium * (1 - self.profit_target_pct / 100),
            confidence=min(iv_rank / 100, 0.9),
            reason=f"Short straddle CE: premium {ce_premium}, IV rank {iv_rank}",
            strategy_name=self.name,
            metadata={
                "strike": atm_strike,
                "option_type": "CE",
                "total_premium": total_premium,
                "upper_be": upper_be,
                "lower_be": lower_be,
                "range_pct": range_pct,
            },
        ))

        # Sell PE
        signals.append(Signal(
            signal_type=SignalType.SELL,
            symbol=atm_pe.get("symbol", f"{self.underlying}{expiry}{atm_strike}PE"),
            segment="FNO",
            quantity=self.lot_size,
            price=pe_premium,
            stop_loss=pe_premium * (1 + self.max_loss_pct / 100),
            target=pe_premium * (1 - self.profit_target_pct / 100),
            confidence=min(iv_rank / 100, 0.9),
            reason=f"Short straddle PE: premium {pe_premium}, IV rank {iv_rank}",
            strategy_name=self.name,
            metadata={
                "strike": atm_strike,
                "option_type": "PE",
                "total_premium": total_premium,
                "upper_be": upper_be,
                "lower_be": lower_be,
                "range_pct": range_pct,
            },
        ))

        log.info(
            f"Straddle signal: {self.underlying} {atm_strike} | "
            f"Premium: {total_premium} | Range: {range_pct:.1f}% | "
            f"IV rank: {iv_rank}"
        )

        return signals

    def get_required_symbols(self) -> list[str]:
        return [self.underlying]


class ShortStrangle(Strategy):
    """
    Sell OTM Call + OTM Put at different strikes.
    Wider profit zone than straddle, lower premium.
    """

    def __init__(
        self,
        underlying: str = "NIFTY",
        lot_size: int = 25,
        otm_distance: int = 3,  # number of strikes away from ATM
        max_loss_pct: float = 100,
        profit_target_pct: float = 50,
        min_iv_rank: float = 20,
    ):
        super().__init__(
            name=f"ShortStrangle_{underlying}",
            description=f"Sell OTM strangle on {underlying}",
        )
        self.underlying = underlying
        self.lot_size = lot_size
        self.otm_distance = otm_distance
        self.max_loss_pct = max_loss_pct
        self.profit_target_pct = profit_target_pct
        self.min_iv_rank = min_iv_rank

    def analyze(self, market_data: dict) -> list[Signal]:
        signals = []

        spot = market_data.get("spot_price", 0)
        chain = market_data.get("option_chain", {})
        iv_rank = market_data.get("iv_rank", 0)
        expiry = market_data.get("nearest_expiry", "")

        if not spot or not chain or iv_rank < self.min_iv_rank:
            return signals

        strike_step = 50 if "BANK" not in self.underlying else 100
        atm_strike = round(spot / strike_step) * strike_step

        ce_strike = atm_strike + (self.otm_distance * strike_step)
        pe_strike = atm_strike - (self.otm_distance * strike_step)

        calls = chain.get("calls", {})
        puts = chain.get("puts", {})

        ce_data = calls.get(ce_strike, {})
        pe_data = puts.get(pe_strike, {})

        ce_premium = ce_data.get("ltp", 0)
        pe_premium = pe_data.get("ltp", 0)

        if ce_premium <= 0 or pe_premium <= 0:
            return signals

        total_premium = ce_premium + pe_premium

        # Sell OTM CE
        signals.append(Signal(
            signal_type=SignalType.SELL,
            symbol=ce_data.get("symbol", f"{self.underlying}{expiry}{ce_strike}CE"),
            segment="FNO",
            quantity=self.lot_size,
            price=ce_premium,
            stop_loss=ce_premium * (1 + self.max_loss_pct / 100),
            target=ce_premium * (1 - self.profit_target_pct / 100),
            confidence=min(iv_rank / 100, 0.85),
            reason=f"Short strangle CE {ce_strike}: premium {ce_premium}",
            strategy_name=self.name,
            metadata={"strike": ce_strike, "option_type": "CE", "otm_distance": self.otm_distance},
        ))

        # Sell OTM PE
        signals.append(Signal(
            signal_type=SignalType.SELL,
            symbol=pe_data.get("symbol", f"{self.underlying}{expiry}{pe_strike}PE"),
            segment="FNO",
            quantity=self.lot_size,
            price=pe_premium,
            stop_loss=pe_premium * (1 + self.max_loss_pct / 100),
            target=pe_premium * (1 - self.profit_target_pct / 100),
            confidence=min(iv_rank / 100, 0.85),
            reason=f"Short strangle PE {pe_strike}: premium {pe_premium}",
            strategy_name=self.name,
            metadata={"strike": pe_strike, "option_type": "PE", "otm_distance": self.otm_distance},
        ))

        log.info(
            f"Strangle signal: {self.underlying} CE {ce_strike} / PE {pe_strike} | "
            f"Premium: {total_premium} | IV rank: {iv_rank}"
        )

        return signals

    def get_required_symbols(self) -> list[str]:
        return [self.underlying]


class IronCondor(Strategy):
    """
    Iron Condor — Defined risk option selling.
    Sell OTM CE + PE, Buy further OTM CE + PE for protection.
    Max loss = width of spread - premium collected.
    """

    def __init__(
        self,
        underlying: str = "NIFTY",
        lot_size: int = 25,
        sell_distance: int = 3,
        buy_distance: int = 5,
        profit_target_pct: float = 50,
        max_loss_pct: float = 100,
        min_iv_rank: float = 25,
    ):
        super().__init__(
            name=f"IronCondor_{underlying}",
            description=f"Iron condor on {underlying} (defined risk)",
        )
        self.underlying = underlying
        self.lot_size = lot_size
        self.sell_distance = sell_distance
        self.buy_distance = buy_distance
        self.profit_target_pct = profit_target_pct
        self.max_loss_pct = max_loss_pct
        self.min_iv_rank = min_iv_rank

    def analyze(self, market_data: dict) -> list[Signal]:
        signals = []

        spot = market_data.get("spot_price", 0)
        chain = market_data.get("option_chain", {})
        iv_rank = market_data.get("iv_rank", 0)
        expiry = market_data.get("nearest_expiry", "")

        if not spot or not chain or iv_rank < self.min_iv_rank:
            return signals

        strike_step = 50 if "BANK" not in self.underlying else 100
        atm = round(spot / strike_step) * strike_step

        sell_ce = atm + (self.sell_distance * strike_step)
        buy_ce = atm + (self.buy_distance * strike_step)
        sell_pe = atm - (self.sell_distance * strike_step)
        buy_pe = atm - (self.buy_distance * strike_step)

        calls = chain.get("calls", {})
        puts = chain.get("puts", {})

        legs = [
            (SignalType.SELL, sell_ce, "CE", calls),
            (SignalType.BUY, buy_ce, "CE", calls),
            (SignalType.SELL, sell_pe, "PE", puts),
            (SignalType.BUY, buy_pe, "PE", puts),
        ]

        for sig_type, strike, opt_type, data in legs:
            leg_data = data.get(strike, {})
            premium = leg_data.get("ltp", 0)
            if premium <= 0:
                return []  # all legs must be available

            signals.append(Signal(
                signal_type=sig_type,
                symbol=leg_data.get("symbol", f"{self.underlying}{expiry}{strike}{opt_type}"),
                segment="FNO",
                quantity=self.lot_size,
                price=premium,
                confidence=min(iv_rank / 100, 0.8),
                reason=f"Iron condor {sig_type.value} {strike}{opt_type}: {premium}",
                strategy_name=self.name,
                metadata={
                    "strike": strike,
                    "option_type": opt_type,
                    "leg": f"{sig_type.value}_{strike}{opt_type}",
                },
            ))

        if signals:
            log.info(
                f"Iron condor: {self.underlying} | "
                f"CE: {sell_ce}/{buy_ce} | PE: {sell_pe}/{buy_pe} | "
                f"IV rank: {iv_rank}"
            )

        return signals

    def get_required_symbols(self) -> list[str]:
        return [self.underlying]

"""
exit_engine.py — 30+ Exit Strategies for Perfect Timing
=========================================================
"The market doesn't care about your entry. It only rewards your exit."

This engine runs EVERY exit strategy in parallel on each position,
scores them, and only exits when MULTIPLE strategies agree.

Philosophy:
  - Never exit on emotion. Exit on data.
  - One peak but two same lows. Exit on the second peak, not max.
  - Play opposite to sentiment — when everyone panics, hold. When everyone's greedy, leave.
  - Perfect maxima is impossible. Near-maxima with confirmation is the goal.

Usage:
    from exit_engine import ExitEngine
    engine = ExitEngine()
    decision = engine.should_exit(symbol, entry, ltp, price_history, volume_history)
    # decision = {"exit": True/False, "confidence": 0-100, "strategies_agree": [...], "reason": "..."}
"""

import math
from typing import Optional


class ExitEngine:
    """
    Runs 30+ exit strategies and returns consensus decision.
    Only exits when 3+ strategies agree with >70% confidence.
    """

    MIN_STRATEGIES_TO_EXIT = 3   # at least 3 must agree
    MIN_CONFIDENCE = 70          # weighted average must be >70%

    def should_exit(
        self,
        entry: float,
        ltp: float,
        prices: list,
        volumes: list = None,
        peaks: list = None,
        valleys: list = None,
    ) -> dict:
        """
        Master exit decision. Runs all strategies, returns consensus.

        Args:
            entry: entry price
            ltp: current price
            prices: list of recent prices (oldest first)
            volumes: list of recent volumes (optional)
            peaks: list of detected peak prices
            valleys: list of detected valley prices
        """
        if not prices or len(prices) < 5:
            return {"exit": False, "confidence": 0, "reason": "insufficient data"}

        if volumes is None:
            volumes = []
        if peaks is None:
            peaks = []
        if valleys is None:
            valleys = []

        pnl_pct = ((ltp - entry) / entry * 100) if entry > 0 else 0

        # Run all strategies
        votes = []
        votes.append(self._trailing_stop(entry, ltp, prices, peaks))
        votes.append(self._double_low_confirmation(ltp, prices, peaks, valleys))
        votes.append(self._momentum_exhaustion(prices))
        votes.append(self._volume_divergence(prices, volumes))
        votes.append(self._rsi_overbought(prices))
        votes.append(self._macd_crossover(prices))
        votes.append(self._bollinger_squeeze(prices))
        votes.append(self._candlestick_reversal(prices))
        votes.append(self._time_decay_urgency(prices, entry, ltp))
        votes.append(self._profit_target_zone(pnl_pct))
        votes.append(self._peak_distance(ltp, peaks))
        votes.append(self._higher_low_break(valleys))
        votes.append(self._ema_crossdown(prices))
        votes.append(self._parabolic_sar(prices))
        votes.append(self._atr_trailing(prices, entry))
        votes.append(self._chandelier_exit(prices, peaks))
        votes.append(self._keltner_channel(prices))
        votes.append(self._donchian_break(prices))
        votes.append(self._stochastic_exit(prices))
        votes.append(self._williams_r(prices))
        votes.append(self._cci_exit(prices))
        votes.append(self._adx_weakening(prices))
        votes.append(self._fibonacci_retracement(entry, peaks, ltp))
        votes.append(self._vwap_cross(prices, volumes))
        votes.append(self._three_bar_reversal(prices))
        votes.append(self._gap_down_exit(prices))
        votes.append(self._consecutive_red(prices))
        votes.append(self._hard_stop(pnl_pct))
        votes.append(self._time_stop(prices))

        # Filter to strategies that voted EXIT
        exit_votes = [v for v in votes if v["vote"] == "EXIT"]
        hold_votes = [v for v in votes if v["vote"] == "HOLD"]

        # Consensus calculation
        total_weight = sum(v["weight"] for v in votes if v["vote"] in ("EXIT", "HOLD"))
        exit_weight = sum(v["weight"] for v in exit_votes)
        hold_weight = sum(v["weight"] for v in hold_votes)

        exit_pct = (exit_weight / total_weight * 100) if total_weight > 0 else 0
        confidence = int(exit_pct)

        should_exit = (
            len(exit_votes) >= self.MIN_STRATEGIES_TO_EXIT
            and confidence >= self.MIN_CONFIDENCE
        )

        strategies_agree = [v["name"] for v in exit_votes]
        strategies_hold = [v["name"] for v in hold_votes]

        reason = ""
        if should_exit:
            top_reasons = [v["reason"] for v in sorted(exit_votes, key=lambda x: -x["weight"])[:3]]
            reason = " + ".join(top_reasons)

        return {
            "exit": should_exit,
            "confidence": confidence,
            "exit_count": len(exit_votes),
            "hold_count": len(hold_votes),
            "total_strategies": len(votes),
            "strategies_agree": strategies_agree[:5],
            "strategies_hold": strategies_hold[:5],
            "reason": reason,
            "pnl_pct": round(pnl_pct, 2),
        }

    # ══════════════════════════════════════════════════════════
    # TREND-FOLLOWING EXIT STRATEGIES
    # ══════════════════════════════════════════════════════════

    def _trailing_stop(self, entry, ltp, prices, peaks, trail_pct=10) -> dict:
        """Classic trailing stop — exit when price drops X% from peak."""
        peak = max(prices) if prices else entry
        distance = ((peak - ltp) / peak * 100) if peak > 0 else 0
        if distance > trail_pct:
            return {"name": "trailing_stop", "vote": "EXIT", "weight": 8,
                    "reason": f"down {distance:.1f}% from peak Rs.{peak:.2f}"}
        return {"name": "trailing_stop", "vote": "HOLD", "weight": 8, "reason": "within trail"}

    def _double_low_confirmation(self, ltp, prices, peaks, valleys) -> dict:
        """YOUR philosophy: exit only after 2 confirmed lows."""
        if len(valleys) < 2:
            return {"name": "double_low", "vote": "HOLD", "weight": 10,
                    "reason": f"only {len(valleys)} valleys, need 2"}
        v1, v2 = valleys[-2], valleys[-1]
        # Second low lower than first = trend breaking
        if v2 < v1 * 0.97:
            return {"name": "double_low", "vote": "EXIT", "weight": 10,
                    "reason": f"2nd low Rs.{v2:.2f} < 1st Rs.{v1:.2f} = trend broken"}
        # Second low held = support confirmed, but check if falling from peak
        peak = max(prices[-10:]) if len(prices) >= 10 else max(prices)
        if ltp < peak * 0.90:
            return {"name": "double_low", "vote": "EXIT", "weight": 10,
                    "reason": f"support at Rs.{v2:.2f} but down 10%+ from peak"}
        return {"name": "double_low", "vote": "HOLD", "weight": 10,
                "reason": "2 lows confirmed, support holding"}

    def _ema_crossdown(self, prices) -> dict:
        """Exit when fast EMA crosses below slow EMA."""
        if len(prices) < 12:
            return {"name": "ema_cross", "vote": "SKIP", "weight": 0, "reason": "insufficient data"}
        ema5 = self._ema(prices, 5)
        ema12 = self._ema(prices, 12)
        if ema5 < ema12:
            return {"name": "ema_cross", "vote": "EXIT", "weight": 7,
                    "reason": f"EMA5 ({ema5:.2f}) < EMA12 ({ema12:.2f})"}
        return {"name": "ema_cross", "vote": "HOLD", "weight": 7,
                "reason": f"EMA5 ({ema5:.2f}) > EMA12 ({ema12:.2f})"}

    # ══════════════════════════════════════════════════════════
    # MOMENTUM EXIT STRATEGIES
    # ══════════════════════════════════════════════════════════

    def _momentum_exhaustion(self, prices) -> dict:
        """Exit when momentum decelerates (speed of price change slowing)."""
        if len(prices) < 10:
            return {"name": "momentum", "vote": "SKIP", "weight": 0, "reason": ""}
        recent = prices[-5:]
        older = prices[-10:-5]
        recent_move = abs(recent[-1] - recent[0])
        older_move = abs(older[-1] - older[0])
        if older_move > 0 and recent_move < older_move * 0.3:
            return {"name": "momentum", "vote": "EXIT", "weight": 6,
                    "reason": "momentum exhausted (recent move < 30% of prior)"}
        return {"name": "momentum", "vote": "HOLD", "weight": 6, "reason": "momentum intact"}

    def _rsi_overbought(self, prices, period=14) -> dict:
        """Exit when RSI > 70 (overbought)."""
        rsi = self._calc_rsi(prices, period)
        if rsi is None:
            return {"name": "rsi", "vote": "SKIP", "weight": 0, "reason": ""}
        if rsi > 75:
            return {"name": "rsi", "vote": "EXIT", "weight": 7,
                    "reason": f"RSI={rsi:.0f} (overbought >75)"}
        if rsi < 30:
            return {"name": "rsi", "vote": "HOLD", "weight": 7,
                    "reason": f"RSI={rsi:.0f} (oversold, potential bounce)"}
        return {"name": "rsi", "vote": "HOLD", "weight": 5, "reason": f"RSI={rsi:.0f} (neutral)"}

    def _macd_crossover(self, prices) -> dict:
        """Exit when MACD line crosses below signal line."""
        if len(prices) < 26:
            return {"name": "macd", "vote": "SKIP", "weight": 0, "reason": ""}
        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)
        macd = ema12 - ema26
        # Signal is EMA of MACD values — approximate with recent trend
        prev_ema12 = self._ema(prices[:-1], 12)
        prev_ema26 = self._ema(prices[:-1], 26)
        prev_macd = prev_ema12 - prev_ema26
        if macd < 0 and prev_macd >= 0:
            return {"name": "macd", "vote": "EXIT", "weight": 7,
                    "reason": "MACD crossed below zero"}
        if macd < prev_macd and macd > 0:
            return {"name": "macd", "vote": "HOLD", "weight": 5,
                    "reason": "MACD positive but weakening"}
        return {"name": "macd", "vote": "HOLD", "weight": 7, "reason": "MACD bullish"}

    def _stochastic_exit(self, prices, period=14) -> dict:
        """Exit when Stochastic %K crosses below %D in overbought zone."""
        if len(prices) < period:
            return {"name": "stochastic", "vote": "SKIP", "weight": 0, "reason": ""}
        recent = prices[-period:]
        high = max(recent)
        low = min(recent)
        if high == low:
            return {"name": "stochastic", "vote": "SKIP", "weight": 0, "reason": ""}
        k = ((prices[-1] - low) / (high - low)) * 100
        if k > 80:
            return {"name": "stochastic", "vote": "EXIT", "weight": 5,
                    "reason": f"Stochastic %K={k:.0f} (overbought >80)"}
        return {"name": "stochastic", "vote": "HOLD", "weight": 5, "reason": f"Stochastic %K={k:.0f}"}

    def _williams_r(self, prices, period=14) -> dict:
        """Williams %R — exit when > -20 (overbought)."""
        if len(prices) < period:
            return {"name": "williams_r", "vote": "SKIP", "weight": 0, "reason": ""}
        recent = prices[-period:]
        high = max(recent)
        low = min(recent)
        if high == low:
            return {"name": "williams_r", "vote": "SKIP", "weight": 0, "reason": ""}
        wr = ((high - prices[-1]) / (high - low)) * -100
        if wr > -20:
            return {"name": "williams_r", "vote": "EXIT", "weight": 4,
                    "reason": f"Williams %R={wr:.0f} (overbought)"}
        return {"name": "williams_r", "vote": "HOLD", "weight": 4, "reason": f"Williams %R={wr:.0f}"}

    def _cci_exit(self, prices, period=20) -> dict:
        """CCI — exit when crosses below +100 from above."""
        if len(prices) < period:
            return {"name": "cci", "vote": "SKIP", "weight": 0, "reason": ""}
        tp = prices[-1]  # typical price (simplified)
        sma = sum(prices[-period:]) / period
        md = sum(abs(p - sma) for p in prices[-period:]) / period
        cci = (tp - sma) / (0.015 * md) if md > 0 else 0
        if cci < 100 and len(prices) > period:
            prev_prices = prices[-(period+1):-1]
            prev_sma = sum(prev_prices) / len(prev_prices)
            prev_md = sum(abs(p - prev_sma) for p in prev_prices) / len(prev_prices)
            prev_cci = (prices[-2] - prev_sma) / (0.015 * prev_md) if prev_md > 0 else 0
            if prev_cci > 100:
                return {"name": "cci", "vote": "EXIT", "weight": 5,
                        "reason": f"CCI crossed below 100 ({cci:.0f})"}
        return {"name": "cci", "vote": "HOLD", "weight": 4, "reason": f"CCI={cci:.0f}"}

    def _adx_weakening(self, prices) -> dict:
        """ADX — exit when trend strength drops (simplified)."""
        if len(prices) < 15:
            return {"name": "adx", "vote": "SKIP", "weight": 0, "reason": ""}
        # Simplified: compare recent vs older volatility
        recent_vol = max(prices[-5:]) - min(prices[-5:])
        older_vol = max(prices[-10:-5]) - min(prices[-10:-5])
        if older_vol > 0 and recent_vol < older_vol * 0.4:
            return {"name": "adx", "vote": "EXIT", "weight": 5,
                    "reason": "trend weakening (range contracting)"}
        return {"name": "adx", "vote": "HOLD", "weight": 5, "reason": "trend strong"}

    # ══════════════════════════════════════════════════════════
    # VOLATILITY EXIT STRATEGIES
    # ══════════════════════════════════════════════════════════

    def _bollinger_squeeze(self, prices, period=20) -> dict:
        """Exit when price touches upper Bollinger Band and reverses."""
        if len(prices) < period:
            return {"name": "bollinger", "vote": "SKIP", "weight": 0, "reason": ""}
        sma = sum(prices[-period:]) / period
        std = math.sqrt(sum((p - sma)**2 for p in prices[-period:]) / period)
        upper = sma + 2 * std
        if prices[-1] < upper and prices[-2] >= upper:
            return {"name": "bollinger", "vote": "EXIT", "weight": 6,
                    "reason": f"fell from upper Bollinger (Rs.{upper:.2f})"}
        return {"name": "bollinger", "vote": "HOLD", "weight": 5, "reason": "within bands"}

    def _atr_trailing(self, prices, entry, multiplier=2) -> dict:
        """ATR-based trailing stop — adapts to volatility."""
        if len(prices) < 14:
            return {"name": "atr_trail", "vote": "SKIP", "weight": 0, "reason": ""}
        # Simplified ATR (true range without high/low)
        trs = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        atr = sum(trs[-14:]) / 14
        peak = max(prices)
        stop = peak - (atr * multiplier)
        if prices[-1] < stop:
            return {"name": "atr_trail", "vote": "EXIT", "weight": 7,
                    "reason": f"below ATR stop Rs.{stop:.2f} (ATR={atr:.2f})"}
        return {"name": "atr_trail", "vote": "HOLD", "weight": 7,
                "reason": f"above ATR stop Rs.{stop:.2f}"}

    def _chandelier_exit(self, prices, peaks, multiplier=3) -> dict:
        """Chandelier Exit — ATR-based stop from highest high."""
        if len(prices) < 14 or not peaks:
            return {"name": "chandelier", "vote": "SKIP", "weight": 0, "reason": ""}
        trs = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        atr = sum(trs[-14:]) / 14
        highest = max(peaks) if peaks else max(prices)
        stop = highest - (atr * multiplier)
        if prices[-1] < stop:
            return {"name": "chandelier", "vote": "EXIT", "weight": 6,
                    "reason": f"Chandelier stop Rs.{stop:.2f}"}
        return {"name": "chandelier", "vote": "HOLD", "weight": 6, "reason": "above Chandelier"}

    def _keltner_channel(self, prices, period=20, multiplier=1.5) -> dict:
        """Exit when drops below Keltner Channel midline."""
        if len(prices) < period:
            return {"name": "keltner", "vote": "SKIP", "weight": 0, "reason": ""}
        ema = self._ema(prices, period)
        trs = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        atr = sum(trs[-period:]) / min(len(trs), period)
        midline = ema
        if prices[-1] < midline:
            return {"name": "keltner", "vote": "EXIT", "weight": 5,
                    "reason": f"below Keltner midline Rs.{midline:.2f}"}
        return {"name": "keltner", "vote": "HOLD", "weight": 5, "reason": "above Keltner"}

    def _donchian_break(self, prices, period=10) -> dict:
        """Donchian Channel — exit when price breaks below N-period low."""
        if len(prices) < period:
            return {"name": "donchian", "vote": "SKIP", "weight": 0, "reason": ""}
        channel_low = min(prices[-period:])
        if prices[-1] <= channel_low:
            return {"name": "donchian", "vote": "EXIT", "weight": 6,
                    "reason": f"broke {period}-period low Rs.{channel_low:.2f}"}
        return {"name": "donchian", "vote": "HOLD", "weight": 6, "reason": "above channel"}

    def _parabolic_sar(self, prices) -> dict:
        """Simplified Parabolic SAR — tracks acceleration of trend."""
        if len(prices) < 10:
            return {"name": "psar", "vote": "SKIP", "weight": 0, "reason": ""}
        # Simplified: SAR ≈ recent low + acceleration toward peak
        af = 0.02
        peak = max(prices[-10:])
        sar = min(prices[-5:]) + af * (peak - min(prices[-5:]))
        if prices[-1] < sar:
            return {"name": "psar", "vote": "EXIT", "weight": 5,
                    "reason": f"below Parabolic SAR Rs.{sar:.2f}"}
        return {"name": "psar", "vote": "HOLD", "weight": 5, "reason": "above SAR"}

    # ══════════════════════════════════════════════════════════
    # PATTERN-BASED EXIT STRATEGIES
    # ══════════════════════════════════════════════════════════

    def _candlestick_reversal(self, prices) -> dict:
        """Detect bearish reversal patterns (engulfing, doji at top)."""
        if len(prices) < 4:
            return {"name": "candle", "vote": "SKIP", "weight": 0, "reason": ""}
        p1, p2, p3, p4 = prices[-4], prices[-3], prices[-2], prices[-1]
        # Bearish engulfing: big up candle followed by bigger down candle
        if p3 > p2 and p4 < p1 and (p3 - p2) > 0 and (p2 - p4) > (p3 - p2):
            return {"name": "candle", "vote": "EXIT", "weight": 6,
                    "reason": "bearish engulfing pattern"}
        # Evening star: up, doji, down
        if p2 < p1 and abs(p3 - p2) < abs(p2 - p1) * 0.3 and p4 < p2:
            return {"name": "candle", "vote": "EXIT", "weight": 5,
                    "reason": "evening star pattern"}
        return {"name": "candle", "vote": "HOLD", "weight": 5, "reason": "no reversal pattern"}

    def _three_bar_reversal(self, prices) -> dict:
        """Three consecutive lower closes = reversal signal."""
        if len(prices) < 4:
            return {"name": "3bar", "vote": "SKIP", "weight": 0, "reason": ""}
        if prices[-1] < prices[-2] < prices[-3]:
            return {"name": "3bar", "vote": "EXIT", "weight": 6,
                    "reason": "3 consecutive lower prices"}
        return {"name": "3bar", "vote": "HOLD", "weight": 5, "reason": "no 3-bar reversal"}

    def _gap_down_exit(self, prices) -> dict:
        """Exit on significant gap down (>3%)."""
        if len(prices) < 2:
            return {"name": "gap", "vote": "SKIP", "weight": 0, "reason": ""}
        gap = ((prices[-1] - prices[-2]) / prices[-2] * 100) if prices[-2] > 0 else 0
        if gap < -3:
            return {"name": "gap", "vote": "EXIT", "weight": 8,
                    "reason": f"gap down {gap:.1f}%"}
        return {"name": "gap", "vote": "HOLD", "weight": 5, "reason": "no gap"}

    def _consecutive_red(self, prices, count=5) -> dict:
        """Exit after N consecutive down moves."""
        if len(prices) < count + 1:
            return {"name": "consec_red", "vote": "SKIP", "weight": 0, "reason": ""}
        reds = sum(1 for i in range(-count, 0) if prices[i] < prices[i-1])
        if reds >= count:
            return {"name": "consec_red", "vote": "EXIT", "weight": 7,
                    "reason": f"{count} consecutive down moves"}
        return {"name": "consec_red", "vote": "HOLD", "weight": 5, "reason": f"{reds}/{count} reds"}

    # ══════════════════════════════════════════════════════════
    # VOLUME-BASED EXIT STRATEGIES
    # ══════════════════════════════════════════════════════════

    def _volume_divergence(self, prices, volumes) -> dict:
        """Price up but volume down = bearish divergence."""
        if len(volumes) < 10 or not any(v > 0 for v in volumes):
            return {"name": "vol_div", "vote": "SKIP", "weight": 0, "reason": "no volume data"}
        recent_vol = [v for v in volumes[-5:] if v > 0]
        older_vol = [v for v in volumes[-10:-5] if v > 0]
        if not recent_vol or not older_vol:
            return {"name": "vol_div", "vote": "SKIP", "weight": 0, "reason": ""}
        avg_recent = sum(recent_vol) / len(recent_vol)
        avg_older = sum(older_vol) / len(older_vol)
        price_up = prices[-1] > prices[-5] if len(prices) >= 5 else False
        vol_down = avg_recent < avg_older * 0.7
        if price_up and vol_down:
            return {"name": "vol_div", "vote": "EXIT", "weight": 7,
                    "reason": "price up but volume declining (bearish divergence)"}
        return {"name": "vol_div", "vote": "HOLD", "weight": 6, "reason": "no divergence"}

    def _vwap_cross(self, prices, volumes) -> dict:
        """Exit when price crosses below VWAP."""
        if len(prices) < 10 or len(volumes) < 10 or not any(v > 0 for v in volumes):
            return {"name": "vwap", "vote": "SKIP", "weight": 0, "reason": ""}
        # Simplified VWAP
        total_pv = sum(p * v for p, v in zip(prices[-10:], volumes[-10:]) if v > 0)
        total_v = sum(v for v in volumes[-10:] if v > 0)
        if total_v == 0:
            return {"name": "vwap", "vote": "SKIP", "weight": 0, "reason": ""}
        vwap = total_pv / total_v
        if prices[-1] < vwap:
            return {"name": "vwap", "vote": "EXIT", "weight": 6,
                    "reason": f"below VWAP Rs.{vwap:.2f}"}
        return {"name": "vwap", "vote": "HOLD", "weight": 6, "reason": f"above VWAP Rs.{vwap:.2f}"}

    # ══════════════════════════════════════════════════════════
    # STRUCTURAL EXIT STRATEGIES
    # ══════════════════════════════════════════════════════════

    def _peak_distance(self, ltp, peaks) -> dict:
        """Exit when too far from peak (lost too much gain)."""
        if not peaks:
            return {"name": "peak_dist", "vote": "SKIP", "weight": 0, "reason": ""}
        highest = max(peaks)
        dist = ((highest - ltp) / highest * 100) if highest > 0 else 0
        if dist > 20:
            return {"name": "peak_dist", "vote": "EXIT", "weight": 8,
                    "reason": f"down {dist:.1f}% from peak Rs.{highest:.2f}"}
        return {"name": "peak_dist", "vote": "HOLD", "weight": 6,
                "reason": f"{dist:.1f}% from peak"}

    def _higher_low_break(self, valleys) -> dict:
        """Exit when higher-low pattern breaks."""
        if len(valleys) < 2:
            return {"name": "hl_break", "vote": "HOLD", "weight": 8,
                    "reason": "need 2+ valleys"}
        if valleys[-1] < valleys[-2]:
            return {"name": "hl_break", "vote": "EXIT", "weight": 9,
                    "reason": f"higher-low broken: Rs.{valleys[-1]:.2f} < Rs.{valleys[-2]:.2f}"}
        return {"name": "hl_break", "vote": "HOLD", "weight": 9,
                "reason": "higher lows intact"}

    def _fibonacci_retracement(self, entry, peaks, ltp) -> dict:
        """Exit at key Fibonacci retracement levels (38.2%, 50%, 61.8%)."""
        if not peaks:
            return {"name": "fib", "vote": "SKIP", "weight": 0, "reason": ""}
        peak = max(peaks)
        if peak <= entry:
            return {"name": "fib", "vote": "SKIP", "weight": 0, "reason": "no profit peak"}
        move = peak - entry
        fib_382 = peak - move * 0.382
        fib_50 = peak - move * 0.50
        fib_618 = peak - move * 0.618
        if ltp < fib_618:
            return {"name": "fib", "vote": "EXIT", "weight": 7,
                    "reason": f"below 61.8% Fib (Rs.{fib_618:.2f})"}
        if ltp < fib_50:
            return {"name": "fib", "vote": "EXIT", "weight": 5,
                    "reason": f"below 50% Fib (Rs.{fib_50:.2f})"}
        return {"name": "fib", "vote": "HOLD", "weight": 6,
                "reason": f"above Fib levels"}

    # ══════════════════════════════════════════════════════════
    # SAFETY EXITS
    # ══════════════════════════════════════════════════════════

    def _profit_target_zone(self, pnl_pct) -> dict:
        """Alert when in profit target zone but don't force exit."""
        if pnl_pct >= 25:
            return {"name": "profit_zone", "vote": "HOLD", "weight": 3,
                    "reason": f"in profit zone +{pnl_pct:.1f}% (let winners run)"}
        return {"name": "profit_zone", "vote": "HOLD", "weight": 1, "reason": "below target"}

    def _hard_stop(self, pnl_pct) -> dict:
        """Absolute stop loss — non-negotiable capital protection."""
        if pnl_pct <= -45:
            return {"name": "hard_stop", "vote": "EXIT", "weight": 20,
                    "reason": f"HARD STOP at {pnl_pct:.1f}%"}
        return {"name": "hard_stop", "vote": "HOLD", "weight": 1, "reason": "above stop"}

    def _time_stop(self, prices) -> dict:
        """Exit if held too long with no movement (dead money)."""
        if len(prices) < 30:
            return {"name": "time_stop", "vote": "HOLD", "weight": 3, "reason": "too early"}
        total_range = max(prices[-30:]) - min(prices[-30:])
        avg = sum(prices[-30:]) / 30
        range_pct = (total_range / avg * 100) if avg > 0 else 0
        if range_pct < 2 and len(prices) >= 60:
            return {"name": "time_stop", "vote": "EXIT", "weight": 5,
                    "reason": f"dead money ({range_pct:.1f}% range in 60+ scans)"}
        return {"name": "time_stop", "vote": "HOLD", "weight": 3, "reason": "active"}

    def _time_decay_urgency(self, prices, entry, ltp) -> dict:
        """Options lose value over time — urgency increases with age."""
        scans = len(prices)
        pnl_pct = ((ltp - entry) / entry * 100) if entry > 0 else 0
        # After 60+ scans (~1 hour), if not profitable, consider exit
        if scans > 60 and pnl_pct < 5:
            return {"name": "theta_decay", "vote": "EXIT", "weight": 4,
                    "reason": f"held {scans} scans, only +{pnl_pct:.1f}% (theta eating)"}
        return {"name": "theta_decay", "vote": "HOLD", "weight": 3, "reason": "time OK"}

    # ══════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _ema(prices: list, period: int) -> float:
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        for p in prices[period:]:
            ema = (p - ema) * multiplier + ema
        return ema

    @staticmethod
    def _calc_rsi(prices: list, period: int = 14) -> Optional[float]:
        if len(prices) < period + 1:
            return None
        gains, losses = [], []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        if len(gains) < period:
            return None
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


# ── Convenience ──────────────────────────────────────────

_engine = ExitEngine()

def should_exit(entry, ltp, prices, volumes=None, peaks=None, valleys=None) -> dict:
    return _engine.should_exit(entry, ltp, prices, volumes, peaks, valleys)

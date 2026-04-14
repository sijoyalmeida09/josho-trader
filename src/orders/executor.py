"""
Order Executor — Places, modifies, cancels orders via Groww API.
Every order goes through Risk Manager first.
Sends Telegram alerts for every trade.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from ..client import get_client
from ..risk.manager import RiskManager

log = logging.getLogger("josho.executor")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TRADE_LOG = Path(__file__).parent.parent.parent / "data" / "trades.json"


class OrderExecutor:
    """Execute trades with risk checks and Telegram notifications."""

    def __init__(self, risk: RiskManager):
        self.client = get_client()
        self.risk = risk
        self.trades: list[dict] = []
        self._load_trades()

    def execute(
        self,
        symbol: str,
        qty: int,
        side: str,  # "BUY" or "SELL"
        order_type: str = "MARKET",
        price: float = 0,
        product: str = "NRML",
        segment: str = "FNO",
        reason: str = "",
    ) -> dict:
        """
        Execute a trade after risk checks.
        Returns order response or rejection reason.
        """
        # Calculate approximate order value
        ltp = self.client.get_ltp(symbol, segment=segment) or price or 0
        order_value = ltp * qty

        # Risk check
        allowed, risk_msg = self.risk.can_trade(order_value)
        if not allowed:
            result = {
                "status": "REJECTED",
                "reason": risk_msg,
                "symbol": symbol,
                "side": side,
                "qty": qty,
            }
            self._alert(f"ORDER REJECTED\n{symbol} {side} {qty}\nReason: {risk_msg}")
            log.warning(f"Order rejected: {risk_msg}")
            return result

        # Paper trading — simulate
        if self.risk.limits.paper_trading:
            result = self._paper_trade(symbol, qty, side, ltp, reason)
            return result

        # Live execution
        log.info(f"Executing: {side} {qty}x {symbol} @ {order_type} {price}")
        response = self.client.place_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            price=price,
            product=product,
            segment=segment,
        )

        # Record
        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": ltp,
            "order_type": order_type,
            "product": product,
            "reason": reason,
            "response": response,
            "paper": False,
        }
        self.trades.append(trade)
        self._save_trades()

        # Alert
        status = response.get("status", "UNKNOWN")
        order_id = response.get("payload", {}).get("groww_order_id", "N/A")
        self._alert(
            f"{'BUY' if side == 'BUY' else 'SELL'} ORDER {'PLACED' if status == 'SUCCESS' else 'FAILED'}\n"
            f"Symbol: {symbol}\n"
            f"Qty: {qty} @ ₹{ltp}\n"
            f"Value: ₹{order_value:,.0f}\n"
            f"Order ID: {order_id}\n"
            f"Reason: {reason}"
        )

        self.risk.record_trade(0, is_open=True)
        return response

    def _paper_trade(self, symbol: str, qty: int, side: str, price: float, reason: str) -> dict:
        """Simulate a trade in paper mode."""
        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "value": price * qty,
            "reason": reason,
            "paper": True,
            "status": "SIMULATED",
        }
        self.trades.append(trade)
        self._save_trades()

        self._alert(
            f"PAPER TRADE\n"
            f"{'BUY' if side == 'BUY' else 'SELL'} {qty}x {symbol}\n"
            f"@ ₹{price:,.2f} = ₹{price * qty:,.0f}\n"
            f"Reason: {reason}"
        )

        self.risk.record_trade(0, is_open=True)
        log.info(f"Paper trade: {side} {qty}x {symbol} @ ₹{price}")
        return trade

    def close_position(
        self,
        symbol: str,
        qty: int,
        side: str,  # opposite of entry
        entry_price: float,
        reason: str = "Close",
    ) -> dict:
        """Close a position and record P&L."""
        ltp = self.client.get_ltp(symbol, segment="FNO") or 0
        pnl = (ltp - entry_price) * qty if side == "SELL" else (entry_price - ltp) * qty

        if self.risk.limits.paper_trading:
            result = self._paper_trade(symbol, qty, side, ltp, reason)
            result["pnl"] = pnl
        else:
            result = self.client.place_order(symbol=symbol, qty=qty, side=side)
            result["pnl"] = pnl

        self.risk.record_trade(pnl, is_open=False)

        self._alert(
            f"POSITION CLOSED\n"
            f"{side} {qty}x {symbol}\n"
            f"Entry: ₹{entry_price:,.2f} → Exit: ₹{ltp:,.2f}\n"
            f"P&L: ₹{pnl:,.0f} {'PROFIT' if pnl > 0 else 'LOSS'}\n"
            f"Reason: {reason}"
        )

        return result

    def _alert(self, message: str):
        """Send trade alert to Telegram."""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
        try:
            prefix = "JOSHO TRADER"
            text = f"*{prefix}*\n\n{message}\n\n_{datetime.now().strftime('%H:%M:%S')}_"
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": text,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
        except Exception as e:
            log.error(f"Telegram alert failed: {e}")

    def _save_trades(self):
        TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
        TRADE_LOG.write_text(
            json.dumps(self.trades[-100:], indent=2, default=str),
            encoding="utf-8",
        )

    def _load_trades(self):
        if TRADE_LOG.exists():
            try:
                self.trades = json.loads(TRADE_LOG.read_text(encoding="utf-8"))
            except Exception:
                self.trades = []

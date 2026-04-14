"""
JoSho Trader — F&O Trading Bot
================================
Main entry point. Connects to Groww, scans market, executes strategies.

Usage:
  python bot.py                  # Full run (scan + trade)
  python bot.py --scan           # Scan only (no trades)
  python bot.py --status         # Show positions + P&L
  python bot.py --paper          # Force paper trading
  python bot.py --kill           # Activate kill switch
  python bot.py --test           # Test connection only
"""

import argparse
import json
import logging
import sys
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from src.client import get_client
from src.risk import RiskManager, RiskLimits
from src.market_data import MarketScanner
from src.orders import OrderExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("josho")


def test_connection():
    """Test Groww API connection and print account info."""
    log.info("Testing Groww API connection...")
    client = get_client()
    try:
        api = client.connect()
        log.info("Connected successfully!")

        # Test market data
        positions = client.get_positions()
        log.info(f"Positions response: {json.dumps(positions, indent=2, default=str)[:500]}")

        holdings = client.get_holdings()
        log.info(f"Holdings response: {json.dumps(holdings, indent=2, default=str)[:500]}")

        # Test LTP
        ltp = client.get_ltp("RELIANCE-EQ", exchange="NSE", segment="CASH")
        log.info(f"RELIANCE LTP: ₹{ltp}")

        return True
    except Exception as e:
        log.error(f"Connection test failed: {e}")
        return False


def scan_market():
    """Scan market for opportunities."""
    scanner = MarketScanner()

    if not scanner.is_market_open():
        log.info("Market is closed. Showing last known data.")

    log.info("Scanning FNO stocks for momentum...")
    signals = scanner.scan_momentum()

    if signals:
        log.info(f"Found {len(signals)} momentum signals:")
        for s in signals:
            log.info(
                f"  {s['direction']} {s['symbol']}: "
                f"{s['change_pct']:+.2f}% | "
                f"Range: {s['day_range_pct']:.1f}% | "
                f"Strength: {s['strength']}"
            )
    else:
        log.info("No strong momentum signals found.")

    return signals


def show_status():
    """Show current positions and risk status."""
    risk = RiskManager()
    status = risk.get_status()

    print("\n" + "=" * 50)
    print("JOSHO TRADER — STATUS")
    print("=" * 50)
    print(f"Date:            {status['date']}")
    print(f"Mode:            {'PAPER' if status['paper'] else 'LIVE'}")
    print(f"Kill switch:     {'ACTIVE — ' + status['kill_reason'] if status['killed'] else 'OFF'}")
    print(f"Realized P&L:    ₹{status['realized_pnl']:,.0f}")
    print(f"Unrealized P&L:  ₹{status['unrealized_pnl']:,.0f}")
    print(f"Total P&L:       ₹{status['total_pnl']:,.0f}")
    print(f"Max drawdown:    ₹{status['max_drawdown']:,.0f}")
    print(f"Trades today:    {status['trades']}")
    print(f"Open positions:  {status['open_positions']}")
    print(f"Loss budget left:₹{status['remaining_loss_budget']:,.0f}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="JoSho Trader — F&O Trading Bot")
    parser.add_argument("--test", action="store_true", help="Test connection only")
    parser.add_argument("--scan", action="store_true", help="Scan market only")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--paper", action="store_true", help="Force paper trading")
    parser.add_argument("--kill", action="store_true", help="Activate kill switch")
    args = parser.parse_args()

    print("""
    ==========================================
     JOSHO TRADER -- F&O Trading Bot
     Groww API | Paper + Live modes
     Risk-managed | Telegram alerts
    ==========================================
    """)

    # Create logs dir
    from pathlib import Path
    Path("logs").mkdir(exist_ok=True)

    if args.test:
        success = test_connection()
        sys.exit(0 if success else 1)

    if args.status:
        show_status()
        return

    if args.kill:
        risk = RiskManager()
        risk.force_kill("Manual kill via CLI")
        log.info("Kill switch activated. No trades will execute today.")
        return

    if args.scan:
        scan_market()
        return

    # Full run
    limits = RiskLimits()
    if args.paper:
        limits.paper_trading = True

    risk = RiskManager(limits)
    executor = OrderExecutor(risk)

    log.info(f"Bot started | Mode: {'PAPER' if limits.paper_trading else 'LIVE'}")
    log.info(f"Risk limits: ₹{limits.max_loss_per_day} max loss, {limits.max_open_positions} max positions")

    # Scan
    signals = scan_market()

    # Show status
    show_status()

    log.info("Bot run complete. Use strategies/ to automate specific F&O plays.")


if __name__ == "__main__":
    main()

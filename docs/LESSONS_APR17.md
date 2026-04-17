# Lessons from April 17, 2026 — First Live Trading Day

## MISTAKES MADE (Never Repeat)

### 1. WRONG LOT SIZES — Cost us 2 hours
- **What happened:** Hardcoded lot sizes from random internet data. SUZLON=4000 (real=9025), PNB=4000 (real=8000), TATASTEEL=1500 (real=5500)
- **Impact:** Every order failed. Wasted 20+ API calls = rate limit risk
- **Fix applied:** `client.get_lot_size()` always queries Groww instrument master
- **Monday rule:** NEVER hardcode. ALWAYS verify via API before ordering.

### 2. TOO MANY SMALL TRADES — Charges ate 56% of profit
- **What happened:** 18 orders in one day. Brokerage Rs.300 + taxes Rs.82 = Rs.382 on Rs.687 gross
- **Impact:** Net profit only Rs.305 instead of Rs.687
- **Fix:** ONE big conviction bet > many small scattered bets
- **Monday rule:** Max 4 orders/day. Each order must have >Rs.500 expected profit.

### 3. AUTOPILOT KEPT DYING — Restarted 5+ times
- **What happened:** Background `nohup` processes on Windows are unreliable. Multiple zombie processes.
- **Fix applied:** PM2 ecosystem with auto-restart, max 3 restarts, 60s delay
- **Monday rule:** ONLY use `pm2 start/restart`. Never `nohup python &`.

### 4. ORDERS REJECTED BUT ASSUMED PLACED
- **What happened:** Order status was "NEW" but actually "REJECTED" by exchange. We moved on thinking it was placed.
- **Impact:** Ended up with 0 positions multiple times
- **Fix needed:** After EVERY order, wait 3-5 seconds and verify via `get_positions()`. If no position, retry or alert.
- **Monday rule:** ALWAYS verify position exists after ordering. Trust positions API, not order status.

### 5. DEEP OTM REJECTED BY GROWW
- **What happened:** Tried buying 530CE, 520CE, 505CE — all rejected "place order in strikes closer to spot price"
- **Impact:** Couldn't get the cheap lottery plays we wanted
- **Groww limit:** ~12-15% OTM maximum for stock options
- **Monday rule:** Never try >15% OTM on Groww. For deeper OTM, need different broker (Upstox/Dhan).

### 6. MARKET ORDER SLIPPAGE
- **What happened:** LTP was Rs.5.40 but market order filled at Rs.6.40 (+18.5% slippage!)
- **Impact:** Rs.663 short on balance, order rejected
- **Fix:** Use LIMIT orders at LTP + Rs.0.10 instead of MARKET orders for >Rs.2 premiums
- **Monday rule:** LIMIT orders for premiums >Rs.2. MARKET only for cheap <Rs.1 options.

### 7. DIDN'T CHECK BALANCE BEFORE ORDERING
- **What happened:** Tried to buy Rs.7,290 worth but only Rs.7,594 available. After charges, insufficient.
- **Impact:** Order rejected, money stuck in pending order blocking balance
- **Fix:** Always check `fno_available` AND subtract 10% for charges buffer
- **Monday rule:** Max order = 90% of available balance (10% for charges).

### 8. PENDING ORDERS BLOCKING BALANCE
- **What happened:** A pending LIMIT order blocked Rs.6,952. Showed Rs.642 available instead of Rs.7,594.
- **Impact:** Couldn't place new orders until cancelled
- **Fix:** Cancel ALL pending orders before placing new ones
- **Monday rule:** Before any new order: cancel all pending first.

## WHAT WENT RIGHT

1. **First F&O trade profitable** — COALINDIA26APR460CE: +Rs.540 (+25%)
2. **Journey tracker worked** — Held through +9%, +15%, sold at +25% near peak
3. **Intelligence engine found real signals** — Trump tariff + Iran = 16 bearish signals (correct!)
4. **ML training completed** — 22 stocks, ICICIBANK best at 69.8%
5. **Fear & Greed = 21** (extreme fear) — contrarian signal correct, COALINDIA rallied +1.6%
6. **System hub connected** — Supabase heartbeat working, all systems visible

## MONDAY PLAYBOOK

### Pre-Market (8:30-9:15 AM)
1. Check Hang Seng + Nikkei (they open before India)
2. Check GIFT Nifty gap
3. Run intelligence scan (Trump weekend tweets, Iran news)
4. Check crude oil (biggest COALINDIA predictor)
5. Decide: keep position or exit at open

### Market Open (9:15-9:30 AM)
1. Check COALINDIA26MAY490CE LTP immediately
2. If green (>Rs.3.00): SELL on first peak, don't wait
3. If red (<Rs.2.00): HOLD, check intelligence, wait for reversal
4. After exit: immediately enter next play (perpetual engine)

### During Market (9:30 AM - 3:00 PM)
1. Max 4 orders
2. Each trade: check lot size + balance + charges BEFORE ordering
3. Use LIMIT orders for >Rs.2 premiums
4. Verify position after every order
5. Intelligence scan every 15 min
6. Exit engine runs 30 strategies — exit only on 3+ consensus

### Pre-Close (3:00-3:25 PM)
1. If profitable: consider holding overnight (NRML)
2. If losing: exit before close if >-30%
3. Position for Tuesday (next market day) if strong signal

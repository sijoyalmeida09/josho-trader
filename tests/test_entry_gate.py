"""Guards POINT 2: SmartConnection.buy() refuses a NEW entry when kill_check is
true OR raises (fail-closed); sell()/exits are NEVER gated under any kill state;
and with no kill_check wired the original buy behavior is preserved."""

import groww_bot


def _boom():
    raise RuntimeError("uncertain")


def test_buy_blocked_when_kill_true():
    conn = groww_bot.SmartConnection(dry_run=True)
    conn._kill_check = lambda: True
    assert conn.buy("TATASTEEL-EQ", 1) is None


def test_buy_blocked_when_kill_raises():
    conn = groww_bot.SmartConnection(dry_run=True)
    conn._kill_check = _boom
    assert conn.buy("TATASTEEL-EQ", 1) is None  # fail-closed on gate error


def test_sell_never_gated():
    conn = groww_bot.SmartConnection(dry_run=True)
    conn._kill_check = lambda: True
    assert conn.sell("TATASTEEL-EQ", 1) is not None
    conn._kill_check = _boom
    assert conn.sell("TATASTEEL-EQ", 1) is not None


def test_default_no_gate_preserves_buy():
    conn = groww_bot.SmartConnection(dry_run=True)  # _kill_check defaults to None
    assert conn.buy("TATASTEEL-EQ", 1) == {"groww_order_id": "DRY-RUN", "order_status": "DRY"}

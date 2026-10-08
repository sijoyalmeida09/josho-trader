"""Guards: under every block state (kill true, kill_check raises, DB-uncertain),
the exit path still executes — the bot can ALWAYS get out. Entries are gated;
exits never are."""

import groww_bot
from fakes import FakeSupabase, make_control_plane, make_trader_db


def _boom():
    raise RuntimeError("uncertain")


def test_sell_runs_under_every_block_state():
    conn = groww_bot.SmartConnection(dry_run=True)
    for gate in (lambda: True, _boom):
        conn._kill_check = gate
        assert conn.sell("X-EQ", 1) is not None  # exit always runs
        assert conn.buy("X-EQ", 1) is None  # entry always blocked


def test_force_exit_runs_while_entries_blocked(kill_file):
    sb = FakeSupabase(fail_reads=True)  # DB outage -> uncertain -> entries blocked
    cp = make_control_plane(make_trader_db(sb), kill_file.path)
    conn = groww_bot.SmartConnection(dry_run=True)
    conn._kill_check = cp.new_entries_blocked
    assert cp.new_entries_blocked() is True

    fno = groww_bot.FnOStrategy(conn)
    fno.positions.append({"symbol": "S", "lot": 1, "entry": 1.0})
    eq = groww_bot.EquityStrategy(conn)
    mr = groww_bot.MeanReversionStrategy(conn)
    out = groww_bot._force_exit(conn, eq, fno, mr, flatten=True)
    assert out["ok"] is True  # liquidation path runs even while entries are blocked
    assert fno.positions == []

"""Guards the bot-side command consumption: flatten_all/manual_exit emit SELLs
only (never buys) even while killed, toggle_strategy places no orders, approve is
refused under kill, and the handler map exposes exactly the expected verbs.

(The HTTP allowlist pause|resume|toggle_strategy|manual_exit|flatten_all is
enforced in command/route.ts — TypeScript, not runnable in pytest — and mapped
onto these DB verbs; this file guards the loop side that the route maps onto.)"""

import groww_bot
from fakes import FakeGrowwConn, FakeSupabase, make_control_plane, make_trader_db


def _setup(kill_path, killed=False):
    sb = FakeSupabase()
    db = make_trader_db(sb, {"kill_switch": killed})
    cp = make_control_plane(db, kill_path)
    conn = FakeGrowwConn()
    eq = groww_bot.EquityStrategy(conn)
    fno = groww_bot.FnOStrategy(conn)
    mr = groww_bot.MeanReversionStrategy(conn)
    handlers = groww_bot._build_command_handlers(cp, conn, eq, fno, mr)
    return cp, conn, eq, fno, mr, handlers


def test_handler_map_keys(kill_file):
    _cp, _conn, _eq, _fno, _mr, handlers = _setup(kill_file.path)
    assert set(handlers) == {"kill", "pause", "resume", "set_setting", "exit_position", "approve", "reject"}


def test_flatten_all_sells_only_even_under_kill(kill_file):
    cp, conn, eq, fno, _mr, handlers = _setup(kill_file.path, killed=True)
    fno.positions.append({"symbol": "COALINDIA26JUN420CE", "lot": 1350, "entry": 2.0})
    eq.position = {"symbol": "TATASTEEL", "qty": 10, "entry": 140.0}
    assert cp.kill_engaged() is True
    out = handlers["exit_position"]({"payload": {"all": True}})
    assert out["ok"] is True
    assert len(conn.sells) >= 2  # fno + equity liquidated
    assert conn.buys == []  # never buys


def test_manual_exit_sells_one(kill_file):
    _cp, conn, _eq, fno, _mr, handlers = _setup(kill_file.path)
    fno.positions.append({"symbol": "HINDALCO26JUN700CE", "lot": 700, "entry": 3.0})
    handlers["exit_position"]({"payload": {"trading_symbol": "HINDALCO26JUN700CE"}})
    assert ("HINDALCO26JUN700CE", 700, "NRML", "FNO") in conn.sells
    assert conn.buys == []


def test_toggle_strategy_places_no_orders(kill_file):
    cp, conn, _eq, _fno, _mr, handlers = _setup(kill_file.path)
    out = handlers["set_setting"]({"payload": {"strategy": "fno", "enabled": False}})
    assert out["ok"] is True
    assert conn.buys == [] and conn.sells == []
    assert cp.strategy_enabled("fno") is False


def test_approve_refused_under_kill(kill_file):
    cp, _conn, _eq, _fno, _mr, handlers = _setup(kill_file.path, killed=True)
    assert cp.kill_engaged() is True
    assert handlers["approve"]({"payload": {}})["ok"] is False  # kill overrides


def test_malformed_payloads_rejected(kill_file):
    _cp, _conn, _eq, _fno, _mr, handlers = _setup(kill_file.path)
    assert handlers["set_setting"]({"payload": {}})["ok"] is False
    assert handlers["exit_position"]({"payload": {}})["ok"] is False


def test_consume_commands_dispatches_exactly_once(kill_file):
    """consume_commands claims+dispatches each row once; a second drain is empty."""
    sb = FakeSupabase()
    db = make_trader_db(sb, {"kill_switch": False})
    cp = make_control_plane(db, kill_file.path)
    conn = FakeGrowwConn()
    eq, fno, mr = (
        groww_bot.EquityStrategy(conn),
        groww_bot.FnOStrategy(conn),
        groww_bot.MeanReversionStrategy(conn),
    )
    handlers = groww_bot._build_command_handlers(cp, conn, eq, fno, mr)
    sb.seed("trader_commands", [{"id": "p1", "command": "pause", "status": "pending", "payload": {}}])

    first = cp.consume_commands(handlers)
    assert [(r["command"], r["ok"]) for r in first] == [("pause", True)]
    assert cp._cmd_pause is True  # handler actually ran
    assert cp.consume_commands(handlers) == []  # not re-processed


def test_consume_commands_audit_rows_not_acted_on(kill_file):
    """payload.audit==True rows are recorded but trigger no handler/behaviour."""
    sb = FakeSupabase()
    db = make_trader_db(sb, {"kill_switch": False})
    cp = make_control_plane(db, kill_file.path)
    conn = FakeGrowwConn()
    handlers = groww_bot._build_command_handlers(
        cp, conn, groww_bot.EquityStrategy(conn), groww_bot.FnOStrategy(conn), groww_bot.MeanReversionStrategy(conn)
    )
    sb.seed("trader_commands", [{"id": "a1", "command": "pause", "status": "pending", "payload": {"audit": True}}])
    out = cp.consume_commands(handlers)
    assert out and out[0].get("audit") is True
    assert cp._cmd_pause is False  # audit row did NOT actually pause


def test_consume_commands_unknown_marked_failed(kill_file):
    """A command with no handler is consumed and recorded failed (never retried)."""
    sb = FakeSupabase()
    db = make_trader_db(sb, {"kill_switch": False})
    cp = make_control_plane(db, kill_file.path)
    sb.seed("trader_commands", [{"id": "u1", "command": "set_mode", "status": "pending", "payload": {}}])
    out = cp.consume_commands({})  # empty handler map -> no handler for set_mode
    assert out and out[0]["ok"] is False
    assert cp.consume_commands({}) == []  # consumed, not retried

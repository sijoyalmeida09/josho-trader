"""Guards: a Supabase read error fails CLOSED (blocks new entries), exits still
run, the process never crashes, and it recovers to unblocked on the next clean
read (no false permanent latch)."""

import groww_bot
import trader_db
from fakes import FakeSupabase, make_control_plane, make_trader_db


def test_read_error_blocks_new_entries(kill_file):
    sb = FakeSupabase(fail_reads=True)
    cp = make_control_plane(make_trader_db(sb), kill_file.path)
    assert cp.refresh_kill().uncertain is True
    assert cp.new_entries_blocked() is True  # fail-closed, not fail-open


def test_exits_run_while_entries_blocked(kill_file):
    sb = FakeSupabase(fail_reads=True)
    cp = make_control_plane(make_trader_db(sb), kill_file.path)
    conn = groww_bot.SmartConnection(dry_run=True)
    conn._kill_check = cp.new_entries_blocked
    assert conn.buy("TATASTEEL-EQ", 1) is None  # entry blocked
    assert conn.sell("TATASTEEL-EQ", 1) is not None  # exit still runs


def test_process_never_crashes_on_read_error(kill_file):
    sb = FakeSupabase(fail_reads=True)
    cp = make_control_plane(make_trader_db(sb), kill_file.path)
    for _ in range(3):  # repeated calls must not raise
        cp.new_entries_blocked()
        cp.read_settings()
        cp.snapshot()


def test_recovers_on_next_clean_read(kill_file):
    sb = FakeSupabase(fail_reads=True)
    cp = make_control_plane(make_trader_db(sb), kill_file.path)
    assert cp.new_entries_blocked() is True
    sb.fail_reads = False  # outage resolves
    assert cp.refresh_kill().uncertain is False
    assert cp.new_entries_blocked() is False  # no false permanent latch
    assert cp._latched_kill is False


def test_configured_but_unreachable_fails_closed(kill_file):
    """A CONFIGURED DB that goes unreachable (enabled flips False) must fail
    closed, not collapse into 'no DB -> allow' mode."""
    db = trader_db.TraderDB()
    db.configured = True  # creds were present
    db.enabled = False  # create_client failed -> unreachable
    cp = make_control_plane(db, kill_file.path)
    assert cp.refresh_kill().uncertain is True
    assert cp.new_entries_blocked() is True


def test_c_is_none_read_flips_read_healthy():
    """The c-is-None early return (conn failed) must flip read_healthy False —
    no read path may swallow-and-default while still reporting healthy."""
    db = trader_db.TraderDB()
    db.enabled = False  # _conn() -> None
    assert db.get_settings() == {}
    assert db.read_healthy is False


def test_genuinely_unconfigured_still_runs(kill_file, monkeypatch):
    """No creds at all (configured False) -> local-file-only authority; entries
    are allowed (the run-without-Supabase contract is preserved, not over-rotated
    into a permanent block)."""
    for var in (
        "SUPABASE_URL",
        "NEXT_PUBLIC_SUPABASE_URL",
        "SOHAYA__NEXT_PUBLIC_SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SOHAYA__SUPABASE_SERVICE_ROLE_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    db = trader_db.TraderDB()  # no creds -> configured False, enabled False
    assert db.configured is False
    cp = make_control_plane(db, kill_file.path)
    assert cp.new_entries_blocked() is False

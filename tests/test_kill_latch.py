"""Guards: the kill latch is sticky across a DB flip-flop (true->false mid-run
stays blocked); clearing needs confirm=true AND no local KILL file; and each of
{db, command, file} arms kill independently."""

from fakes import FakeSupabase, make_control_plane, make_trader_db


def test_latch_sticky_across_db_flip(kill_file, db_factory):
    sb = FakeSupabase()
    cp = make_control_plane(db_factory(sb, {"kill_switch": True}), kill_file.path)
    assert cp.kill_engaged() is True
    sb.tables["trader_settings"][0]["kill_switch"] = False  # DB flips off mid-run
    assert cp.kill_engaged() is True  # still latched
    assert cp.new_entries_blocked() is True


def test_clear_requires_confirm_and_no_file(kill_file, db_factory):
    sb = FakeSupabase()
    cp = make_control_plane(db_factory(sb, {"kill_switch": True}), kill_file.path)
    cp.kill_engaged()  # latch on
    sb.tables["trader_settings"][0]["kill_switch"] = False  # remote operator cleared flag
    assert cp.clear_kill(confirm=False) is False  # confirm required
    kill_file.arm()
    assert cp.clear_kill(confirm=True) is False  # refused while local KILL file present
    kill_file.disarm()
    assert cp.clear_kill(confirm=True) is True  # now disarms
    assert cp.new_entries_blocked() is False


def test_db_arms_independently(kill_file, db_factory):
    sb = FakeSupabase()
    cp = make_control_plane(db_factory(sb, {"kill_switch": True}), kill_file.path)
    assert cp.kill_engaged() is True


def test_command_arms_independently(kill_file, db_factory):
    sb = FakeSupabase()
    cp = make_control_plane(db_factory(sb, {"kill_switch": False}), kill_file.path)
    assert cp.kill_engaged() is False
    cp.engage_kill("command")
    assert cp.kill_engaged() is True


def test_file_arms_independently(kill_file, db_factory):
    sb = FakeSupabase()
    cp = make_control_plane(db_factory(sb, {"kill_switch": False}), kill_file.path)
    assert cp.kill_engaged() is False
    kill_file.arm()
    assert cp.kill_engaged() is True

"""Guards: per_trade_cap clamped <= wallet; daily_loss_cap <= 50%*wallet;
negatives clamp to 0; non-numeric falls back to default (no crash); and
read_settings() returns an immutable per-tick snapshot (no torn read)."""

import dataclasses

import pytest

from fakes import FakeSupabase, make_control_plane


def test_per_trade_clamped_to_wallet(db_factory, kill_file):
    sb = FakeSupabase()
    cp = make_control_plane(db_factory(sb, {"max_per_fno_trade_inr": 999999}), kill_file.path, wallet=10000.0)
    assert cp.read_settings().per_trade_cap_inr == 10000.0


def test_daily_clamped_to_half_wallet(db_factory, kill_file):
    sb = FakeSupabase()
    cp = make_control_plane(db_factory(sb, {"daily_loss_cap_inr": 999999}), kill_file.path, wallet=10000.0)
    assert cp.read_settings().daily_loss_cap_inr == 5000.0


def test_negatives_clamped_to_zero(db_factory, kill_file):
    sb = FakeSupabase()
    cp = make_control_plane(
        db_factory(sb, {"max_per_fno_trade_inr": -50, "daily_loss_cap_inr": -1}),
        kill_file.path,
        wallet=10000.0,
    )
    eff = cp.read_settings()
    assert eff.per_trade_cap_inr == 0.0
    assert eff.daily_loss_cap_inr == 0.0


def test_non_numeric_falls_back_to_default(db_factory, kill_file):
    sb = FakeSupabase()
    cp = make_control_plane(
        db_factory(sb, {"max_per_fno_trade_inr": "not-a-number"}), kill_file.path, wallet=100000.0
    )
    assert cp.read_settings().per_trade_cap_inr == 3000.0  # default, no crash


def test_snapshot_immutable_and_not_torn(db_factory, kill_file):
    sb = FakeSupabase()
    cp = make_control_plane(db_factory(sb, {"max_per_fno_trade_inr": 3000}), kill_file.path, wallet=10000.0)
    eff = cp.read_settings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        eff.per_trade_cap_inr = 1  # frozen snapshot
    sb.tables["trader_settings"][0]["max_per_fno_trade_inr"] = 1234  # DB changes after the read
    assert eff.per_trade_cap_inr == 3000.0  # earlier snapshot unchanged (no torn read)
    assert cp.read_settings().per_trade_cap_inr == 1234.0  # next tick picks it up

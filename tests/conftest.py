"""Shared pytest fixtures for the trader guardrail suite.

Adds the josho-trader dir to sys.path so `import control_plane`, `trader_db`,
`groww_bot` resolve, and exposes the STEP-0 fakes as fixtures. Hermetic: no
network, no secrets, no live Supabase/Groww.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the bot modules importable (tests/ lives under automation/josho-trader/),
# and the tests dir itself so `from fakes import ...` resolves as top-level.
_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fakes import (  # noqa: E402  (path set above)
    FakeClock,
    FakeGrowwConn,
    FakeSupabase,
    KillFile,
    make_control_plane,
    make_trader_db,
)


@pytest.fixture
def fake_sb() -> FakeSupabase:
    """Fresh in-memory Supabase client (reads succeed by default)."""
    return FakeSupabase()


@pytest.fixture
def fake_groww() -> FakeGrowwConn:
    """Order recorder; never networks."""
    return FakeGrowwConn()


@pytest.fixture
def fake_clock() -> FakeClock:
    """Deterministic, controllable clock."""
    return FakeClock()


@pytest.fixture
def kill_file(tmp_path) -> KillFile:
    """Temp KILL-file controller (disarmed by default)."""
    return KillFile(tmp_path / "KILL")


@pytest.fixture
def db_factory():
    """Returns make_trader_db(fake_sb, settings=None) -> real TraderDB on a fake."""
    return make_trader_db


@pytest.fixture
def cp_factory():
    """Returns make_control_plane(db, kill_path, wallet=10000.0) -> real ControlPlane."""
    return make_control_plane

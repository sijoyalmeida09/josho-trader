"""Shared hermetic fakes for the trader guardrail suite.

ZERO network, ZERO live Supabase, ZERO live Groww. Every test imports its
mocks from HERE — no test defines its own. See tests/_HARNESS.md for the
public signatures.

The marquee fake is FakeSupabase: it mimics enough of the supabase-py client
surface that the REAL trader_db.TraderDB runs on top of it unmodified. Crucially
it replicates PostgREST's atomic conditional update — UPDATE ... WHERE
status='pending' hands the row to exactly ONE caller — and offers a
``fail_reads`` mode so a SELECT raises, exercising trader_db's real
swallow-and-default path (which flips read_healthy False -> fail-closed).
"""

from __future__ import annotations

import copy
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# FakeSupabase — client-level fake; real TraderDB runs on top of it.
# ---------------------------------------------------------------------------
class FakeResult:
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data


class _Query:
    """One chained query against one table. Mirrors the supabase-py builder."""

    def __init__(self, sb: "FakeSupabase", table: str):
        self._sb = sb
        self._table = table
        self._op: Optional[str] = None
        self._payload: Any = None
        self._on_conflict: Optional[str] = None
        self._eq: List[tuple] = []
        self._gt: List[tuple] = []
        self._contains: List[tuple] = []
        self._order: Optional[str] = None
        self._desc = False
        self._limit: Optional[int] = None

    # builder verbs (all return self for chaining)
    def select(self, *_a, **_k) -> "_Query":
        self._op = "select"
        return self

    def insert(self, row: Any) -> "_Query":
        self._op = "insert"
        self._payload = row
        return self

    def update(self, patch: Dict[str, Any]) -> "_Query":
        self._op = "update"
        self._payload = patch
        return self

    def upsert(self, row: Dict[str, Any], on_conflict: Optional[str] = None) -> "_Query":
        self._op = "upsert"
        self._payload = row
        self._on_conflict = on_conflict
        return self

    def delete(self) -> "_Query":
        self._op = "delete"
        return self

    def eq(self, col: str, val: Any) -> "_Query":
        self._eq.append((col, val))
        return self

    def gt(self, col: str, val: Any) -> "_Query":
        self._gt.append((col, val))
        return self

    def contains(self, col: str, vals: Any) -> "_Query":
        self._contains.append((col, vals))
        return self

    def order(self, col: str, desc: bool = False) -> "_Query":
        self._order = col
        self._desc = desc
        return self

    def limit(self, n: int) -> "_Query":
        self._limit = n
        return self

    def _match(self, row: Dict[str, Any]) -> bool:
        for c, v in self._eq:
            if row.get(c) != v:
                return False
        for c, v in self._gt:
            cur = row.get(c)
            if cur is None or not (cur > v):
                return False
        for c, vals in self._contains:
            cur = row.get(c) or []
            if not all(x in cur for x in vals):
                return False
        return True

    def execute(self) -> FakeResult:
        # The whole operation is serialized: this is what makes the conditional
        # update atomic for concurrent callers (exactly-one-wins).
        with self._sb.lock:
            rows = self._sb.tables.setdefault(self._table, [])

            if self._op == "select":
                if self._sb.fail_reads:
                    raise RuntimeError("FakeSupabase: simulated read failure")
                out = [copy.deepcopy(r) for r in rows if self._match(r)]
                if self._order:
                    out.sort(
                        key=lambda r: (r.get(self._order) is None, r.get(self._order)),
                        reverse=self._desc,
                    )
                if self._limit is not None:
                    out = out[: self._limit]
                return FakeResult(out)

            if self._op == "insert":
                items = self._payload if isinstance(self._payload, list) else [self._payload]
                inserted = []
                for it in items:
                    r = dict(it)
                    r.setdefault("id", str(uuid.uuid4()))
                    r.setdefault("created_at", _now_iso())
                    rows.append(r)
                    inserted.append(copy.deepcopy(r))
                return FakeResult(inserted)

            if self._op == "update":
                updated = []
                for r in rows:
                    if self._match(r):
                        r.update(self._payload)
                        updated.append(copy.deepcopy(r))
                return FakeResult(updated)

            if self._op == "upsert":
                key = self._on_conflict or "id"
                r = dict(self._payload)
                for ex in rows:
                    if ex.get(key) == r.get(key):
                        ex.update(r)
                        return FakeResult([copy.deepcopy(ex)])
                r.setdefault("id", str(uuid.uuid4()))
                rows.append(r)
                return FakeResult([copy.deepcopy(r)])

            if self._op == "delete":
                removed = [copy.deepcopy(r) for r in rows if self._match(r)]
                self._sb.tables[self._table] = [r for r in rows if not self._match(r)]
                return FakeResult(removed)

            return FakeResult([])


class FakeSupabase:
    """In-memory supabase-py client stand-in. ``fail_reads`` makes SELECTs raise."""

    def __init__(self, fail_reads: bool = False):
        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        self.lock = threading.RLock()
        self.fail_reads = fail_reads

    def table(self, name: str) -> _Query:
        return _Query(self, name)

    def seed(self, table: str, rows: List[Dict[str, Any]]) -> None:
        with self.lock:
            self.tables[table] = [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Factories — build the REAL TraderDB / ControlPlane on top of FakeSupabase.
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "id": 1,
    "approval_mode": "manual",
    "auto_execute_min_confidence": 75,
    "daily_loss_cap_inr": 2000,
    "max_concurrent_fno_positions": 2,
    "max_per_fno_trade_inr": 3000,
    "kill_switch": False,
    "pause_new_entries": False,
}


def make_trader_db(fake_sb: FakeSupabase, settings: Optional[Dict[str, Any]] = None):
    """Real TraderDB wired to a FakeSupabase, seeded with the settings singleton."""
    import trader_db

    db = trader_db.TraderDB()
    db.enabled = True
    db._client = fake_sb
    base = dict(DEFAULT_SETTINGS)
    if settings:
        base.update(settings)
    fake_sb.seed("trader_settings", [base])
    return db


def make_control_plane(db, kill_path: Path, wallet: float = 10000.0):
    """Real ControlPlane over the given db. notifier is a no-op (no Telegram)."""
    import control_plane

    return control_plane.ControlPlane(
        db=db,
        kill_file=Path(kill_path),
        wallet_provider=lambda: wallet,
        notifier=lambda _m: None,
    )


# ---------------------------------------------------------------------------
# FakeGrowwConn — records buy/sell, never networks. Pure recorder (it deliberately
# carries NO kill logic of its own, so the real SmartConnection gate stays the
# single source of truth for entry-gate tests).
# ---------------------------------------------------------------------------
class FakeGrowwConn:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.connected = True
        self.buys: List[tuple] = []
        self.sells: List[tuple] = []

    def buy(self, symbol, qty, product="MIS", segment="CASH"):
        self.buys.append((symbol, qty, product, segment))
        return {"groww_order_id": "FAKE", "order_status": "FAKE"}

    def sell(self, symbol, qty, product="MIS", segment="CASH"):
        self.sells.append((symbol, qty, product, segment))
        return {"groww_order_id": "FAKE", "order_status": "FAKE"}


# ---------------------------------------------------------------------------
# FakeClock — deterministic, controllable time source.
# ---------------------------------------------------------------------------
class FakeClock:
    def __init__(self, start: Optional[datetime] = None):
        self._now = start or datetime(2026, 6, 22, 10, 0, 0, tzinfo=timezone.utc)

    def now(self, tz=None) -> datetime:
        return self._now.astimezone(tz) if tz else self._now

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


# ---------------------------------------------------------------------------
# KillFile — temp KILL-file controller (arm/disarm). os.PathLike for direct use.
# ---------------------------------------------------------------------------
class KillFile:
    def __init__(self, path: Path):
        self.path = Path(path)

    def arm(self) -> None:
        self.path.write_text("KILL")

    def disarm(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def exists(self) -> bool:
        return self.path.exists()

    def __fspath__(self) -> str:
        return str(self.path)

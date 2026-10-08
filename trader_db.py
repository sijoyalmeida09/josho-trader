"""
trader_db.py — Supabase writer/reader for JoSho Trader.

Single class TraderDB providing fail-soft persistence helpers used by
groww_bot.py. Every public method catches Exception, logs at WARNING, and
returns a sensible empty/None default so the bot keeps running even when
Supabase is unreachable.

Schema source of truth: supabase/migrations/040_trader_tables.sql
Contract: automation/josho-trader/_CONTRACT.md ("Supabase writer").
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("trader_db")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

# Lazy import — keeps `import trader_db` cheap and avoids hard failure if SDK
# is missing in some constrained environments.
try:
    from supabase import Client, create_client  # type: ignore
    _SDK_OK = True
except Exception as _e:  # pragma: no cover - missing dep is itself fail-soft
    Client = Any  # type: ignore
    create_client = None  # type: ignore
    _SDK_OK = False
    log.info("supabase SDK not importable (%s); trader_db will be disabled", _e)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_env() -> tuple[Optional[str], Optional[str]]:
    """Resolve Supabase URL + service-role key with vault fallback chain."""
    url = (
        os.environ.get("SUPABASE_URL")
        or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        or os.environ.get("SOHAYA__NEXT_PUBLIC_SUPABASE_URL")
    )
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SOHAYA__SUPABASE_SERVICE_ROLE_KEY")
    )
    return url, key


_REQUIRED_PENDING_FIELDS = (
    "segment",
    "symbol",
    "trading_symbol",
    "qty",
    "entry_price",
    "cost_inr",
    "strategy",
)


class TraderDB:
    """Fail-soft Supabase wrapper. All methods swallow exceptions."""

    def __init__(self) -> None:
        self.enabled: bool = False
        # True after the most recent attempted read succeeded; flipped False
        # whenever a read swallows an exception and returns a default. Lets the
        # control plane distinguish "DB says no kill" from "DB unreadable" so a
        # swallowed failure fails closed instead of fail-open. Existing return
        # values/signatures are unchanged — fail-soft behaviour is byte-identical.
        self.read_healthy: bool = True
        self._client: Optional["Client"] = None
        self._url, self._key = _resolve_env()
        # True iff credentials are present, i.e. a DB was CONFIGURED. Never flips
        # (unlike `enabled`, which create_client failure sets False). Lets the
        # control plane tell "configured but unreachable" (-> fail closed) apart
        # from "genuinely not configured" (-> local-file-only, run without DB).
        self.configured: bool = bool(self._url and self._key)
        if not _SDK_OK:
            log.info("trader_db disabled: supabase SDK not installed")
            return
        if not self._url or not self._key:
            log.info(
                "trader_db disabled: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY missing"
            )
            return
        self.enabled = True  # connection deferred to first call

    # ---- internal -----------------------------------------------------
    def _conn(self) -> Optional["Client"]:
        if not self.enabled:
            return None
        if self._client is not None:
            return self._client
        try:
            self._client = create_client(self._url, self._key)  # type: ignore[arg-type]
            return self._client
        except Exception as e:
            log.warning("trader_db connect failed: %s", e)
            self.enabled = False
            return None

    # ---- state (singleton id=1) ---------------------------------------
    def upsert_state(self, **fields: Any) -> None:
        c = self._conn()
        if c is None:
            return
        try:
            row = dict(fields)
            row["id"] = 1
            row["updated_at"] = _now_iso()
            row["ist_ts"] = _now_iso()
            c.table("trader_state").upsert(row, on_conflict="id").execute()
        except Exception as e:
            log.warning("upsert_state failed: %s", e)

    # Alias preserved for contract compatibility.
    def update_state(self, **fields: Any) -> None:
        self.upsert_state(**fields)

    def get_settings(self) -> Dict[str, Any]:
        c = self._conn()
        if c is None:
            # _conn() returned None — either disabled, or create_client just
            # failed (flipping enabled=False). Either way the read did not
            # succeed, so do not report health as good.
            self.read_healthy = False
            return {}
        try:
            res = (
                c.table("trader_settings").select("*").eq("id", 1).limit(1).execute()
            )
            data = res.data or []
            self.read_healthy = True
            return data[0] if data else {}
        except Exception as e:
            log.warning("get_settings failed: %s", e)
            self.read_healthy = False
            return {}

    def get_kill_switch(self) -> bool:
        return bool(self.get_settings().get("kill_switch", False))

    # ---- scans --------------------------------------------------------
    def insert_scan(
        self,
        equity: Any = None,
        fno_list: Any = None,
        meanrev: Any = None,
        active_situational: Any = None,
        wallet: Optional[float] = None,
    ) -> Optional[int]:
        c = self._conn()
        if c is None:
            return None
        try:
            row = {
                "ts": _now_iso(),
                "wallet_inr": wallet,
                "equity_signal": equity,
                "fno_opportunities": fno_list if fno_list is not None else [],
                "meanrev_signal": meanrev,
                "active_situational": active_situational if active_situational is not None else [],
            }
            res = c.table("trader_scans").insert(row).execute()
            data = res.data or []
            return data[0]["id"] if data else None
        except Exception as e:
            log.warning("insert_scan failed: %s", e)
            return None

    # ---- pending orders -----------------------------------------------
    def create_pending_order(self, order_dict: Dict[str, Any]) -> Optional[str]:
        c = self._conn()
        if c is None:
            return None
        try:
            missing = [f for f in _REQUIRED_PENDING_FIELDS if order_dict.get(f) in (None, "")]
            if missing:
                log.warning("create_pending_order missing fields: %s", missing)
                return None
            res = c.table("trader_pending_orders").insert(order_dict).execute()
            data = res.data or []
            return str(data[0]["id"]) if data else None
        except Exception as e:
            log.warning("create_pending_order failed: %s", e)
            return None

    def mark_pending(self, order_id: str, status: str, **fields: Any) -> None:
        c = self._conn()
        if c is None:
            return
        try:
            patch: Dict[str, Any] = {"status": status}
            patch.update(fields)
            if status in ("approved", "rejected", "expired", "executed", "failed"):
                patch.setdefault("decided_at", _now_iso())
            c.table("trader_pending_orders").update(patch).eq("id", order_id).execute()
        except Exception as e:
            log.warning("mark_pending failed: %s", e)

    # ---- positions ----------------------------------------------------
    def upsert_position(self, position_dict: Dict[str, Any]) -> Optional[str]:
        c = self._conn()
        if c is None:
            return None
        try:
            row = dict(position_dict)
            row["updated_at"] = _now_iso()
            res = (
                c.table("trader_positions")
                .upsert(row, on_conflict="trading_symbol")
                .execute()
            )
            data = res.data or []
            return str(data[0]["id"]) if data and "id" in data[0] else None
        except Exception as e:
            log.warning("upsert_position failed: %s", e)
            return None

    def close_position(self, trading_symbol: str) -> Dict[str, Any]:
        c = self._conn()
        if c is None:
            return {}
        try:
            sel = (
                c.table("trader_positions")
                .select("*")
                .eq("trading_symbol", trading_symbol)
                .limit(1)
                .execute()
            )
            rows = sel.data or []
            if not rows:
                return {}
            row = rows[0]
            c.table("trader_positions").delete().eq(
                "trading_symbol", trading_symbol
            ).execute()
            return row
        except Exception as e:
            log.warning("close_position failed: %s", e)
            return {}

    # ---- trades (immutable) -------------------------------------------
    def insert_trade(self, trade_dict: Dict[str, Any]) -> Optional[str]:
        c = self._conn()
        if c is None:
            return None
        try:
            res = c.table("trader_trades").insert(trade_dict).execute()
            data = res.data or []
            return str(data[0]["id"]) if data and "id" in data[0] else None
        except Exception as e:
            log.warning("insert_trade failed: %s", e)
            return None

    # ---- commands queue ----------------------------------------------
    def fetch_pending_commands(self) -> List[Dict[str, Any]]:
        c = self._conn()
        if c is None:
            self.read_healthy = False  # read did not succeed
            return []
        try:
            res = (
                c.table("trader_commands")
                .select("*")
                .eq("status", "pending")
                .order("created_at", desc=False)
                .limit(50)
                .execute()
            )
            self.read_healthy = True
            return list(res.data or [])
        except Exception as e:
            log.warning("fetch_pending_commands failed: %s", e)
            self.read_healthy = False
            return []

    def mark_command_consumed(self, command_id: str, result: Dict[str, Any]) -> None:
        c = self._conn()
        if c is None:
            return
        try:
            c.table("trader_commands").update(
                {
                    "status": "consumed",
                    "consumed_at": _now_iso(),
                    "consumed_by": "bot",
                    "result": result or {},
                }
            ).eq("id", command_id).execute()
        except Exception as e:
            log.warning("mark_command_consumed failed: %s", e)

    def claim_command(self, command_id: str) -> Optional[Dict[str, Any]]:
        """Atomically transition a single command pending->consumed and return
        the claimed row, or None if another worker already claimed it.

        Exactly-once guarantee: the ``status = 'pending'`` filter in the UPDATE
        is the optimistic-concurrency gate. PostgREST applies it server-side, so
        of N concurrent claimers only the one whose UPDATE actually matched a
        still-pending row gets a non-empty representation back; everyone else
        sees an empty result and must skip. (Schema status vocabulary is
        pending/consumed/failed — this maps the logical queued->processing
        transition onto the conditional pending->consumed update.)
        """
        c = self._conn()
        if c is None:
            self.read_healthy = False  # claim read did not succeed
            return None
        try:
            res = (
                c.table("trader_commands")
                .update(
                    {
                        "status": "consumed",
                        "consumed_at": _now_iso(),
                        "consumed_by": "bot",
                    }
                )
                .eq("id", command_id)
                .eq("status", "pending")  # conditional: only claim if still pending
                .execute()
            )
            data = res.data or []
            self.read_healthy = True
            return data[0] if data else None
        except Exception as e:
            log.warning("claim_command failed: %s", e)
            self.read_healthy = False
            return None

    def finish_command(
        self, command_id: str, result: Dict[str, Any], ok: bool = True
    ) -> None:
        """Record the outcome of an already-claimed command. Writes the result
        payload; flips status to 'failed' when ``ok`` is False (the row is
        already 'consumed' from the claim, so this never re-opens it for another
        worker)."""
        c = self._conn()
        if c is None:
            return
        try:
            patch: Dict[str, Any] = {"result": result or {}}
            if not ok:
                patch["status"] = "failed"
            c.table("trader_commands").update(patch).eq("id", command_id).execute()
        except Exception as e:
            log.warning("finish_command failed: %s", e)

    # ---- signals ------------------------------------------------------
    def fetch_active_signals(self) -> List[Dict[str, Any]]:
        c = self._conn()
        if c is None:
            self.read_healthy = False  # read did not succeed
            return []
        try:
            res = (
                c.table("trader_signals")
                .select("*")
                .gt("expires_at", _now_iso())
                .order("severity", desc=True)
                .limit(50)
                .execute()
            )
            self.read_healthy = True
            return list(res.data or [])
        except Exception as e:
            log.warning("fetch_active_signals failed: %s", e)
            self.read_healthy = False
            return []

    def fetch_active_signals_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        c = self._conn()
        if c is None:
            self.read_healthy = False  # read did not succeed
            return []
        if not symbol:
            return []  # caller passed no symbol — not a read failure
        try:
            res = (
                c.table("trader_signals")
                .select("*")
                .contains("affected_symbols", [symbol])
                .gt("expires_at", _now_iso())
                .order("severity", desc=True)
                .limit(50)
                .execute()
            )
            self.read_healthy = True
            return list(res.data or [])
        except Exception as e:
            log.warning("fetch_active_signals_for_symbol failed: %s", e)
            self.read_healthy = False
            return []


if __name__ == "__main__":
    db = TraderDB()
    status = "ENABLED" if db.enabled else "DISABLED"
    url_host = (db._url or "").split("//", 1)[-1].split("/", 1)[0] if db._url else "no-url"
    print(f"trader_db smoke test: {status} (host={url_host}, sdk={_SDK_OK})")
    # Read-only probes — safe even when disabled.
    settings = db.get_settings()
    print(f"trader_db smoke test: settings_rows={1 if settings else 0}")
    print(f"trader_db smoke test: kill_switch={db.get_kill_switch()}")
    print(f"trader_db smoke test: active_signals={len(db.fetch_active_signals())}")

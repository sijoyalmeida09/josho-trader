"""
control_plane.py — deterministic guardrails for groww_bot.py.

ADDITIVE and self-contained. Nothing here places orders; it only decides
whether NEW entries are allowed, what the effective risk caps are, and
dispatches queued commands exactly once. The bot loop consults it at three
points every tick.

AUTHORITY MODEL
---------------
Kill is engaged if ANY source says so:
  1. Local KILL file (synchronous, read every pass)  — hard local authority.
  2. trader_settings.kill_switch == true             — remote authority (UI).
  3. A claimed 'kill' command in trader_commands      — event authority.

LATCHING
  Once kill is observed engaged it stays engaged in-process until an explicit
  clear() (operator action via the kill route with enable=false + confirm).
  A DB flip back to kill_switch=false does NOT silently re-arm trading.

FAIL-CLOSED
  - DB *configured* but a read raises  -> new entries BLOCKED ("uncertain ==
    unsafe"). Latches an `uncertain` flag that a later clean read clears.
  - DB *not configured / disabled*     -> the local KILL file is the sole
    authority, so the bot still runs in --scan-once without Supabase
    (_CONTRACT.md boundary #6). It does NOT block purely because the DB is off.

KILL SEMANTICS
  Kill halts NEW ENTRIES only. It NEVER force-liquidates open positions
  (force-selling deep-OTM books at market = catastrophic slippage). Exits and
  the explicit 'flatten_all' command remain allowed while killed — kill gates
  buys, never sells.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("control_plane")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

# Default risk caps used when the DB is unreachable/disabled. Mirror groww_bot
# constants but kept independent so this module imports cleanly on its own.
_DEFAULT_DAILY_LOSS_CAP = float(os.environ.get("DAILY_LOSS_CAP_INR", "2000"))
_DEFAULT_PER_TRADE_CAP = float(os.environ.get("MAX_PER_FNO_TRADE_INR", "3000"))
_DEFAULT_MAX_FNO_POS = int(os.environ.get("MAX_CONCURRENT_FNO_POSITIONS", "2"))
# Daily loss cap may not exceed this fraction of wallet (defence-in-depth).
_DAILY_CAP_WALLET_FRACTION = float(os.environ.get("DAILY_LOSS_CAP_WALLET_PCT", "0.5"))


@dataclass(frozen=True)
class EffectiveSettings:
    """Immutable snapshot of the risk caps in force for one tick."""

    per_trade_cap_inr: float
    daily_loss_cap_inr: float
    max_concurrent_fno_positions: int
    pause_new_entries: bool
    kill_switch: bool
    source: str  # 'db' | 'default' | 'last_known'


@dataclass(frozen=True)
class KillView:
    """What the kill sources said on the most recent refresh."""

    engaged: bool
    latched: bool
    local_file: bool
    db_kill_switch: Optional[bool]  # None == unknown (disabled or error)
    uncertain: bool
    db_enabled: bool
    sources: List[str]


class ControlPlane:
    """Single authority the bot loop consults each tick.

    Parameters
    ----------
    db : TraderDB | None
        Fail-soft Supabase wrapper. May be a disabled instance.
    kill_file : Path
        The local KILL file path (hard authority).
    wallet_provider : Callable[[], float]
        Returns the latest known wallet balance (for clamping caps).
    notifier : Callable[[str], None] | None
        Telegram sender; used for high-priority kill alerts.
    """

    def __init__(
        self,
        db: Any,
        kill_file: Path,
        wallet_provider: Callable[[], float],
        notifier: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._db = db
        self._kill_file = Path(kill_file)
        self._wallet = wallet_provider
        self._notify = notifier or (lambda _m: None)

        self._latched_kill = False
        self._uncertain = False
        self._kill_announced = False
        self._last_settings: Optional[EffectiveSettings] = None
        self._strategy_disabled: Dict[str, bool] = {}
        self._cmd_pause = False  # in-memory pause from a 'pause' command

    # ------------------------------------------------------------------ #
    # KILL
    # ------------------------------------------------------------------ #
    @property
    def _db_enabled(self) -> bool:
        return bool(getattr(self._db, "enabled", False))

    def local_kill_present(self) -> bool:
        """Synchronous local KILL-file probe. Fail-closed on filesystem error."""
        try:
            return self._kill_file.exists()
        except Exception as e:  # pragma: no cover - fs errors are rare
            log.warning("local kill-file probe failed (%s) — fail-closed", e)
            return True

    def refresh_kill(self) -> KillView:
        """Re-evaluate every kill source and update the latch. Returns a view."""
        sources: List[str] = []
        local = self.local_kill_present()
        if local:
            sources.append("local_file")

        db_kill: Optional[bool] = None
        configured = bool(getattr(self._db, "configured", False))
        if configured and not self._db_enabled:
            # A DB WAS configured but is now unreachable (create_client failed,
            # which flips enabled False). Never collapse that into "no DB -> allow":
            # fail closed. Distinct from a genuinely unconfigured DB, where the
            # local KILL file is the sole authority (handled by the else below).
            log.warning("configured Supabase is unreachable — fail-closed on new entries")
            self._uncertain = True
            sources.append("db_unreachable")
        elif self._db_enabled:
            try:
                db_kill = bool(self._db.get_kill_switch())
            except Exception as e:
                # Defensive: a read that actually raises -> fail-closed.
                log.warning("kill read raised (%s) — fail-closed on new entries", e)
                self._uncertain = True
                sources.append("db_uncertain")
                db_kill = None
            else:
                # trader_db is fail-soft: it SWALLOWS read errors and returns a
                # default (kill_switch -> False) instead of raising. read_healthy
                # tells us whether the read truly succeeded. A swallowed failure
                # must fail closed, not fail open. uncertain is NOT latched — a
                # later clean read clears it (no false permanent latch).
                if not getattr(self._db, "read_healthy", True):
                    log.warning("kill read swallowed an error — fail-closed on new entries")
                    self._uncertain = True
                    sources.append("db_uncertain")
                    db_kill = None
                else:
                    self._uncertain = False  # a clean read clears prior uncertainty
                    if db_kill:
                        sources.append("db_kill_switch")

        engaged_now = local or bool(db_kill)
        if engaged_now:
            self._latched_kill = True

        if self._latched_kill and not self._kill_announced:
            self._kill_announced = True
            self._notify(
                "🛑 KILL ENGAGED — new entries halted. Open positions are NOT "
                "auto-liquidated. Use flatten_all to liquidate explicitly."
            )
        if not self._latched_kill:
            self._kill_announced = False

        return KillView(
            engaged=self._latched_kill,
            latched=self._latched_kill,
            local_file=local,
            db_kill_switch=db_kill,
            uncertain=self._uncertain,
            db_enabled=self._db_enabled,
            sources=sources,
        )

    def kill_engaged(self) -> bool:
        """Latching kill state. Cheap: refreshes sources, returns the latch.

        Used at loop-top (point 1) and immediately before every order via the
        ``new_entries_blocked`` gate (point 2). Reads the local file every call
        (point 3)."""
        return self.refresh_kill().engaged

    def engage_kill(self, reason: str = "") -> None:
        """Latch kill from an in-process event (e.g. a 'kill' command)."""
        self._latched_kill = True
        if not self._kill_announced:
            self._kill_announced = True
            self._notify(f"🛑 KILL ENGAGED ({reason or 'command'}) — new entries halted.")

    def clear_kill(self, confirm: bool) -> bool:
        """Explicit disarm. Refuses unless confirm=True AND no source still
        asserts kill (won't clear while the local KILL file is present)."""
        if not confirm:
            log.warning("clear_kill refused: confirm flag not set")
            return False
        if self.local_kill_present():
            log.warning("clear_kill refused: local KILL file still present")
            return False
        self._latched_kill = False
        self._uncertain = False
        self._kill_announced = False
        log.info("kill latch cleared by explicit operator confirmation")
        return True

    def new_entries_blocked(self) -> bool:
        """The gate the order path calls before any BUY. Blocks on kill,
        operator pause, or DB uncertainty. NEVER blocks sells/exits."""
        if self.kill_engaged():
            return True
        if self._uncertain:
            return True
        if self._cmd_pause:
            return True
        s = self._last_settings
        if s is not None and s.pause_new_entries:
            return True
        return False

    def set_pause(self, paused: bool) -> None:
        """In-memory pause toggle driven by 'pause'/'resume' commands."""
        self._cmd_pause = bool(paused)

    def set_strategy_disabled(self, name: str, disabled: bool) -> None:
        """Enable/disable a single strategy ('equity'|'fno'|'meanrev')."""
        self._strategy_disabled[name] = bool(disabled)

    # ------------------------------------------------------------------ #
    # SETTINGS (read every tick, clamped)
    # ------------------------------------------------------------------ #
    def read_settings(self) -> EffectiveSettings:
        """Read settings fresh each tick (no startup caching) and clamp to sane
        bounds against the live wallet. Fail-soft: falls back to last-known,
        then hard defaults."""
        if not self._db_enabled:
            eff = self._clamp(self._defaults(), source="default")
            self._last_settings = eff
            return eff
        try:
            raw = self._db.get_settings() or {}
        except Exception as e:
            log.warning("settings read failed (%s) — using last-known/default", e)
            eff = self._last_settings or self._clamp(self._defaults(), "default")
            return eff
        if not raw:
            eff = self._clamp(self._defaults(), source="default")
            self._last_settings = eff
            return eff

        merged = self._defaults()
        merged.update(
            {
                "per_trade_cap_inr": _num(
                    raw.get("max_per_fno_trade_inr"), merged["per_trade_cap_inr"]
                ),
                "daily_loss_cap_inr": _num(
                    raw.get("daily_loss_cap_inr"), merged["daily_loss_cap_inr"]
                ),
                "max_concurrent_fno_positions": int(
                    _num(
                        raw.get("max_concurrent_fno_positions"),
                        merged["max_concurrent_fno_positions"],
                    )
                ),
                "pause_new_entries": bool(raw.get("pause_new_entries", False)),
                "kill_switch": bool(raw.get("kill_switch", False)),
            }
        )
        eff = self._clamp(merged, source="db")
        self._last_settings = eff
        return eff

    def _defaults(self) -> Dict[str, Any]:
        return {
            "per_trade_cap_inr": _DEFAULT_PER_TRADE_CAP,
            "daily_loss_cap_inr": _DEFAULT_DAILY_LOSS_CAP,
            "max_concurrent_fno_positions": _DEFAULT_MAX_FNO_POS,
            "pause_new_entries": False,
            "kill_switch": False,
        }

    def _clamp(self, d: Dict[str, Any], source: str) -> EffectiveSettings:
        try:
            wallet = max(0.0, float(self._wallet() or 0.0))
        except Exception:
            wallet = 0.0
        per_trade = max(0.0, float(d["per_trade_cap_inr"]))
        daily = max(0.0, float(d["daily_loss_cap_inr"]))
        # per-trade cap can never exceed wallet (only clamp when wallet is known)
        if wallet > 0:
            per_trade = min(per_trade, wallet)
            daily = min(daily, wallet * _DAILY_CAP_WALLET_FRACTION)
        return EffectiveSettings(
            per_trade_cap_inr=per_trade,
            daily_loss_cap_inr=daily,
            max_concurrent_fno_positions=max(0, int(d["max_concurrent_fno_positions"])),
            pause_new_entries=bool(d["pause_new_entries"]),
            kill_switch=bool(d["kill_switch"]),
            source=source,
        )

    # ------------------------------------------------------------------ #
    # COMMANDS (exactly-once)
    # ------------------------------------------------------------------ #
    def strategy_enabled(self, name: str) -> bool:
        return not self._strategy_disabled.get(name, False)

    def fetch_pending(self) -> List[Dict[str, Any]]:
        """Read-only peek at the queue (used by the scan-once proof)."""
        if not self._db_enabled:
            return []
        try:
            return list(self._db.fetch_pending_commands())
        except Exception as e:
            log.warning("fetch_pending failed: %s", e)
            return []

    def consume_commands(self, handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Claim each pending command atomically and dispatch it exactly once.

        For every pending row we attempt ``claim_command`` (conditional
        pending->consumed). Only the worker that wins the claim runs the
        handler; losers/already-claimed rows are skipped. The handler's return
        dict is stored as the command result. Unknown commands are recorded as
        failed. Kill overrides everything: handlers themselves re-check the gate
        where relevant (manual entries must respect kill/caps)."""
        results: List[Dict[str, Any]] = []
        if not self._db_enabled:
            return results
        try:
            pending = list(self._db.fetch_pending_commands())
        except Exception as e:
            log.warning("consume_commands fetch failed: %s", e)
            return results

        for row in pending:
            cid = row.get("id")
            command = row.get("command")
            if not cid:
                continue
            claimed = self._db.claim_command(str(cid))
            if not claimed:
                continue  # someone else won the claim — exactly-once preserved
            # Audit-only rows (payload.audit == true) are durable records of a
            # manual action written by the API routes; they trigger no behaviour.
            payload = claimed.get("payload") or {}
            if isinstance(payload, dict) and payload.get("audit") is True:
                self._db.finish_command(str(cid), {"ok": True, "action": "audit_recorded"}, ok=True)
                results.append({"id": cid, "command": command, "ok": True, "audit": True})
                continue
            handler = handlers.get(command)
            try:
                if handler is None:
                    self._db.finish_command(
                        str(cid), {"ok": False, "error": f"no handler for '{command}'"}, ok=False
                    )
                    results.append({"id": cid, "command": command, "ok": False})
                    continue
                out = handler(claimed) or {}
                ok = bool(out.get("ok", True))
                self._db.finish_command(str(cid), out, ok=ok)
                results.append({"id": cid, "command": command, "ok": ok})
            except Exception as e:
                log.warning("command %s (%s) handler raised: %s", cid, command, e)
                self._db.finish_command(str(cid), {"ok": False, "error": str(e)}, ok=False)
                results.append({"id": cid, "command": command, "ok": False})
        return results

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict[str, Any]:
        """One-shot evidence bundle: kill (3 sources), effective settings,
        pending command count. Used by the --scan-once --server proof."""
        kv = self.refresh_kill()
        eff = self.read_settings()
        pending = self.fetch_pending()
        return {
            "kill": {
                "engaged": kv.engaged,
                "latched": kv.latched,
                "local_file": kv.local_file,
                "db_kill_switch": kv.db_kill_switch,
                "uncertain": kv.uncertain,
                "db_enabled": kv.db_enabled,
                "sources": kv.sources,
            },
            "settings": {
                "per_trade_cap_inr": eff.per_trade_cap_inr,
                "daily_loss_cap_inr": eff.daily_loss_cap_inr,
                "max_concurrent_fno_positions": eff.max_concurrent_fno_positions,
                "pause_new_entries": eff.pause_new_entries,
                "kill_switch": eff.kill_switch,
                "source": eff.source,
            },
            "new_entries_blocked": self.new_entries_blocked(),
            "pending_commands": [
                {"id": c.get("id"), "command": c.get("command")} for c in pending
            ],
        }


def _num(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


__all__ = ["ControlPlane", "EffectiveSettings", "KillView"]

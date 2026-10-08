"""Flask HTTP layer for groww_bot.py.

Exposes a status JSON endpoint plus a minimal cockpit HTML page on
127.0.0.1:8080. Designed to be started from the bot's main thread via
``BotServer(state_provider).start_in_background()``.

The bot owns all state; this module owns nothing mutable. The ``state_provider``
callable is invoked on every request and must return a fresh snapshot dict.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Protocol

from flask import Flask, jsonify, make_response

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

StateSnapshot = Dict[str, Any]


class StateProvider(Protocol):
    """Returns a fresh snapshot dict on every call.

    Expected keys (all optional, coerced to safe defaults if missing):
    pnl_today, wallet_inr, positions[], trades_today, strategies[],
    capital_utilization, mode ("paper" | "live").
    """

    def __call__(self) -> StateSnapshot: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_positions(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [
        {
            "symbol": str(p.get("symbol", "")),
            "qty": _i(p.get("qty", p.get("quantity", 0))),
            "entry_price": _f(p.get("entry_price", p.get("entryPrice", 0))),
            "current_price": _f(p.get("current_price", p.get("currentPrice", 0))),
            "pnl": _f(p.get("pnl", 0)),
        }
        for p in raw
        if isinstance(p, dict)
    ]


def _normalize_strategies(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [
        {
            "name": str(s.get("name", s.get("strategy", ""))),
            "win_rate": _f(s.get("win_rate", s.get("winRate", 0))),
            "trades": _i(s.get("trades", 0)),
        }
        for s in raw
        if isinstance(s, dict)
    ]


def _normalize_snapshot(snap: Optional[StateSnapshot]) -> Dict[str, Any]:
    snap = snap or {}
    return {
        "pnl_today": _f(snap.get("pnl_today")),
        "wallet_inr": _f(snap.get("wallet_inr")),
        "positions": _normalize_positions(snap.get("positions")),
        "trades_today": _i(snap.get("trades_today")),
        "strategies": _normalize_strategies(snap.get("strategies")),
        "capital_utilization": _f(snap.get("capital_utilization")),
        "mode": "live" if snap.get("mode") == "live" else "paper",
    }


# ---------------------------------------------------------------------------
# Cockpit HTML (self-contained, no CDNs)
# ---------------------------------------------------------------------------

_COCKPIT_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>JoSho Trader</title>
<style>
:root{--bg:#0a0a0a;--fg:#e5e5e5;--mut:#888;--red:#ef4444;--grn:#22c55e;--card:#141414;--ln:#1f1f1f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}
.wrap{max-width:1100px;margin:0 auto;padding:18px}
h1{font-size:13px;letter-spacing:.18em;color:var(--mut);margin:0 0 14px;text-transform:uppercase}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--ln);border-radius:8px;padding:12px}
.lbl{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.1em}
.val{font-size:22px;margin-top:4px;font-variant-numeric:tabular-nums}
.pos{color:var(--grn)}.neg{color:var(--red)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--ln)}
th{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.1em;font-weight:normal}
.mode{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;letter-spacing:.1em;text-transform:uppercase}
.mode.live{background:#3b0a0a;color:#fca5a5}.mode.paper{background:#0a1a3b;color:#93c5fd}
.empty{color:var(--mut);padding:14px;text-align:center}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--grn);margin-right:6px;vertical-align:middle}
.dot.off{background:var(--red)}
.foot{margin-top:14px;color:var(--mut);font-size:11px}
</style></head><body>
<div class="wrap">
<h1><span id="dot" class="dot"></span>JoSho Trader <span id="mode" class="mode paper">paper</span></h1>
<div class="grid">
<div class="card"><div class="lbl">Wallet (INR)</div><div class="val" id="wallet">--</div></div>
<div class="card"><div class="lbl">P&amp;L Today</div><div class="val" id="pnl">--</div></div>
<div class="card"><div class="lbl">Trades Today</div><div class="val" id="trades">--</div></div>
<div class="card"><div class="lbl">Capital Util</div><div class="val" id="util">--</div></div>
</div>
<div class="card">
<div class="lbl" style="margin-bottom:8px">Open Positions</div>
<table><thead><tr><th>Symbol</th><th>Qty</th><th>Entry</th><th>LTP</th><th>P&amp;L</th></tr></thead>
<tbody id="positions"></tbody></table>
<div id="empty" class="empty" style="display:none">No open positions.</div>
</div>
<div class="foot">Auto-refresh every 5s. Endpoint: /api/status</div>
</div>
<script>
function fmt(n,d){return Number(n).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});}
function cls(n){return n>0?"pos":(n<0?"neg":"");}
async function tick(){
  try{
    const r=await fetch("/api/status",{cache:"no-store"});
    if(!r.ok)throw 0;
    const d=await r.json();
    document.getElementById("dot").className="dot";
    document.getElementById("wallet").textContent=fmt(d.wallet_inr,2);
    const p=document.getElementById("pnl");p.textContent=fmt(d.pnl_today,2);p.className="val "+cls(d.pnl_today);
    document.getElementById("trades").textContent=d.trades_today;
    document.getElementById("util").textContent=fmt((d.capital_utilization||0)*100,1)+"%";
    const m=document.getElementById("mode");m.textContent=d.mode;m.className="mode "+d.mode;
    const tb=document.getElementById("positions");tb.innerHTML="";
    const rows=d.positions||[];
    document.getElementById("empty").style.display=rows.length?"none":"block";
    for(const pos of rows){
      const tr=document.createElement("tr");
      tr.innerHTML="<td>"+pos.symbol+"</td><td>"+pos.qty+"</td><td>"+fmt(pos.entry_price,2)+
        "</td><td>"+fmt(pos.current_price,2)+"</td><td class=\\""+cls(pos.pnl)+"\\">"+fmt(pos.pnl,2)+"</td>";
      tb.appendChild(tr);
    }
  }catch(e){document.getElementById("dot").className="dot off";}
}
tick();setInterval(tick,5000);
</script></body></html>"""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class BotServer:
    """Flask cockpit server. Pass a ``state_provider`` callable; do not share state."""

    def __init__(
        self,
        state_provider: Callable[[], StateSnapshot],
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._provider = state_provider
        self._host = host
        self._port = port
        self._log = logger or logging.getLogger("bot_server")
        self._thread: Optional[threading.Thread] = None
        self._app = self._build_app()

    def _build_app(self) -> Flask:
        app = Flask(__name__)
        # Silence Werkzeug request logs; bot owns the log stream.
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

        @app.get("/api/status")
        def status():  # type: ignore[no-untyped-def]
            try:
                snap = self._provider()
            except Exception as exc:  # fail-soft per contract
                self._log.warning("state_provider raised: %s", exc)
                snap = {}
            return jsonify(_normalize_snapshot(snap))

        @app.get("/healthz")
        def healthz():  # type: ignore[no-untyped-def]
            return jsonify({"ok": True})

        @app.get("/")
        def cockpit():  # type: ignore[no-untyped-def]
            resp = make_response(_COCKPIT_HTML)
            resp.headers["Content-Type"] = "text/html; charset=utf-8"
            resp.headers["Cache-Control"] = "no-store"
            return resp

        return app

    def _serve(self) -> None:
        # Suppress Flask's development banner.
        import flask.cli as _cli

        _cli.show_server_banner = lambda *a, **k: None  # type: ignore[assignment]
        try:
            self._app.run(
                host=self._host,
                port=self._port,
                debug=False,
                use_reloader=False,
                threaded=True,
            )
        except KeyboardInterrupt:
            self._log.info("bot_server: KeyboardInterrupt, shutting down")
        except Exception as exc:
            self._log.error("bot_server crashed: %s", exc)

    def start_in_background(self) -> threading.Thread:
        """Spawn the Flask server on a daemon thread and return it."""
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        t = threading.Thread(target=self._serve, name="bot_server", daemon=True)
        t.start()
        self._thread = t
        self._log.info("bot_server listening on http://%s:%d", self._host, self._port)
        return t


__all__ = ["BotServer", "StateProvider", "StateSnapshot"]

"""Codifies the manual dry-run proof: `groww_bot.py --scan-once --server` exits 0
with a fake broker and NO Supabase env (disabled branch), prints the control-plane
snapshot, and the bot_server /api/status endpoint answers. Fully hermetic."""

import subprocess
import sys
from pathlib import Path

from bot_server import BotServer

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent  # automation/josho-trader
_FAKEGROWW = _HERE / "_fakegroww"


def _hermetic_env() -> dict:
    import os

    env = dict(os.environ)
    # Disabled branch: ensure NO Supabase creds leak in from the parent shell.
    for k in (
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "NEXT_PUBLIC_SUPABASE_URL",
        "SOHAYA__NEXT_PUBLIC_SUPABASE_URL",
        "SOHAYA__SUPABASE_SERVICE_ROLE_KEY",
        "SHARED__SUPABASE_SERVICE_ROLE_KEY",
    ):
        env.pop(k, None)
    env["PYTHONUTF8"] = "1"  # avoid the cp1252 banner crash on Windows
    env["GROWW_ACCESS_TOKEN"] = "fake-token"  # so connect() succeeds (vault won't override a set var)
    # Prepend the fake growwapi so `import growwapi` resolves to the stub.
    env["PYTHONPATH"] = str(_FAKEGROWW) + (";" if sys.platform == "win32" else ":") + env.get("PYTHONPATH", "")
    return env


def test_scan_once_server_exits_zero_and_prints_snapshot():
    proc = subprocess.run(
        [sys.executable, "groww_bot.py", "--scan-once", "--server", "--strategy", "fno"],
        cwd=str(_ROOT),
        env=_hermetic_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"exit={proc.returncode}\n{combined}"
    assert "[CONTROL PLANE]" in combined  # snapshot printed
    assert "new_entries_blocked" in combined
    assert "bot_server listening" in combined  # --server started


def test_api_status_answers_in_process():
    srv = BotServer(lambda: {"wallet_inr": 10100, "pnl_today": 5, "positions": [], "mode": "paper"})
    client = srv._app.test_client()  # exercises routes without binding a socket
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["wallet_inr"] == 10100.0
    assert data["mode"] == "paper"
    assert client.get("/healthz").status_code == 200

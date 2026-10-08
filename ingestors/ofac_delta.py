"""ofac_delta.py -- Layer 4 ingestor #6 (OFAC SDN delta watcher).

Pure ingestion: polls the Treasury OFAC SDN CSV, hashes the body, and emits
one event per ADDED or LIFTED entry since the last poll. Does NOT classify
(Agent 8), does NOT write to DB (Agent 13), and does NOT import from sibling
Layer-4 modules. The OFAC helper in signal_worker.py is intentionally NOT
reused -- this is a fresh implementation per the spec.

First run captures a baseline (current hash + row set) and returns []. Only
subsequent runs that observe a hash change emit signals.

Cache files (under ./data/, created on first run):
  - data/ofac_hash.json   : {"hash": "<sha256>", "captured_at": "<iso>"}
  - data/ofac_state.json  : [{"ent_num","name","type","program"}, ...]

CLI: python ingestors/ofac_delta.py [--dry-run] [--verbose]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

UTC = timezone.utc
SOURCE_NAME: str = "ofac"
USER_AGENT: str = "JoShoTrader-OFACDelta/1.0 (+contact: contact@example.com)"
OFAC_SDN_URL: str = "https://www.treasury.gov/ofac/downloads/sdn.csv"
HTTP_TIMEOUT_S: int = 60
MAX_RETRIES: int = 3
RETRY_BACKOFF_BASE_S: float = 1.5

# OFAC SDN CSV is HEADERLESS. Column order (verified against live feed
# 2026-06-17 and Treasury's SDN.FF spec):
#   ent_num, SDN_Name, SDN_Type, Program, Title, Call_Sign, Vess_type,
#   Tonnage, GRT, Vess_flag, Vess_owner, Remarks
CSV_COLUMNS: tuple[str, ...] = ("ent_num", "SDN_Name", "SDN_Type", "Program", "Title",
                                "Call_Sign", "Vess_type", "Tonnage", "GRT",
                                "Vess_flag", "Vess_owner", "Remarks")
EMPTY_SENTINEL: str = "-0-"  # Treasury's "field N/A" marker

MODULE_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = MODULE_DIR.parent
DATA_DIR: Path = REPO_ROOT / "data"
HASH_CACHE_PATH: Path = DATA_DIR / "ofac_hash.json"
STATE_CACHE_PATH: Path = DATA_DIR / "ofac_state.json"

log = logging.getLogger("ingestors.ofac_delta")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    )

def _ensure_data_dir() -> None:
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

def _load_prev_hash() -> str | None:
    if not HASH_CACHE_PATH.exists():
        return None
    try:
        blob = json.loads(HASH_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("Could not read %s: %s", HASH_CACHE_PATH, exc)
        return None
    h = blob.get("hash")
    return h if isinstance(h, str) and h else None

def _load_prev_state() -> dict[str, dict[str, str]]:
    if not STATE_CACHE_PATH.exists():
        return {}
    try:
        blob = json.loads(STATE_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("Could not read %s: %s", STATE_CACHE_PATH, exc)
        return {}
    if not isinstance(blob, list):
        log.warning("Malformed state cache (not a list); ignoring.")
        return {}
    out: dict[str, dict[str, str]] = {}
    for row in blob:
        if not isinstance(row, dict):
            continue
        ent_num = str(row.get("ent_num") or "").strip()
        if not ent_num:
            continue
        out[ent_num] = {"ent_num": ent_num,
                        "name": str(row.get("name") or ""),
                        "type": str(row.get("type") or ""),
                        "program": str(row.get("program") or "")}
    return out

def _save_hash(new_hash: str) -> None:
    _ensure_data_dir()
    payload = {"hash": new_hash, "captured_at": datetime.now(UTC).isoformat()}
    HASH_CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _save_state(rows_by_ent: dict[str, dict[str, str]]) -> None:
    _ensure_data_dir()
    serialised = sorted(rows_by_ent.values(), key=lambda r: r.get("ent_num", ""))
    STATE_CACHE_PATH.write_text(
        json.dumps(serialised, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

def _fetch_sdn_body() -> bytes | None:
    """Download SDN CSV with bounded retries. Returns body bytes or None."""
    headers = {"User-Agent": USER_AGENT, "Accept": "text/csv,*/*"}
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(OFAC_SDN_URL, headers=headers, timeout=HTTP_TIMEOUT_S)
            if resp.status_code == 200 and resp.content:
                return resp.content
            if 500 <= resp.status_code < 600:
                log.warning("OFAC HTTP %d (attempt %d/%d)", resp.status_code, attempt, MAX_RETRIES)
                time.sleep(RETRY_BACKOFF_BASE_S ** attempt)
                continue
            log.warning("OFAC HTTP %d -- not retrying. body[:160]=%r", resp.status_code, resp.text[:160])
            return None
        except requests.RequestException as exc:
            last_exc = exc
            log.warning("OFAC fetch error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
            time.sleep(RETRY_BACKOFF_BASE_S ** attempt)
    log.warning("OFAC gave up after %d attempts: %s", MAX_RETRIES, last_exc)
    return None

def _clean_cell(val: str | None) -> str:
    """Strip + map Treasury '-0-' sentinel to empty string."""
    if val is None:
        return ""
    s = val.strip()
    return "" if s == EMPTY_SENTINEL else s

def _parse_sdn_rows(body: bytes) -> dict[str, dict[str, str]]:
    """Parse SDN CSV body -> dict keyed by ent_num. Bad rows skipped."""
    text = body.decode("latin-1", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows: dict[str, dict[str, str]] = {}
    bad = 0
    for raw in reader:
        if not raw:
            continue
        if len(raw) < 4:
            bad += 1
            continue
        ent_num = _clean_cell(raw[0])
        if not ent_num:
            bad += 1
            continue
        rows[ent_num] = {"ent_num": ent_num,
                         "name": _clean_cell(raw[1]),
                         "type": _clean_cell(raw[2]),
                         "program": _clean_cell(raw[3])}
    if bad:
        log.info("OFAC parse: skipped %d malformed rows", bad)
    return rows

def _program_themes(program: str) -> list[str]:
    """Map OFAC Program tag -> coarse situational themes. May be empty."""
    p = (program or "").upper()
    out: list[str] = []
    if "RUSSIA" in p or "UKR" in p or "BELARUS" in p:
        out.append("sanction_russia")
    if "IRAN" in p:
        out.append("sanction_iran")
    if "CHINA" in p or "PRC" in p or "HK-EO13936" in p:
        out.append("sanction_china")
    return out

def _build_event(row: dict[str, str], sub_action: str, detected_at_iso: str) -> dict[str, Any]:
    """Construct one raw event dict for an added or lifted SDN row."""
    name = row.get("name") or "(unnamed entity)"
    program = row.get("program") or ""
    sdn_type = row.get("type") or ""
    ent_num = row.get("ent_num") or ""
    themes = ["sanction", sub_action] + _program_themes(program)
    verb = "added" if sub_action == "added" else "lifted"
    headline = f"OFAC SDN {verb}: {name}" + (f" [{program}]" if program else "")
    return {
        "event_id": f"ofac:{ent_num}:{sub_action}",
        "source": SOURCE_NAME,
        "headline": headline[:500],
        "url": OFAC_SDN_URL,
        "timestamp": detected_at_iso,
        "raw_payload": {"ent_num": ent_num, "SDN_Name": name,
                        "SDN_Type": sdn_type, "Program": program,
                        "sub_action": sub_action},
        "entities_hint": {"themes": themes,
                          "organizations": [name] if name else [],
                          "programs": [program] if program else []},
    }

# --- Public API (frozen contract) --------------------------------------------
def poll(  # noqa: ARG001 -- `since` ignored (we diff by hash, not time)
    since: datetime | None = None,
    *,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Fetch OFAC SDN, hash-compare, emit added/lifted events.

    `since` accepted for ingestor-contract symmetry but ignored: OFAC has no
    per-row timestamp. Failsoft per spec sec 12: errors -> []. First run
    captures baseline silently and returns []. When dry_run=True, caches
    are NOT updated so re-runs reproduce the same delta.
    """
    body = _fetch_sdn_body()
    if body is None:
        log.warning("OFAC: empty body after retries; returning [].")
        return []
    new_hash = hashlib.sha256(body).hexdigest()
    prev_hash = _load_prev_hash()
    # First-run baseline path: never emit events.
    if prev_hash is None:
        try:
            current_rows = _parse_sdn_rows(body)
        except Exception as exc:  # noqa: BLE001
            log.warning("OFAC first-run parse failed: %s", exc)
            return []
        if not dry_run:
            _save_hash(new_hash)
            _save_state(current_rows)
            log.info("OFAC first-run baseline captured (hash=%s..., rows=%d).",
                     new_hash[:12], len(current_rows))
        else:
            log.info("OFAC first-run baseline observed but NOT saved (--dry-run); "
                     "hash=%s... rows=%d.", new_hash[:12], len(current_rows))
        return []

    if new_hash == prev_hash:
        log.info("OFAC hash unchanged (%s...). No delta.", new_hash[:12])
        return []
    log.info("OFAC hash changed (%s... -> %s...). Computing delta.",
             prev_hash[:12], new_hash[:12])
    try:
        current_rows = _parse_sdn_rows(body)
    except Exception as exc:  # noqa: BLE001
        log.warning("OFAC parse failed after hash change: %s", exc)
        return []
    prev_rows = _load_prev_state()
    detected_at_iso = datetime.now(UTC).isoformat()
    prev_keys = set(prev_rows.keys())
    cur_keys = set(current_rows.keys())
    added_keys = sorted(cur_keys - prev_keys)
    removed_keys = sorted(prev_keys - cur_keys)
    events: list[dict[str, Any]] = []
    for ent in added_keys:
        events.append(_build_event(current_rows[ent], "added", detected_at_iso))
    for ent in removed_keys:
        events.append(_build_event(prev_rows[ent], "lifted", detected_at_iso))
    log.info("OFAC delta: added=%d lifted=%d total=%d",
             len(added_keys), len(removed_keys), len(events))
    if not dry_run:
        _save_hash(new_hash)
        _save_state(current_rows)
    else:
        log.info("OFAC --dry-run: cache NOT updated; delta will re-surface.")
    return events

def poll_once_cli() -> int:
    """argparse entry. Exit 0 on success (incl. quiet runs), 1 on crash."""
    parser = argparse.ArgumentParser(
        prog="ofac_delta",
        description="Poll OFAC SDN CSV for added/lifted sanctions.",
    )
    parser.add_argument("--since", type=str, default=None, help="Accepted for symmetry; ignored.")
    parser.add_argument("--dry-run", action="store_true", help="Print JSON lines; do NOT update cache.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        log.setLevel(logging.DEBUG)
    log.info("poll_once_cli starting | dry_run=%s verbose=%s", args.dry_run, args.verbose)
    try:
        events = poll(since=None, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001 -- top-level CLI guard
        log.exception("poll() crashed: %s", exc)
        return 1
    if not events:
        log.info("OFAC: no delta this cycle.")
        return 0
    if args.dry_run:
        for ev in events:
            sys.stdout.write(json.dumps(ev, ensure_ascii=False, sort_keys=True) + "\n")
        sys.stdout.flush()
    else:
        log.info("OFAC: %d delta events returned (persistence is Agent 13's job).", len(events))
    return 0

if __name__ == "__main__":
    sys.exit(poll_once_cli())

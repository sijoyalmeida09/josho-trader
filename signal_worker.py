"""
signal_worker.py — JoSho Trader Layer 4 poller.

Standalone process. Polls free geopolitical / macro feeds, normalizes events
into the `trader_signals` Supabase table. Does NOT trade, does NOT touch
positions or settings.

Free sources wired:
  1. GDELT 2.0 ArtList   — global news tone for curated keyword set
  2. NASA EONET v3       — open natural disasters inside the India bbox
  3. OFAC SDN CSV delta  — sanctions list hash change since last run

(NOAA placeholder skipped — US weather rarely tradeable on Indian tape.)

Run:
    python signal_worker.py --once
    python signal_worker.py --interval 15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"
DATA_DIR = HERE / "data"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
OFAC_HASH_FILE = DATA_DIR / "ofac_hash.json"

# Vault loader — reuse groww_bot's logic.
# TODO: refactor groww_bot._load_vault_env into a shared module so this
# fallback can disappear. For now keep both copies in sync manually.
try:
    from groww_bot import _load_vault_env, VAULT_FILE  # type: ignore
    _VAULT_REPORT = _load_vault_env()
except Exception:
    VAULT_FILE = HERE.parent / "secrets-os" / "vault.master.env"
    _MINI_MAP = {
        "SHARED__SUPABASE_URL":              ["SUPABASE_URL"],
        "SOHAYA__NEXT_PUBLIC_SUPABASE_URL":  ["SUPABASE_URL"],
        "SHARED__SUPABASE_SERVICE_ROLE_KEY": ["SUPABASE_SERVICE_ROLE_KEY"],
        "TELEGRAM_BOT_TOKEN":        ["TELEGRAM_BOT_TOKEN"],
        "TELEGRAM_CHAT_ID":    ["TELEGRAM_CHAT_ID"],
    }
    if VAULT_FILE.exists():
        for raw in VAULT_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            for env_name in _MINI_MAP.get(k, []):
                if v and not os.environ.get(env_name):
                    os.environ[env_name] = v
    _VAULT_REPORT = {"vault_path": str(VAULT_FILE), "exists": VAULT_FILE.exists()}

from situational_map import SITUATIONAL_MAP, affected_symbols_for  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] signal_worker | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "signal_worker.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("signal_worker")

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc
USER_AGENT = "JoShoTrader-SignalWorker/1.0 (+contact: contact@example.com)"
HTTP_TIMEOUT = 20

GDELT_QUERIES: list[tuple[str, str]] = [
    ("Iran oil",                  "oil_shock"),
    ("Strait of Hormuz",          "oil_shock"),
    ("Israel Hamas",              "oil_shock"),
    ("India Pakistan border",     "defense_flashpoint"),
    ("India China border",        "defense_flashpoint"),
    ("submarine cable cut",       "cable_cut"),
    ("Red Sea cable",             "cable_cut"),
    ("OFAC sanction Russia",      "sanction"),
    ("RBI repo rate",             "india_policy_event"),
    ("NVIDIA AI compute",         "ai_compute_buildout"),
]
GDELT_REQ_SPACING_S = 1.2  # GDELT politeness cap ~1 req/sec

INDIA_BBOX = {"lat_min": 6.0, "lat_max": 37.0, "lon_min": 68.0, "lon_max": 97.0}
EONET_CATEGORIES = [("cyclones", "cyclone"), ("severeStorms", "cyclone")]
OFAC_SDN_URL = "https://www.treasury.gov/ofac/downloads/sdn.csv"

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "").strip()


def tg(msg: str) -> None:
    if not (TG_TOKEN and TG_CHAT):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg},
            timeout=10,
        )
    except Exception as exc:
        log.debug("Telegram post failed: %s", exc)


# ─────────────────────────────────────────────────────────────
# Supabase REST helpers — same fail-soft pattern as trader_db.
# ─────────────────────────────────────────────────────────────
def _sb_headers() -> dict[str, str] | None:
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not key:
        return None
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _sb_base() -> str | None:
    url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    return f"{url}/rest/v1" if url else None


def fetch_active_signals() -> list[dict]:
    """Return non-expired trader_signals rows. Used for dedup. Fail-soft."""
    base, hdrs = _sb_base(), _sb_headers()
    if not (base and hdrs):
        return []
    try:
        r = requests.get(
            f"{base}/trader_signals",
            headers=hdrs,
            params={
                "select": "id,source,category,url,severity,expires_at",
                "expires_at": f"gt.{datetime.now(UTC).isoformat()}",
            },
            timeout=HTTP_TIMEOUT,
        )
        if r.status_code == 200:
            return r.json() or []
        log.warning("fetch_active_signals HTTP %s: %s", r.status_code, r.text[:200])
    except Exception as exc:
        log.warning("fetch_active_signals error: %s", exc)
    return []


def insert_signal(signal: dict) -> bool:
    base, hdrs = _sb_base(), _sb_headers()
    if not (base and hdrs):
        log.debug("No Supabase creds — would have inserted: %s", signal.get("headline"))
        return False
    try:
        r = requests.post(f"{base}/trader_signals", headers=hdrs, json=signal, timeout=HTTP_TIMEOUT)
        if 200 <= r.status_code < 300:
            return True
        log.warning("insert_signal HTTP %s: %s", r.status_code, r.text[:300])
    except Exception as exc:
        log.warning("insert_signal error: %s", exc)
    return False


# ─────────────────────────────────────────────────────────────
# Normalizer
# ─────────────────────────────────────────────────────────────
def build_signal(*, source: str, category: str, headline: str, url: str,
                 severity: int | None = None, payload: dict | None = None) -> dict:
    spec = SITUATIONAL_MAP.get(category, {})
    sev = severity if severity is not None else spec.get("default_severity", 5)
    sev = max(1, min(10, int(sev)))
    ttl_hours = int(spec.get("ttl_hours", 24))
    expires_at = (datetime.now(UTC) + timedelta(hours=ttl_hours)).isoformat()
    boosts = list((spec.get("indian_symbols_boost") or {}).values())
    blocks = [abs(v) for v in (spec.get("indian_symbols_block") or {}).values()]
    headline_adj = max(boosts + blocks + [0])
    return {
        "source": source,
        "category": category,
        "severity": sev,
        "affected_symbols": affected_symbols_for(category),
        "score_adjustment": int(headline_adj),
        "headline": (headline or "")[:500],
        "url": (url or "")[:1000] or None,
        "payload": payload or {},
        "expires_at": expires_at,
    }


def is_duplicate(active: list[dict], source: str, category: str, url: str | None) -> bool:
    return any(
        s.get("source") == source and s.get("category") == category
        and (s.get("url") or "") == (url or "")
        for s in active
    )


# ─────────────────────────────────────────────────────────────
# Source 1 — GDELT 2.0
# ─────────────────────────────────────────────────────────────
def poll_gdelt(active: list[dict]) -> list[dict]:
    new_signals: list[dict] = []
    for idx, (keyword, category) in enumerate(GDELT_QUERIES):
        if idx > 0:
            time.sleep(GDELT_REQ_SPACING_S)
        try:
            r = requests.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={"query": keyword, "mode": "ArtList", "format": "json",
                        "maxrecords": 20, "sort": "DateDesc"},
                headers={"User-Agent": USER_AGENT},
                timeout=HTTP_TIMEOUT,
            )
            if r.status_code != 200:
                log.info("GDELT %s -> HTTP %s", keyword, r.status_code)
                continue
            articles = (r.json() or {}).get("articles", [])
        except Exception as exc:
            log.warning("GDELT fetch failed for %s: %s", keyword, exc)
            continue

        worst, worst_tone = None, 0.0
        for art in articles:
            try:
                tone = float(art.get("tone", 0.0))
            except (TypeError, ValueError):
                tone = 0.0
            if tone < worst_tone:
                worst_tone, worst = tone, art
        if not worst or worst_tone >= -5.0:
            continue

        severity = 8 if worst_tone < -8.0 else 5
        url = worst.get("url") or ""
        if is_duplicate(active, "gdelt", category, url):
            continue
        new_signals.append(build_signal(
            source="gdelt",
            category=category,
            headline=f"[{keyword}] {worst.get('title', '')}",
            url=url,
            severity=severity,
            payload={"keyword": keyword, "tone": worst_tone,
                     "seendate": worst.get("seendate"), "domain": worst.get("domain")},
        ))
    return new_signals


# ─────────────────────────────────────────────────────────────
# Source 2 — NASA EONET v3
# ─────────────────────────────────────────────────────────────
def _coord_in_india(coord: list) -> bool:
    # EONET geometry coords are [lon, lat]
    if not coord or len(coord) < 2:
        return False
    try:
        lon, lat = float(coord[0]), float(coord[1])
    except (TypeError, ValueError):
        return False
    return (INDIA_BBOX["lat_min"] <= lat <= INDIA_BBOX["lat_max"]
            and INDIA_BBOX["lon_min"] <= lon <= INDIA_BBOX["lon_max"])


def poll_eonet(active: list[dict]) -> list[dict]:
    new_signals: list[dict] = []
    for eonet_cat, our_cat in EONET_CATEGORIES:
        try:
            r = requests.get(
                "https://eonet.gsfc.nasa.gov/api/v3/events",
                params={"status": "open", "category": eonet_cat},
                headers={"User-Agent": USER_AGENT},
                timeout=HTTP_TIMEOUT,
            )
            if r.status_code != 200:
                log.info("EONET %s -> HTTP %s", eonet_cat, r.status_code)
                continue
            events = (r.json() or {}).get("events", [])
        except Exception as exc:
            log.warning("EONET fetch failed for %s: %s", eonet_cat, exc)
            continue

        for ev in events:
            geoms = ev.get("geometry", []) or []
            if not any(_coord_in_india(g.get("coordinates")) for g in geoms):
                continue
            sources = ev.get("sources") or []
            ev_url = (sources[0].get("url", "") if sources else "") or ""
            if is_duplicate(active, "eonet", our_cat, ev_url):
                continue
            severity = None
            for g in geoms:
                mv = g.get("magnitudeValue")
                if mv is not None:
                    try:
                        severity = max(severity or 0, int(round(float(mv))))
                    except (TypeError, ValueError):
                        pass
            new_signals.append(build_signal(
                source="eonet",
                category=our_cat,
                headline=ev.get("title", "EONET event"),
                url=ev_url,
                severity=severity or 6,
                payload={"eonet_id": ev.get("id"), "category": eonet_cat},
            ))
    return new_signals


# ─────────────────────────────────────────────────────────────
# Source 3 — OFAC SDN delta
# ─────────────────────────────────────────────────────────────
def poll_ofac(active: list[dict]) -> list[dict]:
    try:
        r = requests.get(OFAC_SDN_URL, headers={"User-Agent": USER_AGENT}, timeout=60)
        if r.status_code != 200:
            log.info("OFAC SDN -> HTTP %s", r.status_code)
            return []
        new_hash = hashlib.sha256(r.content).hexdigest()
    except Exception as exc:
        log.warning("OFAC fetch failed: %s", exc)
        return []

    prev_hash = None
    if OFAC_HASH_FILE.exists():
        try:
            prev_hash = json.loads(OFAC_HASH_FILE.read_text(encoding="utf-8")).get("hash")
        except Exception:
            prev_hash = None

    if prev_hash == new_hash:
        return []

    OFAC_HASH_FILE.write_text(
        json.dumps({"hash": new_hash, "checked_at": datetime.now(UTC).isoformat()}),
        encoding="utf-8",
    )

    if prev_hash is None:
        log.info("OFAC SDN baseline recorded (hash=%s...)", new_hash[:12])
        return []
    if is_duplicate(active, "ofac", "sanction", OFAC_SDN_URL):
        return []
    return [build_signal(
        source="ofac",
        category="sanction",
        headline="OFAC SDN list delta detected",
        url=OFAC_SDN_URL,
        severity=6,
        payload={"prev_hash": prev_hash[:16], "new_hash": new_hash[:16]},
    )]


# ─────────────────────────────────────────────────────────────
# One full poll cycle
# ─────────────────────────────────────────────────────────────
def run_once() -> int:
    log.info("Starting poll cycle")
    active = fetch_active_signals()
    log.info("Active signals in DB: %d", len(active))

    candidates: list[dict] = []
    for poller in (poll_gdelt, poll_eonet, poll_ofac):
        try:
            candidates.extend(poller(active))
        except Exception as exc:
            log.exception("Poller %s crashed: %s", poller.__name__, exc)

    inserted = 0
    for sig in candidates:
        if insert_signal(sig):
            inserted += 1
            log.info("INSERT [%s/%s] sev=%d adj=+%d aff=%d %s",
                     sig["source"], sig["category"], sig["severity"],
                     sig["score_adjustment"], len(sig["affected_symbols"]),
                     sig["headline"][:80])
            if sig["severity"] >= 7:
                tg(f"[Trader Signal] {sig['source'].upper()} {sig['category']} "
                   f"sev={sig['severity']}\n{sig['headline']}\n{sig.get('url') or ''}")

    log.info("Cycle done. candidates=%d inserted=%d", len(candidates), inserted)
    return inserted


# ─────────────────────────────────────────────────────────────
# Layer-4 orchestrator (Agent 15)
# ─────────────────────────────────────────────────────────────
#
# Wires the eight Layer-4 ingestors → classifier → situational mapping →
# confidence model → dedup+noise filter → signal writer into one pipeline.
#
# Sequential. Per-step failsoft. Pure additive — does not touch the legacy
# poll_gdelt/poll_eonet/poll_ofac path above.
#
# Public entrypoint:
#     orchestrate_cycle(dry_run=False, verbose=False) -> dict
#
# Return shape:
#     {
#       "ingested": int,         # raw events collected across all ingestors
#       "classified": int,       # events that passed the classifier
#       "dedupped": int,         # events surviving dedup + noise filter
#       "written": int,          # signals written (0 when dry_run)
#       "errors": list[str],     # one entry per failed step
#       "per_ingestor": dict,    # {ingestor_name: count}
#       "sample": list[dict],    # up to 3 example signals (verbose only)
#     }


_LAYER4_INGESTORS: tuple[str, ...] = (
    "gdelt_events",
    "gdelt_gkg",
    "rss_india",
    "filings",
    "natural_events",
    "ofac_delta",
    "macro",
)


def _import_layer4_module(name: str):
    """Try `ingestors.<name>` then flat fallback. Returns module or None."""
    # Try package-relative import first
    try:
        from importlib import import_module
        return import_module(f"ingestors.{name}")
    except Exception:
        pass
    try:
        from importlib import import_module
        return import_module(name)
    except Exception:
        return None


def _import_classifier():
    try:
        from importlib import import_module
        return import_module("classifier")
    except Exception:
        return None


def _import_situational_map():
    try:
        from importlib import import_module
        return import_module("situational_map")
    except Exception:
        return None


def _import_confidence_model():
    try:
        from importlib import import_module
        return import_module("confidence_model")
    except Exception:
        return None


def _import_dedup():
    try:
        from importlib import import_module
        return import_module("dedup")
    except Exception:
        return None


def _import_signal_writer():
    try:
        from importlib import import_module
        return import_module("signal_writer")
    except Exception:
        return None


def orchestrate_cycle(dry_run: bool = False, verbose: bool = False) -> dict:
    """Run all ingestors → classifier → mapping → confidence → dedup → writer.

    Returns {ingested, classified, dedupped, written, errors: list[str]}.
    Each step is wrapped — a failure in one ingestor or module is logged and
    the pipeline continues.
    """
    errors: list[str] = []
    per_ingestor: dict[str, int] = {}
    raw_events: list[dict] = []

    # ── Step 1: ingestors ────────────────────────────────────────────────
    for name in _LAYER4_INGESTORS:
        mod = _import_layer4_module(name)
        if mod is None:
            msg = f"ingestor:{name} import failed"
            errors.append(msg)
            log.warning(msg)
            per_ingestor[name] = 0
            continue
        poll_fn = getattr(mod, "poll", None)
        if poll_fn is None:
            msg = f"ingestor:{name} has no poll()"
            errors.append(msg)
            log.warning(msg)
            per_ingestor[name] = 0
            continue
        try:
            got = poll_fn() or []
            if not isinstance(got, list):
                got = list(got)
            per_ingestor[name] = len(got)
            raw_events.extend(got)
            if verbose:
                log.info("orchestrate: %s polled %d events", name, len(got))
        except Exception as exc:  # noqa: BLE001
            msg = f"ingestor:{name} crashed: {exc}"
            errors.append(msg)
            log.exception(msg)
            per_ingestor[name] = 0

    ingested_n = len(raw_events)
    if verbose:
        log.info("orchestrate: ingested=%d", ingested_n)

    # ── Step 2: classifier ───────────────────────────────────────────────
    classifier_mod = _import_classifier()
    classified: list[dict] = []
    if classifier_mod is None or not hasattr(classifier_mod, "classify"):
        errors.append("classifier import failed")
        log.warning("orchestrate: classifier unavailable — skipping")
    else:
        for ev in raw_events:
            try:
                c = classifier_mod.classify(ev)
            except Exception as exc:  # noqa: BLE001
                log.debug("classify failed for event: %s", exc)
                continue
            if c is not None:
                classified.append(c)

    if verbose:
        log.info("orchestrate: classified=%d", len(classified))

    # ── Step 3: situational mapping ──────────────────────────────────────
    sm_mod = _import_situational_map()
    mapped: list[dict] = []
    if sm_mod is None or not hasattr(sm_mod, "instruments_and_direction_for"):
        errors.append("situational_map import failed")
        log.warning("orchestrate: situational_map unavailable — skipping mapping")
        mapped = classified  # pass through unmapped
    else:
        for ev in classified:
            try:
                instruments, direction, base_mag = sm_mod.instruments_and_direction_for(
                    ev.get("event_type"), ev.get("entities") or {},
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("mapping failed for event %s: %s", ev.get("event_id"), exc)
                continue
            ev_out = dict(ev)
            ev_out["instruments"] = instruments
            ev_out["direction"] = direction
            ev_out["base_magnitude"] = base_mag
            mapped.append(ev_out)

    # ── Step 4: confidence scoring ───────────────────────────────────────
    cm_mod = _import_confidence_model()
    scored: list[dict] = []
    if cm_mod is None or not hasattr(cm_mod, "ConfidenceModel"):
        errors.append("confidence_model import failed")
        log.warning("orchestrate: confidence_model unavailable — skipping")
        scored = mapped
    else:
        try:
            model = cm_mod.ConfidenceModel()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ConfidenceModel init failed: {exc}")
            log.warning("orchestrate: ConfidenceModel init failed (%s)", exc)
            model = None
        if model is not None:
            for ev in mapped:
                try:
                    res = model.score(ev, corroboration_count=1)
                except Exception as exc:  # noqa: BLE001
                    log.debug("confidence score failed for %s: %s",
                              ev.get("event_id"), exc)
                    continue
                ev_out = dict(ev)
                ev_out["confidence"] = res.get("confidence")
                ev_out["magnitude"] = res.get("magnitude")
                ev_out["half_life_min"] = res.get("half_life_min")
                exp_at = res.get("expires_at")
                if hasattr(exp_at, "isoformat"):
                    exp_at = exp_at.isoformat()
                ev_out["expires_at"] = exp_at
                scored.append(ev_out)
        else:
            scored = mapped

    # ── Step 5: dedup + noise filter ─────────────────────────────────────
    dd_mod = _import_dedup()
    survivors: list[dict] = scored
    if dd_mod is None or not hasattr(dd_mod, "dedup_and_filter"):
        errors.append("dedup import failed")
        log.warning("orchestrate: dedup unavailable — skipping")
    else:
        try:
            survivors = dd_mod.dedup_and_filter(scored)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"dedup_and_filter crashed: {exc}")
            log.exception("orchestrate: dedup_and_filter crashed")
            survivors = scored

    dedupped_n = len(survivors)
    if verbose:
        log.info("orchestrate: dedupped=%d", dedupped_n)

    # ── Step 6: write (unless dry_run) ───────────────────────────────────
    written = 0
    sample = survivors[:3] if verbose else []
    if dry_run:
        log.info(
            "orchestrate DRY_RUN: ingested=%d classified=%d dedupped=%d (writer skipped)",
            ingested_n, len(classified), dedupped_n,
        )
        if verbose and sample:
            for sig in sample:
                log.info(
                    "  sample: event_type=%s conf=%s instruments=%s headline=%r",
                    sig.get("event_type"),
                    sig.get("confidence"),
                    sig.get("instruments"),
                    (sig.get("headline_normalized") or sig.get("headline") or "")[:80],
                )
    else:
        sw_mod = _import_signal_writer()
        if sw_mod is None or not hasattr(sw_mod, "SignalWriter"):
            errors.append("signal_writer import failed")
            log.warning("orchestrate: signal_writer unavailable — nothing written")
        else:
            try:
                writer = sw_mod.SignalWriter()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"SignalWriter init failed: {exc}")
                log.warning("orchestrate: SignalWriter init failed (%s)", exc)
                writer = None
            if writer is not None:
                for sig in survivors:
                    try:
                        res = writer.write(sig, force_shadow=False)
                    except Exception as exc:  # noqa: BLE001
                        log.debug("writer.write failed for %s: %s",
                                  sig.get("event_id"), exc)
                        continue
                    if res:
                        written += 1

    summary: dict = {
        "ingested": ingested_n,
        "classified": len(classified),
        "dedupped": dedupped_n,
        "written": written,
        "errors": errors,
        "per_ingestor": per_ingestor,
        "sample": sample,
    }
    log.info(
        "orchestrate done: ingested=%d classified=%d dedupped=%d written=%d errors=%d",
        ingested_n, len(classified), dedupped_n, written, len(errors),
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="JoSho Trader — situational signal worker")
    parser.add_argument("--once", action="store_true", help="Run a single poll cycle and exit.")
    parser.add_argument("--interval", type=int, default=15, help="Loop interval in minutes (default 15).")
    parser.add_argument("--orchestrate", action="store_true",
                        help="Run the new Layer-4 orchestrator (ingestors→classifier→"
                             "mapping→confidence→dedup→writer) instead of legacy pollers.")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --orchestrate: skip the writer step.")
    parser.add_argument("--verbose", action="store_true",
                        help="With --orchestrate: log per-step counts + sample signals.")
    args = parser.parse_args()

    log.info("signal_worker starting | vault=%s | interval=%dm",
             _VAULT_REPORT.get("vault_path"), args.interval)

    if args.orchestrate:
        summary = orchestrate_cycle(dry_run=args.dry_run, verbose=args.verbose)
        print(json.dumps({k: v for k, v in summary.items() if k != "sample"},
                         default=str, indent=2))
        if args.verbose and summary.get("sample"):
            print("\nSample signals:")
            for sig in summary["sample"]:
                print(json.dumps({
                    "event_id": sig.get("event_id"),
                    "event_type": sig.get("event_type"),
                    "confidence": sig.get("confidence"),
                    "magnitude": sig.get("magnitude"),
                    "instruments": sig.get("instruments"),
                    "headline": (sig.get("headline_normalized")
                                 or sig.get("headline") or "")[:120],
                }, default=str, indent=2))
        return 0

    if args.once:
        run_once()
        return 0

    while True:
        try:
            run_once()
        except KeyboardInterrupt:
            log.info("Interrupted — exiting.")
            return 0
        except Exception as exc:
            log.exception("Cycle crashed: %s", exc)
        sleep_s = max(60, args.interval * 60)
        log.info("Sleeping %ds until next cycle", sleep_s)
        time.sleep(sleep_s)


if __name__ == "__main__":
    sys.exit(main())

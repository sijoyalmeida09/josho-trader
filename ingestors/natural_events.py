"""natural_events.py -- Layer 4 ingestor #5 (NASA EONET v3 + USGS quakes).

Pure ingestion: polls EONET v3 (open natural events, last 7 days; categories
cyclones / severeStorms / wildfires / floods) AND USGS FDSN earthquakes
(M5.0+, last 30 days). Both filtered to a South-Asia bbox. Returns merged raw
event dicts per SITUATIONAL_SPEC.md sections 1, 3, 4, 10.

Does NOT classify (Agent 8), does NOT write to DB (Agent 13), does NOT import
from any sibling Layer-4 modules. Stdlib + requests only.

CLI:
    python ingestors/natural_events.py --dry-run --verbose
    python ingestors/natural_events.py --since=2026-05-18T00:00:00Z
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

UTC = timezone.utc
SOURCE_EONET = "eonet"
SOURCE_USGS = "usgs"
USER_AGENT = "JoShoTrader-NaturalEvents/1.0 (+contact: contact@example.com)"

EONET_ENDPOINT = "https://eonet.gsfc.nasa.gov/api/v3/events"
USGS_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query"

HTTP_TIMEOUT_S = 25
INTER_ENDPOINT_GAP_S = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF_BASE_S = 1.5

# EONET v3 category IDs are lowerCamelCase slugs.
EONET_CATEGORIES = ["cyclones", "severeStorms", "wildfires", "floods"]
EONET_DAYS_WINDOW = 7

# South-Asia bbox: India + buffer for PK/NP/LK/BD and the Himalayan seismic belt.
INDIA_BBOX = {"lat_min": 6.0, "lat_max": 37.0, "lon_min": 68.0, "lon_max": 97.0}

USGS_MIN_MAG = 5.0
USGS_DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_SINCE_HOURS = 24

log = logging.getLogger("ingestors.natural_events")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    )


def _parse_since(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            log.warning("Bad --since=%r; using default 24h window", value)
    return datetime.now(UTC) - timedelta(hours=DEFAULT_SINCE_HOURS)


def _coord_in_india(coord: Any) -> bool:
    """coord is [lon, lat] per GeoJSON. Tolerates malformed inputs."""
    if not isinstance(coord, (list, tuple)) or len(coord) < 2:
        return False
    try:
        lon, lat = float(coord[0]), float(coord[1])
    except (TypeError, ValueError):
        return False
    return (INDIA_BBOX["lat_min"] <= lat <= INDIA_BBOX["lat_max"]
            and INDIA_BBOX["lon_min"] <= lon <= INDIA_BBOX["lon_max"])


def _any_geom_in_india(geometry_list: list[dict[str, Any]]) -> bool:
    """EONET geometries may be Point, MultiPoint, or Polygon; walk leaves so
    any waypoint matching the bbox counts as in-region."""
    for g in geometry_list or []:
        coords = g.get("coordinates")
        if not isinstance(coords, list) or not coords:
            continue
        # Point: [lon, lat]
        if isinstance(coords[0], (int, float)):
            if _coord_in_india(coords):
                return True
            continue
        for sub in coords:
            if isinstance(sub, list) and sub and isinstance(sub[0], (int, float)):
                if _coord_in_india(sub):
                    return True
            elif isinstance(sub, list):
                for sub2 in sub:
                    if _coord_in_india(sub2):
                        return True
    return False


def _http_get_json(url: str, params: dict[str, Any], label: str,
                   verbose: bool = False) -> dict[str, Any] | None:
    """GET with bounded retries. Failsoft per spec section 12."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if verbose:
                prep = requests.Request("GET", url, params=params).prepare()
                log.info("%s GET %s", label, prep.url)
            resp = requests.get(url, params=params, headers=headers,
                                timeout=HTTP_TIMEOUT_S)
            if resp.status_code == 200:
                try:
                    return resp.json() or {}
                except ValueError:
                    log.warning("%s: non-JSON body (len=%d)", label, len(resp.content))
                    return None
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                log.warning("%s HTTP %d (attempt %d/%d)",
                            label, resp.status_code, attempt, MAX_RETRIES)
                time.sleep(RETRY_BACKOFF_BASE_S ** attempt)
                continue
            log.warning("%s HTTP %d body=%s", label, resp.status_code, resp.text[:160])
            return None
        except requests.RequestException as exc:
            log.warning("%s fetch error (attempt %d/%d): %s",
                        label, attempt, MAX_RETRIES, exc)
            time.sleep(RETRY_BACKOFF_BASE_S ** attempt)
    log.warning("%s gave up after %d attempts", label, MAX_RETRIES)
    return None


def _eonet_event_to_raw(ev: dict[str, Any]) -> dict[str, Any] | None:
    eonet_id = ev.get("id")
    title = (ev.get("title") or "").strip()
    if not eonet_id or not title:
        return None
    geometry = ev.get("geometry") or []
    if not _any_geom_in_india(geometry):
        return None

    sources = ev.get("sources") or []
    src_url = (sources[0].get("url") or "").strip() if (sources and isinstance(sources[0], dict)) else ""

    # Earliest geometry timestamp = event start.
    earliest: datetime | None = None
    magnitude_val: float | None = None
    magnitude_unit: str | None = None
    for g in geometry:
        ts = g.get("date")
        if ts:
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                if earliest is None or dt < earliest:
                    earliest = dt
            except ValueError:
                pass
        mv = g.get("magnitudeValue")
        if mv is not None:
            try:
                mv_f = float(mv)
                if magnitude_val is None or mv_f > magnitude_val:
                    magnitude_val = mv_f
                    magnitude_unit = g.get("magnitudeUnit")
            except (TypeError, ValueError):
                pass
    timestamp_iso = earliest.isoformat() if earliest else datetime.now(UTC).isoformat()

    cats = ev.get("categories") or []
    cat_ids = [c.get("id") for c in cats if isinstance(c, dict) and c.get("id")]
    primary_cat = cat_ids[0] if cat_ids else "unknown"

    entities_hint: dict[str, Any] = {
        "themes": ["natural_disaster", primary_cat],
        "locations": "IN-region",
    }
    if magnitude_val is not None:
        entities_hint["magnitude"] = magnitude_val
        if magnitude_unit:
            entities_hint["magnitude_unit"] = magnitude_unit

    return {
        "event_id": f"eonet:{eonet_id}",
        "source": SOURCE_EONET,
        "headline": title[:500],
        "url": src_url[:1000],
        "timestamp": timestamp_iso,
        "raw_payload": ev,
        "entities_hint": entities_hint,
    }


def _poll_eonet(verbose: bool = False) -> list[dict[str, Any]]:
    """One EONET call. Per spec: don't filter by 'since' here -- an event is
    'still open' even if started days ago, so we trust the 7-day window."""
    params = {"status": "open", "days": EONET_DAYS_WINDOW,
              "category": ",".join(EONET_CATEGORIES)}
    body = _http_get_json(EONET_ENDPOINT, params, label="EONET", verbose=verbose)
    if not body:
        return []
    events_in = body.get("events") or []
    if not isinstance(events_in, list):
        log.warning("EONET: 'events' is not a list")
        return []

    out: list[dict[str, Any]] = []
    cat_set = set(EONET_CATEGORIES)
    for ev in events_in:
        if not isinstance(ev, dict):
            continue
        cats = ev.get("categories") or []
        cat_ids = {c.get("id") for c in cats if isinstance(c, dict)}
        if not (cat_ids & cat_set):
            continue
        try:
            mapped = _eonet_event_to_raw(ev)
        except Exception as exc:  # noqa: BLE001
            log.warning("EONET: skipping malformed event: %s", exc)
            continue
        if mapped is not None:
            out.append(mapped)
    log.info("EONET poll: kept=%d / scanned=%d", len(out), len(events_in))
    return out


def _usgs_feature_to_raw(feat: dict[str, Any]) -> dict[str, Any] | None:
    feat_id = feat.get("id")
    if not feat_id:
        return None
    geom = feat.get("geometry") or {}
    if not _coord_in_india(geom.get("coordinates")):
        return None
    props = feat.get("properties") or {}
    try:
        mag_f = float(props.get("mag")) if props.get("mag") is not None else None
    except (TypeError, ValueError):
        mag_f = None
    if mag_f is None or mag_f < USGS_MIN_MAG:
        return None
    place = (props.get("place") or "").strip() or "Unknown location"
    detail_url = (props.get("url") or "").strip()

    try:
        timestamp_iso = datetime.fromtimestamp(
            int(props.get("time")) / 1000, tz=UTC).isoformat()
    except (TypeError, ValueError):
        timestamp_iso = datetime.now(UTC).isoformat()

    return {
        "event_id": f"usgs:{feat_id}",
        "source": SOURCE_USGS,
        "headline": f"M {mag_f:.1f} - {place}"[:500],
        "url": detail_url[:1000],
        "timestamp": timestamp_iso,
        "raw_payload": feat,
        "entities_hint": {"themes": ["earthquake"], "locations": place,
                          "magnitude": mag_f},
    }


def _poll_usgs(since: datetime, verbose: bool = False) -> list[dict[str, Any]]:
    """One USGS call with starttime + minmagnitude + bbox params."""
    starttime = since.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    params: dict[str, Any] = {
        "format": "geojson",
        "starttime": starttime,
        "minmagnitude": USGS_MIN_MAG,
        "minlatitude": INDIA_BBOX["lat_min"],
        "maxlatitude": INDIA_BBOX["lat_max"],
        "minlongitude": INDIA_BBOX["lon_min"],
        "maxlongitude": INDIA_BBOX["lon_max"],
        "orderby": "time",
    }
    body = _http_get_json(USGS_ENDPOINT, params, label="USGS", verbose=verbose)
    if not body:
        return []
    features = body.get("features") or []
    if not isinstance(features, list):
        log.warning("USGS: 'features' is not a list")
        return []
    out: list[dict[str, Any]] = []
    for feat in features:
        if not isinstance(feat, dict):
            continue
        try:
            mapped = _usgs_feature_to_raw(feat)
        except Exception as exc:  # noqa: BLE001
            log.warning("USGS: skipping malformed feature: %s", exc)
            continue
        if mapped is not None:
            out.append(mapped)
    log.info("USGS poll: kept=%d / scanned=%d starttime=%s",
             len(out), len(features), starttime)
    return out


# ---- Public API (frozen contract) -----------------------------------------

def poll(since: datetime | None = None) -> list[dict[str, Any]]:
    """Poll EONET + USGS once, merge, dedup-within-batch, return event dicts.

    Failsoft: a dead source returns []; the other still flows. A 0.5 s gap
    between endpoints respects free-tier courtesy.
    """
    window_start = since or (datetime.now(UTC) - timedelta(hours=DEFAULT_SINCE_HOURS))
    if window_start.tzinfo is None:
        window_start = window_start.replace(tzinfo=UTC)
    usgs_floor = datetime.now(UTC) - timedelta(days=USGS_DEFAULT_LOOKBACK_DAYS)
    usgs_since = max(window_start, usgs_floor)

    eonet_events = _poll_eonet()
    time.sleep(INTER_ENDPOINT_GAP_S)
    usgs_events = _poll_usgs(usgs_since)

    by_id: dict[str, dict[str, Any]] = {}
    for ev in eonet_events + usgs_events:
        eid = ev.get("event_id")
        if eid and eid not in by_id:
            by_id[eid] = ev
    out = list(by_id.values())
    log.info("natural_events poll: eonet=%d usgs=%d merged=%d window_start=%s",
             len(eonet_events), len(usgs_events), len(out), window_start.isoformat())
    return out


def poll_once_cli() -> int:
    """argparse entry. Exit 0 on success, 1 on top-level crash."""
    parser = argparse.ArgumentParser(
        prog="natural_events",
        description="Poll NASA EONET v3 + USGS quakes for Layer-4 events.",
    )
    parser.add_argument("--since", type=str, default=None,
                        help="ISO 8601 lower-bound timestamp (default: 24h ago).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print JSON lines to stdout; do not persist.")
    parser.add_argument("--verbose", action="store_true",
                        help="Log each request URL.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        log.setLevel(logging.DEBUG)

    since_dt = _parse_since(args.since)
    log.info("poll_once_cli starting | since=%s dry_run=%s verbose=%s",
             since_dt.isoformat(), args.dry_run, args.verbose)

    try:
        events = poll(since=since_dt)
    except Exception as exc:  # noqa: BLE001 -- top-level CLI guard
        log.exception("poll() crashed: %s", exc)
        return 1

    if not events:
        log.warning("No events returned (EONET + USGS both empty or down).")
        return 0

    if args.dry_run:
        # ensure_ascii=True so Windows cp1252 terminals can render USGS place
        # names with diacritics (Hindi, Burmese, etc) without crashing.
        for ev in events:
            sys.stdout.write(json.dumps(
                ev, ensure_ascii=True, sort_keys=True, default=str) + "\n")
        sys.stdout.flush()
    else:
        log.info("Non-dry-run no-op in pure ingestor; "
                 "Agent 13 writes DB. Returned %d events.", len(events))
    return 0


if __name__ == "__main__":
    sys.exit(poll_once_cli())

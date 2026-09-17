"""Nord Pool REMIT UMM collector (public API, no key): https://ummapi.nordpoolgroup.com/messages

Every message version is stored with its publication time, so features can use exactly the
outage information that was public at the decision time (a later version or a cancellation
only counts from its own publication time).

The API pages with `limit` / `skip` and returns newest messages first. Parsing is defensive:
field names follow the public API (productionUnits / generationUnits / consumptionUnits /
transmissionUnits -> timePeriods with eventStart, eventStop, unavailableCapacity,
availableCapacity).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .common import ensure_tables, http_get, utc_naive

log = logging.getLogger("nurex41id.umm")
URL = "https://ummapi.nordpoolgroup.com/messages"
EIC_SHORT = {
    "10YDK-1--------W": "DK1", "10YDK-2--------M": "DK2", "10Y1001A1001A82H": "DE_LU",
    "10YNO-2--------T": "NO2", "10Y1001A1001A46L": "SE3", "10Y1001A1001A47J": "SE4",
    "10YNL----------L": "NL", "10YGB----------A": "GB",
}
NAME_SHORT = {"DK1": "DK1", "DK2": "DK2", "NO2": "NO2", "SE3": "SE3", "SE4": "SE4", "DE": "DE_LU",
              "DE-LU": "DE_LU", "DE_LU": "DE_LU", "NL": "NL", "GB": "GB"}
UNIT_KINDS = {"productionUnits": "production", "generationUnits": "generation",
              "consumptionUnits": "consumption", "transmissionUnits": "transmission"}


def _area(d: dict) -> tuple[str, str]:
    eic = d.get("areaEic") or d.get("inAreaEic") or ""
    name = d.get("areaName") or d.get("inAreaName") or ""
    short = EIC_SHORT.get(eic) or NAME_SHORT.get(str(name).replace(" ", ""), str(name))
    # transmission units connect two areas: keep "IN>OUT"
    if "outAreaEic" in d or "outAreaName" in d:
        o_eic, o_name = d.get("outAreaEic", ""), d.get("outAreaName", "")
        o_short = EIC_SHORT.get(o_eic) or NAME_SHORT.get(str(o_name).replace(" ", ""), str(o_name))
        short = f"{short}>{o_short}"
        eic = f"{eic}>{o_eic}"
    return short, eic


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def parse_messages(items: list[dict]) -> pd.DataFrame:
    rows = []
    for it in items or []:
        base = {
            "message_id": str(it.get("messageId") or it.get("id") or ""),
            "version": int(it.get("version") or 1),
            "publication_utc": it.get("publicationDate") or it.get("publicationTime"),
            "message_type": str(it.get("messageType", "")),
            "unavailability_type": str(it.get("unavailabilityType", "")),
            "event_status": str(it.get("eventStatus", "")),
        }
        for key, kind in UNIT_KINDS.items():
            for u in it.get(key) or []:
                area, eic = _area(u)
                inst = _num(u.get("installedCapacity"))
                for tp in u.get("timePeriods") or []:
                    unav = _num(tp.get("unavailableCapacity"))
                    avail = _num(tp.get("availableCapacity"))
                    if np.isnan(unav) and not np.isnan(inst) and not np.isnan(avail):
                        unav = inst - avail
                    rows.append({**base, "asset_kind": kind,
                                 "asset_name": str(u.get("name") or u.get("unitName") or u.get("eic") or ""),
                                 "area": area, "area_eic": eic,
                                 "fuel": str(u.get("fuelType", "")), "installed_mw": inst,
                                 "available_mw": avail, "unavailable_mw": unav,
                                 "event_start": tp.get("eventStart"), "event_stop": tp.get("eventStop")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("publication_utc", "event_start", "event_stop"):
        df[c] = utc_naive(df[c])
    df = df.dropna(subset=["publication_utc", "event_start", "event_stop"])
    return df.drop_duplicates(["message_id", "version", "asset_kind", "asset_name", "event_start"], keep="last")


def collect(store, since=None, page: int = 500, max_pages: int = 2000, extra_params: dict | None = None,
            getter=None) -> int:
    """Page backwards until messages are older than `since` (publication time) or pages run out."""
    ensure_tables(store)
    since = pd.Timestamp(since) if since is not None else None
    get = getter or (lambda params: http_get(URL, params=params, headers={"Accept": "application/json"}).json())
    total, skip = 0, 0
    for _ in range(max_pages):
        params = {"limit": page, "skip": skip, **(extra_params or {})}
        data = get(params)
        items = data.get("items", data if isinstance(data, list) else [])
        if not items:
            break
        df = parse_messages(items)
        if len(df):
            total += store.upsert("umm_events", df)
        pubs = utc_naive(pd.Series([i.get("publicationDate") for i in items]).dropna())
        log.info("UMM skip=%d: %d messages, %d rows, oldest publication %s", skip, len(items), len(df),
                 pubs.min() if len(pubs) else "?")
        if since is not None and len(pubs) and pubs.max() < since:
            break
        if len(items) < page:
            break
        skip += page
    return total

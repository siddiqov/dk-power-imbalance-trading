"""Dashboard-only circuit breaker for external APIs that reject our credentials (2026-11-02).

When an API answers 401/403 (login rejected), every dashboard load used to wait on it again
(ENTSO-E and the old Nord Pool data-api client added minutes per load). After such an answer the API
is skipped for BLOCK_HOURS, shared by all dashboard sessions through one small JSON file; it is tried
again automatically afterwards, so a fixed key resumes by itself. Not used by the trading models.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

BLOCK_HOURS = 6.0
_FILE = Path(__file__).resolve().parents[1] / "data" / "dashboard_api_breaker.json"


def _read() -> dict:
    try:
        return json.loads(_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def blocked(name: str) -> bool:
    """True while `name` is paused after a rejected login."""
    return float((_read().get(name) or {}).get("until", 0)) > time.time()


def trip(name: str, reason: str = "", hours: float = BLOCK_HOURS) -> None:
    """Pause `name` for `hours` (called after a 401/403)."""
    try:
        d = _read()
        d[name] = {"until": time.time() + hours * 3600, "reason": str(reason)[:200],
                   "since": time.strftime("%Y-%m-%d %H:%M:%S")}
        tmp = _FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
        tmp.replace(_FILE)
    except Exception:
        pass


def is_auth_error(e) -> bool:
    s = str(e)
    return "401" in s or "403" in s or "Unauthorized" in s or "Forbidden" in s

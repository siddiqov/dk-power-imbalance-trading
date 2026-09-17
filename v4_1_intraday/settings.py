"""Settings for Nurex V4.1 Intraday: V4.2 config.yaml deep-merged with config_v41.yaml."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parent            # .../Basic_Approach/v4_1_intraday
BASE = PKG.parent                                  # .../Basic_Approach
V42_ROOT = BASE / "Nurex_V4_2"
if str(V42_ROOT) not in sys.path:
    sys.path.insert(0, str(V42_ROOT))

from nurex42.config import Settings, load_settings  # noqa: E402


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class IntradaySettings(Settings):
    @property
    def db_path(self) -> Path:
        p = Path(self.raw["db_path"])
        return p if p.is_absolute() else V42_ROOT / p

    @property
    def lead(self):
        import pandas as pd
        return pd.Timedelta(minutes=int(self.raw["intraday"]["gate_lead_minutes"]))

    def path(self, key: str) -> Path:
        p = Path(self.raw[key])
        return p if p.is_absolute() else BASE / p


def load(path: str | Path | None = None, db_path: str | None = None) -> IntradaySettings:
    base = load_settings(V42_ROOT / "config.yaml").raw
    with open(path or PKG / "config_v41.yaml", "r", encoding="utf-8") as f:
        over = yaml.safe_load(f) or {}
    raw = _merge(base, over)
    if db_path:
        raw["db_path"] = str(db_path)
    return IntradaySettings(raw)

"""Settings for Nurex V4.1 Intraday: V4.2 config.yaml deep-merged with config_v41.yaml.

Two decision schedules (config `intraday.mode`):
  gate   every quarter is decided on its own at delivery - gate_lead_minutes (original V4.1)
  batch  client schedule: quarters are grouped in local wall-clock blocks of
         `batch_schedule.quarters_per_batch` (4 = one hour). A batch is LOCKED at its deadline =
         first quarter - lead_minutes (2h15), using data up to deadline - final_run_minutes.
All time helpers below take/return tz-naive UTC timestamps.
"""
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

MODES = ("gate", "batch")


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
        """Gate-mode lead (delivery - gate_lead_minutes). Batch mode: use the helpers below."""
        import pandas as pd
        return pd.Timedelta(minutes=int(self.raw["intraday"]["gate_lead_minutes"]))

    # ------------------------------------------------------------------ schedule
    @property
    def mode(self) -> str:
        return str(self.raw["intraday"].get("mode", "gate")).lower()

    @property
    def batch_mode(self) -> bool:
        return self.mode == "batch"

    @property
    def bs(self) -> dict:
        return self.raw.get("batch_schedule") or {}

    @property
    def train_extra_leads(self) -> list[int]:
        src = self.bs if self.batch_mode else self.raw["intraday"]
        return [int(m) for m in (src.get("train_extra_leads_minutes") or [])]

    def _q(self, quarters):
        import pandas as pd
        return pd.to_datetime(pd.Series(quarters)).reset_index(drop=True)

    def batch_starts(self, quarters):
        """Start of the batch each quarter belongs to (gate mode: the quarter itself)."""
        q = self._q(quarters)
        if not self.batch_mode:
            return q
        from nurex42.timeutil import batch_start
        return batch_start(q, self["local_tz"], int(self.bs.get("quarters_per_batch", 4)))

    def deadlines(self, quarters):
        """Moment the decision must be final/submitted (gate mode: the gate closure)."""
        import pandas as pd
        q = self._q(quarters)
        if not self.batch_mode:
            return q - self.lead
        return self.batch_starts(q) - pd.Timedelta(minutes=int(self.bs.get("lead_minutes", 135)))

    def decision_times(self, quarters):
        """Information cut-off of the final decision (as_of). Always strictly before the deadline
        in batch mode; equal to the gate in gate mode."""
        import pandas as pd
        q = self._q(quarters)
        if not self.batch_mode:
            return q - self.lead
        return self.deadlines(q) - pd.Timedelta(minutes=int(self.bs.get("final_run_minutes", 15)))

    def schedule_text(self) -> str:
        if not self.batch_mode:
            return f"each quarter decided at delivery - {int(self.raw['intraday']['gate_lead_minutes'])} min"
        n, lead, fr = (int(self.bs.get("quarters_per_batch", 4)), int(self.bs.get("lead_minutes", 135)),
                       int(self.bs.get("final_run_minutes", 15)))
        return (f"batches of {n} quarters locked {lead // 60}h{lead % 60:02d} before the first quarter "
                f"(data cut-off {(lead + fr) // 60}h{(lead + fr) % 60:02d} before)")

    def path(self, key: str) -> Path:
        p = Path(self.raw[key])
        return p if p.is_absolute() else BASE / p


def load(path: str | Path | None = None, db_path: str | None = None, mode: str | None = None) -> IntradaySettings:
    """`mode` overrides config intraday.mode (e.g. train batch models while live still runs gate)."""
    base = load_settings(V42_ROOT / "config.yaml").raw
    with open(path or PKG / "config_v41.yaml", "r", encoding="utf-8") as f:
        over = yaml.safe_load(f) or {}
    raw = _merge(base, over)
    if db_path:
        raw["db_path"] = str(db_path)
    if mode:
        raw.setdefault("intraday", {})["mode"] = mode
    m = str(raw.get("intraday", {}).get("mode", "gate")).lower()
    if m not in MODES:
        raise ValueError(f"intraday.mode must be one of {MODES}, got {m!r}")
    if m == "batch":
        # batch mode keeps its own models, journal, reports -> separate track record
        for k in ("models_dir", "journal_path", "reports_dir"):
            if (raw.get("batch_schedule") or {}).get(k):
                raw[k] = raw["batch_schedule"][k]
    return IntradaySettings(raw)

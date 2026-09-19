"""Configuration loading (config.yaml + .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # optional
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    raw: dict

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def db_path(self) -> Path:
        p = Path(self.raw["db_path"])
        return p if p.is_absolute() else ROOT / p

    @property
    def collateral_eur(self) -> float:
        r = self.raw["risk"]
        return float(r["collateral_dkk"]) / float(r["dkk_per_eur"])

    @property
    def cost_per_mwh(self) -> float:
        c = self.raw["costs"]
        return float(c["nordpool_dayahead_eur_mwh"] + c["imbalance_fee_eur_mwh"]
                     + c["brp_service_eur_mwh"] + c["slippage_eur_mwh"])


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings(raw)


def secret(name: str) -> str | None:
    """Read a secret from the environment (.env). Returns None when not set."""
    val = os.getenv(name)
    return val if val else None

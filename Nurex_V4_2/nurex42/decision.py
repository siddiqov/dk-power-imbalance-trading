"""Turn spread forecasts into positions (action + MWh) under the client's risk limits.

PnL convention (working assumption until the BRP confirms settlement):
    BUY  x MWh: bought at the day-ahead price, settled at the imbalance price
                pnl = x * (imbalance - day_ahead) - x * cost
    SELL x MWh: pnl = x * (day_ahead - imbalance) - x * cost
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

BUY, SELL, HOLD = "BUY", "SELL", "HOLD"


def pnl(action: pd.Series, mwh: pd.Series, spread: pd.Series, cost_per_mwh: float):
    sign = np.where(action == BUY, 1.0, np.where(action == SELL, -1.0, 0.0))
    gross = sign * mwh * spread
    cost = np.where(sign != 0, mwh * cost_per_mwh, 0.0)
    return gross, cost, gross - cost


def raw_signal(pred: pd.DataFrame, cost: float, params: dict) -> pd.DataFrame:
    """params = {"buy": {"margin", "pmin"} | None, "sell": {...} | None}; a side set to None is disabled."""
    edge_buy = pred["exp_spread"] - cost
    edge_sell = -pred["exp_spread"] - cost
    buy = pd.Series(False, index=pred.index)
    sell = pd.Series(False, index=pred.index)
    if params.get("buy"):
        buy = (edge_buy > params["buy"]["margin"]) & (pred["p_up"] >= params["buy"]["pmin"])
    if params.get("sell"):
        sell = (edge_sell > params["sell"]["margin"]) & (pred["p_down"] >= params["sell"]["pmin"])
    buy &= ~sell
    action = np.where(buy, BUY, np.where(sell, SELL, HOLD))
    edge = np.where(buy, edge_buy, np.where(sell, edge_sell, 0.0))
    return pd.DataFrame({"action": action, "edge": edge}, index=pred.index)


def size_positions(sig: pd.DataFrame, batch_key: pd.Series, stress: dict, cfg) -> pd.Series:
    """MWh per quarter.

    1. Base size grows with edge relative to the stressed adverse move, capped at max_mwh.
    2. Within each client batch the total stressed loss is scaled down to
       batch_risk_fraction x collateral.
    """
    r = cfg["risk"]
    cost = cfg.cost_per_mwh
    max_mwh = float(r["max_mwh_per_quarter"])
    budget = float(r["batch_risk_fraction"]) * cfg.collateral_eur
    loss_buy = max(0.0, -stress["low"]) + cost      # EUR/MWh lost in a stressed quarter
    loss_sell = max(0.0, stress["high"]) + cost
    loss = np.where(sig["action"] == BUY, loss_buy, np.where(sig["action"] == SELL, loss_sell, 0.0))
    base = np.where(loss > 0, max_mwh * np.clip(sig["edge"] / np.maximum(loss, 1e-9) * 4.0, 0, 1), 0.0)
    df = pd.DataFrame({"base": base, "loss": loss, "b": batch_key.values}, index=sig.index)
    df["risk"] = df["base"] * df["loss"]
    tot = df.groupby("b")["risk"].transform("sum")
    scale = np.where(tot > budget, budget / tot.replace(0, np.nan), 1.0)
    mwh = df["base"] * np.nan_to_num(scale, nan=1.0)
    step = float(r["mwh_step"])
    mwh = np.floor(mwh / step) * step
    mwh = np.where(mwh + 1e-9 < float(r["min_mwh_per_quarter"]), 0.0, mwh)
    return pd.Series(np.round(mwh, 3), index=sig.index)


def decide(pred: pd.DataFrame, batch_key: pd.Series, stress: dict, cfg, params: dict,
           size_mult: pd.Series | float = 1.0) -> pd.DataFrame:
    if params.get("no_trade") or not (params.get("buy") or params.get("sell")):
        out = pd.DataFrame({"action": HOLD, "edge": 0.0, "mwh": 0.0}, index=pred.index)
        out["reason"] = "model has no validated edge"
        return out
    sig = raw_signal(pred, cfg.cost_per_mwh, params)
    sig["mwh"] = size_positions(sig, batch_key, stress, cfg)
    if not np.isscalar(size_mult) or size_mult != 1.0:
        step = float(cfg["risk"]["mwh_step"])
        m = np.floor(sig["mwh"] * size_mult / step) * step
        sig["mwh"] = np.where(m + 1e-9 < float(cfg["risk"]["min_mwh_per_quarter"]), 0.0, np.round(m, 3))
    sig.loc[sig["mwh"] <= 0, "action"] = HOLD
    sig.loc[sig["action"] == HOLD, ["mwh", "edge"]] = 0.0
    sig["reason"] = np.where(sig["action"] == HOLD, "edge below threshold",
                             "edge " + sig["edge"].round(1).astype(str) + " EUR/MWh")
    return sig


def tune(pred: pd.DataFrame, spread: pd.Series, batch_key: pd.Series, stress: dict, cfg,
         min_trades: int = 20) -> dict:
    """Pick margin / min-probability separately for BUY and SELL on a validation window by net PnL.

    A setting is only eligible if it is profitable in BOTH halves of the validation window and
    earns at least one extra unit of costs per MWh (guards against fitting noise). A side with
    no eligible setting is switched off."""
    out = {"buy": None, "sell": None, "val_net_eur": 0.0, "val_trades": 0}
    order = pd.Series(pd.to_datetime(batch_key).values, index=pred.index)
    first_half = (order <= order.quantile(0.5)).values
    for side in ("buy", "sell"):
        best, best_net, best_n = None, 0.0, 0
        for margin, pmin in itertools.product(cfg["decision"]["margin_grid"], cfg["decision"]["prob_grid"]):
            p = {side: {"margin": margin, "pmin": pmin}}
            d = decide(pred, batch_key, stress, cfg, p)
            n = int((d["action"] != HOLD).sum())
            if n < min_trades:
                continue
            _, _, net = pnl(d["action"], d["mwh"], spread, cfg.cost_per_mwh)
            net = np.asarray(net, dtype=float)
            tot = float(np.sum(net))
            mwh = float(d["mwh"].sum())
            stable = net[first_half].sum() > 0 and net[~first_half].sum() > 0
            if not stable or mwh <= 0 or tot / mwh < cfg.cost_per_mwh:
                continue
            if tot > best_net:
                best, best_net, best_n = {"margin": margin, "pmin": pmin}, tot, n
        out[side] = best
        out["val_net_eur"] = round(out["val_net_eur"] + best_net, 2)
        out["val_trades"] += best_n
    out["no_trade"] = not (out["buy"] or out["sell"])
    return out

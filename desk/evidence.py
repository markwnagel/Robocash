"""Live trading against the backtest: is the strategy that is running the one that was tested?

Backtest Sharpe barely predicts live results, but volatility and drawdown do (Wiecki et al. 2016, 888
Quantopian strategies). So the daily email compares exactly those two: live 20-day volatility against the
backtest's distribution of 20-day volatility, and live drawdown against the backtest's drawdowns.
Leaving the range is the earliest sign that live trading is not the strategy the backtest described.

reference_from_curve() runs when backtest.py tests the current config (no overrides) and is saved to
state/history/live_reference.json with the config it came from. live_vs_backtest() runs in the report.
"""
from __future__ import annotations

import math

import pandas as pd

from .report import inception

WINDOW = 20             # trading days per volatility estimate, live and backtest alike


def reference_from_curve(equity: pd.Series, window: int = WINDOW) -> dict:
    r = equity.pct_change().dropna()
    vol = (r.rolling(window).std() * math.sqrt(252)).dropna()
    dd = equity / equity.cummax() - 1
    return {"window_days": window, "n_days": int(len(r)),
            "vol_p05": float(vol.quantile(0.05)), "vol_p50": float(vol.quantile(0.50)),
            "vol_p95": float(vol.quantile(0.95)), "vol_full": float(r.std() * math.sqrt(252)),
            "max_dd": float(dd.min()), "dd_p05": float(dd.quantile(0.05))}


def live_vs_backtest(equity: pd.Series, ref: dict | None, window: int = WINDOW) -> dict:
    """Live figures since the account first traded, each with a status against the reference."""
    if equity.empty:
        return {"days": 0, "reference": ref}
    live = equity[equity.index >= pd.Timestamp(inception(equity))]
    r = live.pct_change().dropna()
    dd = live / live.cummax() - 1
    out = {"days": int(len(r)), "window_days": window, "reference": ref,
           "vol": float(r.tail(window).std() * math.sqrt(252)) if len(r) >= window else None,
           "dd_now": float(dd.iloc[-1]), "dd_max": float(dd.min())}
    if ref is None:
        out["vol_status"] = out["dd_status"] = "no backtest reference yet"
        return out
    if out["vol"] is None:
        out["vol_status"] = f"not enough data yet ({len(r)} of {window} trading days)"
    elif out["vol"] > ref["vol_p95"]:
        out["vol_status"] = "ABOVE backtest range"
    elif out["vol"] < ref["vol_p05"]:
        out["vol_status"] = "below backtest range"
    else:
        out["vol_status"] = "within backtest range"
    if out["dd_now"] <= ref["max_dd"]:
        out["dd_status"] = "BEYOND the backtest's worst"
    elif out["dd_now"] <= ref["dd_p05"]:
        out["dd_status"] = "in the backtest's worst 5% of days"
    else:
        out["dd_status"] = "within backtest range"
    out["flag"] = out["vol_status"].startswith("ABOVE") or out["dd_status"].startswith("BEYOND")
    return out

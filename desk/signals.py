from __future__ import annotations

import numpy as np
import pandas as pd


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int) -> pd.DataFrame:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()]).groupby(level=0).max()
    return tr.rolling(n).mean()


def compute_panels(panel: dict, scfg: dict) -> dict:
    """Backward-looking time series for every symbol; row t uses data through t only."""
    close, high, low = panel["close"], panel["high"], panel["low"]
    lb, skip = scfg["momentum_lookback_days"], scfg["momentum_skip_days"]
    ret = close.pct_change()
    return {
        "close": close,
        "momentum": close.shift(skip) / close.shift(lb) - 1,
        "short": close / close.shift(scfg["short_lookback_days"]) - 1,
        "vol": ret.rolling(scfg["vol_days"]).std() * np.sqrt(252),
        "atr": atr(high, low, close, scfg["atr_days"]),
        "ma": close.rolling(scfg["trend_ma_days"]).mean(),
        "returns": ret,
    }


def snapshot(panels: dict, i: int, scfg: dict) -> pd.DataFrame:
    """Cross-sectional table at row i: score and rank (1 = best) among eligible names."""
    snap = pd.DataFrame({
        "close": panels["close"].iloc[i],
        "momentum": panels["momentum"].iloc[i],
        "short": panels["short"].iloc[i],
        "vol": panels["vol"].iloc[i],
        "atr": panels["atr"].iloc[i],
        "ma": panels["ma"].iloc[i],
    }).dropna()
    snap["atr_pct"] = snap["atr"] / snap["close"]
    snap["eligible"] = (
        (snap["close"] > snap["ma"])
        & (snap["momentum"] > 0)
        & (snap["atr_pct"] <= scfg["max_atr_pct"])
    )
    w = scfg["weights"]
    el = snap[snap["eligible"]]
    score = (
        w["momentum"] * el["momentum"].rank(pct=True)
        + w["short"] * el["short"].rank(pct=True)
        + w["low_vol"] * (-el["vol"]).rank(pct=True)
    )
    snap["score"] = score
    snap["rank"] = score.rank(ascending=False, method="first")
    return snap.sort_values("rank")


def rerank(snap: pd.DataFrame, tilt: dict | None = None, blocked: dict | None = None) -> pd.DataFrame:
    """Apply score tilts (backtested features only) and hard blocks, then recompute ranks."""
    snap = snap.copy()
    if blocked:
        snap.loc[snap.index.isin(list(blocked)), "eligible"] = False
    if tilt:
        for sym, t in tilt.items():
            if sym in snap.index and snap.loc[sym, "eligible"]:
                snap.loc[sym, "score"] += t
    snap.loc[~snap["eligible"], ["score", "rank"]] = np.nan
    el = snap["eligible"]
    snap.loc[el, "rank"] = snap.loc[el, "score"].rank(ascending=False, method="first")
    return snap.sort_values("rank")


def compute_signals(panel: dict, scfg: dict) -> pd.DataFrame:
    panels = compute_panels(panel, scfg)
    return snapshot(panels, -1, scfg)

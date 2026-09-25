from __future__ import annotations

import numpy as np
import pandas as pd


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int) -> pd.DataFrame:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()]).groupby(level=0).max()
    return tr.rolling(n).mean()


# The only signal settings compute_panels reads. The rest (weights, max_atr_pct) apply per snapshot, so a
# cache of computed panels keys on these alone - keying on the whole section recomputed (and held in memory)
# a full panel set for every weight or ATR-filter variant.
PANEL_KEYS = ("momentum_lookback_days", "momentum_skip_days", "short_lookback_days", "trend_ma_days",
              "atr_days", "vol_days")


def panel_key(scfg: dict) -> tuple:
    return tuple(scfg[k] for k in PANEL_KEYS)


def compute_panels(panel: dict, scfg: dict) -> dict:
    """Backward-looking time series for every symbol; row t uses data through t only."""
    close, high, low = panel["close"], panel["high"], panel["low"]
    lb, skip = scfg["momentum_lookback_days"], scfg["momentum_skip_days"]
    # A missing bar carries the last close forward (a 0% day), spelled out rather than left to pct_change's
    # deprecated default fill so a pandas upgrade cannot change every result.
    ret = close.ffill().pct_change(fill_method=None)
    return {
        "close": close,
        "momentum": close.shift(skip) / close.shift(lb) - 1,
        "short": close / close.shift(scfg["short_lookback_days"]) - 1,
        "vol": ret.rolling(scfg["vol_days"]).std() * np.sqrt(252),
        "atr": atr(high, low, close, scfg["atr_days"]),
        "ma": close.rolling(scfg["trend_ma_days"]).mean(),
        "returns": ret,
    }


SNAP_FIELDS = ("close", "momentum", "short", "vol", "atr", "ma")


def _rows(panels: dict, i: int, cols=None) -> tuple:
    """(index, {field: array}) for row i: every symbol, or `cols` in that order (NaN where the panel has no
    such column), keeping only symbols with every field present - what building the frame from reindexed
    Series and calling dropna() gave, read straight from the arrays. This runs once per backtest day."""
    columns = panels["close"].columns
    idx = columns if cols is None else pd.Index(cols)
    take = None if cols is None else columns.get_indexer(idx)
    data = {}
    for k in SNAP_FIELDS:
        row = panels[k].to_numpy()[i]
        if take is not None:
            row = np.where(take >= 0, row[take], np.nan)
        data[k] = row
    keep = ~np.isnan(np.column_stack([data[k] for k in SNAP_FIELDS])).any(axis=1)
    return idx[keep], {k: v[keep] for k, v in data.items()}


def snapshot(panels: dict, i: int, scfg: dict, cols=None) -> pd.DataFrame:
    """Cross-sectional table at row i: score and rank (1 = best) among eligible names. cols limits the
    cross-section to those symbols (a point-in-time universe plus holdings): ranks are percentiles, so
    they depend on who else is in the table, as they do live."""
    idx, d = _rows(panels, i, cols)
    atr_pct = d["atr"] / d["close"]
    eligible = (d["close"] > d["ma"]) & (d["momentum"] > 0) & (atr_pct <= scfg["max_atr_pct"])
    w, el = scfg["weights"], idx[eligible]
    pct = lambda values: pd.Series(values, index=el).rank(pct=True)  # noqa: E731
    score_el = (w["momentum"] * pct(d["momentum"][eligible]) + w["short"] * pct(d["short"][eligible])
                + w["low_vol"] * pct(-d["vol"][eligible]))
    score, rank = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
    score[eligible] = score_el.to_numpy()
    rank[eligible] = score_el.rank(ascending=False, method="first").to_numpy()
    # Built in one go: adding the derived columns one by one cost more than everything else in here.
    snap = pd.DataFrame({**d, "atr_pct": atr_pct, "eligible": eligible, "score": score, "rank": rank}, index=idx)
    return snap.sort_values("rank")


def rerank(snap: pd.DataFrame, tilt: dict | None = None, blocked: dict | None = None) -> pd.DataFrame:
    """Apply score tilts (backtested features only) and hard blocks, then recompute ranks. snap is ranked and
    sorted as snapshot() leaves it; when no block or tilt touches an eligible name the ranks cannot change,
    so the copy is returned as is (most backtest days - it was a sixth of the run time)."""
    el0 = snap["eligible"].to_numpy(dtype=bool)
    touches = bool(blocked) and bool(snap.index[el0].isin(list(blocked)).any())
    touches = touches or any(t and sym in snap.index and bool(snap.loc[sym, "eligible"]) for sym, t in (tilt or {}).items())
    snap = snap.copy()
    if not touches:
        return snap
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

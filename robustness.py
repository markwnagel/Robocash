#!/usr/bin/env python
"""Robustness study: config.yaml as it runs today, on 8.7 years of point-in-time data, stress-tested.

  python scripts/build_pit_universe.py      # once: the point-in-time universe and its prices (resumable)
  python robustness.py                      # every section, ~2 minutes on 20 workers
  python robustness.py --sections core,costs --workers 8

backtest.py answers "what did this config do on today's 800 names over three years?". This answers what
that number cannot:
  survivorship   today's 800 names against the names that were liquid at the time, delisted ones included
  out of sample  2018-01 to 2023-09 was never used to choose a parameter
  stress         2018 Q4, the 2020 crash, the 2020 momentum rotation, the 2022 bear, SVB, the 2025 tariffs
  costs          10 / 25 / 50 bps per side
  design         the regime gate, kill switch, earnings block and cash-below-the-MA rule, re-checked on
                 years that contain bear markets
  null           the same machinery buying at random among eligible names: is the ranking the edge?
  frictions      the analyst's observed REMOVE / REDUCE / half-size rates applied at random
  sensitivity    each key parameter one notch each way: a plateau or a spike?
  windows        every 3-year window starting each quarter
  bootstrap      block-resampled 3-year paths: the odds of each drawdown switch firing

This study's own runs do not go into state/history/trials.jsonl (they test the chosen config); the variant counts
are in backtest.prior_untracked_trials. The scripts/check_*.py studies, which test alternatives, log each variant
with log_variant_trials(). excess_vs_spy() is the bar a proposal is judged on.
Output: state/robustness/<date>/results.json, report.md, curves.pkl
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B
from desk import pit
from desk import trials as T
from desk.config import load_config
from desk.signals import compute_panels, panel_key, snapshot

log = logging.getLogger("desk.robustness")

SECTIONS = ("core", "costs", "design", "null", "frictions", "sensitivity", "windows")

# One notch each way around config.yaml. Chosen before any result was seen, at roughly +-33% or the
# nearest customary value, so the grid itself is not fitted.
SENSITIVITY = [
    ("momentum lookback (days)", ("signals", "momentum_lookback_days"), 84, 168),
    ("momentum skip (days)", ("signals", "momentum_skip_days"), 10, 42),
    ("trend MA (days)", ("signals", "trend_ma_days"), 30, 100),
    ("max ATR % of price", ("signals", "max_atr_pct"), 0.06, 0.10),
    ("stop (x ATR)", ("exits", "stop_atr_multiple"), 1.5, 2.5),
    ("target (R)", ("exits", "take_profit_r_multiple"), 2.0, 4.0),
    ("min hold (days)", ("exits", "min_hold_days"), 10, 20),
    ("max hold (days)", ("exits", "max_hold_days"), 40, 90),
    ("decay rank", ("exits", "signal_decay_rank"), 25, 60),
    ("risk per trade", ("portfolio", "risk_per_trade_pct"), 0.0075, 0.0125),
    ("target positions", ("portfolio", "target_positions"), 9, 15),
    ("kill switch drawdown", ("risk", "max_drawdown_pct"), 0.12, 0.20),
]

DESIGN = [
    ("no regime gate", [(("regime", "enabled"), False)]),
    ("cash below SPY 200-day MA", [(("regime", "off_liquidate"), True)]),
    ("no kill switches", [(("risk", "max_drawdown_pct"), 1.0), (("risk", "max_cumulative_drawdown_pct"), None)]),
    ("no earnings block", [(("facts", "exclude_earnings_within_days"), 0)]),
]

# Live entries staged 2026-09-14 to 09-24 (21 of them): 12 marked REDUCE by the analyst, 5 halved by the
# implied-vol rule (2 both). Entry REMOVEs leave no trace in the brief; 10% is an assumption. A small
# sample: this estimates the drag of the veto layer if its calls carry no information either way.
FRICTIONS = {"remove": 0.10, "reduce": 12 / 21, "factor": 0.75, "half": 5 / 21}

STRESS = [
    ("2018 Q4 sell-off", "2018-09-20", "2018-12-24"),
    ("COVID crash", "2020-02-19", "2020-03-23"),
    ("COVID rebound", "2020-03-23", "2020-06-08"),
    ("Vaccine-day momentum reversal", "2020-11-06", "2020-12-04"),
    ("Growth peak to 2022 low", "2021-11-19", "2022-10-12"),
    ("2022 bear market", "2022-01-03", "2022-10-12"),
    ("SVB bank failures", "2023-03-08", "2023-03-24"),
    ("2023 autumn correction", "2023-07-31", "2023-10-27"),
    ("2024 yen carry unwind", "2024-07-16", "2024-08-05"),
    ("2025 tariff shock", "2025-02-19", "2025-04-08"),
]

_G: dict = {}
PANEL_SETS = 2     # computed signal panels a worker keeps: config.yaml's plus the variant it is running


def _set(cfg: dict, path: tuple, value) -> None:
    node = cfg
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value


def _init(inputs: str, panel_path: str) -> None:
    """Worker start: the panel and inputs are read once per process, not shipped with every job."""
    _G.update(pd.read_pickle(inputs))
    _G["panel"] = pd.read_pickle(panel_path)
    _G["signal_panels"] = OrderedDict()
    logging.getLogger().setLevel(logging.WARNING)


def run_job(job: dict) -> dict:
    t0 = time.time()
    cfg = copy.deepcopy(_G["cfg"])
    for path, value in job.get("overrides", []):
        _set(cfg, path, value)
    reg, reg_state = B.regime_inputs(_G["spy"], _G["vix"], cfg)
    # Keyed on what compute_panels reads, and bounded: a worker used to keep one full panel set for every
    # signal variant it had run, including weight and ATR-filter variants that need no new panels at all.
    key, cache = panel_key(cfg["signals"]), _G["signal_panels"]
    if key in cache:
        cache.move_to_end(key)
    else:
        cache[key] = compute_panels(_G["panel"], cfg["signals"])
        while len(cache) > PANEL_SETS:
            cache.popitem(last=False)
    rng = np.random.default_rng(job["seed"]) if job.get("seed") is not None else None
    curve, trades = B.run_backtest(
        _G["panel"], cfg, pd.Timestamp(job["start"]), pd.Timestamp(job["end"]), _G["capital"],
        job.get("slippage", 10), 20, regime=reg, earnings=_G["earnings"], regime_state=reg_state,
        panels=cache[key], members=_G["members"][job.get("universe", "pit")],
        entry_rng=rng if job.get("mode") == "random" else None,
        veto={**FRICTIONS, **job.get("frictions", {}), "rng": rng} if job.get("mode") == "frictions" else None,
        offerings=_G.get("offerings") if job.get("offerings") else None,
        sweep_px=(_G.get("sweeps") or {}).get(job.get("sweep_with", "SPY"), _G.get("spy")))
    return {"name": job["name"], "group": job["group"], "job": {k: v for k, v in job.items() if k != "overrides"},
            "overrides": [[".".join(p), v] for p, v in job.get("overrides", [])],
            "equity": curve["equity"], "n_positions": curve["n_positions"], "cash": curve["cash"],
            "trades": trades if job.get("keep_trades") else None,
            "n_trades": len(trades), "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ statistics (pure; tested)

def full_stats(eq: pd.Series) -> dict:
    eq = eq.dropna()
    r = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    dd = eq / eq.cummax() - 1
    down = math.sqrt((r.clip(upper=0) ** 2).mean()) * math.sqrt(252) if len(r) else float("nan")
    under = dd < 0
    runs = (under != under.shift()).cumsum()
    spells = [(g.index[-1] - g.index[0]).days for _, g in dd[under].groupby(runs[under])] if under.any() else [0]
    monthly = eq.groupby(eq.index.to_period("M")).last().pct_change().dropna()
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    return {"total": total, "cagr": cagr, "maxdd": float(dd.min()), "vol": float(r.std() * math.sqrt(252)),
            "sharpe": float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else float("nan"),
            "sortino": float(r.mean() * 252 / down) if down > 0 else float("nan"),
            "calmar": cagr / abs(dd.min()) if dd.min() < 0 else float("nan"),
            "longest_underwater_days": int(max(spells)), "worst_month": float(monthly.min()) if len(monthly) else None,
            "best_month": float(monthly.max()) if len(monthly) else None,
            "pct_months_up": float((monthly > 0).mean()) if len(monthly) else None, "years": years}


def excess_vs_spy(eq: pd.Series, spy: pd.Series, horizon: int = 126) -> dict:
    """The strategy against buy-and-hold SPY over the same dates: CAGR difference, and over every `horizon`-day
    window (126 = six months) the median and 10th-percentile excess return and how often it beat SPY. The bar
    proposals are judged on (engineering review 2026-10-17): the desk's job is to beat SPY, and an absolute-
    return bar passed changes that only added exposure (reduce_factor 1.0: +1.8%/yr, drawdown 3pt deeper)."""
    eq = eq.dropna()
    s = spy.reindex(eq.index).ffill().dropna()
    eq = eq.reindex(s.index)
    if len(eq) < horizon + 2:
        return {"cagr_excess": None, "excess_6m_median": None, "excess_6m_p10": None, "beat_6m": None}
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = lambda x: (x.iloc[-1] / x.iloc[0]) ** (1 / years) - 1  # noqa: E731
    ex = (eq.pct_change(horizon) - s.pct_change(horizon)).dropna()
    return {"cagr_excess": float(cagr(eq) - cagr(s)), "excess_6m_median": float(ex.median()),
            "excess_6m_p10": float(ex.quantile(0.1)), "beat_6m": float((ex > 0).mean())}


def log_variant_trials(path, jobs: list, results: dict, tag: str) -> int:
    """Append one trial per variant per window (its first start) to the trial log, so every config tested by a
    check script counts toward the Deflated Sharpe's N. They were never logged: by 2026-10-17, 63 variants had been
    tried outside the log, against 20 in it (engineering review). Returns how many were written."""
    seen, n = set(), 0
    for j in jobs:
        r = results.get(j["name"])
        if r is None or j.get("mode") in ("random", "frictions"):
            continue
        ov = {".".join(p): v for p, v in j.get("overrides", [])}
        key = (json.dumps(ov, sort_keys=True, default=str), j.get("universe", "pit"), j["group"])
        if key in seen:
            continue
        seen.add(key)
        st = full_stats(r["equity"])
        T.record(Path(path), {"config_id": T.config_id({"overrides": ov, "universe": j.get("universe", "pit")}),
                              "overrides": ov, "tag": tag, "start": j["start"], "end": j["end"],
                              "universe_id": j.get("universe", "pit"), "stats": T.return_stats(r["equity"]),
                              "total": st["total"], "maxdd": st["maxdd"], "sharpe_annual": st["sharpe"]})
        n += 1
    return n


def period_return(eq: pd.Series, start: str, end: str) -> float | None:
    """Return from the last close on or before `start` to the last close on or before `end`."""
    a, b = eq[:start], eq[:end]
    if a.empty or b.empty or b.index[-1] <= a.index[-1]:
        return None
    return float(b.iloc[-1] / a.iloc[-1] - 1)


def period_drawdown(eq: pd.Series, start: str, end: str) -> float | None:
    seg = eq[eq.index >= eq[:start].index[-1]] if not eq[:start].empty else eq[start:]
    seg = seg[:end]
    return float((seg / seg.cummax() - 1).min()) if len(seg) > 1 else None


def yearly(eq: pd.Series) -> dict:
    y = eq.groupby(eq.index.year).last()
    first = eq.groupby(eq.index.year).first()
    out = {}
    for yr in y.index:
        prev = y.get(yr - 1)
        base = prev if prev is not None and not np.isnan(prev) else first[yr]
        out[int(yr)] = float(y[yr] / base - 1)
    return out


def ew_universe(close: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """Equal-weight buy of every member, rebalanced daily, no costs: what owning the universe earned."""
    r = close.pct_change(fill_method=None)
    m = members.reindex(index=r.index, columns=r.columns, fill_value=False).shift(1, fill_value=False)
    daily = r.where(m).mean(axis=1).fillna(0.0)
    return (1 + daily).cumprod()


def plain_topn(panels: dict, members: pd.DataFrame, scfg: dict, n: int, start: pd.Timestamp,
               end: pd.Timestamp, cost_bps: float = 10) -> pd.Series:
    """The same score, none of the trade management: each month-end buy the top n eligible members in equal
    weight and hold a month. Separates what the ranking earns from what the stops, targets, sizing and
    switches add or cost."""
    close = panels["close"]
    r = close.pct_change(fill_method=None).fillna(0.0)
    dates = close.index[(close.index >= start) & (close.index <= end)]
    month_ends = set(pd.Series(dates, index=dates).groupby(dates.to_period("M")).max())
    mem = members.reindex(index=close.index, columns=close.columns, fill_value=False)
    held, eq, out, last = [], 1.0, [], None
    ix = {d: i for i, d in enumerate(close.index)}
    for d in dates:
        if held:
            eq *= 1 + float(r.loc[d, held].mean())
        out.append((d, eq))
        if d in month_ends:
            cols = close.columns[mem.loc[d].to_numpy()]
            snap = snapshot(panels, ix[d], scfg, cols)
            new = list(snap.index[snap["eligible"].values][:n])
            turnover = len(set(new) ^ set(held)) / max(n, 1) if held else 1.0
            eq *= 1 - turnover * cost_bps / 10000
            held, last = new, d
    return pd.Series(dict(out))


def stationary_bootstrap(r: np.ndarray, bench: np.ndarray, horizon: int, n_paths: int, block: float,
                         seed: int = 7) -> dict:
    """Politis-Romano stationary bootstrap of paired daily returns (strategy, SPY): blocks of geometric length
    (mean `block`) keep volatility clustering and the kill switch's cash spells intact."""
    rng = np.random.default_rng(seed)
    n = len(r)
    tot, dd, spy_tot = np.empty(n_paths), np.empty(n_paths), np.empty(n_paths)
    p = 1.0 / block
    for k in range(n_paths):
        idx = np.empty(horizon, dtype=int)
        idx[0] = rng.integers(n)
        jump = rng.random(horizon) < p
        fresh = rng.integers(n, size=horizon)
        for t in range(1, horizon):
            idx[t] = fresh[t] if jump[t] else (idx[t - 1] + 1) % n
        path = np.cumprod(1 + r[idx])
        tot[k] = path[-1] - 1
        dd[k] = (path / np.maximum.accumulate(path) - 1).min()
        spy_tot[k] = np.prod(1 + bench[idx]) - 1
    q = lambda a: {f"p{p_}": float(np.percentile(a, p_)) for p_ in (5, 25, 50, 75, 95)}  # noqa: E731
    return {"horizon_days": horizon, "paths": n_paths, "mean_block_days": block,
            "total": q(tot), "maxdd": q(dd), "p_loss": float((tot < 0).mean()),
            "p_dd_15": float((dd <= -0.15).mean()), "p_dd_25": float((dd <= -0.25).mean()),
            "p_beat_spy": float((tot > spy_tot).mean())}


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {}
    t = trades.copy()
    t["r"] = t["pnl"] / t["realized_risk"].where(t["realized_risk"] > 0)
    wins, losses = t[t["pnl"] > 0], t[t["pnl"] <= 0]
    net = t["pnl"].sum()
    top = t["pnl"].sort_values(ascending=False)
    by_reason = (t.groupby("reason").agg(n=("pnl", "size"), win_rate=("pnl", lambda s: (s > 0).mean()),
                                         avg_pct=("pnl_pct", "mean"), pnl=("pnl", "sum"))
                 .sort_values("n", ascending=False))
    t["exit_year"] = pd.to_datetime(t["exit_date"]).dt.year
    by_year = t.groupby("exit_year").agg(n=("pnl", "size"), win_rate=("pnl", lambda s: (s > 0).mean()),
                                         pnl=("pnl", "sum"))
    return {"n": len(t), "win_rate": float((t["pnl"] > 0).mean()),
            "avg_win_pct": float(wins["pnl_pct"].mean()) if len(wins) else None,
            "avg_loss_pct": float(losses["pnl_pct"].mean()) if len(losses) else None,
            "payoff": float(wins["pnl"].mean() / -losses["pnl"].mean()) if len(wins) and len(losses) and losses["pnl"].mean() < 0 else None,
            "profit_factor": float(wins["pnl"].sum() / -losses["pnl"].sum()) if losses["pnl"].sum() < 0 else None,
            "expectancy_r": float(t["r"].mean()), "median_r": float(t["r"].median()),
            "median_hold_days": float(t["hold_days"].median()), "net_pnl": float(net),
            "top10_share": float(top.head(10).sum() / net) if net > 0 else None,
            "net_without_top10": float(net - top.head(10).sum()),
            "top5pct_share": float(top.head(max(1, len(t) // 20)).sum() / net) if net > 0 else None,
            "by_reason": {k: {c: (float(v) if c != "n" else int(v)) for c, v in row.items()}
                          for k, row in by_reason.to_dict("index").items()},
            "by_year": {int(k): {c: (float(v) if c != "n" else int(v)) for c, v in row.items()}
                        for k, row in by_year.to_dict("index").items()}}


def switch_events(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"kill-switch": [], "hard-floor": []}
    return {k: sorted({str(d) for d in trades.loc[trades["reason"] == k, "exit_date"]}) for k in ("kill-switch", "hard-floor")}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (pd.Timestamp, date)):
        return str(o)[:10]
    return o


def variant_summary(runs: dict) -> dict:
    """{variant: medians over its start dates + paired wins against config.yaml from the same start}. runs is
    S["runs"]; staggered names look like "sens_min_hold_days_high@2", the reference is "base@2"."""
    groups: dict = {}
    for n, r in runs.items():
        if "@" in n:
            groups.setdefault(n.split("@")[0], {})[int(n.split("@")[1])] = r
    base = groups.get("base", {})
    out = {}
    for v, by in groups.items():
        rows = list(by.values())
        med = {m: float(np.median([r[m] for r in rows])) for m in ("cagr", "sharpe", "maxdd", "total")}
        paired = [(by[k], base[k]) for k in by if k in base]
        out[v] = {"label": rows[0].get("label"), "group": rows[0]["group"], "side": rows[0].get("side"),
                  "value": rows[0].get("value"), "n": len(rows), **med,
                  "cagr_by_start": [by[k]["cagr"] for k in sorted(by)],
                  "beats_base_cagr": sum(a["cagr"] > b["cagr"] for a, b in paired),
                  "beats_base_dd": sum(a["maxdd"] > b["maxdd"] for a, b in paired), "paired": len(paired)}
    return out


# ------------------------------------------------------------------ the study

def build_jobs(sections: set, start: str, end: str, recent: str, seeds: int, veto_seeds: int, n_starts: int = 4) -> list:
    """Variants (costs, design, sensitivity) run from n_starts start dates a quarter apart, next to config.yaml
    from the same dates, and are judged on the median and on how many starts they win. Removing seven notes
    that traded twice for $3 moved one 8.7-year path from +72% to +85%: a single path is mostly noise."""
    J = []
    full = {"start": start, "end": end}
    starts = [str((pd.Timestamp(start) + pd.DateOffset(months=3 * k)).date()) for k in range(n_starts)]

    def staggered(name: str, group: str, **kw) -> list:
        return [{"name": f"{name}@{k}", "group": group, "start": st, "end": end, "offset": k, **kw}
                for k, st in enumerate(starts)]

    if sections & {"costs", "design", "sensitivity"}:
        J += staggered("base", "base", label="config.yaml")
    if "core" in sections:
        J += [{"name": "pit_full", "group": "core", **full, "universe": "pit", "keep_trades": True},
              {"name": "frozen_full", "group": "core", **full, "universe": "frozen", "keep_trades": True},
              {"name": "pit_oos", "group": "core", "start": start, "end": str((pd.Timestamp(recent) - pd.Timedelta(days=1)).date()),
               "universe": "pit", "keep_trades": True},
              {"name": "pit_recent", "group": "core", "start": recent, "end": end, "universe": "pit", "keep_trades": True},
              {"name": "frozen_recent", "group": "core", "start": recent, "end": end, "universe": "frozen", "keep_trades": True}]
        noearn = [(("facts", "exclude_earnings_within_days"), 0)]
        for n, u, s in (("pit_full_noearn", "pit", start), ("frozen_full_noearn", "frozen", start),
                        ("pit_recent_noearn", "pit", recent), ("frozen_recent_noearn", "frozen", recent)):
            J.append({"name": n, "group": "core", "start": s, "end": end, "universe": u, "overrides": noearn})
    if "costs" in sections:
        for b in (25, 50):
            J += staggered(f"slip_{b}", "costs", label=f"Slippage {b} bps per side", slippage=b)
    if "design" in sections:
        for i, (lbl, ov) in enumerate(DESIGN):
            J += staggered(f"design_{i}", "design", label=lbl, overrides=ov)
    if "null" in sections:
        J += [{"name": f"null_{k}", "group": "null", **full, "mode": "random", "seed": 1000 + k} for k in range(seeds)]
    if "frictions" in sections:
        J += [{"name": f"frictions_{k}", "group": "frictions", **full, "mode": "frictions", "seed": 2000 + k}
              for k in range(veto_seeds)]
    if "sensitivity" in sections:
        for lbl, path, lo, hi in SENSITIVITY:
            for side, v in (("low", lo), ("high", hi)):
                J += staggered(f"sens_{path[-1]}_{side}", "sensitivity", label=lbl, side=side, value=v,
                               overrides=[(path, v)])
    if "windows" in sections:
        s0, e = pd.Timestamp(start), pd.Timestamp(end)
        while s0 + pd.DateOffset(years=3) <= e + pd.Timedelta(days=5):
            J.append({"name": f"win_{s0.date()}", "group": "windows", "start": str(s0.date()),
                      "end": str(min(s0 + pd.DateOffset(years=3), e).date())})
            s0 += pd.DateOffset(months=3)
    return J


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-02")
    ap.add_argument("--end", default="2026-09-11")
    ap.add_argument("--recent", default="2023-09-11", help="start of the window config.yaml was tuned on")
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--seeds", type=int, default=40, help="random-entry null runs")
    ap.add_argument("--veto-seeds", type=int, default=20, help="live-frictions runs")
    ap.add_argument("--starts", type=int, default=4, help="start dates (a quarter apart) per variant")
    ap.add_argument("--workers", type=int, default=max(1, min(20, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--panel", default=None, help="price panel (default: from state/history/pit/build.json)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    sections = set(args.sections.split(","))

    cfg = load_config()
    if not cfg["account"]["paper"]:
        raise SystemExit("refusing: config.yaml is not a paper account")
    root = Path(cfg["_root"])
    hist = root / "state" / "history"
    build = json.loads((hist / "pit" / "build.json").read_text())
    panel_path = Path(args.panel) if args.panel else root / build["prices"]
    panel = pd.read_pickle(panel_path)
    close = panel["close"]
    dates = close.index
    members_pit = pit.expand(pd.read_csv(hist / "pit" / "members.csv"), dates)
    frozen = [s for s in json.loads((hist / "universe_bt.json").read_text())["symbols"] if s in close.columns]
    members_frozen = pd.DataFrame(True, index=dates, columns=frozen)

    from alpaca.data.historical import StockHistoricalDataClient
    from desk.config import alpaca_keys
    from desk.regime import vix_series
    key, secret = alpaca_keys()
    data = StockHistoricalDataClient(key, secret)
    spy = B.frozen_prices(data, ["SPY"], 2750, cfg, pd.Timestamp(args.end))[0]["close"]["SPY"].dropna()
    vix = vix_series(root / cfg["data"]["cache_dir"])
    earnings = pd.read_csv(hist / "earnings.csv") if (hist / "earnings.csv").exists() else None

    out_dir = root / "state" / "robustness" / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = out_dir / "_inputs.pkl"
    pd.to_pickle({"cfg": cfg, "spy": spy, "vix": vix, "earnings": earnings, "capital": float(cfg["account"]["starting_capital"]),
                  "members": {"pit": members_pit, "frozen": members_frozen}}, inputs)

    jobs = build_jobs(sections, args.start, args.end, args.recent, args.seeds, args.veto_seeds, args.starts)
    log.info("%d runs on %d workers; universe: %d point-in-time names (%d delisted), %d frozen names in the panel",
             len(jobs), args.workers, members_pit.any().sum(), build.get("delisted_members", 0), len(frozen))
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(str(inputs), str(panel_path))) as ex:
        for k, res in enumerate(ex.map(run_job, jobs, chunksize=1), 1):
            results[res["name"]] = res
            if k % 10 == 0 or k == len(jobs):
                log.info("%d/%d runs done (%.0fs)", k, len(jobs), time.time() - t0)
    inputs.unlink(missing_ok=True)

    study = analyze(results, cfg, panel, members_pit, members_frozen, spy, vix, args, build, hist)
    study["runtime_seconds"] = round(time.time() - t0)
    bench = study.pop("_bench_curves")
    (out_dir / "results.json").write_text(json.dumps(_clean(study), indent=1))
    pd.to_pickle({"runs": {n: {"equity": r["equity"], "n_positions": r["n_positions"], "group": r["group"]}
                           for n, r in results.items()},
                  "benchmarks": bench, "trades": {n: r["trades"] for n, r in results.items() if r["trades"] is not None}},
                 out_dir / "curves.pkl")
    (out_dir / "report.md").write_text(report_md(json.loads((out_dir / "results.json").read_text())))
    print((out_dir / "report.md").read_text())
    print(f"\nSaved {out_dir.relative_to(root)}/results.json, report.md, curves.pkl")


def analyze(results: dict, cfg: dict, panel: dict, members_pit, members_frozen, spy: pd.Series, vix, args,
            build: dict, hist: Path) -> dict:
    close = panel["close"]
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    S = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "window": [args.start, args.end],
         "recent_window": [args.recent, args.end], "config_id": B.current_config_id(cfg), "universe_build": build,
         "frictions": FRICTIONS, "runs": {}}
    for n, r in results.items():
        S["runs"][n] = {"group": r["group"], "label": r["job"].get("label"), "side": r["job"].get("side"),
                        "value": r["job"].get("value"), "start": r["job"]["start"], "end": r["job"]["end"],
                        "overrides": r["overrides"], "n_trades": r["n_trades"], "seconds": r["seconds"],
                        "avg_positions": float(r["n_positions"].mean()),
                        "avg_invested": float(1 - (r["cash"] / r["equity"]).mean()), **full_stats(r["equity"])}

    spy_w = spy[(spy.index >= start) & (spy.index <= end)]
    ew = ew_universe(close, members_pit)
    ew = ew[(ew.index >= start) & (ew.index <= end)]
    ew_frozen = ew_universe(close, members_frozen)
    ew_frozen = ew_frozen[(ew_frozen.index >= start) & (ew_frozen.index <= end)]
    panels = compute_panels(panel, cfg["signals"])
    top = plain_topn(panels, members_pit, cfg["signals"], cfg["portfolio"]["target_positions"], start, end)
    bench = {"SPY": spy_w / spy_w.iloc[0], "Equal-weight universe": ew / ew.iloc[0],
             "Equal-weight frozen 800": ew_frozen / ew_frozen.iloc[0], "Plain monthly top-12": top / top.iloc[0]}
    S["benchmarks"] = {k: full_stats(v) for k, v in bench.items()}
    S["_bench_curves"] = bench

    if "pit_full" in results:
        base = results["pit_full"]
        eq = base["equity"]
        r = eq.pct_change().dropna()
        s_r = spy.pct_change().reindex(r.index).fillna(0.0)
        e_r = ew.pct_change().reindex(r.index).fillna(0.0)
        beta = float(np.cov(r, s_r)[0, 1] / s_r.var())
        S["vs_spy"] = {"beta": beta, "corr": float(r.corr(s_r)),
                       "alpha_annual": float((r.mean() - beta * s_r.mean()) * 252),
                       "beta_ew": float(np.cov(r, e_r)[0, 1] / e_r.var()), "corr_ew": float(r.corr(e_r))}
        S["yearly"] = {"strategy": yearly(eq), "frozen": yearly(results["frozen_full"]["equity"]) if "frozen_full" in results else {},
                       "SPY": yearly(spy_w), "Equal-weight universe": yearly(ew)}
        S["stress"] = [{"label": lbl, "start": a, "end": b, "strategy": period_return(eq, a, b),
                        "strategy_dd": period_drawdown(eq, a, b), "SPY": period_return(spy, a, b),
                        "Equal-weight universe": period_return(ew, a, b),
                        "avg_positions": float(base["n_positions"][a:b].mean()) if len(base["n_positions"][a:b]) else None}
                       for lbl, a, b in STRESS]
        tr = base["trades"]
        S["trades"] = trade_stats(tr)
        S["switches"] = switch_events(tr)
        S["delisted_exits"] = ([] if tr is None or tr.empty else
                               tr.loc[tr["reason"] == "delisted", ["symbol", "entry_date", "exit_date", "pnl_pct", "pnl", "qty", "exit_price"]]
                               .assign(entry_date=lambda d: d["entry_date"].astype(str), exit_date=lambda d: d["exit_date"].astype(str))
                               .to_dict("records"))
        # Capacity: entry size against the name's 20-day dollar volume at the time
        dv = (panel["close"] * panel["volume"]).rolling(20, min_periods=10).mean()
        if tr is not None and len(tr):
            frac = [(t.qty * t.entry_price) / dv.at[pd.Timestamp(t.entry_date), t.symbol]
                    for t in tr.itertuples() if pd.notna(dv.at[pd.Timestamp(t.entry_date), t.symbol])]
            p95 = float(np.percentile(frac, 95))
            S["capacity"] = {"median_pct_adv": float(np.median(frac)), "p95_pct_adv": p95,
                             "equity_at_1pct_adv_p95": float(eq.mean() * 0.01 / p95) if p95 > 0 else None}
        S["bootstrap"] = stationary_bootstrap(r.to_numpy(), s_r.to_numpy(), 756, 5000, 20.0)
        reg_state = B.regime_inputs(spy, vix, cfg)[1]
        if reg_state is not None:
            st = reg_state.reindex(r.index).fillna("neutral")
            S["by_regime"] = {k: {"days": int((st == k).sum()), "ann_return": float(r[st == k].mean() * 252),
                                  "spy_ann_return": float(s_r[st == k].mean() * 252)} for k in ("neutral", "off")}
        # Deflated Sharpe on the long window: N from the trial log; the spread of Sharpes from the sensitivity
        # variants run here, the nearest thing to "the other configs that could have been chosen".
        log_ = T.load(hist / "trials.jsonl")
        n_trials = T.summary(log_, int((cfg.get("backtest") or {}).get("prior_untracked_trials", 0)))["n_trials"] if log_ else 1
        sens_sr = [T.return_stats(results[n]["equity"])["sr_daily"] for n in results
                   if results[n]["group"] == "sensitivity" and n.endswith("@0")]       # same window as pit_full
        S["dsr"] = {**T.deflated_sharpe(T.return_stats(eq), sens_sr + [T.return_stats(eq)["sr_daily"]], max(n_trials, 1)),
                    "n_logged_trials": n_trials}

    nulls = [n for n in results if results[n]["group"] == "null"]
    if nulls and "pit_full" in results:
        base = S["runs"]["pit_full"]
        tots = np.array([S["runs"][n]["total"] for n in nulls])
        shs = np.array([S["runs"][n]["sharpe"] for n in nulls])
        S["null"] = {"n": len(nulls), "total": {"p5": float(np.percentile(tots, 5)), "p50": float(np.median(tots)), "p95": float(np.percentile(tots, 95))},
                     "sharpe": {"p5": float(np.percentile(shs, 5)), "p50": float(np.median(shs)), "p95": float(np.percentile(shs, 95))},
                     "strategy_beats_share_total": float((base["total"] > tots).mean()),
                     "strategy_beats_share_sharpe": float((base["sharpe"] > shs).mean()),
                     "empirical_p_sharpe": float(((shs >= base["sharpe"]).sum() + 1) / (len(shs) + 1))}
    fr = [n for n in results if results[n]["group"] == "frictions"]
    if fr:
        tots = np.array([S["runs"][n]["total"] for n in fr])
        dds = np.array([S["runs"][n]["maxdd"] for n in fr])
        shs = np.array([S["runs"][n]["sharpe"] for n in fr])
        S["frictions_summary"] = {"n": len(fr), "total_p50": float(np.median(tots)), "total_p5": float(np.percentile(tots, 5)),
                                  "total_p95": float(np.percentile(tots, 95)), "maxdd_p50": float(np.median(dds)),
                                  "sharpe_p50": float(np.median(shs)),
                                  "invested_p50": float(np.median([S["runs"][n]["avg_invested"] for n in fr]))}
    wins = sorted(n for n in results if results[n]["group"] == "windows")
    if wins:
        rows = [S["runs"][n] for n in wins]
        spy3 = [period_return(spy, rw["start"], rw["end"]) for rw in rows]
        S["windows"] = {"n": len(rows), "rows": [{"start": rw["start"], "end": rw["end"], "total": rw["total"], "cagr": rw["cagr"],
                                                   "maxdd": rw["maxdd"], "sharpe": rw["sharpe"], "spy": s3}
                                                  for rw, s3 in zip(rows, spy3)],
                        "cagr_p10": float(np.percentile([rw["cagr"] for rw in rows], 10)),
                        "cagr_p50": float(np.median([rw["cagr"] for rw in rows])),
                        "worst_total": float(min(rw["total"] for rw in rows)),
                        "beat_spy": int(sum(rw["total"] > (s or 0) for rw, s in zip(rows, spy3))),
                        "lost_money": int(sum(rw["total"] < 0 for rw in rows)),
                        "hit_15": int(sum(rw["maxdd"] <= -0.15 for rw in rows))}
    V = variant_summary(S["runs"])
    if V:
        S["variants"] = V
        S["base_median"] = V.get("base")
    if any(v["group"] == "sensitivity" for v in V.values()):
        S["sensitivity"] = [{"label": lbl, "param": ".".join(path), "base": _get(cfg, path), "low": lo, "high": hi,
                             "low_run": V.get(f"sens_{path[-1]}_low"), "high_run": V.get(f"sens_{path[-1]}_high")}
                            for lbl, path, lo, hi in SENSITIVITY]
    # earnings-history coverage of the point-in-time members, by year: the block is inert where it is missing
    e_path = hist / "earnings.csv"
    if e_path.exists():
        e = pd.read_csv(e_path).dropna()
        covered = set(e["symbol"].astype(str).str.upper())
        yrs = members_pit.groupby(members_pit.index.year).any()
        S["earnings_coverage"] = {"history_starts": str(e["date"].astype(str).min()),
                                  "share_of_members_with_history": {
                                      int(y): float(np.mean([s in covered for s in yrs.columns[yrs.loc[y].to_numpy()]]))
                                      for y in yrs.index}}
    return S


def _get(cfg: dict, path: tuple):
    node = cfg
    for k in path:
        node = node[k]
    return node


def _pct(x, d=1) -> str:
    return "n/a" if x is None else f"{x * 100:+.{d}f}%"


def report_md(S: dict) -> str:
    R = S["runs"]
    L = [f"# Robustness study - {S['generated'][:10]}", "",
         f"config {S['config_id']} | window {S['window'][0]} to {S['window'][1]} | point-in-time universe: "
         f"{S['universe_build'].get('ever_members')} names ever members, {S['universe_build'].get('delisted_members')} since delisted", ""]

    def row(name, label):
        r = R.get(name)
        if not r:
            return None
        return (f"| {label} | {_pct(r['total'], 0)} | {_pct(r['cagr'])} | {_pct(r['maxdd'])} | {r['sharpe']:.2f} | "
                f"{r['sortino']:.2f} | {r['avg_positions']:.1f} |")
    L += ["## Core", "", "| Run | Total | CAGR | Max DD | Sharpe | Sortino | Avg pos |", "|---|---|---|---|---|---|---|"]
    for n, lbl in (("pit_full", "Point-in-time, full window"), ("frozen_full", "Frozen 800 (survivor list), full window"),
                   ("pit_oos", "Point-in-time, before the tuning window"), ("pit_recent", "Point-in-time, tuning window"),
                   ("frozen_recent", "Frozen 800, tuning window (published baseline)"),
                   ("pit_full_noearn", "Point-in-time, no earnings block"), ("frozen_full_noearn", "Frozen 800, no earnings block"),
                   ("pit_recent_noearn", "Point-in-time, tuning window, no earnings block"),
                   ("frozen_recent_noearn", "Frozen 800, tuning window, no earnings block"),
                   ):
        x = row(n, lbl)
        if x:
            L.append(x)
    V = S.get("variants", {})
    if V:
        b = V.get("base", {})
        L += ["", f"## Rules and costs (median of {b.get('n')} start dates a quarter apart; wins = starts where it beat config.yaml)", "",
              "| Variant | CAGR | Max DD | Sharpe | Wins on CAGR | Wins on DD |", "|---|---|---|---|---|---|",
              f"| config.yaml | {_pct(b.get('cagr'))} | {_pct(b.get('maxdd'))} | {b.get('sharpe', 0):.2f} | - | - |"]
        for k, v in V.items():
            if v["group"] in ("design", "costs"):
                L.append(f"| {v['label']} | {_pct(v['cagr'])} | {_pct(v['maxdd'])} | {v['sharpe']:.2f} | "
                         f"{v['beats_base_cagr']}/{v['paired']} | {v['beats_base_dd']}/{v['paired']} |")
    L += ["", "## Benchmarks (same window)", "", "| Benchmark | Total | CAGR | Max DD | Sharpe |", "|---|---|---|---|---|"]
    for k, b in S.get("benchmarks", {}).items():
        L.append(f"| {k} | {_pct(b['total'], 0)} | {_pct(b['cagr'])} | {_pct(b['maxdd'])} | {b['sharpe']:.2f} |")
    if "vs_spy" in S:
        v = S["vs_spy"]
        L += ["", f"Beta to SPY {v['beta']:.2f} (correlation {v['corr']:.2f}), alpha {_pct(v['alpha_annual'])} a year. "
                  f"Beta to the equal-weight universe {v['beta_ew']:.2f}."]
    if "dsr" in S:
        d = S["dsr"]
        L += [f"Deflated Sharpe {d['dsr']:.2f} (N = {d['n_logged_trials']} logged trials); undeflated {d['psr_vs_zero']:.2f}."]
    if "null" in S:
        n = S["null"]
        L += ["", "## Random-entry null", "",
              f"{n['n']} runs buying at random among eligible names with every other rule unchanged: total return "
              f"p5 {_pct(n['total']['p5'], 0)} / median {_pct(n['total']['p50'], 0)} / p95 {_pct(n['total']['p95'], 0)}; "
              f"Sharpe median {n['sharpe']['p50']:.2f}. The ranked strategy beats {n['strategy_beats_share_sharpe']:.0%} of them on "
              f"Sharpe (empirical p = {n['empirical_p_sharpe']:.3f})."]
    if "frictions_summary" in S:
        f = S["frictions_summary"]
        L += ["", "## Live frictions (analyst REMOVE/REDUCE, implied-vol half size at observed rates)", "",
              f"{f['n']} runs: total median {_pct(f['total_p50'], 0)} (p5 {_pct(f['total_p5'], 0)}, p95 {_pct(f['total_p95'], 0)}), "
              f"max DD median {_pct(f['maxdd_p50'])}, Sharpe median {f['sharpe_p50']:.2f}, invested {f['invested_p50']:.0%}."]
    if "stress" in S:
        L += ["", "## Stress episodes", "", "| Episode | Dates | Strategy | Worst DD inside | SPY | EW universe | Avg pos |", "|---|---|---|---|---|---|---|"]
        for s in S["stress"]:
            L.append(f"| {s['label']} | {s['start']} to {s['end']} | {_pct(s['strategy'])} | {_pct(s['strategy_dd'])} | "
                     f"{_pct(s['SPY'])} | {_pct(s['Equal-weight universe'])} | {(s['avg_positions'] or 0):.1f} |")
    if "yearly" in S:
        L += ["", "## Calendar years", "", "| Year | Strategy | Frozen 800 | SPY | EW universe |", "|---|---|---|---|---|"]
        for y in S["yearly"]["strategy"]:
            L.append(f"| {y} | {_pct(S['yearly']['strategy'][y])} | {_pct(S['yearly']['frozen'].get(y))} | "
                     f"{_pct(S['yearly']['SPY'].get(y))} | {_pct(S['yearly']['Equal-weight universe'].get(y))} |")
    if "sensitivity" in S:
        L += ["", "## Sensitivity (one notch each way; medians over the start dates, wins vs config.yaml on CAGR)", "",
              "| Parameter | Low | CAGR | Max DD | Wins | Base | High | CAGR | Max DD | Wins |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s in S["sensitivity"]:
            lo, hi = s["low_run"] or {}, s["high_run"] or {}
            L.append(f"| {s['label']} | {s['low']} | {_pct(lo.get('cagr'))} | {_pct(lo.get('maxdd'))} | {lo.get('beats_base_cagr')}/{lo.get('paired')} | "
                     f"{s['base']} | {s['high']} | {_pct(hi.get('cagr'))} | {_pct(hi.get('maxdd'))} | {hi.get('beats_base_cagr')}/{hi.get('paired')} |")
    if "windows" in S:
        w = S["windows"]
        L += ["", "## Every 3-year window (quarterly starts)", "",
              f"{w['n']} windows: CAGR median {_pct(w['cagr_p50'])}, 10th percentile {_pct(w['cagr_p10'])}, worst total {_pct(w['worst_total'])}; "
              f"beat SPY in {w['beat_spy']}/{w['n']}, lost money in {w['lost_money']}/{w['n']}, drew down 15%+ in {w['hit_15']}/{w['n']}."]
    if "bootstrap" in S:
        b = S["bootstrap"]
        L += ["", "## Bootstrap (3-year paths)", "",
              f"{b['paths']} paths: total p5 {_pct(b['total']['p5'], 0)} / median {_pct(b['total']['p50'], 0)} / p95 {_pct(b['total']['p95'], 0)}; "
              f"P(loss) {b['p_loss']:.0%}, P(15% drawdown) {b['p_dd_15']:.0%}, P(25% drawdown) {b['p_dd_25']:.0%}, P(beat SPY) {b['p_beat_spy']:.0%}."]
    if "trades" in S and S["trades"]:
        t = S["trades"]
        L += ["", "## Trades (point-in-time, full window)", "",
              f"{t['n']} trades, win rate {t['win_rate']:.0%}, expectancy {t['expectancy_r']:+.2f}R, profit factor "
              f"{(t['profit_factor'] or 0):.2f}, top 10 trades = {(t['top10_share'] or 0):.0%} of net P&L."]
        sw = S.get("switches", {})
        L.append(f"Kill switch fired {len(sw.get('kill-switch', []))}x, hard floor {len(sw.get('hard-floor', []))}x. "
                 f"Delisted while held: {len(S.get('delisted_exits', []))}.")
    return "\n".join(x for x in L if x is not None) + "\n"


if __name__ == "__main__":
    main()

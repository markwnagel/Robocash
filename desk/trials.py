"""Backtest trial log and the Deflated Sharpe Ratio.

Every backtest run appends one record to state/history/trials.jsonl (committed with the repo). The log is
the denominator of every result: a Sharpe ratio picked as the best of N tries is inflated by the picking,
and the Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014, Journal of Portfolio Management 40(5))
corrects for that and for fat-tailed, skewed returns.

    DSR = Phi( (SR - SR0) * sqrt(T - 1) / sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2) )
    SR0 = sqrt(Var[SR_n]) * ((1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e)))      g = Euler-Mascheroni

SR, SR0 and Var[SR_n] are per-period (daily) Sharpe ratios; kurt is ordinary kurtosis (3 for normal
returns); N is the number of independent trials behind the choice; Var[SR_n] is the spread of the trials'
Sharpe ratios. DSR is the probability that the true Sharpe is above zero once all of that is paid for;
0.95 is the usual bar.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import pandas as pd

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def return_stats(equity: pd.Series) -> dict:
    r = equity.pct_change().dropna()
    sd = float(r.std())
    return {"n_obs": int(len(r)), "sr_daily": float(r.mean() / sd) if sd > 0 else 0.0,
            "skew": float(r.skew()) if len(r) > 2 else 0.0,
            "kurt": float(r.kurt()) + 3.0 if len(r) > 3 else 3.0}      # pandas gives excess kurtosis


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """SR0: the best Sharpe you would expect from n_trials strategies with no edge at all."""
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    return math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials)
                                     + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * math.e)))


def probabilistic_sharpe(sr: float, sr0: float, n_obs: int, skew: float, kurt: float) -> float:
    """Probability the true per-period Sharpe exceeds sr0, given the sample's length, skew and kurtosis."""
    if n_obs < 2:
        return float("nan")
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return float("nan")
    return _N.cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom))


def deflated_sharpe(stats: dict, trial_srs: list, n_trials: int) -> dict:
    """stats: return_stats() of the run judged. trial_srs: daily Sharpes of the trials it was chosen from,
    on the same window. n_trials: every independent trial, logged or not."""
    var = float(pd.Series(trial_srs).var()) if len(trial_srs) > 1 else 0.0
    sr0 = expected_max_sharpe(var, n_trials)
    return {"dsr": probabilistic_sharpe(stats["sr_daily"], sr0, stats["n_obs"], stats["skew"], stats["kurt"]),
            "sr0_daily": sr0, "n_trials": n_trials, "trials_with_spread": len(trial_srs),
            "psr_vs_zero": probabilistic_sharpe(stats["sr_daily"], 0.0, stats["n_obs"], stats["skew"], stats["kurt"])}


def config_id(overrides: dict) -> str:
    """One variant = one set of overrides. Re-running a variant does not count as a new trial."""
    return hashlib.sha1(json.dumps(overrides, sort_keys=True, default=str).encode()).hexdigest()[:10]


# What identifies a variant. Lives here, not in backtest.py, so the daily report can tell whether the live
# reference describes the running config without importing the backtest CLI.
STRATEGY_SECTIONS = ("signals", "portfolio", "exits", "risk", "regime", "universe")
BACKTEST_FACTS = ("exclude_earnings_within_days", "insider_tilt_weight", "insider_min_net_usd", "candidate_pool")
# backtest.py's CLI defaults. Integers on purpose: every logged config_id hashed them as 10 and 20, and a
# float here would give today's config a new id and a new trial.
DEFAULT_SLIPPAGE_BPS = 10
DEFAULT_HALT_COOLDOWN = 20


def strategy_fingerprint(cfg: dict, slippage_bps: float, halt_cooldown: int) -> dict:
    """Everything that changes a backtest's result, and nothing that does not (model names, email)."""
    sections = {k: cfg[k] for k in STRATEGY_SECTIONS}
    # The live halt cooldown is the backtest's halt_cooldown, already below; hashing it twice would give
    # today's config a new id without changing a single backtest result.
    sections["risk"] = {k: v for k, v in sections["risk"].items() if k != "halt_cooldown_days"}
    return {**sections,
            "facts": {k: cfg["facts"].get(k) for k in BACKTEST_FACTS},
            "slippage_bps": slippage_bps, "halt_cooldown": halt_cooldown}


def current_config_id(cfg: dict) -> str:
    """The variant id config.yaml runs as today, at the backtest's default slippage and halt cooldown."""
    return config_id(strategy_fingerprint(cfg, DEFAULT_SLIPPAGE_BPS, DEFAULT_HALT_COOLDOWN))


def record(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **rec},
                           default=str) + "\n")


def load(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def judge(trials: list, current: dict, prior_untracked: int) -> dict:
    """DSR for `current` against everything tried: logged variants on any window count toward N (plus the
    ones run before the log existed); the spread of Sharpes uses only variants run on the same window and
    universe, where they are comparable."""
    same = {}
    for t in trials + [current]:
        if (t.get("universe_id"), t.get("start"), t.get("end")) == (current.get("universe_id"), current.get("start"), current.get("end")):
            same[t["config_id"]] = t["stats"]["sr_daily"]          # latest run of each variant
    n = len({t["config_id"] for t in trials + [current]}) + int(prior_untracked)
    return deflated_sharpe(current["stats"], list(same.values()), n)


# ------------------------------------------------------------------ the README table, generated from the log

START_MARK, END_MARK = "<!-- trials:start -->", "<!-- trials:end -->"


def _label(t: dict) -> str:
    ov = t.get("overrides") or {}
    return ", ".join(f"{k.replace('_', '-')} {v}" for k, v in sorted(ov.items())) if ov else "config.yaml"


def summary(log: list, prior_untracked: int) -> dict:
    """Latest run of every variant on the window where the most variants were tested, each with its DSR."""
    if not log:
        return {"rows": [], "runs": 0}
    groups: dict = {}
    for t in log:
        groups.setdefault((t.get("universe_id"), t.get("start"), t.get("end")), set()).add(t["config_id"])
    key = max(groups, key=lambda k: (len(groups[k]), k[2] or "", k[1] or ""))
    latest, runs = {}, {}
    for t in log:
        if (t.get("universe_id"), t.get("start"), t.get("end")) == key:
            latest[t["config_id"]] = t
            runs[t["config_id"]] = runs.get(t["config_id"], 0) + 1
    plain = [t for t in latest.values() if not t.get("overrides")]
    current = max(plain, key=lambda t: t["ts"])["config_id"] if plain else None
    rows = []
    for cid, t in latest.items():
        v = judge(log, t, prior_untracked)
        label = _label(t)
        if not t.get("overrides") and cid != current:
            label = f"config.yaml as of {t['ts'][:10]} (superseded)"
        rows.append({"label": label, "current": cid == current, "dsr": v["dsr"], "total": t["total"],
                     "maxdd": t["maxdd"], "sharpe": t["sharpe_annual"], "runs": runs[cid]})
    rows.sort(key=lambda r: -r["dsr"])
    return {"rows": rows, "runs": len(log), "last_run": max(t["ts"] for t in log), "window": key,
            "n_trials": len({t["config_id"] for t in log}) + int(prior_untracked),
            "current": next((r for r in rows if r["current"]), None)}


def markdown(s: dict) -> str:
    if not s["rows"]:
        return "_No backtest trials logged yet._"
    uid, start, end = s["window"]
    lines = [f"_Generated from {s['runs']} logged runs (latest {s['last_run'][:16].replace('T', ' ')} UTC) by "
             f"`scripts/trials_report.py --write-readme`. Window {start} to {end}, universe `{uid}`, "
             f"N = {s['n_trials']} trials._", "",
             "| Variant | Deflated Sharpe | Return | Max DD | Sharpe | Runs |", "|---|---|---|---|---|---|"]
    for r in s["rows"]:
        name = f"**current: {r['label']}**" if r["current"] else r["label"]
        lines.append(f"| {name} | {r['dsr']:.2f} | {r['total']:+.0%} | {r['maxdd']:.1%} | {r['sharpe']:.2f} | {r['runs']} |")
    return "\n".join(lines)


def write_readme(readme: Path, block: str) -> None:
    text = readme.read_text()
    if START_MARK not in text or END_MARK not in text:
        raise ValueError(f"{readme} has no {START_MARK} ... {END_MARK} block")
    head, rest = text.split(START_MARK, 1)
    _, tail = rest.split(END_MARK, 1)
    readme.write_text(f"{head}{START_MARK}\n{block}\n{END_MARK}{tail}")


def readme_lag(readme: Path, log: list) -> int | None:
    """Runs logged since the README table was generated; None if the README has no generated table."""
    import re
    m = re.search(r"Generated from (\d+) logged runs", readme.read_text()) if readme.exists() else None
    return len(log) - int(m.group(1)) if m else None

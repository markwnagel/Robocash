"""The shadow book: every decision that changed what the desk bought, and what it would have earned.

The quant engine is backtested; the layers on top of it are not - the analyst's REMOVE, the red team behind it,
the headline classifier's blocks, the SEC-filing and earnings blocks, the implied-vol half size. Nothing measured
what they cost or saved (engineering review 2026-10-17). Each desk run appends one row per decision here, and
outcomes() replays each row against later daily bars with the desk's own bracket - entry at the staged close,
stop and target the same distances as a real entry, exit at the first touch or the max hold - next to the
same-day buys as the control group. retirement_due() applies the pre-registered rule in config.yaml (shadow).

Kinds:
  bought            staged and sent (the control group)
  half_size         bought at a fraction of its size by the implied-vol rule; the row is the part not bought
  blocked           a fact-checked candidate kept out by a hard filter (filing, headline, earnings); `reason`
  analyst_remove    a candidate the analyst vetoed
  analyst_exit      a holding the analyst told the desk to sell; the outcome is what holding on would have done
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import state as _state


def book_path() -> Path:
    """Read at call time, so a test that points desk.state.STATE_DIR elsewhere is followed."""
    return _state.STATE_DIR / "shadow" / "book.jsonl"


HORIZONS = (10, 20, 60)


def blocker_class(reason: str) -> str:
    """'dilutive filing: 8-K item 3.02 ...' -> 'filing'; the classes the retirement rule is judged on."""
    r = reason.lower()
    if "filing" in r:
        return "filing"
    if "headline" in r:
        return "headline"
    if "earnings" in r:
        return "earnings"
    return re.sub(r"[^a-z]+", "_", r.split(":")[0]).strip("_")[:30] or "other"


def row(day: date, kind: str, sym: str, price: float, atr: float | None, ecfg: dict, qty: float = 0,
        reason: str = "", rank: int | None = None) -> dict:
    """One decision. Stop and target are measured like a real entry's, so a blocked name is replayed with the
    bracket it would have had."""
    m, r = float(ecfg["stop_atr_multiple"]), float(ecfg.get("take_profit_r_multiple") or 0)
    stop = round(price - m * atr, 2) if atr and atr > 0 else None
    target = round(price + r * m * atr, 2) if stop is not None and r > 0 else None
    return {"date": day.isoformat(), "kind": kind, "symbol": sym, "price": round(float(price), 4),
            "atr": round(float(atr), 4) if atr else None, "stop": stop, "target": target, "qty": float(qty),
            "notional": round(float(qty) * float(price), 2), "reason": reason[:200], "rank": rank}


def record(rows: list, path: Path | None = None) -> None:
    if not rows:
        return
    path = path or book_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def load(path: Path | None = None) -> pd.DataFrame:
    path = path or book_path()
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    df = pd.DataFrame(rows)
    # One row per decision: a second run the same day (a manual re-run) repeats them.
    return df.drop_duplicates(["date", "kind", "symbol"], keep="last").reset_index(drop=True) if len(df) else df


def replay(r: dict, high: pd.Series, low: pd.Series, close: pd.Series, max_hold: int) -> dict:
    """The bracket's outcome from the bars after r['date']: (return, exit reason, days), None while open."""
    after = close.index > pd.Timestamp(r["date"])
    c, h, lo = close[after].dropna(), high[after], low[after]
    if not len(c):
        return {"bracket_ret": None, "exit": None, "days": 0}
    px = float(r["price"])
    for i, d in enumerate(c.index[:max_hold]):
        if r.get("stop") is not None and float(lo.get(d, np.nan)) <= r["stop"]:
            return {"bracket_ret": r["stop"] / px - 1, "exit": "stop", "days": i + 1}
        if r.get("target") is not None and float(h.get(d, np.nan)) >= r["target"]:
            return {"bracket_ret": r["target"] / px - 1, "exit": "target", "days": i + 1}
    if len(c) >= max_hold:
        return {"bracket_ret": float(c.iloc[max_hold - 1]) / px - 1, "exit": "time", "days": max_hold}
    return {"bracket_ret": float(c.iloc[-1]) / px - 1, "exit": None, "days": len(c)}      # still open: marked


def once_per_spell(book: pd.DataFrame, days: int = 28) -> pd.DataFrame:
    """A name blocked by the same filter day after day is one decision, not one a day: SNOW's 8-K 3.02 kept it out
    for weeks, and 30 near-identical rows would outweigh everything else in its group. Keeps a blocked row only
    when the same symbol and blocker class has no kept row in the previous `days` calendar days."""
    if book.empty:
        return book
    keep, last = [], {}
    for i, r in book.sort_values("date").iterrows():
        if r["kind"] != "blocked":
            keep.append(i)
            continue
        k, d = (r["symbol"], blocker_class(str(r.get("reason") or ""))), date.fromisoformat(str(r["date"]))
        if k not in last or (d - last[k]).days >= days:
            keep.append(i)
            last[k] = d
    return book.loc[keep].reset_index(drop=True)


def outcomes(book: pd.DataFrame, panel: dict, spy: pd.Series, max_hold: int = 60) -> pd.DataFrame:
    """Each decision with its forward returns at HORIZONS trading days, the same against SPY, and the bracket
    replay. panel: {'close', 'high', 'low'} DataFrames covering the book's symbols."""
    if book.empty:
        return book
    out = []
    for r in once_per_spell(book).to_dict("records"):
        s = r["symbol"]
        if s not in panel["close"].columns:
            continue
        close = panel["close"][s]
        rec = dict(r, **replay(r, panel["high"][s], panel["low"][s], close, max_hold))
        after = close.index > pd.Timestamp(r["date"])
        c, sp = close[after].dropna(), spy[spy.index > pd.Timestamp(r["date"])]
        s0 = spy[spy.index <= pd.Timestamp(r["date"])]
        for n in HORIZONS:
            if len(c) >= n and len(sp) >= n and len(s0):
                ret = float(c.iloc[n - 1]) / float(r["price"]) - 1
                rec[f"ret_{n}d"], rec[f"excess_{n}d"] = ret, ret - (float(sp.iloc[n - 1]) / float(s0.iloc[-1]) - 1)
            else:
                rec[f"ret_{n}d"] = rec[f"excess_{n}d"] = None
        rec["group"] = f"blocked:{blocker_class(r.get('reason', ''))}" if r["kind"] == "blocked" else r["kind"]
        out.append(rec)
    return pd.DataFrame(out)


def summary(df: pd.DataFrame) -> dict:
    """Per group: decisions, how many have a closed bracket, mean bracket return and mean 20-day excess over SPY."""
    if df.empty:
        return {}
    res = {}
    groups = list(df.groupby("group"))
    full = df[(df["group"] == "bought") & ~df["reason"].fillna("").str.contains("half size")]
    if len(full):
        groups.append(("bought_full", full))       # the control for half_size: buys the rule left alone
    for g, x in groups:
        closed = x[x["exit"].notna()]
        res[g] = {"n": int(len(x)), "n_closed": int(len(closed)),
                  "bracket_ret_mean": float(closed["bracket_ret"].mean()) if len(closed) else None,
                  "excess_20d_mean": float(x["excess_20d"].dropna().mean()) if x["excess_20d"].notna().any() else None,
                  "n_20d": int(x["excess_20d"].notna().sum())}
    return res


def retirement_due(s: dict, rule: dict) -> list:
    """The pre-registered test (config shadow): once a group that keeps names out has `min_closed` closed brackets,
    it is flagged when the names it kept out did at least as well as the names the desk bought. Flagged, not
    switched off: retiring a rule is the owner's change."""
    need = int(rule.get("min_closed", 30))
    flags = []
    for g, v in s.items():
        base = (s.get("bought_full" if g == "half_size" else "bought") or {}).get("bracket_ret_mean")
        if g in ("bought", "bought_full") or v["n_closed"] < need or base is None or v["bracket_ret_mean"] is None:
            continue
        if v["bracket_ret_mean"] >= base:
            flags.append(f"{g}: the {v['n_closed']} names it kept out returned {v['bracket_ret_mean']:+.1%} a trade on "
                         f"the desk's bracket vs {base:+.1%} for the names bought - it is not protecting the book; "
                         f"retire it or show why not")
    return flags


def evaluate(cfg: dict, data=None, book: pd.DataFrame | None = None) -> dict:
    """Load the book, fetch bars for its names and SPY, replay, summarise, and apply the retirement rule.
    {'summary', 'flags', 'rows', 'since'}; {} when the book is empty. Never raises past a missing data client."""
    book = load() if book is None else book
    if book.empty:
        return {}
    from .data import cached_bars
    if data is None:
        from alpaca.data.historical import StockHistoricalDataClient
        from .config import alpaca_keys
        data = StockHistoricalDataClient(*alpaca_keys())
    first = date.fromisoformat(str(book["date"].min()))
    days = (date.today() - first).days + 120
    panel = cached_bars(data, sorted(set(book["symbol"]) | {"SPY"}), days, cfg, tag="shadow")
    spy = panel["close"]["SPY"].dropna()
    df = outcomes(book, panel, spy, int(cfg["exits"]["max_hold_days"]))
    s = summary(df)
    return {"summary": s, "flags": retirement_due(s, cfg.get("shadow") or {}), "rows": len(df), "since": first.isoformat()}


def markdown(ev: dict) -> str:
    """The audit's SHADOW BOOK section."""
    if not ev:
        return "SHADOW BOOK\n  empty - decisions are recorded from 2026-10-19"
    lines = [f"SHADOW BOOK (decisions since {ev['since']}; desk bracket replayed on daily bars)"]
    for g, v in sorted(ev["summary"].items()):
        br = f"{v['bracket_ret_mean']:+.1%}" if v["bracket_ret_mean"] is not None else "n/a"
        ex = f"{v['excess_20d_mean']:+.1%}" if v["excess_20d_mean"] is not None else "n/a"
        lines.append(f"  {g:22} n {v['n']:3}  closed {v['n_closed']:3}  bracket {br:>7}  20d vs SPY {ex:>7} (n {v['n_20d']})")
    lines += [f"  RETIRE? {f}" for f in ev["flags"]] or ["  retirement rule: nothing due (needs "
                                                         "shadow.min_closed closed brackets per group)"]
    return "\n".join(lines)

"""Live vs backtest parity: replay the backtester over the live period and say where the desk did something else.

Three live/backtest mismatches were found by accident - the stop anchored to the signal close, sizing cuts the
backtest did not model (REDUCE, the implied-vol half size) and the idle cash they left (engineering review
2026-10-17, item 9). This runs the backtest from the first live day with today's config on the live universe
(state/universe.json plus every name the desk has held or staged), the earnings dates the desk saw (its daily
earnings cache) and the same starting capital, and compares:

  equity     live (from the daily briefs) against the backtest, and the gap in percentage points
  entries    per signal day, the names the backtest staged vs the names the desk staged (shadow book 'bought'):
             overlap, and the live-only and backtest-only names, each with the shadow book's reason when it has one
  cash       average cash share, live vs backtest

The monitor puts the result in its payload and report and raises SEV2 when the equity gap passes
parity.equity_gap_alarm_pts. The analyst, the red team and fills are not simulated, so some gap is expected; the
point is to see it, and its causes, while they are small.
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import shadow
from . import state as _state

EQUITY = re.compile(r"^Equity \$([\d,]+)\b.*cash \$([\d,]+)")


def live_equity(briefs_dir: Path | None = None) -> pd.DataFrame:
    """date -> equity, cash from each day's brief (the first 'Equity $...' line)."""
    d = briefs_dir or _state.BRIEFS_DIR
    rows = []
    for p in sorted(Path(d).glob("20*.md")):
        for line in p.read_text().splitlines():
            m = EQUITY.match(line)
            if m:
                rows.append({"date": pd.Timestamp(p.stem), "equity": float(m.group(1).replace(",", "")),
                             "cash": float(m.group(2).replace(",", ""))})
                break
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["equity", "cash"])


def earnings_seen(cache_dir: Path) -> pd.DataFrame:
    """The report dates the desk saw, one row per (date, symbol), from its daily earnings cache."""
    rows = set()
    for p in sorted(Path(cache_dir).glob("earnings_*.json")):
        try:
            for sym, d in json.loads(p.read_text()).items():
                if d:
                    rows.add((str(d)[:10], sym))
        except (ValueError, AttributeError):
            continue
    return pd.DataFrame(sorted(rows), columns=["date", "symbol"])


def compare_entries(bt_entries: list, live_bought: pd.DataFrame, book: pd.DataFrame, dates: pd.DatetimeIndex,
                    last_day: str | None = None, tolerance: int = 5) -> dict:
    """Backtest entries are logged on their fill day, live rows on the signal (staging) day; both are compared on
    the signal day (the trading day before a backtest fill). A name staged by both within `tolerance` trading days
    is a match, with the lag recorded: the first run found the backtest buying DDOG, NTAP and HNGE three to five
    days before live - live re-stages an order that did not fill. Live orders staged on the last day cannot have
    filled in the replay and are left out."""
    pos = {d: i for i, d in enumerate(dates)}
    bt = []
    for e in bt_entries:
        i = pos.get(pd.Timestamp(e["date"]))
        if i:
            bt.append((i - 1, e["symbol"]))
    live = []
    for r in live_bought.to_dict("records"):
        i = pos.get(pd.Timestamp(r["date"]))
        if i is not None and r["date"] != last_day:
            live.append((i, r["symbol"]))
    live = sorted(set(live))
    why = {}
    for r in book.to_dict("records") if len(book) else []:
        if r["kind"] in ("blocked", "analyst_remove"):
            why.setdefault(r["symbol"], f"{r['kind']}: {str(r.get('reason') or '')[:60]}")
    used, lags, only_bt = set(), [], []
    for i, sym in sorted(bt):
        cands = [(abs(j - i), j) for j, s in live if s == sym and (j, s) not in used and abs(j - i) <= tolerance]
        if cands:
            _, j = min(cands)
            used.add((j, sym))
            lags.append(j - i)
        else:
            only_bt.append((dates[i].date().isoformat(), sym))
    # A live name re-staged on several days after a missed fill is one decision: count its first staging only.
    first_live = {}
    for j, sym in live:
        if (j, sym) not in used and not any(s == sym and abs(k - j) <= tolerance for k, s in used):
            first_live.setdefault(sym, j)
    only_live = sorted((dates[j].date().isoformat(), s) for s, j in first_live.items())
    total = len(lags) + len(only_bt) + len(only_live)
    return {"overlap": round(len(lags) / total, 3) if total else None, "both": len(lags),
            "backtest_only": len(only_bt), "live_only": len(only_live),
            "median_lag_days": float(pd.Series(lags).median()) if lags else None,
            "backtest_only_names": [f"{d} {s}" + (f" ({why[s]})" if s in why else "") for d, s in only_bt][-10:],
            "live_only_names": [f"{d} {s}" for d, s in only_live][-10:]}


def run(cfg: dict, data=None, start: date | None = None, end: date | None = None) -> dict:
    """Replay and compare. {} before there is a live history."""
    import backtest as B
    from .data import cached_bars
    from .regime import vix_series
    eq = live_equity()
    if eq.empty:
        return {}
    start = start or eq.index[0].date()
    end = end or eq.index[-1].date()
    if data is None:
        from alpaca.data.historical import StockHistoricalDataClient
        from .config import alpaca_keys
        data = StockHistoricalDataClient(*alpaca_keys())
    root = Path(cfg["_root"])
    book = shadow.load()
    uni = json.loads((root / "state" / "universe.json").read_text()).get("symbols", [])
    symbols = sorted(set(uni) | set(book["symbol"] if len(book) else []) | set(_state.load_state()["positions"]))
    warm = int(cfg["signals"]["momentum_lookback_days"]) + int(cfg["signals"]["trend_ma_days"])
    days = (end - start).days + int(warm * 1.6) + 30
    panel = cached_bars(data, symbols, days, cfg, tag="parity")
    spy = cached_bars(data, ["SPY"], days, cfg, tag="parity-spy")["close"]["SPY"].dropna()
    vix = vix_series(root / cfg["data"]["cache_dir"])
    reg, reg_state = B.regime_inputs(spy, vix, cfg)
    log: list = []
    capital = float(eq["equity"].iloc[0])
    curve, trades = B.run_backtest(panel, cfg, pd.Timestamp(start), pd.Timestamp(end), capital, 10, 20, regime=reg,
                                   regime_state=reg_state, earnings=earnings_seen(root / cfg["data"]["cache_dir"]),
                                   entries_log=log)
    bt_eq = curve["equity"]
    live_last, bt_last = float(eq["equity"].iloc[-1]), float(bt_eq.iloc[-1])
    spy_ret = float(spy[spy.index <= pd.Timestamp(end)].iloc[-1] / spy[spy.index <= pd.Timestamp(start)].iloc[-1] - 1)
    bought = book[book["kind"] == "bought"] if len(book) else pd.DataFrame(columns=["date", "symbol"])
    ent = compare_entries(log, bought, book, panel["close"].index, last_day=end.isoformat())
    return {"window": f"{start} to {end}", "live_return": round(live_last / capital - 1, 4),
            "backtest_return": round(bt_last / float(bt_eq.iloc[0]) - 1, 4), "spy_return": round(spy_ret, 4),
            "equity_gap_pts": round(100 * (live_last / capital - bt_last / float(bt_eq.iloc[0])), 2),
            "live_cash_share": round(float((eq["cash"] / eq["equity"]).mean()), 3),
            "backtest_cash_share": round(float((curve["cash"] / curve["equity"]).mean()), 3),
            "entries": ent, "backtest_trades_closed": int(len(trades)), "backtest_entries": len(log)}


def alarms(res: dict, cfg: dict, today: date) -> list:
    """SEV2 when live has drifted from the backtest by more than parity.equity_gap_alarm_pts."""
    lim = float((cfg.get("parity") or {}).get("equity_gap_alarm_pts", 5.0))
    gap = (res or {}).get("equity_gap_pts")
    if gap is None or abs(gap) <= lim:
        return []
    return [{"key": f"parity-gap-{today:%Y-%m-%d}", "severity": "SEV2",
             "title": f"Live is {gap:+.1f} pts from the backtest over {res['window']}",
             "detail": f"live {res['live_return']:+.1%}, backtest {res['backtest_return']:+.1%}, SPY {res['spy_return']:+.1%}; "
                       f"cash live {res['live_cash_share']:.0%} vs backtest {res['backtest_cash_share']:.0%}; entry overlap "
                       f"{res['entries']['overlap']}. The days that differ are in the report's Parity section."}]


def markdown(res: dict) -> list:
    if not res:
        return ["## Live vs backtest", "- no live history yet"]
    if "error" in res:
        return ["## Live vs backtest", f"- unavailable: {res['error']}"]
    e = res["entries"]
    lines = ["## Live vs backtest", f"- {res['window']}: live {res['live_return']:+.1%}, backtest {res['backtest_return']:+.1%} "
             f"(gap {res['equity_gap_pts']:+.1f} pts), SPY {res['spy_return']:+.1%}",
             f"- cash: live {res['live_cash_share']:.0%}, backtest {res['backtest_cash_share']:.0%}",
             f"- entries: {e['both']} in both, {e['backtest_only']} backtest only, {e['live_only']} live only "
             f"(overlap {e['overlap']})"]
    if e.get("median_lag_days") is not None:
        lines.append(f"  - matched names were staged live a median {e['median_lag_days']:+.0f} trading days after the backtest")
    if e["backtest_only_names"]:
        lines.append(f"  - backtest only: {'; '.join(e['backtest_only_names'][-6:])}")
    if e["live_only_names"]:
        lines.append(f"  - live only: {'; '.join(e['live_only_names'][-6:])}")
    return lines

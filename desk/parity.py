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
    # How much of the backtest-only side is a live gate the replay does not apply (a hard-filter block or an analyst
    # veto), as opposed to a genuinely different pick - the monitor kept reading the whole gap as unexplained
    # (R-50, R-53, 2026-10-21).
    gated = sum(1 for _, s in only_bt if s in why)
    return {"overlap": round(len(lags) / total, 3) if total else None, "both": len(lags),
            "match_tolerance_days": tolerance, "backtest_only_gated_live": gated,
            "backtest_only": len(only_bt), "live_only": len(only_live),
            "median_lag_days": float(pd.Series(lags).median()) if lags else None,
            "backtest_only_names": [f"{d} {s}" + (f" ({why[s]})" if s in why else "") for d, s in only_bt][-10:],
            "live_only_names": [f"{d} {s}" for d, s in only_live][-10:]}


def attribute(bt_entries: list, bt_trades: pd.DataFrame, last_close: pd.Series, live_closed: pd.DataFrame,
              live_open: list, live_bought: pd.DataFrame, gated: set, dates: pd.DatetimeIndex, capital: float,
              gap_pts: float, tolerance: int = 5) -> dict:
    """The gap in percentage points, by cause (monitor R-58, 2026-10-22). Each entry is valued at its result -
    a closed trade's P&L, an open one marked at the last close - and sorted: names both sides took (within
    `tolerance` trading days), names only the backtest took because a live gate kept them out (`gated`), other
    backtest-only names, and live-only names. What the entries do not explain - fills, slippage, the size of each
    position, idle cash - is the residual."""
    pos = {d: i for i, d in enumerate(dates)}
    live_days = {}
    for r in live_bought.to_dict("records"):
        if pd.Timestamp(r["date"]) in pos:
            live_days.setdefault(r["symbol"], []).append(pos[pd.Timestamp(r["date"])])
    bt_days = {}
    pnl_closed = {}
    if len(bt_trades):
        for r in bt_trades.to_dict("records"):
            k = (r["symbol"], pd.Timestamp(r["entry_date"]))
            pnl_closed[k] = pnl_closed.get(k, 0.0) + float(r["pnl"])
    pts = {"matched_live": 0.0, "matched_backtest": 0.0, "backtest_only_gated": 0.0, "backtest_only_other": 0.0,
           "live_only": 0.0}
    near = lambda days, i: any(abs(j - i) <= tolerance for j in days)  # noqa: E731
    for e in bt_entries:
        i = pos.get(pd.Timestamp(e["date"]))
        if not i:
            continue
        sig = i - 1
        bt_days.setdefault(e["symbol"], []).append(sig)
        k = (e["symbol"], pd.Timestamp(e["date"]))
        v = pnl_closed[k] if k in pnl_closed else (float(last_close.get(e["symbol"], e["price"])) - e["price"]) * e["qty"]
        if near(live_days.get(e["symbol"], []), sig):
            pts["matched_backtest"] += v
        elif e["symbol"] in gated:
            pts["backtest_only_gated"] += v
        else:
            pts["backtest_only_other"] += v
    rows = [(str(r["symbol"]), str(r["entry_date"]), float(r["pnl"])) for r in live_closed.to_dict("records")]
    rows += [(s, str(p["entry_date"]), (float(p.get("last_price") or p["entry_price"]) - float(p["entry_price"]))
              * float(p["qty"])) for s, p in live_open]
    for sym, d, v in rows:
        i = pos.get(pd.Timestamp(d))
        if i is not None and near(bt_days.get(sym, []), i):
            pts["matched_live"] += v
        else:
            pts["live_only"] += v
    out = {k: round(100 * v / capital, 2) for k, v in pts.items()}
    out.update(matched_split(bt_entries, bt_trades, last_close, live_closed, live_open, pos, capital, tolerance))
    explained = (out["matched_live"] + out["live_only"]) - (out["matched_backtest"] + out["backtest_only_gated"]
                                                           + out["backtest_only_other"])
    out["matched_difference"] = round(out["matched_live"] - out["matched_backtest"], 2)
    out["residual"] = round(gap_pts - explained, 2)
    return out


def matched_split(bt_entries: list, bt_trades: pd.DataFrame, last_close: pd.Series, live_closed: pd.DataFrame,
                  live_open: list, pos: dict, capital: float, tolerance: int = 5) -> dict:
    """The same-names gap split into position size and per-trade return (monitor R-59, 2026-10-23), with the
    pairs that differ most. Each backtest entry is paired with the nearest live entry of that name within
    `tolerance` trading days. size = (live dollars - backtest dollars) x the backtest's return; return = live
    dollars x (live return - backtest return); together they are the pair's P&L difference."""
    live = {}
    for r in live_closed.to_dict("records"):
        k = (str(r["symbol"]), str(r["entry_date"])[:10])
        a = live.setdefault(k, {"pnl": 0.0, "qty": 0.0, "px": float(r.get("entry_price") or 0), "exit": "", "exit_date": ""})
        a["pnl"] += float(r["pnl"])
        a["qty"] += float(r.get("qty") or 0)
        a["exit"], a["exit_date"] = str(r.get("exit_reason") or "")[:24], str(r.get("exit_date") or "")[:10]
    for s, p in live_open:
        px = float(p["entry_price"])
        live[(s, str(p["entry_date"])[:10])] = {"pnl": (float(p.get("last_price") or px) - px) * float(p["qty"]),
                                                "qty": float(p["qty"]), "px": px, "exit": "open", "exit_date": ""}
    bt = {}
    if len(bt_trades):
        for r in bt_trades.to_dict("records"):
            k = (r["symbol"], str(r["entry_date"])[:10])
            a = bt.setdefault(k, {"pnl": 0.0, "exit": "", "exit_date": ""})
            a["pnl"] += float(r["pnl"])
            a["exit"], a["exit_date"] = str(r.get("reason") or ""), str(r.get("exit_date") or "")[:10]
    size = ret = 0.0
    rows, used = [], set()
    for e in bt_entries:
        i = pos.get(pd.Timestamp(e["date"]))
        if not i:
            continue
        sig = i - 1
        near = [(abs(pos[pd.Timestamp(d)] - sig), (s, d)) for (s, d) in live
                if s == e["symbol"] and (s, d) not in used and pd.Timestamp(d) in pos and abs(pos[pd.Timestamp(d)] - sig) <= tolerance]
        if not near:
            continue
        k = min(near)[1]
        used.add(k)
        lv = live[k]
        b_usd, l_usd = float(e["price"]) * float(e["qty"]), lv["px"] * lv["qty"]
        if b_usd <= 0 or l_usd <= 0:
            continue
        b = bt.get((e["symbol"], str(pd.Timestamp(e["date"]).date()))) or {
            "pnl": (float(last_close.get(e["symbol"], e["price"])) - float(e["price"])) * float(e["qty"]), "exit": "open", "exit_date": ""}
        b_ret, l_ret = b["pnl"] / b_usd, lv["pnl"] / l_usd
        size += (l_usd - b_usd) * b_ret
        ret += l_usd * (l_ret - b_ret)
        rows.append({"symbol": e["symbol"], "diff_pts": round(100 * (lv["pnl"] - b["pnl"]) / capital, 2),
                     "size_ratio": round(l_usd / b_usd, 2),
                     "entry_px_diff_pct": round(100 * (lv["px"] / float(e["price"]) - 1), 1),
                     "live": f"{k[1]} {l_ret:+.1%} {lv['exit']} {lv['exit_date']}".strip(),
                     "backtest": f"{pd.Timestamp(e['date']).date()} {b_ret:+.1%} {b['exit']} {b['exit_date']}".strip()})
    rows.sort(key=lambda r: r["diff_pts"])
    return {"matched_size": round(100 * size / capital, 2), "matched_return": round(100 * ret / capital, 2),
            "matched_pairs": len(rows),
            "matched_size_ratio_median": float(pd.Series([r["size_ratio"] for r in rows]).median()) if rows else None,
            "matched_worst": rows[:5]}


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
    gated = {r["symbol"] for r in book.to_dict("records") if r["kind"] in ("blocked", "analyst_remove")} if len(book) else set()
    st = _state.load_state()
    trades_file = _state.TRADES_FILE
    live_closed = pd.read_csv(trades_file) if Path(trades_file).exists() else pd.DataFrame(columns=["symbol", "entry_date", "pnl"])
    live_closed = live_closed[live_closed["symbol"] != "symbol"]
    live_closed = live_closed[pd.to_datetime(live_closed["entry_date"]) >= pd.Timestamp(start)]
    live_open = [(s, p) for s, p in st["positions"].items() if p.get("filled") and p.get("entry_price")]
    gap = round(100 * (live_last / capital - bt_last / float(bt_eq.iloc[0])), 2)
    try:
        attr = attribute(log, trades, panel["close"].ffill().iloc[-1], live_closed, live_open, bought, gated,
                         panel["close"].index, capital, gap)
    except Exception as e:                       # attribution is a breakdown, never a reason the replay fails
        attr = {"error": f"{type(e).__name__}: {e}"}
    from .config import release
    rel = release(cfg)
    since = {}
    cut = pd.Timestamp(rel["since"]) if rel["name"] and rel["since"] else None
    if cut is not None and cut < pd.Timestamp(end) and (eq.index <= cut).any() and (bt_eq.index <= cut).any():
        # The same window for both: from the cut's close. The replay carries its own positions across the cut, as
        # live carries v1's - so this compares the two books under the new release, not two fresh starts.
        s_cut = spy[spy.index <= cut].iloc[-1]
        since = {"release": rel["name"], "from": rel["since"].isoformat(),
                 "live_return": round(live_last / float(eq["equity"][eq.index <= cut].iloc[-1]) - 1, 4),
                 "backtest_return": round(bt_last / float(bt_eq[bt_eq.index <= cut].iloc[-1]) - 1, 4),
                 "spy_return": round(float(spy[spy.index <= pd.Timestamp(end)].iloc[-1] / s_cut - 1), 4)}
        bench = ((cfg.get("governance") or {}).get("stop_rule") or {}).get("benchmark")
        if bench:
            # The stop rule's yardstick (governance.stop_rule): a lens, never a reason the replay fails.
            try:
                b = cached_bars(data, [bench], days, cfg, tag=f"parity-{bench.lower()}")["close"][bench].dropna()
                since["benchmark"] = bench
                since["benchmark_return"] = round(float(b[b.index <= pd.Timestamp(end)].iloc[-1]
                                                        / b[b.index <= cut].iloc[-1] - 1), 4)
            except Exception as e:
                since["benchmark_error"] = f"{type(e).__name__}: {e}"
    return {"window": f"{start} to {end}", "live_return": round(live_last / capital - 1, 4),
            "backtest_return": round(bt_last / float(bt_eq.iloc[0]) - 1, 4), "spy_return": round(spy_ret, 4),
            "equity_gap_pts": round(100 * (live_last / capital - bt_last / float(bt_eq.iloc[0])), 2),
            "live_cash_share": round(float((eq["cash"] / eq["equity"]).mean()), 3),
            "backtest_cash_share": round(float((curve["cash"] / curve["equity"]).mean()), 3),
            "entries": ent, "backtest_trades_closed": int(len(trades)), "backtest_entries": len(log),
            "since_release": since, "attribution_pts": attr}


def alarms(res: dict, cfg: dict, today: date) -> list:
    """SEV2 when live has drifted from the backtest by more than parity.equity_gap_alarm_pts since the current release
    began. The full-window gap is still reported; it mostly predates the release and alarmed every day (2026-10-23)."""
    lim = float((cfg.get("parity") or {}).get("equity_gap_alarm_pts", 5.0))
    sr = (res or {}).get("since_release")
    if sr and sr.get("live_return") is not None and sr.get("backtest_return") is not None:
        gap, span = (sr["live_return"] - sr["backtest_return"]) * 100, f"since {sr['release']} ({sr['from']} close)"
    else:
        gap, span = (res or {}).get("equity_gap_pts"), (res or {}).get("window")
    if gap is None or abs(gap) <= lim:
        return []
    return [{"key": f"parity-gap-{today:%Y-%m-%d}", "severity": "SEV2",
             "title": f"Live is {gap:+.1f} pts from the backtest {span}",
             "detail": f"full window {res['window']}: live {res['live_return']:+.1%}, backtest {res['backtest_return']:+.1%}, "
                       f"SPY {res['spy_return']:+.1%}; cash live {res['live_cash_share']:.0%} vs backtest "
                       f"{res['backtest_cash_share']:.0%}; entry overlap {res['entries']['overlap']}. "
                       f"The days that differ are in the report's Parity section."}]


def markdown(res: dict) -> list:
    if not res:
        return ["## Live vs backtest", "- no live history yet"]
    if "error" in res:
        return ["## Live vs backtest", f"- unavailable: {res['error']}"]
    e = res["entries"]
    lines = ["## Live vs backtest", f"- {res['window']}: live {res['live_return']:+.1%}, backtest {res['backtest_return']:+.1%} "
             f"(gap {res['equity_gap_pts']:+.1f} pts), SPY {res['spy_return']:+.1%}",
             *([f"- since {res['since_release']['release']} ({res['since_release']['from']} close): live "
                f"{res['since_release']['live_return']:+.1%}, backtest {res['since_release']['backtest_return']:+.1%}, "
                f"SPY {res['since_release']['spy_return']:+.1%}"] if res.get("since_release") else []),
             f"- cash: live {res['live_cash_share']:.0%}, backtest {res['backtest_cash_share']:.0%}",
             f"- entries: {e['both']} in both, {e['backtest_only']} backtest only, {e['live_only']} live only "
             f"(overlap {e['overlap']}; a name staged by both within {e.get('match_tolerance_days', 5)} trading days "
             f"counts as a match)"]
    a = res.get("attribution_pts") or {}
    if a and "error" not in a:
        lines.append(f"- the gap by cause, in points: same names {a['matched_difference']:+.1f} (live {a['matched_live']:+.1f} "
                     f"vs backtest {a['matched_backtest']:+.1f}); backtest-only kept out live {-a['backtest_only_gated']:+.1f}; "
                     f"other backtest-only {-a['backtest_only_other']:+.1f}; live-only {a['live_only']:+.1f}; "
                     f"fills, sizing and idle cash {a['residual']:+.1f}")
        if a.get("matched_pairs"):
            lines.append(f"  - same names, split: position size {a['matched_size']:+.1f}, per-trade return "
                         f"{a['matched_return']:+.1f} ({a['matched_pairs']} pairs; live positions a median "
                         f"{a['matched_size_ratio_median']:.0%} of the backtest's)")
            for r in a.get("matched_worst") or []:
                lines.append(f"    - {r['symbol']} {r['diff_pts']:+.1f}: live {r['live']} vs backtest {r['backtest']}; "
                             f"size {r['size_ratio']:.0%}, entry price {r['entry_px_diff_pct']:+.1f}%")
    if e.get("backtest_only_gated_live"):
        lines.append(f"  - {e['backtest_only_gated_live']} of the {e['backtest_only']} backtest-only names were kept out "
                     f"live by a gate the replay does not apply (a hard-filter block or an analyst veto)")
    if e.get("median_lag_days") is not None:
        lines.append(f"  - matched names were staged live a median {e['median_lag_days']:+.0f} trading days after the backtest")
    if e["backtest_only_names"]:
        lines.append(f"  - backtest only: {'; '.join(e['backtest_only_names'][-6:])}")
    if e["live_only_names"]:
        lines.append(f"  - live only: {'; '.join(e['live_only_names'][-6:])}")
    return lines

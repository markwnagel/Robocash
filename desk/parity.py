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


# A live gate explains a backtest-only entry only if it fired on the backtest's signal day (one session either side).
# Matching a name's block from any date labelled EXEL, bought by the replay on 10-22 with its report 7 days out,
# with a 10-26 "earnings in 3d" block (monitor F-45, 2026-10-30).
GATE_DAYS = 1


def gate_index(book: pd.DataFrame, pos: dict) -> dict:
    """symbol -> [(trading-day index, 'kind: reason')] for every live block and analyst veto in the shadow book."""
    out: dict = {}
    for r in book.to_dict("records") if len(book) else []:
        if r["kind"] in ("blocked", "analyst_remove"):
            i = pos.get(pd.Timestamp(r["date"]))
            if i is not None:
                out.setdefault(r["symbol"], []).append((i, f"{r['kind']}: {str(r.get('reason') or '')[:60]}"))
    return out


def gate_near(gates: dict, sym: str, i: int, days: int = GATE_DAYS) -> str | None:
    near = [(abs(j - i), why) for j, why in gates.get(sym, []) if abs(j - i) <= days]
    return min(near)[1] if near else None


def gate_class(why: str, offering_block: bool = False) -> str:
    """retired: a rule live no longer applies (the dilution block, facts.offering_block false since 2026-10-19).
    earnings: a current rule the replay applies too, so a gate here is a real mismatch (the replay blocks reports
    1-5 days out, live 0-5). unsimulated: the analyst and the headline/social blocks, which the replay never runs."""
    if why.startswith("blocked: dilutive filing"):
        return "unsimulated" if offering_block else "retired"
    if why.startswith("blocked: earnings"):
        return "earnings"
    return "unsimulated"


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
    gates = gate_index(book, pos)
    used, lags, only_bt = set(), [], []
    for i, sym in sorted(bt):
        cands = [(abs(j - i), j) for j, s in live if s == sym and (j, s) not in used and abs(j - i) <= tolerance]
        if cands:
            _, j = min(cands)
            used.add((j, sym))
            lags.append(j - i)
        else:
            only_bt.append((i, sym, gate_near(gates, sym, i)))
    # A live name re-staged on several days after a missed fill is one decision: count its first staging only.
    first_live = {}
    for j, sym in live:
        if (j, sym) not in used and not any(s == sym and abs(k - j) <= tolerance for k, s in used):
            first_live.setdefault(sym, j)
    only_live = sorted((dates[j].date().isoformat(), s) for s, j in first_live.items())
    total = len(lags) + len(only_bt) + len(only_live)
    # How much of the backtest-only side is a live gate the replay does not apply (a hard-filter block or an analyst
    # veto) on that day, as opposed to a genuinely different pick (R-50, R-53, 2026-10-21).
    gated = sum(1 for *_, why in only_bt if why)
    return {"overlap": round(len(lags) / total, 3) if total else None, "both": len(lags),
            "match_tolerance_days": tolerance, "backtest_only_gated_live": gated,
            "backtest_only": len(only_bt), "live_only": len(only_live),
            "median_lag_days": float(pd.Series(lags).median()) if lags else None,
            "backtest_only_names": [f"{dates[i].date().isoformat()} {s}" + (f" ({why})" if why else "")
                                    for i, s, why in only_bt][-10:],
            "live_only_names": [f"{d} {s}" for d, s in only_live][-10:]}


def live_positions(live_closed: pd.DataFrame, live_open: list) -> dict:
    """(symbol, entry date) -> one live position: realized P&L of every closed part plus the open rest at its mark.
    An open remainder used to replace a partial close instead of adding to it."""
    live: dict = {}
    for r in live_closed.to_dict("records"):
        k = (str(r["symbol"]), str(r["entry_date"])[:10])
        a = live.setdefault(k, {"pnl": 0.0, "qty": 0.0, "px": float(r.get("entry_price") or 0), "exit": "", "exit_date": ""})
        a["pnl"] += float(r["pnl"])
        a["qty"] += float(r.get("qty") or 0)
        a["exit"], a["exit_date"] = str(r.get("exit_reason") or "")[:24], str(r.get("exit_date") or "")[:10]
    for s, p in live_open:
        px = float(p["entry_price"])
        a = live.setdefault((s, str(p["entry_date"])[:10]), {"pnl": 0.0, "qty": 0.0, "px": px, "exit": "", "exit_date": ""})
        a["pnl"] += (float(p.get("last_price") or px) - px) * float(p["qty"])
        a["qty"] += float(p["qty"])
        a["px"], a["exit"], a["exit_date"] = a["px"] or px, "open", ""
    return live


def pair_entries(bt_entries: list, bt_trades: pd.DataFrame, last_close: pd.Series, live: dict, pos: dict,
                 tolerance: int = 5) -> tuple[list, list, set]:
    """One pairing for the whole attribution: each backtest entry, in date order, takes the nearest unused live
    position of that name within `tolerance` trading days. Returns (pairs, unpaired backtest entries), each
    backtest side valued at its result - closed P&L, or an open one marked at the last close - and the set of
    live positions used."""
    bt = {}
    if len(bt_trades):
        for r in bt_trades.to_dict("records"):
            k = (r["symbol"], str(r["entry_date"])[:10])
            a = bt.setdefault(k, {"pnl": 0.0, "exit": "", "exit_date": ""})
            a["pnl"] += float(r["pnl"])
            a["exit"], a["exit_date"] = str(r.get("reason") or ""), str(r.get("exit_date") or "")[:10]
    pairs, unpaired, used = [], [], set()
    for e in sorted((e for e in bt_entries if pos.get(pd.Timestamp(e["date"]))), key=lambda e: pd.Timestamp(e["date"])):
        sig = pos[pd.Timestamp(e["date"])] - 1
        day = str(pd.Timestamp(e["date"]).date())
        b = bt.get((e["symbol"], day)) or {
            "pnl": (float(last_close.get(e["symbol"], e["price"])) - float(e["price"])) * float(e["qty"]),
            "exit": "open", "exit_date": ""}
        b = dict(b, usd=float(e["price"]) * float(e["qty"]), px=float(e["price"]), day=day)
        near = [(abs(pos[pd.Timestamp(d)] - sig), (s, d)) for (s, d) in live
                if s == e["symbol"] and (s, d) not in used and pd.Timestamp(d) in pos
                and abs(pos[pd.Timestamp(d)] - sig) <= tolerance]
        if near:
            k = min(near)[1]
            used.add(k)
            pairs.append({"symbol": e["symbol"], "sig": sig, "bt": b, "live": dict(live[k], day=k[1])})
        else:
            unpaired.append({"symbol": e["symbol"], "sig": sig, "bt": b})
    return pairs, unpaired, used


def split_pairs(pairs: list, capital: float) -> dict:
    """The same-names gap split into position size and per-trade return (monitor R-59, 2026-10-23), with the pairs
    that differ most. size = (live dollars - backtest dollars) x the backtest's return; return = live dollars x
    (live return - backtest return). Per pair they sum to the P&L difference exactly; a pair with no recorded
    size on either side puts its whole difference in return, so the split always adds up (R-60, F-52)."""
    size = ret = 0.0
    rows = []
    for p in pairs:
        b, lv = p["bt"], p["live"]
        b_usd, l_usd = b["usd"], lv["px"] * lv["qty"]
        diff = lv["pnl"] - b["pnl"]
        if b_usd > 0 and l_usd > 0:
            b_ret, l_ret = b["pnl"] / b_usd, lv["pnl"] / l_usd
            s = (l_usd - b_usd) * b_ret
            size, ret = size + s, ret + (diff - s)
            ratio, px_diff = round(l_usd / b_usd, 2), round(100 * (lv["px"] / b["px"] - 1), 1)
            live_s, bt_s = f"{l_ret:+.1%}", f"{b_ret:+.1%}"
        else:
            ret += diff
            ratio = px_diff = None
            live_s, bt_s = f"${lv['pnl']:+.0f}", f"${b['pnl']:+.0f}"
        rows.append({"symbol": p["symbol"], "diff_pts": round(100 * diff / capital, 2), "size_ratio": ratio,
                     "entry_px_diff_pct": px_diff,
                     "live": f"{lv['day']} {live_s} {lv['exit']} {lv['exit_date']}".strip(),
                     "backtest": f"{b['day']} {bt_s} {b['exit']} {b['exit_date']}".strip()})
    rows.sort(key=lambda r: r["diff_pts"])
    ratios = [r["size_ratio"] for r in rows if r["size_ratio"] is not None]
    total, m_size = round(100 * (size + ret) / capital, 2), round(100 * size / capital, 2)
    return {"matched_size": m_size, "matched_return": round(total - m_size, 2), "matched_split_total": total,
            "matched_pairs": len(rows),
            "matched_size_ratio_median": float(pd.Series(ratios).median()) if ratios else None,
            "matched_worst": rows[:5]}


def matched_split(bt_entries: list, bt_trades: pd.DataFrame, last_close: pd.Series, live_closed: pd.DataFrame,
                  live_open: list, pos: dict, capital: float, tolerance: int = 5) -> dict:
    pairs, _, _ = pair_entries(bt_entries, bt_trades, last_close, live_positions(live_closed, live_open), pos, tolerance)
    return split_pairs(pairs, capital)


def attribute(bt_entries: list, bt_trades: pd.DataFrame, last_close: pd.Series, live_closed: pd.DataFrame,
              live_open: list, live_bought: pd.DataFrame, book: pd.DataFrame, dates: pd.DatetimeIndex, capital: float,
              gap_pts: float, tolerance: int = 5, offering_block: bool = False) -> dict:
    """The gap in percentage points, by cause (monitor R-58, 2026-10-22), from one pairing of backtest entries with
    live positions (pair_entries):
      matched                pairs: live P&L vs the backtest's, split into size and return
      backtest_only_unfilled live staged the name within `tolerance` days but the order never filled
      backtest_only_gated_*  a live gate kept it out that day: retired rule / earnings / unsimulated (gate_class)
      backtest_only_other    a different pick
      live_only              live positions with no backtest entry near them
    What the entries do not explain - fills, slippage, idle cash - is the residual."""
    pos = {d: i for i, d in enumerate(dates)}
    live = live_positions(live_closed, live_open)
    pairs, unpaired, used = pair_entries(bt_entries, bt_trades, last_close, live, pos, tolerance)
    # Staged orders that never became a position: no live entry of that name within `tolerance` days of staging.
    filled = {}
    for s, d in live:
        if pd.Timestamp(d) in pos:
            filled.setdefault(s, []).append(pos[pd.Timestamp(d)])
    staged = {}
    for r in live_bought.to_dict("records"):
        j = pos.get(pd.Timestamp(r["date"]))
        if j is not None and not any(abs(f - j) <= tolerance for f in filled.get(r["symbol"], [])):
            staged.setdefault(r["symbol"], []).append(j)
    gates = gate_index(book, pos)
    pts = {"matched_live": sum(p["live"]["pnl"] for p in pairs), "matched_backtest": sum(p["bt"]["pnl"] for p in pairs),
           "backtest_only_unfilled": 0.0, "backtest_only_gated_retired": 0.0, "backtest_only_gated_earnings": 0.0,
           "backtest_only_gated_unsimulated": 0.0, "backtest_only_other": 0.0,
           "live_only": sum(a["pnl"] for k, a in live.items() if k not in used)}
    for u in unpaired:
        why = gate_near(gates, u["symbol"], u["sig"])
        if any(abs(j - u["sig"]) <= tolerance for j in staged.get(u["symbol"], [])):
            pts["backtest_only_unfilled"] += u["bt"]["pnl"]
        elif why:
            pts[f"backtest_only_gated_{gate_class(why, offering_block)}"] += u["bt"]["pnl"]
        else:
            pts["backtest_only_other"] += u["bt"]["pnl"]
    out = {k: round(100 * v / capital, 2) for k, v in pts.items()}
    out["backtest_only_gated"] = round(sum(out[f"backtest_only_gated_{c}"] for c in ("retired", "earnings", "unsimulated")), 2)
    out.update(split_pairs(pairs, capital))
    out["matched_unfilled"] = sum(1 for u in unpaired
                                  if any(abs(j - u["sig"]) <= tolerance for j in staged.get(u["symbol"], [])))
    bt_side = out["matched_backtest"] + out["backtest_only_unfilled"] + out["backtest_only_gated"] + out["backtest_only_other"]
    out["matched_difference"] = out.pop("matched_split_total")      # the same pairs, so the split sums to it
    out["residual"] = round(gap_pts - ((out["matched_live"] + out["live_only"]) - bt_side), 2)
    return out


def run(cfg: dict, data=None, start: date | None = None, end: date | None = None) -> dict:
    """Replay and compare. {} before there is a live history."""
    import backtest as B
    from .data import cached_bars
    from .regime import vix_series
    from .config import account_since
    eq = live_equity()
    acc = account_since(cfg)
    if acc is not None and not eq.empty:
        eq = eq[eq.index >= pd.Timestamp(acc)]        # an earlier account's briefs are not this account's record
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
    st = _state.load_state()
    trades_file = _state.TRADES_FILE
    live_closed = pd.read_csv(trades_file) if Path(trades_file).exists() else pd.DataFrame(columns=["symbol", "entry_date", "pnl"])
    live_closed = live_closed[live_closed["symbol"] != "symbol"]
    live_closed = live_closed[pd.to_datetime(live_closed["entry_date"]) >= pd.Timestamp(start)]
    live_open = [(s, p) for s, p in st["positions"].items() if p.get("filled") and p.get("entry_price")]
    gap = round(100 * (live_last / capital - bt_last / float(bt_eq.iloc[0])), 2)
    try:
        attr = attribute(log, trades, panel["close"].ffill().iloc[-1], live_closed, live_open, bought, book,
                         panel["close"].index, capital, gap,
                         offering_block=bool((cfg.get("facts") or {}).get("offering_block")))
    except Exception as e:                       # attribution is a breakdown, never a reason the replay fails
        attr = {"error": f"{type(e).__name__}: {e}"}
    from .config import measured_from, release
    rel = release(cfg)
    since = {}
    frm = measured_from(cfg) if rel["name"] else None
    cut = pd.Timestamp(frm) if frm else None
    if cut is not None and cut < pd.Timestamp(end) and (eq.index <= cut).any() and (bt_eq.index <= cut).any():
        # The same window for both: from the cut's close. The replay carries its own positions across the cut, as
        # live carries v1's - so this compares the two books under the new release, not two fresh starts.
        s_cut = spy[spy.index <= cut].iloc[-1]
        since = {"release": rel["name"], "from": frm.isoformat(),
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
        g = {c: -a.get(f"backtest_only_gated_{c}", 0.0) for c in ("retired", "earnings", "unsimulated")}
        lines.append(f"- the gap by cause, in points: same names {a['matched_difference']:+.1f} (live {a['matched_live']:+.1f} "
                     f"vs backtest {a['matched_backtest']:+.1f}); backtest-only kept out live {-a['backtest_only_gated']:+.1f}; "
                     f"staged live but never filled {-a.get('backtest_only_unfilled', 0.0):+.1f}; "
                     f"other backtest-only {-a['backtest_only_other']:+.1f}; live-only {a['live_only']:+.1f}; "
                     f"fills, sizing and idle cash {a['residual']:+.1f}")
        if a['backtest_only_gated']:
            lines.append(f"  - kept out live, by gate: retired rule (dilution block) {g['retired']:+.1f}; earnings block "
                         f"(the replay applies it too: a real mismatch) {g['earnings']:+.1f}; analyst or headline/social "
                         f"block (not simulated) {g['unsimulated']:+.1f}")
        if a.get("matched_pairs"):
            med = a.get("matched_size_ratio_median")
            lines.append(f"  - same names, split: position size {a['matched_size']:+.1f}, per-trade return "
                         f"{a['matched_return']:+.1f} ({a['matched_pairs']} filled pairs"
                         + (f"; live positions a median {med:.0%} of the backtest's" if med is not None else "")
                         + (f"; {a['matched_unfilled']} more staged live but never filled" if a.get("matched_unfilled") else "")
                         + ")")
            for r in a.get("matched_worst") or []:
                size = (f"; size {r['size_ratio']:.0%}, entry price {r['entry_px_diff_pct']:+.1f}%"
                        if r.get("size_ratio") is not None else "; size not recorded")
                lines.append(f"    - {r['symbol']} {r['diff_pts']:+.1f}: live {r['live']} vs backtest {r['backtest']}{size}")
    if e.get("backtest_only_gated_live"):
        lines.append(f"  - {e['backtest_only_gated_live']} of the {e['backtest_only']} backtest-only names were kept out "
                     f"live by a gate that fired on the backtest's signal day (a hard-filter block or an analyst veto)")
    if e.get("median_lag_days") is not None:
        lines.append(f"  - matched names were staged live a median {e['median_lag_days']:+.0f} trading days after the backtest")
    if e["backtest_only_names"]:
        lines.append(f"  - backtest only: {'; '.join(e['backtest_only_names'][-6:])}")
    if e["live_only_names"]:
        lines.append(f"  - live only: {'; '.join(e['live_only_names'][-6:])}")
    return lines

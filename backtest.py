#!/usr/bin/env python
"""Daily-bar backtest of the live logic (signals, sizing, brackets, time/decay exits, kill switch).

  python backtest.py --start 2023-09-11 --end 2026-09-11
  python backtest.py --start 2023-09-11 --end 2026-09-11 --risk-per-trade 0.0125 --stagger 6

Runs read a frozen universe snapshot (state/history/universe_bt.json) and fetch bars only through --end,
so the same command gives the same answer next month. --stagger N reruns the window from N start dates a
month apart, variant against the untouched config.yaml, because a single path is dominated by when the
kill switch happens to fire.

Caveats: the snapshot is a list of names that were liquid when it was built (survivorship bias inflates
results); no analyst veto; fills are approximated from daily OHLC; a same-day stop and target both touched
counts as the stop.
"""
from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import json
import logging
import math
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from desk.config import alpaca_keys, load_config, require_paper
from desk.data import build_universe, fetch_daily_bars
from desk.portfolio import (decay_exit, entry_limit, pyramid_due, select_entries, shrink_qty, size_position,
                            trail_stop)
from desk.regime import regime_params, regime_scale_series, regime_state_series, vix_series
from desk.signals import compute_panels, panel_key, rerank, snapshot
from desk.state import STATE_DIR, atomic_pickle, atomic_write_text
from desk import trials as T
from desk.trials import BACKTEST_FACTS, STRATEGY_SECTIONS, current_config_id, strategy_fingerprint  # noqa: F401
from desk.evidence import reference_from_curve

log = logging.getLogger("desk.backtest")
STALE_EXIT_DAYS = 5        # sessions without a bar before a holding is closed at its last price


def _report_between(dates: list, after, upto) -> bool:
    """A report dated in (after, upto]; `dates` sorted. Replaces a scan of every date for every candidate
    on every day, which was a tenth of a run."""
    k = bisect.bisect_right(dates, after)
    return k < len(dates) and dates[k] <= upto


def _aligned(frame: pd.DataFrame, like: pd.DataFrame):
    """frame's values on like's rows and columns, as an array (a view when they already match)."""
    if not (frame.index.equals(like.index) and frame.columns.equals(like.columns)):
        frame = frame.reindex(index=like.index, columns=like.columns)
    return frame.to_numpy()


def run_backtest(panel: dict, cfg: dict, start: pd.Timestamp, end: pd.Timestamp, capital: float,
                 slippage_bps: float, halt_cooldown: int, regime: pd.Series | None = None,
                 earnings: pd.DataFrame | None = None, insider: pd.DataFrame | None = None,
                 regime_state: pd.Series | None = None, panels: dict | None = None,
                 members: pd.DataFrame | None = None, entry_rng: np.random.Generator | None = None,
                 veto: dict | None = None) -> tuple:
    """panels: precomputed compute_panels(panel, cfg['signals']) - pass it when running the same panel and
    signal settings many times (--stagger), since it is the bulk of each run's cost.

    Used by robustness.py:
      members    date x symbol bool, the point-in-time universe. New entries and ranks come from that
                 day's members plus holdings (live ranks universe + holdings); None = every panel column.
      entry_rng  pick entries in random order among eligible names instead of by score - the null test.
                 Exits still use the real ranks, so only the choice of what to buy changes.
      veto       {"rng", "remove", "reduce", "factor", "half"}: independently per entry, dropped with
                 probability remove (no refill, as live), cut to `factor` with probability reduce, and
                 halved (the implied-vol rule) with probability half."""
    scfg, pcfg, ecfg, rcfg, fcfg = cfg["signals"], cfg["portfolio"], cfg["exits"], cfg["risk"], cfg["facts"]
    gcfg = cfg.get("regime", {})
    slip = slippage_bps / 10000
    panels = panels if panels is not None else compute_panels(panel, scfg)
    op, hi, lo, cl = panel["open"], panel["high"], panel["low"], panel["close"]
    # Last known close (past data only). A held name with a missing bar is still worth what it last traded
    # at: dropping it from equity fakes a loss that can trip the kill switch, and closing at NaN turns cash
    # into NaN for the rest of the run.
    cf = cl.ffill()
    atr_panel = panels["atr"]
    # Plain arrays for the per-position reads below; .at[date, symbol] lookups were a tenth of the run time.
    col = {s: j for j, s in enumerate(cl.columns)}
    OP, HI, LO, CL, CF, ATR = (_aligned(f, cl) for f in (op, hi, lo, cl, cf, atr_panel))
    mem = None
    if members is not None:
        mem = members.reindex(index=cl.index, columns=cl.columns, fill_value=False).to_numpy(dtype=bool)
    dates = cl.index
    warm = scfg["momentum_lookback_days"] + scfg["trend_ma_days"]

    earn_by_sym = {}
    if earnings is not None and len(earnings):
        for r in earnings.dropna(subset=["date"]).itertuples():
            if str(r.date).strip():
                earn_by_sym.setdefault(str(r.symbol).upper(), []).append(pd.Timestamp(r.date))
        for v in earn_by_sym.values():
            v.sort()
    pre_earn = int(ecfg.get("earnings_exit_days_before") or 0)  # trading days, not calendar days
    ins = None
    if insider is not None and len(insider):
        daily = (insider.assign(date=pd.to_datetime(insider["date"]))
                 .groupby(["date", "symbol"])["net_usd"].sum().unstack(fill_value=0.0).resample("D").sum())
        ins = daily.rolling(30, min_periods=1).sum().reindex(dates, method="ffill").fillna(0.0)
    earn_window = pd.Timedelta(days=fcfg["exclude_earnings_within_days"])

    cash, positions, pending, trades, curve = capital, {}, [], [], []
    hwm, ath, last_equity, halted_until = capital, capital, capital, None
    cum_floor = rcfg.get("max_cumulative_drawdown_pct")
    max_new = pcfg.get("max_new_entries_per_day")
    trail = float(ecfg.get("trail_atr_multiple") or 0)
    pyr_n = int(pcfg.get("pyramid_max_adds") or 0)
    stop_mult = ecfg["stop_atr_multiple"]
    trim_above = float(pcfg.get("trim_above_pct") or 0)
    trim_to = float(pcfg.get("trim_to_pct") or pcfg["max_position_pct"])
    tp_r = float(ecfg.get("take_profit_r_multiple") or 0)

    def sell(sym, qty, px, d, reason):
        """Sell `qty` shares of a holding: all of them closes it, fewer is a trim that leaves the rest open with
        the same stop and target. One trade row per sale, its risk the sold shares' share of the position's."""
        nonlocal cash
        p = positions[sym]
        frac = qty / p["qty"]
        cash += qty * px
        risk = p.get("realized_risk")
        trades.append({"symbol": sym, "entry_date": p["entry_date"].date(), "exit_date": d.date(),
                       "entry_price": p["entry_price"], "exit_price": px, "qty": qty,
                       "pnl": (px - p["entry_price"]) * qty, "pnl_pct": px / p["entry_price"] - 1,
                       "hold_days": (d - p["entry_date"]).days, "reason": reason,
                       "intended_risk": p.get("intended_risk"),
                       "realized_risk": risk * frac if risk is not None else None,
                       "adds": p.get("adds", 0), "final_stop": p["stop"],
                       "spanned_earnings": _report_between(earn_by_sym.get(sym, []), p["entry_date"], d)})
        if qty >= p["qty"]:
            positions.pop(sym)
        else:
            p["qty"] -= qty
            if risk is not None:
                p["realized_risk"] = risk * (1 - frac)

    def close_pos(sym, px, d, reason):
        sell(sym, positions[sym]["qty"], px, d, reason)

    def last(sym):
        """Last known close (forward-filled) on today's row."""
        return CF[i, col[sym]]

    def mark():
        return cash + sum(q["qty"] * last(s) for s, q in positions.items() if not np.isnan(last(s)))

    for i in range(warm, len(dates)):
        d = dates[i]
        if d < start:
            continue
        if d > end:
            break
        for e in pending:
            sym = e["symbol"]
            o, l = OP[i, col[sym]], LO[i, col[sym]]
            is_add = bool(e.get("add"))
            if np.isnan(o) or is_add != (sym in positions):
                continue
            if o <= e["limit"]:
                fill = o * (1 + slip)
            elif l <= e["limit"]:
                fill = e["limit"]
            else:
                continue
            qty = min(e["qty"], math.floor(cash / fill))
            if qty < 1:
                continue
            cash -= qty * fill
            if is_add:
                p = positions[sym]
                p["entry_price"] = (p["entry_price"] * p["qty"] + fill * qty) / (p["qty"] + qty)
                p["qty"] += qty
                p["adds"] += 1
                p["last_add_price"] = fill
                continue
            stop = round(fill - stop_mult * e["atr"], 2)
            target = round(fill + tp_r * (fill - stop), 2) if tp_r > 0 else None
            positions[sym] = {"qty": qty, "entry_price": fill, "entry_date": d, "stop": stop, "target": target,
                              "intended_risk": e.get("intended_risk"), "realized_risk": qty * (fill - stop),
                              "high_close": fill, "adds": 0, "last_add_price": fill, "initial_risk_per_share": fill - stop}
        pending = []

        for sym in list(positions):
            p = positions[sym]
            j = col[sym]
            o, h, l, c = OP[i, j], HI[i, j], LO[i, j], CL[i, j]
            if np.isnan(c):
                # No bar: a halt, or the name stopped trading (merger, bankruptcy). Waiting forever would
                # hold the slot and the last price for the rest of the run, so after a week out it goes at
                # the last close. For a failure that price flatters the result; robustness.py lists these.
                p["missing"] = p.get("missing", 0) + 1
                if p["missing"] >= STALE_EXIT_DAYS:
                    close_pos(sym, last(sym) * (1 - slip), d, "delisted")
                continue
            p["missing"] = 0
            if o <= p["stop"]:
                close_pos(sym, o * (1 - slip), d, "stop-gap")
            elif l <= p["stop"]:
                close_pos(sym, p["stop"] * (1 - slip), d, "stop")
            elif p["target"] is not None and o >= p["target"]:
                close_pos(sym, o, d, "target-gap")
            elif p["target"] is not None and h >= p["target"]:
                close_pos(sym, p["target"], d, "target")
            elif (d - p["entry_date"]).days >= ecfg["max_hold_days"]:
                close_pos(sym, c * (1 - slip), d, "time")
        if trail:
            # Ratcheted on the close, so like the live evening job it protects from the next session on.
            for sym, p in positions.items():
                c, a = CL[i, col[sym]], ATR[i, col[sym]]
                if np.isnan(c) or np.isnan(a):
                    continue
                p["high_close"] = max(p["high_close"], c)
                p["stop"] = trail_stop(p["stop"], p["high_close"], a, ecfg)

        equity = mark()
        hwm = max(hwm, equity)
        ath = max(ath, equity)
        drawdown = 1 - equity / hwm
        allow = True
        if halted_until is not None and i < halted_until:
            allow = False
        elif cum_floor and 1 - equity / ath >= cum_floor:
            # Hard floor: flat for three cooldowns, then the all-time high is re-baselined - the closest
            # stand-in for a human running --resume --reset-hwm after a long pause.
            for sym in list(positions):
                close_pos(sym, last(sym) * (1 - slip), d, "hard-floor")
            equity = cash
            hwm = ath = equity
            halted_until = i + 3 * halt_cooldown
            allow = False
        elif drawdown >= rcfg["max_drawdown_pct"]:
            for sym in list(positions):
                close_pos(sym, last(sym) * (1 - slip), d, "kill-switch")
            equity = cash
            hwm = equity
            halted_until = i + halt_cooldown
            allow = False
        if equity / last_equity - 1 <= -rcfg["daily_loss_halt_pct"]:
            allow = False

        cols = None
        if mem is not None:
            cols = cl.columns[mem[i]].union(pd.Index(list(positions)))
        snap = snapshot(panels, i, scfg, cols)
        if pre_earn:
            # Window measured in bars so a Monday report is caught on the preceding Thursday/Friday;
            # calendar-day arithmetic would look at the weekend and miss it.
            window_end = dates[min(i + pre_earn, len(dates) - 1)]
            for sym in list(positions):
                if _report_between(earn_by_sym.get(sym, []), d, window_end):
                    close_pos(sym, last(sym) * (1 - slip), d, "pre-earnings")
            equity = mark()
        for sym in list(positions):
            # "defer" (no complete bar today) keeps the holding, as the live run does
            if decay_exit(snap, sym, (d - positions[sym]["entry_date"]).days, ecfg) == "decay":
                close_pos(sym, last(sym) * (1 - slip), d, "decay")
                equity = mark()

        if trim_above and positions:
            # A winner at or above trim_above_pct of equity is sold back to trim_to_pct, as the live desk does
            # (there at the next open; here at the close, like the other non-stop exits).
            equity = mark()
            for sym in list(positions):
                px = last(sym)
                if np.isnan(px) or equity <= 0 or positions[sym]["qty"] * px / equity < trim_above:
                    continue
                keep = math.floor(equity * trim_to / px)
                if 1 <= keep < positions[sym]["qty"]:
                    sell(sym, positions[sym]["qty"] - keep, px * (1 - slip), d, "trim")
            equity = mark()

        if regime_state is not None:
            rp = regime_params(str(regime_state.get(d, "neutral")), pcfg, gcfg)
        else:
            scale = float(regime.get(d, 1.0)) if regime is not None else 1.0
            rp = {"target_positions": max(1, int(round(pcfg["target_positions"] * scale))),
                  "max_position_pct": pcfg["max_position_pct"], "liquidate": False}
        if rp["liquidate"] and positions:
            for sym in list(positions):
                close_pos(sym, last(sym) * (1 - slip), d, "regime")
            equity = cash
        day_pcfg = dict(pcfg, max_position_pct=rp["max_position_pct"])

        if allow and i + 1 < len(dates):
            slots = rp["target_positions"] - len(positions)
            esnap = snap
            if entry_rng is not None:
                esnap = snap.copy()
                el = esnap["eligible"].to_numpy(dtype=bool)
                esnap.loc[el, "score"] = entry_rng.random(int(el.sum()))
                esnap.loc[el, "rank"] = esnap.loc[el, "score"].rank(ascending=False, method="first")
                esnap = esnap.sort_values("rank")
            if earn_by_sym or ins is not None:
                blocked, tilt = {}, {}
                block_end = d + earn_window
                for sym in esnap.index[esnap["eligible"].values][: fcfg["candidate_pool"] * 2]:
                    if _report_between(earn_by_sym.get(sym, []), d, block_end):
                        blocked[sym] = "earnings"
                    if ins is not None and sym in ins.columns and ins.at[d, sym] >= fcfg["insider_min_net_usd"]:
                        tilt[sym] = float(fcfg["insider_tilt_weight"])
                esnap = rerank(esnap, tilt, blocked)
            rets = panels["returns"].iloc[max(0, i + 1 - pcfg["correlation_days"]): i + 1]
            picks = select_entries(esnap, rets, list(positions), slots, pcfg)
            if max_new:
                picks = picks[:max_new]
            cash_avail = cash - equity * pcfg["cash_floor_pct"]
            for sym in picks:
                row = esnap.loc[sym]
                sz = size_position(equity, cash_avail, float(row["close"]), float(row["atr"]), day_pcfg, ecfg)
                if not sz:
                    continue
                if veto:
                    r = veto["rng"]
                    if r.random() < veto["remove"]:
                        continue
                    cut = ((veto["factor"] if r.random() < veto["reduce"] else 1.0)
                           * (0.5 if r.random() < veto["half"] else 1.0))
                    if cut < 1.0:
                        q = shrink_qty(sz["qty"], cut)
                        sz = {**sz, "qty": q, "notional": q * float(row["close"])}
                cash_avail -= sz["notional"]
                pending.append({"symbol": sym, "qty": sz["qty"], "limit": entry_limit(float(row["close"])), "atr": float(row["atr"]),
                                "intended_risk": equity * pcfg["risk_per_trade_pct"] if pcfg.get("sizing") == "risk_parity" else None})
            if pyr_n:
                # Adds are sized like a fresh unit at today's price and capped by the position limit;
                # they share the position's (trailed) stop and do not count against entry pacing.
                for sym, p in positions.items():
                    if sym not in snap.index:
                        continue
                    row = snap.loc[sym]
                    px, a = float(row["close"]), float(row["atr"])
                    if np.isnan(a) or not pyramid_due(p, px, pcfg):
                        continue
                    sz = size_position(equity, cash_avail, px, a, day_pcfg, ecfg)
                    if not sz:
                        continue
                    room = equity * day_pcfg["max_position_pct"] - p["qty"] * px
                    q = min(sz["qty"], int(room // px)) if room > 0 else 0
                    if q < 1:
                        continue
                    cash_avail -= q * px
                    pending.append({"symbol": sym, "qty": q, "limit": entry_limit(px), "atr": a, "add": True})

        curve.append((d, equity, len(positions), cash))
        last_equity = equity

    curve = pd.DataFrame(curve, columns=["date", "equity", "n_positions", "cash"]).set_index("date")
    return curve, pd.DataFrame(trades)


def curve_stats(curve: pd.DataFrame) -> dict:
    eq = curve["equity"]
    ret = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    return {"total": total, "cagr": (1 + total) ** (1 / years) - 1, "maxdd": (eq / eq.cummax() - 1).min(),
            "sharpe": ret.mean() / ret.std() * math.sqrt(252) if ret.std() > 0 else float("nan"), "years": years}


def summarize(curve: pd.DataFrame, trades: pd.DataFrame, spy: pd.Series | None) -> str:
    eq = curve["equity"]
    st = curve_stats(curve)
    lines = [
        f"Period {eq.index[0].date()} -> {eq.index[-1].date()} ({st['years']:.2f}y)",
        f"Start ${eq.iloc[0]:,.0f}  End ${eq.iloc[-1]:,.0f}  Total {st['total']:+.1%}  CAGR {st['cagr']:+.1%}",
        f"Max drawdown {st['maxdd']:.1%}  Sharpe {st['sharpe']:.2f}  Avg positions {curve['n_positions'].mean():.1f}",
    ]
    if spy is not None and len(spy) > 1:
        s = spy.reindex(eq.index).ffill().dropna()
        lines.append(f"SPY buy-and-hold same period: {s.iloc[-1] / s.iloc[0] - 1:+.1%}")
    if len(trades):
        wins = (trades["pnl"] > 0).mean()
        lines.append(f"Trades {len(trades)}  win rate {wins:.0%}  avg {trades['pnl_pct'].mean():+.2%}  "
                     f"avg win {trades.loc[trades.pnl > 0, 'pnl_pct'].mean():+.2%}  avg loss {trades.loc[trades.pnl <= 0, 'pnl_pct'].mean():+.2%}  "
                     f"avg hold {trades['hold_days'].mean():.0f}d")
        lines.append("Exit reasons: " + ", ".join(f"{k} {v}" for k, v in trades["reason"].value_counts().items()))
        if "final_stop" in trades:
            locked = trades[trades["reason"].isin(["stop", "stop-gap"]) & (trades["pnl"] > 0)]
            if len(locked):
                lines.append(f"Stops hit above entry (gains locked by the trail): {len(locked)}  mean {locked['pnl_pct'].mean():+.2%}")
        if "adds" in trades and (trades["adds"] > 0).any():
            a = trades[trades["adds"] > 0]
            lines.append(f"Pyramided positions: {len(a)} of {len(trades)}  mean {a['pnl_pct'].mean():+.2%}  vs un-added {trades.loc[trades['adds'] == 0, 'pnl_pct'].mean():+.2%}")
    if len(trades) and "realized_risk" in trades and trades["intended_risk"].notna().any():
        r = (trades["realized_risk"] / trades["intended_risk"]).dropna()
        lines.append(f"Realized/intended risk per entry: mean {r.mean():.2f}  p10 {r.quantile(0.1):.2f}  p90 {r.quantile(0.9):.2f}  (1.00 = risk parity holding exactly)")
    if len(trades) and "spanned_earnings" in trades and trades["spanned_earnings"].any():
        sp = trades[trades["spanned_earnings"]]
        lines.append(f"Trades that held through an earnings date: {len(sp)}  mean {sp['pnl_pct'].mean():+.2%}  median {sp['pnl_pct'].median():+.2%}  worst {sp['pnl_pct'].min():+.1%}")
    lines.append("NOTE: survivorship-biased universe, no analyst veto, OHLC fill approximation. Treat as an upper bound.")
    return "\n".join(lines)


UNIVERSE_DIR = "history"   # under state/: backtest inputs that must not drift between runs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=(date.today() - timedelta(days=730)).isoformat())
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--capital", type=float, default=None)
    ap.add_argument("--slippage-bps", type=float, default=T.DEFAULT_SLIPPAGE_BPS)
    ap.add_argument("--halt-cooldown-days", type=int, default=T.DEFAULT_HALT_COOLDOWN,
                    help="trading days to stay flat after the kill switch")
    ap.add_argument("--universe", default=None,
                    help="universe snapshot under state/history/ (default universe_bt.json)")
    ap.add_argument("--rebuild-universe", action="store_true",
                    help="rebuild the snapshot from current liquidity. Moves every result - do it deliberately, "
                         "then re-run the baseline before comparing anything")
    ap.add_argument("--refetch-prices", action="store_true",
                    help="re-download the frozen price history for this universe/window (moves results)")
    ap.add_argument("--stagger", type=int, default=0,
                    help="run N windows, start shifted one month each, this config vs untouched config.yaml")
    ap.add_argument("--sizing", choices=["risk_parity", "notional"], default=None, help="override portfolio.sizing")
    ap.add_argument("--risk-per-trade", type=float, default=None, help="override portfolio.risk_per_trade_pct")
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    ap.add_argument("--max-corr-peers", type=int, default=None, help="override portfolio.max_correlated_peers")
    ap.add_argument("--max-new-per-day", type=int, default=None, help="override portfolio.max_new_entries_per_day (0 = unlimited)")
    ap.add_argument("--cum-floor", type=float, default=None, help="override risk.max_cumulative_drawdown_pct (0 = off)")
    ap.add_argument("--earnings-exit-days", type=int, default=None, help="override exits.earnings_exit_days_before (0 = off)")
    ap.add_argument("--min-shares", type=int, default=None, help="override portfolio.min_shares_per_position")
    ap.add_argument("--earnings-block-days", type=int, default=None, help="override facts.exclude_earnings_within_days")
    ap.add_argument("--trail-atr", type=float, default=None, help="override exits.trail_atr_multiple (0 = off)")
    ap.add_argument("--take-profit", type=float, default=None, help="override exits.take_profit_r_multiple (0 = no target)")
    ap.add_argument("--pyramid-adds", type=int, default=None, help="override portfolio.pyramid_max_adds (0 = off)")
    ap.add_argument("--pyramid-trigger-r", type=float, default=None, help="override portfolio.pyramid_trigger_r")
    ap.add_argument("--ladder", action="store_true", help="regime.ladder = true (strong/neutral/off)")
    ap.add_argument("--off-liquidate", action="store_true", help="regime.off_liquidate = true (cash below the MA)")
    ap.add_argument("--max-atr-pct", type=float, default=None, help="override signals.max_atr_pct")
    ap.add_argument("--target-positions", type=int, default=None, help="override portfolio.target_positions")
    ap.add_argument("--max-position-pct", type=float, default=None, help="override portfolio.max_position_pct")
    ap.add_argument("--min-dollar-volume", type=float, default=None, help="override universe.min_avg_dollar_volume (own snapshot)")
    ap.add_argument("--max-symbols", type=int, default=None, help="override universe.max_symbols (own snapshot)")
    return ap


def apply_overrides(base: dict, args) -> dict:
    """config.yaml plus the CLI overrides, as a copy: --stagger runs the untouched base next to it."""
    cfg = copy.deepcopy(base)
    p, x, r, g = cfg["portfolio"], cfg["exits"], cfg["risk"], cfg["regime"]
    if args.sizing:
        p["sizing"] = args.sizing
    if args.risk_per_trade is not None:
        p["risk_per_trade_pct"] = args.risk_per_trade
    if args.max_corr_peers is not None:
        p["max_correlated_peers"] = args.max_corr_peers
    if args.max_new_per_day is not None:
        p["max_new_entries_per_day"] = args.max_new_per_day or None
    if args.cum_floor is not None:
        r["max_cumulative_drawdown_pct"] = args.cum_floor or None
    if args.earnings_exit_days is not None:
        x["earnings_exit_days_before"] = args.earnings_exit_days
    if args.min_shares is not None:
        p["min_shares_per_position"] = args.min_shares
    if args.earnings_block_days is not None:
        cfg["facts"]["exclude_earnings_within_days"] = args.earnings_block_days
    if args.trail_atr is not None:
        x["trail_atr_multiple"] = args.trail_atr
    if args.take_profit is not None:
        x["take_profit_r_multiple"] = args.take_profit
    if args.pyramid_adds is not None:
        p["pyramid_max_adds"] = args.pyramid_adds
    if args.pyramid_trigger_r is not None:
        p["pyramid_trigger_r"] = args.pyramid_trigger_r
    if args.ladder:
        g["ladder"] = True
    if args.off_liquidate:
        g["off_liquidate"] = True
    if args.max_atr_pct is not None:
        cfg["signals"]["max_atr_pct"] = args.max_atr_pct
    if args.target_positions is not None:
        p["target_positions"] = args.target_positions
    if args.max_position_pct is not None:
        p["max_position_pct"] = args.max_position_pct
    if args.min_dollar_volume is not None:
        cfg["universe"]["min_avg_dollar_volume"] = args.min_dollar_volume
    if args.max_symbols is not None:
        cfg["universe"]["max_symbols"] = args.max_symbols
    return cfg


def frozen_universe(trading, data, cfg: dict, name: str, rebuild: bool = False) -> tuple:
    """(symbols, built_date, path). Never the live universe.json: that one rebuilds weekly, and the rebuild
    on 2026-09-21 moved the same three-year baseline from +247% to +125%. A result nobody can reproduce
    cannot be compared with anything, so the snapshot only changes when --rebuild-universe asks for it."""
    path = Path(cfg["_root"]) / "state" / UNIVERSE_DIR / name
    if path.exists() and not rebuild:
        u = json.loads(path.read_text())
        return list(u["symbols"]), u.get("date", "unknown"), path
    syms = build_universe(trading, data, cfg, force=True, cache_name=f"{UNIVERSE_DIR}/{name}")
    return syms, date.today().isoformat(), path


def universe_id(symbols: list) -> str:
    return hashlib.sha1(",".join(sorted(symbols)).encode()).hexdigest()[:8]


def overrides_of(args) -> dict:
    """The CLI settings that differ from the defaults - a readable label for the variant."""
    skip = {"start", "end", "tag", "capital", "stagger", "universe", "rebuild_universe", "refetch_prices"}
    defaults = vars(build_parser().parse_args([]))
    return {k: v for k, v in vars(args).items() if k not in skip and v != defaults.get(k)}


def writes_live_reference(args, uname: str) -> bool:
    """Only config.yaml as it runs, on the default snapshot and capital. --universe and --capital are not
    overrides (they leave config_id alone), so a run on another universe used to replace the email's
    reference without its staleness check noticing."""
    return not overrides_of(args) and uname == "universe_bt.json" and not args.capital


def data_end(end: pd.Timestamp, now: datetime | None = None) -> datetime:
    """Bars through the requested end date, not through today, so the window does not grow a new tail
    every time the command is re-run. Free-tier SIP data must end at least 15 minutes in the past."""
    now = now or datetime.now(timezone.utc)
    want = datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=1)
    return min(want, now - timedelta(minutes=16))


def frozen_prices(data, symbols: list, days: int, cfg: dict, end: pd.Timestamp, refetch: bool = False) -> tuple:
    """(panel, path). Price history saved beside the universe snapshot on first use and reused on every
    later run. Re-downloading re-exposes a run to feed gaps - one blank TEVA moved the baseline 19 points -
    so a frozen universe needs frozen prices too. Lives outside the pruned cache; --refetch-prices replaces it."""
    folder = Path(cfg["_root"]) / "state" / UNIVERSE_DIR / "prices"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"panel_{universe_id(symbols)}_{len(symbols)}_{days}d_to{end.date()}.pkl"
    if path.exists() and not refetch:
        return pd.read_pickle(path), path
    panel = fetch_daily_bars(data, symbols, days, cfg["data"]["feed"], end=data_end(end))
    atomic_pickle(panel, path)
    return panel, path


def regime_inputs(spy: pd.Series | None, vix: pd.Series | None, cfg: dict) -> tuple:
    if spy is None or not cfg["regime"]["enabled"]:
        return None, None
    return regime_scale_series(spy, vix, cfg["regime"]), regime_state_series(spy, vix, cfg["regime"])


def compare_windows(rows: list) -> str:
    """rows: [{'start', 'base': curve_stats, 'variant': curve_stats}]. Reports how often the variant wins
    rather than one number: a single window is mostly a statement about kill-switch timing."""
    n = len(rows)
    diffs = [r["variant"]["total"] - r["base"]["total"] for r in rows]
    won_ret = sum(d > 0 for d in diffs)
    won_dd = sum(r["variant"]["maxdd"] >= r["base"]["maxdd"] for r in rows)
    won_sh = sum(r["variant"]["sharpe"] > r["base"]["sharpe"] for r in rows)
    lines = [f"{'start':<12}{'base total':>11}{'DD':>8}{'Sharpe':>8}   {'variant':>9}{'DD':>8}{'Sharpe':>8}   {'diff':>8}"]
    for r, d in zip(rows, diffs):
        b, v = r["base"], r["variant"]
        lines.append(f"{str(r['start']):<12}{b['total']:>+11.1%}{b['maxdd']:>8.1%}{b['sharpe']:>8.2f}   "
                     f"{v['total']:>+9.1%}{v['maxdd']:>8.1%}{v['sharpe']:>8.2f}   {d * 100:>+7.1f}pt")
    lines.append(f"Variant beat base on return in {won_ret}/{n} windows, on drawdown in {won_dd}/{n}, "
                 f"on Sharpe in {won_sh}/{n}. Median return difference {statistics.median(diffs) * 100:+.1f} pts "
                 f"(range {min(diffs) * 100:+.1f} to {max(diffs) * 100:+.1f}).")
    return "\n".join(lines)


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    base_cfg = load_config()
    cfg = apply_overrides(base_cfg, args)
    custom_universe = args.min_dollar_volume is not None or args.max_symbols is not None
    if args.stagger and custom_universe:
        ap.error("--stagger compares against config.yaml on one universe; drop --min-dollar-volume/--max-symbols")
    key, secret = alpaca_keys()
    trading = TradingClient(key, secret, paper=require_paper(cfg))
    data = StockHistoricalDataClient(key, secret)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    capital = args.capital or cfg["account"]["starting_capital"]

    uname = args.universe or (
        f"universe_bt_{int(cfg['universe']['min_avg_dollar_volume'] / 1e6)}m_{cfg['universe']['max_symbols']}.json"
        if custom_universe else "universe_bt.json")
    universe, built, upath = frozen_universe(trading, data, cfg, uname, args.rebuild_universe)
    days = (end.date() - start.date()).days + 300
    panel, ppath = frozen_prices(data, universe, days, cfg, end, args.refetch_prices)
    close = panel["close"]
    absent = sorted(s for s in universe if s not in close.columns or close[s].notna().sum() == 0)
    # Coverage belongs in the headline: one stock silently missing from a download moved the same
    # three-year baseline by 19 points. Two results with different coverage are not comparable.
    ident = (f"UNIVERSE: state/{UNIVERSE_DIR}/{uname} built {built}, {len(universe)} symbols, "
             f"id {universe_id(universe)} | bars through {end.date()} | data for {len(universe) - len(absent)}"
             f"/{len(universe)}" + (f" (missing: {', '.join(absent[:8])}{' ...' if len(absent) > 8 else ''})" if absent else "")
             + f"\nPRICES: state/{UNIVERSE_DIR}/prices/{ppath.name}")
    log.info(ident)

    spy, vix = None, None
    try:
        spy = frozen_prices(data, ["SPY"], days + 320, cfg, end, args.refetch_prices)[0]["close"]["SPY"].dropna()
        if cfg["regime"]["enabled"] or base_cfg["regime"]["enabled"]:
            vix = vix_series(Path(cfg["_root"]) / cfg["data"]["cache_dir"])
    except Exception as e:  # benchmark and regime gate are optional
        log.warning("SPY unavailable (no benchmark, no regime gate): %s", e)
    hist = Path(cfg["_root"]) / "state" / "history"
    earnings = pd.read_csv(hist / "earnings.csv") if (hist / "earnings.csv").exists() else None
    insider = pd.read_csv(hist / "insider.csv") if (hist / "insider.csv").exists() else None
    if earnings is not None:
        covered = set(earnings["symbol"].astype(str).str.upper())
        missing = [s for s in universe if s not in covered]
        if missing:
            log.warning("earnings history missing for %d/%d universe names (block inert for them): run "
                        "scripts/build_earnings_history.py --symbols-file state/%s/%s",
                        len(missing), len(universe), UNIVERSE_DIR, uname)
    log.info("features: regime=%s (ladder=%s, off_liquidate=%s) earnings_history=%s insider_history=%s (tilt weight %s) "
             "trail=%s take_profit=%s pyramid_adds=%s",
             spy is not None and cfg["regime"]["enabled"], cfg["regime"].get("ladder"), cfg["regime"].get("off_liquidate"),
             earnings is not None, insider is not None, cfg["facts"]["insider_tilt_weight"],
             cfg["exits"].get("trail_atr_multiple"), cfg["exits"].get("take_profit_r_multiple"),
             cfg["portfolio"].get("pyramid_max_adds"))

    signal_panels = {}     # keyed by what compute_panels reads: base and variant usually share them

    def run(c: dict, s: pd.Timestamp) -> tuple:
        reg, reg_state = regime_inputs(spy, vix, c)
        k = panel_key(c["signals"])
        if k not in signal_panels:
            signal_panels[k] = compute_panels(panel, c["signals"])
        return run_backtest(panel, c, s, end, capital, args.slippage_bps, args.halt_cooldown_days,
                            regime=reg, earnings=earnings, insider=insider, regime_state=reg_state,
                            panels=signal_panels[k])

    STATE_DIR.mkdir(exist_ok=True)
    trial_log = Path(cfg["_root"]) / "state" / UNIVERSE_DIR / "trials.jsonl"
    prior = int((cfg.get("backtest") or {}).get("prior_untracked_trials", 0))
    uid = universe_id(universe)

    def trial(c: dict, s: pd.Timestamp, curve: pd.DataFrame, label: dict) -> dict:
        st = curve_stats(curve)
        return {"config_id": T.config_id(strategy_fingerprint(c, args.slippage_bps, args.halt_cooldown_days)),
                "overrides": label, "tag": args.tag, "start": str(s.date()), "end": str(end.date()),
                "universe_id": uid, "data_coverage": f"{len(universe) - len(absent)}/{len(universe)}",
                "stats": T.return_stats(curve["equity"]), "total": st["total"], "maxdd": st["maxdd"],
                "sharpe_annual": st["sharpe"]}

    if args.stagger:
        tag = f"{start.date()}_{end.date()}_x{args.stagger}{('_' + args.tag) if args.tag else ''}"
        rows = []
        for k in range(args.stagger):
            s = start + pd.DateOffset(months=k)
            cb, cv = run(base_cfg, s)[0], run(cfg, s)[0]
            rows.append({"start": s.date(), "base": curve_stats(cb), "variant": curve_stats(cv)})
            for c, curve, label in ((base_cfg, cb, {}), (cfg, cv, overrides_of(args))):
                T.record(trial_log, trial(c, s, curve, label))
            log.info("window %d/%d (%s) done", k + 1, args.stagger, s.date())
        out = STATE_DIR / f"backtest_stagger_{tag}.csv"
        pd.DataFrame([{"start": r["start"], **{f"{w}_{m}": r[w][m] for w in ("base", "variant")
                                                for m in ("total", "maxdd", "sharpe")}} for r in rows]).to_csv(out, index=False)
        print("\n" + ident)
        print(compare_windows(rows))
        print(f"Saved state/{out.name}")
        return

    curve, trades = run(cfg, start)
    this = trial(cfg, start, curve, overrides_of(args))
    verdict = T.judge(T.load(trial_log), this, prior)
    T.record(trial_log, {**this, "dsr": verdict["dsr"], "n_trials": verdict["n_trials"]})
    if writes_live_reference(args, uname):
        # The live-vs-backtest comparison in the daily email reads this; stamped with the config it describes
        # so a later config change shows up as a stale reference rather than a silent mismatch.
        ref = {**reference_from_curve(curve["equity"]), "config_id": this["config_id"], "universe_id": uid,
               "window": f"{start.date()} to {end.date()}",
               "generated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        atomic_write_text(Path(cfg["_root"]) / "state" / UNIVERSE_DIR / "live_reference.json", json.dumps(ref, indent=1))
    tag = f"{start.date()}_{end.date()}{('_' + args.tag) if args.tag else ''}"
    curve.to_csv(STATE_DIR / f"backtest_curve_{tag}.csv")
    trades.to_csv(STATE_DIR / f"backtest_trades_{tag}.csv", index=False)
    print("\n" + ident)
    print(f"SIZING: {cfg['portfolio']['sizing']}"
          + (f" @ {cfg['portfolio']['risk_per_trade_pct']:.2%}/trade" if cfg["portfolio"]["sizing"] == "risk_parity" else ""))
    print(summarize(curve, trades, spy))
    print(f"Deflated Sharpe {verdict['dsr']:.2f} - probability the edge is real after {verdict['n_trials']} trials "
          f"({verdict['trials_with_spread']} comparable on this window) and this run's skew/kurtosis; 0.95 is the usual bar. "
          f"Undeflated: {verdict['psr_vs_zero']:.2f}.")
    print(f"Saved state/backtest_curve_{tag}.csv and state/backtest_trades_{tag}.csv; trial logged to "
          f"state/{UNIVERSE_DIR}/trials.jsonl")


if __name__ == "__main__":
    main()

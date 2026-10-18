#!/usr/bin/env python
"""Daily desk run. Schedule after the US close (e.g. 16:45 ET) on weekdays.

  python run_daily.py --dry-run        stage orders without submitting; writes no state, brief or webhook
  python run_daily.py                  live run against the configured (paper) account
  python run_daily.py --resume         clear a kill-switch halt, then run

Three phases. DECIDE reads bars, facts and the AI roles and plans exits and entries; it places no order
except a trailing-stop ratchet. PROTECT checks every holding for a live broker-side stop. EXECUTE sends exits,
entries and adds. If DECIDE fails (a data outage, a bug), PROTECT and the brief still run and the error is
re-raised afterwards, so a crash never leaves a position filled this morning without a stop tonight.

Launched before the day's bars are final (market open, or before 16:20 ET - a Mac that woke late), the run
is PROTECT-ONLY: sync, the kill switch and the stop check, no decisions. --force runs everything anyway.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide

from desk.analyst import analyst_rules, apply_verdicts, red_team, review
from desk.config import alpaca_keys, load_config, release, require_paper
from desk.data import SESSION_DONE, build_universe, cached_bars
from desk.execution import Broker, sync_positions
from desk.facts import FactBuilder, candidate_blockers, compact, insider_tilt, size_multiplier
from desk.notify import send
from desk.portfolio import (decay_exit, entry_limit, pyramid_due, select_entries, shrink_qty, size_position,
                            trail_stop)
from desk.regime import current_regime, regime_params, vix_series
from desk.risk import can_exit_today, enforce_entry_limits, evaluate
from desk.signals import compute_panels, rerank, snapshot
from desk import schedule, shadow, telemetry
from desk.state import load_state, save_state, write_brief

log = logging.getLogger("desk")
ET = ZoneInfo("America/New_York")
NO_STOP = "has NO protective stop"          # telemetry and the monitor count notes containing this


def now_et() -> datetime:
    """Wall clock in New York. One function so tests can pin the time of day."""
    return datetime.now(ET)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="compute and print; submit nothing, save no state")
    ap.add_argument("--no-analyst", action="store_true")
    ap.add_argument("--resume", action="store_true", help="clear a kill-switch halt")
    ap.add_argument("--refresh-universe", action="store_true")
    ap.add_argument("--force", action="store_true", help="run even if today is not a trading day")
    ap.add_argument("--reset-hwm", action="store_true",
                    help="with --resume after a HARD FLOOR halt: acknowledge re-baselining the all-time high")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = load_config()
    today = date.today()
    st = load_state()
    telemetry.note(dry_run=args.dry_run, release=release(cfg)["name"])
    if args.resume:
        if st.get("hard_halted") and not args.reset_hwm:
            msg = (f"HARD FLOOR halt ({st.get('halt_reason')}) is not cleared by --resume alone. "
                   "Re-run with --resume --reset-hwm to acknowledge re-baselining the all-time high.")
            log.error(msg)
            publish(msg, today, cfg, args.dry_run, brief=False)
            return
        st.update(halted=False, hard_halted=False, halt_reason="", halt_date="")
        st["high_water_mark"] = 0.0  # episode HWM re-baselines so the same drawdown doesn't re-trip immediately
        if args.reset_hwm:
            st["all_time_hwm"] = 0.0
        if not args.dry_run:
            save_state(st)
        log.info("halt cleared%s", " (all-time high reset)" if args.reset_hwm else "")

    key, secret = alpaca_keys()
    trading = TradingClient(key, secret, paper=require_paper(cfg))
    data = StockHistoricalDataClient(key, secret)
    broker = Broker(trading, dry_run=args.dry_run)
    mode = "PAPER" + (" DRY-RUN" if args.dry_run else "")

    if not args.force and not broker.is_trading_day(today):
        log.info("not a trading day; exiting")
        telemetry.note(status="skipped", reason="not a trading day")
        return

    acct = broker.account()
    if acct["trading_blocked"]:
        publish(f"[{mode}] account is blocked by the broker; no action taken", today, cfg, args.dry_run, brief=False)
        telemetry.note(status="blocked", reason="account blocked by broker")
        return
    live = broker.positions()
    notes = sync_positions(st, broker, live, today)
    if not args.dry_run:
        # Closes just appended to trades.csv and their removal from state land together: a run that dies
        # later cannot log the same close again tomorrow.
        save_state(st)
    decision = evaluate(acct["equity"], acct["last_equity"], st, cfg["risk"])
    notes += decision.notes

    if decision.halt:
        run_halted(args, cfg, today, st, broker, live, acct, decision, notes, mode)
        return

    session_open = broker.market_open()
    clock = now_et()
    settling = time(16, 0) <= clock.time() < time(*SESSION_DONE)       # the close's prints still arriving
    protect_only = (session_open or settling) and not args.force
    plan, failure = {"exits": {}, "entries": [], "adds": [], "trims": {}, "analyst_out": None, "analyst_status": "skipped",
                     "stale": False, "target_positions": cfg["portfolio"]["target_positions"]}, None
    headline = ""
    if protect_only:
        # Signals would read a partial day and exits would fill at once, not at the next open: leave every
        # decision to the evening run. Sync, the kill switch (above) and the stop check still run.
        headline = (f"PROTECT-ONLY RUN at {clock:%H:%M} ET, before today's bars are final: stops checked, no "
                    f"exits or entries - the evening run decides (--force overrides)")
    else:
        try:
            plan = decide(args, cfg, today, st, acct, live, decision, broker, trading, data, (key, secret), notes,
                          session_open)
        except Exception as e:
            failure = e
            log.exception("decision phase failed")
            notes.append(f"MANUAL ACTION: the decision phase failed ({type(e).__name__}: "
                         f"{telemetry.redact(str(e))[:200]}) - no exits or entries today; protective stops were "
                         f"still checked. See state/launchd.log.")
    exits, entries, adds = plan["exits"], plan["entries"], plan["adds"]
    notes += schedule.schedule_problems(schedule.loaded_jobs(), schedule.repeating_wake())
    try:
        protect(st, broker, exits, notes)
        execute(args, cfg, today, st, broker, acct, exits, entries, adds, notes, session_open, plan.get("trims"))
        if not args.dry_run:
            record_shadow(today, cfg, st, entries, plan.get("shadow") or [])
    finally:
        st["manual_open"] = age_manual_actions(notes, st.get("manual_open") or {}, today)
        if not args.dry_run:
            st["last_run"] = today.isoformat()
            save_state(st)

    notes.sort(key=lambda n: not n.startswith("MANUAL ACTION"))
    brief = render_brief(mode, today, acct, decision, st, exits, entries, notes, headline, cfg, adds,
                         target=plan["target_positions"])
    if plan["analyst_out"] and plan["analyst_out"].get("brief"):
        brief += "\n\nANALYST\n" + plan["analyst_out"]["brief"]
    publish(brief, today, cfg, args.dry_run)
    count = lambda text: sum(text in n for n in notes)  # noqa: E731
    telemetry.note(entries=len(entries), exits=len(exits), adds=len(adds), notes=len(notes), stale=plan["stale"],
                   analyst=plan["analyst_status"], equity=round(acct["equity"], 2),
                   # Held and staged apart, counted as the brief counts them: one "positions" number that included
                   # pending entries read as 12 beside a brief saying "11 + 1 staged" (monitor F-21, 2026-10-06).
                   positions=sum(1 for p in st["positions"].values() if p.get("filled", True)),
                   positions_staged=sum(1 for p in st["positions"].values() if not p.get("filled", True)),
                   manual_actions=count("MANUAL ACTION"), cap_breaches=count("% cap - sell"),
                   unprotected=count(NO_STOP), protection_unverified=count("protection UNVERIFIED"),
                   protection_replaced=count("protective stop re-placed"), order_errors=count("ORDER FAILED"),
                   protection_mismatch=count("protection MISMATCH"),
                   stop_not_raised=count("stop NOT raised"), decay_deferred=count("signal-decay check deferred"),
                   data_concerns=count("data concern:"), session_open=session_open,
                   partial_bar=count("session still in progress") > 0, regime=plan.get("regime_label", "n/a"),
                   regime_state=plan.get("regime", "n/a"),
                   protect_only=protect_only, decision_failed=failure is not None)
    if failure is not None:
        raise failure


def publish(text: str, today: date, cfg: dict, dry_run: bool, brief: bool = True) -> None:
    """A dry run prints only. Writing its brief would replace the live run's under the same date, and the
    14:00 email and the monitor both read that file."""
    if dry_run:
        print(text)
        return
    if brief:
        write_brief(text, today)
    send(text, cfg)


def run_halted(args, cfg, today, st, broker, live, acct, decision, notes, mode) -> None:
    if not st.get("halted"):
        try:
            broker.liquidate_all()
            flat = "All positions liquidated."
        except Exception as e:
            # Record the halt regardless: the next run sees it and retries every survivor below.
            log.error("liquidate_all failed: %s", e)
            flat = f"LIQUIDATION FAILED ({type(e).__name__}: {e}) - retried next run; check the broker now."
        for sym, pos in st["positions"].items():
            pos["pending_exit_reason"] = "KILL SWITCH: " + decision.halt_reason
        st.update(halted=True, hard_halted=decision.hard, halt_reason=decision.halt_reason, halt_date=today.isoformat())
        how = "--resume --reset-hwm" if decision.hard else "--resume"
        headline = f"KILL SWITCH FIRED - {decision.halt_reason}. {flat} Run with {how} to re-enable."
    else:
        how = "--resume --reset-hwm" if st.get("hard_halted") else "--resume"
        headline = f"HALTED since {st['halt_date']} ({st['halt_reason']}). No trading. Run with {how} to re-enable."
        # Anything still held survived the liquidation, and liquidate_all cancelled its stops first, so
        # it is open with no protection. Re-send the close every halted run until it is gone.
        resent = []
        for sym in live:
            try:
                broker.close(sym)
                resent.append(sym)
            except Exception as e:
                log.error("kill-switch retry close %s failed: %s", sym, e)
                notes.append(f"MANUAL ACTION: {sym} STILL OPEN and the close was rejected ({type(e).__name__}: {e}) "
                             f"- it {NO_STOP}; close it at the broker")
            if sym in st["positions"]:
                st["positions"][sym]["pending_exit_reason"] = f"KILL SWITCH (retry): {st['halt_reason']}"
        if resent:
            headline += (f" {len(resent)} position(s) survived the liquidation with no stop; "
                         f"close re-sent: {', '.join(sorted(resent))}.")
    if not args.dry_run:
        st["last_run"] = today.isoformat()
        save_state(st)
    publish(render_brief(mode, today, acct, decision, st, {}, [], notes, headline, cfg), today, cfg, args.dry_run)
    telemetry.note(status="halted", reason=st.get("halt_reason"), hard=bool(st.get("hard_halted")),
                   halt_date=st.get("halt_date"), open_positions=len(live), notes=len(notes),
                   unprotected=sum(NO_STOP in n for n in notes))


def decide(args, cfg, today, st, acct, live, decision, broker, trading, data, keys, notes, session_open) -> dict:
    """Everything that needs bars, facts or the AI roles. Places no order except a trailing-stop ratchet."""
    key, secret = keys
    universe = build_universe(trading, data, cfg, force=args.refresh_universe)
    symbols = sorted(set(universe) | set(st["positions"]))
    panel = cached_bars(data, symbols, cfg["data"]["history_days"], cfg)
    scfg, pcfg, ecfg, rcfg = cfg["signals"], cfg["portfolio"], cfg["exits"], cfg["risk"]
    panels = compute_panels(panel, scfg)
    snap = snapshot(panels, -1, scfg)

    # Entries are priced off the last bar; if that bar is not the latest session's, the limits and
    # stops are stale and no entry should be placed (exits still run - they are date-based or broker-side).
    last_bar = panel["close"].index[-1].date()
    expected_bar = today
    while expected_bar.weekday() >= 5:
        expected_bar -= timedelta(days=1)
    stale = last_bar < expected_bar and not args.force
    notes.append(f"data through {last_bar.isoformat()}" + (" - STALE, entries suppressed" if stale else ""))
    co_note = telemetry.checkout_warning(telemetry.checkout())
    if co_note:
        notes.append(co_note)
    clock = now_et()
    if session_open and last_bar >= clock.date():
        # Only under --force: otherwise a run during the session is protect-only and never gets here.
        notes.append(f"WARNING: run at {clock:%H:%M} ET while the market is open - today's session still in "
                     f"progress, so signals, limits and stops are priced off a partial day, and exits fill now, "
                     f"not at the next open")

    day_pcfg = dict(pcfg)
    target_positions = pcfg["target_positions"]
    # regime_status is the internal state (off / neutral / strong); regime_label is what the brief says for it
    # (risk-off / risk-on / risk-on (strong)). The run log recorded the first beside a brief showing the second,
    # so "neutral" read as disagreeing with "risk-on" (monitor F-24); both are recorded now. "disabled" when the
    # gate is switched off - "off" there read as risk-off.
    regime_liquidate, regime_status, regime_label = False, "disabled", "disabled"
    if cfg["regime"]["enabled"]:
        try:
            spy = cached_bars(data, ["SPY"], 320, cfg, tag="spy")["close"]["SPY"].dropna()
            reg = current_regime(spy, vix_series(Path(cfg["_root"]) / cfg["data"]["cache_dir"]), cfg["regime"])
            state = reg.get("state") or ("off" if reg["scale"] < 1.0 else "neutral")
            rp = regime_params(state, pcfg, cfg["regime"])
            target_positions, regime_liquidate = rp["target_positions"], rp["liquidate"]
            day_pcfg["max_position_pct"] = rp["max_position_pct"]
            regime_status, regime_label = state, reg["label"]
            vix_txt = f"{reg['vix']:.1f}" if reg["vix"] is not None else "unavailable"
            notes.append(f"regime {reg['label']}: SPY {reg['spy']:.0f} vs {reg['spy_ma'] or 0:.0f} MA, VIX {vix_txt}; "
                         f"target positions {target_positions}, position cap {day_pcfg['max_position_pct']:.0%}"
                         + (" - FLAT: exiting everything" if regime_liquidate else ""))
        except Exception as e:
            # Fail safe: without a reading, size as risk-off until the feed is back. Never liquidate on a failed
            # feed, even with off_liquidate set - selling everything is a decision a data outage cannot make.
            log.warning("regime unavailable: %s", e)
            rp = regime_params("off", pcfg, dict(cfg["regime"], off_liquidate=False))
            target_positions, day_pcfg["max_position_pct"] = rp["target_positions"], rp["max_position_pct"]
            regime_status, regime_label = "unavailable", "unavailable (sized as risk-off)"
            notes.append(f"WARNING: regime gate unavailable ({type(e).__name__}) - fail-safe: risk-off target of "
                         f"{target_positions} positions until the SPY/VIX feeds are back; nothing is sold for it")

    # The cap is enforced at entry; a winner can drift through it. Up to trim_above_pct that is a note. At or
    # above it the holding is trimmed back to trim_to_pct at the next open (owner's rule, 2026-09-29) - see
    # execute() for how the broker-side stop is handled.
    trims = plan_trims(live, acct["equity"], day_pcfg, notes, st["positions"])

    # Trailing stop: ratchet each held name's stop to (highest close since entry - k x ATR), never lower,
    # and push the new level to the broker so it protects from the next session on.
    if float(ecfg.get("trail_atr_multiple") or 0) > 0:
        for sym, pos in st["positions"].items():
            if not pos.get("filled") or pos.get("stop") is None or pos.get("pending_exit_reason"):
                continue
            if sym not in panel["close"].columns or panel["close"][sym].dropna().empty:
                continue
            c = float(panel["close"][sym].dropna().iloc[-1])
            a = panels["atr"][sym].dropna()
            if a.empty:
                continue
            pos["high_close"] = max(float(pos.get("high_close") or pos["entry_price"]), c)
            new = trail_stop(float(pos["stop"]), pos["high_close"], float(a.iloc[-1]), ecfg)
            if new > float(pos["stop"]):
                try:
                    effective = broker.raise_stop(sym, new)
                except Exception as e:
                    log.warning("raise_stop %s failed: %s", sym, e)
                    notes.append(f"{sym} stop NOT raised to {new:.2f}: the replace was rejected ({type(e).__name__}) "
                                 f"- the previous stop is checked below")
                    continue
                if effective is None:
                    notes.append(f"{sym} stop NOT raised to {new:.2f}: no open stop order found at the broker - check manually")
                else:
                    if effective > float(pos["stop"]):
                        notes.append(f"{sym} stop raised {pos['stop']:.2f} -> {effective:.2f}")
                    # Never record a level below what the broker is holding.
                    pos["stop"] = max(float(pos["stop"]), effective)

    facts, pool, facts_failed = {}, [], False
    if cfg["facts"]["enabled"] and not cfg["facts"].get("social_enabled", True):
        hist_path = Path(cfg["_root"]) / "state" / "social_history.json"
        try:
            hist = json.loads(hist_path.read_text()) if hist_path.exists() else {}
        except (OSError, ValueError):
            hist = {}
        ready = sum(1 for v in hist.values() if len(v) >= 15)
        # The furthest name's count shows the history growing: "0 of 58" alone read as a stuck counter (R-26).
        most = max((len(v) for v in hist.values()), default=0)
        notes.append(f"social crowding guard: off, collecting history ({ready} of {len(hist)} names have the 15 "
                     f"sessions it needs; most so far {most}) - not screening for pumps yet")
    if cfg["facts"]["enabled"]:
        pool = [s for s in snap[snap["eligible"]].index if s not in st["positions"]][: cfg["facts"]["candidate_pool"]]
        try:
            fb = FactBuilder(key, secret, cfg, today, asset_lookup=getattr(trading, "get_asset", None))
            facts = fb.build(sorted(set(pool) | set(st["positions"])), panel["close"])
        except Exception as e:
            # Fail closed, like the analyst: an unscreened candidate could be reporting tomorrow or mid-offering.
            log.warning("facts unavailable: %s", e)
            facts_failed = True
            notes.append(f"fact sheets unavailable ({type(e).__name__}) -> fail-safe: no new entries today "
                         f"(candidates cannot be screened for earnings or offerings)")
        if pool and facts and all((facts.get(s) or {}).get("earnings_lookup_failed") for s in pool):
            notes.append("earnings calendar unavailable for every candidate (FINNHUB_API_KEY missing or Finnhub "
                         "down) -> fail-safe: all blocked, no new entries today")
    blocked, tilt = {}, {}
    for s, f in facts.items():
        if s in st["positions"]:
            continue
        reasons = candidate_blockers(f, cfg["facts"])
        if reasons:
            blocked[s] = "; ".join(reasons)
        t = insider_tilt(f, cfg["facts"])
        if t:
            tilt[s] = t
    shadow_rows = [shadow.row(today, "blocked", s, float(snap.loc[s, "close"]), float(snap.loc[s, "atr"]), ecfg,
                              reason=r, rank=int(snap.loc[s, "rank"]))
                   for s, r in blocked.items() if s in snap.index and bool(snap.loc[s, "eligible"])]
    snap = rerank(snap, tilt, blocked)
    notes += [f"blocked {s}: {r}" for s, r in list(blocked.items())[:8]]
    if facts:
        # Only fact-checked names may be bought. The correlation filter can skip past the screened pool,
        # and anything beyond it is invisible to the earnings/offering blocks.
        unscreened = {s: "not fact-checked" for s in snap.index[snap["eligible"].values]
                      if s not in facts and s not in st["positions"]}
        snap = rerank(snap, None, unscreened)

    exits = {}
    for sym, pos in st["positions"].items():
        if not pos.get("filled") or pos.get("pending_exit_reason"):
            continue
        held = (today - date.fromisoformat(pos["entry_date"])).days
        earn_date = (facts.get(sym) or {}).get("earnings_date")
        pre_earn = ecfg.get("earnings_exit_days_before") or 0
        # Trading days, not calendar days: a Monday report must be caught on Thursday/Friday.
        eid_td = int(np.busday_count(today, date.fromisoformat(earn_date))) if earn_date else None
        if regime_liquidate:
            exits[sym] = "regime: SPY below its MA (or VIX high) -> cash"
        elif held >= ecfg["max_hold_days"]:
            exits[sym] = f"max hold {ecfg['max_hold_days']}d"
        elif pre_earn and eid_td is not None and 0 <= eid_td <= pre_earn:
            # Overrides min_hold_days on purpose: the binary event is the reason to be flat.
            exits[sym] = f"pre-earnings exit ({eid_td} trading days to report)"
        else:
            verdict = decay_exit(snap, sym, held, ecfg)
            if verdict == "defer":
                # No complete bar today (a trading halt, a feed gap). Absence from the snapshot says nothing
                # about the ranking, so wait for data instead of selling into the reopening on a gap.
                notes.append(f"{sym}: no complete bar today - signal-decay check deferred")
            elif verdict == "decay":
                exits[sym] = "signal decay"

    held_after = [s for s in st["positions"] if s not in exits]
    may_add = decision.allow_entries and not stale and not facts_failed
    slots = target_positions - len(held_after) if may_add else 0
    picks = select_entries(snap, panels["returns"], list(st["positions"]), slots, pcfg)
    equity = acct["equity"]
    cash_avail = min(acct["cash"], acct["non_marginable_buying_power"]) - equity * pcfg["cash_floor_pct"]
    entries = []
    for sym in picks:
        row = snap.loc[sym]
        sz = size_position(equity, cash_avail, float(row["close"]), float(row["atr"]), day_pcfg, ecfg)
        if not sz:
            continue
        mult = size_multiplier(facts.get(sym, {}), cfg["facts"]) if facts else 1.0
        if mult < 1.0:
            # One share is already the smallest expressible position; shrinking below it would drop the
            # name entirely, turning a risk-reduction rule into a silent exclusion.
            full = sz["qty"]
            sz["qty"] = shrink_qty(sz["qty"], mult)
            shadow_rows.append(shadow.row(today, "half_size", sym, float(row["close"]), float(row["atr"]), ecfg,
                                          qty=full - sz["qty"], reason="implied vol >> realized", rank=int(row["rank"])))
            sz["notional"] = sz["qty"] * float(row["close"])
        cash_avail -= sz["notional"]
        entries.append({
            "symbol": sym, "price": float(row["close"]), "limit": entry_limit(float(row["close"])),
            "rank": int(row["rank"]), "momentum_6m": float(row["momentum"]), "return_20d": float(row["short"]),
            "atr_pct": float(row["atr_pct"]), **sz,
            "thesis": f"rank {int(row['rank'])}, 6m mom {row['momentum']:.0%}, 20d {row['short']:.0%}"
                      + (" | half size: implied vol >> realized" if mult < 1.0 else "")
                      + (" | insider buying tilt" if sym in tilt else "")
                      + (" | dilutive filing flagged" if (facts.get(sym) or {}).get("offering_filings") else ""),
        })

    max_new = pcfg.get("max_new_entries_per_day")
    if max_new and len(entries) > max_new:
        # Paced before the analyst so review cost and freshness track what is actually bought today;
        # deferred names are re-selected (and re-reviewed) on later sessions if still ranked.
        notes.append(f"entry pacing: {len(entries) - max_new} deferred to later sessions (max {max_new}/day)")
        entries = entries[:max_new]

    analyst_out, analyst_status = None, "skipped"
    if cfg["analyst"]["enabled"] and not args.no_analyst and (entries or st["positions"]):
        n_review = cfg["analyst"]["candidates_to_review"]
        if len(entries) > n_review:
            # Fail closed: nothing the analyst has not seen gets bought.
            notes.append(f"{len(entries) - n_review} entries beyond analyst review capacity dropped")
            entries = entries[:n_review]
        payload = {
            "date": today.isoformat(), "mode": "PAPER" + (" DRY-RUN" if args.dry_run else ""),
            "account": {"equity": equity, "cash": acct["cash"], "drawdown_from_hwm": round(decision.drawdown, 4),
                        "day_return": round(decision.daily_return, 4), "kill_switch_at": rcfg["max_drawdown_pct"]},
            # The prompt asks for the regime, and without this the brief guessed it from the drawdown: "Regime is
            # risk-off" on a risk-on day (monitor F-37, 2026-10-16).
            "regime": {"label": regime_label, "state": regime_status},
            "limits": {"max_positions": pcfg["max_positions"], "target_positions": target_positions,
                       "max_position_pct": day_pcfg["max_position_pct"], "max_hold_days": ecfg["max_hold_days"]},
            "rules": analyst_rules(cfg),
            "holdings": [
                {"symbol": s, **{k: p.get(k) for k in ("entry_date", "entry_price", "qty", "stop", "target", "last_price", "adds")},
                 "thesis_at_entry": p.get("thesis"),
                 "days_held": (today - date.fromisoformat(p["entry_date"])).days,
                 "exit_planned": exits.get(s),
                 "facts": compact(facts[s]) if s in facts else None}
                for s, p in st["positions"].items()
            ],
            "candidates": [{**e, "facts": compact(facts[e["symbol"]]) if e["symbol"] in facts else None}
                           for e in entries[: cfg["analyst"]["candidates_to_review"]]],
            "blocked_by_hard_filters": blocked,
        }
        if cfg["analyst"].get("red_team_enabled"):
            cases = red_team(red_team_view(payload, cfg), cfg)
            if cases:
                payload["red_team"] = cases
                flagged = [f"{s} {c['trap_type']} {c['severity']:.1f}" for s, c in cases.items() if c["severity"] >= 0.5]
                if flagged:
                    notes.append("red team flags: " + ", ".join(flagged))
            else:
                notes.append("red team unavailable; analyst ran without it")
        analyst_out = review(payload, cfg)
        analyst_status = "ok" if analyst_out is not None else "unavailable"
        if analyst_out is None:
            entries = []
            notes.append("analyst unavailable -> fail-safe: no new entries today")
        else:
            before = {e["symbol"]: e for e in entries}
            entries, extra, verdict_notes = apply_verdicts(analyst_out["verdicts"], st["positions"], entries,
                                                           cfg["analyst"]["reduce_factor"])
            kept = {e["symbol"] for e in entries}
            reasons = {str(v.get("symbol", "")).upper(): str(v.get("reason", "")) for v in analyst_out["verdicts"]}
            for s, e in before.items():
                if s not in kept and s in snap.index:
                    shadow_rows.append(shadow.row(today, "analyst_remove", s, e["price"], float(snap.loc[s, "atr"]), ecfg,
                                                  qty=e["qty"], reason=reasons.get(s, ""), rank=e.get("rank")))
            for s in extra:
                pos = st["positions"].get(s) or {}
                if s in snap.index:
                    shadow_rows.append(shadow.row(today, "analyst_exit", s, float(snap.loc[s, "close"]),
                                                  float(snap.loc[s, "atr"]), ecfg, qty=float(pos.get("qty") or 0),
                                                  reason=reasons.get(s, "")))
            exits.update(extra)
            notes += verdict_notes
            if analyst_out.get("regime_mismatch"):
                notes.append(f"WARNING: {analyst_out['regime_mismatch']}")
            notes += [f"data concern: {c}" for c in analyst_out.get("data_concerns", [])]
            for c in analyst_out.get("data_concerns", []):
                telemetry.emit({"event": "data_concern", "source": "analyst", "text": telemetry.redact(c)[:500],
                                "signature": telemetry.signature(c)})

    n_held = len([s for s in st["positions"] if s not in exits])
    entries = enforce_entry_limits(entries, equity, acct["cash"], n_held, day_pcfg)

    # Pyramid adds: a winner that has moved one initial-risk unit past its last entry gets another unit,
    # sized like a fresh entry at today's price, capped by the position limit, sharing the trailed stop.
    # Holdings the analyst is removing or reducing never get added to.
    adds = []
    if int(pcfg.get("pyramid_max_adds") or 0) > 0 and may_add:
        for sym, pos in st["positions"].items():
            if not pos.get("filled") or sym in exits or pos.get("add_order_id") or pos.get("stop") is None or sym not in snap.index:
                continue
            row = snap.loc[sym]
            px, a = float(row["close"]), float(row["atr"])
            if np.isnan(a) or not pyramid_due(pos, px, pcfg):
                continue
            sz = size_position(equity, cash_avail, px, a, day_pcfg, ecfg)
            if not sz:
                continue
            room = equity * day_pcfg["max_position_pct"] - float(pos["qty"]) * px
            q = min(sz["qty"], int(room // px)) if room > 0 else 0
            if q < 1:
                continue
            cash_avail -= q * px
            adds.append({"symbol": sym, "qty": q, "price": px, "limit": entry_limit(px), "stop": float(pos["stop"]),
                         "notional": q * px, "add_n": int(pos.get("adds") or 0) + 1})
    return {"exits": exits, "entries": entries, "adds": adds, "trims": trims, "analyst_out": analyst_out,
            "analyst_status": analyst_status, "stale": stale, "target_positions": target_positions,
            "regime": regime_status, "regime_label": regime_label, "shadow": shadow_rows}


def record_shadow(today: date, cfg: dict, st: dict, entries: list, rows: list) -> None:
    """Today's decisions for the shadow book (desk/shadow.py), with the entries actually staged as the control.
    A failure here must never touch trading: it is logged and the run goes on."""
    try:
        ecfg = cfg["exits"]
        staged = [shadow.row(today, "bought", e["symbol"], e["price"], (e["price"] - e["stop"]) / float(ecfg["stop_atr_multiple"]),
                             ecfg, qty=e["qty"], reason=e.get("thesis", ""), rank=e.get("rank"))
                  for e in entries if e["symbol"] in st["positions"]]
        name = release(cfg)["name"]
        shadow.record([dict(r, release=name) for r in staged + rows])
    except Exception as e:
        log.warning("shadow book not written: %s", e)


def red_team_view(payload: dict, cfg: dict) -> dict:
    """What the red team reviews. With analyst.red_team_scope 'candidates' it gets only today's candidates (holdings
    as bare symbols, for concentration): 30% of API spend went on bear cases for every holding as well, and in 18
    runs they produced one high-severity case, while a holding's REDUCE is never acted on (engineering review
    2026-10-17). The analyst still reviews holdings, with their fact sheets."""
    if cfg["analyst"].get("red_team_scope", "all") != "candidates":
        return payload
    return {**payload, "holdings": [], "held_symbols": [h["symbol"] for h in payload.get("holdings", [])]}


def protect(st: dict, broker, exits: dict, notes: list) -> None:
    """Every position keeps a live broker-side stop. Bracket legs inherit the entry's DAY tif and die at the
    close of the day the entry fills, so from the next session on the only protection is what is re-placed
    here. A position already through its stop with nothing resting is closed outright. Anything that leaves
    a holding without a stop is a MANUAL ACTION note, every run, until it is fixed."""
    for sym, pos in st["positions"].items():
        if not pos.get("filled") or sym in exits or pos.get("pending_exit_reason"):
            continue
        if pos.get("stop") is None:
            notes.append(f"MANUAL ACTION: {sym} {NO_STOP}: adopted from the broker with no stop on record - place "
                         f"one at the broker or close it (the desk only time-exits it)")
            continue
        last = float(pos.get("last_price") or 0)
        if last and last <= float(pos["stop"]):
            exits[sym] = f"stop {pos['stop']:.2f} breached (no live broker order)"
            continue
        ensure_stop(broker, sym, pos, notes)


def ensure_stop(broker, sym: str, pos: dict, notes: list) -> None:
    """Re-place a GTC stop (OCO with the target) if the broker has none resting for this holding."""
    legs = broker.live_exit_orders(sym)
    if legs.get("error"):
        notes.append(f"MANUAL ACTION: {sym} protection UNVERIFIED: the broker order query failed - check its stop "
                     f"at the broker")
        return
    moved = pos.pop("reanchor_pending", False)
    if legs["stop"] is not None:
        if moved and abs(float(legs["stop"].stop_price) - float(pos["stop"])) >= 0.01:
            # A leg still resting at the staged level after the stop was measured from the fill (anchor_to_fill).
            # Bracket legs normally expire at the close; this covers one that did not.
            try:
                broker.move_exit_levels(sym, float(pos["stop"]), pos.get("target"))
                notes.append(f"{sym} resting stop moved {float(legs['stop'].stop_price):.2f} -> {pos['stop']:.2f} "
                             f"(measured from the fill)")
            except Exception as e:
                log.error("moving %s exit legs failed: %s", sym, e)
                # State must match what actually rests: back to the staged levels.
                pos["stop"], pos["target"] = float(legs["stop"].stop_price), pos.get("staged_target", pos.get("target"))
                notes.append(f"{sym} stop left at {pos['stop']:.2f}: moving it to the fill-based level was "
                             f"rejected ({type(e).__name__})")
        check_coverage(sym, pos, legs, notes)
        return
    try:
        broker.submit_exit_protection(sym, int(float(pos["qty"])), float(pos["stop"]), pos.get("target"))
        notes.append(f"{sym} protective stop re-placed at {pos['stop']:.2f} (GTC)")
    except Exception as e:
        log.error("protective stop for %s rejected: %s", sym, e)
        notes.append(f"MANUAL ACTION: {sym} {NO_STOP}: re-placing it at {pos['stop']:.2f} was rejected "
                     f"({type(e).__name__}: {telemetry.redact(str(e))[:160]}) - place it at the broker")


def check_coverage(sym: str, pos: dict, legs: dict, notes: list) -> None:
    """A resting stop must cover exactly the shares held, less any sale already working. A manual trim leaves
    it too large (when it fires it sells short); a manual add leaves the new shares bare. Existence alone
    was all that was checked."""
    covered = getattr(legs["stop"], "qty", None)
    if covered is None:
        return
    held, covered = float(pos["qty"]), float(covered)
    selling = float(getattr(legs.get("close"), "qty", 0) or 0)
    if covered + selling < held:
        notes.append(f"MANUAL ACTION: {sym} {NO_STOP} for {held - covered - selling:g} of {held:g} share(s): the "
                     f"resting stop covers {covered:g} - resize it at the broker")
    elif covered > held:
        notes.append(f"MANUAL ACTION: {sym} protection MISMATCH: the resting stop is for {covered:g} shares but "
                     f"{held:g} are held - when it fires it sells {covered - held:g} short; resize it at the broker")


def execute(args, cfg, today, st, broker, acct, exits, entries, adds, notes, session_open, trims=None) -> None:
    """Exits, then trims, then entries, then adds. One rejected order never stops the rest, and state is saved
    after each submitted entry so a run killed half way leaves state agreeing with the broker."""
    ecfg, rcfg = cfg["exits"], cfg["risk"]
    for sym, reason in exits.items():
        pos = st["positions"][sym]
        if not can_exit_today(pos, today, acct["daytrade_count"], rcfg, session_open):
            notes.append(f"{sym} exit deferred a day (day-trade guard)")
            continue
        try:
            broker.close(sym)
        except Exception as e:
            log.error("close %s failed: %s", sym, e)
            notes.append(f"ORDER FAILED: {sym} exit ({reason}) was rejected ({type(e).__name__}: "
                         f"{telemetry.redact(str(e))[:160]}) - retried next run")
            # close() cancels the stops before selling, so a rejected sale can leave the holding bare.
            if pos.get("stop") is not None:
                ensure_stop(broker, sym, pos, notes)
            continue
        pos["pending_exit_reason"] = reason
    for sym, t in (trims or {}).items():
        execute_trim(args, today, st, broker, acct, sym, t, exits, notes, rcfg, session_open)
    for e in entries:
        try:
            oid = broker.submit_bracket(e["symbol"], e["qty"], e["limit"], e["stop"], e.get("target"))
        except Exception as ex:
            log.error("entry %s failed: %s", e["symbol"], ex)
            notes.append(f"ORDER FAILED: BUY {e['symbol']} x{e['qty']} was rejected ({type(ex).__name__}: "
                         f"{telemetry.redact(str(ex))[:160]}) - nothing was bought")
            continue
        if not args.dry_run:
            telemetry.order_event("entry_staged", e["symbol"], order_id=oid, qty=e["qty"], limit=e["limit"],
                                  stop=e["stop"], target=e.get("target"))
            st["positions"][e["symbol"]] = {
                "entry_date": today.isoformat(), "entry_price": e["price"], "qty": e["qty"], "stop": e["stop"],
                "target": e.get("target"), "filled": False, "order_id": oid, "thesis": e["thesis"],
                "exit_by": (today + timedelta(days=ecfg["max_hold_days"])).isoformat(),
                "high_close": e["price"], "adds": 0, "last_add_price": e["price"],
                "initial_risk_per_share": round(e["price"] - e["stop"], 4),
                "release": release(cfg)["name"],
            }
            save_state(st)
    for a in adds:
        try:
            oid = broker.submit_bracket(a["symbol"], a["qty"], a["limit"], a["stop"], None)
        except Exception as ex:
            log.error("add %s failed: %s", a["symbol"], ex)
            notes.append(f"ORDER FAILED: ADD {a['symbol']} x{a['qty']} was rejected ({type(ex).__name__})")
            continue
        if not args.dry_run:
            st["positions"][a["symbol"]].update(add_order_id=oid, add_date=today.isoformat())
            save_state(st)


def plan_trims(live: dict, equity: float, pcfg: dict, notes: list, positions: dict | None = None) -> dict:
    """{sym: {"sell", "keep", "weight"}} for holdings at or above trim_above_pct of equity, sized back to
    trim_to_pct. Between the entry cap and the trim line: a note the first run a holding crosses the cap, not
    every run (UMC was noted five days running; the audit of 2026-10-03 called it an alarm with no mechanism
    behind it). `positions` is state["positions"]: the flag lives there and clears when the holding falls back."""
    positions = positions if positions is not None else {}
    cap = float(pcfg["max_position_pct"])
    above = float(pcfg.get("trim_above_pct") or 0)
    to = float(pcfg.get("trim_to_pct") or cap)
    trims = {}
    for sym, p in sorted(live.items()):
        px = float(p.get("current_price") or 0)
        weight = float(p["market_value"]) / equity if equity else 0
        if px <= 0 or weight <= cap * 1.02:
            positions.get(sym, {}).pop("cap_drift_noted", None)
            continue
        held = int(float(p["qty"]))
        keep = math.floor(equity * to / px)
        if above and weight >= above and 1 <= keep < held:
            trims[sym] = {"sell": held - keep, "keep": keep, "weight": weight}
            notes.append(f"{sym} is {weight:.1%} of equity, at or above the {above:.0%} trim line - selling "
                         f"{held - keep} of {held} share(s) at the next open to bring it back to {to:.0%}")
        elif not positions.get(sym, {}).get("cap_drift_noted"):
            line = f"; trims automatically at {above:.0%}" if above else ""
            notes.append(f"{sym} is {weight:.1%} of equity, over the {cap:.0%} entry cap{line}")
            if sym in positions:
                positions[sym]["cap_drift_noted"] = True
    return trims


def execute_trim(args, today, st, broker, acct, sym, t, exits, notes, rcfg, session_open) -> None:
    """Sell part of a holding and keep the rest protected. The broker-side stop reserves every share, so the
    sale is refused unless it is cancelled first: cancel, sell, then re-place the stop for what is left - the
    sequence done by hand for UMC on 2026-09-24. A failure at any step re-protects the holding."""
    pos = st["positions"].get(sym)
    if pos is None or sym in exits or pos.get("pending_exit_reason") or not pos.get("filled"):
        return
    if not can_exit_today(pos, today, acct["daytrade_count"], rcfg, session_open):
        notes.append(f"{sym} trim deferred a day (day-trade guard)")
        return
    try:
        broker.cancel_symbol_orders(sym, OrderSide.SELL)
        broker.sell_market(sym, t["sell"])
    except Exception as e:
        log.error("trim %s failed: %s", sym, e)
        notes.append(f"ORDER FAILED: {sym} trim of {t['sell']} share(s) was rejected ({type(e).__name__}: "
                     f"{telemetry.redact(str(e))[:160]}) - retried next run")
        if pos.get("stop") is not None:
            ensure_stop(broker, sym, pos, notes)            # the cancel may already have gone through
        return
    held = pos["qty"]
    if not args.dry_run:
        pos["qty"] = t["keep"]                              # the next run's sync confirms it from the broker
        pos["last_trim"] = {"date": today.isoformat(), "sold": t["sell"], "weight": round(t["weight"], 4)}
    try:
        if pos.get("stop") is None:
            raise ValueError("no stop on record")
        broker.submit_exit_protection(sym, t["keep"], float(pos["stop"]), pos.get("target"))
        notes.append(f"{sym} trimmed: {t['sell']} of {held:g} share(s) sell at the next open; stop re-placed for "
                     f"{t['keep']} at {pos['stop']:.2f}")
    except Exception as e:
        log.error("protective stop after trimming %s rejected: %s", sym, e)
        notes.append(f"MANUAL ACTION: {sym} {NO_STOP}: the trim sale is queued but re-placing the stop for the "
                     f"{t['keep']} remaining share(s) failed ({type(e).__name__}) - place it at the broker")


def age_manual_actions(notes: list, open_since: dict, today: date) -> dict:
    """Mark each MANUAL ACTION note with the day it was first raised, and return the open set to save. A
    cap breach used to read the same on day 1 and day 5 (monitor F-12). Keyed on the note with its numbers
    blanked, so UMC's cap breach stays one item as its weight moves; an item that stops appearing is closed."""
    now_open = {}
    for i, n in enumerate(notes):
        if not n.startswith("MANUAL ACTION"):
            continue
        k = re.sub(r"\d[\d.,]*", "#", n)[:90]
        first = open_since.get(k, today.isoformat())
        now_open[k] = first
        if first < today.isoformat():
            notes[i] = f"{n} [open since {first}]"
    return now_open


def render_brief(mode, today, acct, decision, st, exits, entries, notes, headline, cfg, adds=None, target=None) -> str:
    pcfg, ecfg, rcfg = cfg["portfolio"], cfg["exits"], cfg["risk"]
    eq = acct["equity"]
    rel = release(cfg)["name"]
    lines = [f"DESK BRIEF {today.isoformat()} [{mode}]" + (f" {rel}" if rel else "")]
    if headline:
        lines.append(headline)
    lines.append(
        f"Equity ${eq:,.0f} (day {decision.daily_return:+.1%}) | HWM ${st['high_water_mark']:,.0f} | "
        f"drawdown {decision.drawdown:.1%} (kill at {rcfg['max_drawdown_pct']:.0%}) | "
        f"from all-time high ${st.get('all_time_hwm') or eq:,.0f}: {decision.cumulative_drawdown:.1%} "
        f"(floor {rcfg.get('max_cumulative_drawdown_pct', 0):.0%}) | cash ${acct['cash']:,.0f} ({acct['cash'] / eq:.0%})"
    )
    # Held positions only: staged entries are orders, listed under "Entries staged" (monitor F-21 - MXL, ASST
    # and CORT showed as holdings at day 0 and the listed values summed to $1,645 over equity).
    held = {s: p for s, p in st["positions"].items() if p.get("filled", True)}
    staged = len(st["positions"]) - len(held)
    # The day's target (the regime gate halves it in risk-off), not the configured one.
    lines.append(f"Positions {len(held)}/{pcfg['max_positions']} "
                 f"(target {target if target is not None else pcfg['target_positions']})"
                 + (f" · {staged} staged" if staged else ""))
    value = sum(float(p["qty"]) * float(p.get("last_price") or p["entry_price"]) for p in held.values())
    if eq and abs(value + acct["cash"] - eq) > 0.02 * eq:
        log.warning("brief: holdings $%.0f + cash $%.0f = $%.0f vs equity $%.0f", value, acct["cash"],
                    value + acct["cash"], eq)
        notes.append(f"data concern: held positions (${value:,.0f}) plus cash (${acct['cash']:,.0f}) differ from "
                     f"equity (${eq:,.0f}) by more than 2%")
    for sym, p in sorted(held.items()):
        held = (today - date.fromisoformat(p["entry_date"])).days
        last = p.get("last_price") or p["entry_price"]
        pnl = last / p["entry_price"] - 1 if p["entry_price"] else 0
        flag = f"  -> EXIT: {exits[sym]}" if sym in exits else (f"  -> {p['pending_exit_reason']}" if p.get("pending_exit_reason") else "")
        stop = f"{p['stop']:.2f}" if p.get("stop") else "NONE"
        tgt = f"{p['target']:.2f}" if p.get("target") else ("trail" if p.get("stop") else "-")
        adds_s = f" +{p['adds']}" if p.get("adds") else ""
        lines.append(f"  {sym:<6} {p['qty']:>5.0f}{adds_s} @ {p['entry_price']:.2f} -> {last:.2f} ({pnl:+.1%}) day {held}/{ecfg['max_hold_days']} stop {stop} tgt {tgt}{flag}")
    if entries:
        lines.append("Entries staged for next open:")
        for e in entries:
            tgt = f"{e['target']:.2f}" if e.get("target") else "trail"
            lines.append(f"  BUY {e['symbol']:<6} x{e['qty']} lim {e['limit']:.2f} stop {e['stop']:.2f} tgt {tgt} (${e['notional']:,.0f}) - {e['thesis']}")
    elif not headline:
        lines.append("Entries staged: none")
    if adds:
        lines.append("Pyramid adds staged for next open:")
        for a in adds:
            lines.append(f"  ADD {a['symbol']:<6} x{a['qty']} lim {a['limit']:.2f} stop {a['stop']:.2f} (${a['notional']:,.0f}) - add {a['add_n']}")
    manual = [n for n in notes if n.startswith("MANUAL ACTION")]
    if manual:
        lines.insert(2, f"MANUAL ACTION NEEDED: {len(manual)} item(s), listed first under Notes")
    if notes:
        lines.append("Notes:")
        lines += [f"  - {n}" for n in notes]
    return "\n".join(lines)


if __name__ == "__main__":
    telemetry.run_job("desk", main)

#!/usr/bin/env python
"""Daily desk run. Schedule after the US close (e.g. 16:45 ET) on weekdays.

  python run_daily.py --dry-run        stage orders without submitting
  python run_daily.py                  live run against the configured (paper) account
  python run_daily.py --resume         clear a kill-switch halt, then run
"""
from __future__ import annotations

import argparse
import logging
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from desk.analyst import apply_verdicts, red_team, review
from desk.config import alpaca_keys, load_config
from desk.data import build_universe, cached_bars
from desk.execution import Broker, sync_positions
from desk.facts import FactBuilder, candidate_blockers, compact, insider_tilt, size_multiplier
from desk.notify import send
from desk.portfolio import pyramid_due, select_entries, size_position, trail_stop
from desk.regime import current_regime, regime_params, vix_series
from desk.risk import can_exit_today, enforce_entry_limits, evaluate
from desk.signals import compute_panels, rerank, snapshot
from desk import telemetry
from desk.state import load_state, save_state, write_brief

log = logging.getLogger("desk")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="compute and print, submit nothing, save no state")
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
    telemetry.note(dry_run=args.dry_run)
    if args.resume:
        if st.get("hard_halted") and not args.reset_hwm:
            msg = (f"HARD FLOOR halt ({st.get('halt_reason')}) is not cleared by --resume alone. "
                   "Re-run with --resume --reset-hwm to acknowledge re-baselining the all-time high.")
            log.error(msg)
            send(msg, cfg)
            return
        st.update(halted=False, hard_halted=False, halt_reason="", halt_date="")
        st["high_water_mark"] = 0.0  # episode HWM re-baselines so the same drawdown doesn't re-trip immediately
        if args.reset_hwm:
            st["all_time_hwm"] = 0.0
        save_state(st)
        log.info("halt cleared%s", " (all-time high reset)" if args.reset_hwm else "")

    key, secret = alpaca_keys()
    trading = TradingClient(key, secret, paper=cfg["account"]["paper"])
    data = StockHistoricalDataClient(key, secret)
    broker = Broker(trading, dry_run=args.dry_run)
    mode = ("PAPER" if cfg["account"]["paper"] else "LIVE") + (" DRY-RUN" if args.dry_run else "")

    if not args.force and not broker.is_trading_day(today):
        log.info("not a trading day; exiting")
        telemetry.note(status="skipped", reason="not a trading day")
        return

    acct = broker.account()
    if acct["trading_blocked"]:
        send(f"[{mode}] account is blocked by the broker; no action taken", cfg)
        telemetry.note(status="blocked", reason="account blocked by broker")
        return
    live = broker.positions()
    notes = sync_positions(st, broker, live, today)
    decision = evaluate(acct["equity"], acct["last_equity"], st, cfg["risk"])
    notes += decision.notes

    if decision.halt:
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
                    notes.append(f"{sym} STILL OPEN and the close was rejected ({type(e).__name__}: {e}) - "
                                 f"close it at the broker")
                if sym in st["positions"]:
                    st["positions"][sym]["pending_exit_reason"] = f"KILL SWITCH (retry): {st['halt_reason']}"
            if resent:
                headline += (f" {len(resent)} position(s) survived the liquidation with no stop; "
                             f"close re-sent: {', '.join(sorted(resent))}.")
        if not args.dry_run:
            st["last_run"] = today.isoformat()
            save_state(st)
        brief = render_brief(mode, today, acct, decision, st, {}, [], notes, headline, cfg)
        write_brief(brief, today)
        send(brief, cfg)
        telemetry.note(status="halted", reason=st.get("halt_reason"), hard=bool(st.get("hard_halted")),
                       open_positions=len(live), notes=len(notes))
        return

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

    day_pcfg = dict(pcfg)
    target_positions = pcfg["target_positions"]
    regime_liquidate = False
    if cfg["regime"]["enabled"]:
        try:
            spy = cached_bars(data, ["SPY"], 320, cfg, tag="spy")["close"]["SPY"].dropna()
            reg = current_regime(spy, vix_series(Path(cfg["_root"]) / cfg["data"]["cache_dir"]), cfg["regime"])
            state = reg.get("state") or ("off" if reg["scale"] < 1.0 else "neutral")
            rp = regime_params(state, pcfg, cfg["regime"])
            target_positions, regime_liquidate = rp["target_positions"], rp["liquidate"]
            day_pcfg["max_position_pct"] = rp["max_position_pct"]
            vix_txt = f"{reg['vix']:.1f}" if reg["vix"] is not None else "unavailable"
            notes.append(f"regime {reg['label']}: SPY {reg['spy']:.0f} vs {reg['spy_ma'] or 0:.0f} MA, VIX {vix_txt}; "
                         f"target positions {target_positions}, position cap {day_pcfg['max_position_pct']:.0%}"
                         + (" - FLAT: exiting everything" if regime_liquidate else ""))
        except Exception as e:
            log.warning("regime unavailable: %s", e)

    # The cap is enforced at entry only; a winner can drift through it. Say so every run until it is fixed -
    # the desk does not trim held positions itself (a partial sell would orphan the broker-side OCO).
    cap = day_pcfg["max_position_pct"]
    for sym, p in sorted(live.items()):
        weight = p["market_value"] / acct["equity"] if acct["equity"] else 0
        if weight > cap * 1.02 and p["current_price"] > 0:
            over = int(p["qty"]) - math.floor(acct["equity"] * cap / p["current_price"])
            notes.append(f"MANUAL ACTION: {sym} is {weight:.1%} of equity, above the {cap:.0%} cap - sell {over} "
                         f"share(s) to get back under (the desk does not trim held positions automatically)")

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
                effective = broker.raise_stop(sym, new)
                if effective is None:
                    notes.append(f"{sym} stop NOT raised to {new:.2f}: no open stop order found at the broker - check manually")
                else:
                    if effective > float(pos["stop"]):
                        notes.append(f"{sym} stop raised {pos['stop']:.2f} -> {effective:.2f}")
                    # Never record a level below what the broker is holding.
                    pos["stop"] = max(float(pos["stop"]), effective)

    facts = {}
    if cfg["facts"]["enabled"] and not cfg["facts"].get("social_enabled", True):
        notes.append("social crowding guard: disabled in config (no history yet) - not screening for pumps")
    if cfg["facts"]["enabled"]:
        pool = [s for s in snap[snap["eligible"]].index if s not in st["positions"]][: cfg["facts"]["candidate_pool"]]
        try:
            fb = FactBuilder(key, secret, cfg, today, asset_lookup=getattr(trading, "get_asset", None))
            facts = fb.build(sorted(set(pool) | set(st["positions"])), panel["close"])
        except Exception as e:
            log.warning("facts unavailable: %s", e)
            notes.append("fact sheets unavailable -> candidates not screened for earnings/offerings")
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
        elif held >= ecfg["min_hold_days"]:
            if sym not in snap.index:
                # No complete bar today (a trading halt, a feed gap). Absence from the snapshot says nothing
                # about the ranking, so wait for data instead of selling into the reopening on a gap.
                notes.append(f"{sym}: no complete bar today - signal-decay check deferred")
            elif not bool(snap.loc[sym, "eligible"]) or snap.loc[sym, "rank"] > ecfg["signal_decay_rank"]:
                exits[sym] = "signal decay"

    held_after = [s for s in st["positions"] if s not in exits]
    slots = target_positions - len(held_after) if decision.allow_entries and not stale else 0
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
            sz["qty"] = max(1, int(sz["qty"] * mult))
            sz["notional"] = sz["qty"] * float(row["close"])
        cash_avail -= sz["notional"]
        entries.append({
            "symbol": sym, "price": float(row["close"]), "limit": round(float(row["close"]) * 1.005, 2),
            "rank": int(row["rank"]), "momentum_6m": float(row["momentum"]), "return_20d": float(row["short"]),
            "atr_pct": float(row["atr_pct"]), **sz,
            "thesis": f"rank {int(row['rank'])}, 6m mom {row['momentum']:.0%}, 20d {row['short']:.0%}"
                      + (" | half size: implied vol >> realized" if mult < 1.0 else "")
                      + (" | insider buying tilt" if sym in tilt else ""),
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
            "date": today.isoformat(), "mode": mode,
            "account": {"equity": equity, "cash": acct["cash"], "drawdown_from_hwm": round(decision.drawdown, 4),
                        "day_return": round(decision.daily_return, 4), "kill_switch_at": rcfg["max_drawdown_pct"]},
            "limits": {"max_positions": pcfg["max_positions"], "target_positions": target_positions,
                       "max_position_pct": day_pcfg["max_position_pct"], "max_hold_days": ecfg["max_hold_days"]},
            "holdings": [
                {"symbol": s, **{k: p.get(k) for k in ("entry_date", "entry_price", "qty", "stop", "target", "last_price", "thesis", "adds")},
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
            cases = red_team(payload, cfg)
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
            entries, extra, verdict_notes = apply_verdicts(analyst_out["verdicts"], st["positions"], entries,
                                                           cfg["analyst"].get("reduce_factor", 0.75))
            exits.update(extra)
            notes += verdict_notes
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
    if int(pcfg.get("pyramid_max_adds") or 0) > 0 and decision.allow_entries and not stale:
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
            adds.append({"symbol": sym, "qty": q, "price": px, "limit": round(px * 1.005, 2), "stop": float(pos["stop"]),
                         "notional": q * px, "add_n": int(pos.get("adds") or 0) + 1})

    # Every position keeps a live broker-side stop. Bracket legs inherit the entry's DAY tif and die at the
    # close of the day the entry fills, so from the next session on the only protection is what is re-placed
    # here. A position already through its stop with nothing resting is closed outright.
    for sym, pos in st["positions"].items():
        if not pos.get("filled") or sym in exits or pos.get("stop") is None or pos.get("pending_exit_reason"):
            continue
        last = float(pos.get("last_price") or 0)
        if last and last <= float(pos["stop"]):
            exits[sym] = f"stop {pos['stop']:.2f} breached (no live broker order)"
            continue
        legs = broker.live_exit_orders(sym)
        if legs.get("error"):
            notes.append(f"{sym} protection UNVERIFIED: broker order query failed - check manually")
            continue
        if legs["stop"] is None:
            oid = broker.submit_exit_protection(sym, int(float(pos["qty"])), float(pos["stop"]), pos.get("target"))
            notes.append(f"{sym} protective stop re-placed at {pos['stop']:.2f} (GTC)" + ("" if oid else " - FAILED"))

    for sym, reason in exits.items():
        pos = st["positions"][sym]
        if not can_exit_today(pos, today, acct["daytrade_count"], rcfg):
            notes.append(f"{sym} exit deferred a day (day-trade guard)")
            continue
        broker.close(sym)
        pos["pending_exit_reason"] = reason
    for e in entries:
        oid = broker.submit_bracket(e["symbol"], e["qty"], e["limit"], e["stop"], e.get("target"))
        if not args.dry_run:
            st["positions"][e["symbol"]] = {
                "entry_date": today.isoformat(), "entry_price": e["price"], "qty": e["qty"], "stop": e["stop"],
                "target": e.get("target"), "filled": False, "order_id": oid, "thesis": e["thesis"],
                "exit_by": (today + timedelta(days=ecfg["max_hold_days"])).isoformat(),
                "high_close": e["price"], "adds": 0, "last_add_price": e["price"],
                "initial_risk_per_share": round(e["price"] - e["stop"], 4),
            }
    for a in adds:
        oid = broker.submit_bracket(a["symbol"], a["qty"], a["limit"], a["stop"], None)
        if not args.dry_run:
            st["positions"][a["symbol"]].update(add_order_id=oid, add_date=today.isoformat())

    if not args.dry_run:
        st["last_run"] = today.isoformat()
        save_state(st)
    notes.sort(key=lambda n: not n.startswith("MANUAL ACTION"))
    brief = render_brief(mode, today, acct, decision, st, exits, entries, notes, "", cfg, adds)
    if analyst_out and analyst_out.get("brief"):
        brief += "\n\nANALYST\n" + analyst_out["brief"]
    write_brief(brief, today)
    send(brief, cfg)
    count = lambda text: sum(text in n for n in notes)
    telemetry.note(entries=len(entries), exits=len(exits), adds=len(adds), notes=len(notes), stale=stale,
                   analyst=analyst_status, positions=len(st["positions"]), equity=round(acct["equity"], 2),
                   manual_actions=count("MANUAL ACTION"), cap_breaches=count("% cap - sell"),
                   protection_unverified=count("protection UNVERIFIED"), protection_replaced=count("protective stop re-placed"),
                   stop_not_raised=count("stop NOT raised"), decay_deferred=count("signal-decay check deferred"),
                   data_concerns=count("data concern:"))


def render_brief(mode, today, acct, decision, st, exits, entries, notes, headline, cfg, adds=None) -> str:
    pcfg, ecfg, rcfg = cfg["portfolio"], cfg["exits"], cfg["risk"]
    eq = acct["equity"]
    lines = [f"DESK BRIEF {today.isoformat()} [{mode}]"]
    if headline:
        lines.append(headline)
    lines.append(
        f"Equity ${eq:,.0f} (day {decision.daily_return:+.1%}) | HWM ${st['high_water_mark']:,.0f} | "
        f"drawdown {decision.drawdown:.1%} (kill at {rcfg['max_drawdown_pct']:.0%}) | "
        f"from all-time high ${st.get('all_time_hwm') or eq:,.0f}: {decision.cumulative_drawdown:.1%} "
        f"(floor {rcfg.get('max_cumulative_drawdown_pct', 0):.0%}) | cash ${acct['cash']:,.0f} ({acct['cash'] / eq:.0%})"
    )
    lines.append(f"Positions {len(st['positions'])}/{pcfg['max_positions']} (target {pcfg['target_positions']})")
    for sym, p in sorted(st["positions"].items()):
        held = (today - date.fromisoformat(p["entry_date"])).days
        last = p.get("last_price") or p["entry_price"]
        pnl = last / p["entry_price"] - 1 if p["entry_price"] else 0
        flag = f"  -> EXIT: {exits[sym]}" if sym in exits else (f"  -> {p['pending_exit_reason']}" if p.get("pending_exit_reason") else "")
        stop = f"{p['stop']:.2f}" if p.get("stop") else "-"
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

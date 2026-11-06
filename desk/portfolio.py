from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Rules the live run and the backtest must apply identically. Each used to be written out twice.
ENTRY_LIMIT_MULT = 1.005        # entry limit = signal close x this (0.5% above)


def entry_limit(price: float) -> float:
    return round(price * ENTRY_LIMIT_MULT, 2)


def pace_entries(entries: list, max_new, positions: dict, today) -> tuple:
    """(entries kept, note or ""). max_new_entries_per_day is per day, not per run: entries an earlier run placed
    today count against it, so a second run the same day (a forced daytime run, then the evening job) cannot double
    the pace. Falsy max_new = no cap."""
    if not max_new:
        return entries, ""
    placed = sum(1 for p in (positions or {}).values() if p.get("entry_date") == today.isoformat())
    room = max(0, int(max_new) - placed)
    if len(entries) <= room:
        return entries, ""
    earlier = f"; {placed} already placed today" if placed else ""
    return entries[:room], (f"entry pacing: {len(entries) - room} deferred to later sessions "
                            f"(max {max_new}/day{earlier})")


def too_small(qty: float, full_qty: float, min_fraction) -> bool:
    """A pick the remaining cash can fund only as a sliver: under portfolio.min_entry_fraction of the shares it
    would get with cash to spare. Skipped rather than bought (HST: 2 shares, $50 of a ~$7k slot, 2026-11-06); it is
    picked again on a later session if still ranked. Judged before any deliberate cut (half size, analyst REDUCE),
    so a cut is never mistaken for a sliver. 0 or None = off."""
    return bool(min_fraction) and full_qty > 0 and qty < float(min_fraction) * full_qty


def shrink_qty(qty: int, factor: float) -> int:
    """Every size cut (analyst REDUCE, implied-vol half size, simulated vetoes): floored so it never rounds
    back up, never below one share so a cut never turns into a silent exclusion, and the factor is clamped
    to at most 1 so no rule written as a cut can enlarge a position."""
    return max(1, math.floor(qty * min(1.0, max(0.0, float(factor)))))


def decay_exit(snap: pd.DataFrame, sym: str, held_days: int, ecfg: dict) -> str | None:
    """Signal-decay rule for a holding: None = keep, "decay" = exit, "defer" = no complete bar today. A name
    missing from the snapshot (trading halt, feed gap) says nothing about its ranking, so the check waits."""
    if held_days < ecfg["min_hold_days"]:
        return None
    if sym not in snap.index:
        return "defer"
    if not bool(snap.loc[sym, "eligible"]) or snap.loc[sym, "rank"] > ecfg["signal_decay_rank"]:
        return "decay"
    return None


def too_correlated(sym: str, basket: list, returns: pd.DataFrame, pcfg: dict) -> bool:
    """More than max_correlated_peers holdings correlated above the threshold over the last correlation_days.
    Pairwise-complete Pearson, computed exactly as DataFrame.corrwith does (np.corrcoef on the rows where both
    have data) without its per-column apply machinery, which was a fifth of the backtest's run time."""
    peers = [b for b in basket if b in returns.columns and b != sym]
    if not peers or sym not in returns.columns:
        return False
    window = returns.tail(pcfg["correlation_days"])
    x = window[sym].to_numpy(dtype=float)
    x_ok = ~np.isnan(x)
    threshold, hits = pcfg["correlation_threshold"], 0
    with np.errstate(all="ignore"):
        for p in peers:
            y = window[p].to_numpy(dtype=float)
            ok = x_ok & ~np.isnan(y)
            if ok.any() and np.corrcoef(x[ok], y[ok])[0, 1] > threshold:
                hits += 1
    return hits > pcfg["max_correlated_peers"]


def select_entries(snap: pd.DataFrame, returns: pd.DataFrame, held: list, slots: int, pcfg: dict) -> list:
    if slots <= 0:
        return []
    cands = snap[snap["eligible"] & ~snap.index.isin(held)].sort_values("rank")
    chosen, basket = [], list(held)
    for sym in cands.index:
        if len(chosen) >= slots:
            break
        if too_correlated(sym, basket, returns, pcfg):
            continue
        chosen.append(sym)
        basket.append(sym)
    return chosen


def _bracket(price: float, atr: float, ecfg: dict) -> tuple | None:
    stop = round(price - ecfg["stop_atr_multiple"] * atr, 2)
    if stop <= 0 or stop >= price:
        return None
    r = float(ecfg.get("take_profit_r_multiple") or 0)
    # No take-profit leg when r is 0: the position is meant to trail instead of being capped.
    return stop, (round(price + r * (price - stop), 2) if r > 0 else None)


def trail_stop(stop: float | None, high_close: float, atr: float, ecfg: dict) -> float | None:
    """Chandelier ratchet: highest close since entry minus a multiple of ATR, and it only ever rises."""
    m = float(ecfg.get("trail_atr_multiple") or 0)
    if m <= 0 or stop is None or atr <= 0:
        return stop
    return max(stop, round(high_close - m * atr, 2))


def pyramid_due(pos: dict, price: float, pcfg: dict) -> bool:
    """A winner has moved trigger_r initial-risk units past its last entry and has adds remaining."""
    n = int(pcfg.get("pyramid_max_adds") or 0)
    if n <= 0 or int(pos.get("adds") or 0) >= n:
        return False
    r0 = float(pos.get("initial_risk_per_share") or 0)
    if r0 <= 0:
        return False
    last = float(pos.get("last_add_price") or pos["entry_price"])
    return price >= last + float(pcfg.get("pyramid_trigger_r") or 1.0) * r0


def _pack(qty: int, price: float, stop: float, target: float, pcfg: dict) -> dict | None:
    # Below the minimum lot the risk budget cannot be expressed: a 1-share position of a $400 stock
    # carries a third of the intended risk and every size modifier becomes a no-op.
    if qty < max(1, int(pcfg.get("min_shares_per_position") or 1)):
        return None
    return {"qty": qty, "notional": qty * price, "stop": stop, "target": target}


def size_notional(equity: float, cash_available: float, price: float, atr: float,
                  pcfg: dict, ecfg: dict) -> dict | None:
    """Each position gets equity/target_positions, scaled toward a fixed ATR reference."""
    bracket = _bracket(price, atr, ecfg)
    if not bracket:
        return None
    notional = equity / pcfg["target_positions"]
    if pcfg.get("vol_weighting") and atr > 0:
        # Scale toward a 3% ATR reference so a 6% ATR name gets half the notional.
        notional *= min(1.5, max(0.5, 0.03 / (atr / price)))
    notional = min(notional, equity * pcfg["max_position_pct"], cash_available)
    return _pack(math.floor(notional / price), price, *bracket, pcfg)


def size_risk_parity(equity: float, cash_available: float, price: float, atr: float,
                     pcfg: dict, ecfg: dict) -> dict | None:
    """Each position risks the same fraction of equity; the stop distance sets the share count.
    Volatile names get fewer shares as a consequence rather than via a separate scaling factor."""
    bracket = _bracket(price, atr, ecfg)
    if not bracket:
        return None
    stop, target = bracket
    qty = math.floor(equity * pcfg["risk_per_trade_pct"] / (price - stop))
    qty = min(qty, math.floor(equity * pcfg["max_position_pct"] / price),
              math.floor(cash_available / price) if cash_available > 0 else 0)
    return _pack(qty, price, stop, target, pcfg)


def size_position(equity: float, cash_available: float, price: float, atr: float,
                  pcfg: dict, ecfg: dict) -> dict | None:
    fn = size_risk_parity if pcfg.get("sizing") == "risk_parity" else size_notional
    return fn(equity, cash_available, price, atr, pcfg, ecfg)

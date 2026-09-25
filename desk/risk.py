from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class RiskDecision:
    halt: bool = False
    hard: bool = False            # hard floor: needs --resume --reset-hwm, not just --resume
    halt_reason: str = ""
    allow_entries: bool = True
    drawdown: float = 0.0         # from the episode high-water mark (reset on --resume)
    cumulative_drawdown: float = 0.0  # from the all-time high (never reset without --reset-hwm)
    daily_return: float = 0.0
    notes: list = field(default_factory=list)


def evaluate(equity: float, last_equity: float, state: dict, rcfg: dict) -> RiskDecision:
    """Updates both high-water marks in state and decides whether to halt or pause entries."""
    d = RiskDecision()
    hwm = max(float(state.get("high_water_mark") or 0.0), equity)
    state["high_water_mark"] = hwm
    ath = max(float(state.get("all_time_hwm") or 0.0), equity)
    state["all_time_hwm"] = ath
    d.drawdown = 1 - equity / hwm if hwm > 0 else 0.0
    d.cumulative_drawdown = 1 - equity / ath if ath > 0 else 0.0
    d.daily_return = equity / last_equity - 1 if last_equity > 0 else 0.0

    if state.get("halted"):
        d.halt = True
        d.hard = bool(state.get("hard_halted"))
        d.halt_reason = state.get("halt_reason", "previously halted")
        d.allow_entries = False
        return d
    floor = rcfg.get("max_cumulative_drawdown_pct")
    if floor and d.cumulative_drawdown >= floor:
        d.halt = d.hard = True
        d.halt_reason = (f"HARD FLOOR: cumulative drawdown {d.cumulative_drawdown:.1%} >= {floor:.0%} "
                         f"from all-time high ${ath:,.0f}")
        d.allow_entries = False
        return d
    if d.drawdown >= rcfg["max_drawdown_pct"]:
        d.halt = True
        d.halt_reason = f"drawdown {d.drawdown:.1%} >= {rcfg['max_drawdown_pct']:.0%} limit"
        d.allow_entries = False
        return d
    if d.daily_return <= -rcfg["daily_loss_halt_pct"]:
        d.allow_entries = False
        d.notes.append(f"day {d.daily_return:.1%}: new entries paused")
    return d


def can_exit_today(pos: dict, today: date, daytrade_count: int, rcfg: dict, session_open: bool = False) -> bool:
    """Avoid same-day round trips that would count toward the pattern-day-trader limit.

    entry_date is the day the entry was STAGED, the evening before it fills, so on its own it never matches
    a position bought today. After the close (the scheduled run) that is harmless: the sale executes next
    session and cannot complete a round trip. During the session (a run launched late while the market is
    open) the sale fills at once, so a position whose fill was first seen today (filled_date) counts too."""
    same_day = pos.get("entry_date") == today.isoformat()
    if session_open and pos.get("filled_date") == today.isoformat():
        same_day = True
    if not same_day:
        return True
    return daytrade_count < rcfg["max_daytrades_per_5d"]


def enforce_entry_limits(entries: list, equity: float, cash: float, n_held: int, pcfg: dict) -> list:
    """Final code-level gate on staged entries: count, per-position size, cash floor."""
    kept = []
    cash_left = cash - equity * pcfg["cash_floor_pct"]
    max_notional = equity * pcfg["max_position_pct"]
    for e in entries:
        if n_held + len(kept) >= pcfg["max_positions"]:
            break
        if e["notional"] > max_notional * 1.001:
            continue
        if e["notional"] > cash_left:
            continue
        cash_left -= e["notional"]
        kept.append(e)
    return kept

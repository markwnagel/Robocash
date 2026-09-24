from datetime import date

from desk.analyst import _parse, apply_verdicts
from desk.risk import can_exit_today, enforce_entry_limits, evaluate


def test_drawdown_kill_switch(cfg):
    st = {"high_water_mark": 10000.0, "halted": False}
    d = evaluate(equity=8400.0, last_equity=8500.0, state=st, rcfg=cfg["risk"])
    assert d.halt and not d.allow_entries
    assert "drawdown" in d.halt_reason


def test_hwm_ratchets_up_and_no_halt_on_gain(cfg):
    st = {"high_water_mark": 10000.0, "halted": False}
    d = evaluate(11000.0, 10500.0, st, cfg["risk"])
    assert st["high_water_mark"] == 11000.0
    assert not d.halt and d.allow_entries


def test_daily_loss_pauses_entries_only(cfg):
    st = {"high_water_mark": 10000.0, "halted": False}
    d = evaluate(9400.0, 10000.0, st, cfg["risk"])
    assert not d.halt and not d.allow_entries


def test_hard_floor_fires_on_cumulative_drawdown_even_after_episode_reset(cfg):
    # Episode HWM was re-baselined by --resume (10,000) but the all-time high is 14,000:
    # equity 10,400 is +4% on the episode and -25.7% from the all-time high.
    st = {"high_water_mark": 10000.0, "all_time_hwm": 14000.0, "halted": False}
    d = evaluate(10400.0, 10300.0, st, cfg["risk"])
    assert d.halt and d.hard and "HARD FLOOR" in d.halt_reason
    assert d.drawdown == 0.0 and abs(d.cumulative_drawdown - (1 - 10400 / 14000)) < 1e-9
    # Just above the floor: no hard halt, and the episode rule alone does not fire either.
    st = {"high_water_mark": 10000.0, "all_time_hwm": 14000.0, "halted": False}
    d = evaluate(10600.0, 10300.0, st, cfg["risk"])
    assert not d.halt and not d.hard
    # All-time high ratchets up with equity.
    st = {"high_water_mark": 0.0, "all_time_hwm": 0.0, "halted": False}
    evaluate(12000.0, 11000.0, st, cfg["risk"])
    assert st["all_time_hwm"] == 12000.0 and st["high_water_mark"] == 12000.0


def test_halted_state_sticks(cfg):
    st = {"high_water_mark": 10000.0, "halted": True, "halt_reason": "x"}
    d = evaluate(12000.0, 11000.0, st, cfg["risk"])
    assert d.halt and not d.allow_entries


def test_entry_limits(cfg):
    pcfg = cfg["portfolio"]
    entries = [{"symbol": f"S{i}", "notional": 900.0} for i in range(10)]
    kept = enforce_entry_limits(entries, equity=10000, cash=4000, n_held=12, pcfg=pcfg)
    assert len(kept) == pcfg["max_positions"] - 12
    kept = enforce_entry_limits(entries, equity=10000, cash=2400, n_held=0, pcfg=pcfg)
    assert sum(e["notional"] for e in kept) <= 2400 - 10000 * pcfg["cash_floor_pct"]
    oversized = [{"symbol": "BIG", "notional": 10000 * pcfg["max_position_pct"] * 2}]
    assert enforce_entry_limits(oversized, 10000, 9000, 0, pcfg) == []


def test_day_trade_guard(cfg):
    today = date(2026, 9, 14)
    assert can_exit_today({"entry_date": "2026-09-10"}, today, 3, cfg["risk"])
    assert not can_exit_today({"entry_date": "2026-09-14"}, today, 3, cfg["risk"])
    assert can_exit_today({"entry_date": "2026-09-14"}, today, 0, cfg["risk"])


def test_verdicts_can_only_reduce_risk():
    holdings = {"HLD": {"qty": 10}}
    entries = [
        {"symbol": "AAA", "qty": 10, "price": 50.0, "notional": 500.0},
        {"symbol": "BBB", "qty": 3, "price": 100.0, "notional": 300.0},
        {"symbol": "CCC", "qty": 1, "price": 400.0, "notional": 400.0},
    ]
    verdicts = [
        {"symbol": "AAA", "action": "REDUCE", "reason": "r", "confidence": 0.6},
        {"symbol": "BBB", "action": "REMOVE", "reason": "offering", "confidence": 0.9},
        {"symbol": "CCC", "action": "REDUCE", "reason": "r", "confidence": 0.5},
        {"symbol": "HLD", "action": "REMOVE", "reason": "guidance cut", "confidence": 0.8},
        {"symbol": "ZZZ", "action": "REMOVE", "reason": "unknown symbol", "confidence": 1.0},
        {"symbol": "AAA", "action": "ADD", "reason": "not allowed", "confidence": 1.0},
    ]
    kept, exits, notes = apply_verdicts(verdicts, holdings, entries)
    assert [e["symbol"] for e in kept] == ["AAA", "CCC"]
    assert kept[0]["qty"] == 7 and kept[0]["notional"] == 350.0   # 10 x 0.75, floored
    assert kept[1]["qty"] == 1  # REDUCE on a one-share position keeps the share; it never becomes REMOVE
    assert exits == {"HLD": "analyst REMOVE: guidance cut"}
    assert notes == []                                    # no REDUCE landed on a holding here


def test_reduce_on_a_held_position_is_reported_not_swallowed():
    # Trimming a holding would leave the broker-side stop covering more shares than we own, so the
    # verdict is surfaced for the owner instead. Silently doing nothing is the bug being guarded.
    holdings = {"HLD": {"qty": 10}}
    kept, exits, notes = apply_verdicts(
        [{"symbol": "HLD", "action": "REDUCE", "reason": "insider selling", "confidence": 0.5}],
        holdings, [])
    assert kept == [] and exits == {}                     # never turns into a sale
    assert len(notes) == 1 and "HLD" in notes[0] and "insider selling" in notes[0]
    assert holdings["HLD"]["analyst_flag"] == "analyst REDUCE: insider selling"


def test_parse_tolerates_fenced_json():
    txt = 'Here you go:\n```json\n{"verdicts": [], "brief": "ok"}\n```'
    assert _parse(txt)["verdicts"] == []
    assert _parse("no json here") is None
    assert _parse("[1, 2]") is None


def test_analyst_call_never_raises_so_the_run_reaches_its_exits(monkeypatch):
    """An unresolvable API key raises TypeError, not anthropic.APIError. If that escapes, run_daily
    dies before the exit and stop-re-placement loops and holdings are left naked overnight."""
    from desk import analyst

    class _Boom:
        def __init__(self, *a, **k):
            self.messages = self

        def create(self, **k):
            raise TypeError("Could not resolve authentication method. Expected one of api_key, ...")

    monkeypatch.setattr(analyst.anthropic, "Anthropic", _Boom)
    assert analyst._call_json("sys", "user", {}, "model", 0) is None
    assert analyst.review({}, {"analyst": {"model": "m", "max_web_searches": 0}}) is None
    assert analyst.red_team({}, {"analyst": {"model": "m", "red_team_max_web_searches": 0}}) == {}


def test_reduce_factor_is_configurable_and_can_never_enlarge():
    def reduced(qty, factor):
        e = [{"symbol": "AAA", "qty": qty, "price": 10.0, "notional": qty * 10.0}]
        v = [{"symbol": "AAA", "action": "REDUCE", "reason": "r", "confidence": 0.5}]
        return apply_verdicts(v, {}, e, factor)[0][0]["qty"]

    assert reduced(10, 0.5) == 5 and reduced(10, 0.75) == 7
    assert reduced(3, 0.75) == 2          # floored: 2.25 -> 2, never rounded back up to the full 3
    assert reduced(1, 0.75) == 1          # one share is the floor
    # A typo in config must not let the analyst buy MORE than the engine sized: the veto-only rule.
    assert reduced(10, 1.5) == 10 and reduced(10, -1) == 1


def test_a_crowded_trap_without_crowding_data_carries_no_severity(monkeypatch):
    from desk import analyst
    payload = {"holdings": [{"symbol": "AMD", "facts": {"social_msgs_per_day": None, "social_z": None}}],
               "candidates": [{"symbol": "GME", "facts": {"social_msgs_per_day": 900.0, "social_z": 4.1}}]}
    monkeypatch.setattr(analyst, "_call_json", lambda *a, **k: {"cases": [
        {"symbol": "AMD", "bear_case": "Everyone owns it.", "trap_type": "crowded", "severity": 0.6},
        {"symbol": "GME", "bear_case": "Retail pile-in.", "trap_type": "crowded", "severity": 0.8}]})
    cases = analyst.red_team(payload, {"analyst": {"model": "m", "red_team_max_web_searches": 0}})
    assert cases["AMD"]["severity"] == 0.0 and "unverifiable" in cases["AMD"]["bear_case"]
    assert cases["GME"]["severity"] == 0.8                          # real crowding data: untouched


def test_the_analyst_is_told_it_never_executes_trades():
    from desk.analyst import ANALYST_SYSTEM
    assert "Never write that a trade happened" in ANALYST_SYSTEM and "manual action" in ANALYST_SYSTEM

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


def test_reduce_on_a_held_position_is_noted_once_and_never_trades():
    # Trimming a holding would leave the broker-side stop covering more shares than we own, and the tested
    # strategy manages holdings by stop, target and decay. The concern is noted the first run it appears -
    # repeating it every run put the same advisory under "needs your action" daily.
    holdings = {"HLD": {"qty": 10}}
    reduce = [{"symbol": "HLD", "action": "REDUCE", "reason": "insider selling", "confidence": 0.5}]
    kept, exits, notes = apply_verdicts(reduce, holdings, [])
    assert kept == [] and exits == {}                     # never turns into a sale
    assert len(notes) == 1 and "HLD" in notes[0] and "insider selling" in notes[0]
    assert not notes[0].startswith("MANUAL ACTION")       # advisory, not an action item
    assert holdings["HLD"]["analyst_flag"] == "analyst REDUCE: insider selling"
    # The next run's REDUCE on the same holding is recorded silently, with the latest reason.
    _, _, notes = apply_verdicts([{**reduce[0], "reason": "still selling"}], holdings, [])
    assert notes == [] and holdings["HLD"]["analyst_flag"] == "analyst REDUCE: still selling"
    # A KEEP clears it, so a later concern is news again.
    apply_verdicts([{"symbol": "HLD", "action": "KEEP", "reason": "ok", "confidence": 0.5}], holdings, [])
    assert "analyst_flag" not in holdings["HLD"]
    _, _, notes = apply_verdicts(reduce, holdings, [])
    assert len(notes) == 1


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
        {"symbol": "AMD", "bear_case": "Everyone owns it.", "trap_type": "crowded", "severity": 0.6, "evidence": "social_z"},
        {"symbol": "GME", "bear_case": "Retail pile-in.", "trap_type": "crowded", "severity": 0.8, "evidence": "social_z: 4.1"}]})
    cases = analyst.red_team(payload, {"analyst": {"model": "m", "red_team_max_web_searches": 0}})
    assert cases["AMD"]["severity"] == 0.0 and "unverifiable" in cases["AMD"]["bear_case"]
    assert cases["GME"]["severity"] == 0.8                          # real crowding data: untouched


def test_the_analyst_is_told_it_never_executes_trades():
    from desk.analyst import ANALYST_SYSTEM
    assert "Never write that a trade happened" in ANALYST_SYSTEM and "concern on X (no trade)" in ANALYST_SYSTEM
    assert "manual action" not in ANALYST_SYSTEM          # a held REDUCE is not a request for the owner to trade


def test_an_uncited_bear_case_cannot_carry_a_veto_level_severity(monkeypatch):
    # ASST was scored "dilution 0.7" on 2026-09-24 with no offering or shelf filing in its fact sheet.
    from desk import analyst
    payload = {"holdings": [], "candidates": [{"symbol": "ASST", "facts": {"shelf_filings": []}},
                                              {"symbol": "TWST", "facts": {"shelf_filings": ["S-3ASR 2026-08-12"]}}]}
    monkeypatch.setattr(analyst, "_call_json", lambda *a, **k: {"cases": [
        {"symbol": "ASST", "bear_case": "Treasury model implies issuance.", "trap_type": "dilution", "severity": 0.7,
         "evidence": "none"},
        {"symbol": "TWST", "bear_case": "Live shelf.", "trap_type": "dilution", "severity": 0.7,
         "evidence": "shelf_filings: S-3ASR 2026-08-12"}]})
    cases = analyst.red_team(payload, {"analyst": {"model": "m", "red_team_max_web_searches": 0}})
    assert cases["ASST"]["severity"] == analyst.UNCITED_SEVERITY_CAP and "no cited evidence" in cases["ASST"]["bear_case"]
    assert cases["TWST"]["severity"] == 0.7                          # cited: untouched


def test_a_bear_case_resting_only_on_web_pages_is_capped_below_veto_level(monkeypatch):
    """10-08: the strongest GTLB/DDOG case was a year-old, unconfirmed bid report found on the web (F-31)."""
    from desk import analyst
    payload = {"holdings": [], "candidates": [{"symbol": s, "facts": {}} for s in ("GTLB", "ASST", "CORT")]}
    monkeypatch.setattr(analyst, "_call_json", lambda *a, **k: {"cases": [
        {"symbol": "GTLB", "bear_case": "Bid talk caps upside.", "trap_type": "other", "severity": 0.7,
         "evidence": "web: reuters.com, 2025-09-30; web: bloomberg.com, 2025-10-01"},
        {"symbol": "ASST", "bear_case": "ATM selling.", "trap_type": "dilution", "severity": 0.7,
         "evidence": "prospectus_supplements_12m: 424B5 2026-06-08; web: strive.com, 2026-10-01"},
        {"symbol": "CORT", "bear_case": "Minor.", "trap_type": "other", "severity": 0.4, "evidence": "web: x.com, 2026-10-01"}]})
    cases = analyst.red_team(payload, {"analyst": {"model": "m", "red_team_max_web_searches": 0}})
    assert cases["GTLB"]["severity"] == analyst.WEB_ONLY_SEVERITY_CAP and "web sources only" in cases["GTLB"]["bear_case"]
    assert cases["ASST"]["severity"] == 0.7                     # a fact-sheet field beside the web page: untouched
    assert cases["CORT"]["severity"] == 0.4                     # already below the cap
    assert analyst.web_only("web: a; web: b") and not analyst.web_only("shelf_filings: S-3 2026-01-01; web: a")
    assert not analyst.web_only("none") and not analyst.web_only("")


def test_the_ai_roles_are_told_the_desk_s_own_earnings_rule(cfg):
    # The analyst was told to veto "an earnings report inside the hold window", which is a 60-day block - the
    # setting the backtest rejected (MXL vetoed at 26 days out, 2026-09-25). Code blocks within the configured
    # days; the prompts must say later reports are allowed, and the payload must carry the number.
    from desk.analyst import ANALYST_SYSTEM, RED_TEAM_SYSTEM, analyst_rules
    r = analyst_rules(cfg)
    assert r["earnings_block_days"] == cfg["facts"]["exclude_earnings_within_days"]
    assert str(r["earnings_block_days"]) in r["earnings_policy"]
    assert "inside the hold window that the" not in ANALYST_SYSTEM
    assert "rules.earnings_block_days" in ANALYST_SYSTEM and "rules.earnings_block_days" in RED_TEAM_SYSTEM
    assert r["social_screen"].startswith("off") == (not cfg["facts"].get("social_enabled", True))
    assert "15%" in r["position_cap"] and "not a data concern" in r["position_cap"]   # UMC drift, 5 days running


def test_the_fact_sheet_says_how_many_headlines_it_left_out():
    """AMD 10-02: news_7d 14 and 14 dropped as unrelated beside 8 listed read as counts that "cannot all be
    right" (F-18). They were right: 14 related, the newest 8 shown."""
    from desk.facts import compact
    hl = [{"age_h": float(i), "source": "x", "headline": f"AMD item {i}"} for i in range(10)]
    c = compact({"symbol": "AMD", "news_7d": 14, "headlines_dropped_unrelated": 14, "headlines": hl})
    assert len(c["headlines"]) == 8 and c["headlines_not_shown"] == 6
    assert "headlines_not_shown" not in compact({"symbol": "X", "news_7d": None, "headlines": []})


def test_an_undated_web_page_counts_as_no_evidence(monkeypatch):
    """2026-10-15: 1 web citation, 0 dated, though the prompt requires a publication date (monitor F-31/F-34)."""
    from desk import analyst
    payload = {"holdings": [], "candidates": [{"symbol": s, "facts": {}} for s in ("IOVA", "FTNT", "CAKE", "MXL")]}
    monkeypatch.setattr(analyst, "_call_json", lambda *a, **k: {"cases": [
        {"symbol": "IOVA", "bear_case": "ATM.", "trap_type": "dilution", "severity": 0.6, "evidence": "web: sec.gov"},
        {"symbol": "FTNT", "bear_case": "Event.", "trap_type": "event-in-window", "severity": 0.6,
         "evidence": "web: reuters.com, 2026-10-12"},
        {"symbol": "CAKE", "bear_case": "Event.", "trap_type": "event-in-window", "severity": 0.6,
         "evidence": "earnings_date: 2026-10-27; web: cnbc.com"},
        {"symbol": "MXL", "bear_case": "Squeeze.", "trap_type": "squeeze", "severity": 0.6,
         "evidence": "web: a.com; web: b.com, May 2026"}]})
    events = []
    monkeypatch.setattr(analyst.telemetry, "agent_output", lambda role, *a, **k: events.append(k))
    cases = analyst.red_team(payload, {"analyst": {"model": "m", "red_team_max_web_searches": 1}})
    assert cases["IOVA"]["severity"] == analyst.UNCITED_SEVERITY_CAP and "web sources undated" in cases["IOVA"]["bear_case"]
    assert cases["FTNT"]["severity"] == analyst.WEB_ONLY_SEVERITY_CAP     # dated web only: the web cap
    assert cases["CAKE"]["severity"] == 0.6                               # a fact-sheet field still carries it
    assert cases["MXL"]["severity"] == analyst.WEB_ONLY_SEVERITY_CAP      # the dated page is left
    assert events[-1]["web_undated_capped"] == 1
    assert analyst.drop_undated_web("web: a.com; shelf_filings: S-3 2026-01-01; web: b.com, 2026-10-01") == \
        "shelf_filings: S-3 2026-01-01; web: b.com, 2026-10-01"
    assert "without its\npublication date counts as no evidence" in analyst.RED_TEAM_SYSTEM

"""Fixes from the 2026-11-16 health review: hold wording from config, no "absence" data concerns, a capped
shelf_filings list, owner-resolved findings marked in the email, and the real baseline date per period."""
from datetime import date

import pandas as pd

from desk.analyst import ANALYST_SYSTEM, RED_TEAM_SYSTEM, analyst_rules, holding_text
from desk.config import load_config
from desk.edgar import shelf_filings, shelf_tally
from desk.report import build_report, mark_resolved, owner_resolutions, performance


def test_prompts_carry_no_fixed_hold_or_take_profit():
    for p in (ANALYST_SYSTEM, RED_TEAM_SYSTEM):
        assert "15-60" not in p and "60 days" not in p and "take-profit orders" not in p
        assert "rules.holding" in p
    assert "small account" not in RED_TEAM_SYSTEM


def test_holding_text_follows_config():
    cfg = load_config()
    x = cfg["exits"]
    r = analyst_rules(cfg)
    assert r["holding"] == holding_text(cfg)
    assert f"{x['min_hold_days']} to {x['max_hold_days']} calendar days" in r["holding"]
    assert f"{x['min_hold_days']}-{x['max_hold_days']} day hold" in r["earnings_policy"]
    tp = {**cfg, "exits": {**x, "take_profit_r_multiple": 3.0, "trail_atr_multiple": 2}}
    assert "take-profit order at 3R" in holding_text(tp) and "2x ATR under the highest close" in holding_text(tp)
    if not float(x.get("take_profit_r_multiple") or 0):
        assert "no take-profit order" in r["holding"]


def test_data_concerns_are_problems_not_their_absence():
    assert "never their absence" in ANALYST_SYSTEM


def test_shelf_filings_capped_keeps_newest_of_each_form():
    notes = [{"form": "424B2", "date": date(2026, 11, 1) - pd.Timedelta(days=i).to_pytimedelta()} for i in range(300)]
    filings = notes + [{"form": "S-3ASR", "date": date(2026, 8, 20)}, {"form": "424B5", "date": date(2026, 9, 2)}]
    rows = shelf_filings(filings, 8)
    assert len(rows) == 8
    assert "S-3ASR 2026-08-20" in rows and "424B5 2026-09-02" in rows and rows[0] == "424B2 2026-11-01"
    assert [r.split()[1] for r in rows] == sorted((r.split()[1] for r in rows), reverse=True)
    assert shelf_tally(filings) == {"424B2": 300, "S-3ASR": 1, "424B5": 1}
    assert shelf_filings(filings[-2:]) == ["424B5 2026-09-02", "S-3ASR 2026-08-20"]     # no limit: all, newest first


def test_owner_resolutions_mark_carried_findings():
    res = owner_resolutions([
        "F-60 / R-67: fixed 2026-11-14 - the summarizer ...",
        "F-63 / R-70: not a defect 2026-11-14 - the lag ...",
        "F-58 / R-66: fixed 2026-11-06 (PR #53) - the analyst prompt ...",
        "F-53 / R-61: not a defect - no date here",
    ])
    assert res["F-60"] == ("fixed", "2026-11-14") and res["R-70"] == ("not a defect", "2026-11-14")
    assert res["R-66"] == ("fixed", "2026-11-06") and "F-53" not in res
    items = [{"id": "F-60"}, {"id": "F-58"}, {"id": "F-99"}]
    out = mark_resolved(items, res, "2026-11-13")
    assert out[0]["resolved"] == "fixed 2026-11-14"
    assert "resolved" not in out[1]          # resolved before the review that still carries it: the review knew
    assert "resolved" not in out[2] and "resolved" not in items[0]
    same_day = mark_resolved([{"id": "F-60"}], res, "2026-11-14T14:00:05-08:00")
    assert same_day[0]["resolved"] == "fixed 2026-11-14"     # fixed after the same day's review


def test_email_shows_owner_resolution():
    h = {"status": "healthy", "summary": "Rules-only check.", "analysis": "deterministic", "reviewed_on": "2026-11-13",
         "findings": [{"id": "F-60", "severity": "SEV3", "title": "Summarizer truncated", "trend": "ongoing",
                       "resolved": "fixed 2026-11-14"}],
         "recommendations": [{"id": "R-67", "priority": "P2", "title": "Size the budget", "resolved": "fixed 2026-11-14"}]}
    text = _render_health(h)
    assert "F-60 Summarizer truncated (ongoing) - owner: fixed 2026-11-14, awaiting the next model review" in text
    assert "R-67 Size the budget - owner: fixed 2026-11-14" in text


def _render_health(h):
    """The SYSTEM HEALTH block alone, through build_report with a minimal context."""
    ctx = {"day": date(2026, 11, 16), "equity": 100000.0, "day_pl": 0.0, "day_pct": 0.0, "cash": 5000.0,
           "positions": [], "unrealized": 0.0, "closed_today": [], "staged": [], "perf": [], "moves_plain": {},
           "movers": [], "news": {}, "quote_links": {}, "holdings": {"rows": [], "pnl": 0.0}, "changes": [],
           "halted": None, "notes": [], "health": h}
    return build_report(ctx)


def test_week_row_shows_the_close_it_is_measured_from():
    idx = pd.to_datetime(["2026-11-05", "2026-11-06", "2026-11-13", "2026-11-16"])
    eq, spy = pd.Series([100, 101, 102, 101.5], idx), pd.Series([10, 10.1, 10.3, 10.1], idx)
    rows = {r["period"]: r for r in performance(eq, spy, date(2026, 11, 16))}
    assert rows["This week"]["from"] == date(2026, 11, 13)       # Friday's close, not Sunday 11-15
    assert abs(rows["This week"]["desk_pct"] - (101.5 / 102 - 1)) < 1e-12


def test_held_concern_note_names_a_target_only_when_there_is_one():
    from desk.analyst import apply_verdicts
    v = [{"symbol": s, "action": "REDUCE", "reason": "insider selling", "confidence": 0.6} for s in ("NOTGT", "TGT")]
    holdings = {"NOTGT": {"stop": 10.0, "target": None}, "TGT": {"stop": 10.0, "target": 30.0}}
    _, exits, notes = apply_verdicts(v, holdings, [])
    assert not exits
    assert any("NOTGT (held; no trade - it keeps its stop):" in n for n in notes)
    assert any("TGT (held; no trade - it keeps its stop and target):" in n for n in notes)


def test_brief_takes_the_position_count_from_the_payload():
    assert "limits.positions_held" in ANALYST_SYSTEM
    assert '"positions_held": len(st["positions"])' in open("run_daily.py").read()

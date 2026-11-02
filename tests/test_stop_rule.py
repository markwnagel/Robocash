"""Quant review 2026-11-01: the stop rule the weekly audit checks (governance.stop_rule)."""
from datetime import date

import audit

RULE = {"benchmark": "SPMO", "warn_on": date(2027, 4, 30), "warn_trail_pts": 20,
        "decide_on": date(2027, 10, 29), "stop_trail_pts": 15}


def sr(live, bench, spy):
    return {"release": "v2.1", "from": "2026-10-30", "live_return": live, "benchmark": "SPMO",
            "benchmark_return": bench, "spy_return": spy}


def test_running_before_the_checkpoints_names_the_next_one():
    out = audit.stop_rule_status(RULE, sr(-0.30, 0.10, 0.05), date(2027, 1, 15))
    assert out["status"] == "RUNNING" and out["next_checkpoint"] == "2027-04-30"
    assert out["desk_vs_benchmark_pts"] == -40.0 and out["desk_vs_spy_pts"] == -35.0


def test_review_at_six_months_only_when_far_behind_the_benchmark():
    assert audit.stop_rule_status(RULE, sr(-0.15, 0.06, 0.02), date(2027, 5, 3))["status"] == "REVIEW"
    out = audit.stop_rule_status(RULE, sr(-0.05, 0.10, 0.02), date(2027, 5, 3))
    assert out["status"] == "RUNNING" and out["next_checkpoint"] == "2027-10-29"


def test_stop_needs_both_the_benchmark_gap_and_trailing_spy():
    assert audit.stop_rule_status(RULE, sr(0.02, 0.20, 0.08), date(2027, 11, 1))["status"] == "STOP"
    # 22pt behind SPMO but ahead of SPY: momentum ran away, the desk still beat the market - not a stop.
    assert audit.stop_rule_status(RULE, sr(0.10, 0.32, 0.08), date(2027, 11, 1))["status"] == "REVIEW"
    assert audit.stop_rule_status(RULE, sr(0.10, 0.20, 0.08), date(2027, 11, 1))["status"] == "RUNNING"


def test_missing_returns_are_unknown_and_no_rule_is_empty():
    assert audit.stop_rule_status(RULE, {}, date(2027, 11, 1))["status"] == "UNKNOWN"
    assert audit.stop_rule_status({}, sr(0.1, 0.1, 0.1), date(2027, 11, 1)) == {}


def test_the_live_config_carries_the_rule_and_the_audit_payload_includes_it(cfg, monkeypatch):
    live_cfg = cfg
    rule = live_cfg["governance"]["stop_rule"]
    assert rule["benchmark"] == "SPMO" and rule["stop_trail_pts"] == 15
    monkeypatch.setattr(audit.parity, "run", lambda cfg: {"since_release": sr(0.01, 0.02, 0.0)})
    monkeypatch.setattr(audit, "api_spend", lambda today, cfg: {})
    monkeypatch.setattr(audit, "changes_since", lambda since: [])
    ctx = audit.current_context(live_cfg, date(2026, 11, 7))
    assert ctx["stop_rule"]["status"] == "RUNNING" and ctx["stop_rule"]["desk_vs_benchmark_pts"] == -1.0
    assert "stop_rule" in audit.AUDIT_SYSTEM

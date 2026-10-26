"""The audit's current context (2026-10-24): it quoted a three-week-old parity run and v1 costs from config
comments, read the spend alarm as a cap, and missed a bug fix merged the day before."""
import json
from datetime import date

import audit


def test_spend_is_measured_from_telemetry_and_labelled_an_alarm_not_a_cap(tmp_path, monkeypatch):
    from desk import telemetry
    d = tmp_path / "telemetry"; d.mkdir()
    monkeypatch.setattr(telemetry, "DIR", d)
    calls = [("2026-10-01", "desk", 3.0), ("2026-10-19", "desk", 0.5), ("2026-10-20", "monitor", 1.0),
             ("2026-10-23", "desk", 0.25), ("2026-09-30", "desk", 9.0)]
    for day, job, usd in calls:
        with open(d / f"{day}.jsonl", "a") as f:
            f.write(json.dumps({"ts": f"{day}T20:00:00+00:00", "event": "llm_call", "job": job, "cost_usd": usd,
                                "web_search_usd": 0.0}) + "\n")
    s = audit.api_spend(date(2026, 10, 24), {"budget": {"monthly_api_usd": 40}})
    assert s["month_to_date_usd"] == 4.75 and s["alarm_threshold_usd"] == 40       # September is not this month
    assert s["per_trading_day_usd_last_5"] == {"desk": 0.15, "monitor": 0.2}       # 10-19..10-23
    assert "does not cap" in s["note"]


def test_the_last_audit_sets_the_change_window(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "STATE_DIR", tmp_path)
    assert audit.last_audit_date(date(2026, 10, 24)) == date(2026, 10, 17)          # none yet: a week back
    (tmp_path / "audits").mkdir()
    for d in ("2026-10-10", "2026-10-17", "2026-10-24"):
        (tmp_path / "audits" / f"{d}.md").write_text("x")
    assert audit.last_audit_date(date(2026, 10, 24)) == date(2026, 10, 17)          # not today's own file
    assert all(" " in c for c in audit.changes_since(date(2026, 10, 1))[:3])        # "date hash subject"


def test_the_context_survives_a_parity_failure_and_carries_owner_decisions(monkeypatch):
    def boom(cfg):
        raise RuntimeError("no bars")
    monkeypatch.setattr(audit.parity, "run", boom)
    cfg = {"parity": {"enabled": True}, "budget": {"monthly_api_usd": 40}, "monitor": {"owner_decisions": ["a"]}}
    c = audit.current_context(cfg, date(2026, 10, 24))
    assert c["parity"] == {"error": "RuntimeError: no bars"} and c["owner_decisions"] == ["a"]
    assert set(c) == {"changes_since_last_audit", "parity", "api_spend", "owner_decisions"}
    assert "it does not limit or reduce spending" in audit.AUDIT_SYSTEM

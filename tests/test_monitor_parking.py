"""Parked findings (2026-10-23): a SEV3/4 finding with no new evidence for 3 reviews leaves the daily email."""
from desk.monitor import MONITOR_SCHEMA, parked
from desk.report import build_report  # noqa: F401  (import check)


def test_parking_needs_three_quiet_reviews_and_a_low_severity():
    assert parked({"severity": "SEV4", "quiet_windows": 3})
    assert not parked({"severity": "SEV4", "quiet_windows": 2})
    assert not parked({"severity": "SEV2", "quiet_windows": 9})             # serious findings never park
    assert not parked({"severity": "SEV3"})                                 # older state has no counter
    assert "quiet_windows" in MONITOR_SCHEMA["properties"]["findings"]["items"]["required"]


def test_the_email_drops_parked_findings_and_counts_them(tmp_path, monkeypatch):
    import json
    from datetime import date
    import report_daily
    (tmp_path / "monitor").mkdir()
    (tmp_path / "monitor" / "latest.json").write_text(json.dumps({"date": "2026-10-26", "status": "healthy", "findings": [
        {"id": "F-51", "severity": "SEV3", "title": "a", "quiet_windows": 0},
        {"id": "F-38", "severity": "SEV4", "title": "b", "quiet_windows": 4},
        {"id": "F-33", "severity": "SEV4", "title": "c", "quiet_windows": 3}]}))
    monkeypatch.setattr(report_daily, "STATE_DIR", tmp_path)
    h = report_daily.health_today(date(2026, 10, 26))
    assert [f["id"] for f in h["findings"]] == ["F-51"] and h["parked"] == 2


def test_owner_decisions_reach_the_model_review():
    from desk.config import load_config
    od = load_config()["monitor"]["owner_decisions"]
    assert any("offering_block" in d for d in od) and any("red_team_scope" in d for d in od)
    assert "owner_decisions" in open("monitor.py").read()

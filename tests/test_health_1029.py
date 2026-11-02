"""2026-10-29 health review: open findings F-30, F-33, F-40/R-48, R-42 and the email's carried findings."""
import json
from datetime import date

from desk import telemetry
from desk.facts import CLASSIFIER_SCHEMA, CLASSIFIER_SYSTEM, earnings_conflict
from desk.monitor import api_access_alarms, summary_for_email
from desk.report import build_report
from desk.report_html import build_html
from tests.test_report import NOW


def _events():
    return [json.loads(x) for p in sorted(telemetry.DIR.glob("*.jsonl")) for x in p.read_text().splitlines()]


def test_a_spend_limit_refusal_is_named_and_still_alarms():
    import time
    telemetry.llm_call("auditor", "claude-opus-5", time.monotonic(), outcome="error",
                       error="BadRequestError: Error code: 400 - You have reached your specified API usage limits")
    telemetry.llm_call("analyst", "claude-opus-5", time.monotonic(), outcome="error", error="APIConnectionError: timeout")
    ev = _events()
    assert [e["outcome"] for e in ev if e["event"] == "llm_call"] == ["api_access", "error"]
    a = api_access_alarms(ev)
    assert len(a) == 1 and a[0]["severity"] == "SEV1" and "auditor" in a[0]["detail"]
    old = [{"event": "llm_call", "ts": "2026-10-18T01:00:00+00:00", "role": "auditor", "outcome": "error",
            "error": "You have reached your specified API usage limits"}]               # recorded before the rename
    assert len(api_access_alarms(old)) == 1


def test_a_next_report_too_close_to_the_last_results_is_flagged():
    assert "earnings_date_conflict" in earnings_conflict("2026-10-30", "2026-10-22")       # 8 days apart
    assert earnings_conflict("2027-01-28", "2026-10-22") == {}                              # a quarter on
    assert earnings_conflict(None, "2026-10-22") == {} and earnings_conflict("2026-10-30", None) == {}


def test_the_classifier_reports_unrelated_headlines():
    item = CLASSIFIER_SCHEMA["properties"]["results"]["items"]
    assert "unrelated" in item["required"] and item["properties"]["unrelated"]["type"] == "integer"
    assert "unrelated" in CLASSIFIER_SYSTEM


def test_the_email_keeps_quiet_windows_and_dates_carried_findings():
    result = {"status": "healthy", "summary": "Rules-only check.", "findings": [
        {"id": "F-51", "severity": "SEV3", "title": "gap", "trend": "new", "quiet_windows": 0},
        {"id": "F-38", "severity": "SEV4", "title": "repost", "trend": "ongoing", "quiet_windows": 4}],
        "recommendations": []}
    h = summary_for_email(result, "r.md", "deterministic", reviewed_on="2026-10-23")
    assert [f["quiet_windows"] for f in h["findings"]] == [0, 4] and h["reviewed_on"] == "2026-10-23"
    ctx = {"day": date(2026, 10, 29), "equity": 9300.0, "day_pl": 0.0, "day_pct": 0.0, "cash": 3000.0, "positions": [],
           "unrealized": 0.0, "closed_today": [], "staged": [], "perf": [], "movers": {"notable": [], "rest": 0},
           "news": [], "changes": [], "halted": None, "notes": [], "now": NOW, "moves_plain": {},
           "health": {**h, "findings": h["findings"][:1], "parked": 1}}
    text, html = build_report(ctx), build_html(ctx)
    assert "as of the 2026-10-23 model review" in text and "as of the 2026-10-23 model review" in html
    assert "+ 1 parked" in text
    ctx["health"]["analysis"] = "claude"                            # a review day's findings are today's
    assert "as of the" not in build_report(ctx)


def test_owner_resolutions_reach_the_model_review():
    from desk.config import load_config
    res = load_config()["monitor"]["owner_resolutions"]
    assert any(r.startswith("F-41") for r in res) and any(r.startswith("F-47") for r in res)
    assert '"owner_resolutions"' in open("monitor.py").read()
    assert "owner_resolutions" in open("agent-monitor-prompt.md").read()

"""Telemetry and the agent reliability monitor: what gets recorded, and what the monitor concludes from it."""
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from desk import monitor as M
from desk import telemetry as T

ROOT = Path(__file__).resolve().parent.parent


def _events():
    return T.load_events(datetime(2000, 1, 1, tzinfo=timezone.utc), datetime(2100, 1, 1, tzinfo=timezone.utc))


def _resp(inp=1000, out=500, stop="end_turn", model="claude-opus-5", searches=0):
    usage = SimpleNamespace(input_tokens=inp, output_tokens=out, cache_creation_input_tokens=0,
                            cache_read_input_tokens=0, server_tool_use=SimpleNamespace(web_search_requests=searches))
    return SimpleNamespace(usage=usage, stop_reason=stop, model=model, content=[])


# ------------------------------------------------------------------ telemetry

def test_run_job_records_start_end_and_warnings_under_one_run_id():
    def job():
        T.note(entries=3)
        logging.getLogger("desk.facts").warning("options %s: timed out", "AAPL")
        logging.getLogger("desk.facts").info("routine")                       # below WARNING: not recorded
    T.run_job("desk", job)
    ev = _events()
    kinds = [e["event"] for e in ev]
    assert kinds == ["run_start", "log", "run_end"]
    assert len({e["run_id"] for e in ev}) == 1 and all(e["job"] == "desk" for e in ev)
    end = ev[-1]
    assert end["status"] == "ok" and end["entries"] == 3 and end["duration_s"] >= 0
    assert ev[1]["signature"] == "options <SYM>: timed out"


def test_a_crash_is_recorded_then_reraised_unchanged():
    def boom():
        raise RuntimeError("broker 503 token=abc123")
    with pytest.raises(RuntimeError, match="broker 503"):
        T.run_job("desk", boom)
    end = _events()[-1]
    assert end["event"] == "run_end" and end["status"] == "error"
    assert "RuntimeError: broker 503" in end["error"] and "abc123" not in end["error"]   # redacted
    assert "test_monitor.py" in end["error"]                                          # where it happened


def test_status_set_by_the_job_wins_over_ok():
    T.run_job("desk", lambda: T.note(status="skipped", reason="not a trading day"))
    assert _events()[-1]["status"] == "skipped"


def test_secrets_never_reach_telemetry():
    for secret in ("sk-ant-api03-SECRETVALUE", "https://finnhub.io/x?symbol=A&token=SECRETVALUE",
                   "Authorization: Bearer SECRETVALUE", "api_key=SECRETVALUE"):
        assert "SECRETVALUE" not in T.redact(secret), secret


def test_signature_counts_one_problem_once():
    a = T.signature("no bars for 1 symbol(s), retrying once: TEVA")
    b = T.signature("no bars for 3 symbol(s), retrying once: AAPL, MSFT")
    assert T.signature("run 3f9a2c1b7d8e failed") == T.signature("run 0a1b2c3d4e5f failed")
    assert a.startswith("no bars for # symbol(s), retrying once") and b.startswith(a[: a.index(":")])


def test_llm_call_prices_tokens_and_leaves_unknown_models_unpriced():
    started = __import__("time").monotonic()
    T.llm_call("analyst", "claude-opus-5", started, [_resp(1000, 1000, searches=2), _resp(2000, 0)])
    T.llm_call("summarizer", "some-new-model", started, [_resp(model="some-new-model")])
    a, s = _events()[-2:]
    # 3000 input x $5/M + 1000 output x $25/M, summed over both pause_turn responses.
    assert a["input_tokens"] == 3000 and a["output_tokens"] == 1000 and a["turns"] == 2 and a["web_searches"] == 2
    assert a["cost_usd"] == pytest.approx(0.015 + 0.025)
    assert s["cost_usd"] is None                      # unknown price stays unknown, never zero


def test_a_fallback_answer_is_marked_and_priced_at_the_serving_model():
    T.llm_call("monitor", "claude-opus-5", 0.0, [_resp(1000, 0, model="claude-opus-4-8")])
    e = _events()[-1]
    assert e["fallback"] is True and e["served_by"] == "claude-opus-4-8" and e["cost_usd"] == pytest.approx(0.005)


# ------------------------------------------------------------------ schedules and runs

PT = timezone(timedelta(hours=-7))


def test_launchd_weekdays_map_to_python_weekdays():
    plist = {"StartCalendarInterval": [{"Weekday": 1, "Hour": 16, "Minute": 45},     # Monday
                                       {"Weekday": 0, "Hour": 9, "Minute": 0},       # Sunday (0)
                                       {"Weekday": 7, "Hour": 9, "Minute": 0}]}      # Sunday (7)
    assert M.schedule_slots(plist) == [(0, 16, 45), (6, 9, 0)]


def test_a_missed_desk_slot_on_a_trading_day_is_a_hard_alarm():
    slots = [(d, 16, 45) for d in range(5)]
    start, end = datetime(2026, 9, 21, 14, 0, tzinfo=PT), datetime(2026, 9, 23, 14, 0, tzinfo=PT)   # Mon -> Wed
    exp = M.expected_runs(slots, start, end, timedelta(minutes=60))
    assert [t.day for t in exp] == [21, 22]
    runs = {"desk": M.match_runs(exp, [datetime(2026, 9, 21, 16, 45, 6, tzinfo=PT)])}
    assert [t.day for t in runs["desk"]["missed"]] == [22]
    alarms = M.hard_alarms(runs, {}, {date(2026, 9, 21), date(2026, 9, 22)})
    assert len(alarms) == 1 and alarms[0]["severity"] == "SEV1" and "2026-09-22" in alarms[0]["title"]
    # A market holiday is not a missed trading day.
    assert M.hard_alarms(runs, {}, {date(2026, 9, 21)}) == []


def test_a_late_run_still_counts_and_reports_how_late():
    t = datetime(2026, 9, 15, 13, 45, tzinfo=PT)
    m = M.match_runs([t], [t + timedelta(minutes=13, seconds=36)])
    assert not m["missed"] and m["matched"][0]["late_min"] == 13.6


def _run_events(job, rid, start, end=None, **end_fields):
    ev = [{"ts": start.isoformat(), "run_id": rid, "job": job, "event": "run_start"}]
    if end:
        ev.append({"ts": end.isoformat(), "run_id": rid, "job": job, "event": "run_end", **end_fields})
    return ev


def test_crashed_and_orphaned_desk_runs_are_hard_alarms_but_a_running_one_is_not():
    now = datetime(2026, 9, 24, 14, 0, tzinfo=timezone.utc)
    ev = (_run_events("desk", "a", now - timedelta(hours=20), now - timedelta(hours=19.9), status="error",
                      error="KeyError: 'stop'", duration_s=5)
          + _run_events("desk", "b", now - timedelta(hours=5))             # never ended: orphaned
          + _run_events("desk", "c", now - timedelta(minutes=10)))         # still running
    jobs = M.job_signals(ev, now)
    assert jobs["desk"]["status"] == {"error": 1} and len(jobs["desk"]["orphaned"]) == 1
    assert len(jobs["desk"]["in_progress"]) == 1
    keys = {a["key"] for a in M.hard_alarms({}, jobs, None)}
    assert keys == {"desk-error-a", "desk-orphan-b"}


# ------------------------------------------------------------------ signals

def test_role_signals_measure_errors_coverage_and_contract_breaks():
    ev = [
        {"event": "llm_call", "role": "analyst", "outcome": "ok", "latency_s": 40, "cost_usd": 0.3,
         "input_tokens": 30000, "output_tokens": 2000, "turns": 2},
        {"event": "llm_call", "role": "analyst", "outcome": "error", "latency_s": 1, "cost_usd": None, "turns": 0,
         "error": "APIConnectionError"},
        {"event": "agent_output", "role": "analyst", "requested": 10, "returned": 9, "missing": ["TEAM"],
         "brief_over_limit": True},
        {"event": "agent_output", "role": "analyst", "requested": 10, "returned": None},
    ]
    s = M.role_signals(ev)["analyst"]
    assert s["calls"] == 2 and s["error_rate"] == 0.5 and s["cost_unknown_calls"] == 1
    assert s["coverage"] == 0.45 and s["no_answer"] == 1 and s["missing_examples"] == ["TEAM"]
    assert s["contract"] == {"brief_over_limit": 1} and s["max_turns"] == 2


def test_delta_uses_robust_statistics():
    base = M.robust([10, 11, 9, 10, 12, 10, 60])            # one outlier barely moves the median
    assert base["median"] == 10
    d = M.delta(M.robust([20]), base)
    assert d["rel_change"] == 1.0 and d["robust_z"] > 3


def test_new_signatures_are_flagged_against_the_baseline():
    base = [{"event": "log", "job": "desk", "signature": "options <SYM>: timed out", "ts": "2026-09-20T00:00:00+00:00"}]
    win = [{"event": "log", "job": "desk", "signature": "options <SYM>: timed out", "level": "WARNING",
            "message": "options AAPL: timed out", "ts": "2026-09-24T00:00:00+00:00", "run_id": "r1"},
           {"event": "log", "job": "desk", "signature": "STILL no bars for # symbol(s)", "level": "WARNING",
            "message": "STILL no bars for 1 symbol(s)", "ts": "2026-09-24T00:00:01+00:00", "run_id": "r1"}]
    rows = M.log_signatures(win, base)
    assert [r["new"] for r in rows] == [True, False] and rows[0]["signature"].startswith("STILL")


def test_brief_notes_are_grouped_so_soft_failures_surface():
    brief = ("DESK BRIEF\nNotes:\n  - AMD protection UNVERIFIED: broker order query failed - check manually\n"
             "  - TEAM protection UNVERIFIED: broker order query failed - check manually\n  - data through 2026-09-23\n")
    rows = M.brief_signatures({"2026-09-24": brief}, {"2026-09-20": "Notes:\n  - data through 2026-09-19\n"})
    top = rows[0]
    assert top["count"] == 2 and top["new"] and "protection UNVERIFIED" in top["example"]
    assert next(r for r in rows if "data through" in r["example"])["new"] is False


def test_incident_days_are_kept_out_of_the_baseline():
    ev = [{"event": "run_start", "ts": "2026-09-20T18:00:00+00:00"}, {"event": "run_start", "ts": "2026-09-21T18:00:00+00:00"}]
    v = M.baseline_view(ev, {"2026-09-21"})
    assert v["runs"] == 1 and v["windows"] == 1


# ------------------------------------------------------------------ prompt, fallback, email

def test_the_users_prompt_is_used_verbatim_apart_from_its_config_block():
    template = (ROOT / "agent-monitor-prompt.md").read_text()
    cfg_yaml = M.render_config({"agents": [{"name": "desk-job"}], "baseline_days": 7}, "owner@example.com", "/r")
    out = M.build_system_prompt(template, cfg_yaml)
    assert "{{agent_name}}" not in out and "name: desk-job" in out and "owner@example.com" in out
    head, tail = template.split("## Configuration")[0], template.split("If a field is missing")[1]
    assert out.startswith(head) and out.endswith(tail)
    with pytest.raises(ValueError):
        M.build_system_prompt("no config block here", cfg_yaml)


def test_the_monitor_schema_is_strict_everywhere():
    def walk(s):
        if s.get("type") == "object":
            assert s.get("additionalProperties") is False and set(s["required"]) == set(s["properties"])
            for v in s["properties"].values():
                walk(v)
        if s.get("type") == "array":
            walk(s["items"])
    walk(M.MONITOR_SCHEMA)


def test_fallback_report_still_states_the_alarms_and_numbers():
    payload = {"run": {"window_start": "a", "window_end": "b"},
               "hard_alarms": [{"severity": "SEV1", "title": "Desk job crashed", "detail": "KeyError"}],
               "runs": {"desk": {"matched": [1], "missed": []}}, "jobs": {"desk": {"status": {"error": 1}}},
               "signals": {"analyst": {"calls": 1, "error_rate": 0.0, "latency_s": {"p95": 40.0},
                                       "cost_total_usd": 0.31, "coverage": 1.0}},
               "coverage": {"telemetry": "files 2026-09-24..2026-09-24"}}
    text = M.fallback_report(payload, "APIConnectionError")
    assert "Incident" in text and "Desk job crashed" in text and "| analyst | 1 |" in text and "$0.31" in text


def test_email_shows_health_and_says_so_when_the_monitor_did_not_run():
    sys.path.insert(0, str(ROOT / "tests"))
    from test_report import _ctx
    from desk.report import build_report
    from desk.report_html import build_html
    health = {"status": "degraded", "summary": "Analyst coverage fell.", "report": "state/monitor/reports/x.md",
              "findings": [{"id": "F-2", "severity": "SEV2", "title": "Analyst skips holdings <b>", "trend": "new"}],
              "recommendations": [{"id": "R-1", "priority": "P1", "title": "Validate verdict coverage"}]}
    text, html = build_report(_ctx(health=health)), build_html(_ctx(health=health))
    assert "SYSTEM HEALTH" in text and "[SEV2] F-2 Analyst skips holdings" in text and "R-1" in text
    # The title is model-written text: it must arrive escaped, never as markup.
    assert "System health" in html and "Analyst skips holdings &lt;b&gt;" in html
    assert "Analyst skips holdings <b>" not in html
    missing = {"status": "missing", "summary": "The agent monitor did not run today - no health check.",
               "findings": [], "recommendations": []}
    assert "did not run today" in build_report(_ctx(health=missing))


def test_report_job_treats_a_stale_health_file_as_missing(tmp_path, monkeypatch):
    import report_daily
    monkeypatch.setattr(report_daily, "STATE_DIR", tmp_path)
    (tmp_path / "monitor").mkdir()
    (tmp_path / "monitor" / "latest.json").write_text(json.dumps({"date": "2026-09-01", "status": "healthy"}))
    assert report_daily.health_today(date(2026, 9, 24))["status"] == "missing"
    (tmp_path / "monitor" / "latest.json").write_text(json.dumps({"date": "2026-09-24", "status": "healthy"}))
    assert report_daily.health_today(date(2026, 9, 24))["status"] == "healthy"


# ------------------------------------------------------------------ the job end to end (no network)

class _Stream:
    def __init__(self, resp):
        self.resp = resp

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.resp


def _fake_anthropic(result=None, stop="end_turn", seen=None):
    text = json.dumps(result) if result is not None else ""

    class Client:
        def __init__(self, *a, **k):
            self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self.stream))

        def stream(self, **kw):
            if seen is not None:
                seen.update(kw)
            r = _resp(20000, 3000, stop=stop)
            r.content = [SimpleNamespace(type="text", text=text)]
            return _Stream(r)
    return Client


@pytest.fixture
def mon(tmp_path, monkeypatch):
    import monitor as job
    monkeypatch.setattr(job, "STATE_DIR", tmp_path)
    monkeypatch.setattr(job, "MON_DIR", tmp_path / "monitor")
    monkeypatch.setattr(job, "installed_schedules",
                        lambda: {"desk": [(d, 16, 45) for d in range(5)], "report": [(d, 14, 0) for d in range(5)],
                                 "audit": [(5, 9, 0)]})
    monkeypatch.setattr(job, "trading_days", lambda cfg, s, e: None)
    sent = []
    monkeypatch.setattr(job, "send_email", lambda subject, body, cfg, html=None: sent.append((subject, body)) or "emailed x")
    return job, sent, tmp_path


def _run_monitor(job, argv):
    sys.argv = ["monitor.py"] + argv
    job.main()


def _seed_telemetry(hours_ago: float) -> None:
    t = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    T.DIR.mkdir(parents=True, exist_ok=True)
    with open(T.DIR / f"{t.date().isoformat()}.jsonl", "a") as f:
        f.write(json.dumps({"ts": t.isoformat(timespec="seconds"), "run_id": "old", "job": "report",
                            "event": "run_start"}) + "\n")


def test_runs_scheduled_before_telemetry_existed_are_not_called_missed(mon, monkeypatch):
    """The first monitor run after the rollout saw no record of the previous evening's desk run and raised a
    SEV1 'did not run' - the run had happened; recording had not."""
    job, sent, tmp = mon
    slot = datetime.now().astimezone() - timedelta(hours=2)
    monkeypatch.setattr(job, "installed_schedules",
                        lambda: {"desk": [(d, slot.hour, slot.minute) for d in range(7)], "report": None, "audit": None})
    _run_monitor(job, ["--no-llm"])                                    # telemetry starts with this very run
    latest = json.loads((tmp / "monitor" / "latest.json").read_text())
    assert latest["findings"] == [] and sent == []


def test_monitor_without_the_model_still_escalates_a_missed_desk_run(mon, monkeypatch):
    job, sent, tmp = mon
    slot = datetime.now().astimezone() - timedelta(hours=2)          # always inside the window, past its grace
    monkeypatch.setattr(job, "installed_schedules",
                        lambda: {"desk": [(d, slot.hour, slot.minute) for d in range(7)], "report": None, "audit": None})
    _seed_telemetry(hours_ago=3)                                       # recording was on before the slot
    _run_monitor(job, ["--no-llm"])                                    # ...and no desk run was recorded
    latest = json.loads((tmp / "monitor" / "latest.json").read_text())
    report = (tmp / "monitor" / "reports" / f"{date.today().isoformat()}.md").read_text()
    assert latest["analysis"] == "deterministic" and latest["status"] == "incident"
    assert latest["findings"][0]["severity"] == "SEV1" and "did not run" in latest["findings"][0]["title"]
    assert "did not run" in report and len(sent) == 1 and sent[0][0].startswith("[SEV1] TeamRobo monitor:")
    st = json.loads((tmp / "monitor" / "state.json").read_text())
    assert st["last_run"] and "fingerprint" in st and st["incident_days"] == [date.today().isoformat()]

    # Next run: the window starts where the last one ended, so the same missed slot is neither re-counted
    # nor re-emailed.
    _run_monitor(job, ["--no-llm"])
    assert len(sent) == 1


def test_monitor_uses_the_models_report_and_the_fallback_opt_in(mon, monkeypatch):
    job, sent, tmp = mon
    result = {"status": "degraded", "summary": "Analyst dropped one holding.", "report_markdown": "# Agent Reliability Report\n...",
              "findings": [{"id": "F-1", "signature": "analyst-missing-holding", "title": "Analyst skipped TEAM",
                            "severity": "SEV3", "trend": "new", "agents": ["analyst"]}],
              "recommendations": [{"id": "R-1", "addresses": ["F-1"], "priority": "P2", "title": "Check verdict coverage",
                                   "status": "open"}],
              "resolved_finding_ids": [], "escalate": False, "escalation_summary": ""}
    seen = {}
    monkeypatch.setattr(job.anthropic, "Anthropic", _fake_anthropic(result, seen=seen))
    monkeypatch.setattr(job, "hard_alarms", lambda runs, jobs, tdays, events=None: [])
    _run_monitor(job, [])
    assert seen["fallbacks"] == "default" and seen["betas"] == ["server-side-fallback-2026-07-01"]
    assert seen["output_config"]["format"]["schema"] is M.MONITOR_SCHEMA
    assert "{{agent_name}}" not in seen["system"] and "desk-job" in seen["system"]
    latest = json.loads((tmp / "monitor" / "latest.json").read_text())
    assert latest["status"] == "degraded" and latest["analysis"] == "claude" and latest["findings"][0]["id"] == "F-1"
    st = json.loads((tmp / "monitor" / "state.json").read_text())
    assert st["next_finding"] == 2 and st["next_recommendation"] == 2 and st["open_findings"][0]["id"] == "F-1"
    assert sent == []                                       # SEV3 is never emailed on its own
    calls = [e for e in _events() if e.get("event") == "llm_call" and e["role"] == "monitor"]
    assert calls and calls[-1]["outcome"] == "ok" and calls[-1]["cost_usd"] == pytest.approx(0.1 + 0.075)


def test_monitor_falls_back_when_the_model_refuses(mon, monkeypatch):
    job, sent, tmp = mon
    monkeypatch.setattr(job.anthropic, "Anthropic", _fake_anthropic(None, stop="refusal"))
    monkeypatch.setattr(job, "hard_alarms", lambda runs, jobs, tdays, events=None: [])
    _run_monitor(job, [])
    latest = json.loads((tmp / "monitor" / "latest.json").read_text())
    assert latest["analysis"] == "deterministic" and "refused" in latest["summary"]
    assert "deterministic fallback" in (tmp / "monitor" / "reports" / f"{date.today().isoformat()}.md").read_text()


# ------------------------------------------------------------------ review round 3

def test_jobs_keep_their_console_log(monkeypatch):
    """basicConfig is a no-op once root has a handler. Attaching the telemetry handler first silenced every
    job's launchd log (09-24 15:06 desk run: 0 lines)."""
    order = []
    real = logging.basicConfig
    monkeypatch.setattr(logging, "basicConfig",
                        lambda **k: order.append(("basicConfig", k.get("level"),
                                                  any(isinstance(h, T._LogHandler) for h in logging.getLogger().handlers))))
    T.run_job("desk", lambda: None)
    assert order and order[0] == ("basicConfig", logging.INFO, False)   # configured before our handler existed
    monkeypatch.setattr(logging, "basicConfig", real)


def test_a_crash_after_a_noted_status_still_reads_as_a_crash():
    def job():
        T.note(status="halted")
        raise OSError("disk full")
    with pytest.raises(OSError):
        T.run_job("desk", job)
    end = _events()[-1]
    assert end["status"] == "error" and end["noted_status"] == "halted"


def test_expected_slots_follow_daylight_saving(monkeypatch):
    import time as _t
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    _t.tzset()
    try:
        start = datetime(2027, 3, 12, 12, 0).astimezone()        # Friday, PST
        end = datetime(2027, 3, 16, 12, 0).astimezone()          # Tuesday, PDT (DST began Sun 03-14)
        slots = M.expected_runs([(0, 9, 30), (4, 13, 45)], start, end, timedelta(minutes=5))
        assert [(t.day, t.hour, t.minute) for t in slots] == [(12, 13, 45), (15, 9, 30)]
        assert slots[0].utcoffset() != slots[1].utcoffset()        # each slot on its own date's clock
    finally:
        monkeypatch.delenv("TZ")
        _t.tzset()


def test_a_failed_sev1_email_is_retried_next_run(mon, monkeypatch):
    job, sent, tmp = mon
    slot = datetime.now().astimezone() - timedelta(hours=2)
    monkeypatch.setattr(job, "installed_schedules",
                        lambda: {"desk": [(d, slot.hour, slot.minute) for d in range(7)], "report": None, "audit": None})
    _seed_telemetry(hours_ago=3)
    attempts = []
    monkeypatch.setattr(job, "send_email", lambda s, b, c, html=None: attempts.append(s) or "SMTP FAILED (auth)")
    _run_monitor(job, ["--no-llm"])
    st = json.loads((tmp / "monitor" / "state.json").read_text())
    assert len(attempts) == 1 and st["escalated"] == [] and len(st["pending_escalations"]) == 1

    # Gmail is back. The missed slot is now outside the window, so only the pending list can deliver it.
    monkeypatch.setattr(job, "send_email", lambda s, b, c, html=None: attempts.append(s) or "emailed x")
    _run_monitor(job, ["--no-llm"])
    st = json.loads((tmp / "monitor" / "state.json").read_text())
    assert len(attempts) == 2 and "did not run" in attempts[1]
    assert st["pending_escalations"] == [] and len(st["escalated"]) == 1
    _run_monitor(job, ["--no-llm"])
    assert len(attempts) == 2                                          # delivered once, never again

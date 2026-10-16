"""Telemetry and the agent reliability monitor: what gets recorded, and what the monitor concludes from it."""
import json
import time
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

def test_the_monitor_sees_the_commits_in_its_window(tmp_path):
    import subprocess
    assert M.commits_since(tmp_path, datetime(2026, 1, 1, tzinfo=timezone.utc)) == []     # not a checkout
    git = lambda *a: subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                                    check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / "f").write_text("x")
    git("add", "f")
    git("commit", "-q", "-m", "Run the desk at the 13:40 wake")
    got = M.commits_since(tmp_path, datetime.now(timezone.utc) - timedelta(days=1))
    assert [c["subject"] for c in got] == ["Run the desk at the 13:40 wake"] and len(got[0]["sha"]) >= 7
    assert M.commits_since(tmp_path, datetime.now(timezone.utc) + timedelta(days=1)) == []


def test_a_call_records_what_server_tools_added_to_its_context():
    """10-06: analyst input 54k tokens vs 25k, same prompt, web_search_requests 0 - nothing recorded said why."""
    from types import SimpleNamespace as NS

    class Block(NS):
        def model_dump(self):
            return dict(self.__dict__)
    resp = NS(model="claude-opus-5", stop_reason="end_turn",
              usage=NS(input_tokens=54458, output_tokens=6255, cache_creation_input_tokens=0, cache_read_input_tokens=0,
                       server_tool_use=NS(web_search_requests=0, web_fetch_requests=2)),
              content=[Block(type="server_tool_use", name="web_fetch"), Block(type="web_fetch_tool_result", content="x" * 500),
                       Block(type="server_tool_use", name="web_fetch"), Block(type="text", text="{}")])
    T.llm_call("analyst", "claude-opus-5", time.monotonic(), [resp])
    e = [x for x in _events() if x["event"] == "llm_call"][-1]
    assert e["web_fetches"] == 2 and e["web_searches"] == 0
    assert e["server_blocks"] == {"server_tool_use:web_fetch": 2, "web_fetch_tool_result": 1}
    assert e["server_result_chars"] > 500


def test_the_monitor_gets_numbers_for_the_headline_filter_and_the_web_cap():
    """R-38: both fixes were being verified from the analyst's wording in the brief."""
    from datetime import date
    from desk.analyst import web_citation_stats
    w = web_citation_stats(["web: reuters.com, 2025-10-01; shelf_registration: S-3ASR 2025-09-15", "web: strive.com",
                            "none", "web: bloomberg.com, 2026-10-05"], today=date(2026, 10, 12))
    assert w == {"web_citations": 3, "web_citations_dated": 2, "web_citation_max_age_days": 376}
    ev = [{"event": "agent_output", "role": "red_team", "requested": 12, "returned": 12, "web_only_capped": 1,
           "high_severity": 0, **w},
          {"event": "feed_quality", "feed": "news", "symbol": "DELL", "kept": 6, "dropped_unrelated": 5},
          {"event": "feed_quality", "feed": "news", "symbol": "INTC", "kept": 8, "dropped_unrelated": 2},
          {"event": "feed_quality", "feed": "news", "symbol": "UMC", "kept": 1, "dropped_unrelated": 0}]
    rt = M.role_signals(ev)["red_team"]
    assert rt["contract"]["web_only_capped"] == 1 and rt["contract"]["web_citations_dated"] == 2
    assert rt["web_citation_max_age_days"] == 376
    assert M.feed_signals(ev) == {"news": {"symbols": 3, "kept": 15, "dropped_unrelated": 7,
                                           "most_dropped": [["DELL", 5], ["INTC", 2]]}}


def test_role_signals_separate_prompt_growth_from_search_results():
    """10-02: the red team's input tokens "doubled" and the monitor could not say why, though each call
    recorded its prompt size and search count. Flat prompt, tokens by search count: search results (F-19)."""
    def call(tokens, searches):
        return {"event": "llm_call", "role": "red_team", "outcome": "ok", "input_tokens": tokens,
                "web_searches": searches, "input_chars": 26500, "prompt_symbols": 12, "cost_usd": 0.5}
    s = M.role_signals([call(17000, 0), call(43538, 1), call(108758, 2), call(59477, 2)])["red_team"]
    assert s["input_chars"]["median"] == 26500 and s["input_chars"]["mad"] == 0 and s["prompt_symbols"]["median"] == 12
    assert {k: v["n"] for k, v in s["input_tokens_by_web_searches"].items()} == {"0": 1, "1": 1, "2": 2}
    assert len(s["calls_detail"]) == 4 and s["calls_detail"][1]["web_searches"] == 1


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


def test_the_previous_days_desk_run_is_not_reported_as_unscheduled(mon, monkeypatch):
    """Desk at 13:40, monitor at 14:00: the previous day's 13:40 run sits 20 minutes before the window, inside
    the 10-minute early tolerance, and was listed as "unscheduled" in every report (F-28, 2026-10-06)."""
    job, sent, tmp = mon
    now = datetime.now().astimezone().replace(microsecond=0)
    slot_today = now - timedelta(minutes=20)
    monkeypatch.setattr(job, "installed_schedules",
                        lambda: {"desk": [(d, slot_today.hour, slot_today.minute) for d in range(7)],
                                 "report": None, "audit": None})
    (tmp / "monitor").mkdir(parents=True, exist_ok=True)
    (tmp / "monitor" / "state.json").write_text(json.dumps({"last_run": (now - timedelta(days=1)).isoformat()}))
    T.DIR.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate([now - timedelta(days=2), slot_today - timedelta(days=1) + timedelta(seconds=7),
                           slot_today + timedelta(seconds=7)]):
        u = t.astimezone(timezone.utc)
        with open(T.DIR / f"{u.date().isoformat()}.jsonl", "a") as f:
            f.write(json.dumps({"ts": u.isoformat(timespec="seconds"), "run_id": f"r{i}", "job": "desk",
                                "event": "run_start", "trigger": "launchd"}) + "\n")
    seen = {}
    monkeypatch.setattr(job, "analyze", lambda system, payload, model: seen.update(payload) or (None, "test"))
    _run_monitor(job, [])
    desk = seen["runs"]["desk"]
    assert desk["unscheduled"] == [] and desk["missed"] == [] and len(desk["matched"]) == 1


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


def test_only_a_sev1_alarm_is_emailed_as_one(mon, monkeypatch):
    """2026-10-01: a SEV2 manual-coverage alarm went out under a "[SEV1]" subject."""
    job, sent, tmp = mon
    sev2 = {"key": "report-manual-cover-2026-09-30", "severity": "SEV2", "title": "slot covered by hand", "detail": "d"}
    monkeypatch.setattr(job, "hard_alarms", lambda runs, jobs, tdays, events=None: [sev2])
    _run_monitor(job, ["--no-llm"])
    assert sent == []
    assert "slot covered by hand" in (tmp / "monitor" / "reports" / f"{date.today().isoformat()}.md").read_text()
    sev1 = dict(sev2, key="desk-missed-2026-10-01", severity="SEV1", title="Desk job did not run")
    monkeypatch.setattr(job, "hard_alarms", lambda runs, jobs, tdays, events=None: [sev2, sev1])
    _run_monitor(job, ["--no-llm"])
    assert len(sent) == 1 and "Desk job did not run" in sent[0][0] and "slot covered by hand" not in sent[0][1]


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


def test_runs_record_who_started_them_and_help_is_not_a_run(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_daily.py", "--dry-run"])
    monkeypatch.setenv("XPC_SERVICE_NAME", "application.com.apple.Terminal.123")
    T.run_job("desk", lambda: T.note(dry_run=True))
    monkeypatch.setenv("XPC_SERVICE_NAME", "com.alpaca-desk.daily")          # what launchd sets for the job
    monkeypatch.setattr(sys, "argv", ["run_daily.py"])
    T.run_job("desk", lambda: None)
    monkeypatch.setattr(sys, "argv", ["run_daily.py", "--help"])
    T.run_job("desk", lambda: None)                                          # records nothing
    starts = [e for e in _events() if e["event"] == "run_start"]
    assert [(s["trigger"], s["args"]) for s in starts] == [("manual", "--dry-run"), ("launchd", "")]


def test_only_scheduled_real_runs_set_a_job_s_latency():
    now = datetime(2026, 9, 25, 21, 0, tzinfo=timezone.utc)

    def run(rid, trigger, dur, **end):
        ts = (now - timedelta(minutes=30)).isoformat()
        return [{"event": "run_start", "run_id": rid, "job": "desk", "ts": ts, "trigger": trigger},
                {"event": "run_end", "run_id": rid, "job": "desk", "ts": ts, "status": "ok", "duration_s": dur, **end}]
    ev = run("a", "launchd", 290.0) + run("b", "manual", 1.2, dry_run=True) + run("c", "launchd", 1.0, dry_run=True) \
        + run("d", "manual", 300.0)
    j = M.job_signals(ev, now)["desk"]
    assert j["kinds"] == {"scheduled": 1, "manual_or_test": 3}
    assert j["duration_s_scheduled"]["n"] == 1 and j["duration_s_scheduled"]["median"] == 290.0
    # Older events carry no trigger: judged by their dry_run/print_only notes alone, as before.
    assert T.is_production({"job": "desk"}, {"status": "ok"}) and not T.is_production({}, {"print_only": True})


def test_a_late_scheduled_start_and_a_missed_slot_are_both_on_record(monkeypatch):
    from desk import schedule
    PTZ = timezone(timedelta(hours=-7))
    monkeypatch.setenv("XPC_SERVICE_NAME", "com.alpaca-desk.daily")
    monkeypatch.setattr(schedule, "installed_slots", lambda job: [(d, 13, 45) for d in range(5)] if job == "desk" else [])
    monkeypatch.setattr(schedule, "slot_lateness", lambda label, now, slots=None:
                        {"slot": "2026-09-29T13:45:00-07:00", "late_min": 47.0})
    T.start_run("desk")                                       # launchd fired it on wake, 47 minutes late
    T.end_run("ok")
    ev = _events()
    assert ev[0]["late_min"] == 47.0 and ev[1]["event"] == "slot_deferred"
    # The next job to start finds Monday's desk slot with no scheduled run and records it once.
    monkeypatch.setattr(T, "first_event_time", lambda directory=None: datetime(2026, 9, 25, 15, 0, tzinfo=PTZ))
    now = datetime(2026, 9, 29, 13, 44, tzinfo=PTZ)
    monkeypatch.setattr(T, "load_events", lambda since, until=None, directory=None: [])
    assert ("desk", datetime(2026, 9, 28, 13, 45, tzinfo=PTZ)) in T.record_missed_slots(now)


def test_a_schedule_change_does_not_rewrite_slots_already_judged(monkeypatch):
    """10-02: after the desk slot moved 13:45 -> 13:40, the first run re-checked the past week against 13:40 and
    recorded 09-28, 09-29 and 09-30 as missed a second time (F-22)."""
    from desk import schedule
    PTZ = timezone(timedelta(hours=-7))
    monkeypatch.setattr(T, "first_event_time", lambda directory=None: datetime(2026, 9, 24, 15, 0, tzinfo=PTZ))
    monkeypatch.setattr(schedule, "installed_slots", lambda job: [(d, 13, 40) for d in range(5)] if job == "desk" else [])
    prev = {"event": "run_start", "run_id": "prev", "job": "report", "trigger": "launchd",
            "ts": datetime(2026, 10, 1, 14, 0, tzinfo=PTZ).isoformat()}
    desk_1001 = {"event": "run_start", "run_id": "d", "job": "desk", "trigger": "launchd",    # deferred to a wake
                 "ts": datetime(2026, 10, 1, 13, 57, 42, tzinfo=PTZ).isoformat()}
    monkeypatch.setattr(T, "load_events", lambda since, until=None, directory=None: [desk_1001, prev])
    assert T.record_missed_slots(datetime(2026, 10, 2, 13, 40, 7, tzinfo=PTZ)) == []
    # Days with no runs at all: the previous start is old, so every slot since then is still checked.
    prev["ts"] = desk_1001["ts"] = datetime(2026, 9, 28, 9, 0, tzinfo=PTZ).isoformat()
    got = T.record_missed_slots(datetime(2026, 10, 2, 13, 40, 7, tzinfo=PTZ))
    assert [t.day for _, t in got] == [28, 29, 30, 1]


def test_a_scheduled_run_is_recognised_by_the_job_name_its_plist_sets(monkeypatch):
    # Under the named scripts Python saw XPC_SERVICE_NAME=0, and every 2026-09-30 run read as manual (F-17).
    monkeypatch.delenv("TEAMROBO_TRIGGER", raising=False)
    monkeypatch.setenv("XPC_SERVICE_NAME", "0")
    monkeypatch.delenv("TEAMROBO_JOB", raising=False)
    assert T.trigger() == "manual" and T.launchd_job() is None
    monkeypatch.setenv("TEAMROBO_JOB", "com.alpaca-desk.daily")
    assert T.trigger() == "launchd" and T.launchd_job() == "com.alpaca-desk.daily"


def test_a_run_records_the_processes_that_started_it(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_daily.py", "--dry-run"])
    T.run_job("desk", lambda: None)
    chain = next(e for e in _events() if e["event"] == "run_start")["launched_via"]
    assert chain.split("<")[0].lower().startswith("python")      # this process first, then its parents
    assert len(chain.split("<")) >= 2


def test_a_chained_email_run_counts_as_scheduled_work(monkeypatch):
    monkeypatch.setenv("TEAMROBO_TRIGGER", "chained")
    assert T.trigger() == "chained"
    assert T.is_production({"trigger": "chained"}, {"status": "ok"})


def test_events_written_after_the_utc_date_rolls_over_are_still_found():
    """Files are named by UTC date. At 17:26 Pacific on 09-29 it was already 09-30 UTC, and a window from local
    midnight picked only the 09-29 file: the email's wait-for-desk check could not see an evening desk run."""
    PTZ = timezone(timedelta(hours=-7))
    T.DIR.mkdir(parents=True, exist_ok=True)
    ev = {"event": "run_start", "job": "desk", "run_id": "x", "ts": "2026-09-30T00:26:00+00:00"}
    (T.DIR / "2026-09-30.jsonl").write_text(json.dumps(ev) + "\n")
    got = T.load_events(datetime(2026, 9, 29, 0, 0, tzinfo=PTZ), datetime(2026, 9, 29, 17, 30, tzinfo=PTZ))
    assert [e["run_id"] for e in got] == ["x"]


def test_a_slot_covered_by_hand_is_an_alarm_and_a_chained_one_is_not():
    """09-28 and 09-29: the desk slot 'matched' a run started by hand, so nothing was raised (monitor R-17)."""
    from datetime import date as _date
    runs = {"desk": {"matched": [{"slot": "2026-09-29T13:45:00-07:00", "late_min": 42.7, "covered_by": "manual"}]},
            "report": {"matched": [{"slot": "2026-09-29T14:00:00-07:00", "late_min": 39.6, "covered_by": "chained"},
                                   {"slot": "2026-09-25T14:00:00-07:00", "late_min": 4.0, "covered_by": None}]}}
    alarms = M.coverage_alarms(runs, {_date(2026, 9, 29), _date(2026, 9, 25)})
    assert [a["key"] for a in alarms] == ["desk-manual-cover-2026-09-29"] and alarms[0]["severity"] == "SEV2"
    assert M.coverage_alarms(runs, set()) == []                        # not a trading day: nothing to alarm on


def test_each_llm_call_carries_its_prompt_make_up_and_each_run_its_code_version():
    from desk.analyst import prompt_composition
    payload = {"rules": {"a": 1}, "holdings": [{"symbol": "AMD", "facts": {"x": "y" * 100}}],
               "candidates": [{"symbol": "MXL", "facts": None}]}
    c = prompt_composition(payload)
    assert set(c["prompt_chars"]) == {"rules", "holdings", "candidates"} and c["prompt_symbols"] == 2
    assert c["prompt_fact_chars"] > 100
    assert T.code_version() is None or len(T.code_version().split("+")[0]) >= 7


def test_run_start_records_the_branch_the_job_ran_from(monkeypatch):
    monkeypatch.setitem(T._CODE, "checkout", {"branch": "r38-counters", "behind_main": 5})
    T.start_run("desk")
    T.end_run("ok")
    assert _events()[0]["branch"] == "r38-counters" and _events()[0]["behind_main"] == 5

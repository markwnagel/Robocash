"""Deterministic half of the agent reliability monitor: expected runs, golden signals per agent, baselines,
error signatures, hard alarms, and a fallback report. No network and no LLM here - monitor.py gathers the
inputs, calls Claude with agent-monitor-prompt.md, and stores the result.

The arithmetic lives in code on purpose: the model interprets numbers, it does not compute them.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

import yaml

from .telemetry import is_production, signature

from .schedule import expected_runs, match_runs, schedule_slots  # noqa: F401  (shared with the jobs)


def _ts(e: dict) -> datetime:
    return datetime.fromisoformat(e["ts"])


def pct(xs: list, q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def robust(xs: list) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    med = statistics.median(xs)
    return {"n": len(xs), "median": round(med, 4), "p95": round(pct(xs, 0.95), 4),
            "mad": round(statistics.median([abs(x - med) for x in xs]), 4)}


def delta(window: dict, base: dict) -> dict:
    """Relative change and robust z of the window median against the baseline, per the prompt's thresholds."""
    if not window.get("n") or not base.get("n") or base.get("median") in (None, 0):
        return {}
    rel = (window["median"] - base["median"]) / abs(base["median"])
    sd = 1.4826 * base.get("mad", 0)
    return {"rel_change": round(rel, 3), "robust_z": round((window["median"] - base["median"]) / sd, 2) if sd else None}


def role_signals(events: list) -> dict:
    """Golden signals per LLM role: rate, errors, duration, tokens and cost, plus output-contract checks."""
    calls, outputs = defaultdict(list), defaultdict(list)
    for e in events:
        if e.get("event") == "llm_call":
            calls[e["role"]].append(e)
        elif e.get("event") == "agent_output":
            outputs[e["role"]].append(e)
    out = {}
    for role in sorted(set(calls) | set(outputs)):
        c, o = calls.get(role, []), outputs.get(role, [])
        outcomes = Counter(x.get("outcome") for x in c)
        costs = [x.get("cost_usd") for x in c]
        req = sum(x.get("requested") or 0 for x in o)
        ret = sum(x.get("returned") or 0 for x in o if x.get("returned") is not None)
        out[role] = {
            "calls": len(c), "outcomes": dict(outcomes),
            "error_rate": round(1 - outcomes.get("ok", 0) / len(c), 3) if c else None,
            "fallback_served": sum(1 for x in c if x.get("fallback")),
            "stop_reasons": dict(Counter(x.get("stop_reason") for x in c)),
            "latency_s": robust([x.get("latency_s") for x in c]),
            "input_tokens": robust([x.get("input_tokens") for x in c]),
            "output_tokens": robust([x.get("output_tokens") for x in c]),
            "cost_usd": robust(costs), "cost_total_usd": round(sum(x for x in costs if x), 4),
            "cost_unknown_calls": sum(1 for x in costs if x is None),
            "web_searches": sum(x.get("web_searches") or 0 for x in c),
            "web_search_usd_total": round(sum(x.get("web_search_usd") or 0 for x in c), 4),
            "max_turns": max((x.get("turns") or 0 for x in c), default=0),
            "outputs": len(o), "no_answer": sum(1 for x in o if x.get("returned") is None),
            "coverage": round(ret / req, 3) if req else None,
            "contract": {k: sum(x.get(k) or 0 for x in o)
                         for k in ("brief_over_limit", "severity_out_of_range", "fell_back", "trimmed_for_length")
                         if any(k in x for x in o)},
            "unrequested_examples": sorted({s for x in o for s in (x.get("unrequested") or [])})[:10],
            "missing_examples": sorted({s for x in o for s in (x.get("missing") or [])})[:10],
            "errors": [x.get("error") for x in c if x.get("error")][:3],
        }
    return out


def job_signals(events: list, now: datetime, orphan_after: timedelta = timedelta(hours=2)) -> dict:
    starts = {e["run_id"]: e for e in events if e.get("event") == "run_start"}
    ends = {e["run_id"]: e for e in events if e.get("event") == "run_end"}
    jobs = defaultdict(lambda: {"started": 0, "ended": 0, "status": Counter(), "durations": [], "orphaned": [],
                                "in_progress": [], "errors": [], "fields": [], "kinds": Counter()})
    for rid, s in starts.items():
        j = jobs[s["job"]]
        j["started"] += 1
        end = ends.get(rid)
        prod = is_production(s, end)
        j["kinds"]["scheduled" if prod else "manual_or_test"] += 1
        if end is None:
            (j["orphaned"] if now - _ts(s) > orphan_after else j["in_progress"]).append(
                {"run_id": rid, "started": s["ts"]})
            continue
        j["ended"] += 1
        j["status"][end.get("status")] += 1
        if prod:                  # a dry run's 1-second duration is not the job's latency
            j["durations"].append(end.get("duration_s"))
        if end.get("error"):
            j["errors"].append({"run_id": rid, "ts": end["ts"], "error": end["error"]})
        extra = {k: v for k, v in end.items()
                 if k not in ("ts", "run_id", "job", "event", "status", "duration_s", "error")}
        j["fields"].append({"run_id": rid, "ts": end["ts"], "status": end.get("status"),
                            "trigger": s.get("trigger"), "scheduled": prod, **extra})
    return {k: {**{x: v[x] for x in ("started", "ended", "orphaned", "in_progress", "errors")},
                "status": dict(v["status"]), "kinds": dict(v["kinds"]),
                "duration_s_scheduled": robust(v["durations"]), "runs": v["fields"][-5:]}
            for k, v in jobs.items()}


def log_signatures(window_events: list, baseline_events: list, limit: int = 40) -> list:
    """WARNING+ log lines grouped by normalized signature; `new` = never seen in the baseline period."""
    seen = {(e.get("job"), e.get("signature")) for e in baseline_events if e.get("event") == "log"}
    groups = defaultdict(list)
    for e in window_events:
        if e.get("event") == "log":
            groups[(e.get("job"), e.get("signature"))].append(e)
    rows = [{"job": job, "signature": sig, "count": len(es), "level": es[0].get("level"),
             "new": (job, sig) not in seen, "example": es[0].get("message", "")[:300], "first_ts": es[0]["ts"],
             "run_ids": sorted({x.get("run_id") for x in es if x.get("run_id")})[:5]}
            for (job, sig), es in groups.items()]
    rows.sort(key=lambda r: (not r["new"], r["level"] != "ERROR", -r["count"]))
    return rows[:limit]


def brief_signatures(briefs: dict, baseline_briefs: dict, limit: int = 40) -> list:
    """The desk brief's Notes carry its soft failures ('protection UNVERIFIED', 'analyst unavailable', ...)
    that never reach a log. Grouped the same way as log lines."""
    def notes(text: str) -> list:
        out, on = [], False
        for line in text.splitlines():
            if line.startswith("Notes:"):
                on = True
                continue
            if on and line.startswith("  - "):
                out.append(line[4:])
            elif on and line and not line.startswith(" "):
                break
        return out
    seen = {signature(n) for t in baseline_briefs.values() for n in notes(t)}
    groups = defaultdict(list)
    for day, text in sorted(briefs.items()):
        for n in notes(text):
            groups[signature(n)].append((day, n))
    rows = [{"signature": sig, "count": len(v), "days": sorted({d for d, _ in v}), "new": sig not in seen,
             "example": v[0][1][:300]} for sig, v in groups.items()]
    rows.sort(key=lambda r: (not r["new"], -r["count"]))
    return rows[:limit]


def baseline_view(baseline_events: list, incident_days: set) -> dict:
    kept = [e for e in baseline_events if _ts(e).astimezone().date().isoformat() not in incident_days]
    days = {_ts(e).astimezone().date() for e in kept}
    runs = sum(1 for e in kept if e.get("event") == "run_start")
    return {"events": kept, "windows": len(days), "runs": runs,
            "excluded_incident_days": sorted(incident_days)}


def compare_roles(window: dict, base: dict) -> dict:
    out = {}
    for role, w in window.items():
        b = base.get(role, {})
        out[role] = {m: delta(w.get(m, {}), b.get(m, {})) for m in ("latency_s", "cost_usd", "input_tokens", "output_tokens")}
        if b.get("error_rate") is not None and w.get("error_rate") is not None:
            out[role]["error_rate"] = {"window": w["error_rate"], "baseline": b["error_rate"]}
        if b.get("coverage") is not None and w.get("coverage") is not None:
            out[role]["coverage"] = {"window": w["coverage"], "baseline": b["coverage"]}
    return out


def hard_alarms(runs: dict, jobs: dict, trading_days: set | None, events: list | None = None) -> list:
    """Conditions that escalate whether or not the model is reachable. The desk is the job that maintains
    exits and protective stops, so a missed, crashed or hung desk run is SEV1 on its own; so is a holding
    the desk reported without a broker-side stop, a kill-switch halt, and a daily email that did not go out.
    `events` are the window's telemetry, for the outcome fields each run_end carries."""
    alarms = []
    desk = runs.get("desk", {})
    for t in desk.get("missed", []):
        if trading_days is None or t.date() in trading_days:
            alarms.append({"key": f"desk-missed-{t.date()}", "severity": "SEV1",
                           "title": f"Desk job did not run for its {t:%a %Y-%m-%d %H:%M} slot",
                           "detail": "No run_start near the scheduled time. Exits, the kill switch and stop "
                                     "re-placement did not run that day. Stops placed on earlier days still stand "
                                     "at the broker, but a position whose entry filled that day has none - its "
                                     "bracket legs expired at the close. Likely cause: the Mac was asleep or off."})
    for r in jobs.get("desk", {}).get("errors", []):
        alarms.append({"key": f"desk-error-{r['run_id']}", "severity": "SEV1",
                       "title": "Desk job crashed", "detail": f"run {r['run_id']} at {r['ts']}: {r['error']}"})
    for r in jobs.get("desk", {}).get("orphaned", []):
        alarms.append({"key": f"desk-orphan-{r['run_id']}", "severity": "SEV1",
                       "title": "Desk job started and never finished",
                       "detail": f"run {r['run_id']} started {r['started']} with no run_end: killed, hung, or the "
                                 f"machine slept mid-run. Orders may be half-placed - check the broker."})
    for t in runs.get("report", {}).get("missed", []):
        if trading_days is None or t.date() in trading_days:
            alarms.append({"key": f"report-missed-{t.date()}", "severity": "SEV1",
                           "title": f"Daily report did not run for its {t:%a %Y-%m-%d %H:%M} slot",
                           "detail": "No email that day: its notes, manual actions and health summary were not sent."})
    for r in jobs.get("report", {}).get("errors", []):
        alarms.append({"key": f"report-error-{r['run_id']}", "severity": "SEV1", "title": "Daily report crashed",
                       "detail": f"run {r['run_id']} at {r['ts']}: {r['error']}. No email went out."})
    ends = [e for e in events or [] if e.get("event") == "run_end"]
    for e in ends:
        if e.get("job") == "desk":
            bare = (int(e.get("unprotected") or 0) + int(e.get("protection_unverified") or 0)
                    + int(e.get("protection_mismatch") or 0))
            if bare:
                alarms.append({"key": f"desk-unprotected-{str(e['ts'])[:10]}", "severity": "SEV1",
                               "title": f"{bare} holding(s) without a verified broker-side stop",
                               "detail": f"desk run {e.get('run_id')} at {e['ts']} reported {e.get('unprotected') or 0} "
                                         f"holding(s) with no stop (or not for every share), "
                                         f"{e.get('protection_unverified') or 0} it could not verify and "
                                         f"{e.get('protection_mismatch') or 0} whose stop is larger than the position. "
                                         f"The brief's MANUAL ACTION notes name them; fix the stops at the broker."})
            if e.get("status") == "halted":
                alarms.append({"key": f"desk-halted-{e.get('halt_date') or str(e['ts'])[:10]}", "severity": "SEV1",
                               "title": "Kill switch halted trading",
                               "detail": f"{e.get('reason')}. Nothing trades until --resume"
                                         f"{' --reset-hwm' if e.get('hard') else ''}; "
                                         f"{e.get('open_positions', 0)} position(s) still open at that run."})
        elif e.get("job") == "report" and e.get("email") == "failed":
            alarms.append({"key": f"report-email-{e.get('run_id')}", "severity": "SEV1",
                           "title": "Daily email failed to send",
                           "detail": f"run {e.get('run_id')} at {e['ts']}: {e.get('email_detail')}. The report is in "
                                     f"state/reports/; the next email will carry the changes this one missed."})
    return alarms


def render_config(mcfg: dict, report_to: str, root: str) -> str:
    """The prompt's Configuration block, filled from config.yaml's monitor section."""
    return yaml.safe_dump({
        "monitored_agents": mcfg.get("agents", []),
        "telemetry_sources": {
            "logs": f"{root}/state/telemetry/*.jsonl (event=log, WARNING and above) + state/launchd*.log",
            "traces": "unavailable - no tracing; run_id joins run_start, run_end, llm_call and log events of one job run",
            "metrics": "derived from state/telemetry (llm_call, agent_output, run_start/run_end)",
            "transcripts": "unavailable - prompts and full model responses are not stored; the desk brief "
                           "(state/briefs) and daily report (state/reports) are the closest record",
            "evals": "unavailable - no eval or golden-task suite exists",
            "change_log": "config.yaml values and code-file hashes diffed against the previous monitor run; "
                          "installed launchd schedules. Not a git repository - no commit history",
        },
        "report_cadence": "weekdays, immediately before the 14:00 local daily report",
        "report_window": "since the previous monitor run (normally 24h; up to 7 days after a gap)",
        "baseline_window": f"trailing {mcfg.get('baseline_days', 7)} days before the window, incident days excluded",
        "report_destination": f"{root}/state/monitor/reports/<date>.md; summary in the daily email",
        "state_store": f"{root}/state/monitor/state.json",
        "escalation": f"SEV1 only: immediate email to {report_to}",
        "permissions": {"can_open_prs": False, "can_create_issues": False},
    }, sort_keys=False, width=110)


def build_system_prompt(template: str, config_yaml: str) -> str:
    """Swap the template's placeholder Configuration block for the real one; the rest is used verbatim."""
    pat = re.compile(r"(## Configuration\s*\n+```yaml\n)(.*?)(\n```)", re.S)
    if not pat.search(template):
        raise ValueError("monitor prompt has no '## Configuration' yaml block to fill")
    return pat.sub(lambda m: m.group(1) + config_yaml.rstrip() + m.group(3), template, count=1)


SEVERITIES = ["SEV1", "SEV2", "SEV3", "SEV4"]

MONITOR_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["healthy", "degraded", "incident"]},
        "summary": {"type": "string"},
        "report_markdown": {"type": "string"},
        "findings": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"}, "signature": {"type": "string"}, "title": {"type": "string"},
                "severity": {"type": "string", "enum": SEVERITIES},
                "trend": {"type": "string", "enum": ["new", "ongoing", "worsening", "improving"]},
                "agents": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["id", "signature", "title", "severity", "trend", "agents"], "additionalProperties": False}},
        "recommendations": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"}, "addresses": {"type": "array", "items": {"type": "string"}},
                "priority": {"type": "string", "enum": ["P1", "P2", "P3"]}, "title": {"type": "string"},
                "status": {"type": "string", "enum": ["open", "applied", "not_applied", "unknown"]},
            },
            "required": ["id", "addresses", "priority", "title", "status"], "additionalProperties": False}},
        "resolved_finding_ids": {"type": "array", "items": {"type": "string"}},
        "escalate": {"type": "boolean"},
        "escalation_summary": {"type": "string"},
    },
    "required": ["status", "summary", "report_markdown", "findings", "recommendations", "resolved_finding_ids",
                 "escalate", "escalation_summary"],
    "additionalProperties": False,
}


def fallback_report(payload: dict, reason: str) -> str:
    """Used when the model call fails: the numbers and hard alarms still get written down."""
    w = payload["run"]
    alarms = payload.get("hard_alarms", [])
    status = "🔴 Incident" if alarms else "⚪ Unknown (analysis unavailable)"
    lines = [f"# Agent Reliability Report: {w['window_start']} to {w['window_end']}", "", "## Summary",
             f"Overall status: {status}",
             f"The monitor's analysis step failed ({reason}); this is the deterministic fallback. It lists the "
             f"measured signals and hard alarms without diagnosis.", "", "## Hard alarms"]
    lines += [f"- **{a['severity']}** {a['title']}: {a['detail']}" for a in alarms] or ["- none"]
    lines += ["", "## Jobs"]
    for job, r in payload["runs"].items():
        j = payload["jobs"].get(job, {})
        lines.append(f"- **{job}**: {len(r.get('matched', []))} of {len(r.get('matched', [])) + len(r.get('missed', []))} "
                     f"scheduled runs seen; statuses {j.get('status', {})}; orphaned {len(j.get('orphaned', []))}")
    lines += ["", "## Agents (LLM roles)", "| Role | Calls | Error rate | p95 latency (s) | Cost | Coverage |",
              "|---|---|---|---|---|---|"]
    for role, s in payload["signals"].items():
        lines.append(f"| {role} | {s['calls']} | {s['error_rate']} | {s['latency_s'].get('p95', 'n/a')} | "
                     f"${s['cost_total_usd']:.2f} | {s['coverage']} |")
    lines += ["", "## Data coverage", *[f"- {k}: {v}" for k, v in payload["coverage"].items()]]
    return "\n".join(lines)


def summary_for_email(result: dict, report_rel: str, analysis: str) -> dict:
    """What the daily email shows: status, findings and recommendations, and where the full report is."""
    return {
        "date": date.today().isoformat(), "status": result.get("status", "unknown"), "analysis": analysis,
        "summary": result.get("summary", ""), "report": report_rel,
        "findings": [{"id": f["id"], "severity": f["severity"], "title": f["title"], "trend": f.get("trend", "")}
                     for f in sorted(result.get("findings", []), key=lambda f: f["severity"])],
        "recommendations": [{"id": r["id"], "priority": r["priority"], "title": r["title"]}
                            for r in result.get("recommendations", []) if r.get("status") == "open"],
    }

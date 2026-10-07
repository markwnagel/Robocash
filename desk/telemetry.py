"""Structured telemetry: one JSON object per line in state/telemetry/YYYY-MM-DD.jsonl.

Events
  run_start / run_end   one pair per job run (desk, report, audit, monitor), joined by run_id
  llm_call              every Claude request: role, model, tokens, cost, latency, outcome
  agent_output          what an LLM role returned, checked against what it was asked for
  log                   WARNING and above from any logger while a run is open
  order                 the desk's order lifecycle: entry_staged, entry_filled, entry_expired, position_adopted,
                        position_closed - so a name that appears or disappears can be traced to an order

monitor.py is the only reader. Nothing that trades depends on these files, and writing one must never
break a run, so every write swallows its own errors.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import state

DIR: Path | None = None          # tests point this at a temp dir; default is state/telemetry

# USD per million tokens, Anthropic first-party list prices (claude-api reference, cached 2026-06-24).
# Cache writes bill at 1.25x input (5-minute TTL), cache reads at 0.1x. cost_usd is tokens only; web searches
# bill separately at $10 per 1,000 and are recorded as web_search_usd beside it.
PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
CACHE_WRITE_X, CACHE_READ_X = 1.25, 0.10
WEB_SEARCH_USD = 0.01

_run = {"id": None, "job": None, "t0": None, "fields": {}, "handler": None}

_SECRETS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]+"), "sk-ant-<redacted>"),
    (re.compile(r"(?i)(token|api_key|apikey|secret|password)=([^&\s]+)"), r"\1=<redacted>"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+"), r"\1<redacted>"),
]


def redact(text: str) -> str:
    s = str(text)
    for pat, rep in _SECRETS:
        s = pat.sub(rep, s)
    return s


def signature(message: str) -> str:
    """Normalized form of a message, so one problem counts once however many symbols or numbers it names."""
    s = redact(message)
    s = re.sub(r"\b[0-9a-f]{8,}\b", "<id>", s)
    s = re.sub(r"\d+(?:\.\d+)?", "#", s)
    s = re.sub(r"\b[A-Z]{1,5}\b", "<SYM>", s)
    return re.sub(r"\s+", " ", s).strip()[:200]


def _dir() -> Path:
    return DIR if DIR is not None else state.STATE_DIR / "telemetry"


def emit(event: dict) -> None:
    try:
        now = datetime.now(timezone.utc)
        rec = {"ts": now.isoformat(timespec="seconds"), "run_id": _run["id"], "job": _run["job"], **event}
        d = _dir()
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{now.date().isoformat()}.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception as e:  # telemetry must never be the reason a run fails - but a gap in it must be visible
        try:
            sys.stderr.write(f"telemetry: event not recorded ({type(e).__name__}: {e})\n")
        except Exception:
            pass


class _LogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("desk.telemetry"):
            return
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        emit({"event": "log", "level": record.levelname, "logger": record.name,
              "message": redact(msg)[:500], "signature": signature(msg)})


def start_run(job: str) -> str:
    _run.update(id=uuid.uuid4().hex[:12], job=job, t0=time.monotonic(), fields={})
    # Configure console logging first: each job's own logging.basicConfig() is a no-op once the root logger
    # has any handler, so adding ours first silently switched off every job's launchd log.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    h = _LogHandler(level=logging.WARNING)
    logging.getLogger().addHandler(h)
    _run["handler"] = h
    trig = trigger()
    slot = {}
    if trig == "launchd":
        try:
            from . import schedule
            slot = schedule.slot_lateness(launchd_job() or "", datetime.now().astimezone())
        except Exception:           # the record is a nicety; it must never stop a run
            slot = {}
    emit({"event": "run_start", "pid": os.getpid(), "trigger": trig, "args": redact(" ".join(sys.argv[1:]))[:200],
          "code": code_version(), "launched_via": process_chain(), **slot})
    if slot.get("late_min", 0) > 10:
        # Fired by launchd but late: the Mac was asleep (launchd runs a missed calendar job on wake) - monitor R-14.
        emit({"event": "slot_deferred", "slot": slot["slot"], "late_min": slot["late_min"]})
    return _run["id"]


_CODE = {}


def code_version() -> str | None:
    """Short git commit, plus "+dirty" when tracked files have uncommitted edits: ties each run to the code it
    ran, so a change in behavior or cost can be dated to a commit (monitor R-19). None outside a git checkout."""
    if "v" not in _CODE:
        root = Path(__file__).resolve().parent.parent
        try:
            git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True,  # noqa: E731
                                            text=True, timeout=5).stdout.strip()
            sha = git("rev-parse", "--short", "HEAD")
            _CODE["v"] = (sha + ("+dirty" if git("status", "--porcelain", "--untracked-files=no") else "")) or None
        except Exception:
            _CODE["v"] = None
    return _CODE["v"]


def record_missed_slots(now: datetime | None = None) -> list:
    """Write a slot_missed event for each weekday slot that passed with no scheduled start (the job was not
    loaded, or the Mac was off). Every job checks at start, so a missed slot is on record by the next run of
    anything - including the Saturday audit - instead of being silence (monitor R-14, 2026-09-28)."""
    try:
        from . import schedule
        now = now or datetime.now().astimezone()
        began = first_event_time()
        if began is None:
            return []
        events = load_events(now - timedelta(days=8), now)
        since = max(began.astimezone(), now - timedelta(days=7), _previous_check(events, now))
        missed = schedule.missed_slots(events, now, since)
    except Exception:
        return []
    for job, slot in missed:
        emit({"event": "slot_missed", "slot_job": job, "slot": slot.isoformat()})
    return missed


def _previous_check(events: list, now: datetime) -> datetime:
    """Where the previous run's missed-slot check left off. Every run checks at start, and a slot counts once it
    is MISSED_AFTER old, so slots older than (previous run start - MISSED_AFTER) were judged then. Re-judging
    them against today's schedule recorded 09-28..09-30 as missed at 13:40 after the slot moved from 13:45
    (monitor F-22, 2026-10-02). After days with no runs at all, the previous start is old and they are all checked."""
    from .schedule import MISSED_AFTER
    starts = [datetime.fromisoformat(e["ts"]).astimezone() for e in events
              if e.get("event") == "run_start" and e.get("run_id") != _run["id"]]
    starts = [t for t in starts if t < now]
    return max(starts) - MISSED_AFTER if starts else datetime.min.replace(tzinfo=timezone.utc)


def launchd_job() -> str | None:
    """The launchd label this run was started under, or None. The plists set TEAMROBO_JOB to their label.
    launchd's own XPC_SERVICE_NAME reaches only the program it starts: the named scripts start Python two
    processes down, and every scheduled run on 2026-09-30 recorded itself as manual (monitor F-17).
    XPC_SERVICE_NAME still counts, for a job whose plist predates TEAMROBO_JOB and starts Python directly."""
    for name in ("TEAMROBO_JOB", "XPC_SERVICE_NAME"):
        value = os.environ.get(name, "")
        if value.startswith("com.alpaca-desk."):
            return value
    return None


def trigger() -> str:
    """'launchd' for a scheduled run, 'manual' for anything a person or script started. Without this, dev and
    print-only runs were mixed into the scheduled runs' durations and baselines (monitor F-14, 2026-09-25)."""
    # "chained": the desk job ran late and sent the day's email itself (launchd/teamrobo-desk sets this).
    if os.environ.get("TEAMROBO_TRIGGER"):
        return os.environ["TEAMROBO_TRIGGER"]
    return "launchd" if launchd_job() else "manual"


def process_chain(pid: int | None = None, depth: int = 6) -> str | None:
    """Program names from this process up to launchd: "Python<sh<launchd" for a scheduled run (the job script
    runs under sh), a login shell and terminal for one started by hand. It records how a run started without
    trusting the run's own label (monitor R-20). None if ps fails."""
    names, pid = [], pid or os.getpid()
    try:
        for _ in range(depth):
            out = subprocess.run(["ps", "-o", "ppid=,comm=", "-p", str(pid)], capture_output=True, text=True,
                                 timeout=5).stdout.strip()
            if not out:
                break
            ppid, comm = out.split(None, 1)
            names.append(Path(comm).name)
            if pid == 1:
                break
            pid = int(ppid)
    except Exception:
        return None
    return "<".join(names) or None


def order_event(kind: str, symbol: str, **fields) -> None:
    """One step of an order's life. The brief says the same in prose, but on 2026-09-29 it listed an expired
    MXL order and a new MXL order side by side, and the monitor read the next day's fill as a position with
    no entry (F-23)."""
    emit({"event": "order", "kind": kind, "symbol": symbol, **fields})


def is_production(start: dict, end: dict | None) -> bool:
    """A scheduled run that did real work: launched by launchd (or from before trigger was recorded), not a
    dry run and not print-only."""
    if start.get("trigger", "launchd") not in ("launchd", "chained"):
        return False
    return not (end and (end.get("dry_run") or end.get("print_only")))


def note(**fields) -> None:
    """Attach outcome fields to the current run's run_end (entries staged, email status, ...)."""
    _run["fields"].update(fields)


def end_run(status: str, error: str | None = None) -> None:
    fields = dict(_run["fields"])
    noted = fields.pop("status", None)
    if status != "error" and noted:        # a crash always reads as a crash, whatever the job noted first
        status = noted
    elif noted:
        fields["noted_status"] = noted
    dur = round(time.monotonic() - _run["t0"], 2) if _run["t0"] is not None else None
    emit({"event": "run_end", "status": status, "duration_s": dur, "error": error, **fields})
    if _run["handler"] is not None:
        logging.getLogger().removeHandler(_run["handler"])
    _run.update(id=None, job=None, t0=None, fields={}, handler=None)


def run_job(job: str, fn) -> None:
    """Run a job's main() between run_start and run_end. A crash is recorded, then re-raised unchanged.
    --help is not a run: it was being logged as one and counted by the monitor."""
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        fn()
        return
    start_run(job)
    record_missed_slots()
    try:
        fn()
    except SystemExit as e:
        end_run("ok" if e.code in (None, 0) else "error", None if e.code in (None, 0) else f"exit {e.code}")
        raise
    except BaseException as e:
        tb = traceback.extract_tb(e.__traceback__)
        where = f" at {Path(tb[-1].filename).name}:{tb[-1].lineno}" if tb else ""
        end_run("error", redact(f"{type(e).__name__}: {e}")[:500] + where)
        raise
    else:
        end_run("ok")


def usage_of(responses: list) -> dict:
    """Token counts summed over every response of one logical call (pause_turn continuations included)."""
    tot = {"input_tokens": 0, "output_tokens": 0, "cache_write_tokens": 0, "cache_read_tokens": 0, "web_searches": 0,
           "web_fetches": 0}
    for r in responses:
        u = getattr(r, "usage", None)
        if u is None:
            continue
        tot["input_tokens"] += int(getattr(u, "input_tokens", 0) or 0)
        tot["output_tokens"] += int(getattr(u, "output_tokens", 0) or 0)
        tot["cache_write_tokens"] += int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        tot["cache_read_tokens"] += int(getattr(u, "cache_read_input_tokens", 0) or 0)
        stu = getattr(u, "server_tool_use", None)
        tot["web_searches"] += int(getattr(stu, "web_search_requests", 0) or 0) if stu else 0
        tot["web_fetches"] += int(getattr(stu, "web_fetch_requests", 0) or 0) if stu else 0
    return tot


def server_tool_activity(responses: list) -> dict:
    """What server-side tools put into a call's context, from the response content: block counts by type
    (server tool calls by name) and the size of their results. On 2026-10-06 the analyst's input doubled to
    54k tokens with an unchanged prompt and web_search_requests 0, and nothing recorded could say why."""
    blocks, result_chars = {}, 0
    for r in responses:
        for b in getattr(r, "content", None) or []:
            t = getattr(b, "type", None)
            if t in (None, "text", "thinking", "redacted_thinking"):
                continue
            key = f"{t}:{getattr(b, 'name', '')}" if t == "server_tool_use" else t
            blocks[key] = blocks.get(key, 0) + 1
            if t.endswith("_tool_result"):
                try:
                    result_chars += len(json.dumps(b.model_dump(), default=str))
                except Exception:
                    pass
    return {"server_blocks": blocks, "server_result_chars": result_chars}


def cost_usd(model: str | None, u: dict) -> float | None:
    """None for a model without a listed price - an unknown cost is reported as unknown, not as zero."""
    p = PRICES.get(model or "")
    if not p:
        return None
    inp, out = p
    return round((u["input_tokens"] * inp + u["cache_write_tokens"] * inp * CACHE_WRITE_X
                  + u["cache_read_tokens"] * inp * CACHE_READ_X + u["output_tokens"] * out) / 1e6, 6)


def llm_call(role: str, model: str, started: float, responses: list | None = None, outcome: str = "ok",
             error: str | None = None, **fields) -> None:
    """outcome: ok | refusal | parse_error | paused_out | error. `started` is time.monotonic() at the call."""
    responses = [r for r in (responses or []) if r is not None]
    u = usage_of(responses)
    last = responses[-1] if responses else None
    served = getattr(last, "model", None) or model
    emit({"event": "llm_call", "role": role, "model": model, "served_by": served,
          "fallback": bool(last is not None and served != model), "outcome": outcome,
          "stop_reason": getattr(last, "stop_reason", None), "turns": len(responses),
          "latency_s": round(time.monotonic() - started, 2), **u, **server_tool_activity(responses),
          "cost_usd": cost_usd(served, u),
          "web_search_usd": round(u["web_searches"] * WEB_SEARCH_USD, 4),
          "error": redact(error)[:500] if error else None, **fields})


def agent_output(role: str, requested: int | None, returned: int | None, **fields) -> None:
    """requested/returned: how many items (symbols, trades) the role was asked about vs answered."""
    emit({"event": "agent_output", "role": role, "requested": requested, "returned": returned, **fields})


def first_event_time(directory: Path | None = None) -> datetime | None:
    """When recording began. A run scheduled before this cannot be judged missing - there was no way to see it."""
    d = directory or _dir()
    for p in sorted(d.glob("*.jsonl")) if d.exists() else []:
        for line in p.read_text().splitlines():
            try:
                return datetime.fromisoformat(json.loads(line)["ts"])
            except Exception:
                continue
    return None


def load_events(since: datetime, until: datetime | None = None, directory: Path | None = None) -> list:
    d = directory or _dir()
    until = until or datetime.now(timezone.utc)
    out = []
    if not d.exists():
        return out
    for p in sorted(d.glob("*.jsonl")):
        try:
            day = datetime.fromisoformat(p.stem).date()
        except ValueError:
            continue
        # Files are named by the UTC date of their events; since/until may be local times. Compare in UTC, or
        # every event after 17:00 Pacific (next day's file) was invisible to a same-day local window.
        if day < since.astimezone(timezone.utc).date() or day > until.astimezone(timezone.utc).date():
            continue
        for line in p.read_text().splitlines():
            try:
                e = json.loads(line)
                ts = datetime.fromisoformat(e["ts"])
            except Exception:
                continue
            if since <= ts < until:
                out.append(e)
    return out

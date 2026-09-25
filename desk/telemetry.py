"""Structured telemetry: one JSON object per line in state/telemetry/YYYY-MM-DD.jsonl.

Events
  run_start / run_end   one pair per job run (desk, report, audit, monitor), joined by run_id
  llm_call              every Claude request: role, model, tokens, cost, latency, outcome
  agent_output          what an LLM role returned, checked against what it was asked for
  log                   WARNING and above from any logger while a run is open

monitor.py is the only reader. Nothing that trades depends on these files, and writing one must never
break a run, so every write swallows its own errors.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
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
    emit({"event": "run_start", "pid": os.getpid()})
    return _run["id"]


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
    """Run a job's main() between run_start and run_end. A crash is recorded, then re-raised unchanged."""
    start_run(job)
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
    tot = {"input_tokens": 0, "output_tokens": 0, "cache_write_tokens": 0, "cache_read_tokens": 0, "web_searches": 0}
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
    return tot


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
          "latency_s": round(time.monotonic() - started, 2), **u, "cost_usd": cost_usd(served, u),
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
        if day < since.date() or day > until.date():
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

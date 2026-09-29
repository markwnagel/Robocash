"""The launchd schedule as the jobs themselves can see it: which jobs are loaded, the wake that lets them fire,
their slots, how late a run started, and which slots passed with no scheduled run at all.

On 2026-09-28 neither weekday job was loaded (macOS had them switched off after a restart) and nothing recorded
it: no run, no error, no event. This module turns that silence into something the monitor, the brief and the
email can report. No telemetry import here - telemetry calls in at run start - so there is no import cycle.
"""
from __future__ import annotations

import plistlib
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

JOBS = {"desk": "com.alpaca-desk.daily", "report": "com.alpaca-desk.report", "audit": "com.alpaca-desk.weekly-audit"}
AGENTS = Path.home() / "Library" / "LaunchAgents"
WEEKDAY_JOBS = ("desk", "report")
MISSED_AFTER = timedelta(hours=3)      # a slot with no scheduled start this long after it was missed, not late
DEFERRED_AFTER_MIN = 10                # a scheduled start later than this is recorded as deferred

# launchd Weekday: 0 or 7 = Sunday, 1 = Monday ... 6 = Saturday.
def _py_weekday(launchd_wd: int) -> int:
    return (int(launchd_wd) - 1) % 7


def schedule_slots(plist: dict) -> list:
    """[(python_weekday, hour, minute)] from a launchd plist's StartCalendarInterval."""
    sci = plist.get("StartCalendarInterval") or []
    if isinstance(sci, dict):
        sci = [sci]
    out = []
    for s in sci:
        if "Hour" not in s:
            continue
        days = [_py_weekday(s["Weekday"])] if "Weekday" in s else list(range(7))
        out += [(d, int(s["Hour"]), int(s.get("Minute", 0))) for d in days]
    return sorted(set(out))


def expected_runs(slots: list, start: datetime, end: datetime, grace: timedelta) -> list:
    """Scheduled times inside [start, end) that are old enough (by `grace`) to have produced a run."""
    out, day = [], start.date()
    while day <= end.date():
        for wd, h, m in slots:
            if day.weekday() == wd:
                # launchd fires on local wall-clock time. Resolving each slot through the system's rules for
                # that date keeps daylight-saving changes from shifting slots by an hour.
                t = datetime(day.year, day.month, day.day, h, m).astimezone()
                if start <= t and t + grace <= end:
                    out.append(t)
        day += timedelta(days=1)
    return sorted(out)


def match_runs(expected: list, starts: list, early: timedelta = timedelta(minutes=10),
               late: timedelta = timedelta(hours=3)) -> dict:
    """Pair each scheduled slot with the first run that started near it. Unpaired slots are missed runs."""
    starts = sorted(starts)
    used, matched, missed = set(), [], []
    for t in expected:
        hit = next((s for s in starts if s not in used and t - early <= s <= t + late), None)
        if hit is None:
            missed.append(t)
        else:
            used.add(hit)
            matched.append({"slot": t, "started": hit, "late_min": round((hit - t).total_seconds() / 60, 1)})
    return {"matched": matched, "missed": missed, "unscheduled": [s for s in starts if s not in used]}


def installed_slots(job: str) -> list:
    try:
        return schedule_slots(plistlib.loads((AGENTS / f"{JOBS[job]}.plist").read_bytes()))
    except Exception:
        return []


def parse_launchctl_list(out: str) -> dict:
    labels = {ln.split("\t")[-1].strip() for ln in out.splitlines()}
    return {job: label in labels for job, label in JOBS.items()}


def parse_pmset_sched(out: str) -> str | None:
    m = re.search(r"Repeating power events:\s*\n((?:[ \t]+.*\n?)+)", out)
    lines = [ln.strip() for ln in (m.group(1).splitlines() if m else []) if "wake" in ln]
    return lines[0] if lines else None


def loaded_jobs() -> dict | None:
    """{job: loaded?} from launchctl, or None where launchctl is unavailable (not macOS, or it failed)."""
    try:
        return parse_launchctl_list(subprocess.run(["launchctl", "list"], capture_output=True, text=True,
                                                   timeout=10).stdout)
    except Exception:
        return None


def repeating_wake() -> str | None:
    """The repeating wake from `pmset -g sched` ("wakepoweron at 1:40PM weekdays only"), or None if there is
    none. Without it the Mac is asleep at 13:45 and launchd only runs the job when something else wakes it."""
    try:
        return parse_pmset_sched(subprocess.run(["pmset", "-g", "sched"], capture_output=True, text=True,
                                                timeout=10).stdout)
    except Exception:
        return None


def schedule_problems(loaded: dict | None, wake: str | None) -> list:
    """Owner-facing MANUAL ACTION text for a schedule that cannot run. Empty when unknown (loaded is None)."""
    if loaded is None:
        return []
    out = []
    for job, ok in loaded.items():
        if not ok:
            label = JOBS[job]
            out.append(f"MANUAL ACTION: the scheduled {job} job ({label}) is not loaded in launchd, so it will not run. "
                       f"Check it is on under System Settings > General > Login Items & Extensions > Allow in the "
                       f"Background, then run: launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/{label}.plist")
    if wake is None:
        out.append("MANUAL ACTION: no repeating wake is scheduled, so the Mac may be asleep at 13:45 and the desk "
                   "job will not start until something wakes it. Restore it with: sudo pmset repeat wakepoweron "
                   "MTWRF 13:40:00")
    return out


def slot_lateness(label: str, now: datetime, slots: list | None = None) -> dict:
    """{slot, late_min} for the most recent slot of the launchd job `label` at or before now (last 24 hours)."""
    job = next((j for j, lb in JOBS.items() if lb == label), None)
    if job is None:
        return {}
    slots = installed_slots(job) if slots is None else slots
    past = expected_runs(slots, now - timedelta(days=1), now + timedelta(seconds=1), timedelta(0))
    if not past:
        return {}
    return {"slot": past[-1].isoformat(), "late_min": round((now - past[-1]).total_seconds() / 60, 1)}


def missed_slots(events: list, now: datetime, since: datetime, slots_by_job: dict | None = None) -> list:
    """[(job, slot)] for weekday slots that passed with no scheduled start within MISSED_AFTER and are not
    already recorded as slot_missed. A manual run afterwards does not un-miss the slot: the point is that the
    schedule did not fire. Runs from before `trigger` was recorded count as scheduled."""
    slots_by_job = slots_by_job if slots_by_job is not None else {j: installed_slots(j) for j in WEEKDAY_JOBS}
    recorded = {(e.get("slot_job"), e.get("slot")) for e in events if e.get("event") == "slot_missed"}
    out = []
    for job, slots in slots_by_job.items():
        if not slots:
            continue
        starts = [datetime.fromisoformat(e["ts"]).astimezone() for e in events
                  if e.get("event") == "run_start" and e.get("job") == job and e.get("trigger", "launchd") == "launchd"]
        expected = expected_runs(slots, since, now, MISSED_AFTER)
        for t in match_runs(expected, starts, late=MISSED_AFTER)["missed"]:
            if (job, t.isoformat()) not in recorded:
                out.append((job, t))
    return out

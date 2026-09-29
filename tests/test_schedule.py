"""desk/schedule.py: turning a schedule that silently does not run into something on record (monitor R-13, R-14).
On 2026-09-28 neither weekday job was loaded after a restart, and nothing anywhere said so."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import desk.schedule as S

PT = ZoneInfo("America/Los_Angeles")
SLOTS = {"desk": [(d, 13, 45) for d in range(5)], "report": [(d, 14, 0) for d in range(5)]}


def test_loaded_jobs_and_the_wake_are_read_from_launchctl_and_pmset():
    listing = "PID\tStatus\tLabel\n-\t0\tcom.alpaca-desk.daily\n-\t0\tcom.alpaca-desk.weekly-audit\n"
    assert S.parse_launchctl_list(listing) == {"desk": True, "report": False, "audit": True}
    sched = ("Repeating power events:\n  wakepoweron at 1:40PM weekdays only\nScheduled power events:\n"
             " [0]  wake at 09/28/2026 11:25:13 by 'com.apple.alarm'\n")
    assert S.parse_pmset_sched(sched) == "wakepoweron at 1:40PM weekdays only"
    assert S.parse_pmset_sched("Scheduled power events:\n [0]  wake at 09/28/2026\n") is None


def test_schedule_problems_name_the_job_and_the_fix():
    p = S.schedule_problems({"desk": True, "report": False, "audit": True}, "wakepoweron at 1:40PM weekdays only")
    assert len(p) == 1 and p[0].startswith("MANUAL ACTION: the scheduled report job (com.alpaca-desk.report)")
    assert "Allow in the Background" in p[0] and "launchctl bootstrap" in p[0]
    assert any("no repeating wake" in x for x in S.schedule_problems({"desk": True}, None))
    assert S.schedule_problems(None, None) == []      # launchctl unavailable: say nothing rather than guess


def test_a_late_start_is_measured_against_its_own_slot():
    now = datetime(2026, 9, 29, 14, 32, tzinfo=PT)    # woke late: the 13:45 slot fired 47 minutes after it
    got = S.slot_lateness("com.alpaca-desk.daily", now, SLOTS["desk"])
    assert got == {"slot": datetime(2026, 9, 29, 13, 45, tzinfo=PT).isoformat(), "late_min": 47.0}
    assert S.slot_lateness("com.example.other", now, SLOTS["desk"]) == {}


def test_a_slot_with_only_a_manual_run_after_it_is_missed_once():
    """2026-09-28: the desk ran by hand at 15:31 and the email by hand at 15:42; neither launchd slot fired."""
    now = datetime(2026, 9, 29, 13, 44, tzinfo=PT)
    since = datetime(2026, 9, 25, 15, 0, tzinfo=PT)
    ev = [{"event": "run_start", "job": "desk", "ts": "2026-09-28T15:31:42-07:00", "trigger": "manual"},
          {"event": "run_start", "job": "report", "ts": "2026-09-28T15:42:00-07:00", "trigger": "manual"},
          {"event": "run_start", "job": "desk", "ts": "2026-09-25T13:45:06-07:00"}]   # before trigger existed
    missed = S.missed_slots(ev, now, since, SLOTS)
    assert missed == [("desk", datetime(2026, 9, 28, 13, 45, tzinfo=PT)),
                      ("report", datetime(2026, 9, 28, 14, 0, tzinfo=PT))]
    recorded = ev + [{"event": "slot_missed", "slot_job": j, "slot": t.isoformat()} for j, t in missed]
    assert S.missed_slots(recorded, now, since, SLOTS) == []                  # never recorded twice

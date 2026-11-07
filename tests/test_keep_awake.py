"""2026-11-07: the weekly audit ran with the Mac asleep (10-second maintenance wakes every 90 s); the Alpaca download
was reset and the Claude stream timed out after 17 minutes."""
import subprocess
import sys

import audit
from desk import analyst
from desk import telemetry as T

GOOD = {"summary": "The book lost money in a momentum unwind; stops did their job.", "findings": [{"title": "t"}],
        "proposed_changes": [], "sample_size_warning": "small"}


def test_a_job_holds_a_sleep_assertion_for_its_own_life(monkeypatch):
    seen = {}
    monkeypatch.delenv("TEAMROBO_NO_CAFFEINATE", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **k: seen.setdefault("cmd", cmd) or object())
    assert T.keep_awake() is not None
    assert seen["cmd"][0] == "/usr/bin/caffeinate" and "-i" in seen["cmd"] and seen["cmd"][-2:] == ["-w", str(__import__("os").getpid())]


def test_keep_awake_is_off_in_tests_and_never_raises(monkeypatch):
    assert T.keep_awake() is None                                         # conftest sets TEAMROBO_NO_CAFFEINATE
    monkeypatch.delenv("TEAMROBO_NO_CAFFEINATE", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")

    def boom(*a, **k):
        raise OSError("no")
    monkeypatch.setattr(subprocess, "Popen", boom)
    assert T.keep_awake() is None


def _calls(*replies):
    seq = list(replies)

    def call():
        analyst.LAST_CALL.clear()
        out, err = seq.pop(0)
        if err:
            analyst.LAST_CALL["error"] = err
        return out
    return call, seq


def test_the_auditor_is_asked_again_after_a_call_that_never_completed():
    call, left = _calls((None, "ReadTimeout"), (GOOD, None))
    assert audit.ask_auditor(call) == GOOD and not left


def test_a_refusal_or_bad_reply_is_not_retried_and_two_failures_give_up():
    call, left = _calls((None, None), (GOOD, None))
    assert audit.ask_auditor(call) is None and len(left) == 1
    call, left = _calls((None, "ReadTimeout"), (None, "ReadTimeout"))
    assert audit.ask_auditor(call) is None and not left


def test_an_empty_reply_is_still_asked_again():
    call, left = _calls((dict(GOOD, summary="placeholder"), None), (GOOD, None))
    assert audit.ask_auditor(call) == GOOD and not left

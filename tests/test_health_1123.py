"""Fixes from the 2026-11-17 review, the 2026-11-20 outage and the 2026-11-21 audit."""
from datetime import datetime

import pytest
import requests

from desk import data as D
from desk.analyst import ANALYST_SYSTEM
from desk.config import load_config
from desk.monitor import model_review_due, stale_checkout_alarms
from desk.report import _thesis_bits, fallback_summary
from desk.signals import history_bars_needed, history_bars_requested, momentum_months


def test_bar_requests_retry_a_dropped_connection():
    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise requests.exceptions.ConnectionError("Connection aborted.", TimeoutError(60, "Operation timed out"))
        return "bars"

    assert D.with_retries(flaky, "bars 1-200", waits=(1, 2, 3), sleep=waits.append) == "bars"
    assert len(calls) == 3 and waits == [1, 2]


def test_bar_requests_give_up_after_the_last_retry_and_never_retry_other_errors():
    def down():
        raise requests.exceptions.ConnectionError("down")
    with pytest.raises(requests.exceptions.ConnectionError):
        D.with_retries(down, "x", waits=(1, 1), sleep=lambda s: None)
    n = []

    def bad():
        n.append(1)
        raise ValueError("not a network error")
    with pytest.raises(ValueError):
        D.with_retries(bad, "x", waits=(1, 1), sleep=lambda s: None)
    assert len(n) == 1


def test_monitor_retries_a_dropped_review(monkeypatch):
    import monitor as M
    replies = iter([(None, "APIConnectionError: Connection error."), ({"status": "healthy"}, None)])
    monkeypatch.setattr(M, "_analyze_once", lambda *a: next(replies))
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    assert M.analyze("sys", {}, "m") == ({"status": "healthy"}, None)
    monkeypatch.setattr(M, "_analyze_once", lambda *a: (None, "model refused (x)"))
    assert M.analyze("sys", {}, "m") == (None, "model refused (x)")      # a refusal is final


def test_a_failed_review_is_due_on_the_next_run():
    cfg = {"model_review_weekdays": [4]}
    mon = datetime(2026, 11, 23, 14, 0)
    assert not model_review_due(cfg, mon, [], "2026-11-17T14:00:00")[0]
    due, why = model_review_due(cfg, mon, [], "2026-11-17T14:00:00", failed_on="2026-11-20")
    assert due and "2026-11-20" in why


def test_stale_alarm_names_the_missing_commits():
    ev = [{"event": "run_start", "job": "desk", "branch": "main", "behind_main": 2, "ts": "2026-11-17T21:40:02+00:00",
           "run_id": "r1", "code": "6ca3e6a", "missing_commits": ["99f0732 Fix the 2026-11-16 health review findings"],
           "missing_paths": ["desk/analyst.py", "desk/edgar.py"]}]
    a = stale_checkout_alarms(ev)[0]
    assert "Missing: 99f0732 Fix the 2026-11-16" in a["detail"] and "Touching: desk/analyst.py, desk/edgar.py" in a["detail"]


def test_momentum_label_follows_the_lookback_and_history_covers_it():
    cfg = load_config()
    s = cfg["signals"]
    assert momentum_months({"momentum_lookback_days": 252}) == 12
    assert momentum_months({"momentum_lookback_days": 126}) == 6
    run = open("run_daily.py").read()
    assert "6m mom" not in run and "{mom_m}m mom" in run and 'f"momentum_{mom_m}m"' in run
    # data.history_days must fetch enough bars for the slowest signal (audit 2026-11-21 asked)
    assert history_bars_requested(int(cfg["data"]["history_days"])) >= history_bars_needed(s) + 20


def test_old_and_new_theses_both_parse():
    old, new = _thesis_bits("rank 3, 6m mom 158%, 20d 8%"), _thesis_bits("rank 7, 12m mom 55%, 20d 16%")
    assert (old["mom"], old["mom_months"]) == (158, 6) and (new["mom"], new["mom_months"]) == (55, 12)
    assert "after gaining 55% over 12 months" in fallback_summary("bought", {"thesis": "rank 7, 12m mom 55%, 20d 16%"})


def test_an_immaterial_sliver_is_not_a_data_concern():
    assert "even one so small you\njudge it immaterial" in ANALYST_SYSTEM

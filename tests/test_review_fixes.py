"""Engineering review 2026-10-17: spend and API-access alarms, the weekly model review, the red team's scope, the
honest trial count, and the SPY-relative bar."""
from datetime import date, datetime

import numpy as np
import pandas as pd

from desk import monitor as M


def _call(err, outcome="error", day="2026-10-18", role="analyst", cost=0.0):
    return {"event": "llm_call", "ts": f"{day}T20:00:00+00:00", "role": role, "outcome": outcome, "error": err,
            "cost_usd": cost, "web_search_usd": 0.0}


def test_an_api_spend_limit_is_a_sev1_alarm_and_an_ordinary_error_is_not():
    """2026-10-17: every role failed on the Console spend limit and nothing alarmed."""
    limit = ("BadRequestError: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': "
             "'You have reached your specified API usage limits. You will regain access on 2026-11-01 at 00:00 UTC.'}}")
    a = M.api_access_alarms([_call(limit), _call(limit, role="red_team"), _call("APITimeoutError: timed out")])
    assert len(a) == 1 and a[0]["severity"] == "SEV1" and a[0]["key"] == "api-access-2026-10-18"
    assert "analyst, red_team" in a[0]["detail"] and "buys nothing" in a[0]["detail"]
    assert M.api_access_alarms([_call("AuthenticationError: Error code: 401 - invalid x-api-key")])
    assert not M.api_access_alarms([_call(limit, outcome="ok")])
    assert M.hard_alarms({}, {}, None, [_call(limit)])[0]["key"] == "api-access-2026-10-18"


def test_month_spend_and_its_budget_alarms():
    ev = [_call(None, outcome="ok", cost=10.0), _call(None, outcome="ok", cost=12.5), {"event": "run_end"}]
    assert M.month_spend(ev) == 22.5
    assert M.spend_alarms(22.5, 40, date(2026, 10, 18)) == []
    sev2 = M.spend_alarms(34.0, 40, date(2026, 10, 18))
    assert sev2[0]["severity"] == "SEV2" and "85%" in sev2[0]["title"]
    assert M.spend_alarms(41.0, 40, date(2026, 10, 18))[0]["severity"] == "SEV1"
    assert M.spend_alarms(99.0, None, date(2026, 10, 18)) == []          # no budget set: no alarm


def test_the_model_review_runs_weekly_on_alarms_and_never_lapses():
    cfg = {"model_review_weekdays": [4]}
    fri, mon = datetime(2026, 10, 23, 14), datetime(2026, 10, 19, 14)
    assert M.model_review_due({}, mon, [], None) == (True, "daily")             # unset: the old behaviour
    assert M.model_review_due(cfg, fri, [], "2026-10-22T14:00:00")[0]
    due, why = M.model_review_due(cfg, mon, [], "2026-10-16T14:00:00")
    assert not due and "Fri" in why
    assert M.model_review_due(cfg, mon, [{"key": "x"}], "2026-10-16T14:00:00")[0]
    assert M.model_review_due(cfg, mon, [], "2026-10-09T14:00:00")[0]         # 10 days: overdue
    assert M.model_review_due(cfg, mon, [], None)[0]
    assert M.model_review_due(cfg, mon, [], "2026-10-16T14:00:00", forced=True)[0]


def test_a_rules_only_report_says_so_and_shows_the_spend():
    payload = {"run": {"window_start": "a", "window_end": "b"}, "hard_alarms": [], "runs": {}, "jobs": {},
               "signals": {}, "coverage": {}, "api_spend_month_to_date": {"usd": 12.34, "budget_usd": 40.0}}
    r = M.fallback_report(payload, "rules-only day: the model review runs Fri or when a hard alarm fires", scheduled=True)
    assert "No hard alarms" in r and "No model review today" in r and "$12.34 of a $40 budget" in r
    assert "failed" not in r
    assert "analysis step failed" in M.fallback_report(payload, "timeout")


def test_the_red_team_reviews_candidates_only_when_scoped():
    from run_daily import red_team_view
    p = {"holdings": [{"symbol": "AMD", "facts": {}}], "candidates": [{"symbol": "VEEV"}], "rules": {}}
    assert red_team_view(p, {"analyst": {}}) is p
    v = red_team_view(p, {"analyst": {"red_team_scope": "candidates"}})
    assert v["holdings"] == [] and v["held_symbols"] == ["AMD"] and v["candidates"] == p["candidates"]
    assert p["holdings"]                                              # the analyst's payload is untouched


def test_excess_vs_spy_measures_six_month_windows_against_the_benchmark():
    from robustness import excess_vs_spy
    idx = pd.bdate_range("2020-01-01", periods=600)
    spy = pd.Series(np.linspace(100, 160, 600), index=idx)
    eq = spy * 1.0
    same = excess_vs_spy(eq, spy)
    assert abs(same["cagr_excess"]) < 1e-12 and same["beat_6m"] == 0.0
    better = excess_vs_spy(spy * pd.Series(np.linspace(1.0, 1.3, 600), index=idx), spy)
    assert better["cagr_excess"] > 0 and better["beat_6m"] == 1.0 and better["excess_6m_p10"] > 0
    assert excess_vs_spy(eq.iloc[:50], spy)["beat_6m"] is None


def test_check_script_variants_are_logged_as_trials_once_each(tmp_path):
    from desk import trials as T
    from robustness import log_variant_trials
    eq = pd.Series(np.linspace(10000, 12000, 300), index=pd.bdate_range("2023-01-02", periods=300))
    jobs = [{"name": f"v{k}", "group": "g", "start": "2023-01-02", "end": "2024-02-01",
             "overrides": [(("exits", "signal_decay_rank"), r)], "offset": k} for k, r in enumerate((25, 25, 30))]
    jobs.append({"name": "f", "group": "g", "start": "2023-01-02", "end": "2024-02-01", "mode": "frictions"})
    res = {j["name"]: {"equity": eq} for j in jobs}
    assert log_variant_trials(tmp_path / "t.jsonl", jobs, res, "test") == 2      # 25 twice (two starts), 30; no frictions
    log = T.load(tmp_path / "t.jsonl")
    assert {t["overrides"]["exits.signal_decay_rank"] for t in log} == {25, 30} and len({t["config_id"] for t in log}) == 2


def test_the_config_carries_the_review_settings(cfg):
    from desk.config import load_config
    c = load_config()
    assert c["monitor"]["model_review_weekdays"] == [4] and c["analyst"]["red_team_scope"] == "candidates"
    assert c["analyst"]["red_team_max_web_searches"] <= 2 and c["budget"]["monthly_api_usd"] > 0
    assert c["backtest"]["prior_untracked_trials"] >= 63

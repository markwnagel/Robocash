"""The shadow book (desk/shadow.py): every decision that changed what was bought, replayed against what was."""
import json
from datetime import date

import numpy as np
import pandas as pd
from pytest import approx

from desk import shadow
from tests.test_run_daily import FakeBroker, run, wired  # noqa: F401  (fixture)

ECFG = {"stop_atr_multiple": 2.0, "take_profit_r_multiple": 3.0}


def test_a_row_carries_the_bracket_a_real_entry_would_have():
    r = shadow.row(date(2026, 10, 19), "blocked", "SNOW", 200.0, 5.0, ECFG, reason="dilutive filing: 8-K item 3.02")
    assert r["stop"] == 190.0 and r["target"] == 230.0 and r["kind"] == "blocked"
    assert shadow.row(date(2026, 10, 19), "blocked", "X", 10.0, None, ECFG)["stop"] is None
    assert shadow.blocker_class("dilutive filing: 8-K item 3.02 2026-08-20") == "filing"
    assert shadow.blocker_class("earnings in 2d") == "earnings" and shadow.blocker_class("headline: offering") == "headline"


def _bars(path):
    idx = pd.bdate_range("2026-10-20", periods=len(path))
    c = pd.Series(path, index=idx, dtype=float)
    return c + 0.5, c - 0.5, c


def test_the_replay_exits_at_the_first_touch_or_the_max_hold():
    r = {"date": "2026-10-19", "price": 100.0, "stop": 90.0, "target": 130.0}
    h, lo, c = _bars([101, 95, 89, 120])
    assert shadow.replay(r, h, lo, c, 60) == {"bracket_ret": approx(-0.1), "exit": "stop", "days": 3}
    h, lo, c = _bars([105, 131])
    assert shadow.replay(r, h, lo, c, 60)["exit"] == "target"
    h, lo, c = _bars([100, 101, 102])
    t = shadow.replay(r, h, lo, c, 2)
    assert t["exit"] == "time" and abs(t["bracket_ret"] - 0.01) < 1e-12
    assert shadow.replay(r, h, lo, c, 60)["exit"] is None                     # still open: marked, not closed


def test_a_name_blocked_every_day_counts_once_per_four_weeks():
    rows = [shadow.row(date(2026, 10, d), "blocked", "SNOW", 200.0, 5.0, ECFG, reason="dilutive filing: x") for d in range(1, 30)]
    rows.append(shadow.row(date(2026, 10, 5), "blocked", "SNOW", 200.0, 5.0, ECFG, reason="earnings in 3d"))
    rows.append(shadow.row(date(2026, 10, 5), "bought", "AMD", 600.0, 20.0, ECFG))
    kept = shadow.once_per_spell(pd.DataFrame(rows))
    assert sorted(kept[kept["kind"] == "blocked"]["date"]) == ["2026-10-01", "2026-10-05", "2026-10-29"]
    assert (kept["kind"] == "bought").sum() == 1


def test_the_retirement_rule_fires_only_on_enough_closed_brackets_that_did_no_worse():
    s = {"bought": {"n": 80, "n_closed": 60, "bracket_ret_mean": 0.01},
         "bought_full": {"n": 50, "n_closed": 40, "bracket_ret_mean": 0.02},
         "blocked:filing": {"n": 40, "n_closed": 35, "bracket_ret_mean": 0.015},
         "blocked:headline": {"n": 40, "n_closed": 35, "bracket_ret_mean": -0.05},
         "analyst_remove": {"n": 10, "n_closed": 8, "bracket_ret_mean": 0.09},
         "half_size": {"n": 40, "n_closed": 31, "bracket_ret_mean": 0.015}}
    flags = shadow.retirement_due(s, {"min_closed": 30})
    assert len(flags) == 1 and flags[0].startswith("blocked:filing")      # half_size is judged against bought_full
    assert shadow.retirement_due(s, {"min_closed": 5}) and any(f.startswith("analyst_remove") for f in
                                                               shadow.retirement_due(s, {"min_closed": 5}))


def test_outcomes_and_summary_on_a_small_book():
    idx = pd.bdate_range("2026-10-19", periods=80)
    up, flat = np.linspace(100, 140, 80), np.full(80, 100.0)
    panel = {f: pd.DataFrame({"AMD": up + d, "SNOW": flat + d, "SPY": flat + d}, index=idx)
             for f, d in (("close", 0.0), ("high", 0.5), ("low", -0.5))}
    book = pd.DataFrame([shadow.row(date(2026, 10, 19), "bought", "AMD", 100.0, 5.0, ECFG),
                         shadow.row(date(2026, 10, 19), "blocked", "SNOW", 100.0, 5.0, ECFG, reason="dilutive filing: x")])
    df = shadow.outcomes(book, panel, panel["close"]["SPY"], 60)
    s = shadow.summary(df)
    assert s["bought"]["bracket_ret_mean"] == approx(0.3) and s["blocked:filing"]["bracket_ret_mean"] == approx(0.0)   # target vs time
    assert s["bought"]["excess_20d_mean"] > 0 and "SHADOW BOOK" in shadow.markdown({"summary": s, "flags": [],
                                                                                    "rows": 2, "since": "2026-10-19"})


def test_a_desk_run_writes_its_decisions_to_the_shadow_book(wired, cfg):   # noqa: F811
    from desk import state as state_mod
    holder, sent = wired
    holder["broker"] = FakeBroker()
    run(["--no-analyst"])
    book = state_mod.STATE_DIR / "shadow" / "book.jsonl"
    rows = [json.loads(x) for x in book.read_text().splitlines()]
    bought = [r for r in rows if r["kind"] == "bought"]
    assert len(bought) == len(holder["broker"].brackets) >= 1
    assert all(r["stop"] is not None and r["stop"] < r["price"] for r in bought)
    holder["broker"] = FakeBroker()
    run(["--no-analyst", "--dry-run"])                                     # a dry run records nothing
    assert len(book.read_text().splitlines()) == len(rows)

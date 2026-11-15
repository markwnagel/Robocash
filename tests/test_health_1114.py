"""Monitor 2026-11-13 and audit 2026-11-14: F-60, F-62, F-63, F-64, F-65, the candidate funnel and the stop comment."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from desk import monitor as M
from desk import parity
from desk import report as R
from desk.analyst import ANALYST_SYSTEM, size_note
from tests.test_run_daily import FakeBroker, run, wired  # noqa: F401  (wired is a fixture)

ROOT = Path(__file__).resolve().parent.parent
DATES = pd.bdate_range("2026-11-02", periods=12)


# ---------------------------------------------------------------- 1. stale checkout is a hard alarm (F-62)

def _start(branch, behind, day="2026-11-09"):
    return {"event": "run_start", "job": "desk", "ts": f"{day}T21:40:00+00:00", "run_id": "r1", "code": "dadaf11",
            "branch": branch, "behind_main": behind}


def test_a_desk_run_behind_main_is_a_sev2_and_off_main_a_sev1():
    behind = M.stale_checkout_alarms([_start("main", 3)])
    assert [a["severity"] for a in behind] == ["SEV2"] and "3 commits behind origin/main" in behind[0]["title"]
    off = M.stale_checkout_alarms([_start("r38-counters", 5)])
    assert off[0]["severity"] == "SEV1" and "not main" in off[0]["title"]
    assert M.stale_checkout_alarms([_start("main", 0)]) == []
    assert M.stale_checkout_alarms([_start("main", None)]) == []                     # never fetched: no claim
    assert M.stale_checkout_alarms([{**_start("main", 3), "job": "report"}]) == []   # the desk is what trades
    keys = {a["key"] for a in M.hard_alarms({}, {}, None, [_start("main", 3), _start("main", 3)])}
    assert keys == {"desk-stale-code-2026-11-09"}                                    # one per day


# ---------------------------------------------------------------- 2. summaries survive a long trade day (F-60)

def test_the_summary_budget_grows_with_the_trade_count():
    assert R.summary_budget(1) < R.summary_budget(19) <= 16000
    assert R.summary_budget(19) > 2000                                  # the fixed budget that ran out on 11-06


def test_a_cut_off_reply_keeps_every_summary_it_finished():
    full = [{"symbol": s, "kind": "bought", "summary": f"Bought {s}. Momentum leader."} for s in ("AZN", "CPAY", "DE")]
    text = json.dumps({"summaries": full})
    cut = text[: text.index('"DE"') + 10]                                # stops inside the third object
    got = R.salvage_summaries(cut)
    assert [g["symbol"] for g in got] == ["AZN", "CPAY"]


def test_a_truncated_reply_paraphrases_the_trades_it_reached(monkeypatch):
    moves = [{"kind": "bought", "symbol": s, "qty": 1, "thesis": "rank 1, 6m mom 50%, 20d 10%", "entry_price": 10.0,
              "stop": 8.0} for s in ("AZN", "CPAY", "DE")]
    full = json.dumps({"summaries": [{"symbol": s, "kind": "bought", "summary": f"Bought {s} on strength."}
                                     for s in ("AZN", "CPAY", "DE")]})
    text = full[: full.index('"DE"') + 10]
    resp = SimpleNamespace(stop_reason="max_tokens", content=[SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=0,
                                                 cache_creation_input_tokens=0, server_tool_use=None))
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kw: resp)))
    out = R.plain_summaries(moves, {"report": {"summary_model": "claude-sonnet-5"}})
    assert out["bought:AZN"] == "Bought AZN on strength." and out["bought:CPAY"] == "Bought CPAY on strength."
    assert out["bought:DE"] == R.fallback_summary("bought", moves[2])     # the unfinished one keeps its template


# ---------------------------------------------------------------- 3. the analyst sees why a holding is small (F-65)

def test_a_holding_says_why_it_is_small():
    assert "cash-limited" in size_note("rank 20, 6m mom 28%, 20d 9% | cash-limited: 1 of 54 shares")
    assert "half size" in size_note("rank 18 | half size: implied vol >> realized")
    assert "REDUCE" in size_note("rank 17 | analyst REDUCE: insider selling")
    assert size_note("rank 2, 6m mom 96%, 20d 17%") is None and size_note(None) is None
    assert "size_note" in ANALYST_SYSTEM


def test_the_holdings_the_analyst_reviews_carry_the_size_note(wired, monkeypatch):
    import run_daily
    holder, _ = wired
    seen = {}
    monkeypatch.setattr(run_daily, "review", lambda payload, cfg: seen.update(payload) or None)
    holder["broker"] = FakeBroker(equity=10000.0, last_equity=10000.0, cash=1200.0)
    run(["--no-analyst"])                                  # stages a cash-limited entry into state
    run([])
    held = seen.get("holdings") or []
    assert held and all("size_note" in h for h in held)
    assert any(h["size_note"] and "cash-limited" in h["size_note"] for h in held)


# ---------------------------------------------------------------- 4. "live had no cash" is its own cause (F-64)

def test_a_backtest_entry_live_could_not_pay_for_is_not_unexplained():
    log = [{"date": DATES[5], "symbol": "JNJ", "price": 200.0, "qty": 30},     # $6,000, signal day 4
           {"date": DATES[8], "symbol": "ROIV", "price": 20.0, "qty": 100}]    # $2,000, signal day 7
    bt = pd.DataFrame([{"symbol": "JNJ", "entry_date": DATES[5].date(), "pnl": 120.0},
                       {"symbol": "ROIV", "entry_date": DATES[8].date(), "pnl": -40.0}])
    # Live at the 5% floor on day 4 ($5,074 cash of $101,000); $9,000 investable by day 7.
    eq = pd.DataFrame({"equity": [101000.0, 101000.0], "cash": [5074.0, 14050.0]}, index=[DATES[4], DATES[7]])
    none = pd.DataFrame(columns=["date", "symbol"])
    empty = pd.DataFrame(columns=["date", "kind", "symbol", "reason"])
    a = parity.attribute(log, bt, pd.Series(dtype=float), pd.DataFrame(columns=["symbol", "entry_date", "pnl"]), [],
                         none, empty, DATES, 100000.0, gap_pts=-0.08, live_eq=eq, floor_pct=0.05)
    assert a["backtest_only_no_capacity"] == 0.12 and a["backtest_only_other"] == -0.04
    bt_side = (a["matched_backtest"] + a["backtest_only_unfilled"] + a["backtest_only_gated"]
               + a["backtest_only_no_capacity"] + a["backtest_only_other"])
    assert a["residual"] == round(-0.08 - ((a["matched_live"] + a["live_only"]) - bt_side), 2)
    # Without the briefs it cannot tell, and says "other" as before.
    b = parity.attribute(log, bt, pd.Series(dtype=float), pd.DataFrame(columns=["symbol", "entry_date", "pnl"]), [],
                         none, empty, DATES, 100000.0, gap_pts=0.0)
    assert b["backtest_only_no_capacity"] == 0.0 and b["backtest_only_other"] == 0.08


# ---------------------------------------------------------------- 5. lag wording and truncated lists (F-63)

def test_the_lag_says_before_or_after_and_shows_its_spread():
    assert parity.lag_text(-1.0, {"-3": 2, "-1": 4, "0": 7}).startswith(
        "matched names were staged live a median 1 trading day before the backtest")
    assert "-3d x2" in parity.lag_text(-1.0, {"-3": 2, "0": 7})
    assert "after the backtest" in parity.lag_text(2.0, {})
    assert "on the same day" in parity.lag_text(0.0, {})


def test_compare_entries_reports_lag_counts_with_live_first_negative():
    log = [{"date": DATES[4], "symbol": "AZN", "price": 1.0, "qty": 1},    # replay signal day 3; live staged day 0
           {"date": DATES[2], "symbol": "SN", "price": 1.0, "qty": 1}]     # replay signal day 1; live day 1
    bought = pd.DataFrame([{"date": DATES[0].date().isoformat(), "symbol": "AZN"},
                           {"date": DATES[1].date().isoformat(), "symbol": "SN"}])
    e = parity.compare_entries(log, bought, pd.DataFrame(columns=["date", "kind", "symbol", "reason"]), DATES)
    assert e["lag_counts"] == {"-3": 1, "0": 1} and e["median_lag_days"] == -1.5


def test_cut_lists_say_how_many_were_left_out():
    assert parity.shown(["a"] * 6, 9) == " (latest 6 of 9)"
    assert parity.shown(["a"] * 3, 3) == ""


# ---------------------------------------------------------------- 6. config comment, 7. candidate funnel

def test_the_stop_comment_no_longer_says_live_is_unchanged():
    text = (ROOT / "config.yaml").read_text()
    assert "Live not changed" not in text and "anchored_to_fill" in text
    yaml.safe_load(text)


def test_the_brief_says_where_the_candidates_went(wired):
    holder, sent = wired
    holder["broker"] = FakeBroker(equity=100000.0, last_equity=100000.0, cash=100000.0)
    run(["--no-analyst"])
    assert "candidate funnel:" in sent[-1] and " staged" in sent[-1]


def test_the_funnel_names_the_cash_floor_when_it_stops_a_pick():
    import run_daily
    line = run_daily.funnel_text({"eligible": 41, "blocked": 3, "slots": 1, "picked": 1, "investable_cash": 0.0,
                                  "unaffordable": 1, "sized": 0, "after_pacing": 0, "after_analyst": 0, "staged": 0})
    assert line == ("candidate funnel: 41 eligible (3 blocked by facts) -> 1 open slot -> 1 picked -> "
                    "1 unaffordable ($0 investable above the cash floor) -> 0 staged")

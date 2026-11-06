"""Monitor 2026-11-06, F-55..F-59: the findings from the switch to the new paper account."""
from datetime import date

from desk import monitor as M
from desk import parity as P
from desk.execution import sync_positions
from tests.test_engineering_review import _held, _live
from tests.test_run_daily import FakeBroker, run, wired  # noqa: F401  (wired is a fixture)


def test_the_fill_note_carries_the_share_count_and_names_a_partial_fill():
    b = FakeBroker(positions=_live("AAA", qty=5))
    st = {"positions": {"AAA": {**_held("AAA", 0, qty=8, price=48.0)["AAA"], "filled": False, "order_id": "o1"}}}
    notes = sync_positions(st, b, b.positions(), date.today())
    assert any("filled at 50.00 vs 48.00 staged" in n and "5 of 8 shares filled" in n for n in notes)
    assert any(n.startswith("AAA partial fill: 5 of 8 shares") for n in notes)


def test_a_full_fill_is_not_called_partial():
    b = FakeBroker(positions=_live("AAA", qty=8))
    st = {"positions": {"AAA": {**_held("AAA", 0, qty=8, price=48.0)["AAA"], "filled": False, "order_id": "o1"}}}
    notes = sync_positions(st, b, b.positions(), date.today())
    assert any("8 of 8 shares filled" in n for n in notes) and not any("partial fill" in n for n in notes)


def test_the_owner_override_raises_the_daily_cap_for_one_run_and_says_so(wired, cfg):
    holder, sent = wired
    holder["broker"] = FakeBroker(equity=100000.0, last_equity=100000.0, cash=100000.0)
    run(["--no-analyst", "--max-new-entries", "6"])
    assert len(holder["broker"].brackets) == 6 > cfg["portfolio"]["max_new_entries_per_day"]
    assert "owner override: up to 6 new entries this run" in sent[-1]


def test_an_entry_cut_by_the_remaining_cash_is_labelled(wired, cfg):
    holder, sent = wired
    holder["broker"] = FakeBroker(equity=10000.0, last_equity=10000.0, cash=1200.0)
    run(["--no-analyst"])
    assert holder["broker"].brackets and "cash-limited:" in sent[-1]


def test_the_classified_news_feed_is_counted_under_the_monitor_names():
    ev = [{"event": "feed_quality", "feed": "news_classified", "symbol": "A", "shown": 10, "unrelated": 3},
          {"event": "feed_quality", "feed": "news_classified", "symbol": "B", "shown": 4, "unrelated": 0},
          {"event": "feed_quality", "feed": "news", "symbol": "A", "kept": 12, "dropped_unrelated": 2}]
    f = M.feed_signals(ev)
    assert f["news_classified"] == {"symbols": 2, "kept": 11, "dropped_unrelated": 3, "most_dropped": [["A", 3]]}
    assert f["news"]["kept"] == 12 and f["news"]["dropped_unrelated"] == 2


def test_a_replay_with_no_entries_says_its_gap_means_nothing():
    res = {"window": "2026-11-06 to 2026-11-06", "live_return": 0.0, "backtest_return": 0.0, "equity_gap_pts": 0.0,
           "spy_return": 0.0, "since_release": {}, "live_cash_share": 0.05, "backtest_cash_share": 1.0,
           "entries": {"both": 0, "backtest_only": 0, "live_only": 19, "overlap": 0.0, "backtest_only_names": [],
                       "live_only_names": []}, "backtest_entries": 0,
           "attribution_pts": {}}
    assert any("no positions yet" in line for line in P.markdown(res))
    assert not any("no positions yet" in line for line in P.markdown({**res, "backtest_entries": 3}))

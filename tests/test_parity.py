"""Live vs backtest parity (desk/parity.py, engineering review 2026-10-17 item 9)."""
from datetime import date

import pandas as pd

from desk import parity


def test_live_equity_reads_each_brief_s_first_equity_line(tmp_path):
    (tmp_path / "2026-10-15.md").write_text("DESK BRIEF\nEquity $9,710 (day -2.6%) | HWM $10,443 | cash $4,100 (42%)\n")
    (tmp_path / "2026-10-16.md").write_text("Equity $9,361 (day -3.3%) | HWM | cash $5,929 (63%)\nEquity $1 | cash $1\n")
    eq = parity.live_equity(tmp_path)
    assert list(eq["equity"]) == [9710.0, 9361.0] and list(eq["cash"]) == [4100.0, 5929.0]


def test_entries_match_within_a_few_days_and_restages_count_once():
    dates = pd.bdate_range("2026-10-01", periods=15)
    d = lambda i: dates[i].date().isoformat()  # noqa: E731
    bt = [{"date": dates[2], "symbol": "DDOG"}, {"date": dates[3], "symbol": "MRNA"}, {"date": dates[5], "symbol": "AMD"}]
    live = pd.DataFrame([{"date": d(4), "symbol": "DDOG"},                 # staged 3 days after the backtest
                         {"date": d(6), "symbol": "TXG"}, {"date": d(7), "symbol": "TXG"},   # re-staged: once
                         {"date": d(4), "symbol": "AMD"},
                         {"date": d(14), "symbol": "VEEV"}])               # the last day: cannot have filled
    book = pd.DataFrame([{"date": d(2), "kind": "blocked", "symbol": "MRNA", "reason": "dilutive filing: 8-K 3.02"}])
    e = parity.compare_entries(bt, live, book, dates, last_day=d(14))
    assert e["both"] == 2 and e["backtest_only"] == 1 and e["live_only"] == 1
    assert e["backtest_only_names"] == [f"{d(2)} MRNA (blocked: dilutive filing: 8-K 3.02)"]
    assert e["live_only_names"] == [f"{d(6)} TXG"] and e["median_lag_days"] == 1.5      # lags 3 (DDOG) and 0 (AMD)


def test_a_large_gap_raises_sev2_and_the_report_shows_it():
    res = {"window": "a to b", "live_return": -0.064, "backtest_return": -0.025, "spy_return": -0.001,
           "equity_gap_pts": -6.1, "live_cash_share": 0.5, "backtest_cash_share": 0.22,
           "entries": {"overlap": 0.58, "both": 22, "backtest_only": 6, "live_only": 10, "median_lag_days": 0.0,
                       "backtest_only_names": ["x DELL"], "live_only_names": []}}
    a = parity.alarms(res, {"parity": {"equity_gap_alarm_pts": 5}}, date(2026, 10, 19))
    assert a[0]["severity"] == "SEV2" and "-6.1 pts" in a[0]["title"]
    assert parity.alarms(dict(res, equity_gap_pts=-3.9), {}, date(2026, 10, 19)) == []
    md = "\n".join(parity.markdown(res))
    assert "gap -6.1 pts" in md and "cash: live 50%, backtest 22%" in md and "x DELL" in md
    assert "unavailable" in "\n".join(parity.markdown({"error": "boom"}))


def test_the_comparison_states_its_tolerance_and_how_much_is_a_live_gate():
    """2026-10-21: the monitor called the matcher date-exact and the backtest-only side unexplained (R-50, R-53)."""
    dates = pd.bdate_range("2026-10-01", periods=15)
    d = lambda i: dates[i].date().isoformat()  # noqa: E731
    bt = [{"date": dates[3], "symbol": "MRNA"}, {"date": dates[5], "symbol": "CNC"}]
    book = pd.DataFrame([{"date": d(2), "kind": "blocked", "symbol": "MRNA", "reason": "dilutive filing: 8-K 3.02"}])
    e = parity.compare_entries(bt, pd.DataFrame(columns=["date", "symbol"]), book, dates, tolerance=5)
    assert e["match_tolerance_days"] == 5 and e["backtest_only"] == 2 and e["backtest_only_gated_live"] == 1
    res = {"window": "a to b", "live_return": 0.0, "backtest_return": 0.0, "spy_return": 0.0, "equity_gap_pts": 0.0,
           "live_cash_share": 0.3, "backtest_cash_share": 0.2, "entries": e}
    md = "\n".join(parity.markdown(res))
    assert "within 5 trading days counts as a match" in md and "1 of the 2 backtest-only names were kept out live" in md


def test_the_gap_is_attributed_in_points_and_adds_up():
    """Monitor R-58 (2026-10-22): the 5.2-point gap had no breakdown. Each entry is valued and sorted by cause."""
    dates = pd.bdate_range("2026-10-01", periods=20)
    d = lambda i: dates[i].date().isoformat()  # noqa: E731
    log = [{"date": dates[2], "symbol": "TEAM", "price": 100.0, "qty": 10},     # both took it
           {"date": dates[4], "symbol": "MRNA", "price": 50.0, "qty": 10},      # live gate kept it out
           {"date": dates[6], "symbol": "IOVA", "price": 10.0, "qty": 100}]     # backtest only, still open
    bt_trades = pd.DataFrame([{"symbol": "TEAM", "entry_date": dates[2].date(), "pnl": 200.0},
                              {"symbol": "MRNA", "entry_date": dates[4].date(), "pnl": -100.0}])
    last = pd.Series({"IOVA": 12.0})
    live_closed = pd.DataFrame([{"symbol": "TEAM", "entry_date": d(1), "pnl": -150.0},
                                {"symbol": "NTNX", "entry_date": d(9), "pnl": -50.0}])
    live_open = [("SN", {"entry_date": d(10), "entry_price": 20.0, "last_price": 21.0, "qty": 10})]
    bought = pd.DataFrame([{"date": d(1), "symbol": "TEAM"}, {"date": d(9), "symbol": "NTNX"}, {"date": d(10), "symbol": "SN"}])
    a = parity.attribute(log, bt_trades, last, live_closed, live_open, bought, {"MRNA"}, dates, 10000.0, gap_pts=-5.4)
    assert a["matched_live"] == -1.5 and a["matched_backtest"] == 2.0 and a["matched_difference"] == -3.5
    assert a["backtest_only_gated"] == -1.0 and a["backtest_only_other"] == 2.0 and a["live_only"] == -0.4
    assert a["residual"] == round(-5.4 - ((-1.5 - 0.4) - (2.0 - 1.0 + 2.0)), 2)       # -0.5: fills, sizing, cash
    md = "\n".join(parity.markdown({"window": "a", "live_return": 0, "backtest_return": 0, "spy_return": 0,
                                    "equity_gap_pts": -5.4, "live_cash_share": 0.5, "backtest_cash_share": 0.2,
                                    "attribution_pts": a,
                                    "entries": {"overlap": 1, "both": 1, "backtest_only": 2, "live_only": 2,
                                                "median_lag_days": None, "backtest_only_names": [],
                                                "live_only_names": []}}))
    assert "same names -3.5" in md and "backtest-only kept out live +1.0" in md and "fills, sizing and idle cash -0.5" in md

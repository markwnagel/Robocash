"""The same-names parity gap, split into position size and per-trade return."""
import pandas as pd

from desk import parity


def test_the_same_names_gap_splits_into_size_and_return():
    """Monitor R-59 (2026-10-23): -4.6 of the -5.0 pt gap was on names both sides bought, with no reason why."""
    dates = pd.bdate_range("2026-10-01", periods=20)
    d = lambda i: dates[i].date().isoformat()  # noqa: E731
    log = [{"date": dates[2], "symbol": "TEAM", "price": 100.0, "qty": 10},      # $1,000 in the backtest, +20%
           {"date": dates[4], "symbol": "GTLB", "price": 50.0, "qty": 20}]       # $1,000, still open, +10%
    bt_trades = pd.DataFrame([{"symbol": "TEAM", "entry_date": dates[2].date(), "pnl": 200.0, "reason": "target",
                               "exit_date": dates[9].date()}])
    last = pd.Series({"GTLB": 55.0})
    live_closed = pd.DataFrame([{"symbol": "TEAM", "entry_date": d(1), "entry_price": 100.0, "qty": 5, "pnl": -50.0,
                                 "exit_reason": "bracket/stop/target", "exit_date": d(6)}])        # half size, -10%
    live_open = [("GTLB", {"entry_date": d(3), "entry_price": 50.0, "last_price": 55.0, "qty": 20})]   # the same
    a = parity.matched_split(log, bt_trades, last, live_closed, live_open, {t: i for i, t in enumerate(dates)}, 10000.0)
    # TEAM: size (500 - 1000) x 20% = -100; return 500 x (-10% - 20%) = -150. GTLB: identical, 0.
    assert a["matched_size"] == -1.0 and a["matched_return"] == -1.5 and a["matched_pairs"] == 2
    assert a["matched_worst"][0]["symbol"] == "TEAM" and a["matched_worst"][0]["size_ratio"] == 0.5
    assert a["matched_size_ratio_median"] == 0.75

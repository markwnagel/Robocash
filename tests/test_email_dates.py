"""The email's equity history: each Alpaca daily point belongs to the New York session it closed."""
from datetime import date
from types import SimpleNamespace

import report_daily
from desk.report import performance


class FakeTrading:
    # Alpaca 1D history as returned on 2026-11-01: stamped 00:00 UTC after each session.
    def get_portfolio_history(self, req):
        return SimpleNamespace(timestamp=[1792195200, 1793318400, 1793404800],      # 10-17, 10-30, 10-31 00:00 UTC
                               equity=[9368.6, 9343.33, 9114.19])


def test_each_close_is_dated_by_its_new_york_session():
    s = report_daily.equity_series(FakeTrading())
    assert [d.date() for d in s.index] == [date(2026, 10, 16), date(2026, 10, 29), date(2026, 10, 30)]
    assert s[s.index == "2026-10-16"].iloc[0] == 9368.6           # the v2.0 baseline, as the README states it


def test_since_the_release_counts_from_that_close():
    import pandas as pd
    eq = report_daily.equity_series(FakeTrading())
    spy = pd.Series([700.0, 760.0, 756.0], index=eq.index)
    rows = performance(eq, spy, date(2026, 11, 1), {"name": "v2.1", "since": date(2026, 10, 30)})
    assert rows[0]["period"] == "Since v2.1" and abs(rows[0]["desk_pct"]) < 1e-12

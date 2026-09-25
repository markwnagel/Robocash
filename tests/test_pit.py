import numpy as np
import pandas as pd

from desk import pit


def test_symbol_rules_match_the_live_universe_but_keep_names_that_fell_to_otc():
    bad = ["ETF", "Trust", "Acquisition"]
    assert pit.keep_symbol("AAPL", "Apple Inc. Common Stock", "NASDAQ", bad)
    assert not pit.keep_symbol("BRK.B", "Berkshire", "NYSE", bad)          # not alphabetic
    assert not pit.keep_symbol("ABCDEF", "Six letters", "NASDAQ", bad)
    assert not pit.keep_symbol("SPY", "SPDR S&P 500 ETF TRUST", None, bad)
    assert not pit.keep_symbol("QQQX", "Some fund", "ARCA", bad)           # ETF venue
    assert not pit.keep_symbol("SPAC", "Big Acquisition Corp", "NASDAQ", bad)
    # A company that was NYSE-listed and now trades OTC (or that Alpaca no longer lists at all) is exactly
    # the name a survivor list forgets. Dropping it would bring the bias back.
    assert pit.keep_symbol("SIVB", "SVB FINANCIAL GROUP", None, bad)
    assert pit.keep_symbol("BBBYQ", "Bed Bath & Beyond", "OTC", bad)
    # A delisted exchange-traded note is filed under its issuing bank: no "ETN" in the name to catch.
    assert not pit.keep_symbol("DGAZ", "CREDIT SUISSE AG", None, bad)
    assert not pit.keep_symbol("FNGB", "BANK OF MONTREAL /CAN/", None, bad)
    assert pit.keep_symbol("BMO", "Bank of Montreal Common Stock", "NYSE", bad)   # the bank itself, named by Alpaca


def _frame(values: dict, dates) -> pd.DataFrame:
    return pd.DataFrame({k: np.full(len(dates), v, dtype=float) for k, v in values.items()}, index=dates)


def test_weekly_members_apply_the_live_rule_with_the_price_that_traded_then():
    dates = pd.bdate_range("2019-01-01", periods=40)
    close = _frame({"BIG": 100, "SPLIT": 4, "REV": 20, "THIN": 50, "DUP": 100}, dates)
    volume = _frame({"BIG": 1e6, "SPLIT": 1e7, "REV": 2e6, "THIN": 1e4, "DUP": 1e6}, dates)
    # SPLIT later split 10:1, so its adjusted $4 was a $40 stock at the time: in. REV later reverse-split
    # 1:10, so its adjusted $20 was a $2 stock: out. Adjusted prices alone would get both backwards.
    factor = _frame({"BIG": 1, "SPLIT": 10, "REV": 0.1, "THIN": 1, "DUP": 1}, dates)
    weeks = pit.week_starts(dates)[-2:]
    m = pit.weekly_members(close, volume, factor, weeks, min_adv=20e6, min_price=5, max_symbols=10)
    for w in weeks:
        assert "SPLIT" in m[w] and "BIG" in m[w]
        assert "REV" not in m[w]                  # $2 as traded
        assert "THIN" not in m[w]                 # $0.5M a day
        assert not ({"BIG", "DUP"} <= set(m[w]))  # one company, two tickers: counted once
    capped = pit.weekly_members(close, volume, factor, weeks, 20e6, 5, max_symbols=1)
    assert all(len(v) == 1 for v in capped.values())


def test_membership_round_trips_through_the_committed_interval_file():
    dates = pd.bdate_range("2020-01-06", periods=30)
    weeks = pit.week_starts(dates)
    members = {w: (["AAA", "BBB"] if i % 2 == 0 else ["AAA", "CCC"]) for i, w in enumerate(weeks)}
    iv = pit.to_intervals(members)
    assert len(iv[iv.symbol == "AAA"]) == 1                       # a continuous member is one row
    daily = pit.expand(iv.assign(first_week=iv.first_week.dt.date, last_week=iv.last_week.dt.date), dates)
    for d in dates:
        w = weeks[weeks <= d].max()
        assert set(daily.columns[daily.loc[d].to_numpy()]) == set(members[w])


def test_monthly_superset_is_generous_but_bounded():
    months = pd.period_range("2019-01", periods=3, freq="M")
    dv = pd.DataFrame({"A": [30e6, 5e6, 5e6], "B": [5e6, 5e6, 25e6], "C": [1e6, 1e6, 1e6]}, index=months)
    px = pd.DataFrame({"A": [10, 10, 10], "B": [10, 10, 10], "C": [10, 10, 10]}, index=months, dtype=float)
    assert pit.monthly_superset(dv, px, 10e6, 4, top=10) == {"A", "B"}   # any month counts
    assert pit.monthly_superset(dv, px, 10e6, 4, top=1) == {"A", "B"}    # top per month, not overall

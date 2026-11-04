"""The parity attribution reconciles (monitor F-52 / R-60) and a live gate counts only on its own day (F-45 / R-53)."""
import pandas as pd

from desk import parity

DATES = pd.bdate_range("2026-10-01", periods=25)
d = lambda i: DATES[i].date().isoformat()  # noqa: E731


def test_one_pairing_feeds_every_number_so_the_split_adds_up():
    """2026-11-03: 'same names +0.3' beside a split of -0.9 and +0.7, and 17 matches beside 15 pairs. The
    headline used a many-to-many match, the split a one-to-one match, and an open rest replaced a partial close."""
    log = [{"date": DATES[2], "symbol": "TEAM", "price": 100.0, "qty": 10},
           {"date": DATES[4], "symbol": "TEAM", "price": 102.0, "qty": 10},     # re-entry inside the window
           {"date": DATES[6], "symbol": "HNGE", "price": 50.0, "qty": 20},
           {"date": DATES[8], "symbol": "SN", "price": 20.0, "qty": 50}]       # live staged it, never filled
    bt_trades = pd.DataFrame([{"symbol": "TEAM", "entry_date": DATES[2].date(), "pnl": 100.0},
                              {"symbol": "TEAM", "entry_date": DATES[4].date(), "pnl": -40.0},
                              {"symbol": "HNGE", "entry_date": DATES[6].date(), "pnl": 150.0}])
    last = pd.Series({"SN": 22.0})
    live_closed = pd.DataFrame([{"symbol": "TEAM", "entry_date": d(1), "entry_price": 100.0, "qty": 5, "pnl": -30.0},
                                {"symbol": "HNGE", "entry_date": d(5), "entry_price": 51.0, "qty": 5, "pnl": 20.0}])
    live_open = [("HNGE", {"entry_date": d(5), "entry_price": 51.0, "last_price": 49.0, "qty": 5})]   # the rest
    bought = pd.DataFrame([{"date": d(1), "symbol": "TEAM"}, {"date": d(5), "symbol": "HNGE"},
                           {"date": d(7), "symbol": "SN"}])
    empty = pd.DataFrame(columns=["date", "kind", "symbol", "reason"])
    a = parity.attribute(log, bt_trades, last, live_closed, live_open, bought, empty, DATES, 10000.0, gap_pts=-3.0)
    assert a["matched_pairs"] == 2 and a["matched_unfilled"] == 1          # TEAM once, HNGE once; SN never filled
    assert a["matched_live"] == round((-30 + 20 - 10) / 100, 2)             # HNGE's closed half plus its open half
    assert a["matched_backtest"] == 2.5                                      # TEAM's first entry + HNGE
    assert round(a["matched_size"] + a["matched_return"], 2) == a["matched_difference"]
    assert a["backtest_only_unfilled"] == 1.0 and a["backtest_only_other"] == -0.4   # SN marked; TEAM's re-entry
    bt_side = a["matched_backtest"] + a["backtest_only_unfilled"] + a["backtest_only_gated"] + a["backtest_only_other"]
    assert a["residual"] == round(-3.0 - ((a["matched_live"] + a["live_only"]) - bt_side), 2)


def test_a_gate_explains_a_backtest_only_entry_only_on_its_signal_day_and_is_split_by_rule():
    """2026-10-30: EXEL, bought by the replay on 10-22 with its report 7 days out, was labelled with a 10-26
    'earnings in 3d' block, and four retired dilution blocks were reported in one number with a current rule."""
    log = [{"date": DATES[3], "symbol": "EXEL", "price": 60.0, "qty": 10},    # signal day 2; blocked live from day 6
           {"date": DATES[5], "symbol": "DELL", "price": 100.0, "qty": 5},    # signal day 4: a retired dilution block
           {"date": DATES[7], "symbol": "CNC", "price": 50.0, "qty": 10},     # signal day 6: the current earnings rule
           {"date": DATES[9], "symbol": "MRNA", "price": 30.0, "qty": 10}]    # signal day 8: an analyst veto
    bt_trades = pd.DataFrame([{"symbol": s, "entry_date": DATES[i].date(), "pnl": v}
                              for s, i, v in (("EXEL", 3, 30.0), ("DELL", 5, -50.0), ("CNC", 7, 20.0), ("MRNA", 9, -10.0))])
    book = pd.DataFrame([{"date": d(6), "kind": "blocked", "symbol": "EXEL", "reason": "earnings in 3d"},
                         {"date": d(4), "kind": "blocked", "symbol": "DELL", "reason": "dilutive filing: 8-K item 3.02 2026-06-17"},
                         {"date": d(6), "kind": "blocked", "symbol": "CNC", "reason": "earnings in 5d"},
                         {"date": d(9), "kind": "analyst_remove", "symbol": "MRNA", "reason": "Severity 0.7"}])
    none = pd.DataFrame(columns=["date", "symbol"])
    e = parity.compare_entries(log, none, book, DATES)
    assert e["backtest_only_gated_live"] == 3                                 # not EXEL: its block came 4 days later
    assert f"{d(2)} EXEL" in e["backtest_only_names"]
    assert f"{d(4)} DELL (blocked: dilutive filing: 8-K item 3.02 2026-06-17)" in e["backtest_only_names"]
    a = parity.attribute(log, bt_trades, pd.Series(dtype=float), pd.DataFrame(columns=["symbol", "entry_date", "pnl"]),
                         [], none, book, DATES, 10000.0, gap_pts=0.0)
    assert a["backtest_only_other"] == 0.3 and a["backtest_only_gated_retired"] == -0.5
    assert a["backtest_only_gated_earnings"] == 0.2 and a["backtest_only_gated_unsimulated"] == -0.1
    assert a["backtest_only_gated"] == -0.4
    md = "\n".join(parity.markdown({"window": "a", "live_return": 0, "backtest_return": 0, "spy_return": 0,
                                    "equity_gap_pts": 0.0, "live_cash_share": 0.5, "backtest_cash_share": 0.2,
                                    "attribution_pts": a, "entries": e}))
    assert "retired rule (dilution block) +0.5" in md and "earnings block (the replay applies it too" in md
    assert parity.gate_class("blocked: dilutive filing: x", offering_block=True) == "unsimulated"

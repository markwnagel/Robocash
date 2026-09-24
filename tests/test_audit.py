import numpy as np
import pandas as pd

from audit import post_exit_return, trade_stats


def _closes():
    idx = pd.bdate_range("2026-06-01", periods=80)
    return pd.DataFrame({"AAA": np.linspace(100, 140, 80), "BBB": np.linspace(50, 30, 80)}, index=idx)


def test_post_exit_return():
    closes = _closes()
    exit_date = closes.index[10]
    r = post_exit_return(closes, "AAA", exit_date, float(closes["AAA"].iloc[10]), 20)
    assert abs(r - (closes["AAA"].iloc[30] / closes["AAA"].iloc[10] - 1)) < 1e-9
    assert post_exit_return(closes, "ZZZ", exit_date, 100.0, 20) is None
    assert post_exit_return(closes, "AAA", closes.index[-5], 100.0, 20) is None  # not enough history yet


def test_trade_stats_by_reason():
    closes = _closes()
    trades = pd.DataFrame([
        {"symbol": "AAA", "exit_date": closes.index[5].date(), "exit_price": float(closes["AAA"].iloc[5]),
         "pnl": 50.0, "pnl_pct": 0.05, "hold_days": 20, "exit_reason": "signal decay"},
        {"symbol": "AAA", "exit_date": closes.index[8].date(), "exit_price": float(closes["AAA"].iloc[8]),
         "pnl": 30.0, "pnl_pct": 0.03, "hold_days": 61, "exit_reason": "max hold 60d"},
        {"symbol": "BBB", "exit_date": closes.index[5].date(), "exit_price": float(closes["BBB"].iloc[5]),
         "pnl": -40.0, "pnl_pct": -0.08, "hold_days": 4, "exit_reason": "analyst REMOVE: guidance cut"},
    ])
    s = trade_stats(trades, closes, 20)
    assert s["n_trades"] == 3 and abs(s["win_rate"] - 2 / 3) < 1e-3
    assert s["sum_pnl_usd"] == 40.0
    by = s["by_exit_reason"]
    assert set(by) == {"signal decay", "max hold 60d", "analyst remove"}
    assert by["signal decay"]["avg_post_exit_20d_ret"] > 0      # we sold a winner that kept going
    assert by["analyst remove"]["avg_post_exit_20d_ret"] < 0    # the veto was right
    assert trade_stats(pd.DataFrame(), None, 20) == {"n_trades": 0}

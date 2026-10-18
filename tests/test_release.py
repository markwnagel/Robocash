"""Release tracking (config release, v2.0 from 2026-10-17): results since the cut beside results since inception."""
from datetime import date

import pandas as pd

from desk import state as state_mod
from tests.test_run_daily import wired  # noqa: F401  (fixture)


def test_the_release_helper_reads_the_config(cfg):
    from desk.config import release
    r = release(cfg)
    assert r["name"] == "v2.0" and r["since"] == date(2026, 10, 16)
    assert release({}) == {"name": None, "since": None}


def test_performance_leads_with_the_release_s_own_result():
    from desk.report import performance
    idx = pd.to_datetime(["2026-09-11", "2026-09-30", "2026-10-16", "2026-10-23"])
    eq = pd.Series([10000.0, 10200.0, 9361.0, 9800.0], index=idx)
    spy = pd.Series([600.0, 610.0, 605.0, 611.05], index=idx)
    rows = performance(eq, spy, date(2026, 10, 23), {"name": "v2.0", "since": date(2026, 10, 16)})
    assert rows[0]["period"] == "Since v2.0" and abs(rows[0]["desk_pct"] - (9800 / 9361 - 1)) < 1e-12
    assert abs(rows[0]["spy_pct"] - 0.01) < 1e-9 and any(r["period"] == "Since inception" for r in rows)
    assert performance(eq, spy, date(2026, 10, 23))[0]["period"] != "Since v2.0"          # no release: as before


def test_the_trade_log_gains_a_release_column_and_labels_the_v1_book(tmp_path, monkeypatch):
    monkeypatch.setattr(state_mod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(state_mod, "TRADES_FILE", tmp_path / "trades.csv")
    old = [f for f in state_mod.TRADE_FIELDS if f != "release"]
    (tmp_path / "trades.csv").write_text(",".join(old) + "\nHUM,2026-09-14,405.05,2026-09-17,382.46,1.0,-22.59,-0.0558,3,"
                                         "analyst REMOVE,t,379.8,488.68\n" + ",".join(old) + "\n")       # stray header
    from desk.execution import _log_close
    _log_close({"entry_date": "2026-09-16", "entry_price": 500.0, "qty": 1, "stop": 450.0}, "DELL", 480.0,
               date(2026, 10, 20), "signal decay")
    _log_close({"entry_date": "2026-10-19", "entry_price": 300.0, "qty": 2, "stop": 280.0, "release": "v2.0"}, "VEEV",
               330.0, date(2026, 10, 30), "bracket/stop/target")
    t = pd.read_csv(tmp_path / "trades.csv")
    assert list(t["release"]) == ["v1.0", "v1.0 (carried)", "v2.0"] and len(t) == 3


def test_a_v2_run_stages_positions_and_the_brief_under_the_release(wired, cfg):   # noqa: F811
    import json
    from tests.test_run_daily import FakeBroker, run
    holder, sent = wired
    holder["broker"] = FakeBroker()
    run(["--no-analyst"])
    st = json.loads(state_mod.STATE_FILE.read_text())
    assert st["positions"] and all(p["release"] == "v2.0" for p in st["positions"].values())
    assert sent[-1].splitlines()[0].endswith("v2.0")
    book = [json.loads(x) for x in (state_mod.STATE_DIR / "shadow" / "book.jsonl").read_text().splitlines()]
    assert book and all(r["release"] == "v2.0" for r in book)


def test_the_shadow_and_parity_sections_show_the_release():
    from desk import parity, shadow
    ev = {"summary": {"bought": {"n": 5, "n_closed": 2, "bracket_ret_mean": 0.01, "excess_20d_mean": None, "n_20d": 0}},
          "flags": [], "rows": 5, "since": "2026-09-12", "release": "v2.0",
          "release_summary": {"bought": {"n": 2, "n_closed": 1, "bracket_ret_mean": 0.03, "excess_20d_mean": None, "n_20d": 0}}}
    assert "v2.0 only: bought n 2 closed 1 +3.0%" in shadow.markdown(ev)
    res = {"window": "a to b", "live_return": -0.05, "backtest_return": -0.02, "spy_return": 0.0, "equity_gap_pts": -3.0,
           "live_cash_share": 0.3, "backtest_cash_share": 0.2,
           "entries": {"overlap": 1.0, "both": 1, "backtest_only": 0, "live_only": 0, "median_lag_days": 0.0,
                       "backtest_only_names": [], "live_only_names": []},
           "since_release": {"release": "v2.0", "from": "2026-10-16", "live_return": 0.02, "backtest_return": 0.01,
                             "spy_return": 0.005}}
    assert "since v2.0 (2026-10-16 close): live +2.0%, backtest +1.0%, SPY +0.5%" in "\n".join(parity.markdown(res))

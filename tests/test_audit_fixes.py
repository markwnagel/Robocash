"""Fixes from the 2026-09-12 audit: insider data hygiene, minimum lot, pre-earnings exit (backtest + live)."""
from datetime import date, timedelta

import pandas as pd

from backtest import run_backtest
from desk.facts import earnings_horizon_days, filter_insider
from desk.portfolio import size_risk_parity
from tests.test_run_daily import FakeBroker, _patched_cfg, run, wired  # noqa: F401  (wired is a fixture)


def test_filter_insider_drops_artifacts_and_reports_them():
    spot = 22.75
    rows = [
        {"code": "S", "usd": 5_000.0, "price": 22.10, "shares": 226, "filed": "2026-09-01"},     # real
        {"code": "P", "usd": 40_000.0, "price": 23.00, "shares": 1739, "filed": "2026-09-02"},   # real
        {"code": "S", "usd": 296_004_000.0, "price": 740.01, "shares": 400_000, "filed": "2026-08-20"},  # TWD-priced block
        {"code": "S", "usd": 60_000_000.0, "price": 22.50, "shares": 2_666_667, "filed": "2026-08-25"},  # affiliate block
    ]
    out = filter_insider(rows, spot, max_abs_usd=50_000_000, price_band=3.0)
    assert [t["usd"] for t in out["kept"]] == [5_000.0, 40_000.0]
    assert out["excluded_n"] == 2 and "2 of 4" in out["note"]
    # No spot price: only the size bound applies.
    out2 = filter_insider(rows, None, max_abs_usd=50_000_000, price_band=3.0)
    assert out2["excluded_n"] == 2 and all(t["usd"] <= 50_000_000 for t in out2["kept"])
    assert filter_insider([], spot, 1e8, 3.0) == {"kept": [], "excluded_n": 0, "note": None}


def test_min_shares_rejects_lots_that_cannot_carry_the_risk(cfg):
    pcfg, ecfg = cfg["portfolio"], cfg["exits"]
    # $412 stock on $10k: risk parity wants 3 shares but the 10% cap allows 2.
    assert size_risk_parity(10000, 9500, 411.85, 14.79, pcfg, ecfg)["qty"] == 2
    strict = dict(pcfg, min_shares_per_position=3)
    assert size_risk_parity(10000, 9500, 411.85, 14.79, strict, ecfg) is None
    assert size_risk_parity(10000, 9500, 68.93, 3.81, strict, ecfg)["qty"] >= 3
    assert size_risk_parity(10000, 9500, 411.85, 14.79, dict(pcfg, min_shares_per_position=1), ecfg)["qty"] == 2


def test_backtest_pre_earnings_exit_and_risk_logging(cfg, panel):
    dates = panel["close"].index
    # Every symbol reports every 30 trading days (~42 calendar days): an entry allowed by the 21-day block
    # can still be alive when the next report arrives, so the pre-earnings rule has something to do.
    syms = list(panel["close"].columns)
    earnings = pd.DataFrame([(d.date().isoformat(), s) for s in syms for d in dates[200::30]], columns=["date", "symbol"])
    earnings = pd.concat([earnings, pd.DataFrame([("", "S00")], columns=["date", "symbol"])])  # sentinel row must be ignored
    off = dict(cfg, exits=dict(cfg["exits"], earnings_exit_days_before=0))
    on = dict(cfg, exits=dict(cfg["exits"], earnings_exit_days_before=2))
    _, t_off = run_backtest(panel, off, dates[0], dates[-1], 10000.0, 10, 5, earnings=earnings)
    _, t_on = run_backtest(panel, on, dates[0], dates[-1], 10000.0, 10, 5, earnings=earnings)
    assert "pre-earnings" not in set(t_off["reason"])
    assert (t_on["reason"] == "pre-earnings").any()
    # With the rule on, no trade may span a report date.
    assert not t_on["spanned_earnings"].any()
    # Risk logging is populated and sane under risk parity.
    r = (t_on["realized_risk"] / t_on["intended_risk"]).dropna()
    assert len(r) and (r > 0).all() and (r <= 1.0 + 1e-9).all()


def test_runner_pre_earnings_exit_overrides_min_hold(wired, cfg, monkeypatch):
    from desk import state as state_mod
    holder, sent = wired
    _patched_cfg(monkeypatch, exits={"earnings_exit_days_before": 2})
    recent = (date.today() - timedelta(days=3)).isoformat()  # well inside min_hold_days
    live = {"S02": {"qty": 5, "avg_entry_price": 80.0, "current_price": 90.0, "market_value": 450.0, "unrealized_plpc": 0.1}}
    import numpy as np
    holder["broker"] = FakeBroker(cash=9000.0, positions=live)
    # Report on the next business day: for a weekend run that is Monday, which calendar-day
    # arithmetic would place 2 days out and miss - trading-day arithmetic must still catch it.
    report = np.busday_offset(date.today(), 1, roll="forward").astype("datetime64[D]").astype(object)
    holder["facts"] = {"S02": {"symbol": "S02", "earnings_date": report.isoformat(),
                               "earnings_in_days": (report - date.today()).days}}
    state_mod.save_state({**state_mod.load_state(), "positions": {
        "S02": {"entry_date": recent, "entry_price": 80.0, "qty": 5, "stop": 70.0, "target": 110.0, "filled": True, "thesis": "t"}}})
    run(["--no-analyst"])
    assert "S02" in holder["broker"].closed
    expected_td = int(np.busday_count(date.today(), report))
    assert f"pre-earnings exit ({expected_td} trading days to report)" in sent[-1]


def test_earnings_lookup_spans_the_hold_not_just_the_block(cfg):
    # Shrinking the entry block to 5 days once cut the lookup to 35 days, hiding every Q3 report in the hold.
    for block in (5, 21, 90):
        c = dict(cfg, facts=dict(cfg["facts"], exclude_earnings_within_days=block))
        assert earnings_horizon_days(c) >= max(block, cfg["exits"]["max_hold_days"])

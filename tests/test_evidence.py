"""Live vs backtest: volatility and drawdown against the current config's backtest."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from desk.evidence import live_vs_backtest, reference_from_curve

REF = {"window_days": 20, "vol_p05": 0.09, "vol_p50": 0.22, "vol_p95": 0.44, "vol_full": 0.25,
       "max_dd": -0.19, "dd_p05": -0.165, "config_id": "abc", "window": "2023-09-11 to 2026-09-11"}


def _curve(daily_returns, flat_days=0, start="2026-06-01"):
    r = [0.0] * flat_days + list(daily_returns)
    return pd.Series(10000 * np.cumprod([1.0] + [1 + x for x in r]), index=pd.bdate_range(start, periods=len(r) + 1))


def test_reference_reads_the_spread_of_20_day_volatility_and_drawdowns():
    rng = np.random.default_rng(1)
    ref = reference_from_curve(_curve(rng.normal(0.0008, 0.015, 500)))
    assert ref["vol_p05"] < ref["vol_p50"] < ref["vol_p95"]
    assert 0.15 < ref["vol_p50"] < 0.35                      # ~1.5% daily = ~24% annualized
    assert ref["max_dd"] <= ref["dd_p05"] <= 0 and ref["n_days"] == 500


def test_too_little_live_data_is_said_rather_than_guessed():
    ev = live_vs_backtest(_curve([0.01, -0.005, 0.004, 0.002, -0.001, 0.003], flat_days=3), REF)
    assert ev["vol"] is None and ev["vol_status"] == "not enough data yet (6 of 20 trading days)"
    assert ev["days"] == 6                                     # pre-trading flat days are not live days
    assert ev["dd_status"] == "within backtest range" and not ev["flag"]


def test_volatility_outside_the_backtest_range_is_flagged():
    wild = live_vs_backtest(_curve([0.05, -0.05] * 15), REF)        # ~80% annualized
    calm = live_vs_backtest(_curve([0.001, 0.0012] * 15), REF)      # near-zero
    mid = live_vs_backtest(_curve([0.015, -0.012] * 15), REF)       # ~21%
    assert wild["vol_status"] == "ABOVE backtest range" and wild["flag"]
    assert calm["vol_status"] == "below backtest range" and not calm["flag"]
    assert mid["vol_status"] == "within backtest range"


def test_drawdown_is_judged_against_the_backtests_worst():
    worst = live_vs_backtest(_curve([0.01] * 5 + [-0.03] * 8), REF)     # ~-21.6% from the peak
    bad = live_vs_backtest(_curve([0.01] * 5 + [-0.03] * 6), REF)       # ~-16.7%
    assert worst["dd_status"] == "BEYOND the backtest's worst" and worst["flag"]
    assert bad["dd_status"] == "in the backtest's worst 5% of days" and not bad["flag"]
    assert worst["dd_max"] == pytest.approx(worst["dd_now"])


def test_no_reference_says_how_to_make_one():
    ev = live_vs_backtest(_curve([0.01] * 3), None)
    assert ev["vol_status"] == "no backtest reference yet" and ev["reference"] is None


def test_the_research_line_appears_only_when_backtests_were_run(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import report_daily
    from desk import trials as T
    (tmp_path / "state" / "history").mkdir(parents=True)
    (tmp_path / "README.md").write_text("no table")
    log = tmp_path / "state" / "history" / "trials.jsonl"
    rec = {"config_id": "a", "overrides": {}, "start": "s", "end": "e", "universe_id": "u", "total": 1.4,
           "maxdd": -0.19, "sharpe_annual": 1.3, "stats": {"sr_daily": 0.08, "n_obs": 750, "skew": 0, "kurt": 3}}
    T.record(log, rec)
    cfg = {"_root": str(tmp_path), "backtest": {"prior_untracked_trials": 0}}
    assert report_daily.research_line(cfg, None) == (None, 1)       # first email: record, say nothing
    assert report_daily.research_line(cfg, 1) == (None, 1)          # nothing new
    T.record(log, rec)
    line, n = report_daily.research_line(cfg, 1)
    assert n == 2 and line.startswith("1 new backtest run(s)") and "Deflated Sharpe" in line


def test_email_shows_the_comparison_and_warns_on_a_stale_reference():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_report import _ctx
    from desk.report import build_report
    from desk.report_html import build_html
    ev = {**live_vs_backtest(_curve([0.05, -0.05] * 15), REF), "stale_reference": True}
    ctx = _ctx(evidence=ev, research="2 new backtest run(s) since the last email")
    text, html = build_report(ctx), build_html(ctx)
    assert "LIVE VS BACKTEST" in text and "ABOVE backtest range" in text and "older config" in text
    assert text.index("LIVE VS BACKTEST") < text.index("RESULTS TODAY") and "research: 2 new" in text
    assert "Live vs backtest" in html and "ABOVE backtest range" in html and "#fdecea" in html
    assert "Research:" in html and "older config" in html


def test_the_reference_is_tied_to_the_config_that_produced_it():
    from backtest import current_config_id
    from desk.config import load_config
    cfg = load_config()
    same = current_config_id(cfg)
    cfg["portfolio"]["risk_per_trade_pct"] = 0.02
    assert current_config_id(cfg) != same

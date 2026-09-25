import numpy as np
import pandas as pd

import robustness as R


def test_full_stats_on_a_known_path():
    idx = pd.bdate_range("2020-01-01", periods=5)
    eq = pd.Series([100.0, 120.0, 90.0, 110.0, 130.0], index=idx)
    s = R.full_stats(eq)
    assert abs(s["total"] - 0.30) < 1e-12
    assert abs(s["maxdd"] - (90 / 120 - 1)) < 1e-12
    assert s["longest_underwater_days"] >= 1


def test_period_return_uses_the_last_close_on_or_before_each_date():
    idx = pd.to_datetime(["2020-03-02", "2020-03-04", "2020-03-06"])
    eq = pd.Series([100.0, 80.0, 90.0], index=idx)
    assert abs(R.period_return(eq, "2020-03-03", "2020-03-05") - (80 / 100 - 1)) < 1e-12   # weekend-proof
    assert R.period_return(eq, "2020-01-01", "2020-03-05") is None                     # before the data


def test_bootstrap_of_a_constant_path_is_that_path():
    r = np.full(500, 0.001)
    b = R.stationary_bootstrap(r, np.zeros(500), horizon=100, n_paths=50, block=10.0)
    assert abs(b["total"]["p5"] - (1.001 ** 100 - 1)) < 1e-9 and abs(b["total"]["p95"] - b["total"]["p5"]) < 1e-12
    assert b["p_loss"] == 0 and b["p_dd_15"] == 0 and b["p_beat_spy"] == 1.0


def test_trade_stats_measure_how_much_rides_on_a_few_trades():
    t = pd.DataFrame({"pnl": [1000.0] + [10.0] * 20 + [-5.0] * 10, "pnl_pct": [0.5] + [0.02] * 20 + [-0.03] * 10,
                      "realized_risk": [100.0] * 31, "reason": ["target"] * 21 + ["stop"] * 10,
                      "hold_days": [20] * 31, "exit_date": ["2021-06-01"] * 31})
    s = R.trade_stats(t)
    assert s["n"] == 31 and abs(s["win_rate"] - 21 / 31) < 1e-12
    assert abs(s["top10_share"] - 1090 / 1150) < 1e-12  # net 1150, of which the best ten made 1090
    assert s["net_without_top10"] == 60.0
    assert s["by_reason"]["stop"]["n"] == 10


def test_equal_weight_benchmark_uses_yesterdays_membership():
    idx = pd.bdate_range("2021-01-04", periods=4)
    close = pd.DataFrame({"A": [10, 11, 12.1, 13.31], "B": [10, 20, 40, 80]}, index=idx, dtype=float)
    members = pd.DataFrame({"A": [True] * 4, "B": [False, True, True, True]}, index=idx)
    ew = R.ew_universe(close, members)
    # B joins on day 2 and earns from day 3 on: its +100% on day 2 was not knowable when the list was made.
    assert abs(ew.iloc[1] / ew.iloc[0] - 1.10) < 1e-12
    assert abs(ew.iloc[2] / ew.iloc[1] - (0.10 + 1.0) / 2 - 1) < 1e-12


def test_the_job_list_covers_every_section_and_window():
    jobs = R.build_jobs(set(R.SECTIONS), "2018-01-02", "2026-09-11", "2023-09-11", seeds=5, veto_seeds=3)
    groups = pd.Series([j["group"] for j in jobs]).value_counts()
    assert groups["sensitivity"] == 4 * 2 * len(R.SENSITIVITY)          # every variant from 4 start dates
    assert groups["base"] == 4 and groups["null"] == 5 and groups["frictions"] == 3
    assert groups["windows"] == 23                      # quarterly starts 2018-01 .. 2023-07
    assert len({j["name"] for j in jobs}) == len(jobs)


def test_variants_are_judged_against_config_yaml_from_the_same_start():
    run = lambda cagr, dd: {"cagr": cagr, "sharpe": 1.0, "maxdd": dd, "total": cagr, "group": "design", "label": "x"}
    runs = {"base@0": run(0.10, -0.20), "base@1": run(0.02, -0.30), "base@2": run(0.05, -0.25),
            "v@0": run(0.08, -0.15), "v@1": run(0.04, -0.35), "v@2": run(0.06, -0.20)}
    v = R.variant_summary(runs)["v"]
    assert v["n"] == 3 and abs(v["cagr"] - 0.06) < 1e-12
    assert v["beats_base_cagr"] == 2 and v["beats_base_dd"] == 2 and v["paired"] == 3

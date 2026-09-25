"""Trial log and Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)."""
import math

import pandas as pd
import pytest

from desk import trials as T


def test_probabilistic_sharpe_matches_the_closed_form():
    # SR 0.1/day over 250 days, normal returns: Phi(0.1*sqrt(249)/sqrt(1 + (3-1)/4*0.01)) = Phi(1.5740) = 0.9423
    assert T.probabilistic_sharpe(0.1, 0.0, 250, 0.0, 3.0) == pytest.approx(0.9423, abs=5e-4)
    assert T.probabilistic_sharpe(0.0, 0.0, 250, 0.0, 3.0) == pytest.approx(0.5)


def test_fat_tails_and_negative_skew_lower_confidence():
    base = T.probabilistic_sharpe(0.1, 0.0, 250, 0.0, 3.0)
    assert T.probabilistic_sharpe(0.1, 0.0, 250, -1.0, 3.0) < base        # crash-prone, like momentum
    assert T.probabilistic_sharpe(0.1, 0.0, 250, 0.0, 9.0) < base         # fat tails


def test_more_trials_raise_the_bar():
    assert T.expected_max_sharpe(0.001, 1) == 0.0
    bars = [T.expected_max_sharpe(0.001, n) for n in (2, 10, 100)]
    assert bars[0] < bars[1] < bars[2]
    stats = {"sr_daily": 0.08, "n_obs": 750, "skew": -0.3, "kurt": 6.0}
    few = T.deflated_sharpe(stats, [0.05, 0.08, 0.06], 3)["dsr"]
    many = T.deflated_sharpe(stats, [0.05, 0.08, 0.06], 30)["dsr"]
    assert many < few < T.deflated_sharpe(stats, [0.08], 1)["dsr"]


def test_return_stats_reads_skew_and_ordinary_kurtosis():
    eq = pd.Series([100, 101, 100.5, 102, 101.8, 103, 102.9, 104])
    s = T.return_stats(eq)
    r = eq.pct_change().dropna()
    assert s["n_obs"] == 7 and s["sr_daily"] == pytest.approx(r.mean() / r.std())
    assert s["kurt"] == pytest.approx(r.kurt() + 3)                        # pandas' excess kurtosis + 3


def _t(cid, sr, start="2023-09-11", uid="u1"):
    return {"config_id": cid, "start": start, "end": "2026-09-11", "universe_id": uid,
            "stats": {"sr_daily": sr, "n_obs": 750, "skew": 0.0, "kurt": 3.0}}


def test_judge_counts_variants_not_reruns_and_compares_like_with_like():
    log = [_t("base", 0.07), _t("base", 0.07), _t("risk125", 0.09), _t("pos15", 0.06),
           _t("pos15", 0.05, start="2024-01-11"),                          # another window: counts toward N only
           _t("other", 0.2, uid="u0")]                                      # another universe: same
    v = T.judge(log, _t("base", 0.07), prior_untracked=14)
    assert v["n_trials"] == 4 + 14                                         # base, risk125, pos15, other
    assert v["trials_with_spread"] == 3                                     # same window + universe only
    assert 0 < v["dsr"] < v["psr_vs_zero"]


def test_variant_identity_ignores_key_order_and_the_log_round_trips(tmp_path):
    assert T.config_id({"a": 1, "b": {"c": 2}}) == T.config_id({"b": {"c": 2}, "a": 1})
    p = tmp_path / "trials.jsonl"
    T.record(p, _t("base", 0.07))
    p.write_text(p.read_text() + "not json\n")                             # a torn line is skipped, not fatal
    assert [t["config_id"] for t in T.load(p)] == ["base"]


def test_backtest_labels_a_variant_by_its_non_default_flags():
    from backtest import build_parser, overrides_of
    args = build_parser().parse_args(["--start", "2023-09-11", "--risk-per-trade", "0.0125", "--tag", "x"])
    assert overrides_of(args) == {"risk_per_trade": 0.0125}


def _rec(cid, sr, ts, overrides=None, start="2023-09-11"):
    return {**_t(cid, sr, start=start), "ts": ts, "overrides": overrides or {}, "total": sr * 20, "maxdd": -0.19,
            "sharpe_annual": sr * 16}


def test_summary_marks_the_current_config_and_supersedes_old_ones():
    log = [_rec("old", 0.07, "2026-09-20T00:00:00+00:00"),                 # config.yaml before an edit
           _rec("risk", 0.09, "2026-09-24T00:00:00+00:00", {"risk_per_trade": 0.0125}),
           _rec("new", 0.08, "2026-09-25T00:00:00+00:00"),                 # config.yaml now
           _rec("x", 0.05, "2026-09-25T00:00:00+00:00", {"ladder": True}, start="2024-01-11")]  # smaller window
    s = T.summary(log, prior_untracked=1)
    assert s["window"][1] == "2023-09-11" and s["n_trials"] == 5 and s["current"]["label"] == "config.yaml"
    labels = [r["label"] for r in s["rows"]]
    assert "risk-per-trade 0.0125" in labels and any("superseded" in l for l in labels) and len(labels) == 3
    md = T.markdown(s)
    assert "Generated from 4 logged runs" in md and "**current: config.yaml**" in md
    assert T.markdown(T.summary([], 0)) == "_No backtest trials logged yet._"


def test_readme_block_is_rewritten_in_place_and_staleness_is_measured(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{T.START_MARK}\nold table\n{T.END_MARK}\noutro\n")
    log = [_rec("a", 0.07, "2026-09-24T00:00:00+00:00")]
    T.write_readme(readme, T.markdown(T.summary(log, 0)))
    text = readme.read_text()
    assert text.startswith("intro\n") and text.endswith("outro\n") and "old table" not in text
    assert T.readme_lag(readme, log) == 0 and T.readme_lag(readme, log + log) == 1
    bare = tmp_path / "bare.md"
    bare.write_text("no markers here")
    with pytest.raises(ValueError):
        T.write_readme(bare, "x")

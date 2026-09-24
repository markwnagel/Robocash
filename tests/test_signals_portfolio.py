import numpy as np
import pandas as pd

from desk.portfolio import (select_entries, size_notional, size_position, size_risk_parity,
                            too_correlated)
from desk.signals import atr, compute_panels, compute_signals


def test_momentum_and_ranks(cfg, panel):
    snap = compute_signals(panel, cfg["signals"])
    close = panel["close"]
    lb, skip = cfg["signals"]["momentum_lookback_days"], cfg["signals"]["momentum_skip_days"]
    expected = close["S00"].iloc[-1 - skip] / close["S00"].iloc[-1 - lb] - 1
    assert abs(snap.loc["S00", "momentum"] - expected) < 1e-9
    eligible = snap[snap["eligible"]]
    assert len(eligible) > 0
    assert sorted(eligible["rank"].tolist()) == list(range(1, len(eligible) + 1))
    assert eligible["rank"].iloc[0] == 1
    assert not snap.loc["S29", "eligible"]  # downtrending name is below its MA


def test_atr_is_positive_and_bounded(panel):
    a = atr(panel["high"], panel["low"], panel["close"], 14).iloc[-1]
    assert (a > 0).all()
    assert (a / panel["close"].iloc[-1] < 0.1).all()


def test_correlated_duplicate_is_skipped(cfg, panel):
    panels = compute_panels(panel, cfg["signals"])
    snap = compute_signals(panel, cfg["signals"])
    strict = dict(cfg["portfolio"], max_correlated_peers=0)
    assert too_correlated("S01", ["S00"], panels["returns"], strict)
    assert not too_correlated("S02", ["S00"], panels["returns"], strict)
    assert not too_correlated("S01", ["S00"], panels["returns"], cfg["portfolio"])  # default allows one peer
    assert too_correlated("S01", ["S00", "S02"], panels["returns"], dict(cfg["portfolio"], max_correlated_peers=0))
    picks = select_entries(snap, panels["returns"], held=["S00"], slots=5, pcfg=strict)
    assert "S01" not in picks
    assert "S00" not in picks
    assert len(picks) == 5


def test_slots_cap(cfg, panel):
    panels = compute_panels(panel, cfg["signals"])
    snap = compute_signals(panel, cfg["signals"])
    assert select_entries(snap, panels["returns"], [], 0, cfg["portfolio"]) == []
    assert len(select_entries(snap, panels["returns"], [], 3, cfg["portfolio"])) == 3


def test_notional_sizing_caps_and_bracket_geometry(cfg):
    pcfg, ecfg = dict(cfg["portfolio"], sizing="notional"), cfg["exits"]
    sz = size_notional(equity=10000, cash_available=9500, price=20.0, atr=0.3, pcfg=pcfg, ecfg=ecfg)
    assert sz["notional"] <= 10000 * pcfg["max_position_pct"] + 1e-9
    assert sz["stop"] == round(20.0 - ecfg["stop_atr_multiple"] * 0.3, 2)
    risk = 20.0 - sz["stop"]
    assert abs((sz["target"] - 20.0) - ecfg["take_profit_r_multiple"] * risk) < 0.011
    assert size_notional(10000, 9500, price=2500.0, atr=30, pcfg=pcfg, ecfg=ecfg) is None  # can't buy 1 share
    assert size_notional(10000, 50, price=20.0, atr=0.3, pcfg=pcfg, ecfg=ecfg)["notional"] <= 50
    high_vol = size_notional(10000, 9500, 20.0, atr=1.2, pcfg=pcfg, ecfg=ecfg)
    low_vol = size_notional(10000, 9500, 20.0, atr=0.3, pcfg=pcfg, ecfg=ecfg)
    assert high_vol["notional"] < low_vol["notional"]


def test_risk_parity_equalizes_dollar_risk(cfg):
    pcfg, ecfg = cfg["portfolio"], cfg["exits"]
    equity, budget = 10000.0, 10000.0 * cfg["portfolio"]["risk_per_trade_pct"]
    # Two names, very different volatility: dollar risk should match, share counts should not.
    calm = size_risk_parity(equity, 9500, 50.0, atr=0.5, pcfg=pcfg, ecfg=ecfg)
    wild = size_risk_parity(equity, 9500, 50.0, atr=4.0, pcfg=pcfg, ecfg=ecfg)
    for sz in (calm, wild):
        assert sz["qty"] * (50.0 - sz["stop"]) <= budget + 1e-9
        assert sz["notional"] <= equity * pcfg["max_position_pct"] + 1e-9
    assert wild["qty"] < calm["qty"]
    # Wild is bounded by the risk budget; calm is bounded by the position cap instead.
    assert wild["qty"] * (50.0 - wild["stop"]) > budget * 0.8
    assert calm["notional"] == equity * pcfg["max_position_pct"]
    assert size_risk_parity(equity, 9500, 2500.0, atr=30, pcfg=pcfg, ecfg=ecfg) is None
    assert size_risk_parity(equity, 40, 50.0, atr=0.5, pcfg=pcfg, ecfg=ecfg) is None  # cash floor binds
    assert size_risk_parity(equity, 0, 50.0, atr=0.5, pcfg=pcfg, ecfg=ecfg) is None


def test_size_position_dispatches_on_config(cfg):
    ecfg = cfg["exits"]
    rp = size_position(10000, 9500, 50.0, 2.0, dict(cfg["portfolio"], sizing="risk_parity"), ecfg)
    nt = size_position(10000, 9500, 50.0, 2.0, dict(cfg["portfolio"], sizing="notional"), ecfg)
    assert rp == size_risk_parity(10000, 9500, 50.0, 2.0, cfg["portfolio"], ecfg)
    assert nt == size_notional(10000, 9500, 50.0, 2.0, dict(cfg["portfolio"], sizing="notional"), ecfg)
    assert rp["qty"] != nt["qty"]

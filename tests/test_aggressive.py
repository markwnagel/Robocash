"""Aggression levers: trailing stop, pyramiding, regime ladder / cash rule. All OFF by default in config."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
from alpaca.trading.enums import OrderClass

from backtest import run_backtest
from desk import state as state_mod
from desk.execution import Broker
from desk.portfolio import pyramid_due, size_risk_parity, trail_stop
from desk.regime import regime_params, regime_state_series
from tests.test_run_daily import FakeBroker, _patched_cfg, run, wired  # noqa: F401  (wired is a fixture)


class TrailBroker(FakeBroker):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.raised = []

    def raise_stop(self, symbol, new_stop):
        self.raised.append((symbol, new_stop))
        return new_stop      # the level now resting at the broker, as the real Broker returns

    def order_fill(self, order_id):
        return None, None


def test_trail_stop_only_rises():
    e = {"trail_atr_multiple": 2.0}
    assert trail_stop(10.0, 12.0, 1.0, e) == 10.0
    assert trail_stop(10.0, 14.0, 1.0, e) == 12.0
    assert trail_stop(12.0, 11.0, 1.0, e) == 12.0   # never lowers
    assert trail_stop(10.0, 14.0, 1.0, {"trail_atr_multiple": 0}) == 10.0
    assert trail_stop(None, 14.0, 1.0, e) is None


def test_no_target_when_take_profit_is_zero(cfg):
    ecfg = dict(cfg["exits"], take_profit_r_multiple=0)
    sz = size_risk_parity(10000, 9500, 68.93, 3.81, cfg["portfolio"], ecfg)
    assert sz["target"] is None and sz["stop"] < 68.93


def test_pyramid_due_needs_a_winner_and_headroom():
    pcfg = {"pyramid_max_adds": 2, "pyramid_trigger_r": 1.0}
    pos = {"entry_price": 40.0, "last_add_price": 40.0, "initial_risk_per_share": 5.0, "adds": 0}
    assert not pyramid_due(pos, 44.0, pcfg)
    assert pyramid_due(pos, 45.0, pcfg)
    assert not pyramid_due(dict(pos, adds=2), 60.0, pcfg)
    assert not pyramid_due(pos, 60.0, dict(pcfg, pyramid_max_adds=0))
    assert not pyramid_due(dict(pos, initial_risk_per_share=0), 60.0, pcfg)


def test_backtest_trailing_stop_replaces_target(cfg, panel):
    dates = panel["close"].index
    trail = dict(cfg, exits=dict(cfg["exits"], take_profit_r_multiple=0, trail_atr_multiple=2.0))
    curve, trades = run_backtest(panel, trail, dates[0], dates[-1], 10000.0, 10, 5)
    assert len(trades)
    assert not trades["reason"].isin(["target", "target-gap"]).any()
    assert (trades["final_stop"] > trades["entry_price"]).any()   # the stop ratcheted above entry somewhere
    assert trades["hold_days"].max() <= cfg["exits"]["max_hold_days"] + 3
    assert (curve["cash"] >= -1e-6).all()


def test_backtest_pyramids_winners_without_leverage(cfg, panel):
    dates = panel["close"].index
    c = dict(cfg, exits=dict(cfg["exits"], take_profit_r_multiple=0, trail_atr_multiple=2.0),
             portfolio=dict(cfg["portfolio"], pyramid_max_adds=2, pyramid_trigger_r=0.5, max_new_entries_per_day=None))
    curve, trades = run_backtest(panel, c, dates[0], dates[-1], 10000.0, 10, 5)
    assert (trades["adds"] > 0).any()
    assert (trades["adds"] <= 2).all()
    assert (curve["cash"] >= -1e-6).all()
    assert curve["n_positions"].max() <= cfg["portfolio"]["max_positions"]


def test_regime_ladder_states_and_params():
    idx = pd.bdate_range("2024-01-01", periods=300)
    spy = pd.Series(np.linspace(400, 500, 300), index=idx)
    vix = pd.Series(15.0, index=idx)
    base = {"spy_ma_days": 200, "vix_risk_off": 30, "risk_off_scale": 0.5}
    ladder = dict(base, ladder=True, strong_spy_above_ma_pct=0.03, strong_vix_below=20,
                  strong_target_positions=8, strong_max_position_pct=0.15, off_liquidate=True)
    assert regime_state_series(spy, vix, base).iloc[-1] == "neutral"        # no ladder -> never strong
    assert regime_state_series(spy, vix, ladder).iloc[-1] == "strong"       # 500 vs MA ~450: well above
    vix.iloc[-1] = 25
    assert regime_state_series(spy, vix, ladder).iloc[-1] == "neutral"      # calm-VIX condition fails
    spy.iloc[-1] = 380
    assert regime_state_series(spy, vix, ladder).iloc[-1] == "off"
    assert (regime_state_series(spy, vix, ladder).iloc[:199] == "neutral").all()  # no MA yet
    pcfg = {"target_positions": 12, "max_position_pct": 0.10}
    assert regime_params("strong", pcfg, ladder) == {"target_positions": 8, "max_position_pct": 0.15, "liquidate": False}
    assert regime_params("neutral", pcfg, ladder) == {"target_positions": 12, "max_position_pct": 0.10, "liquidate": False}
    assert regime_params("off", pcfg, ladder) == {"target_positions": 0, "max_position_pct": 0.10, "liquidate": True}
    assert regime_params("off", pcfg, base) == {"target_positions": 6, "max_position_pct": 0.10, "liquidate": False}


def test_backtest_off_liquidate_goes_to_cash(cfg, panel):
    dates = panel["close"].index
    state = pd.Series("neutral", index=dates, dtype=object)
    state.iloc[250:] = "off"
    c = dict(cfg, regime=dict(cfg["regime"], off_liquidate=True))
    curve, trades = run_backtest(panel, c, dates[0], dates[-1], 10000.0, 10, 5, regime_state=state)
    assert (trades["reason"] == "regime").any()
    assert (curve["n_positions"].iloc[251:] == 0).all()


def test_oto_order_when_no_target():
    class T:
        def __init__(self):
            self.reqs = []

        def submit_order(self, req):
            self.reqs.append(req)
            return type("O", (), {"id": "x"})()

    t = T()
    b = Broker(t)
    b.submit_bracket("ABC", 3, 10.05, 9.0, None)
    b.submit_bracket("ABC", 3, 10.05, 9.0, 13.0)
    assert t.reqs[0].order_class == OrderClass.OTO and t.reqs[0].take_profit is None
    assert t.reqs[1].order_class == OrderClass.BRACKET and float(t.reqs[1].take_profit.limit_price) == 13.0


def test_runner_trails_stop_and_submits_oto(wired, cfg, monkeypatch):
    holder, sent = wired
    _patched_cfg(monkeypatch, exits={"trail_atr_multiple": 2.0, "take_profit_r_multiple": 0})
    old = (date.today() - timedelta(days=3)).isoformat()
    live = {"S00": {"qty": 10, "avg_entry_price": 40.0, "current_price": 80.0, "market_value": 800.0, "unrealized_plpc": 1.0}}
    holder["broker"] = TrailBroker(cash=9000.0, positions=live)
    state_mod.save_state({**state_mod.load_state(), "positions": {
        "S00": {"entry_date": old, "entry_price": 40.0, "qty": 10, "stop": 35.0, "target": None, "filled": True, "thesis": "t",
                "high_close": 40.0, "adds": 0, "last_add_price": 40.0, "initial_risk_per_share": 5.0}}})
    run(["--no-analyst"])
    b = holder["broker"]
    assert b.raised and b.raised[0][0] == "S00" and b.raised[0][1] > 35.0
    st = state_mod.load_state()
    assert st["positions"]["S00"]["stop"] == b.raised[0][1]
    assert "stop raised" in sent[-1]
    assert b.brackets and all(t is None for *_, t in b.brackets)   # new entries carry no take-profit leg


def test_runner_pyramids_a_winner(wired, cfg, monkeypatch):
    holder, sent = wired
    _patched_cfg(monkeypatch, exits={"trail_atr_multiple": 2.0, "take_profit_r_multiple": 0},
                 portfolio={"pyramid_max_adds": 1, "pyramid_trigger_r": 1.0})
    old = (date.today() - timedelta(days=3)).isoformat()
    live = {"S00": {"qty": 5, "avg_entry_price": 40.0, "current_price": 80.0, "market_value": 400.0, "unrealized_plpc": 1.0}}
    holder["broker"] = TrailBroker(cash=9000.0, positions=live)
    state_mod.save_state({**state_mod.load_state(), "positions": {
        "S00": {"entry_date": old, "entry_price": 40.0, "qty": 5, "stop": 35.0, "target": None, "filled": True, "thesis": "t",
                "high_close": 40.0, "adds": 0, "last_add_price": 40.0, "initial_risk_per_share": 5.0}}})
    run(["--no-analyst"])
    b = holder["broker"]
    adds = [x for x in b.brackets if x[0] == "S00"]
    assert len(adds) == 1 and adds[0][4] is None
    st = state_mod.load_state()
    assert st["positions"]["S00"].get("add_order_id") == "oid-S00"
    px = float(st["positions"]["S00"]["last_price"])
    assert adds[0][1] * px + 5 * px <= 10000 * 0.10 + px   # position cap respected
    assert "Pyramid adds staged" in sent[-1]


def test_runner_regime_liquidate_exits_everything(wired, cfg, monkeypatch):
    holder, sent = wired
    _patched_cfg(monkeypatch, regime={"off_liquidate": True})
    holder["regime"] = {"scale": 0.5, "state": "off", "label": "risk-off", "spy": 380.0, "spy_ma": 450.0, "vix": 35.0}
    recent = (date.today() - timedelta(days=2)).isoformat()   # inside min_hold: the regime rule overrides it
    live = {"S02": {"qty": 5, "avg_entry_price": 80.0, "current_price": 82.0, "market_value": 410.0, "unrealized_plpc": 0.02}}
    holder["broker"] = TrailBroker(cash=9000.0, positions=live)
    state_mod.save_state({**state_mod.load_state(), "positions": {
        "S02": {"entry_date": recent, "entry_price": 80.0, "qty": 5, "stop": 70.0, "target": 110.0, "filled": True, "thesis": "t"}}})
    run(["--no-analyst"])
    b = holder["broker"]
    assert "S02" in b.closed
    assert not b.brackets                      # target positions 0: nothing new
    assert "FLAT" in sent[-1] and "regime:" in sent[-1]


class ProtBroker(TrailBroker):
    """Fills are already live; exposes what protection the runner re-places."""


def _held(sym, price, stop, target=None, days=3):
    from datetime import date, timedelta
    return {sym: {"entry_date": (date.today() - timedelta(days=days)).isoformat(), "entry_price": price,
                  "qty": 5, "stop": stop, "target": target, "filled": True, "thesis": "t"}}


def test_runner_replaces_missing_broker_stop_with_gtc(wired, cfg):
    """Bracket legs inherit the entry's DAY tif and die that evening; the next run must re-protect."""
    holder, sent = wired
    live = {"S02": {"qty": 5, "avg_entry_price": 80.0, "current_price": 90.0, "market_value": 450.0, "unrealized_plpc": 0.12}}
    b = ProtBroker(cash=9000.0, positions=live)
    holder["broker"] = b
    state_mod.save_state({**state_mod.load_state(), "positions": _held("S02", 80.0, 70.0, 110.0)})
    run(["--no-analyst"])
    assert ("S02", 5, 70.0, 110.0) in b.protected
    assert "protective stop re-placed at 70.00" in sent[-1]


def test_runner_leaves_an_existing_stop_alone(wired, cfg):
    holder, sent = wired
    live = {"S02": {"qty": 5, "avg_entry_price": 80.0, "current_price": 90.0, "market_value": 450.0, "unrealized_plpc": 0.12}}
    b = ProtBroker(cash=9000.0, positions=live)
    b.existing_stops["S02"] = type("O", (), {"stop_price": 70.0, "id": "x"})()
    holder["broker"] = b
    state_mod.save_state({**state_mod.load_state(), "positions": _held("S02", 80.0, 70.0, 110.0)})
    run(["--no-analyst"])
    assert not b.protected


def test_runner_closes_a_position_already_through_its_stop(wired, cfg):
    """No resting order and price below the stop: the backstop must exit, not re-place a stop under water."""
    holder, sent = wired
    live = {"S02": {"qty": 5, "avg_entry_price": 80.0, "current_price": 65.0, "market_value": 325.0, "unrealized_plpc": -0.19}}
    b = ProtBroker(cash=9000.0, positions=live)
    holder["broker"] = b
    state_mod.save_state({**state_mod.load_state(), "positions": _held("S02", 80.0, 70.0, 110.0)})
    run(["--no-analyst"])
    assert "S02" in b.closed and not b.protected
    assert "breached" in sent[-1]


def test_runner_flags_when_it_cannot_verify_protection(wired, cfg):
    holder, sent = wired
    live = {"S02": {"qty": 5, "avg_entry_price": 80.0, "current_price": 90.0, "market_value": 450.0, "unrealized_plpc": 0.12}}
    b = ProtBroker(cash=9000.0, positions=live)
    b.query_fails = True
    holder["broker"] = b
    state_mod.save_state({**state_mod.load_state(), "positions": _held("S02", 80.0, 70.0, 110.0)})
    run(["--no-analyst"])
    assert not b.protected and "S02" not in b.closed
    assert "UNVERIFIED" in sent[-1]


def test_vix_falls_back_when_the_primary_source_is_unreachable(tmp_path, monkeypatch):
    """FRED is unreachable on some networks; a silent None disables the volatility half of the gate."""
    import pandas as pd
    from desk import regime as reg
    idx = pd.bdate_range("2026-01-01", periods=5)
    monkeypatch.setattr(reg, "_vix_from_fred", lambda p: (_ for _ in ()).throw(TimeoutError("blocked")))
    monkeypatch.setattr(reg, "_vix_from_yahoo", lambda p: pd.Series([15.0, 16.0, 31.0, 18.0, 17.0], index=idx, name="vix"))
    s = reg.vix_series(tmp_path)
    assert s is not None and len(s) == 5 and s.iloc[2] == 31.0
    # Both sources down: None, and the caller still runs on the SPY trend alone.
    monkeypatch.setattr(reg, "_vix_from_yahoo", lambda p: (_ for _ in ()).throw(OSError("no network")))
    assert reg.vix_series(tmp_path / "empty") is None


def test_vix_cache_roundtrips(tmp_path, monkeypatch):
    import pandas as pd
    from desk import regime as reg
    idx = pd.bdate_range("2026-01-01", periods=3)
    monkeypatch.setattr(reg, "_vix_from_fred", lambda p: (_ for _ in ()).throw(TimeoutError("blocked")))
    monkeypatch.setattr(reg, "_vix_from_yahoo", reg._vix_from_yahoo.__wrapped__ if hasattr(reg._vix_from_yahoo, "__wrapped__") else
                        (lambda p: (pd.Series([15.0, 16.0, 17.0], index=idx, name="vix").rename_axis("date").to_frame().to_csv(p, header=["vix"]) or
                                    pd.Series([15.0, 16.0, 17.0], index=idx, name="vix"))))
    first = reg.vix_series(tmp_path)
    assert len(first) == 3
    monkeypatch.setattr(reg, "_vix_from_yahoo", lambda p: (_ for _ in ()).throw(AssertionError("should have used the cache")))
    again = reg.vix_series(tmp_path)
    assert len(again) == 3 and float(again.iloc[-1]) == 17.0

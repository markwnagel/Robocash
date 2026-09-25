"""End-to-end run of run_daily.main() against a fake broker: no network, no keys."""
import json
import sys
import time
from datetime import date, datetime, time as dtime, timedelta

import pandas as pd
import pytest

import run_daily
from desk import state as state_mod


class FakeBroker:
    def __init__(self, equity=10000.0, last_equity=10000.0, cash=10000.0, positions=None, dry_run=False):
        self.dry_run = dry_run
        self._acct = {"equity": equity, "last_equity": last_equity, "cash": cash,
                      "non_marginable_buying_power": cash, "daytrade_count": 0,
                      "pattern_day_trader": False, "trading_blocked": False}
        self._pos = positions or {}
        self.brackets, self.closed, self.liquidated = [], [], False
        self.statuses = {}
        self.protected = []          # (symbol, qty, stop, target) re-placed as GTC
        self.existing_stops = {}     # symbol -> object with .stop_price, or absent = unprotected
        self.working_closes = {}     # symbol -> a market sell still working at the broker
        self.query_fails = False
        self.session_open = False    # the scheduled run is after the close
        self.cancelled = []          # (symbol, side) leftover-order sweeps

    def live_exit_orders(self, symbol):
        if self.query_fails:
            return {"stop": None, "target": None, "close": None, "error": True}
        return {"stop": self.existing_stops.get(symbol), "target": None,
                "close": self.working_closes.get(symbol), "error": False}

    def market_open(self):
        return self.session_open

    def cancel_symbol_orders(self, symbol, side=None):
        self.cancelled.append((symbol, side))
        return 0

    def submit_exit_protection(self, symbol, qty, stop, target):
        self.protected.append((symbol, qty, stop, target))
        return f"prot-{symbol}"

    def order_status(self, order_id):
        return self.statuses.get(order_id)

    def account(self):
        return dict(self._acct)

    def positions(self):
        return dict(self._pos)

    def is_trading_day(self, day):
        return True

    def close(self, symbol):
        self.closed.append(symbol)

    def liquidate_all(self):
        self.liquidated = True

    def submit_bracket(self, symbol, qty, limit, stop, target):
        self.brackets.append((symbol, qty, limit, stop, target))
        return f"oid-{symbol}"

    def last_sell_fill(self, symbol, since):
        return None


@pytest.fixture
def wired(monkeypatch, tmp_path, panel):
    """Redirect state files to tmp and stub every network touchpoint."""
    monkeypatch.setattr(state_mod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(state_mod, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(state_mod, "TRADES_FILE", tmp_path / "trades.csv")
    monkeypatch.setattr(state_mod, "BRIEFS_DIR", tmp_path / "briefs")
    monkeypatch.setattr(run_daily, "alpaca_keys", lambda: ("k", "s"))
    monkeypatch.setattr(run_daily, "TradingClient", lambda *a, **k: object())
    monkeypatch.setattr(run_daily, "StockHistoricalDataClient", lambda *a, **k: object())
    monkeypatch.setattr(run_daily, "build_universe", lambda *a, **k: list(panel["close"].columns))
    spy_panel = {"close": pd.DataFrame({"SPY": panel["close"]["S00"]})}
    monkeypatch.setattr(run_daily, "cached_bars", lambda data, symbols, *a, **k: spy_panel if list(symbols) == ["SPY"] else panel)
    monkeypatch.setattr(run_daily, "vix_series", lambda cache_dir: None)
    monkeypatch.setattr(run_daily, "current_regime", lambda spy, vix, rcfg: holder.get("regime", {
        "scale": 1.0, "label": "risk-on", "spy": 500.0, "spy_ma": 480.0, "vix": 15.0}))
    monkeypatch.setattr(run_daily, "red_team", lambda payload, cfg_: holder.get("red_team", {}))
    holder = {"facts": {}}

    class FakeFB:
        def __init__(self, *a, **k):
            pass

        def build(self, symbols, closes):
            holder["facts_requested"] = list(symbols)
            return {s: holder["facts"].get(s, {"symbol": s}) for s in symbols}

    monkeypatch.setattr(run_daily, "FactBuilder", FakeFB)
    sent = []
    monkeypatch.setattr(run_daily, "send", lambda text, cfg: sent.append(text))
    monkeypatch.setattr(run_daily, "Broker", lambda trading, dry_run=False: holder["broker"])
    # The scheduled run: 16:45 ET on the fixture's last bar date, after the close. Without this the suite's
    # outcome would depend on the time of day it runs (16:00-16:20 ET is protect-only).
    holder["clock"] = datetime.combine(date.today(), dtime(16, 45), tzinfo=run_daily.ET)
    monkeypatch.setattr(run_daily, "now_et", lambda: holder["clock"])
    return holder, sent


def run(argv):
    sys.argv = ["run_daily.py"] + argv
    run_daily.main()


def test_fresh_account_stages_entries(wired, cfg):
    holder, sent = wired
    holder["broker"] = FakeBroker()
    holder["analyst"] = None
    run(["--no-analyst"])
    b = holder["broker"]
    assert 1 <= len(b.brackets) <= cfg["portfolio"]["target_positions"]
    total = sum(q * lim for _, q, lim, _, _ in b.brackets)
    assert total <= 10000 * (1 - cfg["portfolio"]["cash_floor_pct"]) + 1e-6
    for sym, qty, lim, stop, target in b.brackets:
        assert stop < lim < target and qty >= 1
    st = json.loads((state_mod.STATE_FILE).read_text())
    assert len(st["positions"]) == len(b.brackets)
    assert "Entries staged" in sent[-1]


def test_kill_switch_liquidates_and_halts(wired, cfg):
    holder, sent = wired
    holder["broker"] = FakeBroker(equity=8000.0, last_equity=8100.0, cash=1000.0,
                                  positions={"S00": {"qty": 10, "avg_entry_price": 60.0, "current_price": 55.0,
                                                     "market_value": 550.0, "unrealized_plpc": -0.08}})
    state_mod.save_state({**state_mod.load_state(), "high_water_mark": 10000.0})
    run(["--no-analyst"])
    b = holder["broker"]
    assert b.liquidated and not b.brackets
    st = state_mod.load_state()
    assert st["halted"] and "drawdown" in st["halt_reason"]
    assert "KILL SWITCH" in sent[-1]
    # A second run while halted must not trade, even with a recovered account.
    holder["broker"] = FakeBroker(equity=12000.0, last_equity=11000.0)
    run(["--no-analyst"])
    assert not holder["broker"].brackets and "HALTED" in sent[-1]
    # --resume clears it and trading restarts.
    holder["broker"] = FakeBroker(equity=12000.0, last_equity=11000.0, cash=12000.0)
    run(["--no-analyst", "--resume"])
    assert holder["broker"].brackets


def test_hard_floor_needs_explicit_reset(wired, cfg):
    holder, sent = wired
    # Episode HWM already re-baselined once; equity is 26% below the all-time high.
    state_mod.save_state({**state_mod.load_state(), "high_water_mark": 10000.0, "all_time_hwm": 14000.0})
    holder["broker"] = FakeBroker(equity=10300.0, last_equity=10200.0, cash=2000.0,
                                  positions={"S00": {"qty": 10, "avg_entry_price": 60.0, "current_price": 55.0,
                                                     "market_value": 550.0, "unrealized_plpc": -0.08}})
    run(["--no-analyst"])
    assert holder["broker"].liquidated and "HARD FLOOR" in sent[-1] and "--reset-hwm" in sent[-1]
    st = state_mod.load_state()
    assert st["halted"] and st["hard_halted"]
    # Plain --resume is refused and trades nothing.
    holder["broker"] = FakeBroker(equity=12000.0, last_equity=11000.0, cash=12000.0)
    run(["--no-analyst", "--resume"])
    assert not holder["broker"].brackets and "not cleared by --resume alone" in sent[-1]
    assert state_mod.load_state()["halted"]
    # Explicit acknowledgement clears it and re-baselines the all-time high.
    holder["broker"] = FakeBroker(equity=12000.0, last_equity=11000.0, cash=12000.0)
    run(["--no-analyst", "--resume", "--reset-hwm"])
    assert holder["broker"].brackets
    st = state_mod.load_state()
    assert not st["halted"] and not st["hard_halted"] and st["all_time_hwm"] == 12000.0


def test_entry_pacing_caps_new_positions_per_day(wired, cfg):
    holder, sent = wired
    holder["broker"] = FakeBroker()
    run(["--no-analyst"])
    assert len(holder["broker"].brackets) == cfg["portfolio"]["max_new_entries_per_day"]
    assert "entry pacing" in sent[-1]


def test_time_exit_and_analyst_veto(wired, cfg, monkeypatch):
    holder, sent = wired
    old = (date.today() - timedelta(days=cfg["exits"]["max_hold_days"] + 1)).isoformat()
    recent = (date.today() - timedelta(days=3)).isoformat()
    live = {
        "S02": {"qty": 5, "avg_entry_price": 80.0, "current_price": 90.0, "market_value": 450.0, "unrealized_plpc": 0.1},
        "S03": {"qty": 5, "avg_entry_price": 80.0, "current_price": 90.0, "market_value": 450.0, "unrealized_plpc": 0.1},
    }
    holder["broker"] = FakeBroker(cash=9000.0, positions=live)
    state_mod.save_state({**state_mod.load_state(), "positions": {
        "S02": {"entry_date": old, "entry_price": 80.0, "qty": 5, "stop": 70.0, "target": 110.0, "filled": True, "thesis": "t"},
        "S03": {"entry_date": recent, "entry_price": 80.0, "qty": 5, "stop": 70.0, "target": 110.0, "filled": True, "thesis": "t"},
    }})
    captured = {}

    def fake_review(payload, cfg_):
        captured["payload"] = payload
        first_candidate = payload["candidates"][0]["symbol"]
        return {"data_concerns": [], "brief": "analyst brief here",
                "verdicts": [{"symbol": "S03", "action": "REMOVE", "reason": "guidance cut", "confidence": 0.9},
                             {"symbol": first_candidate, "action": "REMOVE", "reason": "offering", "confidence": 0.9}]}

    monkeypatch.setattr(run_daily, "review", fake_review)
    run([])
    b = holder["broker"]
    assert "S02" in b.closed  # past max hold
    assert "S03" in b.closed  # analyst veto on a holding
    vetoed = captured["payload"]["candidates"][0]["symbol"]
    assert vetoed not in [s for s, *_ in b.brackets]
    assert "analyst brief here" in sent[-1]
    assert captured["payload"]["holdings"][0]["exit_planned"] is not None


def test_analyst_failure_blocks_new_entries_only(wired, cfg, monkeypatch):
    holder, sent = wired
    holder["broker"] = FakeBroker()
    monkeypatch.setattr(run_daily, "review", lambda payload, cfg_: None)
    run([])
    assert not holder["broker"].brackets
    assert "fail-safe" in sent[-1]


def test_dry_run_touches_nothing(wired, cfg):
    holder, sent = wired
    holder["broker"] = FakeBroker(dry_run=True)
    run(["--no-analyst", "--dry-run"])
    assert holder["broker"].brackets  # staged (fake broker records them; real one logs only)
    assert not state_mod.STATE_FILE.exists()


def test_facts_block_and_resize_candidates(wired, cfg, panel):
    holder, sent = wired
    from desk.signals import compute_signals
    ranked = compute_signals(panel, cfg["signals"])
    top, second, third = list(ranked.index[:3])
    holder["facts"] = {
        top: {"symbol": top, "earnings_in_days": 3},
        second: {"symbol": second, "offering_filings": ["S-3 2026-09-01"]},
        third: {"symbol": third, "iv_over_rv": 2.5},
    }
    holder["broker"] = FakeBroker()
    run(["--no-analyst"])
    b = holder["broker"]
    bought = {s: q for s, q, *_ in b.brackets}
    assert top not in bought and second not in bought
    assert third in bought
    others = [q * lim for s, q, lim, *_ in b.brackets if s != third]
    third_notional = bought[third] * dict((s, lim) for s, q, lim, *_ in b.brackets)[third]
    assert third_notional < 0.7 * max(others)
    assert f"blocked {top}: earnings in 3d" in sent[-1]
    assert set(holder["facts_requested"]) >= {top, second, third}


def test_half_size_never_drops_a_one_share_position(wired, cfg, panel):
    """A high-IV name that can only afford 1 share must still be bought, not silently excluded."""
    holder, sent = wired
    from desk.signals import compute_signals
    ranked = compute_signals(panel, cfg["signals"])
    top = ranked.index[0]
    price = float(ranked.loc[top, "close"])
    holder["facts"] = {top: {"symbol": top, "iv_over_rv": 3.0}}
    # Equity where one share is all the budget allows, so the multiplier would round qty to 0.
    holder["broker"] = FakeBroker(equity=price * 12, last_equity=price * 12, cash=price * 12)
    run(["--no-analyst"])
    bought = {s: q for s, q, *_ in holder["broker"].brackets}
    assert bought.get(top) == 1


def test_risk_off_regime_halves_target(wired, cfg):
    holder, sent = wired
    holder["regime"] = {"scale": 0.5, "label": "risk-off", "spy": 400.0, "spy_ma": 450.0, "vix": 35.0}
    holder["broker"] = FakeBroker()
    run(["--no-analyst"])
    assert len(holder["broker"].brackets) <= round(cfg["portfolio"]["target_positions"] * 0.5)
    assert "regime risk-off" in sent[-1]


def _patched_cfg(monkeypatch, **overrides):
    """Return a load_config replacement with nested overrides like {'facts': {'candidate_pool': 3}}."""
    from desk.config import load_config as real
    base = real()
    for section, kv in overrides.items():
        base[section].update(kv)
    monkeypatch.setattr(run_daily, "load_config", lambda: base)
    return base


def test_only_fact_checked_names_are_bought(wired, cfg, monkeypatch):
    holder, sent = wired
    _patched_cfg(monkeypatch, facts={"candidate_pool": 3})
    holder["broker"] = FakeBroker()
    run(["--no-analyst"])
    bought = [s for s, *_ in holder["broker"].brackets]
    assert 1 <= len(bought) <= 3
    assert set(bought) <= set(holder["facts_requested"])


def test_unreviewed_entries_are_dropped(wired, cfg, monkeypatch):
    holder, sent = wired
    _patched_cfg(monkeypatch, analyst={"candidates_to_review": 2, "red_team_enabled": False})
    monkeypatch.setattr(run_daily, "review", lambda payload, cfg_: {"data_concerns": [], "brief": "", "verdicts": []})
    holder["broker"] = FakeBroker()
    run([])
    assert len(holder["broker"].brackets) <= 2
    assert "beyond analyst review capacity dropped" in sent[-1]


def test_expired_entry_order_frees_its_slot(wired, cfg):
    holder, sent = wired
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    state_mod.save_state({**state_mod.load_state(), "positions": {
        "S05": {"entry_date": yesterday, "entry_price": 50.0, "qty": 4, "stop": 45.0, "target": 65.0,
                "filled": False, "order_id": "o-expired", "thesis": "t"},
        "S06": {"entry_date": yesterday, "entry_price": 50.0, "qty": 4, "stop": 45.0, "target": 65.0,
                "filled": False, "order_id": "o-open", "thesis": "t"},
    }})
    holder["broker"] = FakeBroker()
    holder["broker"].statuses = {"o-expired": "expired", "o-open": "new"}
    run(["--no-analyst"])
    st = state_mod.load_state()
    assert "S05" not in st["positions"]          # expired -> dropped the same evening
    assert "S06" in st["positions"]              # still working -> kept, slot still reserved
    assert "S05 entry order never filled (expired)" in sent[-1]
    assert "S06" not in [s for s, *_ in holder["broker"].brackets]


def test_stale_bars_suppress_entries(wired, cfg, panel, monkeypatch):
    holder, sent = wired
    stale = {k: v.iloc[:-5] for k, v in panel.items()}
    monkeypatch.setattr(run_daily, "cached_bars",
                        lambda data, symbols, *a, **k: {"close": pd.DataFrame({"SPY": stale["close"]["S00"]})}
                        if list(symbols) == ["SPY"] else stale)
    holder["broker"] = FakeBroker()
    run(["--no-analyst"])
    assert holder["broker"].brackets == []
    assert "STALE, entries suppressed" in sent[-1]
    holder["broker"] = FakeBroker()
    run(["--no-analyst", "--force"])          # explicit override for weekend/holiday test runs
    assert holder["broker"].brackets


def test_red_team_feeds_analyst(wired, cfg, monkeypatch):
    holder, sent = wired
    holder["broker"] = FakeBroker()
    seen = {}

    def fake_review(payload, cfg_):
        seen["payload"] = payload
        return {"data_concerns": [], "brief": "ok", "verdicts": []}

    monkeypatch.setattr(run_daily, "review", fake_review)
    first = None
    original_red_team = run_daily.red_team

    def fake_red_team(payload, cfg_):
        nonlocal first
        first = payload["candidates"][0]["symbol"]
        return {first: {"symbol": first, "bear_case": "crowded", "trap_type": "crowded", "severity": 0.8}}

    monkeypatch.setattr(run_daily, "red_team", fake_red_team)
    run([])
    assert seen["payload"]["red_team"][first]["severity"] == 0.8
    assert seen["payload"]["candidates"][0]["facts"] is not None
    assert "red team flags" in sent[-1]


class _Order:
    """Minimal stand-in for an Alpaca order object as live_exit_orders inspects it."""
    def __init__(self, stop_price=None, limit_price=None, symbol="AAA", oid="o1"):
        from alpaca.trading.enums import OrderSide
        self.stop_price, self.limit_price, self.symbol, self.id = stop_price, limit_price, symbol, oid
        self.side = OrderSide.SELL
        self.status = type("S", (), {"value": "held"})()


class _FakeTrading:
    def __init__(self, orders):
        self.orders, self.replaced = orders, []

    def get_orders(self, req):
        return self.orders

    def replace_order_by_id(self, oid, req):
        self.replaced.append((oid, req.stop_price))


def test_raise_stop_reports_the_level_actually_resting_at_the_broker():
    from desk.execution import Broker
    t = _FakeTrading([_Order(stop_price=20.0)])
    assert Broker(t).raise_stop("AAA", 22.0) == 22.0 and t.replaced == [("o1", 22.0)]
    # Broker already higher than the level we computed: its own stop comes back and nothing is replaced,
    # so state can never record a stop below the one really protecting the position.
    t2 = _FakeTrading([_Order(stop_price=25.0)])
    assert Broker(t2).raise_stop("AAA", 22.0) == 25.0 and t2.replaced == []
    # Nothing protecting it at all is distinguishable from both of the above.
    assert Broker(_FakeTrading([])).raise_stop("AAA", 22.0) is None


def test_a_close_that_did_not_complete_is_retried_rather_than_stranded():
    """While pending_exit_reason is set the position is skipped by the exit, trailing-stop AND
    stop-re-placement loops. If the close never filled, leaving it set holds the name forever unprotected."""
    from desk.execution import sync_positions
    st = {"positions": {"AAA": {"entry_date": "2026-09-01", "entry_price": 10.0, "qty": 5,
                                "filled": True, "stop": 9.0, "pending_exit_reason": "signal decay"}}}
    live = {"AAA": {"avg_entry_price": 10.0, "qty": 5.0, "current_price": 10.5,
                    "market_value": 52.5, "unrealized_plpc": 0.05}}
    notes = sync_positions(st, FakeBroker(), live, date(2026, 9, 17))
    assert "pending_exit_reason" not in st["positions"]["AAA"]
    assert any("did not close" in n and "AAA" in n for n in notes)


def test_bar_cache_distinguishes_same_sized_symbol_sets(tmp_path, monkeypatch):
    """Keyed on the count alone, a second run whose universe swapped one name for another silently got
    the first run's panel - and a held name missing from the snapshot is read as signal decay and sold."""
    from desk import data as data_mod
    cfg = {"_root": str(tmp_path), "data": {"cache_dir": "cache", "feed": "sip"}}
    calls = []

    def fake_fetch(d, symbols, days, feed, end=None):
        calls.append(tuple(symbols))
        return {"close": pd.DataFrame({s: [1.0] for s in symbols})}

    monkeypatch.setattr(data_mod, "fetch_daily_bars", fake_fetch)
    data_mod.cached_bars(None, ["AAA", "BBB"], 30, cfg)
    b = data_mod.cached_bars(None, ["AAA", "CCC"], 30, cfg)     # same length, different members
    assert len(calls) == 2, "second set was wrongly served from the first set's cache"
    assert list(b["close"].columns) == ["AAA", "CCC"]
    data_mod.cached_bars(None, ["AAA", "BBB"], 30, cfg)
    assert len(calls) == 2, "the cache stopped caching"


def test_reduce_on_a_holding_reaches_the_brief_and_never_sells(wired, cfg, monkeypatch):
    """The analyst is invited to REDUCE holdings but the code can only shrink staged entries. The verdict
    must therefore be reported to the owner rather than silently discarded."""
    holder, sent = wired
    _patched_cfg(monkeypatch, analyst={"red_team_enabled": False})
    recent = (date.today() - timedelta(days=3)).isoformat()
    live = {"S03": {"qty": 5, "avg_entry_price": 80.0, "current_price": 90.0,
                    "market_value": 450.0, "unrealized_plpc": 0.1}}
    holder["broker"] = FakeBroker(cash=9000.0, positions=live)
    state_mod.save_state({**state_mod.load_state(), "positions": {
        "S03": {"entry_date": recent, "entry_price": 80.0, "qty": 5, "stop": 70.0,
                "target": 110.0, "filled": True, "thesis": "t"},
    }})
    monkeypatch.setattr(run_daily, "review", lambda payload, cfg_: {
        "data_concerns": [], "brief": "b",
        "verdicts": [{"symbol": "S03", "action": "REDUCE", "reason": "insider selling", "confidence": 0.5}]})
    run([])
    assert "S03" not in holder["broker"].closed        # a REDUCE is never escalated into a sale
    assert "flagged REDUCE" in sent[-1] and "insider selling" in sent[-1]


def test_bar_cache_prunes_only_stale_panels(tmp_path):
    """One universe panel per run, never cleaned, reached 243MB in a fortnight. Pruning must not touch
    the fact sheets, earnings calendar or VIX history that share the directory."""
    import os
    from desk.data import prune_bar_cache
    old = time.time() - 10 * 86400
    for name in ("bars_daily_2026-09-01_300_800_abc.pkl", "bars_bt_2026-09-01_1397_800_def.pkl"):
        p = tmp_path / name
        p.write_bytes(b"x")
        os.utime(p, (old, old))
    for name in ("facts_2026-09-01.json", "earnings_2026-09-01.json", "vixcls.csv", "preview_2026-09-01.pkl"):
        p = tmp_path / name
        p.write_bytes(b"x")
        os.utime(p, (old, old))
    fresh = tmp_path / "bars_daily_2026-09-17_300_800_ghi.pkl"
    fresh.write_bytes(b"x")

    assert prune_bar_cache(tmp_path, keep_days=2) == 2
    assert fresh.exists()                                   # today's panel survives
    survivors = {p.name for p in tmp_path.iterdir()}
    assert survivors == {"facts_2026-09-01.json", "earnings_2026-09-01.json", "vixcls.csv",
                         "preview_2026-09-01.pkl", fresh.name}
    assert prune_bar_cache(tmp_path, keep_days=0) == 0      # 0 disables it
    assert prune_bar_cache(tmp_path / "does-not-exist", keep_days=2) == 0   # never raises


def test_fetching_a_panel_prunes_the_old_ones(tmp_path, monkeypatch):
    import os
    from desk import data as data_mod
    cache = tmp_path / "cache"
    cache.mkdir()
    stale = cache / "bars_daily_2026-09-01_300_2_old.pkl"
    stale.write_bytes(b"x")
    os.utime(stale, (time.time() - 10 * 86400,) * 2)
    monkeypatch.setattr(data_mod, "fetch_daily_bars",
                        lambda d, symbols, days, feed, end=None: {"close": pd.DataFrame({s: [1.0] for s in symbols})})
    data_mod.cached_bars(None, ["AAA", "BBB"], 30,
                         {"_root": str(tmp_path), "data": {"cache_dir": "cache", "feed": "sip"}})
    assert not stale.exists()
    assert len(list(cache.glob("bars_*.pkl"))) == 1


def test_run_daily_applies_the_configured_reduce_factor(wired, cfg, monkeypatch):
    holder, sent = wired
    _patched_cfg(monkeypatch, analyst={"red_team_enabled": False, "reduce_factor": 0.75})
    holder["broker"] = FakeBroker()
    seen = {}

    def fake_review(payload, cfg_):
        c = max(payload["candidates"], key=lambda e: e["qty"])        # biggest lot, so 0.75 is visible
        seen["symbol"], seen["qty"] = c["symbol"], c["qty"]
        return {"data_concerns": [], "brief": "",
                "verdicts": [{"symbol": c["symbol"], "action": "REDUCE", "reason": "r", "confidence": 0.5}]}

    monkeypatch.setattr(run_daily, "review", fake_review)
    run([])
    bought = {s: q for s, q, *_ in holder["broker"].brackets}
    assert seen["qty"] >= 2
    assert bought[seen["symbol"]] == max(1, int(seen["qty"] * 0.75))


def test_halted_runs_keep_closing_positions_that_survived_the_kill_switch(wired, cfg):
    """liquidate_all cancels every stop first. A position whose close did not fill used to sit open and
    unprotected for as long as the halt lasted, because later halted runs returned before doing anything."""
    holder, sent = wired
    live = {"S05": {"qty": 5, "avg_entry_price": 80.0, "current_price": 70.0, "market_value": 350.0,
                    "unrealized_plpc": -0.12}}
    holder["broker"] = FakeBroker(positions=live)
    state_mod.save_state({**state_mod.load_state(), "halted": True, "halt_reason": "drawdown 16%",
                          "halt_date": "2026-09-20", "high_water_mark": 10000.0, "positions": {
                              "S05": {"entry_date": "2026-09-01", "entry_price": 80.0, "qty": 5, "stop": 70.0,
                                      "target": 110.0, "filled": True, "thesis": "t"}}})
    run(["--no-analyst"])
    assert holder["broker"].closed == ["S05"]
    assert "survived the liquidation" in sent[-1] and "S05" in sent[-1]
    assert state_mod.load_state()["positions"]["S05"]["pending_exit_reason"].startswith("KILL SWITCH (retry)")


def test_first_kill_switch_records_the_halt_even_if_liquidation_throws(wired, cfg):
    holder, sent = wired

    class Boom(FakeBroker):
        def liquidate_all(self):
            raise RuntimeError("broker 503")

    holder["broker"] = Boom(equity=8000.0, last_equity=8100.0, cash=8000.0)
    state_mod.save_state({**state_mod.load_state(), "high_water_mark": 10000.0, "all_time_hwm": 10000.0})
    run(["--no-analyst"])
    assert state_mod.load_state()["halted"]                # so the next run retries the survivors
    assert "LIQUIDATION FAILED" in sent[-1] and "broker 503" in sent[-1]


def test_a_holding_with_no_bar_today_is_not_sold_as_signal_decay(wired, cfg, panel):
    """A trading halt or a feed gap drops the name from the snapshot; that is missing data, not a ranking."""
    holder, sent = wired
    panel["close"].iloc[-1, panel["close"].columns.get_loc("S03")] = float("nan")
    held = (date.today() - timedelta(days=20)).isoformat()             # past min_hold, inside max_hold
    live = {"S03": {"qty": 5, "avg_entry_price": 80.0, "current_price": 90.0, "market_value": 450.0,
                    "unrealized_plpc": 0.1}}
    holder["broker"] = FakeBroker(cash=9000.0, positions=live)
    state_mod.save_state({**state_mod.load_state(), "positions": {
        "S03": {"entry_date": held, "entry_price": 80.0, "qty": 5, "stop": 70.0, "target": 110.0,
                "filled": True, "thesis": "t"}}})
    run(["--no-analyst"])
    assert "S03" not in holder["broker"].closed
    assert "S03: no complete bar today - signal-decay check deferred" in sent[-1]


def test_a_cap_breach_is_raised_every_run_and_leads_the_brief(wired, cfg):
    """UMC drifted to 10.6-10.9% of equity and stayed there; the only trace was one analyst sentence."""
    holder, sent = wired
    recent = (date.today() - timedelta(days=3)).isoformat()
    live = {"S03": {"qty": 45, "avg_entry_price": 20.0, "current_price": 24.0, "market_value": 1080.0,
                    "unrealized_plpc": 0.2}}
    holder["broker"] = FakeBroker(equity=10000.0, cash=8920.0, positions=live)
    state_mod.save_state({**state_mod.load_state(), "positions": {
        "S03": {"entry_date": recent, "entry_price": 20.0, "qty": 45, "stop": 18.0, "target": 27.0,
                "filled": True, "thesis": "t"}}})
    run(["--no-analyst"])
    brief = sent[-1]
    assert "MANUAL ACTION NEEDED: 1 item(s)" in brief
    notes = brief.split("Notes:", 1)[1].splitlines()
    assert notes[1].startswith("  - MANUAL ACTION: S03 is 10.8% of equity, above the 10% cap - sell 4 share(s)")
    assert "S03" not in holder["broker"].closed                      # reported, never auto-trimmed

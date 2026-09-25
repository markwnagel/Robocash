"""Engineering review 2026-09-24: risk-path resilience, idempotency, monitoring, consistency and the
hot-loop rewrites. Every test here failed (or could not run) against the code before the review."""
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import run_daily
from desk import monitor as M
from desk import state as state_mod
from desk import telemetry as T
from tests.test_run_daily import FakeBroker, _patched_cfg, run, wired  # noqa: F401  (wired is a fixture)


def _held(sym, days_ago, stop=40.0, target=70.0, qty=5, price=50.0):
    return {sym: {"entry_date": (date.today() - timedelta(days=days_ago)).isoformat(), "entry_price": price,
                  "qty": qty, "stop": stop, "target": target, "filled": True, "thesis": "t"}}


def _live(sym, qty=5, price=55.0):
    return {sym: {"qty": qty, "avg_entry_price": 50.0, "current_price": price, "market_value": qty * price,
                  "unrealized_plpc": 0.1}}


# ------------------------------------------------------------------ run_daily: one failure never strands the rest

def test_one_rejected_protective_stop_does_not_leave_the_others_bare(wired):
    holder, sent = wired

    class Picky(FakeBroker):
        def submit_exit_protection(self, symbol, qty, stop, target):
            if symbol == "S02":
                raise RuntimeError("403 insufficient qty available")
            return super().submit_exit_protection(symbol, qty, stop, target)

    holder["broker"] = Picky(cash=9000.0, positions={**_live("S02"), **_live("S03")})
    state_mod.save_state({**state_mod.load_state(), "positions": {**_held("S02", 3), **_held("S03", 3)}})
    run(["--no-analyst"])
    b = holder["broker"]
    assert [p[0] for p in b.protected] == ["S03"]                     # the next holding is still protected
    assert b.brackets                                                  # and the day's entries still go in
    assert "MANUAL ACTION: S02 has NO protective stop" in sent[-1]
    assert state_mod.load_state()["last_run"] == date.today().isoformat()


def test_a_rejected_entry_keeps_the_ones_already_placed_in_state(wired):
    holder, sent = wired

    class Flaky(FakeBroker):
        def submit_bracket(self, symbol, qty, limit, stop, target):
            if len(self.brackets) == 1 and not getattr(self, "_failed", False):
                self._failed = True
                raise RuntimeError("422 limit price too far")
            return super().submit_bracket(symbol, qty, limit, stop, target)

    holder["broker"] = Flaky()
    run(["--no-analyst"])
    placed = [s for s, *_ in holder["broker"].brackets]
    assert len(placed) >= 2
    assert set(state_mod.load_state()["positions"]) == set(placed)    # every order at the broker is in state
    assert "ORDER FAILED: BUY" in sent[-1]


def test_a_rejected_close_restores_the_stop_it_cancelled(wired, cfg):
    holder, sent = wired

    class NoClose(FakeBroker):
        def close(self, symbol):
            raise RuntimeError("403 position is locked")

    holder["broker"] = NoClose(cash=9000.0, positions=_live("S04"))
    state_mod.save_state({**state_mod.load_state(), "positions": _held("S04", cfg["exits"]["max_hold_days"] + 1)})
    run(["--no-analyst"])
    assert [p[0] for p in holder["broker"].protected] == ["S04"]
    assert "pending_exit_reason" not in state_mod.load_state()["positions"]["S04"]   # retried next run
    assert "ORDER FAILED: S04 exit" in sent[-1]


def test_a_data_outage_still_protects_holdings_then_fails_the_run(wired, monkeypatch):
    holder, sent = wired

    def down(*a, **k):
        raise RuntimeError("bars feed down")

    monkeypatch.setattr(run_daily, "cached_bars", down)
    holder["broker"] = FakeBroker(cash=9000.0, positions=_live("S05"))
    state_mod.save_state({**state_mod.load_state(), "positions": _held("S05", 1)})
    with pytest.raises(RuntimeError, match="bars feed down"):
        run(["--no-analyst"])
    b = holder["broker"]
    assert [p[0] for p in b.protected] == ["S05"] and not b.brackets and not b.closed
    assert "MANUAL ACTION: the decision phase failed" in sent[-1]
    assert state_mod.load_state()["last_run"] == date.today().isoformat()


def test_a_holding_without_a_stop_is_flagged_on_every_run(wired):
    holder, sent = wired
    for _ in range(2):
        holder["broker"] = FakeBroker(cash=9000.0, positions=_live("S07"))
        run(["--no-analyst"])
        assert "MANUAL ACTION: S07 has NO protective stop" in sent[-1]
        assert "stop NONE" in sent[-1]


def test_dry_run_writes_no_brief_sends_nothing_and_resume_changes_nothing(wired, capsys):
    holder, sent = wired
    state_mod.save_state({**state_mod.load_state(), "halted": True, "halt_reason": "drawdown", "halt_date": "2026-09-01"})
    holder["broker"] = FakeBroker(dry_run=True)
    run(["--no-analyst", "--dry-run", "--resume"])
    assert sent == []                                                  # no webhook
    assert not (state_mod.BRIEFS_DIR / f"{date.today().isoformat()}.md").exists()
    assert "DESK BRIEF" in capsys.readouterr().out                    # printed instead
    assert state_mod.load_state()["halted"]                            # --resume under --dry-run cleared nothing


def test_the_brief_shows_the_days_target_not_the_configured_one(wired, cfg):
    holder, sent = wired
    holder["regime"] = {"state": "off", "scale": 0.5, "label": "risk-off", "spy": 400.0, "spy_ma": 450.0, "vix": 35.0}
    holder["broker"] = FakeBroker()
    run(["--no-analyst"])
    half = max(1, round(cfg["portfolio"]["target_positions"] * cfg["regime"]["risk_off_scale"]))
    assert f"(target {half})" in sent[-1]


def test_a_regime_failure_runs_at_the_risk_off_target_and_sells_nothing(wired, cfg, monkeypatch):
    holder, sent = wired

    def boom(*a):
        raise ValueError("SPY frame empty")

    monkeypatch.setattr(run_daily, "current_regime", boom)
    _patched_cfg(monkeypatch, regime={"off_liquidate": True})          # even with the cash rule switched on
    holder["broker"] = FakeBroker(cash=9000.0, positions=_live("S05"))
    state_mod.save_state({**state_mod.load_state(), "positions": _held("S05", 3)})
    run(["--no-analyst"])
    half = max(1, round(cfg["portfolio"]["target_positions"] * cfg["regime"]["risk_off_scale"]))
    b = holder["broker"]
    assert "regime gate unavailable" in sent[-1] and f"(target {half})" in sent[-1]
    assert 1 + len(b.brackets) <= half and "S05" not in b.closed


# ------------------------------------------------------------------ fail-safes: analyst coverage, fact sheets

def test_an_entry_the_analyst_gave_no_verdict_for_is_not_bought(wired, monkeypatch):
    holder, sent = wired
    seen = {}

    def partial_review(payload, cfg_):
        seen["candidates"] = [c["symbol"] for c in payload["candidates"]]
        return {"data_concerns": [], "brief": "", "verdicts": [
            {"symbol": seen["candidates"][0], "action": "KEEP", "reason": "fine", "confidence": 0.8},
            {"symbol": seen["candidates"][1], "action": "ADD", "reason": "not an action", "confidence": 1.0}]}

    monkeypatch.setattr(run_daily, "review", partial_review)
    _patched_cfg(monkeypatch, analyst={"red_team_enabled": False})
    holder["broker"] = FakeBroker()
    run([])
    assert len(seen["candidates"]) >= 3
    assert [s for s, *_ in holder["broker"].brackets] == seen["candidates"][:1]
    assert "analyst gave no verdict for" in sent[-1] and seen["candidates"][1] in sent[-1]


def test_apply_verdicts_drops_what_it_was_not_given_a_verdict_for():
    from desk.analyst import apply_verdicts
    entries = [{"symbol": s, "qty": 4, "price": 10.0, "notional": 40.0} for s in ("AAA", "BBB", "CCC")]
    kept, exits, notes = apply_verdicts([{"symbol": "aaa", "action": "keep", "reason": "", "confidence": 1},
                                         {"symbol": "BBB", "action": "ADD", "reason": "", "confidence": 1}], {}, entries)
    assert [e["symbol"] for e in kept] == ["AAA"] and exits == {}
    assert notes == ["analyst gave no verdict for BBB, CCC -> not bought (nothing is bought unreviewed)"]


def test_no_fact_sheets_means_no_new_entries(wired, monkeypatch):
    holder, sent = wired

    class Down:
        def __init__(self, *a, **k):
            raise ConnectionError("news API unreachable")

    monkeypatch.setattr(run_daily, "FactBuilder", Down)
    holder["broker"] = FakeBroker(cash=9000.0, positions=_live("S05"))
    state_mod.save_state({**state_mod.load_state(), "positions": _held("S05", 3)})
    run(["--no-analyst"])
    b = holder["broker"]
    assert not b.brackets and [p[0] for p in b.protected] == ["S05"]
    assert "fact sheets unavailable (ConnectionError) -> fail-safe: no new entries today" in sent[-1]


def test_an_unknown_earnings_calendar_blocks_every_candidate(wired, cfg, tmp_path, monkeypatch):
    from desk import facts as F
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    fb = F.FactBuilder("k", "s", dict(cfg, _root=str(tmp_path)), date.today())
    assert fb._earnings_calendar(["AAA", "BBB"]) == {} and fb.earnings_failed == {"AAA", "BBB"}
    assert F.candidate_blockers(fb._earnings("AAA", {}), cfg["facts"]) == ["earnings date lookup failed - not bought blind"]

    holder, sent = wired
    holder["facts"] = {f"S{i:02d}": {"symbol": f"S{i:02d}", "earnings_lookup_failed": True} for i in range(30)}
    holder["broker"] = FakeBroker()
    run(["--no-analyst"])
    assert not holder["broker"].brackets
    assert "earnings calendar unavailable for every candidate" in sent[-1]


def test_a_run_while_the_market_is_open_only_checks_stops(wired, monkeypatch):
    """A Mac that wakes late starts the missed job mid-session: signals would read a partial day and exits
    would fill at once. It syncs, runs the kill switch and checks stops; the evening run decides."""
    holder, sent = wired
    reviewed = []
    monkeypatch.setattr(run_daily, "review", lambda payload, cfg_: reviewed.append(1))
    holder["broker"] = FakeBroker(cash=9000.0, positions=_live("S05"))
    holder["broker"].session_open = True
    state_mod.save_state({**state_mod.load_state(), "positions": _held("S05", 70)})     # past max hold
    run([])
    b = holder["broker"]
    assert [p[0] for p in b.protected] == ["S05"]                       # its stop is still checked
    assert not b.brackets and not b.closed and not reviewed             # no decisions, no LLM spend
    assert "PROTECT-ONLY RUN" in sent[-1]


def test_the_minutes_after_the_close_are_protect_only_too(wired):
    holder, sent = wired
    holder["clock"] = holder["clock"].replace(hour=16, minute=10)       # closed, but the bars are not final
    holder["broker"] = FakeBroker()
    run(["--no-analyst"])
    assert not holder["broker"].brackets and "PROTECT-ONLY RUN at 16:10 ET" in sent[-1]


def test_force_runs_everything_mid_session_and_says_the_bar_is_partial(wired):
    holder, sent = wired
    holder["clock"] = holder["clock"].replace(hour=11, minute=0)
    holder["broker"] = FakeBroker()
    holder["broker"].session_open = True
    run(["--no-analyst", "--force"])
    assert holder["broker"].brackets and "session still in progress" in sent[-1]


# ------------------------------------------------------------------ broker sync

def test_a_close_still_working_is_left_to_fill():
    from desk.execution import sync_positions
    b = FakeBroker(positions=_live("AAA"))
    b.working_closes = {"AAA": object()}
    st = {"positions": {"AAA": {**_held("AAA", 20)["AAA"], "pending_exit_reason": "signal decay"}}}
    notes = sync_positions(st, b, b.positions(), date.today())
    assert st["positions"]["AAA"]["pending_exit_reason"] == "signal decay"
    assert any("close still working" in n for n in notes)


def test_leftover_sell_orders_are_cancelled_when_a_position_is_gone(tmp_path, monkeypatch):
    from alpaca.trading.enums import OrderSide
    from desk.execution import sync_positions
    monkeypatch.setattr(state_mod, "TRADES_FILE", tmp_path / "trades.csv")
    b = FakeBroker()
    st = {"positions": _held("BBB", 10)}
    sync_positions(st, b, {}, date.today())
    assert b.cancelled == [("BBB", OrderSide.SELL)] and "BBB" not in st["positions"]


def test_first_sight_of_a_fill_records_its_date():
    from desk.execution import sync_positions
    b = FakeBroker(positions=_live("CCC"))
    st = {"positions": {"CCC": {**_held("CCC", 1)["CCC"], "filled": False, "order_id": "o1"}}}
    sync_positions(st, b, b.positions(), date.today())
    assert st["positions"]["CCC"]["filled_date"] == date.today().isoformat()


def test_day_trade_guard_counts_this_mornings_fill_only_while_the_market_is_open(cfg):
    from desk.risk import can_exit_today
    today = date.today()
    pos = {"entry_date": (today - timedelta(days=1)).isoformat(), "filled_date": today.isoformat()}
    used_up = cfg["risk"]["max_daytrades_per_5d"]
    assert can_exit_today(pos, today, used_up, cfg["risk"])                          # evening: sells next session
    assert not can_exit_today(pos, today, used_up, cfg["risk"], session_open=True)   # intraday: a day trade
    assert can_exit_today(pos, today, 0, cfg["risk"], session_open=True)


# ------------------------------------------------------------------ state files

def test_logging_the_same_close_twice_writes_one_row(tmp_path, monkeypatch):
    monkeypatch.setattr(state_mod, "TRADES_FILE", tmp_path / "trades.csv")
    monkeypatch.setattr(state_mod, "STATE_DIR", tmp_path)
    row = {"symbol": "AAA", "entry_date": "2026-09-01", "exit_date": "2026-09-20", "exit_price": 12.0, "qty": 3}
    assert state_mod.record_trade(row) and not state_mod.record_trade(row)
    assert len((tmp_path / "trades.csv").read_text().splitlines()) == 2        # header + one trade


def test_atomic_write_leaves_no_temp_file(tmp_path):
    p = tmp_path / "x" / "state.json"
    state_mod.atomic_write_text(p, "{}")
    state_mod.atomic_pickle({"a": 1}, tmp_path / "x" / "p.pkl")
    assert p.read_text() == "{}" and sorted(f.name for f in p.parent.iterdir()) == ["p.pkl", "state.json"]


# ------------------------------------------------------------------ bar cache

def test_last_close_cutoff_steps_back_over_weekends_and_dst():
    from desk.data import last_close_cutoff
    at = lambda s: last_close_cutoff(datetime.fromisoformat(s)).isoformat()  # noqa: E731
    assert at("2026-09-24T22:30:00+00:00") == "2026-09-24T16:20:00-04:00"    # Thu evening: today's close
    assert at("2026-09-24T20:00:00+00:00") == "2026-09-23T16:20:00-04:00"    # Thu 16:00 ET: session running
    assert at("2026-09-28T13:00:00+00:00") == "2026-09-25T16:20:00-04:00"    # Mon pre-market: Friday
    assert at("2026-11-02T14:00:00+00:00") == "2026-10-30T16:20:00-04:00"    # across the DST change


def test_a_panel_fetched_before_the_close_is_refetched_after_it(tmp_path, monkeypatch):
    from desk import data as D
    fetched = []
    monkeypatch.setattr(D, "fetch_daily_bars", lambda *a, **k: fetched.append(1) or {"close": pd.DataFrame({"A": [1.0]})})
    cfg = {"_root": str(tmp_path), "data": {"cache_dir": "cache", "feed": "iex", "cache_retention_days": 0}}
    D.cached_bars(None, ["A"], 10, cfg)
    D.cached_bars(None, ["A"], 10, cfg)
    assert len(fetched) == 1                                          # fetched after the cutoff: reused
    (key,) = (tmp_path / "cache").glob("bars_*.pkl")
    old = D.last_close_cutoff().timestamp() - 3600                    # e.g. a dry run at 11:00 ET
    os.utime(key, (old, old))
    D.cached_bars(None, ["A"], 10, cfg)
    assert len(fetched) == 2


# ------------------------------------------------------------------ config

def test_a_missing_risk_key_stops_the_load(tmp_path):
    import yaml
    from desk.config import ROOT, load_config
    raw = yaml.safe_load((ROOT / "config.yaml").read_text())
    del raw["risk"]["max_cumulative_drawdown_pct"]
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="risk.max_cumulative_drawdown_pct: missing"):
        load_config(p)


def test_a_live_account_is_refused(tmp_path):
    import yaml
    from desk.config import ROOT, load_config, require_paper
    raw = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw["account"]["paper"] = False
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="account.paper"):
        load_config(p)
    with pytest.raises(RuntimeError):
        require_paper(raw)


# ------------------------------------------------------------------ shared rules and the rewritten hot loop

def _ref_too_correlated(sym, basket, returns, pcfg):
    peers = [b for b in basket if b in returns.columns and b != sym]
    if not peers or sym not in returns.columns:
        return False
    window = returns.tail(pcfg["correlation_days"])
    corr = window[peers].corrwith(window[sym])
    return int((corr > pcfg["correlation_threshold"]).sum()) > pcfg["max_correlated_peers"]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")   # the pandas reference on a constant column
def test_too_correlated_matches_pandas_corrwith_exactly(cfg):
    from desk.portfolio import too_correlated
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.01, size=(80, 1))
    rets = pd.DataFrame(base + rng.normal(0, 0.006, size=(80, 12)), columns=[f"S{i}" for i in range(12)])
    rets.iloc[rng.integers(0, 80, 40), rng.integers(0, 12, 40)] = np.nan
    rets["FLAT"] = 0.0
    for thr in (0.3, 0.5, 0.75):
        pcfg = dict(cfg["portfolio"], correlation_threshold=thr)
        for k in range(12):
            basket = [c for c in rets.columns if c != f"S{k}"][:6] + ["FLAT"]
            assert too_correlated(f"S{k}", basket, rets, pcfg) == _ref_too_correlated(f"S{k}", basket, rets, pcfg)


def _ref_snapshot(panels, i, scfg, cols=None):
    def row(k):
        r = panels[k].iloc[i]
        return r if cols is None else r.reindex(cols)
    snap = pd.DataFrame({k: row(k) for k in ("close", "momentum", "short", "vol", "atr", "ma")}).dropna()
    snap["atr_pct"] = snap["atr"] / snap["close"]
    snap["eligible"] = (snap["close"] > snap["ma"]) & (snap["momentum"] > 0) & (snap["atr_pct"] <= scfg["max_atr_pct"])
    w = scfg["weights"]
    el = snap[snap["eligible"]]
    score = (w["momentum"] * el["momentum"].rank(pct=True) + w["short"] * el["short"].rank(pct=True)
             + w["low_vol"] * (-el["vol"]).rank(pct=True))
    snap["score"] = score
    snap["rank"] = score.rank(ascending=False, method="first")
    return snap.sort_values("rank")


def test_snapshot_matches_the_dataframe_reference_exactly(cfg, panel):
    from desk.signals import compute_panels, snapshot
    panel = {k: v.copy() for k, v in panel.items()}
    panel["close"].iloc[-3:, 4] = np.nan                               # a holding with missing bars
    panels = compute_panels(panel, cfg["signals"])
    cols = pd.Index(["S09", "S02", "NOPE", "S04", "S11", "S00"])      # members + holdings, one not in the panel
    for i in (-1, -2, -40, 200):
        for c in (None, cols):
            pd.testing.assert_frame_equal(snapshot(panels, i, cfg["signals"], c),
                                          _ref_snapshot(panels, i, cfg["signals"], c), check_exact=True)


def test_rerank_shortcut_matches_the_full_rerank(cfg, panel):
    from desk.signals import compute_panels, rerank, snapshot
    snap = snapshot(compute_panels(panel, cfg["signals"]), -1, cfg["signals"])
    top = list(snap.index[snap["eligible"].values][:3])
    ineligible = [s for s in snap.index if not snap.loc[s, "eligible"]]
    full = rerank(snap, {top[1]: 0.0}, {"ZZZZ": "x", **{s: "x" for s in ineligible[:1]}, top[0]: "earnings"})
    assert not full.loc[top[0], "eligible"] and full.loc[top[1], "rank"] == 1
    pd.testing.assert_frame_equal(rerank(snap, {top[1]: 0.0}, {"ZZZZ": "x"}), snap, check_exact=True)


def test_shared_rules_never_enlarge_and_never_drop():
    from desk.portfolio import entry_limit, shrink_qty
    assert [shrink_qty(q, f) for q, f in ((7, 0.75), (1, 0.5), (10, 3.0), (4, -1))] == [5, 1, 10, 1]
    assert entry_limit(100.0) == 100.5 and entry_limit(21.845) == round(21.845 * 1.005, 2)


def test_decay_exit_holds_defers_and_exits(cfg):
    from desk.portfolio import decay_exit
    snap = pd.DataFrame({"eligible": [True, False], "rank": [3.0, np.nan]}, index=["A", "B"])
    e = cfg["exits"]
    assert decay_exit(snap, "B", e["min_hold_days"] - 1, e) is None           # too young to judge
    assert decay_exit(snap, "A", e["min_hold_days"], e) is None
    assert decay_exit(snap, "B", e["min_hold_days"], e) == "decay"
    assert decay_exit(snap, "C", e["min_hold_days"], e) == "defer"            # no bar today


def test_panel_cache_key_ignores_snapshot_only_settings(cfg):
    from desk.signals import panel_key
    s = cfg["signals"]
    assert panel_key(s) == panel_key(dict(s, max_atr_pct=0.2, weights={"momentum": 1, "short": 0, "low_vol": 0}))
    assert panel_key(s) != panel_key(dict(s, momentum_lookback_days=84))


def test_robustness_workers_keep_a_bounded_panel_cache(cfg, panel, monkeypatch):
    import robustness as R
    R._G.clear()
    R._G.update(cfg=cfg, spy=None, vix=None, earnings=None, capital=10000.0, panel=panel,
                members={"pit": pd.DataFrame(True, index=panel["close"].index, columns=panel["close"].columns)})
    from collections import OrderedDict
    R._G["signal_panels"] = OrderedDict()
    d = panel["close"].index
    job = {"name": "x", "group": "g", "start": str(d[200].date()), "end": str(d[-1].date())}
    for ov in ([], [(("signals", "max_atr_pct"), 0.2)], [(("signals", "trend_ma_days"), 30)],
               [(("signals", "momentum_skip_days"), 10)]):
        R.run_job({**job, "overrides": ov})
    assert len(R._G["signal_panels"]) == R.PANEL_SETS
    R._G.clear()


# ------------------------------------------------------------------ monitor

PT = timezone(timedelta(hours=-7))


def test_a_missed_desk_slot_is_judged_by_the_very_next_monitor_run():
    """Steady state: the previous monitor ran yesterday at 14:00, the desk slot was 13:45 today. The slot was
    inside the old 60-minute grace today and before the window tomorrow, so it was never judged."""
    now = datetime(2026, 9, 24, 14, 0, 30, tzinfo=PT)
    prev = now - timedelta(days=1)
    grace = timedelta(minutes=10)
    slots = M.expected_runs([(d, 13, 45) for d in range(5)], prev - grace, now, grace)
    assert slots == [datetime(2026, 9, 24, 13, 45, tzinfo=PT)]
    assert M.match_runs(slots, [])["missed"] == slots
    assert M.match_runs(slots, [datetime(2026, 9, 24, 13, 45, 4, tzinfo=PT)])["missed"] == []


def test_monitor_job_alarms_on_todays_missed_desk_run(tmp_path, monkeypatch):
    import monitor as job
    now = datetime.now().astimezone()
    slot = now - timedelta(minutes=15)
    monkeypatch.setattr(job, "STATE_DIR", tmp_path)
    monkeypatch.setattr(job, "MON_DIR", tmp_path / "monitor")
    monkeypatch.setattr(job, "installed_schedules",
                        lambda: {"desk": [(d, slot.hour, slot.minute) for d in range(7)], "report": None, "audit": None})
    monkeypatch.setattr(job, "trading_days", lambda cfg, s, e: None)
    sent = []
    monkeypatch.setattr(job, "send_email", lambda s, b, c, html=None: sent.append(s) or "emailed x")
    (tmp_path / "monitor").mkdir()
    (tmp_path / "monitor" / "state.json").write_text(json.dumps({"last_run": (now - timedelta(days=1)).isoformat()}))
    T.DIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(timezone.utc) - timedelta(days=2)
    with open(T.DIR / f"{t0.date().isoformat()}.jsonl", "a") as f:
        f.write(json.dumps({"ts": t0.isoformat(timespec="seconds"), "run_id": "r0", "job": "report",
                            "event": "run_start"}) + "\n")
    sys.argv = ["monitor.py", "--no-llm"]
    job.main()
    assert sent and "Desk job did not run" in sent[0]


def test_unprotected_holdings_halts_and_failed_emails_are_hard_alarms():
    ev = [{"event": "run_end", "job": "desk", "run_id": "d1", "ts": "2026-09-24T20:50:00+00:00", "status": "ok",
           "unprotected": 1, "protection_unverified": 0},
          {"event": "run_end", "job": "desk", "run_id": "d2", "ts": "2026-09-25T20:50:00+00:00", "status": "halted",
           "reason": "drawdown 15.2% >= 15% limit", "hard": False, "halt_date": "2026-09-25", "open_positions": 2},
          {"event": "run_end", "job": "report", "run_id": "r1", "ts": "2026-09-24T21:05:00+00:00", "status": "ok",
           "email": "failed", "email_detail": "SMTP FAILED (SMTPAuthenticationError)"},
          {"event": "run_end", "job": "report", "run_id": "r2", "ts": "2026-09-23T21:05:00+00:00", "status": "ok",
           "email": "sent"}]
    keys = {a["key"] for a in M.hard_alarms({}, {}, None, ev)}
    assert keys == {"desk-unprotected-2026-09-24", "desk-halted-2026-09-25", "report-email-r1"}


def test_a_report_that_crashed_or_never_ran_is_a_hard_alarm():
    t = datetime(2026, 9, 23, 14, 0, tzinfo=PT)
    alarms = M.hard_alarms({"report": {"missed": [t]}},
                           {"report": {"errors": [{"run_id": "r9", "ts": "x", "error": "KeyError"}]}}, None)
    assert {a["key"] for a in alarms} == {"report-missed-2026-09-23", "report-error-r9"}


# ------------------------------------------------------------------ report, telemetry, facts, backtest

def test_a_failed_email_does_not_advance_the_change_log(tmp_path, monkeypatch):
    import report_daily
    monkeypatch.setattr(report_daily, "STATE_DIR", tmp_path)
    fp = tmp_path / "fingerprint.json"
    monkeypatch.setattr(report_daily, "send_email", lambda *a, **k: "SMTP FAILED (SMTPServerDisconnected): x")
    report_daily.deliver(date.today(), "s", "text", "<p>", {}, {fp: "{}"})
    assert not fp.exists() and (tmp_path / "reports" / f"{date.today().isoformat()}.txt").exists()
    monkeypatch.setattr(report_daily, "send_email", lambda *a, **k: "emailed x via smtp")
    report_daily.deliver(date.today(), "s", "text", "<p>", {}, {fp: "{}"})
    assert fp.exists()


def test_web_search_charges_are_recorded_beside_the_token_cost():
    from tests.test_monitor import _events, _resp
    T.llm_call("red_team", "claude-opus-5", time.monotonic(), [_resp(1000, 1000, searches=3)])
    e = _events()[-1]
    assert e["web_search_usd"] == pytest.approx(0.03) and e["cost_usd"] == pytest.approx(0.005 + 0.025)
    assert M.role_signals([e])["red_team"]["web_search_usd_total"] == pytest.approx(0.03)


def test_each_symbol_is_looked_up_at_the_broker_once(cfg, tmp_path, monkeypatch):
    from desk import facts as F
    monkeypatch.setenv("SEC_USER_AGENT", "")
    calls = []
    lookup = lambda s: calls.append(s) or SimpleNamespace(name="Acme Corp", easy_to_borrow=True, shortable=True)  # noqa: E731
    fb = F.FactBuilder("k", "s", dict(cfg, _root=str(tmp_path)), date.today(), asset_lookup=lookup)
    assert fb._company("ACME") == "Acme Corp" and fb._asset("ACME") == {"easy_to_borrow": True, "shortable": True}
    assert calls == ["ACME"]


def test_only_the_default_snapshot_rewrites_the_live_reference():
    import backtest as B
    p = B.build_parser()
    assert B.writes_live_reference(p.parse_args([]), "universe_bt.json")
    assert not B.writes_live_reference(p.parse_args(["--universe", "universe_bt_5m_1500.json"]), "universe_bt_5m_1500.json")
    assert not B.writes_live_reference(p.parse_args(["--capital", "50000"]), "universe_bt.json")
    assert not B.writes_live_reference(p.parse_args(["--trail-atr", "2"]), "universe_bt.json")


def test_the_config_id_lives_outside_the_backtest_cli_and_is_unchanged(cfg):
    import backtest as B
    from desk import trials as Tr
    assert Tr.current_config_id(cfg) == B.current_config_id(cfg)
    fp = Tr.strategy_fingerprint(cfg, Tr.DEFAULT_SLIPPAGE_BPS, Tr.DEFAULT_HALT_COOLDOWN)
    assert fp["slippage_bps"] == 10 and isinstance(fp["slippage_bps"], int)   # hashed as an int in every logged id


def test_a_stop_that_does_not_match_the_shares_held_is_flagged(wired):
    """UMC 2026-09-24: trimmed by hand from 45 to 42 shares. The OCO had to be resized too; had it stayed at 45,
    the desk (which only checked that a stop existed) would have called the position protected while the stop
    would have sold 3 shares short when it fired."""
    holder, sent = wired
    stop = lambda q: SimpleNamespace(stop_price=40.0, qty=q)  # noqa: E731
    live = {**_live("S02", qty=42), **_live("S03", qty=45), **_live("S04", qty=45)}
    holder["broker"] = FakeBroker(cash=9000.0, positions=live)
    holder["broker"].existing_stops = {"S02": stop(45), "S03": stop(40), "S04": stop(42)}
    holder["broker"].working_closes = {"S04": SimpleNamespace(qty=3)}        # the trim still working: fine
    state_mod.save_state({**state_mod.load_state(), "positions": {**_held("S02", 3, qty=42), **_held("S03", 3, qty=45),
                                                                   **_held("S04", 3, qty=45)}})
    run(["--no-analyst"])
    assert "S02 protection MISMATCH: the resting stop is for 45 shares but 42 are held" in sent[-1]
    assert "S03 has NO protective stop for 5 of 45 share(s)" in sent[-1]
    assert "S04 protection" not in sent[-1] and "S04 has NO" not in sent[-1]

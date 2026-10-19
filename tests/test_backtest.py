import pandas as pd

from backtest import run_backtest, summarize


def test_backtest_respects_constraints(cfg, panel):
    pcfg = cfg["portfolio"]
    dates = panel["close"].index
    curve, trades = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, slippage_bps=10, halt_cooldown=5)
    assert len(curve) > 50
    assert curve["n_positions"].max() <= pcfg["max_positions"]
    assert (curve["equity"] > 0).all()
    if len(trades):
        assert trades["hold_days"].max() <= cfg["exits"]["max_hold_days"] + 3
        assert (trades["qty"] >= 1).all()
        assert set(trades["reason"]) <= {"stop", "stop-gap", "target", "target-gap", "time", "decay", "kill-switch",
                                         "hard-floor", "regime", "delisted"}
    # Entry pacing: never more new positions in a day than the cap allows.
    cap = cfg["portfolio"]["max_new_entries_per_day"]
    assert curve["n_positions"].diff().max() <= cap
    text = summarize(curve, trades, None)
    assert "Max drawdown" in text


def test_backtest_with_qualitative_features(cfg, panel):
    dates = panel["close"].index
    syms = list(panel["close"].columns)
    # Earnings for every symbol every 40 days -> almost everything is blocked most of the time.
    earnings = pd.DataFrame([(d.date().isoformat(), s) for s in syms for d in dates[::40]], columns=["date", "symbol"])
    insider = pd.DataFrame([(dates[10].date().isoformat(), "S00", 250000.0)], columns=["date", "symbol", "net_usd"])
    regime = pd.Series(0.5, index=dates)
    tilted_cfg = dict(cfg, facts=dict(cfg["facts"], insider_tilt_weight=0.2))
    curve, trades = run_backtest(panel, tilted_cfg, dates[0], dates[-1], 10000.0, 10, 5,
                                 regime=regime, earnings=earnings, insider=insider)
    assert curve["n_positions"].max() <= round(cfg["portfolio"]["target_positions"] * 0.5)
    base_curve, base_trades = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5)
    assert len(trades) < len(base_trades)  # the earnings blocks bite


def test_no_leverage_ever(cfg, panel):
    dates = panel["close"].index
    curve, trades = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5)
    assert (curve["cash"] >= -1e-6).all()
    assert (curve["cash"] >= curve["equity"] * cfg["portfolio"]["cash_floor_pct"] * 0.5).mean() > 0.9


def test_same_inputs_give_the_same_answer(cfg, panel):
    """The whole point of the frozen universe and pinned end date. Guards against hidden nondeterminism
    (set ordering, dict iteration) creeping into the loop itself."""
    dates = panel["close"].index
    c1, t1 = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5)
    c2, t2 = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5)
    pd.testing.assert_frame_equal(c1, c2)
    pd.testing.assert_frame_equal(t1, t2)


def test_overrides_never_touch_the_base_config(cfg):
    from backtest import apply_overrides, build_parser
    args = build_parser().parse_args(["--risk-per-trade", "0.0125", "--target-positions", "15", "--ladder"])
    before = cfg["portfolio"]["risk_per_trade_pct"], cfg["portfolio"]["target_positions"], cfg["regime"]["ladder"]
    out = apply_overrides(cfg, args)
    assert out["portfolio"]["risk_per_trade_pct"] == 0.0125 and out["portfolio"]["target_positions"] == 15
    assert out["regime"]["ladder"] is True
    # --stagger runs this base next to the variant; a shared nested dict would make both the variant.
    assert (cfg["portfolio"]["risk_per_trade_pct"], cfg["portfolio"]["target_positions"], cfg["regime"]["ladder"]) == before


def test_frozen_universe_is_read_not_rebuilt(tmp_path, monkeypatch):
    """The live universe.json rebuilds weekly and moved the baseline from +247% to +125%. A backtest must
    read its own snapshot and never rebuild - or touch - the live file unless told to."""
    import json
    import backtest
    (tmp_path / "state" / "history").mkdir(parents=True)
    snap = tmp_path / "state" / "history" / "universe_bt.json"
    snap.write_text(json.dumps({"date": "2026-09-21", "symbols": ["BBB", "AAA"]}))
    cfg = {"_root": str(tmp_path)}

    def must_not_build(*a, **k):
        raise AssertionError("snapshot exists; nothing should be rebuilt")

    monkeypatch.setattr(backtest, "build_universe", must_not_build)
    syms, built, path = backtest.frozen_universe(None, None, cfg, "universe_bt.json")
    assert syms == ["BBB", "AAA"] and built == "2026-09-21" and path == snap
    assert not (tmp_path / "state" / "universe.json").exists()          # live file untouched

    calls = []
    monkeypatch.setattr(backtest, "build_universe",
                        lambda t, d, c, force, cache_name: calls.append((force, cache_name)) or ["CCC"])
    syms, _, _ = backtest.frozen_universe(None, None, cfg, "universe_bt.json", rebuild=True)
    assert syms == ["CCC"] and calls == [(True, "history/universe_bt.json")]
    assert backtest.universe_id(["AAA", "BBB"]) == backtest.universe_id(["BBB", "AAA"])   # order-independent


def test_bars_stop_at_the_requested_end_date():
    from datetime import datetime, timezone
    from backtest import data_end
    now = datetime(2026, 9, 24, 17, 0, tzinfo=timezone.utc)
    # A past end date includes that day's bar and nothing after it, whenever the command is run.
    assert data_end(pd.Timestamp("2026-09-11"), now) == datetime(2026, 9, 12, tzinfo=timezone.utc)
    # An end of today is capped at the free-tier SIP delay.
    assert data_end(pd.Timestamp("2026-09-24"), now) == datetime(2026, 9, 24, 16, 44, tzinfo=timezone.utc)


def test_backtest_prices_are_frozen_not_redownloaded(tmp_path, monkeypatch):
    """A re-download re-exposes the run to feed gaps (one blank TEVA moved the baseline 19 points), so the
    price history is saved beside the universe snapshot and reused until --refetch-prices."""
    import backtest
    from desk.data import prune_bar_cache
    fetched = []
    monkeypatch.setattr(backtest, "fetch_daily_bars",
                        lambda d, syms, days, feed, end=None: fetched.append(end) or {"close": pd.DataFrame({"A": [1.0]})})
    cfg = {"_root": str(tmp_path), "data": {"feed": "sip"}}
    end = pd.Timestamp("2026-09-11")
    _, path = backtest.frozen_prices(None, ["A"], 30, cfg, end)
    backtest.frozen_prices(None, ["A"], 30, cfg, end)                       # read back, not fetched
    assert len(fetched) == 1 and path.parent == tmp_path / "state" / "history" / "prices"
    assert fetched[0].date().isoformat() == "2026-09-12"                     # pinned to --end, not today
    backtest.frozen_prices(None, ["A"], 30, cfg, end, refetch=True)
    assert len(fetched) == 2
    # Different window or universe -> its own file.
    assert backtest.frozen_prices(None, ["A", "B"], 30, cfg, end)[1] != path
    # The daily cache pruner never reaches it.
    (tmp_path / "cache").mkdir()
    prune_bar_cache(tmp_path / "cache", keep_days=2)
    assert path.exists()


def test_compare_windows_counts_wins_not_just_one_number():
    from backtest import compare_windows

    def s(total, dd, sh):
        return {"total": total, "maxdd": dd, "sharpe": sh}

    rows = [{"start": "2023-09-11", "base": s(1.25, -0.19, 1.20), "variant": s(2.01, -0.19, 1.54)},
            {"start": "2023-10-11", "base": s(1.10, -0.18, 1.10), "variant": s(1.00, -0.22, 1.00)},
            {"start": "2023-11-11", "base": s(0.90, -0.20, 1.00), "variant": s(1.00, -0.21, 1.05)}]
    text = compare_windows(rows)
    assert "return in 2/3 windows" in text and "drawdown in 1/3" in text and "Sharpe in 2/3" in text
    assert "Median return difference +10.0 pts" in text and "range -10.0 to +76.0" in text


class _Bars:
    def __init__(self, df):
        self.df = df


class _FlakyFeed:
    """Returns nothing for the named symbols on the first request, as the feed did for TEVA on 2026-09-24."""
    def __init__(self, drop_first=(), drop_always=()):
        self.calls, self.drop_first, self.drop_always = [], set(drop_first), set(drop_always)

    def get_stock_bars(self, req):
        syms = list(req.symbol_or_symbols)
        self.calls.append(syms)
        skip = self.drop_always | (self.drop_first if len(self.calls) == 1 else set())
        rows = [{"symbol": s, "timestamp": d, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 10}
                for s in syms if s not in skip for d in pd.date_range("2026-09-01", periods=3, tz="UTC")]
        cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        return _Bars(pd.DataFrame(rows, columns=cols).set_index(["symbol", "timestamp"]))


def test_a_symbol_the_feed_drops_is_fetched_again():
    """Live, a held name with no bars is read as signal decay and sold; in a backtest it silently changes
    the universe. One blank TEVA moved a three-year result by 19 points."""
    from desk.data import fetch_daily_bars
    feed = _FlakyFeed(drop_first={"TEVA"})
    panel = fetch_daily_bars(feed, ["AAA", "TEVA"], 10)
    assert feed.calls == [["AAA", "TEVA"], ["TEVA"]]                 # one retry, only for the missing name
    assert panel["close"]["TEVA"].notna().all()


def test_a_symbol_still_missing_after_the_retry_is_reported(caplog):
    from desk.data import fetch_daily_bars
    feed = _FlakyFeed(drop_always={"GONE"})
    with caplog.at_level("WARNING", logger="desk.data"):
        panel = fetch_daily_bars(feed, ["AAA", "GONE"], 10)
    assert len(feed.calls) == 2 and "GONE" not in panel["close"].columns
    assert any("STILL no bars" in r.message and "GONE" in r.message for r in caplog.records)
    healthy = _FlakyFeed()
    fetch_daily_bars(healthy, ["AAA", "BBB"], 10)
    assert len(healthy.calls) == 1                                     # a complete response is not re-requested


def test_a_day_with_missing_bars_neither_fakes_a_loss_nor_poisons_cash(cfg, panel):
    """Held names used to vanish from equity on a gap day (a fake loss that can trip the kill switch) and
    were then closed at a NaN price, turning cash - and every later equity point - into NaN."""
    dates = panel["close"].index
    gap = dates[-40]
    holey = {f: v.copy() for f, v in panel.items()}
    for f in holey:
        holey[f].loc[gap] = float("nan")
    curve, trades = run_backtest(holey, cfg, dates[0], dates[-1], 10000.0, 10, 5)
    assert curve["equity"].notna().all() and curve["cash"].notna().all()
    assert trades["exit_price"].notna().all() if len(trades) else True
    day = curve["equity"].pct_change().loc[gap]
    assert abs(day) < 0.05, f"gap day moved equity {day:+.1%} with no prices at all"


def _held_on(trades, dates, min_left=10):
    """A (symbol, day) where the symbol is held with at least min_left sessions of the trade still to run."""
    for t in trades.itertuples():
        inside = dates[(dates > pd.Timestamp(t.entry_date)) & (dates < pd.Timestamp(t.exit_date))]
        if len(inside) > min_left:
            return t.symbol, inside[1]
    raise AssertionError("no long enough holding in the synthetic run")


def test_a_holding_whose_bars_stop_is_closed_at_its_last_price_after_a_week(cfg, panel):
    """Delisted names (mergers, failures) used to keep their slot and their last price for the rest of the
    run: the time exit sits behind the missing-bar check, so it never fired."""
    from backtest import STALE_EXIT_DAYS
    dates = panel["close"].index
    _, base = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5)
    sym, day = _held_on(base, dates)
    gone = {f: v.copy() for f, v in panel.items()}
    for f in gone:
        gone[f].loc[gone[f].index >= day, sym] = float("nan")
    _, trades = run_backtest(gone, cfg, dates[0], dates[-1], 10000.0, 10, 5)
    out = trades[(trades.symbol == sym) & (trades.reason == "delisted")]
    assert len(out) == 1
    i = dates.get_loc(day)
    assert pd.Timestamp(out.iloc[0].exit_date) == dates[i + STALE_EXIT_DAYS - 1]
    last_close = panel["close"].loc[dates[i - 1], sym]
    assert abs(out.iloc[0].exit_price - last_close * (1 - 10 / 10000)) < 1e-9


def test_a_short_halt_is_not_a_delisting(cfg, panel):
    from backtest import STALE_EXIT_DAYS
    dates = panel["close"].index
    _, base = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5)
    sym, day = _held_on(base, dates)
    i = dates.get_loc(day)
    halted = {f: v.copy() for f, v in panel.items()}
    for f in halted:
        halted[f].loc[dates[i:i + STALE_EXIT_DAYS - 1], sym] = float("nan")
    _, trades = run_backtest(halted, cfg, dates[0], dates[-1], 10000.0, 10, 5)
    assert "delisted" not in set(trades["reason"])


def test_membership_limits_what_can_be_bought(cfg, panel):
    dates = panel["close"].index
    c0, t0 = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5)
    everyone = pd.DataFrame(True, index=dates, columns=panel["close"].columns)
    c1, t1 = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5, members=everyone)
    pd.testing.assert_frame_equal(c0, c1)                      # all members = no membership at all
    traded = sorted(set(t0["symbol"]))
    banned = everyone.copy()
    banned[traded[:3]] = False
    _, t2 = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5, members=banned)
    assert not set(t2["symbol"]) & set(traded[:3])


def test_random_entries_are_reproducible_and_differ_from_the_ranking(cfg, panel):
    import numpy as np
    dates = panel["close"].index
    _, ranked = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5)
    runs = [run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5, entry_rng=np.random.default_rng(s))[1]
            for s in (1, 1, 2)]
    pd.testing.assert_frame_equal(runs[0], runs[1])
    assert list(runs[0]["symbol"]) != list(ranked["symbol"]) or list(runs[2]["symbol"]) != list(ranked["symbol"])


def test_simulated_vetoes_only_ever_shrink_or_drop_entries(cfg, panel):
    import numpy as np
    dates = panel["close"].index
    _, base = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5)
    none = {"remove": 1.0, "reduce": 0.0, "factor": 0.75, "half": 0.0, "rng": np.random.default_rng(0)}
    _, dropped = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5, veto=none)
    assert dropped.empty
    halve = {"remove": 0.0, "reduce": 0.0, "factor": 0.75, "half": 1.0, "rng": np.random.default_rng(0)}
    _, halved = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5, veto=halve)
    first = base.iloc[0]
    same = halved[(halved.symbol == first.symbol) & (halved.entry_date == first.entry_date)]
    assert len(same) == 1 and same.iloc[0].qty == max(1, int(first.qty * 0.5))


def _exits(cfg, **kw):
    return dict(cfg, exits=dict(cfg["exits"], **kw))


def test_a_stop_on_the_close_sells_at_the_next_open_and_only_after_a_close_below_it(cfg, panel):
    """Backtest-only variant (scripts/check_stop_variants.py): an intraday dip through the stop that closes
    above it keeps the holding; a close at or below it sells at the next session's open."""
    dates = panel["close"].index
    c2 = _exits(cfg, stop_on="close")
    _, trades = run_backtest(panel, c2, dates[0], dates[-1], 10000.0, 10, 5)
    stops = trades[trades.reason.str.startswith("stop")]
    assert len(stops) and set(stops.reason) == {"stop-close"}
    for t in stops.itertuples():
        i = dates.get_loc(pd.Timestamp(t.exit_date))
        assert panel["close"].loc[dates[i - 1], t.symbol] <= t.final_stop
        assert abs(t.exit_price - panel["open"].loc[dates[i], t.symbol] * (1 - 10 / 10000)) < 1e-9


def test_the_resting_disaster_stop_still_fires_intraday_under_a_close_stop(cfg, panel):
    dates = panel["close"].index
    _, trades = run_backtest(panel, _exits(cfg, stop_on="close", disaster_atr_multiple=2.5), dates[0], dates[-1],
                             10000.0, 10, 5)
    _, none = run_backtest(panel, _exits(cfg, stop_on="close"), dates[0], dates[-1], 10000.0, 10, 5)
    assert not set(none.reason) & {"stop", "stop-gap"}
    assert set(trades.reason) & {"stop", "stop-gap"}
    _, default = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5)
    assert "stop-close" not in set(default.reason)



def test_a_signal_anchored_stop_is_measured_from_the_staged_price_not_the_fill(cfg, panel):
    """The live desk sets the bracket from the signal-day close (desk/portfolio.py _bracket); the backtest from the
    fill. stop_anchor: signal models live, so the two can be compared (audit 2026-10-17). Same ATR distance, so
    for the same entry the two stops differ by exactly (signal close - fill)."""
    dates = panel["close"].index
    _, live = run_backtest(panel, _exits(cfg, stop_anchor="signal"), dates[0], dates[-1], 10000.0, 10, 5)
    _, bt = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5)
    first = lambda t: t.sort_values("exit_date").drop_duplicates(["symbol", "entry_date"]).set_index(["symbol", "entry_date"])
    both = first(live).join(first(bt), lsuffix="_live", rsuffix="_bt", how="inner")
    assert len(both) >= 5
    checked = 0
    for (sym, entry), r in both.iterrows():
        if abs(r.entry_price_live - r.entry_price_bt) > 1e-9:
            continue                                   # paths diverged; only like-for-like fills compare
        i = dates.get_loc(pd.Timestamp(entry))
        ref = panel["close"].loc[dates[i - 1], sym]
        assert abs((r.final_stop_live - r.final_stop_bt) - (ref - r.entry_price_bt)) <= 0.011
        checked += 1
    assert checked >= 5 and (both.final_stop_live != both.final_stop_bt).any()


def test_a_recent_dilutive_filing_keeps_a_candidate_out_for_ninety_days(cfg, panel):
    """The live dilution block (S-1 or 8-K 3.02 in the last 90 days) modelled in the backtest (review item 6)."""
    dates = panel["close"].index
    _, base = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5)
    sym = base["symbol"].value_counts().index[0]                          # the name the strategy buys most
    first = pd.Timestamp(base[base.symbol == sym]["entry_date"].min())
    filing = first - pd.Timedelta(days=10)
    off = pd.DataFrame({"date": [filing.date().isoformat()], "symbol": [sym]})
    _, blocked = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5, offerings=off)
    entries = pd.to_datetime(blocked[blocked.symbol == sym]["entry_date"])
    # No entry filled within the 90 days after the filing: a signal on day 90 is the last one blocked, and its fill
    # lands on the next session - day 91 or later. (A bound of 91 failed on 2026-10-19, when the synthetic
    # calendar put a fill exactly on day 91.)
    assert not ((entries > filing + pd.Timedelta(days=1)) & (entries <= filing + pd.Timedelta(days=90))).any()
    _, none = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5, offerings=pd.DataFrame(columns=["date", "symbol"]))
    pd.testing.assert_frame_equal(none, base)                             # an empty history changes nothing


def test_the_cash_sweep_holds_idle_cash_in_spy_without_leverage(cfg, panel):
    """Review item 7: idle cash above the floor and tonight's orders goes into SPY; off, nothing changes."""
    dates = panel["close"].index
    spy = panel["close"]["S00"]
    base, t0 = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5)
    same, t1 = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5, sweep_px=spy)       # disabled in config
    pd.testing.assert_frame_equal(base, same)
    on = dict(cfg, portfolio=dict(cfg["portfolio"], cash_sweep={"enabled": True, "band_pct": 0.02}))
    curve, trades = run_backtest(panel, on, dates[0], dates[-1], 10000.0, 10, 5, sweep_px=spy)
    assert (curve["cash"] >= -1e-6).all()                                     # never borrows
    assert (curve["cash"] / curve["equity"]).mean() < (base["cash"] / base["equity"]).mean()
    assert (curve["cash"] / curve["equity"]).median() < (base["cash"] / base["equity"]).median()   # floor + staged orders
    assert len(trades) > 0


def test_the_sweep_funds_new_entries_instead_of_starving_them(cfg, panel):
    """First version budgeted entries from cash alone, so a swept account bought almost nothing new."""
    dates = panel["close"].index
    spy = panel["close"]["S00"]
    _, base = run_backtest(panel, cfg, dates[0], dates[-1], 10000.0, 10, 5)
    on = dict(cfg, portfolio=dict(cfg["portfolio"], cash_sweep={"enabled": True, "band_pct": 0.02}))
    _, swept = run_backtest(panel, on, dates[0], dates[-1], 10000.0, 10, 5, sweep_px=spy)
    assert len(swept) >= 0.8 * len(base)

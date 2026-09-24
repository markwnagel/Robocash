"""Daily report: performance windows, change detection, movers, news, rendering."""
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from desk.report import (baselines, build_report, diff_fingerprint, fallback_summary, inception, movers,
                         notable_news, performance, plain_summaries)


NOW = datetime(2026, 9, 17, 14, 0, tzinfo=timezone(timedelta(hours=-7), "PDT"))


def _series(pairs):
    return pd.Series([v for _, v in pairs], index=pd.to_datetime([d for d, _ in pairs]))


def test_inception_is_the_last_flat_point_before_the_first_trade():
    # Funded Friday, first fill Tuesday: SPY must not be credited with Mon-Tue moves the desk missed.
    eq = _series([("2026-09-11", 10000.0), ("2026-09-14", 10000.0), ("2026-09-15", 10120.0)])
    assert inception(eq) == date(2026, 9, 14)
    assert inception(_series([("2026-09-11", 10000.0), ("2026-09-14", 10000.0)])) == date(2026, 9, 11)


def test_performance_measures_desk_and_spy_from_the_same_dates():
    eq = _series([("2026-09-14", 10000.0), ("2026-09-15", 10100.0), ("2026-09-17", 10200.0)])
    spy = _series([("2026-09-14", 100.0), ("2026-09-15", 101.0), ("2026-09-17", 102.0)])
    rows = {r["period"]: r for r in performance(eq, spy, date(2026, 9, 17))}
    inc = rows["Since inception"]
    assert inc["from"] == date(2026, 9, 14)
    assert abs(inc["desk_pct"] - 0.02) < 1e-9
    assert abs(inc["spy_pct"] - 0.02) < 1e-9
    assert abs(inc["diff_pct"]) < 1e-9
    assert set(rows) == {"Since inception", "This week", "This month", "Year to date"}
    # No SPY data: the desk number still reports, the comparison is dropped rather than faked.
    assert performance(eq, pd.Series(dtype=float), date(2026, 9, 17))[0]["spy_pct"] is None
    assert performance(pd.Series(dtype=float), spy, date(2026, 9, 17)) == []


def test_week_baseline_is_the_session_before_monday():
    b = baselines(date(2026, 9, 17), date(2020, 1, 1))          # a Thursday
    assert b["This week"] == date(2026, 9, 13)                   # Sunday -> asof picks Friday's close
    assert b["This month"] == date(2026, 8, 31)
    assert b["Year to date"] == date(2025, 12, 31)
    # Periods that predate the account are clamped to inception.
    assert baselines(date(2026, 9, 17), date(2026, 9, 14))["Year to date"] == date(2026, 9, 14)


def test_diff_fingerprint_reports_settings_and_code():
    old = {"config": {"facts.exclude_earnings_within_days": 21, "analyst.model": "claude-opus-5"},
           "code": {"desk/risk.py": "aaa", "desk/facts.py": "bbb"}}
    new = {"config": {"facts.exclude_earnings_within_days": 5, "analyst.model": "claude-opus-5",
                      "exits.trail_atr_multiple": 2},
           "code": {"desk/risk.py": "aaa", "desk/facts.py": "ccc", "desk/report.py": "ddd"}}
    lines = diff_fingerprint(old, new)
    assert "setting facts.exclude_earnings_within_days: 21 -> 5" in lines
    assert "setting exits.trail_atr_multiple: <unset> -> 2" in lines
    assert "code desk/facts.py: edited" in lines
    assert "code desk/report.py: added" in lines
    assert not any("risk.py" in ln for ln in lines)          # unchanged files stay silent
    assert diff_fingerprint({}, new) == []                   # first ever run reports nothing


def test_movers_falls_back_when_nothing_moved_much():
    quiet = [{"symbol": "A", "day_pct": 0.001}, {"symbol": "B", "day_pct": -0.004}, {"symbol": "C", "day_pct": 0.002}]
    assert len(movers(quiet)["notable"]) == 2
    loud = quiet + [{"symbol": "D", "day_pct": 0.09}]
    m = movers(loud)
    assert [p["symbol"] for p in m["notable"]] == ["D"] and m["rest"] == 3


def test_notable_news_prefers_flagged_then_fresh():
    facts = {
        "A": {"headlines": [{"headline": "quiet day", "age_h": 2.0, "source": "x"}]},
        "B": {"cls_offering": True, "headlines": [{"headline": "share sale", "age_h": 20.0, "source": "x"}]},
        "C": {"headlines": [{"headline": "ancient", "age_h": 400.0, "source": "x"}]},
        "D": {},
    }
    out = notable_news(facts, ["A", "B", "C", "D"])
    assert [n["symbol"] for n in out] == ["B", "A"]           # flagged first, then fresh; stale and empty dropped
    assert out[0]["flags"] == ["offering"]


JARGON = ("volatility", "implied", "iv/rv", "atr", "momentum", "severity", "catalyst", "basis point")


def test_fallback_summary_explains_every_exit_reason_in_plain_words():
    # Each string run_daily.py can write must map to a sentence; an unmapped reason would ship a sale
    # with no explanation under it.
    for reason in ("analyst REMOVE: severity 0.72 unpaid tail", "max hold 60d", "signal decay",
                   "pre-earnings exit (3 trading days to report)", "KILL SWITCH: drawdown 15%",
                   "regime: SPY below its MA (or VIX high) -> cash",
                   "stop 20.10 breached (no live broker order)", "bracket/stop/target", "closed at broker"):
        s = fallback_summary("sold", {"exit_reason": reason, "pnl": -22.59, "hold_days": 3})
        assert s.endswith("lost $22.59.") and len(s.split(". ")) <= 2
        assert not any(j in s.lower() for j in JARGON), (reason, s)
    assert "no longer held" in fallback_summary("sold", {"exit_reason": "analyst REMOVE: x", "pnl": 0,
                                                         "hold_days": 1})
    assert "Held 1 day," in fallback_summary("sold", {"exit_reason": "max hold 60d", "pnl": 5, "hold_days": 1})


def test_fallback_summary_reads_the_rank_and_size_decision_off_a_buy():
    s = fallback_summary("bought", {"thesis": "rank 4, 6m mom 214%, 20d 21% | analyst REDUCE: blah"})
    assert "ranked 4 out of the ~800" in s and "214% over six months" in s and "21% in the past month" in s
    assert "smaller stake" in s                       # REDUCE has to reach the reader, not just the log
    assert "smaller stake" not in fallback_summary("bought", {"thesis": "rank 4, 6m mom 214%, 20d 21%"})
    assert fallback_summary("bought", {"thesis": ""})  # a thesis-less adopted position still says something


def test_trim_never_ships_half_a_sentence():
    from desk.report import trim_text
    assert trim_text("One. Two. Three.") == "One. Two."                   # third sentence dropped
    assert trim_text("  spaced\n out.  ") == "spaced out."
    long_one = "We sold because " + "the price moved a very long way " * 20 + "today."
    out = trim_text(long_one)
    assert len(out) <= 303 and out.endswith("...")
    assert not out.rstrip(".").endswith(" ") and "  " not in out      # cut at a word, not mid-word
    assert trim_text("") == ""


def test_plain_summaries_falls_back_when_no_model_is_configured():
    moves = [{"kind": "sold", "symbol": "HUM", "exit_reason": "max hold 60d", "pnl": -22.59, "hold_days": 3},
             {"kind": "bought", "symbol": "DELL", "thesis": "rank 4, 6m mom 214%, 20d 21%"}]
    out = plain_summaries(moves, {"report": {}})
    assert set(out) == {"sold:HUM", "bought:DELL"}
    assert "reached the limit" in out["sold:HUM"] and "ranked 4" in out["bought:DELL"]
    assert plain_summaries([], {"report": {"summary_model": "claude-sonnet-5"}}) == {}


def test_build_report_covers_every_requested_section():
    text = build_report({
        "day": date(2026, 9, 17), "equity": 10195.82, "day_pl": 173.10, "day_pct": 0.0173, "cash": 5907.71,
        "positions": [{"symbol": "UMC", "qty": 45, "price": 24.5, "entry_price": 21.85, "day_pct": 0.087,
                       "total_pct": 0.121, "stop": 20.1, "filled_today": False}],
        "unrealized": 218.41,
        "closed_today": [{"symbol": "HUM", "qty": 1, "exit_price": 382.46, "pnl": -22.59, "pnl_pct": -0.0558,
                          "hold_days": 3, "exit_reason": "analyst REMOVE"}],
        "staged": [{"side": "buy", "symbol": "AMD", "qty": 1, "limit": 523.99, "stop": 481.08, "note": "rank 3"}],
        "perf": [{"period": "Since inception", "from": date(2026, 9, 14), "desk_pct": 0.0196,
                  "spy_pct": 0.0068, "diff_pct": 0.0128}],
        "movers": {"notable": [{"symbol": "UMC", "day_pct": 0.087, "total_pct": 0.121, "price": 24.5, "stop": 20.1}], "rest": 5},
        "news": [{"symbol": "UMC", "headline": "chip demand", "age_h": 5.0, "source": "x", "flags": []}],
        "changes": ["setting facts.exclude_earnings_within_days: 21 -> 5"],
        "halted": None, "notes": ["regime risk-on"], "now": NOW,
        "moves_plain": {"sold:HUM": "We closed it early because the reason we bought it stopped holding."},
    })
    for heading in ("RESULTS TODAY", "ORDERS WORKING", "NOTABLE MOVES", "NOTABLE NEWS",
                    "PERFORMANCE VS SPY", "CHANGES TO MODELS"):
        assert heading in text
    assert "sold HUM" in text and "-5.58%" in text
    assert "AMD" in text and "523.99" in text
    assert "+1.96%" in text and "+0.68%" in text
    assert "21 -> 5" in text
    assert "Not investment advice" in text
    assert "the reason we bought it stopped holding" in text     # the plain summary reaches the reader
    assert "14:00 PDT" in text and "will differ" in text          # snapshot time is stated, not implied


def test_build_report_handles_an_empty_halted_account():
    text = build_report({
        "day": date(2026, 9, 17), "equity": 8400.0, "day_pl": -900.0, "day_pct": -0.097, "cash": 8400.0,
        "positions": [], "unrealized": 0.0, "closed_today": [], "staged": [], "perf": [],
        "movers": {"notable": [], "rest": 0}, "news": [], "changes": [],
        "halted": "drawdown 16% from high-water mark", "notes": [],
    })
    assert "HALTED" in text and "no positions closed" in text
    assert "none since the last report" in text and "not enough history yet" in text


def _ctx(**over):
    base = {
        "day": date(2026, 9, 17), "equity": 10195.82, "day_pl": 173.10, "day_pct": 0.0173, "cash": 5907.71,
        "positions": [{"symbol": "UMC", "qty": 45, "price": 24.5, "entry_price": 21.85, "day_pct": 0.087,
                       "total_pct": 0.121, "stop": 20.1, "filled_today": False},
                      {"symbol": "DELL", "qty": 1, "price": 586.94, "entry_price": 569.83, "day_pct": 0.03,
                       "total_pct": 0.03, "stop": 496.03, "filled_today": True}],
        "unrealized": 218.41,
        "closed_today": [{"symbol": "HUM", "qty": 1, "exit_price": 382.46, "pnl": -22.59, "pnl_pct": -0.0558,
                          "hold_days": 3, "exit_reason": "analyst REMOVE: policy risk"}],
        # The summary is model-written prose built from headlines: the escaping guarantee has to hold here.
        "moves_plain": {"sold:HUM": "We closed it early. <script>alert(1)</script>",
                        "bought:DELL": "It was the 4th strongest name we screen."},
        "now": NOW,
        "staged": [{"side": "buy", "symbol": "AMD", "qty": 1, "limit": 523.99, "stop": 481.08, "note": "rank 3"},
                   {"side": "sell", "symbol": "TXG", "qty": 6, "limit": 0.0, "stop": 0.0, "note": "exit"}],
        "perf": [{"period": "Since inception", "from": date(2026, 9, 15), "desk_pct": 0.0196,
                  "spy_pct": 0.0070, "diff_pct": 0.0126}],
        "movers": {"notable": [{"symbol": "UMC", "day_pct": 0.087, "total_pct": 0.121, "price": 24.5, "stop": 20.1}],
                   "rest": 4},
        "news": [{"symbol": "UMC", "headline": "chip demand rises", "age_h": 5.0, "source": "x", "flags": ["offering"]}],
        "changes": ["setting facts.exclude_earnings_within_days: 21 -> 5"], "halted": None, "notes": ["regime risk-on"],
    }
    base.update(over)
    return base


def test_html_report_is_self_contained_and_escaped():
    from desk.report_html import build_html
    html = build_html(_ctx())
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    # Email clients strip these: nothing may depend on them.
    for banned in ("<script", "<link", "position:", "display:flex", "@media", "http://", "https://"):
        assert banned not in html, banned
    # Untrusted text (headlines, analyst prose) must be escaped, not rendered.
    assert "<script>alert" not in html and "&lt;script&gt;" in html
    # No raw entity leaks through the escaper into a number cell.
    assert "&amp;mdash;" not in html and "&amp;#" not in html


def test_html_report_shows_the_numbers_and_colours_them():
    from desk.report_html import NEG, POS, build_html
    html = build_html(_ctx())
    for want in ("$10,195.82", "+1.73%", "+1.96%", "+0.70%", "+1.26%", "HUM", "-5.58%", "AMD", "523.99",
                 "21 -&gt; 5", "4 other holdings moved less than 3%"):
        assert want in html, want
    assert POS in html and NEG in html                      # gains and losses are coloured differently
    assert "market" in html                                 # the sell order has no limit price
    assert "TeamRobo daily 2026-09-17" in html
    assert "4th strongest name" in html                     # a buy is explained, not just listed
    assert "14:00 PDT" in html


def test_html_report_handles_an_empty_halted_account():
    from desk.report_html import build_html
    html = build_html(_ctx(positions=[], closed_today=[], staged=[], perf=[], movers={"notable": [], "rest": 0},
                           news=[], changes=[], notes=[], unrealized=0.0, halted="drawdown 16%"))
    assert "HALTED" in html and "drawdown 16%" in html
    assert "not enough history yet" in html and "no orders resting" in html
    assert "no change to config, models or code" in html


def test_staged_orders_show_whole_shares_and_never_cut_a_word():
    """The 09-17 email read 'BUY HPE x6.0' and ended the note '...$16.1M of 30-day insider sel'."""
    from desk.report_html import build_html
    note = ("rank 7, 6m mom 159%, 20d 15% | analyst REDUCE: Missing earnings date that plausibly lands at the "
            "tail of a 60-day hold, 84% realized vol against a stop only ~13% away, and $16.1M of 30-day insider "
            "selling across several officers in the last month.")
    ctx = _ctx(staged=[{"side": "buy", "symbol": "HPE", "qty": 6.0, "limit": 61.35, "stop": 52.79, "note": note}])
    text, html = build_report(ctx), build_html(ctx)
    line = next(ln for ln in text.splitlines() if "HPE" in ln and "limit" in ln)
    assert "x6 " in line and "x6.0" not in line
    shown = line.split("52.79", 1)[1].strip()
    words = set(note.replace("...", "").split())
    assert shown.endswith("...") and all(w in words for w in shown.rstrip(".").split()), shown
    assert "insider sel " not in text and "insider sel..." not in html


def test_a_trim_is_reported_as_a_partial_sale_not_a_closed_position(tmp_path, monkeypatch):
    """Selling 3 of 45 UMC shares to get back under the cap must not read 'we closed the position'."""
    from datetime import datetime, timezone
    from types import SimpleNamespace
    import report_daily
    monkeypatch.setattr(report_daily, "STATE_DIR", tmp_path)
    order = SimpleNamespace(side=SimpleNamespace(value="sell"), symbol="UMC", filled_avg_price=24.10,
                            filled_qty="3", qty="3", filled_at=datetime(2026, 9, 25, 13, 30, tzinfo=timezone.utc))
    st = {"positions": {"UMC": {"entry_price": 21.85, "entry_date": "2026-09-14"}}}
    row = report_daily.closed_today(date(2026, 9, 25), [order], st, {"UMC"})[0]
    assert row["qty"] == 3 and row["exit_reason"].startswith("partial sale")
    s = fallback_summary("sold", row)
    assert "part of the position" in s and "closed" not in s
    # Fully gone from the account: still an ordinary broker-side close.
    assert report_daily.closed_today(date(2026, 9, 25), [order], st, set())[0]["exit_reason"] == "closed at broker"


def test_manual_actions_lead_the_email():
    from desk.report_html import build_html
    ctx = _ctx(manual=["UMC is 10.6% of equity, above the 10% cap - sell 3 share(s) to get back under"])
    text, html = build_report(ctx), build_html(ctx)
    assert text.index("NEEDS YOUR ACTION") < text.index("RESULTS TODAY") and "sell 3 share(s)" in text
    assert html.index("Needs your action") < html.index("Performance vs SPY")

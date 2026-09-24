from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from desk.edgar import offering_filings, parse_form4, recent_8k_items, shelf_filings
from desk.facts import (candidate_blockers, insider_tilt, parse_occ, size_multiplier, social_rate,
                        summarize_chain, zscore)
from desk.regime import current_regime, regime_scale_series
from desk.signals import compute_signals, rerank

FORM4 = b"""<?xml version="1.0"?>
<ownershipDocument>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-09-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>25.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-09-02</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>200</value></transactionShares>
        <transactionPricePerShare><value>26.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>5000</value></transactionShares>
        <transactionPricePerShare><value>1.00</value></transactionPricePerShare></transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def test_parse_form4_counts_open_market_only():
    p = parse_form4(FORM4)
    assert p["buys"] == 25500.0 and p["sells"] == 5200.0 and p["n"] == 2
    assert p["net_usd"] == 20300.0
    assert [t["code"] for t in p["transactions"]] == ["P", "S"]


def test_offering_and_8k_detection():
    filings = [
        {"form": "8-K", "date": date(2026, 9, 1), "items": "2.02,9.01", "accession": "a", "doc": "d"},
        {"form": "8-K", "date": date(2026, 9, 3), "items": "3.02", "accession": "b", "doc": "d"},
        {"form": "424B5", "date": date(2026, 9, 4), "items": "", "accession": "c", "doc": "d"},
        {"form": "10-Q", "date": date(2026, 8, 1), "items": "", "accession": "e", "doc": "d"},
    ]
    hits = offering_filings(filings)
    # 424B5 is a shelf takedown (may be debt): flagged for the analyst, never a hard block.
    assert any("3.02" in h for h in hits) and len(hits) == 1
    assert not any("424B5" in h for h in hits)
    assert shelf_filings(filings) == ["424B5 2026-09-04"]
    assert offering_filings([{"form": "S-1", "date": date(2026, 9, 5), "items": "", "accession": "f", "doc": "d"}])
    assert offering_filings(filings[:1]) == []
    assert len(recent_8k_items(filings)) == 2


def test_candidate_blockers(cfg):
    fcfg = cfg["facts"]
    inside = fcfg["exclude_earnings_within_days"] - 1
    assert candidate_blockers({"earnings_in_days": inside}, fcfg) == [f"earnings in {inside}d"]
    assert candidate_blockers({"earnings_in_days": fcfg["exclude_earnings_within_days"] + 1}, fcfg) == []
    assert candidate_blockers({"earnings_in_days": None}, fcfg) == []
    assert any("dilutive" in r for r in candidate_blockers({"offering_filings": ["S-1 2026-09-01"]}, fcfg))
    # A shelf filing alone is an analyst flag, never a hard block.
    assert candidate_blockers({"shelf_filings": ["S-3ASR 2026-08-13"]}, fcfg) == []
    assert any("acquisition" in r for r in candidate_blockers({"cls_acquisition_target": True}, fcfg))
    assert any("social" in r for r in candidate_blockers({"social_z": 3.5}, fcfg))
    assert candidate_blockers({"social_z": 2.0, "cls_offering": False}, fcfg) == []


def test_size_multiplier_and_tilt(cfg):
    fcfg = cfg["facts"]
    assert size_multiplier({"iv_over_rv": 2.0}, fcfg) == 0.5
    assert size_multiplier({"iv_over_rv": 1.1}, fcfg) == 1.0
    assert size_multiplier({}, fcfg) == 1.0
    assert insider_tilt({"insider_net_buy_30d": 500000}, fcfg) == 0.0  # default weight is 0: unbacktested
    assert insider_tilt({"insider_net_buy_30d": 500000}, dict(fcfg, insider_tilt_weight=0.1)) == 0.1
    assert insider_tilt({"insider_net_buy_30d": 1000}, dict(fcfg, insider_tilt_weight=0.1)) == 0.0


def test_occ_and_chain_summary():
    assert parse_occ("AAPL260918C00150000") == ("AAPL", date(2026, 9, 18), "C", 150.0)
    assert parse_occ("not-an-option") is None
    spot = 100.0
    chain = {
        "XYZ261016C00100000": SimpleNamespace(implied_volatility=0.40),
        "XYZ261016P00100000": SimpleNamespace(implied_volatility=0.42),
        "XYZ261016P00090000": SimpleNamespace(implied_volatility=0.55),
        "XYZ261016C00120000": SimpleNamespace(implied_volatility=0.30),
        "XYZ261016P00095000": SimpleNamespace(implied_volatility=None),
    }
    s = summarize_chain(chain, spot)
    assert s["atm_iv"] == 0.41 and abs(s["put_skew"] - 0.14) < 1e-9
    assert summarize_chain({}, spot) == {"atm_iv": None, "put_skew": None}


def test_social_rate_and_zscore():
    now = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    msgs = [{"created_at": (now - timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "entities": {"sentiment": {"basic": "Bullish" if h % 3 else "Bearish"}}} for h in range(1, 7)]
    r = social_rate(msgs, now)
    assert r["social_msgs_per_day"] == 24.0  # 6 messages over 6 hours
    assert abs(r["social_bullish_share"] - 4 / 6) < 1e-9
    assert social_rate([], now)["social_msgs_per_day"] == 0.0
    assert zscore([10] * 10, 50) is None  # too few observations
    assert zscore([10, 12, 8, 11, 9] * 4, 11) is not None
    assert zscore([10] * 20, 50) is None  # zero variance


def test_regime_gate():
    idx = pd.bdate_range("2024-01-01", periods=300)
    spy = pd.Series(np.linspace(400, 500, 300), index=idx)
    spy.iloc[-1] = 380  # crash below the MA on the last day
    vix = pd.Series(15.0, index=idx)
    rcfg = {"spy_ma_days": 200, "vix_risk_off": 30, "risk_off_scale": 0.5}
    scale = regime_scale_series(spy, vix, rcfg)
    assert scale.iloc[-2] == 1.0 and scale.iloc[-1] == 0.5
    assert (scale.iloc[:199] == 1.0).all()  # no MA yet -> no gate
    vix.iloc[-2] = 40
    assert regime_scale_series(spy, vix, rcfg).iloc[-2] == 0.5
    r = current_regime(spy, vix, rcfg)
    assert r["label"] == "risk-off" and r["scale"] == 0.5
    assert regime_scale_series(spy, None, rcfg).iloc[-1] == 0.5


def test_rerank_blocks_and_tilts(cfg, panel):
    snap = compute_signals(panel, cfg["signals"])
    top = snap.index[0]
    second = snap.index[1]
    out = rerank(snap, tilt=None, blocked={top: "earnings"})
    assert not out.loc[top, "eligible"] and np.isnan(out.loc[top, "rank"])
    assert out.loc[second, "rank"] == 1
    out2 = rerank(snap, tilt={second: 5.0}, blocked=None)
    assert out2.loc[second, "rank"] == 1 and out2.loc[top, "rank"] == 2
    el = out2[out2["eligible"]]
    assert sorted(el["rank"].tolist()) == list(range(1, len(el) + 1))


def test_failed_earnings_lookup_blocks_the_candidate_and_is_not_cached(tmp_path, monkeypatch):
    """None used to mean both 'no report in the window' and 'the lookup failed', so a Finnhub timeout let a
    candidate reporting in 3 days straight through the hard block - and the cached sheet kept it that way."""
    import io
    import json
    from datetime import date
    import pandas as pd
    from desk import facts as F
    fb = F.FactBuilder.__new__(F.FactBuilder)
    fb.cfg = {"facts": {"exclude_earnings_within_days": 5, "classifier_model": "m", "social_z_exclude_at": 3.0},
              "exits": {"max_hold_days": 60}}
    fb.fcfg, fb.today, fb.earnings_failed = fb.cfg["facts"], date(2026, 9, 24), set()
    fb.cache_dir, fb.cache_file, fb.social_file = tmp_path, tmp_path / "facts.json", tmp_path / "social.json"
    monkeypatch.setenv("FINNHUB_API_KEY", "k")

    def fake_urlopen(url, timeout=30):
        if "symbol=BAD" in url:
            raise TimeoutError("finnhub timed out")
        return io.BytesIO(json.dumps({"earningsCalendar": [{"date": "2026-11-05"}]}).encode())

    monkeypatch.setattr(F.urllib.request, "urlopen", fake_urlopen)
    for m in ("_news", "_options", "_social", "_asset"):
        monkeypatch.setattr(fb, m, lambda *a, **k: {})
    monkeypatch.setattr(fb, "_edgar", lambda s, spot=None: {})
    monkeypatch.setattr(F, "classify_headlines", lambda by, model: {})

    out = fb.build(["OK", "BAD"], pd.DataFrame({"OK": [1.0], "BAD": [1.0]}))
    assert F.candidate_blockers(out["BAD"], fb.fcfg) == ["earnings date lookup failed - not bought blind"]
    assert F.candidate_blockers(out["OK"], fb.fcfg) == []           # a real 'no report soon' still passes
    cached = json.loads(fb.cache_file.read_text())
    assert "OK" in cached and "BAD" not in cached                   # a same-day re-run retries BAD


def test_form4_amendment_does_not_double_count_the_same_trade(monkeypatch):
    """A 4/A restating an $8M sale used to report -$16M of insider selling - the figure the analyst cites
    most often when it REDUCEs an entry."""
    from datetime import date
    from desk import edgar as E
    sale = {"code": "S", "usd": 8_000_000.0, "price": 100.0, "shares": 80_000, "date": "2026-09-10"}
    buy = {"code": "P", "usd": 50_000.0, "price": 100.0, "shares": 500, "date": "2026-09-12"}
    docs = {"a1": [sale], "a2": [sale], "a3": [buy]}           # a2 is the 4/A of a1
    monkeypatch.setattr(E, "parse_form4", lambda xml: {"transactions": docs[xml.decode()]})
    ed = E.Edgar.__new__(E.Edgar)
    monkeypatch.setattr(ed, "_get", lambda url, name, ttl_hours=0: name[3:5].encode(), raising=False)
    filings = [{"form": "4", "date": date(2026, 9, 11), "accession": "a1", "doc": "x.xml"},
               {"form": "4/A", "date": date(2026, 9, 15), "accession": "a2", "doc": "x.xml"},
               {"form": "4", "date": date(2026, 9, 13), "accession": "a3", "doc": "x.xml"}]
    out = ed.form4_activity(1, filings, date(2026, 6, 1))
    assert out["sells"] == 8_000_000.0 and out["buys"] == 50_000.0 and out["n"] == 2
    assert out["net_usd"] == 50_000.0 - 8_000_000.0


def test_roundup_headlines_about_other_companies_are_dropped():
    from desk.facts import headline_is_about
    # 2026-09-23: Broadcom news under INTC, tagged with a dozen tickers.
    assert not headline_is_about("INTC", "Intel Corporation Common Stock",
                                 "Broadcom, Melania Trump And Other Big Stocks Moving Higher", ["AVGO", "INTC", "QCOM"])
    assert headline_is_about("INTC", "Intel Corporation Common Stock", "Intel wins foundry deal", ["INTC", "AAPL"])
    assert headline_is_about("INTC", None, "Why INTC shares are rising", ["INTC", "NVDA"])
    assert headline_is_about("INTC", None, "Chipmaker beats estimates", ["INTC"])          # tagged to it alone
    # A generic first word must not match everyone's headlines.
    assert not headline_is_about("AMD", "Advanced Micro Devices, Inc.", "Advanced packaging demand soars", ["TSM", "AMD"])
    assert headline_is_about("AMD", "Advanced Micro Devices, Inc.", "Advanced Micro Devices raises guidance", ["AMD", "NVDA"])
    assert not headline_is_about("TEAM", "Atlassian Corporation", "Team USA wins gold", ["TEAM", "NKE"])


def _fb_with_edgar(tmp_path, transactions):
    from datetime import date
    from desk import facts as F
    fb = F.FactBuilder.__new__(F.FactBuilder)
    fb.today = date(2026, 9, 24)
    fb.fcfg = {"insider_max_abs_usd": 50_000_000, "insider_price_band": 3.0}

    class Ed:
        def cik(self, s):
            return 1

        def filings(self, cik, since):
            return []

        def form4_activity(self, cik, filings, since):
            return {"transactions": transactions}
    fb.edgar = Ed()
    return fb


def test_an_insider_feed_with_every_row_excluded_is_unknown_not_zero(tmp_path):
    rows = [{"code": "S", "usd": 1e9, "price": 3.0, "shares": 1, "filed": "2026-09-20"}]      # off-market artifact
    out = _fb_with_edgar(tmp_path, rows)._edgar("UMC", spot=24.0)
    assert out["insider_signal"] == "unavailable" and out["insider_net_buy_30d"] is None
    assert _fb_with_edgar(tmp_path, [])._edgar("X", spot=10.0)["insider_signal"] == "no Form 4 trades"


def test_plan_sales_are_kept_out_of_the_insider_signal(tmp_path):
    rows = [{"code": "S", "usd": 9_000_000, "price": 100.0, "shares": 90_000, "filed": "2026-09-20", "plan": True},
            {"code": "S", "usd": 200_000, "price": 100.0, "shares": 2_000, "filed": "2026-09-21", "plan": False}]
    out = _fb_with_edgar(tmp_path, rows)._edgar("DELL", spot=100.0)
    assert out["insider_net_buy_30d"] == -200_000 and out["insider_plan_sales_30d"] == 9_000_000


def test_form4_parser_reads_the_10b5_1_checkbox():
    xml = (b"<ownershipDocument><aff10b5One>1</aff10b5One><nonDerivativeTable><nonDerivativeTransaction>"
           b"<transactionDate><value>2026-09-10</value></transactionDate><transactionCoding><transactionCode>S"
           b"</transactionCode></transactionCoding><transactionAmounts><transactionShares><value>10</value>"
           b"</transactionShares><transactionPricePerShare><value>5</value></transactionPricePerShare>"
           b"</transactionAmounts></nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>")
    assert parse_form4(xml)["transactions"][0]["plan"] is True
    assert parse_form4(xml.replace(b"<aff10b5One>1</aff10b5One>", b""))["transactions"][0]["plan"] is False

import numpy as np
import pandas as pd

from audit import post_exit_return, trade_stats


def _closes():
    idx = pd.bdate_range("2026-06-01", periods=80)
    return pd.DataFrame({"AAA": np.linspace(100, 140, 80), "BBB": np.linspace(50, 30, 80)}, index=idx)


def test_post_exit_return():
    closes = _closes()
    exit_date = closes.index[10]
    r = post_exit_return(closes, "AAA", exit_date, float(closes["AAA"].iloc[10]), 20)
    assert abs(r - (closes["AAA"].iloc[30] / closes["AAA"].iloc[10] - 1)) < 1e-9
    assert post_exit_return(closes, "ZZZ", exit_date, 100.0, 20) is None
    assert post_exit_return(closes, "AAA", closes.index[-5], 100.0, 20) is None  # not enough history yet


def test_trade_stats_by_reason():
    closes = _closes()
    trades = pd.DataFrame([
        {"symbol": "AAA", "exit_date": closes.index[5].date(), "exit_price": float(closes["AAA"].iloc[5]),
         "pnl": 50.0, "pnl_pct": 0.05, "hold_days": 20, "exit_reason": "signal decay"},
        {"symbol": "AAA", "exit_date": closes.index[8].date(), "exit_price": float(closes["AAA"].iloc[8]),
         "pnl": 30.0, "pnl_pct": 0.03, "hold_days": 61, "exit_reason": "max hold 60d"},
        {"symbol": "BBB", "exit_date": closes.index[5].date(), "exit_price": float(closes["BBB"].iloc[5]),
         "pnl": -40.0, "pnl_pct": -0.08, "hold_days": 4, "exit_reason": "analyst REMOVE: guidance cut"},
    ])
    s = trade_stats(trades, closes, 20)
    assert s["n_trades"] == 3 and abs(s["win_rate"] - 2 / 3) < 1e-3
    assert s["sum_pnl_usd"] == 40.0
    by = s["by_exit_reason"]
    assert set(by) == {"signal decay", "max hold 60d", "analyst remove"}
    assert by["signal decay"]["avg_post_exit_20d_ret"] > 0      # we sold a winner that kept going
    assert by["analyst remove"]["avg_post_exit_20d_ret"] < 0    # the veto was right
    assert trade_stats(pd.DataFrame(), None, 20) == {"n_trades": 0}


def test_trade_stats_reports_each_post_exit_window():
    closes = _closes()
    trades = pd.DataFrame([{"symbol": "AAA", "exit_date": closes.index[5].date(), "exit_price": float(closes["AAA"].iloc[5]),
                            "pnl": 50.0, "pnl_pct": 0.05, "hold_days": 20, "exit_reason": "signal decay"}])
    row = trade_stats(trades, closes, [10, 20])["by_exit_reason"]["signal decay"]
    assert row["avg_post_exit_10d_ret"] > 0 and row["avg_post_exit_20d_ret"] > 0
    assert row["n_post_exit_10d"] == row["n_post_exit_20d"] == 1


# ------------------------------------------------------------------ the Saturday email

import json as _json
from datetime import date as _date

import audit as A

OUT = {"summary": "Six trades <b>is</b> too few.", "sample_size_warning": "n=6",
       "findings": [{"severity": "high", "title": "Half the book is one trade", "evidence": "AMD, DELL"},
                    {"severity": "high", "title": "Censored sample", "evidence": "losers close first"},
                    {"severity": "low", "title": "Noisy alerts", "evidence": "UMC x5"}],
       "proposed_changes": [{"config_key": "audit.post_exit_days", "current": 20, "proposed": 10,
                             "rationale": "fields empty", "backtest_check": "none"}]}


def _cfg(email=True):
    return {"audit": {"email": email}, "report": {"to": "owner@example.com"}}


def test_the_audit_email_names_its_findings_and_escapes_the_model_text():
    assert A.audit_subject(_date(2026, 10, 3), OUT) == \
        "TeamRobo weekly audit 2026-10-03 | 2 HIGH, 1 LOW | 1 proposed change"
    html = A.audit_html(_date(2026, 10, 3), OUT, {"n_trades": 6}, "TRIALS")
    assert "&lt;b&gt;is&lt;/b&gt;" in html and "<b>is</b>" not in html
    assert "audit.post_exit_days" in html and "Half the book is one trade" in html


def test_an_audit_is_emailed_once_and_a_failed_send_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "STATE_DIR", tmp_path)
    sent, result = [], {"status": "send failed: SMTPAuthenticationError"}
    monkeypatch.setattr(A, "send_email", lambda subj, body, cfg, html=None: sent.append(subj) or result["status"])
    day = _date(2026, 10, 3)
    assert A.email_audit(day, OUT, {}, "text", _cfg()).startswith("send failed")
    assert not A.sent_marker(day).exists()                      # not marked: the next attempt still sends
    result["status"] = "emailed owner@example.com via smtp"
    assert A.email_audit(day, OUT, {}, "text", _cfg()).startswith("emailed")
    assert A.email_audit(day, OUT, {}, "text", _cfg()).startswith("already emailed")
    assert A.email_audit(day, OUT, {}, "text", _cfg(), resend=True).startswith("emailed")
    assert len(sent) == 3
    assert A.email_audit(day, OUT, {}, "text", _cfg(email=False)) == "email off (audit.email)"


def test_a_saved_audit_can_be_emailed_without_running_it_again(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "STATE_DIR", tmp_path)
    sent = []
    monkeypatch.setattr(A, "send_email", lambda subj, body, cfg, html=None: sent.append((subj, body, html)) or "emailed x")
    d = tmp_path / "audits"
    d.mkdir()
    (d / "2026-09-26.md").write_text("WEEKLY AUDIT 2026-09-26")              # before the .json was saved
    (d / "2026-10-03.md").write_text("WEEKLY AUDIT 2026-10-03")
    (d / "2026-10-03.json").write_text(_json.dumps({"audit": OUT, "stats": {"n_trades": 6}, "trials": "T"}))
    assert A.email_saved(_cfg(), None, False) == "emailed x"
    assert sent[-1][0].endswith("1 proposed change") and sent[-1][1] == "WEEKLY AUDIT 2026-10-03" and sent[-1][2]
    assert A.email_saved(_cfg(), "2026-09-26", False) == "emailed x"
    assert sent[-1][0] == "TeamRobo weekly audit 2026-09-26" and sent[-1][2] is None      # text only
    assert A.email_saved(_cfg(), "2026-01-01", False).startswith("no saved audit")

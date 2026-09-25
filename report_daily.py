#!/usr/bin/env python
"""Daily activity report, emailed after the desk job has finished.

  python report_daily.py              build, save, email, webhook
  python report_daily.py --print      build and print only (no email, no state written)

Read-only with respect to trading: it never submits, cancels or modifies an order.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest

from desk.config import alpaca_keys, load_config, require_paper
from desk.data import fetch_daily_bars
from desk.notify import send, send_email
from desk.report import (build_report, diff_fingerprint, fingerprint, holdings_at_close, load_json, movers,
                         notable_news, performance, plain_summaries, quote_url)
from desk.report_html import build_html
from desk import telemetry
from desk import trials as T
from desk.evidence import live_vs_backtest
from desk.state import STATE_DIR, atomic_write_text, load_state

log = logging.getLogger("desk.report")
DEAD = {"filled", "canceled", "expired", "rejected", "replaced", "done_for_day", "stopped", "suspended"}


def equity_series(trading: TradingClient) -> pd.Series:
    h = trading.get_portfolio_history(GetPortfolioHistoryRequest(period="1A", timeframe="1D"))
    idx = [datetime.fromtimestamp(t, tz=timezone.utc).date() for t in (h.timestamp or [])]
    vals = [float(e) for e in (h.equity or [])]
    s = pd.Series(vals, index=pd.to_datetime(idx)).dropna()
    return s[s > 0]


def closed_today(today: date, orders: list, st: dict, still_held: set = frozenset()) -> list:
    """trades.csv is written by the desk job; a sale that filled since its last run is picked up
    from the broker so the report never silently omits an exit. A sale of a name that is still held is
    a trim, and is labelled as one rather than as a closed position."""
    rows, seen = [], set()
    p = STATE_DIR / "trades.csv"
    if p.exists():
        with open(p) as f:
            for r in csv.DictReader(f):
                if r.get("exit_date") == today.isoformat():
                    seen.add(r["symbol"])
                    rows.append({**r, "qty": float(r["qty"]), "exit_price": float(r["exit_price"]),
                                 "pnl": float(r["pnl"]), "pnl_pct": float(r["pnl_pct"]),
                                 "hold_days": int(float(r["hold_days"]))})
    for o in orders:
        if (o.side.value != "sell" or not o.filled_at or o.filled_at.date() != today
                or o.symbol in seen or not o.filled_avg_price):
            continue
        pos = st["positions"].get(o.symbol, {})
        entry = float(pos.get("entry_price") or 0)
        px, qty = float(o.filled_avg_price), float(o.filled_qty or o.qty)
        held = (today - date.fromisoformat(pos["entry_date"])).days if pos.get("entry_date") else 0
        rows.append({"symbol": o.symbol, "qty": qty, "exit_price": px,
                     "pnl": (px - entry) * qty if entry else 0.0,
                     "pnl_pct": (px / entry - 1) if entry else 0.0, "hold_days": held,
                     "exit_reason": ("partial sale - position still held" if o.symbol in still_held
                                     else pos.get("pending_exit_reason") or "closed at broker"),
                     "thesis": pos.get("thesis", "")})
    return rows


def health_today(today: date) -> dict:
    """The agent monitor's summary if it ran today. A missing check is shown as missing, not as healthy."""
    h = load_json(STATE_DIR / "monitor" / "latest.json", {})
    if h.get("date") != today.isoformat():
        return {"status": "missing", "summary": "The agent monitor did not run today - no health check.",
                "findings": [], "recommendations": []}
    return h


def evidence_today(cfg: dict, equity: pd.Series) -> dict:
    """Live volatility and drawdown against the current config's backtest, plus whether the reference
    still describes the config that is running."""
    ref = load_json(Path(cfg["_root"]) / "state" / "history" / "live_reference.json", None)
    ev = live_vs_backtest(equity, ref)
    if ref:
        ev["stale_reference"] = ref.get("config_id") != T.current_config_id(cfg)
    return ev


def research_line(cfg: dict, seen: int | None) -> tuple:
    """(line or None, runs now). Shown only when backtests were logged since the last email."""
    log = T.load(Path(cfg["_root"]) / "state" / "history" / "trials.jsonl")
    if seen is None or len(log) <= seen:
        return None, len(log)
    s = T.summary(log, int((cfg.get("backtest") or {}).get("prior_untracked_trials", 0)))
    cur = s.get("current")
    line = f"{len(log) - seen} new backtest run(s) since the last email"
    if cur:
        line += f"; current config Deflated Sharpe {cur['dsr']:.2f} (N = {s['n_trials']}, bar 0.95)"
    lag = T.readme_lag(Path(cfg["_root"]) / "README.md", log)
    if lag:
        line += f"; README table {lag} run(s) behind"
    return line, len(log)


def brief_notes(today: date) -> list:
    p = STATE_DIR / "briefs" / f"{today.isoformat()}.md"
    if not p.exists():
        return []
    out, seen = [], False
    for line in p.read_text().splitlines():
        if line.startswith("Notes:"):
            seen = True
            continue
        if seen and line.startswith("  - "):
            out.append(line[4:])
        elif seen and line and not line.startswith(" "):
            break
    return out


def deliver(today: date, subject: str, text: str, html: str, cfg: dict, markers: dict) -> str:
    """Save the report, email it, then write `markers` ({path: text}: the change-log fingerprint and the
    research counter). The markers advance only once a report went out, or mail is deliberately off: they
    were written before sending, so a failed send lost that day's "what changed" from every later email."""
    out = STATE_DIR / "reports"
    atomic_write_text(out / f"{today.isoformat()}.txt", text)
    atomic_write_text(out / f"{today.isoformat()}.html", html)
    status = send_email(subject, text, cfg, html=html)
    log.info("report: %s", status)
    if "FAILED" not in status:
        for path, body in markers.items():
            atomic_write_text(path, body)
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="print_only", action="store_true", help="build and print; send nothing, save nothing")
    ap.add_argument("--html-out", default=None, help="with --print: also write the HTML to this path")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = load_config()
    root = Path(cfg["_root"])
    today = date.today()
    key, secret = alpaca_keys()
    trading = TradingClient(key, secret, paper=require_paper(cfg))
    data = StockHistoricalDataClient(key, secret)
    st = load_state()

    acct = trading.get_account()
    equity, last_equity = float(acct.equity), float(acct.last_equity)
    live = trading.get_all_positions()
    orders = trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, nested=False, limit=500,
                                                 after=datetime.now() - timedelta(days=5)))
    filled_today = {o.symbol for o in orders if o.filled_at and o.filled_at.date() == today and o.side.value == "buy"}

    positions = []
    for p in live:
        sp = st["positions"].get(p.symbol, {})
        positions.append({
            "symbol": p.symbol, "qty": float(p.qty), "price": float(p.current_price),
            "entry_price": float(p.avg_entry_price), "day_pct": float(p.unrealized_intraday_plpc or 0),
            "total_pct": float(p.unrealized_plpc or 0), "stop": sp.get("stop"),
            "filled_today": p.symbol in filled_today, "thesis": sp.get("thesis", ""), "entry_date": sp.get("entry_date"),
        })

    staged = []
    for o in orders:
        if o.status.value in DEAD or o.side.value == "buy" and not o.limit_price:
            continue
        if o.side.value == "buy":
            sp = st["positions"].get(o.symbol, {})
            staged.append({"side": "buy", "symbol": o.symbol, "qty": float(o.qty), "limit": float(o.limit_price),
                           "stop": float(sp.get("stop") or 0), "note": sp.get("thesis") or ""})
        elif not o.stop_price and not o.limit_price:
            staged.append({"side": "sell", "symbol": o.symbol, "qty": float(o.qty), "limit": 0.0, "stop": 0.0,
                           "note": st["positions"].get(o.symbol, {}).get("pending_exit_reason") or "exit"})

    eq_series = equity_series(trading)
    # Portfolio history's last daily point lags the live account, which would report a stale return.
    eq_series.loc[pd.Timestamp(today)] = equity
    eq_series = eq_series.sort_index()
    try:
        spy = fetch_daily_bars(data, ["SPY"], 420, cfg["data"]["feed"])["close"]["SPY"].dropna()
    except Exception as e:
        log.warning("SPY unavailable: %s", e)
        spy = pd.Series(dtype=float)
    perf = performance(eq_series, spy, today)

    facts = load_json(root / cfg["data"]["cache_dir"] / f"facts_{today.isoformat()}.json", {})
    fp_path = STATE_DIR / "fingerprint.json"
    new_fp = fingerprint(root)
    changes = diff_fingerprint(load_json(fp_path, {}), new_fp)

    seen_path = STATE_DIR / "reports" / "research_seen.json"
    research = research_line(cfg, load_json(seen_path, {}).get("runs"))
    closed = closed_today(today, orders, st, {p["symbol"] for p in positions})
    held = [p["symbol"] for p in positions]
    try:
        closes = fetch_daily_bars(data, held, 10, cfg["data"]["feed"])["close"] if held else None
    except Exception as e:
        log.warning("closing prices unavailable, holdings marked at the live price: %s", e)
        closes = None
    news = notable_news(facts, held)
    tickers = set(held) | {s["symbol"] for s in staged} | {c["symbol"] for c in closed} | {n["symbol"] for n in news} | {"SPY"}
    links = {}
    for sym in sorted(tickers):
        try:
            url = quote_url(sym, trading.get_asset(sym).exchange.value)
        except Exception:
            url = None
        if url:
            links[sym] = url
    moves = ([{"kind": "sold", **c} for c in closed]
             + [{"kind": "bought", **p} for p in positions if p.get("filled_today")])
    holdings = holdings_at_close(positions, closes, float(acct.cash))

    ctx = {
        "day": today, "now": datetime.now().astimezone(), "equity": equity, "day_pl": equity - last_equity,
        "day_pct": (equity / last_equity - 1) if last_equity else 0.0,
        # One unrealized figure in the whole email: the holdings table's, at the official close. The broker's
        # unrealized_pl is at the after-hours mark and put a second, different number two lines above it.
        "cash": float(acct.cash), "positions": positions, "unrealized": holdings["pnl"],
        "closed_today": closed, "staged": staged, "perf": perf,
        "moves_plain": plain_summaries(moves, cfg),
        "movers": movers(positions), "news": news, "quote_links": links,
        "holdings": holdings,
        "changes": changes, "halted": st.get("halt_reason") if st.get("halted") else None,
        "notes": brief_notes(today), "health": health_today(today),
        "evidence": evidence_today(cfg, eq_series), "research": research[0],
        "manual": [n.split(":", 1)[1].strip() for n in brief_notes(today) if n.startswith("MANUAL ACTION")],
    }
    text, html = build_report(ctx), build_html(ctx)

    telemetry.note(print_only=args.print_only, moves=len(moves))
    if args.print_only:
        print(text)
        if args.html_out:
            Path(args.html_out).write_text(html)
            print(f"\n[html written to {args.html_out}]")
        return
    subject = (f"TeamRobo daily {today.isoformat()} | {equity:,.0f} ({equity / last_equity - 1:+.2%})"
               if last_equity else f"TeamRobo daily {today.isoformat()}")
    status = deliver(today, subject, text, html, cfg, {fp_path: json.dumps(new_fp, indent=1, sort_keys=True, default=str),
                                                        seen_path: json.dumps({"runs": research[1]})})
    telemetry.note(email="sent" if status.startswith("emailed") else "failed" if "FAILED" in status else "skipped",
                   email_detail=status)
    print(text)
    send(f"[daily report] {status}", cfg)


if __name__ == "__main__":
    telemetry.run_job("report", main)

#!/usr/bin/env python
"""Daily activity report, emailed after the desk job has finished.

  python report_daily.py              build, save, email, webhook
  python report_daily.py --print      build and print only (no email, no state written)

Read-only with respect to trading: it never submits, cancels or modifies an order.
"""
from __future__ import annotations

import argparse
import csv
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest

from desk.config import alpaca_keys, load_config
from desk.data import fetch_daily_bars
from desk.notify import send, send_email
from desk.report import (build_report, diff_fingerprint, fingerprint, load_json, movers, notable_news,
                         performance, plain_summaries)
from desk.report_html import build_html
from desk import telemetry
from desk.state import STATE_DIR, load_state

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
    trading = TradingClient(key, secret, paper=cfg["account"]["paper"])
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
            "filled_today": p.symbol in filled_today, "thesis": sp.get("thesis", ""),
        })
    unrealized = sum(float(p.unrealized_pl or 0) for p in live)

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

    closed = closed_today(today, orders, st, {p["symbol"] for p in positions})
    moves = ([{"kind": "sold", **c} for c in closed]
             + [{"kind": "bought", **p} for p in positions if p.get("filled_today")])

    ctx = {
        "day": today, "now": datetime.now().astimezone(), "equity": equity, "day_pl": equity - last_equity,
        "day_pct": (equity / last_equity - 1) if last_equity else 0.0,
        "cash": float(acct.cash), "positions": positions, "unrealized": unrealized,
        "closed_today": closed, "staged": staged, "perf": perf,
        "moves_plain": plain_summaries(moves, cfg),
        "movers": movers(positions), "news": notable_news(facts, [p["symbol"] for p in positions]),
        "changes": changes, "halted": st.get("halt_reason") if st.get("halted") else None,
        "notes": brief_notes(today), "health": health_today(today),
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
    out = STATE_DIR / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{today.isoformat()}.txt").write_text(text)
    (out / f"{today.isoformat()}.html").write_text(html)
    fp_path.write_text(__import__("json").dumps(new_fp, indent=1, sort_keys=True, default=str))
    status = send_email(f"TeamRobo daily {today.isoformat()} | {equity:,.0f} ({equity / last_equity - 1:+.2%})"
                        if last_equity else f"TeamRobo daily {today.isoformat()}", text, cfg, html=html)
    log.info("report: %s", status)
    telemetry.note(email="sent" if status.startswith("emailed") else "failed" if "FAILED" in status else "skipped",
                   email_detail=status)
    print(text)
    send(f"[daily report] {status}", cfg)


if __name__ == "__main__":
    telemetry.run_job("report", main)

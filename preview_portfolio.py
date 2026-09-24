#!/usr/bin/env python
"""Keyless preview: run the signal engine, regime gate, sizing, and correlation filter on public Yahoo data
for the S&P 500 + S&P 400 universe and print the portfolio it would stage today.

Not the real run: no Alpaca liquidity universe, no EDGAR/news/options facts, no red team or analyst.
Earnings dates come from Yahoo where available so the earnings-in-window block is applied.
"""
from __future__ import annotations

import io
import logging
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from desk.config import load_config
from desk.facts import candidate_blockers
from desk.portfolio import select_entries, size_position
from desk.regime import current_regime
from desk.signals import compute_panels, rerank, snapshot

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
WIKI = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
}


def constituents() -> list:
    syms = set()
    for name, url in WIKI.items():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (alpaca-desk preview)"})
        html = urllib.request.urlopen(req, timeout=60).read()
        for t in pd.read_html(io.BytesIO(html)):
            if "Symbol" in t.columns:
                syms |= {str(s).replace(".", "-") for s in t["Symbol"]}
                break
    return sorted(s for s in syms if s.replace("-", "").isalpha())


def download(symbols: list, cache: Path) -> dict:
    if cache.exists():
        return pd.read_pickle(cache)
    fields = [("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"), ("volume", "Volume")]
    frames = {f: [] for f, _ in fields}
    todo = list(symbols)
    for attempt in range(4):  # Yahoo silently drops symbols under load; retry the stragglers
        raw = yf.download(todo, period="18mo", auto_adjust=False, group_by="column", threads=attempt == 0, progress=False)
        for f, col in fields:
            df = raw[col].dropna(axis=1, how="all") if col in raw else pd.DataFrame()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            frames[f].append(df)
        got = set().union(*(set(d.columns) for d in frames["close"]))
        todo = [s for s in todo if s not in got]
        print(f"download attempt {attempt + 1}: {len(got)} symbols, {len(todo)} missing", file=sys.stderr)
        if not todo:
            break
    panel = {f: pd.concat(frames[f], axis=1).sort_index() for f, _ in fields}
    # Rows where most equities are NaN are holiday artifacts (an index printed, stocks did not); they would
    # poison every rolling window. Drop them from all fields.
    keep = panel["close"].isna().mean(axis=1) <= 0.5
    panel = {f: df.loc[keep] for f, df in panel.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(panel, cache)
    return panel


def earnings_days(symbols: list, today: date, cache: Path) -> dict:
    import json
    import time
    known = json.loads(cache.read_text()) if cache.exists() else {}
    out = {}
    for s in symbols:
        if s in known:
            out[s] = known[s]
            continue
        for attempt in range(3):
            try:
                ed = yf.Ticker(s).get_earnings_dates(limit=4)
                future = [ts.date() for ts in ed.index if ts.date() >= today] if ed is not None else []
                out[s] = known[s] = (min(future) - today).days if future else None
                break
            except Exception as e:
                if "Too Many" in str(e) and attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                out[s] = None
                break
        time.sleep(0.5)
    cache.write_text(json.dumps(known))
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--earnings-block-days", type=int, default=None, help="override facts.exclude_earnings_within_days")
    args = ap.parse_args()
    cfg = load_config()
    if args.earnings_block_days is not None:
        cfg["facts"]["exclude_earnings_within_days"] = args.earnings_block_days
    scfg, pcfg, ecfg, ucfg, fcfg = cfg["signals"], cfg["portfolio"], cfg["exits"], cfg["universe"], cfg["facts"]
    equity = float(cfg["account"]["starting_capital"])
    today = date.today()

    print("universe: fetching S&P 500 + 400 constituents...", file=sys.stderr)
    syms = constituents()
    print(f"universe: {len(syms)} symbols; downloading 18 months of daily bars...", file=sys.stderr)
    panel = download(syms + ["SPY", "^VIX"], Path(cfg["_root"]) / cfg["data"]["cache_dir"] / f"preview_{today.isoformat()}.pkl")
    spy = panel["close"]["SPY"].dropna()
    vix = panel["close"]["^VIX"].dropna()
    for f in panel:
        panel[f] = panel[f].drop(columns=["SPY", "^VIX"], errors="ignore")

    close, vol = panel["close"], panel["volume"]
    dollar_vol = (close * vol).tail(20).mean()
    liquid = dollar_vol[(dollar_vol >= ucfg["min_avg_dollar_volume"]) & (close.ffill().iloc[-1] >= ucfg["min_price"])].index
    for f in panel:
        panel[f] = panel[f][liquid]
    print(f"liquid universe: {len(liquid)}; last bar {close.index[-1].date()}", file=sys.stderr)

    panels = compute_panels(panel, scfg)
    snap = snapshot(panels, -1, scfg)
    reg = current_regime(spy, vix, cfg["regime"])
    target = max(1, int(round(pcfg["target_positions"] * reg["scale"])))

    pool = list(snap[snap["eligible"]].index[: fcfg["candidate_pool"]])
    print(f"checking earnings dates for top {len(pool)}...", file=sys.stderr)
    edays = earnings_days(pool, today, Path(cfg["_root"]) / cfg["data"]["cache_dir"] / f"preview_earnings_{today.isoformat()}.json")
    blocked = {}
    for s in pool:
        r = candidate_blockers({"earnings_in_days": edays.get(s)}, fcfg)
        if r:
            blocked[s] = "; ".join(r)
    snap = rerank(snap, None, blocked)

    picks = select_entries(snap, panels["returns"], [], target, pcfg)
    cash_avail = equity * (1 - pcfg["cash_floor_pct"])
    rows = []
    for s in picks:
        r = snap.loc[s]
        sz = size_position(equity, cash_avail, float(r["close"]), float(r["atr"]), pcfg, ecfg)
        if not sz:
            continue
        cash_avail -= sz["notional"]
        rows.append({"symbol": s, "rank": int(r["rank"]), "price": round(float(r["close"]), 2), "qty": sz["qty"],
                     "notional": round(sz["notional"]), "stop": sz["stop"], "target": sz["target"],
                     "mom_6m": f"{r['momentum']:+.0%}", "ret_20d": f"{r['short']:+.0%}", "atr%": f"{r['atr_pct']:.1%}",
                     "earn_days": edays.get(s)})
    port = pd.DataFrame(rows)

    print(f"\nPREVIEW PORTFOLIO {today} (Yahoo data, S&P 500+400 universe, ${equity:,.0f})")
    print(f"regime {reg['label']}: SPY {reg['spy']:.0f} vs 200d MA {reg['spy_ma']:.0f}, VIX {reg['vix']:.1f} -> target {target} positions")
    print(f"eligible names {int(snap['eligible'].sum())} of {len(snap)}; blocked in top {len(pool)}: {blocked or 'none'}")
    print(port.to_string(index=False))
    print(f"\ninvested ${port['notional'].sum():,.0f}  cash ${equity - port['notional'].sum():,.0f}  "
          f"max loss if every stop hits: ${sum((r['price'] - r['stop']) * r['qty'] for r in rows):,.0f}")
    nxt = snap[snap["eligible"] & ~snap.index.isin(picks)].head(8)
    print("next in line: " + ", ".join(f"{s}(#{int(v)})" for s, v in nxt["rank"].items()))
    print("\nNot applied here: Alpaca liquidity universe, EDGAR/news/options/social facts, red team, analyst.")


if __name__ == "__main__":
    main()

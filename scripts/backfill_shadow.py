#!/usr/bin/env python
"""Seed the shadow book (desk/shadow.py) from the daily briefs written before it existed (2026-09-14 to 10-16).

What the briefs carry: staged buys (price, stop, target, the half-size tag), the first eight hard-filter blocks a
day, and holdings the analyst told the desk to sell. Not carried: candidates the analyst vetoed before they were
staged - those start with the live recording. Blocked names are priced at that day's close with the desk's ATR.
Rows are marked backfill: true. Refuses to run twice.

  python scripts/backfill_shadow.py
"""
import re, sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from desk import shadow
from desk.config import load_config, alpaca_keys
from desk.data import cached_bars
from desk.state import BRIEFS_DIR

BUY = re.compile(r"^\s+BUY (\S+)\s+x(\d+) lim ([\d.]+) stop ([\d.]+) tgt ([\d.]+|trail)[^-]*- (.*)$")
BLOCK = re.compile(r"^\s+- blocked (\S+): (.+)$")
EXIT = re.compile(r"^\s+(\S+)\s+([\d.]+) @ [\d.]+ -> ([\d.]+) .*-> EXIT: analyst REMOVE: ?(.*)$")
CUTOFF = "2026-10-19"          # live recording starts with the first desk run after the merge


def main():
    cfg = load_config()
    if any(r.get("backfill") for r in shadow.load().to_dict("records")):
        sys.exit("already backfilled")
    ecfg, m = cfg["exits"], float(cfg["exits"]["stop_atr_multiple"])
    rows, need_atr = [], []
    for p in sorted(BRIEFS_DIR.glob("2026-*.md")):
        day = date.fromisoformat(p.stem)
        if p.stem >= CUTOFF:
            continue
        for line in p.read_text().splitlines():
            if (b := BUY.match(line)):
                sym, qty, lim, stop, tgt, thesis = b.groups()
                price = float(lim) / 1.005
                atr = (price - float(stop)) / m
                rows.append(shadow.row(day, "bought", sym, price, atr, ecfg, qty=int(qty), reason=thesis))
                if "half size" in thesis:
                    rows.append(shadow.row(day, "half_size", sym, price, atr, ecfg, qty=int(qty),
                                           reason="implied vol >> realized"))
            elif (b := BLOCK.match(line)):
                need_atr.append((day, b.group(1), b.group(2)))
            elif (b := EXIT.match(line)):
                sym, qty, px, why = b.groups()
                need_atr.append((day, sym, "EXIT:" + why, float(qty), float(px)))
    data = __import__("alpaca.data.historical", fromlist=["StockHistoricalDataClient"]).StockHistoricalDataClient(*alpaca_keys())
    panel = cached_bars(data, sorted({x[1] for x in need_atr}), 200, cfg, tag="shadow-backfill")
    n = int(cfg["signals"]["atr_days"])
    tr = pd.concat([panel["high"] - panel["low"], (panel["high"] - panel["close"].shift()).abs(),
                    (panel["low"] - panel["close"].shift()).abs()]).groupby(level=0).max()
    atr = tr.rolling(n).mean()
    for x in need_atr:
        day, sym = x[0], x[1]
        if sym not in panel["close"].columns:
            continue
        c = panel["close"][sym][panel["close"].index <= pd.Timestamp(day)].dropna()
        a = atr[sym][atr.index <= pd.Timestamp(day)].dropna()
        if not len(c) or not len(a):
            continue
        if x[2].startswith("EXIT:"):
            rows.append(shadow.row(day, "analyst_exit", sym, x[4], float(a.iloc[-1]), ecfg, qty=x[3], reason=x[2][5:]))
        else:
            rows.append(shadow.row(day, "blocked", sym, float(c.iloc[-1]), float(a.iloc[-1]), ecfg, reason=x[2]))
    for r in rows:
        r["backfill"] = True
    shadow.record(rows)
    print(f"{len(rows)} rows from {len(list(BRIEFS_DIR.glob('2026-*.md')))} briefs -> {shadow.book_path()}")


if __name__ == "__main__":
    main()

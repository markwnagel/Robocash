#!/usr/bin/env python
"""Build state/history/earnings.csv (date, symbol) so the earnings rules can be backtested.

  python scripts/build_earnings_history.py --source yfinance --start 2023-06-01   # per-symbol, multi-year
  python scripts/build_earnings_history.py --source finnhub  --start 2026-08-01   # bulk, but the free tier
                                                                                  # only serves ~30 trailing days

yfinance is the practical choice for history: one request per universe symbol, resumable, ~10 minutes for
800 names. Rows are appended incrementally so an interrupted run can be re-run without repeating work.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desk.config import load_config  # noqa: E402

log = logging.getLogger("build_earnings")


def month_windows(start: date, end: date):
    cur = start.replace(day=1)
    while cur <= end:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield cur, min(nxt - timedelta(days=1), end)
        cur = nxt


def from_finnhub(start: date, end: date) -> set:
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        sys.exit("FINNHUB_API_KEY not set")
    rows = set()
    for a, b in month_windows(start, end):
        url = f"https://finnhub.io/api/v1/calendar/earnings?from={a.isoformat()}&to={b.isoformat()}&token={key}"
        try:
            raw = json.loads(urllib.request.urlopen(url, timeout=60).read())
        except Exception as e:
            log.warning("%s..%s: %s", a, b, e)
            continue
        for e in raw.get("earningsCalendar", []):
            if e.get("symbol") and e.get("date"):
                rows.add((e["date"], e["symbol"].upper()))
        log.info("%s: %d rows so far", a.strftime("%Y-%m"), len(rows))
        time.sleep(1.1)
    return rows


def from_yfinance(symbols: list, start: date, out: Path) -> int:
    import yfinance as yf
    done = set()
    if out.exists():
        with open(out) as f:
            done = {r["symbol"] for r in csv.DictReader(f)}
    new_file = not out.exists()
    n = 0
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date", "symbol"])
        for i, s in enumerate(symbols):
            if s in done:
                continue
            for attempt in range(4):
                try:
                    ed = yf.Ticker(s).get_earnings_dates(limit=20)
                    dates = sorted({ts.date() for ts in ed.index if ts.date() >= start}) if ed is not None else []
                    for d in dates:
                        w.writerow([d.isoformat(), s])
                        n += 1
                    if not dates:  # keep a sentinel so the symbol is not re-queried on resume
                        w.writerow(["", s])
                    f.flush()
                    break
                except Exception as e:
                    if "Too Many" in str(e) and attempt < 3:
                        time.sleep(5 * (attempt + 1))
                        continue
                    log.warning("%s: %s", s, e)
                    break
            time.sleep(0.4)
            if i % 50 == 49:
                log.info("%d/%d symbols, %d rows", i + 1, len(symbols), n)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--source", choices=["yfinance", "finnhub"], default="yfinance")
    ap.add_argument("--symbols-file", default=None, help="defaults to state/universe.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    root = Path(cfg["_root"])
    out_dir = root / "state" / "history"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "earnings.csv"
    start = date.fromisoformat(args.start)

    if args.source == "finnhub":
        rows = from_finnhub(start, date.fromisoformat(args.end))
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "symbol"])
            w.writerows(sorted(rows))
        print(f"wrote {out} ({len(rows)} rows)")
        return

    symbols = json.loads(Path(args.symbols_file or root / "state" / "universe.json").read_text())
    symbols = symbols["symbols"] if isinstance(symbols, dict) else symbols
    n = from_yfinance(symbols, start, out)
    print(f"appended {n} rows to {out}")


if __name__ == "__main__":
    main()

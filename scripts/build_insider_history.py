#!/usr/bin/env python
"""Pull Form 4 open-market buys/sells for the universe so the insider tilt can be backtested.

  python scripts/build_insider_history.py --start 2023-01-01
Writes state/history/insider.csv (date, symbol, net_usd). Slow: ~10 EDGAR requests/second, tens of thousands
of filings for 800 names. Re-runs are cached, so it can be interrupted and resumed.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desk.config import load_config  # noqa: E402
from desk.edgar import Edgar  # noqa: E402

log = logging.getLogger("build_insider")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--symbols-file", default=None, help="defaults to state/universe.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    root = Path(cfg["_root"])
    start = date.fromisoformat(args.start)
    symbols = json.loads(Path(args.symbols_file or root / "state" / "universe.json").read_text())
    symbols = symbols["symbols"] if isinstance(symbols, dict) else symbols
    edgar = Edgar(root / cfg["data"]["cache_dir"] / "edgar")
    out_dir = root / "state" / "history"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "insider.csv"
    done = set()
    if out.exists():
        with open(out) as f:
            done = {r["symbol"] for r in csv.DictReader(f)}
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["date", "symbol", "net_usd"])
        for i, s in enumerate(symbols):
            if s in done:
                continue
            try:
                cik = edgar.cik(s)
                if not cik:
                    continue
                filings = edgar.filings(cik, start)
                act = edgar.form4_activity(cik, filings, start)
                for t in act["transactions"]:
                    w.writerow([t["filed"], s, round(t["usd"] if t["code"] == "P" else -t["usd"], 2)])
                f.flush()
            except Exception as e:
                log.warning("%s: %s", s, e)
            if i % 25 == 0:
                log.info("%d/%d", i, len(symbols))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Print the backtest trial table, or rewrite the README's generated block with it.

  python scripts/trials_report.py                 print the table
  python scripts/trials_report.py --write-readme  replace the block between the trials markers in README.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desk import trials as T  # noqa: E402
from desk.config import ROOT, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-readme", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    log = T.load(ROOT / "state" / "history" / "trials.jsonl")
    block = T.markdown(T.summary(log, int((cfg.get("backtest") or {}).get("prior_untracked_trials", 0))))
    if args.write_readme:
        T.write_readme(ROOT / "README.md", block)
        print(f"README.md trials table rewritten from {len(log)} runs")
    else:
        print(block)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Point-in-time check of skipping sliver entries (portfolio.min_entry_fraction; owner request 2026-11-06, after the
new account's first build bought HST at 2 shares / $50 and KO at 1 share with the last of the cash).

Variants, 4 starts a quarter apart in 2018-26 and 2023-26, point-in-time universe, config as it is today:
  today   0     the last picks of a day take whatever cash is left
  skip25  0.25  skip a pick the cash funds at under 25% of a full position
  skip50  0.50  under 50%
Reported on return, drawdown and against SPY (robustness.excess_vs_spy). Each variant is logged to trials.jsonl.

  python scripts/check_min_entry.py                    # at account.starting_capital ($10k)
  python scripts/check_min_entry.py --capital 100000   # the live account's size
"""
import argparse, json, os, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)
import backtest as B, robustness as R
from desk import pit
from desk.config import load_config

WINDOWS = {"full": "2018-01-02", "recent": "2023-09-11"}
END = "2026-09-11"
VARIANTS = (("today", 0), ("skip25", 0.25), ("skip50", 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float)
    args = ap.parse_args()
    cfg = load_config()
    assert cfg["account"]["paper"] is True
    capital = args.capital or float(cfg["account"]["starting_capital"])
    root = Path(cfg["_root"]); hist = root / "state" / "history"
    build = json.loads((hist / "pit" / "build.json").read_text())
    panel_path = root / build["prices"]
    panel = pd.read_pickle(panel_path); dates = panel["close"].index
    members_pit = pit.expand(pd.read_csv(hist / "pit" / "members.csv"), dates)
    from alpaca.data.historical import StockHistoricalDataClient
    from desk.config import alpaca_keys
    from desk.regime import vix_series
    spy = B.frozen_prices(StockHistoricalDataClient(*alpaca_keys()), ["SPY"], 2750, cfg, pd.Timestamp(END))[0]["close"]["SPY"].dropna()
    vix = vix_series(root / cfg["data"]["cache_dir"])
    earnings = pd.read_csv(hist / "earnings.csv") if (hist / "earnings.csv").exists() else None
    out = root / "state" / "robustness" / f"2026-11-06-min-entry-{int(capital)}"; out.mkdir(parents=True, exist_ok=True)
    inputs = out / "_inputs.pkl"
    pd.to_pickle({"cfg": cfg, "spy": spy, "vix": vix, "earnings": earnings, "capital": capital,
                  "members": {"pit": members_pit}}, inputs)
    jobs = []
    for wname, ws in WINDOWS.items():
        for k in range(4):
            st = str((pd.Timestamp(ws) + pd.DateOffset(months=3 * k)).date())
            for v, frac in VARIANTS:
                jobs.append({"name": f"{v}|{wname}@{k}", "group": wname, "start": st, "end": END, "variant": v,
                             "offset": k, "overrides": [(("portfolio", "min_entry_fraction"), frac)]})
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(20, (os.cpu_count() or 2) - 2)),
                             initializer=R._init, initargs=(str(inputs), str(panel_path))) as ex:
        res = {r["name"]: r for r in ex.map(R.run_job, jobs, chunksize=1)}
    inputs.unlink(missing_ok=True)
    R.log_variant_trials(root / "state" / "history" / "trials.jsonl", jobs, res, f"min-entry-2026-11-06-{int(capital)}")
    rows = []
    for j in jobs:
        r = res[j["name"]]; st = R.full_stats(r["equity"])
        rows.append({"variant": j["variant"], "window": j["group"], "offset": j["offset"], "n_trades": r["n_trades"],
                     "cagr": st["cagr"], "maxdd": st["maxdd"], "sharpe": st["sharpe"], **R.excess_vs_spy(r["equity"], spy),
                     "invested": float(1 - (r["cash"] / r["equity"]).mean())})
    df = pd.DataFrame(rows); df.to_csv(out / "runs.csv", index=False)
    print(f"{len(jobs)} runs at ${capital:,.0f} in {time.time() - t0:.0f}s -> {out}")
    for w, g in df.groupby("window", sort=False):
        base = g[g.variant == "today"].set_index("offset")
        for v, _ in VARIANTS:
            x = g[g.variant == v].set_index("offset")
            print(f"{w:6} {v:7} CAGR {x.cagr.median():6.2%} (better {int((x.cagr > base.cagr).sum())}/4)  "
                  f"max DD {x.maxdd.median():6.1%} (better {int((x.maxdd > base.maxdd).sum())}/4)  "
                  f"6m vs SPY median {x.excess_6m_median.median():+6.2%} (better {int((x.excess_6m_median > base.excess_6m_median).sum())}/4)  "
                  f"p10 {x.excess_6m_p10.median():+6.2%}  trades {x.n_trades.median():.0f}  invested {x.invested.median():4.0%}")


if __name__ == "__main__":
    main()

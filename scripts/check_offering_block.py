#!/usr/bin/env python
"""Point-in-time check of the live dilution block (engineering review 2026-10-17, item 6): the backtest has never
modelled it, so every published result is for a strategy that buys names the live desk refuses.

Variants, 4 starts a quarter apart in 2018-26 and 2023-26, point-in-time universe:
  no block     today's backtest
  block        an S-1 or 8-K item 3.02 in the last 90 days blocks a candidate, as live
Reported on return, drawdown and against SPY (robustness.excess_vs_spy). Needs state/history/offerings.csv from
scripts/build_offering_history.py. Each variant is logged to trials.jsonl (robustness.log_variant_trials).

  python scripts/check_offering_block.py
"""
import json, os, sys, time
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


def main():
    cfg = load_config()
    assert cfg["account"]["paper"] is True
    root = Path(cfg["_root"]); hist = root / "state" / "history"
    build = json.loads((hist / "pit" / "build.json").read_text())
    panel_path = root / build["prices"]
    panel = pd.read_pickle(panel_path); dates = panel["close"].index
    members_pit = pit.expand(pd.read_csv(hist / "pit" / "members.csv"), dates)
    offerings = pd.read_csv(hist / "offerings.csv")
    from alpaca.data.historical import StockHistoricalDataClient
    from desk.config import alpaca_keys
    from desk.regime import vix_series
    spy = B.frozen_prices(StockHistoricalDataClient(*alpaca_keys()), ["SPY"], 2750, cfg, pd.Timestamp(END))[0]["close"]["SPY"].dropna()
    vix = vix_series(root / cfg["data"]["cache_dir"])
    earnings = pd.read_csv(hist / "earnings.csv") if (hist / "earnings.csv").exists() else None
    out = root / "state" / "robustness" / "2026-10-17-offerings"; out.mkdir(parents=True, exist_ok=True)
    inputs = out / "_inputs.pkl"
    pd.to_pickle({"cfg": cfg, "spy": spy, "vix": vix, "earnings": earnings, "offerings": offerings,
                  "capital": float(cfg["account"]["starting_capital"]), "members": {"pit": members_pit}}, inputs)
    jobs = []
    for wname, ws in WINDOWS.items():
        for k in range(4):
            st = str((pd.Timestamp(ws) + pd.DateOffset(months=3 * k)).date())
            for v in ("no block", "block"):
                jobs.append({"name": f"{v}|{wname}@{k}", "group": wname, "start": st, "end": END, "variant": v,
                             "offset": k, "keep_trades": True, "offerings": v == "block",
                             # a distinct config id for the trial log
                             "overrides": [(("facts", "offering_block_backtest"), v == "block")]})
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(20, (os.cpu_count() or 2) - 2)),
                             initializer=R._init, initargs=(str(inputs), str(panel_path))) as ex:
        res = {r["name"]: r for r in ex.map(R.run_job, jobs, chunksize=1)}
    inputs.unlink(missing_ok=True)
    R.log_variant_trials(root / "state" / "history" / "trials.jsonl", jobs, res, "offerings-2026-10-17")
    rows = []
    for j in jobs:
        r = res[j["name"]]; st = R.full_stats(r["equity"])
        rows.append({"variant": j["variant"], "window": j["group"], "offset": j["offset"], "n_trades": r["n_trades"],
                     "cagr": st["cagr"], "maxdd": st["maxdd"], "sharpe": st["sharpe"], **R.excess_vs_spy(r["equity"], spy),
                     "invested": float(1 - (r["cash"] / r["equity"]).mean())})
    df = pd.DataFrame(rows); df.to_csv(out / "runs.csv", index=False)
    print(f"{len(jobs)} runs in {time.time() - t0:.0f}s -> {out}")
    for w, g in df.groupby("window", sort=False):
        base = g[g.variant == "no block"].set_index("offset"); blk = g[g.variant == "block"].set_index("offset")
        print(f"\n{w}: median CAGR no block {base.cagr.median():.2%} / block {blk.cagr.median():.2%}; "
              f"return better with block in {int((blk.cagr > base.cagr).sum())}/4; "
              f"max DD {base.maxdd.median():.1%} / {blk.maxdd.median():.1%}; "
              f"6m excess vs SPY median {base.excess_6m_median.median():+.2%} / {blk.excess_6m_median.median():+.2%}, "
              f"p10 {base.excess_6m_p10.median():+.2%} / {blk.excess_6m_p10.median():+.2%}; "
              f"invested {base.invested.median():.0%} / {blk.invested.median():.0%}; trades {base.n_trades.median():.0f} / {blk.n_trades.median():.0f}")


if __name__ == "__main__":
    main()

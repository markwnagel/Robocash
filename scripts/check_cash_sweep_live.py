#!/usr/bin/env python
"""The cash sweep re-tested at live cash levels (weekly audit 2026-10-31, for the 2026-11-06 review).

The 2026-10-17 study (check_cash_sweep.py) ran the sweep against the backtest's own idle cash, ~19-30%. Live
holds ~35-45%, mostly because 42% of entries are half-sized on implied vol (shadow book half_size 23 of 55),
which the backtest cannot see. Here both arms run in robustness.py's "frictions" mode with the live rates -
half-size 42%, analyst REMOVE 10%, REDUCE retired (reduce_factor 1.0 since 2026-10-17) - on paired seeds, so
each seed halves the same entries with and without the sweep.

  today   idle cash stays cash
  sweep   idle cash above the floor and staged orders held in SPY (band 2%)

Bar (owner's, as excess over SPY from robustness.excess_vs_spy): median six-month excess better in >=3/4 starts
in BOTH windows, 10th percentile no worse in >=3/4. Max drawdown and invested share are reported beside it.
Frictions-mode runs are not logged to trials.jsonl (robustness.log_variant_trials skips them); the sweep variant
itself was logged by the 2026-10-17 study.

  python scripts/check_cash_sweep_live.py [--seeds 5]
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
LIVE = {"remove": 0.10, "reduce": 0.0, "factor": 1.0, "half": 0.42}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config()
    assert cfg["account"]["paper"] is True
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
    out = root / "state" / "robustness" / "2026-11-01-sweep-live"; out.mkdir(parents=True, exist_ok=True)
    inputs = out / "_inputs.pkl"
    pd.to_pickle({"cfg": cfg, "spy": spy, "vix": vix, "earnings": earnings,
                  "capital": float(cfg["account"]["starting_capital"]), "members": {"pit": members_pit}}, inputs)
    jobs = []
    for wname, ws in WINDOWS.items():
        for k in range(4):
            st = str((pd.Timestamp(ws) + pd.DateOffset(months=3 * k)).date())
            for s in range(args.seeds):
                for v, sw in (("today", {"enabled": False}), ("sweep", {"enabled": True, "band_pct": 0.02})):
                    jobs.append({"name": f"{v}|{wname}@{k}#{s}", "group": wname, "start": st, "end": END,
                                 "variant": v, "offset": k, "seed": 3000 + 100 * k + s, "mode": "frictions",
                                 "frictions": LIVE, "overrides": [(("portfolio", "cash_sweep"), sw)]})
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(20, (os.cpu_count() or 2) - 2)),
                             initializer=R._init, initargs=(str(inputs), str(panel_path))) as ex:
        res = {r["name"]: r for r in ex.map(R.run_job, jobs, chunksize=1)}
    inputs.unlink(missing_ok=True)
    rows = []
    for j in jobs:
        r = res[j["name"]]; st = R.full_stats(r["equity"])
        rows.append({"variant": j["variant"], "window": j["group"], "offset": j["offset"], "seed": j["seed"],
                     "n_trades": r["n_trades"], "cagr": st["cagr"], "maxdd": st["maxdd"], "sharpe": st["sharpe"],
                     **R.excess_vs_spy(r["equity"], spy), "invested": float(1 - (r["cash"] / r["equity"]).mean())})
    df = pd.DataFrame(rows); df.to_csv(out / "runs.csv", index=False)
    print(f"{len(jobs)} runs in {time.time() - t0:.0f}s -> {out}")
    # One number per start: the median across seeds.
    per = df.groupby(["window", "variant", "offset"]).median(numeric_only=True).reset_index()
    for w in WINDOWS:
        g = per[per.window == w]
        base = g[g.variant == "today"].set_index("offset")
        for v in ("today", "sweep"):
            x = g[g.variant == v].set_index("offset")
            print(f"{w:6} {v:6} CAGR {x.cagr.median():6.2%} (better {int((x.cagr > base.cagr).sum())}/4)  "
                  f"max DD {x.maxdd.median():6.1%} (better {int((x.maxdd > base.maxdd).sum())}/4)  "
                  f"6m vs SPY median {x.excess_6m_median.median():+6.2%} "
                  f"(better {int((x.excess_6m_median > base.excess_6m_median).sum())}/4)  "
                  f"p10 {x.excess_6m_p10.median():+6.2%} (no worse {int((x.excess_6m_p10 >= base.excess_6m_p10).sum())}/4)  "
                  f"invested {x.invested.median():4.0%}")
    # Paired by seed: how often the sweep beats today on six-month excess, same entries halved.
    pair = df.pivot_table(index=["window", "offset", "seed"], columns="variant", values="excess_6m_median")
    for w in WINDOWS:
        p = pair.loc[w]
        print(f"{w:6} paired seeds: sweep better on 6m excess in {int((p.sweep > p.today).sum())}/{len(p)}")


if __name__ == "__main__":
    main()

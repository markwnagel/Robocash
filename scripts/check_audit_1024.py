#!/usr/bin/env python
"""Point-in-time checks for the 2026-10-24 audit's two signal proposals:
  1. portfolio.correlation_days: 60 -> 120 (three refiners passed the 0.75 / one-peer filter together)
  2. signals.weights: short 0.2 -> 0.15 and 0.10, the difference to low_vol (entries after a +20% 20-day move
     averaged -6.9% vs -3.7%, n=10)
4 starts a quarter apart in 2018-26 and 2023-26, point-in-time universe, today's config otherwise. The owner's bar
(2026-10-17): median six-month excess over SPY better in >= 3/4 starts in both windows, its 10th percentile and
the median max drawdown no worse; invested % and trade count are reported so exposure is not read as edge.
Each variant is logged to trials.jsonl (robustness.log_variant_trials).

  python scripts/check_audit_1024.py
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


def variants(cfg):
    w = dict(cfg["signals"]["weights"])
    shifted = lambda s: {**w, "short": s, "low_vol": round(w["low_vol"] + w["short"] - s, 2)}  # noqa: E731
    return {"base": [],
            "corr120": [(("portfolio", "correlation_days"), 120)],
            "short0.15": [(("signals", "weights"), shifted(0.15))],
            "short0.10": [(("signals", "weights"), shifted(0.10))]}


def main():
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
    out = root / "state" / "robustness" / "2026-10-24-audit"; out.mkdir(parents=True, exist_ok=True)
    inputs = out / "_inputs.pkl"
    pd.to_pickle({"cfg": cfg, "spy": spy, "vix": vix, "earnings": earnings,
                  "capital": float(cfg["account"]["starting_capital"]), "members": {"pit": members_pit}}, inputs)
    jobs = []
    for wname, ws in WINDOWS.items():
        for k in range(4):
            st = str((pd.Timestamp(ws) + pd.DateOffset(months=3 * k)).date())
            for v, ov in variants(cfg).items():
                jobs.append({"name": f"{v}|{wname}@{k}", "group": wname, "start": st, "end": END, "variant": v,
                             "offset": k, "keep_trades": True, "overrides": ov})
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(20, (os.cpu_count() or 2) - 2)),
                             initializer=R._init, initargs=(str(inputs), str(panel_path))) as ex:
        res = {r["name"]: r for r in ex.map(R.run_job, jobs, chunksize=1)}
    inputs.unlink(missing_ok=True)
    R.log_variant_trials(root / "state" / "history" / "trials.jsonl", [j for j in jobs if j["variant"] != "base"], res,
                         "audit-2026-10-24")
    rows = []
    for j in jobs:
        r = res[j["name"]]; st = R.full_stats(r["equity"])
        rows.append({"variant": j["variant"], "window": j["group"], "offset": j["offset"], "n_trades": r["n_trades"],
                     "cagr": st["cagr"], "maxdd": st["maxdd"], "sharpe": st["sharpe"], **R.excess_vs_spy(r["equity"], spy),
                     "invested": float(1 - (r["cash"] / r["equity"]).mean())})
    df = pd.DataFrame(rows); df.to_csv(out / "runs.csv", index=False)
    print(f"{len(jobs)} runs in {time.time() - t0:.0f}s -> {out}")
    for w, g in df.groupby("window", sort=False):
        base = g[g.variant == "base"].set_index("offset")
        print(f"\n{w}: base median CAGR {base.cagr.median():.2%}, max DD {base.maxdd.median():.1%}, 6m excess "
              f"{base.excess_6m_median.median():+.2%} (p10 {base.excess_6m_p10.median():+.2%}), invested "
              f"{base.invested.median():.0%}, trades {base.n_trades.median():.0f}")
        for v in [x for x in variants(cfg) if x != "base"]:
            x = g[g.variant == v].set_index("offset")
            print(f"  {v:10} CAGR {x.cagr.median():.2%} (better {int((x.cagr > base.cagr).sum())}/4); 6m excess "
                  f"{x.excess_6m_median.median():+.2%} (better {int((x.excess_6m_median > base.excess_6m_median).sum())}/4), "
                  f"p10 {x.excess_6m_p10.median():+.2%} (no worse {int((x.excess_6m_p10 >= base.excess_6m_p10).sum())}/4); "
                  f"max DD {x.maxdd.median():.1%} (no worse {int((x.maxdd >= base.maxdd).sum())}/4); "
                  f"invested {x.invested.median():.0%}; trades {x.n_trades.median():.0f}")


if __name__ == "__main__":
    main()

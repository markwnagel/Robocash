#!/usr/bin/env python
"""Re-baseline check (engineering review 2026-10-17, item 8). config.yaml was tuned on the frozen 800-name list over
2023-26; its header says two settings beat it point-in-time (robustness 2026-09-24): 15 target positions and
regime.off_liquidate. Tested alone and together against today's config, 4 starts in 2018-26 and 2023-26.

Rule, fixed before the run: adopt a variant only if its median six-month excess over SPY is better than today's in
>= 3 of 4 starts in BOTH windows, and its median max drawdown is no worse in either window.

  python scripts/check_rebaseline.py
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
POS15 = [(("portfolio", "target_positions"), 15)]
OFFLIQ = [(("regime", "off_liquidate"), True)]
VARIANTS = {"today": [], "15 positions": POS15, "cash below the MA": OFFLIQ, "both": POS15 + OFFLIQ}
END = "2026-09-11"


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
    out = root / "state" / "robustness" / "2026-10-17-rebaseline"; out.mkdir(parents=True, exist_ok=True)
    inputs = out / "_inputs.pkl"
    pd.to_pickle({"cfg": cfg, "spy": spy, "vix": vix, "earnings": earnings,
                  "capital": float(cfg["account"]["starting_capital"]), "members": {"pit": members_pit}}, inputs)
    jobs = []
    for wname, ws in WINDOWS.items():
        for k in range(4):
            st = str((pd.Timestamp(ws) + pd.DateOffset(months=3 * k)).date())
            for v, ov in VARIANTS.items():
                jobs.append({"name": f"{v}|{wname}@{k}", "group": wname, "start": st, "end": END, "variant": v,
                             "offset": k, "overrides": ov})
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(20, (os.cpu_count() or 2) - 2)),
                             initializer=R._init, initargs=(str(inputs), str(panel_path))) as ex:
        res = {r["name"]: r for r in ex.map(R.run_job, jobs, chunksize=1)}
    inputs.unlink(missing_ok=True)
    R.log_variant_trials(root / "state" / "history" / "trials.jsonl", jobs, res, "rebaseline-2026-10-17")
    rows = []
    for j in jobs:
        r = res[j["name"]]; st = R.full_stats(r["equity"])
        rows.append({"variant": j["variant"], "window": j["group"], "offset": j["offset"], "n_trades": r["n_trades"],
                     "cagr": st["cagr"], "maxdd": st["maxdd"], "sharpe": st["sharpe"], **R.excess_vs_spy(r["equity"], spy),
                     "invested": float(1 - (r["cash"] / r["equity"]).mean())})
    df = pd.DataFrame(rows); df.to_csv(out / "runs.csv", index=False)
    print(f"{len(jobs)} runs in {time.time() - t0:.0f}s -> {out}")
    for w, g in df.groupby("window", sort=False):
        base = g[g.variant == "today"].set_index("offset")
        for v in VARIANTS:
            x = g[g.variant == v].set_index("offset")
            print(f"{w:6} {v:9} CAGR {x.cagr.median():6.2%} (better {int((x.cagr > base.cagr).sum())}/4)  "
                  f"max DD {x.maxdd.median():6.1%} (better {int((x.maxdd > base.maxdd).sum())}/4)  "
                  f"6m vs SPY median {x.excess_6m_median.median():+6.2%} (better {int((x.excess_6m_median > base.excess_6m_median).sum())}/4)  "
                  f"p10 {x.excess_6m_p10.median():+6.2%}  beat {x.beat_6m.median():4.0%}  invested {x.invested.median():4.0%}")
    verdict = {}
    for v in VARIANTS:
        if v == "today":
            continue
        ok = True
        for w, g in df.groupby("window"):
            base = g[g.variant == "today"].set_index("offset"); x = g[g.variant == v].set_index("offset")
            ok &= int((x.excess_6m_median > base.excess_6m_median).sum()) >= 3 and x.maxdd.median() >= base.maxdd.median()
        verdict[v] = "ADOPT" if ok else "reject"
    print("\nRule (>=3/4 better vs SPY in both windows, median drawdown no worse in either):", verdict)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Point-in-time checks for the 2026-10-03 audit's proposals 2 (correlation_threshold) and 3 (REDUCE sizing).
Reuses robustness.py's runner; writes state/robustness/2026-10-03-audit/. Each variant is logged to trials.jsonl (robustness.log_variant_trials).

  python scripts/check_audit_proposals.py      ~20 min on 20 workers; summary: see config.yaml comments
"""
import json, os, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)
import backtest as B, robustness as R
from desk import pit
from desk.config import load_config



def main():
    FULL, RECENT, END = "2018-01-02", "2023-09-11", "2026-09-11"
    cfg = load_config()
    assert cfg["account"]["paper"] is True
    root = Path(cfg["_root"]); hist = root / "state" / "history"
    build = json.loads((hist / "pit" / "build.json").read_text())
    panel_path = root / build["prices"]
    panel = pd.read_pickle(panel_path); dates = panel["close"].index
    members_pit = pit.expand(pd.read_csv(hist / "pit" / "members.csv"), dates)
    frozen = [s for s in json.loads((hist / "universe_bt.json").read_text())["symbols"] if s in panel["close"].columns]
    from alpaca.data.historical import StockHistoricalDataClient
    from desk.config import alpaca_keys
    from desk.regime import vix_series
    data = StockHistoricalDataClient(*alpaca_keys())
    spy = B.frozen_prices(data, ["SPY"], 2750, cfg, pd.Timestamp(END))[0]["close"]["SPY"].dropna()
    vix = vix_series(root / cfg["data"]["cache_dir"])
    earnings = pd.read_csv(hist / "earnings.csv") if (hist / "earnings.csv").exists() else None
    out = root / "state" / "robustness" / "2026-10-03-audit"; out.mkdir(parents=True, exist_ok=True)
    inputs = out / "_inputs.pkl"
    pd.to_pickle({"cfg": cfg, "spy": spy, "vix": vix, "earnings": earnings, "capital": float(cfg["account"]["starting_capital"]),
                  "members": {"pit": members_pit, "frozen": pd.DataFrame(True, index=dates, columns=frozen)}}, inputs)

    jobs = []
    for wname, ws in (("full", FULL), ("recent", RECENT)):
        starts = [str((pd.Timestamp(ws) + pd.DateOffset(months=3 * k)).date()) for k in range(4)]
        for th in (0.75, 0.70, 0.65):
            for k, st in enumerate(starts):
                jobs.append({"name": f"corr{th}_{wname}@{k}", "group": f"corr_{wname}", "start": st, "end": END,
                             "overrides": [(("portfolio", "correlation_threshold"), th)], "keep_trades": True,
                             "variant": th, "offset": k})
    SIZE = {"stacked 0.75 (today)": {"factor": 0.75, "stack": True}, "stacked 0.85": {"factor": 0.85, "stack": True},
            "not stacked 0.75": {"factor": 0.75, "stack": False}}
    for reduce_rate in (R.FRICTIONS["reduce"], 0.9):
        for lbl, fr in SIZE.items():
            for k in range(20):
                jobs.append({"name": f"size[{lbl}|r{reduce_rate:.2f}]#{k}", "group": "size", "start": FULL, "end": END,
                             "mode": "frictions", "seed": 2000 + k, "frictions": {**fr, "reduce": reduce_rate},
                             "variant": lbl, "reduce_rate": reduce_rate})
    jobs.append({"name": "nofrictions", "group": "size", "start": FULL, "end": END, "variant": "no frictions"})

    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(max_workers=20, initializer=R._init, initargs=(str(inputs), str(panel_path))) as ex:
        for r in ex.map(R.run_job, jobs, chunksize=1):
            res[r["name"]] = r
    inputs.unlink(missing_ok=True)
    R.log_variant_trials(root / "state" / "history" / "trials.jsonl", jobs, res, "audit-2026-10-03")
    rows = []
    for j in jobs:
        r = res[j["name"]]
        st = R.full_stats(r["equity"])
        rows.append({**{k: j.get(k) for k in ("name", "group", "variant", "offset", "seed", "reduce_rate", "start")},
                     "n_trades": r["n_trades"], "total": st["total"], "cagr": st["cagr"], "maxdd": st["maxdd"],
                     "sharpe": st["sharpe"], "invested": float(1 - (r["cash"] / r["equity"]).mean())})
    df = pd.DataFrame(rows); df.to_csv(out / "runs.csv", index=False)
    trades = {n: r["trades"] for n, r in res.items() if r.get("trades") is not None}
    pd.to_pickle(trades, out / "trades.pkl")
    print(f"{len(jobs)} runs in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()

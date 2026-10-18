#!/usr/bin/env python
"""Point-in-time checks for the 2026-10-17 audit's three proposals:
  1. exits.stop_anchor: the live desk measures the bracket from the signal-day close, the backtest from the fill
  2. exits.signal_decay_rank: 40 -> 20 / 25 / 30
  3. analyst.reduce_factor: 0.75 -> 1.0 (paired seeds, REDUCE modelled as a random cut, as on 2026-10-03)
Reuses robustness.py's runner; writes state/robustness/2026-10-17-audit/. Each variant is logged to trials.jsonl (robustness.log_variant_trials).

  python scripts/check_audit_1017.py      summary: see config.yaml comments
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
    from alpaca.data.historical import StockHistoricalDataClient
    from desk.config import alpaca_keys
    from desk.regime import vix_series
    spy = B.frozen_prices(StockHistoricalDataClient(*alpaca_keys()), ["SPY"], 2750, cfg, pd.Timestamp(END))[0]["close"]["SPY"].dropna()
    vix = vix_series(root / cfg["data"]["cache_dir"])
    earnings = pd.read_csv(hist / "earnings.csv") if (hist / "earnings.csv").exists() else None
    out = root / "state" / "robustness" / "2026-10-17-audit"; out.mkdir(parents=True, exist_ok=True)
    inputs = out / "_inputs.pkl"
    pd.to_pickle({"cfg": cfg, "spy": spy, "vix": vix, "earnings": earnings, "capital": float(cfg["account"]["starting_capital"]),
                  "members": {"pit": members_pit}}, inputs)

    jobs = []
    for wname, ws in WINDOWS.items():
        starts = [str((pd.Timestamp(ws) + pd.DateOffset(months=3 * k)).date()) for k in range(4)]
        for k, st in enumerate(starts):
            for anchor in ("signal", "fill"):
                jobs.append({"name": f"anchor_{anchor}_{wname}@{k}", "group": f"anchor_{wname}", "start": st, "end": END,
                             "overrides": [(("exits", "stop_anchor"), anchor)], "keep_trades": True,
                             "variant": anchor, "offset": k})
            for rank in (20, 25, 30, 40):
                jobs.append({"name": f"decay{rank}_{wname}@{k}", "group": f"decay_{wname}", "start": st, "end": END,
                             "overrides": [(("exits", "signal_decay_rank"), rank)], "keep_trades": True,
                             "variant": rank, "offset": k})
        for reduce_rate in (R.FRICTIONS["reduce"], 0.9):
            for factor in (0.75, 1.0):
                for s in range(20):
                    jobs.append({"name": f"reduce{factor}|r{reduce_rate:.2f}_{wname}#{s}", "group": f"reduce_{wname}",
                                 "start": ws, "end": END, "mode": "frictions", "seed": 2000 + s,
                                 "frictions": {"factor": factor, "reduce": reduce_rate},
                                 "variant": factor, "reduce_rate": reduce_rate, "offset": s})

    t0 = time.time()
    res = {}
    workers = max(1, min(20, (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(max_workers=workers, initializer=R._init, initargs=(str(inputs), str(panel_path))) as ex:
        for r in ex.map(R.run_job, jobs, chunksize=1):
            res[r["name"]] = r
    inputs.unlink(missing_ok=True)
    R.log_variant_trials(root / "state" / "history" / "trials.jsonl", jobs, res, "audit-2026-10-17")
    rows = []
    for j in jobs:
        r = res[j["name"]]
        st, t = R.full_stats(r["equity"]), r["trades"]
        row = {**{k: j.get(k) for k in ("name", "group", "variant", "offset", "reduce_rate", "start")},
               "n_trades": r["n_trades"], "total": st["total"], "cagr": st["cagr"], "maxdd": st["maxdd"], "sharpe": st["sharpe"],
               **R.excess_vs_spy(r["equity"], spy)}
        if t is not None and len(t):
            stops = t[t["reason"].isin(["stop", "stop-gap"])]
            decay = t[t["reason"] == "decay"]
            risk = (t["realized_risk"] / t["intended_risk"]).dropna()
            row.update(stop_share=len(stops) / len(t), win_rate=float((t["pnl"] > 0).mean()),
                       decay_n=len(decay), decay_avg_pct=float(decay["pnl_pct"].mean()) if len(decay) else None,
                       risk_ratio_p10=float(risk.quantile(0.1)) if len(risk) else None,
                       risk_ratio_median=float(risk.median()) if len(risk) else None)
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(out / "runs.csv", index=False)
    print(f"{len(jobs)} runs in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()

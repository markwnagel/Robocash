#!/usr/bin/env python
"""Point-in-time check of two stop changes after the live desk's stop-outs kept rebounding (2026-10-13):
  - stop 2.5x ATR with the target held at 6x ATR (2.4R), the variant the 2026-09-24 study left untested
  - a stop judged on the close and sold at the next open, alone and with a 3x ATR resting stop under it
Reuses robustness.py's runner; writes state/robustness/2026-10-13-stops/. Each variant is logged to trials.jsonl (robustness.log_variant_trials).

  python scripts/check_stop_variants.py      ~60 runs; summary: see config.yaml comments
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

STOP, TP = ("exits", "stop_atr_multiple"), ("exits", "take_profit_r_multiple")
ON, HARD = ("exits", "stop_on"), ("exits", "disaster_atr_multiple")
VARIANTS = {
    "today: 2.0 intraday, 3R": [],
    "2.5 intraday, 3R (tested 09-24)": [(STOP, 2.5)],
    "2.5 intraday, 2.4R (same target)": [(STOP, 2.5), (TP, 2.4)],
    "2.0 on close": [(ON, "close")],
    "2.0 on close + 3.0 resting": [(ON, "close"), (HARD, 3.0)],
}
STOP_REASONS = ("stop", "stop-gap", "stop-close")


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
    out = root / "state" / "robustness" / "2026-10-13-stops"; out.mkdir(parents=True, exist_ok=True)
    inputs = out / "_inputs.pkl"
    pd.to_pickle({"cfg": cfg, "spy": spy, "vix": vix, "earnings": earnings, "capital": float(cfg["account"]["starting_capital"]),
                  "members": {"pit": members_pit, "frozen": pd.DataFrame(True, index=dates, columns=frozen)}}, inputs)

    jobs = []
    for wname, ws, universe in (("pit_full", FULL, "pit"), ("pit_recent", RECENT, "pit"), ("frozen_recent", RECENT, "frozen")):
        starts = [str((pd.Timestamp(ws) + pd.DateOffset(months=3 * k)).date()) for k in range(4)]
        for lbl, ov in VARIANTS.items():
            for k, st in enumerate(starts):
                jobs.append({"name": f"{lbl}|{wname}@{k}", "group": wname, "start": st, "end": END, "universe": universe,
                             "overrides": ov, "keep_trades": True, "variant": lbl, "offset": k})

    t0 = time.time()
    res = {}
    workers = max(1, min(20, (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(max_workers=workers, initializer=R._init, initargs=(str(inputs), str(panel_path))) as ex:
        for r in ex.map(R.run_job, jobs, chunksize=1):
            res[r["name"]] = r
    inputs.unlink(missing_ok=True)
    R.log_variant_trials(root / "state" / "history" / "trials.jsonl", jobs, res, "stops-2026-10-13")
    rows = []
    for j in jobs:
        r = res[j["name"]]
        st, t = R.full_stats(r["equity"]), r["trades"]
        stops = t[t["reason"].isin(STOP_REASONS)] if len(t) else t
        rows.append({"variant": j["variant"], "window": j["group"], "offset": j["offset"], "start": j["start"],
                     "n_trades": r["n_trades"], "total": st["total"], "cagr": st["cagr"], "maxdd": st["maxdd"],
                     "sharpe": st["sharpe"], **R.excess_vs_spy(r["equity"], spy), "win_rate": float((t["pnl"] > 0).mean()) if len(t) else None,
                     "stop_share": len(stops) / len(t) if len(t) else None,
                     "avg_stop_loss_pct": float(stops["pnl_pct"].mean()) if len(stops) else None,
                     "median_hold": float(t["hold_days"].median()) if len(t) else None})
    df = pd.DataFrame(rows); df.to_csv(out / "runs.csv", index=False)
    pd.to_pickle({n: r["trades"] for n, r in res.items()}, out / "trades.pkl")
    print(f"{len(jobs)} runs in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Does the backtest's $10k account (account.starting_capital) tell the same story as the live $100k account?

The strategy sizes by risk and a percent cap, so in theory it scales. In practice whole shares, the cash floor and
the last-picks-of-a-day cash cut (monitor F-55) act differently at $10k and $100k: at $10k a high-priced name can
round to zero or one share, and slivers are a bigger share of the book. This runs today's config.yaml unchanged at
both sizes from the same 8 starts (4 a quarter apart in 2018-26 and in 2023-26, point-in-time universe).

  python scripts/check_capital.py
  python scripts/check_capital.py --capitals 10000,25000,100000

Capital is not a strategy setting, so nothing is written to trials.jsonl (both sizes share one config id).
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


def trade_facts(trades) -> dict:
    """Entry-size facts from a run's trade list: median entry in dollars and as a share of equity at entry is not
    kept per trade, so this reports shares and dollars bought, which is what whole-share rounding bites on."""
    df = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame(trades or [])
    if df.empty:
        return {"one_share": 0.0, "median_entry_usd": float("nan")}
    qty = next((c for c in ("qty", "shares", "quantity") if c in df), None)
    px = next((c for c in ("entry_px", "entry_price", "entry") if c in df), None)
    out = {}
    out["one_share"] = float((df[qty] <= 1).mean()) if qty else float("nan")
    out["median_entry_usd"] = float((df[qty] * df[px]).median()) if qty and px else float("nan")
    return out


def run_at(capital, cfg, spy, vix, earnings, members_pit, panel_path, out, jobs):
    inputs = out / f"_inputs_{int(capital)}.pkl"
    pd.to_pickle({"cfg": cfg, "spy": spy, "vix": vix, "earnings": earnings, "capital": capital,
                  "members": {"pit": members_pit}}, inputs)
    with ProcessPoolExecutor(max_workers=max(1, min(20, (os.cpu_count() or 2) - 2)),
                             initializer=R._init, initargs=(str(inputs), str(panel_path))) as ex:
        res = {r["name"]: r for r in ex.map(R.run_job, jobs, chunksize=1)}
    inputs.unlink(missing_ok=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capitals", default="10000,100000")
    args = ap.parse_args()
    capitals = [float(c) for c in args.capitals.split(",")]
    cfg = load_config()
    assert cfg["account"]["paper"] is True
    root = Path(cfg["_root"]); hist = root / "state" / "history"
    build = json.loads((hist / "pit" / "build.json").read_text())
    panel_path = root / build["prices"]
    dates = pd.read_pickle(panel_path)["close"].index
    members_pit = pit.expand(pd.read_csv(hist / "pit" / "members.csv"), dates)
    from alpaca.data.historical import StockHistoricalDataClient
    from desk.config import alpaca_keys
    from desk.regime import vix_series
    spy = B.frozen_prices(StockHistoricalDataClient(*alpaca_keys()), ["SPY"], 2750, cfg, pd.Timestamp(END))[0]["close"]["SPY"].dropna()
    vix = vix_series(root / cfg["data"]["cache_dir"])
    earnings = pd.read_csv(hist / "earnings.csv") if (hist / "earnings.csv").exists() else None
    out = root / "state" / "robustness" / "2026-11-14-capital"; out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for wname, ws in WINDOWS.items():
        for k in range(4):
            st = str((pd.Timestamp(ws) + pd.DateOffset(months=3 * k)).date())
            jobs.append({"name": f"{wname}@{k}", "group": wname, "start": st, "end": END, "offset": k,
                         "keep_trades": True, "overrides": []})
    rows, t0 = [], time.time()
    for cap in capitals:
        res = run_at(cap, cfg, spy, vix, earnings, members_pit, panel_path, out, jobs)
        for j in jobs:
            r = res[j["name"]]; st = R.full_stats(r["equity"])
            rows.append({"capital": cap, "window": j["group"], "offset": j["offset"], "start": j["start"],
                         "n_trades": r["n_trades"], "cagr": st["cagr"], "maxdd": st["maxdd"], "sharpe": st["sharpe"],
                         **R.excess_vs_spy(r["equity"], spy), "invested": float(1 - (r["cash"] / r["equity"]).mean()),
                         "positions": float(r["n_positions"].mean()), **trade_facts(r["trades"])})
    df = pd.DataFrame(rows); df.to_csv(out / "runs.csv", index=False)
    print(f"{len(jobs) * len(capitals)} runs in {time.time() - t0:.0f}s -> {out}")
    base_cap = capitals[0]
    for w, g in df.groupby("window", sort=False):
        base = g[g.capital == base_cap].set_index("offset")
        for cap in capitals:
            x = g[g.capital == cap].set_index("offset")
            print(f"{w:6} ${cap:>9,.0f}  CAGR {x.cagr.median():6.2%} (higher {int((x.cagr > base.cagr).sum())}/4)  "
                  f"max DD {x.maxdd.median():6.1%} (shallower {int((x.maxdd > base.maxdd).sum())}/4)  "
                  f"Sharpe {x.sharpe.median():4.2f}  6m vs SPY median {x.excess_6m_median.median():+6.2%}  "
                  f"p10 {x.excess_6m_p10.median():+6.2%}  trades {x.n_trades.median():.0f}  "
                  f"invested {x.invested.median():4.0%}  positions {x.positions.median():4.1f}  "
                  f"1-share entries {x.one_share.median():4.1%}")


if __name__ == "__main__":
    main()

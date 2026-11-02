#!/usr/bin/env python
"""Quant review 2026-11-01: candidate changes for beating SPY with a higher risk tolerance.

Pre-registered before any result was seen (one run per variant; no tuning after the fact):
  dd25          kill switch 15% -> 25%, hard floor 25% -> 35%
  dd_floor      kill switch off (99%); the 25% all-time hard floor is the only drawdown stop
  cap15_r125    per-position cap 10% -> 15% (trim 15/10 -> 20/15) with 1.25% risk per trade
  cap15_r150    the same with 1.5% risk per trade
  decay80       signal_decay_rank 40 -> 80 (hold winners longer)
  slow_wide     2026-10-27 exploratory direction: stop 4x ATR, no target, decay 80, hold 21-126 days,
                20 names at a 7% cap, 6-month momentum lookback kept (the 252-day lookback lost alone)
  regime_full   risk_off_scale 0.5 -> 1.0 (no cut in target positions when SPY < MA or VIX high)
Combinations are run by --combo after the singles are read; see the PR.

4 starts a quarter apart in 2018-26 and 2023-26, point-in-time universe with delistings. Bar (owner, with the
2026-11-01 risk mandate): median six-month excess vs SPY better in >= 3/4 starts in BOTH windows; p10 no worse in
3/4; median max drawdown no deeper than 1.5x today's. Also reported: excess vs SPMO (the "worth running" bar),
invested share (exposure) and trades per year. Every variant is logged to trials.jsonl.

  python scripts/check_quant_review_1101.py [--combo NAME ...] [--sweep]
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

SINGLES = {
    "dd25": [(("risk", "max_drawdown_pct"), 0.25), (("risk", "max_cumulative_drawdown_pct"), 0.35)],
    "dd_floor": [(("risk", "max_drawdown_pct"), 0.99)],
    "cap15_r125": [(("portfolio", "max_position_pct"), 0.15), (("portfolio", "trim_above_pct"), 0.20),
                   (("portfolio", "trim_to_pct"), 0.15), (("portfolio", "risk_per_trade_pct"), 0.0125)],
    "cap15_r150": [(("portfolio", "max_position_pct"), 0.15), (("portfolio", "trim_above_pct"), 0.20),
                   (("portfolio", "trim_to_pct"), 0.15), (("portfolio", "risk_per_trade_pct"), 0.015)],
    "decay80": [(("exits", "signal_decay_rank"), 80)],
    "slow_wide": [(("exits", "stop_atr_multiple"), 4.0), (("exits", "take_profit_r_multiple"), 0.0),
                  (("exits", "signal_decay_rank"), 80), (("exits", "min_hold_days"), 21),
                  (("exits", "max_hold_days"), 126), (("portfolio", "target_positions"), 20),
                  (("portfolio", "max_positions"), 25), (("portfolio", "max_position_pct"), 0.07)],
    "regime_full": [(("regime", "risk_off_scale"), 1.0)],
}
SWEEP = [(("portfolio", "cash_sweep"), {"enabled": True, "band_pct": 0.02})]

# Round 2, registered after round 1 showed every risk increase losing and every variant trailing SPMO by 5-9pt:
# is the gap the construction (12 names, tight stops, ~200 trades/yr) or the universe (800 names incl. mid caps)?
SLOW = [(("exits", "stop_atr_multiple"), 4.0), (("exits", "take_profit_r_multiple"), 0.0),
        (("exits", "signal_decay_rank"), 80), (("exits", "min_hold_days"), 21), (("exits", "max_hold_days"), 126)]
SPMO_LIKE = [(("signals", "momentum_lookback_days"), 252),
             (("signals", "weights"), {"momentum": 1.0, "short": 0.0, "low_vol": 0.0}),
             (("signals", "max_atr_pct"), 0.15),
             (("portfolio", "sizing"), "notional"), (("portfolio", "vol_weighting"), False),
             (("portfolio", "target_positions"), 25), (("portfolio", "max_positions"), 30),
             (("portfolio", "max_position_pct"), 0.06),
             (("exits", "stop_atr_multiple"), 10.0), (("exits", "take_profit_r_multiple"), 0.0),
             (("exits", "signal_decay_rank"), 100), (("exits", "min_hold_days"), 63), (("exits", "max_hold_days"), 252)]
ROUND2 = {
    "slow_wide252": SLOW + [(("signals", "momentum_lookback_days"), 252), (("portfolio", "target_positions"), 20),
                            (("portfolio", "max_positions"), 25), (("portfolio", "max_position_pct"), 0.07)],
    "spmo_like_800": SPMO_LIKE,
    "spmo_like_top200": SPMO_LIKE,          # run on the top-200 universe (JOB_EXTRA)
    "sweep_spmo": SWEEP,                     # idle cash into SPMO instead of SPY (JOB_EXTRA)
}
LIVE_FRICTIONS = {"remove": 0.10, "reduce": 0.0, "factor": 1.0, "half": 0.42}
JOB_EXTRA = {"spmo_like_top200": {"universe": "top200"}, "sweep_spmo": {"sweep_with": "SPMO"}}


def combos(names):
    out = {}
    for n in names:
        parts = n.split("+")
        ov = []
        for p in parts:
            ov += SWEEP if p == "sweep" else {**SINGLES, **ROUND2}[p]
        out[n] = ov
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", nargs="*", default=None, help="run these '+'-joined variants instead of the singles")
    ap.add_argument("--round2", action="store_true", help="the construction variants (ROUND2)")
    ap.add_argument("--starts", type=int, default=4, help="starts per window")
    ap.add_argument("--step-months", type=int, default=3, help="months between starts")
    ap.add_argument("--frictions", type=int, default=0,
                    help="N paired seeds with the live frictions (half-size 42%%, REMOVE 10%%); median per start")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="quant-review-2026-11-01")
    args = ap.parse_args()
    cfg = load_config()
    assert cfg["account"]["paper"] is True
    variants = {"base": [], **(combos(args.combo) if args.combo else ROUND2 if args.round2 else SINGLES)}
    root = Path(cfg["_root"]); hist = root / "state" / "history"
    build = json.loads((hist / "pit" / "build.json").read_text())
    panel_path = root / build["prices"]
    panel = pd.read_pickle(panel_path); dates = panel["close"].index
    members_pit = pit.expand(pd.read_csv(hist / "pit" / "members.csv"), dates)
    from alpaca.data.historical import StockHistoricalDataClient
    from desk.config import alpaca_keys
    from desk.regime import vix_series
    px = B.frozen_prices(StockHistoricalDataClient(*alpaca_keys()), ["SPY", "SPMO", "MTUM"], 2750, cfg,
                         pd.Timestamp(END))[0]["close"]
    spy, spmo, mtum = (px[s].dropna() for s in ("SPY", "SPMO", "MTUM"))
    vix = vix_series(root / cfg["data"]["cache_dir"])
    earnings = pd.read_csv(hist / "earnings.csv") if (hist / "earnings.csv").exists() else None
    # Large-cap universe: each day, the 200 point-in-time members with the highest 20-day dollar volume.
    dv = (panel["close"] * panel["volume"]).rolling(20, min_periods=10).mean().where(members_pit.reindex(
        index=dates, columns=panel["close"].columns, fill_value=False))
    top200 = dv.rank(axis=1, ascending=False) <= 200
    out = root / "state" / "robustness" / (args.out or ("2026-11-01-quant" + ("-combo" if args.combo else "-r2" if args.round2 else "")))
    out.mkdir(parents=True, exist_ok=True)
    inputs = out / "_inputs.pkl"
    pd.to_pickle({"cfg": cfg, "spy": spy, "vix": vix, "earnings": earnings,
                  "capital": float(cfg["account"]["starting_capital"]), "members": {"pit": members_pit, "top200": top200},
                  "sweeps": {"SPY": spy, "SPMO": spmo}}, inputs)
    jobs = []
    seeds = range(args.frictions) if args.frictions else [None]
    for wname, ws in WINDOWS.items():
        for k in range(args.starts):
            st = str((pd.Timestamp(ws) + pd.DateOffset(months=args.step_months * k)).date())
            for v, ov in variants.items():
                extra = {}
                for part in v.split("+"):
                    extra.update(JOB_EXTRA.get(part, {}))
                for sd in seeds:
                    fr = ({"mode": "frictions", "seed": 5000 + 100 * k + sd, "frictions": LIVE_FRICTIONS}
                          if sd is not None else {})
                    jobs.append({"name": f"{v}|{wname}@{k}#{sd}", "group": wname, "start": st, "end": END,
                                 "variant": v, "offset": k, "overrides": ov, **extra, **fr})
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(20, (os.cpu_count() or 2) - 2)),
                             initializer=R._init, initargs=(str(inputs), str(panel_path))) as ex:
        res = {r["name"]: r for r in ex.map(R.run_job, jobs, chunksize=1)}
    inputs.unlink(missing_ok=True)
    R.log_variant_trials(root / "state" / "history" / "trials.jsonl", [j for j in jobs if j["variant"] != "base"],
                         res, args.tag)
    rows = []
    for j in jobs:
        r = res[j["name"]]; st = R.full_stats(r["equity"]); eq = r["equity"].dropna()
        years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
        vs_spmo = R.excess_vs_spy(r["equity"], spmo)
        rows.append({"variant": j["variant"], "window": j["group"], "offset": j["offset"], "n_trades": r["n_trades"],
                     "trades_yr": r["n_trades"] / years, "cagr": st["cagr"], "maxdd": st["maxdd"],
                     "sharpe": st["sharpe"], **R.excess_vs_spy(r["equity"], spy),
                     "spmo_6m_median": vs_spmo["excess_6m_median"], "spmo_cagr_excess": vs_spmo["cagr_excess"],
                     "invested": float(1 - (r["cash"] / r["equity"]).mean())})
    df = pd.DataFrame(rows); df.to_csv(out / "runs.csv", index=False)
    df = df.groupby(["variant", "window", "offset"], sort=False).median(numeric_only=True).reset_index()
    print(f"{len(jobs)} runs in {time.time() - t0:.0f}s -> {out}")
    for w, g in df.groupby("window", sort=False):
        base = g[g.variant == "base"].set_index("offset")
        print(f"\n{w}")
        for v in variants:
            x = g[g.variant == v].set_index("offset")
            ok_ex = int((x.excess_6m_median > base.excess_6m_median).sum())
            ok_p10 = int((x.excess_6m_p10 >= base.excess_6m_p10).sum())
            dd_ok = x.maxdd.median() >= 1.5 * base.maxdd.median()
            n = len(x)
            print(f"  {v:28} 6m vs SPY {x.excess_6m_median.median():+6.2%} ({ok_ex}/{n})  p10 {x.excess_6m_p10.median():+7.2%} "
                  f"({ok_p10}/{n})  maxDD {x.maxdd.median():6.1%}{'' if dd_ok else ' X'}  CAGR {x.cagr.median():6.2%}  "
                  f"vs SPMO {x.spmo_6m_median.median():+6.2%}  inv {x.invested.median():3.0%}  tr/yr {x.trades_yr.median():4.0f}")


if __name__ == "__main__":
    main()

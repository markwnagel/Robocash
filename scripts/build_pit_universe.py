#!/usr/bin/env python
"""Build the point-in-time universe and its price history for robustness.py.

  python scripts/build_pit_universe.py                      # 2017-01 to 2026-09-11, resumable
  python scripts/build_pit_universe.py --end 2026-09-11 --refetch-prices

Ticker lists come from two places. Alpaca serves bars for delisted names (SIVB, FRC, TWTR, ATVI) but its
asset list leaves many of them out, so the other source is the SEC's company_tickers.json as the Wayback
Machine archived it about once a month since 2017-08: every company that filed with the SEC under a ticker
in that time. Monthly bars (unadjusted and split-adjusted) screen the whole pool; daily bars are fetched
only for names that came near the liquidity bar in some month; desk.pit.weekly_members() then applies the
live rule week by week.

Outputs, under state/history/:
  pit/members.csv    symbol, first_week, last_week - the universe definition (committed)
  pit/build.json     parameters, sources and counts (committed)
  pit/sec/, pit/pool.json, pit/monthly_*.pkl   download caches (not committed)
  prices/pit_panel_<start>_to<end>.pkl         daily split-adjusted bars (not committed, ~300 MB)
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpaca.common.exceptions import APIError  # noqa: E402
from alpaca.data.enums import Adjustment, DataFeed  # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # noqa: E402
from alpaca.trading.client import TradingClient  # noqa: E402

from desk import pit  # noqa: E402
from desk.config import alpaca_keys, load_config, require_paper  # noqa: E402
from desk.data import fetch_daily_bars  # noqa: E402

log = logging.getLogger("build_pit")
CDX = ("https://web.archive.org/cdx/search/cdx?url=sec.gov/files/company_tickers.json&output=json"
       "&from={y0}&to={y1}&filter=statuscode:200&collapse=timestamp:6")
RAW = "https://web.archive.org/web/{ts}id_/https://www.sec.gov/files/company_tickers.json"
UA = {"User-Agent": "alpaca-desk-backtest/1.0 (research; point-in-time universe)"}


def _get(url: str, tries: int = 5) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                body = r.read()
            # The archive stores some snapshots as the gzip bytes SEC sent, with no Content-Encoding header.
            return gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body
        except Exception as e:
            if k == tries - 1:
                raise
            log.warning("retry %d for %s: %s", k + 1, url[:90], e)
            time.sleep(30 * (k + 1))          # the archive refuses connections when crawled fast
    raise RuntimeError("unreachable")


def sec_snapshots(folder: Path, y0: int, y1: int) -> dict:
    """{archive timestamp: {ticker: title}} from the Wayback Machine, cached one file per snapshot."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = json.loads(_get(CDX.format(y0=y0, y1=y1)))[1:]
    out = {}
    for r in rows:
        ts = r[1]
        p = folder / f"{ts}.json"
        if not p.exists():
            try:
                raw = json.loads(_get(RAW.format(ts=ts)))
            except Exception as e:           # one bad snapshot costs a month of coverage, not the build
                log.warning("snapshot %s unusable: %s", ts, e)
                continue
            p.write_text(json.dumps(raw))
            time.sleep(4.0)                   # archive.org throttles faster crawls
        raw = json.loads(p.read_text())
        out[ts] = {str(v["ticker"]).upper(): v.get("title", "") for v in raw.values()}
    log.info("SEC ticker snapshots: %d (%s to %s)", len(out), min(out), max(out))
    return out


def candidate_pool(trading, snapshots: dict, bad: list) -> dict:
    """{symbol: {name, exchange, source}}. Alpaca's name and venue win where Alpaca knows the symbol, so the
    filters match the live universe exactly for names that still trade."""
    pool = {}
    for ts in sorted(snapshots):
        for sym, title in snapshots[ts].items():
            pool[sym] = {"name": title, "exchange": None, "source": "sec"}
    # The raw list, as desk/data.py reads it: get_all_assets() rejects the whole list when one asset sits on an
    # exchange the SDK does not know (TXSE, 2026-10-19).
    for status in ("active", "inactive"):
        for a in trading.get("/assets", {"status": status, "asset_class": "us_equity"}) or []:
            pool[a["symbol"]] = {"name": a.get("name") or "", "exchange": a.get("exchange") or "",
                                 "source": "alpaca-" + status}
    kept = {s: v for s, v in pool.items() if pit.keep_symbol(s, v["name"], v["exchange"], bad)}
    log.info("candidate pool: %d symbols seen, %d pass the name/venue rules", len(pool), len(kept))
    return kept


def monthly_bars(data, symbols: list, start: datetime, end: datetime, adjustment: Adjustment) -> pd.DataFrame:
    frames = []

    def pull(chunk: list) -> None:
        req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame(1, TimeFrameUnit.Month),
                               start=start, end=end, adjustment=adjustment, feed=DataFeed.SIP)
        try:
            df = data.get_stock_bars(req).df
        except APIError as e:
            if len(chunk) == 1:
                log.debug("no monthly bars for %s: %s", chunk[0], e)
                return
            half = len(chunk) // 2          # one malformed symbol rejects the whole request
            pull(chunk[:half])
            pull(chunk[half:])
            return
        if not df.empty:
            frames.append(df.reset_index())

    for i in range(0, len(symbols), 200):
        pull(symbols[i:i + 200])
        if (i // 200) % 10 == 0:
            log.info("monthly %s: %d/%d", adjustment.value, min(i + 200, len(symbols)), len(symbols))
    df = pd.concat(frames, ignore_index=True)
    df["month"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.to_period("M")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-start", default="2017-01-01", help="first daily bar (signal warm-up needs ~9 months)")
    ap.add_argument("--first-week", default="2017-12-01", help="first week with a membership list")
    ap.add_argument("--end", default="2026-09-11")
    ap.add_argument("--refetch-prices", action="store_true")
    ap.add_argument("--rebuild-pool", action="store_true", help="re-apply the name rules to the cached ticker lists")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg = load_config()
    u = cfg["universe"]
    root = Path(cfg["_root"]) / "state" / "history"
    folder = root / "pit"
    folder.mkdir(parents=True, exist_ok=True)
    key, secret = alpaca_keys()
    trading = TradingClient(key, secret, paper=require_paper(cfg))
    data = StockHistoricalDataClient(key, secret)
    d0, first_week, end = pd.Timestamp(args.data_start), pd.Timestamp(args.first_week), pd.Timestamp(args.end)

    pool_path = folder / "pool.json"
    if pool_path.exists() and not args.rebuild_pool:
        pool = json.loads(pool_path.read_text())
    else:
        snaps = sec_snapshots(folder / "sec", d0.year, end.year)
        pool = candidate_pool(trading, snaps, u["exclude_name_keywords"])
        pool_path.write_text(json.dumps(pool))
    symbols = sorted(pool)

    m0 = datetime(first_week.year, first_week.month, 1, tzinfo=timezone.utc) - pd.DateOffset(months=1)
    m1 = datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + pd.Timedelta(days=1)
    monthly = {}
    for adj in (Adjustment.RAW, Adjustment.SPLIT):
        p = folder / f"monthly_{adj.value}.pkl"
        if not p.exists():
            pd.to_pickle(monthly_bars(data, symbols, m0, m1, adj), p)
        m = pd.read_pickle(p)
        monthly[adj.value] = m[m["symbol"].isin(pool)]      # the cache may predate a narrower pool
    raw_m = monthly["raw"].pivot_table(index="month", columns="symbol", values="close", aggfunc="last")
    adj_m = monthly["split"].pivot_table(index="month", columns="symbol", values="close", aggfunc="last")
    rm = monthly["raw"]
    days = pd.Series({m: len(pd.bdate_range(m.start_time, m.end_time)) for m in rm["month"].unique()})
    dv_m = (rm.assign(dv=rm["vwap"] * rm["volume"] / rm["month"].map(days))
            .pivot_table(index="month", columns="symbol", values="dv", aggfunc="last"))
    superset = pit.monthly_superset(dv_m, raw_m, u["min_avg_dollar_volume"] * 0.5, u["min_price"] * 0.8,
                                    int(u["max_symbols"] * 1.5))
    log.info("superset for daily bars: %d names with monthly data %d", len(superset), dv_m.shape[1])

    prices = root / "prices"
    prices.mkdir(parents=True, exist_ok=True)
    ppath = prices / f"pit_panel_{d0.date()}_to{end.date()}.pkl"
    if ppath.exists() and not args.refetch_prices:
        panel = pd.read_pickle(ppath)
    else:
        start = datetime(d0.year, d0.month, d0.day, tzinfo=timezone.utc)
        stop = min(datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + pd.Timedelta(days=1),
                   datetime.now(timezone.utc) - pd.Timedelta(minutes=16))
        panel = fetch_daily_bars(data, sorted(superset), 0, cfg["data"]["feed"], end=stop, start=start)
        pd.to_pickle(panel, ppath)
    keep = [c for c in panel["close"].columns if c in superset]     # likewise for a cached panel
    close, vol = panel["close"][keep], panel["volume"][keep]

    # adjusted x (raw / adjusted, that month) = the price that traded, for the min-price rule
    ratio = (raw_m / adj_m).reindex(columns=close.columns)
    factor = ratio.reindex(close.index.to_period("M"))
    factor.index = close.index
    weeks = pit.week_starts(close.index)
    weeks = weeks[(weeks >= first_week) & (weeks <= end)]
    members = pit.weekly_members(close, vol, factor, weeks, u["min_avg_dollar_volume"], u["min_price"],
                                 u["max_symbols"])
    iv = pit.to_intervals(members)
    iv.assign(first_week=iv["first_week"].dt.date, last_week=iv["last_week"].dt.date).to_csv(folder / "members.csv", index=False)

    frozen = set(json.loads((root / "universe_bt.json").read_text())["symbols"])
    ever = set(iv["symbol"])
    last = members[max(members)]
    gone = sorted(s for s in ever if close[s].last_valid_index() is not None
                  and close[s].last_valid_index() < end - pd.Timedelta(days=10))
    info = {"built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_start": str(d0.date()), "first_week": str(weeks[0].date()), "last_week": str(weeks[-1].date()),
            "end": str(end.date()), "weeks": len(weeks), "rule": {k: u[k] for k in ("min_avg_dollar_volume", "min_price", "max_symbols")},
            "pool": len(pool), "pool_sources": pd.Series([v["source"] for v in pool.values()]).value_counts().to_dict(),
            "superset": len(superset), "ever_members": len(ever), "members_last_week": len(last),
            "members_median": int(pd.Series([len(v) for v in members.values()]).median()),
            "delisted_members": len(gone), "delisted_examples": gone[:40],
            "frozen_overlap_last_week": len(frozen & set(last)),
            "frozen_never_member": sorted(frozen - ever),
            "prices": f"state/history/prices/{ppath.name}"}
    (folder / "build.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k not in ("delisted_examples", "frozen_never_member")}, indent=1))
    print(f"delisted examples: {', '.join(gone[:40])}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

log = logging.getLogger("desk.data")
CHUNK = 200
FIELDS = ["open", "high", "low", "close", "volume"]


def list_candidate_assets(trading: TradingClient, ucfg: dict) -> list:
    assets = trading.get_all_assets(
        GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    )
    exchanges = set(ucfg["exchanges"])
    bad = [k.lower() for k in ucfg["exclude_name_keywords"]]
    out = set()
    for a in assets:
        if a.exchange.value not in exchanges or not a.tradable:
            continue
        if not a.symbol.isalpha() or len(a.symbol) > 5:
            continue
        name = (a.name or "").lower()
        if any(k in name for k in bad):
            continue
        out.add(a.symbol)
    return sorted(out)


def fetch_daily_bars(data: StockHistoricalDataClient, symbols: list, days: int,
                     feed: str = "sip", end: datetime | None = None) -> dict:
    """Returns {'open','high','low','close','volume'} -> DataFrame[date x symbol]."""
    # Free-tier SIP requires the request window to end >= 15 minutes in the past.
    end = end or (datetime.now(timezone.utc) - timedelta(minutes=16))
    start = end - timedelta(days=int(days * 1.45) + 10)
    feed_state = {"feed": DataFeed(feed)}

    def pull(syms: list) -> list:
        out = []
        for i in range(0, len(syms), CHUNK):
            chunk = list(syms[i:i + CHUNK])
            req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                                   start=start, end=end, adjustment=Adjustment.SPLIT, feed=feed_state["feed"])
            try:
                bars = data.get_stock_bars(req)
            except APIError as e:
                if feed_state["feed"] != DataFeed.IEX and "subscription" in str(e).lower():
                    log.warning("SIP feed not permitted; falling back to IEX (volumes will be partial)")
                    feed_state["feed"] = req.feed = DataFeed.IEX
                    bars = data.get_stock_bars(req)
                else:
                    raise
            df = bars.df
            if df.empty:
                continue
            out.append(df.reset_index())
            log.info("bars: %d/%d symbols", min(i + CHUNK, len(syms)), len(syms))
        return out

    def with_data(frames: list) -> set:
        return set().union(*(set(f.loc[f["close"].notna(), "symbol"]) for f in frames)) if frames else set()

    frames = pull(symbols)
    missing = [s for s in symbols if s not in with_data(frames)]
    if missing and frames:
        # The feed occasionally returns nothing for one name in an otherwise good response (TEVA,
        # 2026-09-24: every day blank). Nothing downstream can tell "no data" from "not ranked", so live
        # the desk would sell a holding as signal decay, and a backtest silently trades a different universe.
        log.warning("no bars for %d symbol(s), retrying once: %s", len(missing), ", ".join(missing[:10]))
        frames += pull(missing)
        missing = [s for s in symbols if s not in with_data(frames)]
        if missing:
            log.warning("STILL no bars for %d symbol(s) - absent from this panel: %s",
                        len(missing), ", ".join(missing[:20]))
    if not frames:
        raise RuntimeError("no bar data returned")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    panel = {}
    for f in FIELDS:
        panel[f] = df.pivot_table(index="date", columns="symbol", values=f, aggfunc="last").sort_index()
    return panel


def cached_bars(data, symbols, days, cfg, tag="daily", asof=None) -> dict:
    cache_dir = Path(cfg["_root"]) / cfg["data"]["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    asof = asof or date.today()
    # Hash the symbols themselves, not just how many there are: two runs on one day can have the same
    # count and different members (a holding leaves the universe as another joins), and reusing the wrong
    # panel drops a held name out of the snapshot, where the exit logic reads it as signal decay and sells.
    digest = hashlib.sha1(",".join(sorted(symbols)).encode()).hexdigest()[:10]
    key = cache_dir / f"bars_{tag}_{asof.isoformat()}_{days}_{len(symbols)}_{digest}.pkl"
    if key.exists():
        return pd.read_pickle(key)
    panel = fetch_daily_bars(data, symbols, days, cfg["data"]["feed"])
    pd.to_pickle(panel, key)
    prune_bar_cache(cache_dir, int(cfg["data"].get("cache_retention_days", 2)))
    return panel


def prune_bar_cache(cache_dir: Path, keep_days: int = 2) -> int:
    """Delete bar panels too old to be reachable. Every key carries the date it was fetched and only
    today's is ever looked up, so yesterday's panel is already dead weight - and one universe panel per
    run adds up fast (243MB inside a fortnight). Touches bars_*.pkl only: fact sheets, the earnings
    calendar, the EDGAR store and the VIX history live in the same directory and are not regenerated
    as cheaply."""
    if keep_days <= 0:
        return 0
    cutoff = time.time() - keep_days * 86400
    removed = 0
    for p in cache_dir.glob("bars_*.pkl"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError as e:       # housekeeping must never take a run down with it
            log.debug("cache prune skipped %s: %s", p.name, e)
    if removed:
        log.info("pruned %d stale bar cache file(s)", removed)
    return removed


def build_universe(trading, data, cfg, force=False, cache_name="universe.json") -> list:
    ucfg = cfg["universe"]
    cache = Path(cfg["_root"]) / "state" / cache_name
    if cache.exists() and not force:
        u = json.loads(cache.read_text())
        age = (date.today() - date.fromisoformat(u["date"])).days
        if age < ucfg["refresh_days"]:
            return u["symbols"]
    log.info("refreshing universe")
    syms = list_candidate_assets(trading, ucfg)
    panel = fetch_daily_bars(data, syms, 30, cfg["data"]["feed"])
    close, vol = panel["close"], panel["volume"]
    dollar_vol = (close * vol).tail(20).mean()
    last = close.ffill().iloc[-1]
    ok = dollar_vol[(dollar_vol >= ucfg["min_avg_dollar_volume"]) & (last >= ucfg["min_price"])]
    symbols = list(ok.sort_values(ascending=False).index[: ucfg["max_symbols"]])
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps({"date": date.today().isoformat(), "symbols": symbols}))
    log.info("universe: %d symbols", len(symbols))
    return symbols

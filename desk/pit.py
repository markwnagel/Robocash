"""Point-in-time universe: the names that were liquid at each date, including ones that no longer trade.

The frozen snapshot (state/history/universe_bt.json) is today's 800 most liquid names. A backtest on it
already knows who survived and who grew: a name that was a $3M/day small cap in 2019 is tradable in 2019
because it is liquid in 2026. Membership here is rebuilt every week from what was known that week, with
the same rule as build_universe(): the top max_symbols names by 20-day average dollar volume, with a price
at or above min_price as it traded then (not split-adjusted).
"""
from __future__ import annotations

import pandas as pd

ETF_VENUES = {"ARCA", "BATS", "AMEX"}      # live trades NYSE and NASDAQ names only
# Exchange-traded notes file with the SEC under their issuing bank, so a delisted one (DGAZ, TVIX, XIV) carries
# the bank's name and slips past the "ETN" keyword. The banks' own shares trade as ADRs or under Alpaca-listed
# symbols, which are named by Alpaca instead, so this only ever matches notes.
ETN_ISSUERS = {"CREDIT SUISSE AG", "CREDIT SUISSE GROUP AG", "BANK OF MONTREAL /CAN/", "UBS AG", "BARCLAYS BANK PLC",
               "DEUTSCHE BANK AG", "DEUTSCHE BANK AG\\", "ROYAL BANK OF CANADA", "JPMORGAN CHASE FINANCIAL CO. LLC"}


def keep_symbol(symbol: str, name: str, exchange: str | None, bad_keywords: list) -> bool:
    """The live universe's name rules (data.list_candidate_assets). exchange is Alpaca's current listing;
    None for a delisted name Alpaca no longer lists. OTC is kept on purpose: a company that fell from
    NYSE to OTC was listed when it mattered, and dropping it would bring survivorship back in."""
    if not symbol.isalpha() or len(symbol) > 5:
        return False
    if exchange in ETF_VENUES:
        return False
    if exchange is None and (name or "").strip().upper() in ETN_ISSUERS:
        return False
    low = (name or "").lower()
    return not any(k.lower() in low for k in bad_keywords)


def monthly_superset(dollar_vol: pd.DataFrame, raw_close: pd.DataFrame, min_adv: float, min_price: float,
                     top: int) -> set:
    """Names worth downloading daily bars for: in some month, near the liquidity and price bars and among
    the `top` most liquid. dollar_vol is average dollar volume per day for the month (month x symbol).
    Generous on purpose - the exact weekly rule runs later on daily data, and a name missed here is
    missing from the universe for good."""
    out = set()
    for m in dollar_vol.index:
        dv, px = dollar_vol.loc[m], raw_close.loc[m].reindex(dollar_vol.columns)
        ok = dv[(dv >= min_adv) & (px >= min_price)].dropna()
        out.update(ok.sort_values(ascending=False).index[:top])
    return out


def week_starts(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First trading day of each calendar week (Tuesday after a Monday holiday)."""
    iso = dates.isocalendar()
    return pd.DatetimeIndex(pd.Series(dates, index=dates).groupby([iso.year.values, iso.week.values]).min().sort_values())


def weekly_members(close: pd.DataFrame, volume: pd.DataFrame, split_factor: pd.DataFrame, weeks: pd.DatetimeIndex,
                   min_adv: float, min_price: float, max_symbols: int) -> dict:
    """{week_start: [symbols]}. close/volume are split-adjusted daily bars; split_factor turns an adjusted
    close back into the price that traded (daily, same shape). Uses bars through the week's first day,
    as the live rebuild does after that day's close."""
    dv = (close * volume).rolling(20, min_periods=15).mean()
    raw = close * split_factor.reindex(index=close.index, columns=close.columns)
    out = {}
    for w in weeks:
        adv, px = dv.loc[w], raw.loc[w]
        ok = adv[(adv >= min_adv) & (px >= min_price)].dropna()
        # One company under two tickers (a rename the feed serves under both) has identical bars, so an
        # identical ADV to the cent: keep one.
        ok = ok[~ok.round(2).duplicated()]
        out[w] = list(ok.sort_values(ascending=False).index[:max_symbols])
    return out


def to_intervals(members: dict) -> pd.DataFrame:
    """Run-length form (symbol, first_week, last_week): a name is a member every week in between. Small
    enough to commit, and it diffs readably when the universe is rebuilt."""
    weeks = sorted(members)
    pos = {w: i for i, w in enumerate(weeks)}
    rows, open_ = [], {}
    for w in weeks:
        cur = set(members[w])
        for s in list(open_):
            if s not in cur:
                rows.append((s, open_.pop(s), weeks[pos[w] - 1]))
        for s in cur:
            open_.setdefault(s, w)
    rows += [(s, w0, weeks[-1]) for s, w0 in open_.items()]
    df = pd.DataFrame(rows, columns=["symbol", "first_week", "last_week"])
    return df.sort_values(["first_week", "symbol"]).reset_index(drop=True)


def expand(intervals: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily membership (date x symbol, bool). A week's list holds until the next week's is built."""
    iv = intervals.assign(first_week=pd.to_datetime(intervals["first_week"]),
                          last_week=pd.to_datetime(intervals["last_week"]))
    weeks = pd.DatetimeIndex(sorted(set(iv["first_week"]) | set(iv["last_week"])))
    ws = week_starts(dates)
    ws = ws[(ws >= weeks.min()) & (ws <= weeks.max())]
    wk = pd.DataFrame(False, index=ws, columns=sorted(iv["symbol"].unique()))
    for r in iv.itertuples():
        wk.loc[(wk.index >= r.first_week) & (wk.index <= r.last_week), r.symbol] = True
    daily = wk.reindex(dates, method="ffill")
    # After the last rebuild the list is stale, not empty: keep it for the rest of that week only.
    last = ws.max()
    daily.loc[daily.index >= last + pd.Timedelta(days=7)] = False
    return daily.fillna(False).astype(bool)

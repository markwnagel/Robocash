# alpaca-desk

Systematic, long-only, no-leverage US equity strategy for a small Alpaca account. Runs once a day after the close,
holds 8-15 names for 15-60 days with broker-side bracket orders, and has a hard-coded drawdown kill switch.
An intelligence layer turns SEC filings, news, options, and social data into structured facts; Claude plays a
red team and a veto-only analyst on top of that; a weekly auditor attacks the process itself.

**This is not investment advice and the strategy is not proven.** Run it in paper mode first and expect a year
of data before you can distinguish it from noise.

## How it works

```
run_daily.py (16:45 ET, weekdays)
  1. sync state with the broker; log any positions closed by stop/target
  2. RISK GOVERNOR  drawdown from episode high-water mark >= 15% -> liquidate, halt until --resume
                    drawdown from ALL-TIME high >= 25%           -> liquidate, halt until --resume --reset-hwm
                    day down >= 5%                               -> no new entries today
                    (--resume re-baselines the episode HWM, so each episode is capped at 15%; the
                    all-time floor is the cumulative backstop and needs an explicit acknowledgement)
  3. universe       NYSE/Nasdaq common stock, price >= $5, 20d avg dollar volume >= $20M, top 800 (cached weekly)
  4. signals        6-1 month momentum (60%), 20-day return (20%), low realized vol (20%);
                    must be above 50-day MA, positive momentum, 14-day ATR <= 8% of price
  5. REGIME GATE    SPY below its 200-day MA or VIX > 30 -> target position count halved (code, from Alpaca + FRED)
                    If the SPY/regime data fails, the halved (risk-off) target applies until it is back;
                    nothing is sold for it, even with off_liquidate.
  6. FACT SHEETS    for the top 30 candidates + holdings (desk/facts.py). Each source degrades on its own,
                    but if the sheets cannot be built at all, or the earnings calendar is unknown (no
                    FINNHUB_API_KEY, Finnhub down), no new entries are made that day:
                      SEC EDGAR   dilution filings (8-K 3.02, S-1), shelf filings (424B, S-3) flagged
                                  separately, 8-K items, Form 4 insider buys/sells
                      Finnhub     next earnings date (one query per symbol - see the note below)
                      Alpaca      Benzinga headlines (Claude-classified: tone + offering/guidance/M&A/legal flags)
                      Alpaca      options chain -> ATM implied vol vs 20d realized vol, put skew
                      StockTwits  message rate vs the name's own history (z-score), bullish share
                    HARD BLOCKS (candidates only): earnings within 5 calendar days (a report today
                      included; a failed lookup blocks too), unambiguous dilution filing, headline
                      offering or acquisition target, social-volume z >= 3.
                    HALF SIZE: implied/realized vol >= 1.5 (floored at 1 share).
                    ANALYST FLAG, not a block: shelf registrations - a 424B takedown may be debt, and an
                      S-3ASR is routine for a large issuer, so the analyst reads the filing instead.
  7. exits          held >= 60 days -> close; held >= 15 days and rank fell past 40 -> close
                    TRAILING STOP (exits.trail_atr_multiple > 0): each evening the stop is raised to
                      (highest close since entry - k x ATR), never lowered, and the broker-side stop is
                      replaced. Pair with take_profit_r_multiple: 0 so winners are not capped at 3R.
                    PYRAMIDING (portfolio.pyramid_max_adds > 0): a holding that has moved one initial-risk
                      unit past its last entry gets another unit, sized like a fresh entry, capped by the
                      position limit, sharing the trailed stop. Never for a name the analyst is cutting.
                    REGIME LADDER (regime.ladder): strong = SPY well above its MA with VIX calm ->
                      fewer, larger positions; off = below the MA -> half size, or flat if off_liquidate.
  8. entries        fill open slots from the re-ranked list, at most 4 new names per session, skipping any
                    candidate correlated > 0.75 with more than one holding (zero tolerance was backtested
                    and lost on both return and drawdown); each position risks 1% of equity
                    with the stop distance setting the share count (`portfolio.sizing: risk_parity`),
                    capped at 10% of equity. Only fact-checked names are eligible; anything the analyst
                    has not reviewed (past its capacity, or given no verdict) is dropped; no entries at all if
                    the last bar is not the latest session.
  9. RED TEAM       Claude writes the strongest bear case per name (trap type + severity) from the fact sheet
 10. ANALYST        Claude rules KEEP / REMOVE / REDUCE per name with facts + bear case in hand. Veto-only.
                    REDUCE keeps `analyst.reduce_factor` (0.75) of an entry's shares, floored, min 1 share;
                    capped at 1.0 in code so no config value can make a veto enlarge a position.
                    A REDUCE on a stock already held is a concern, not a trade: noted once, see limitations.
                    If the API call fails: no new entries today, holdings untouched. An entry it gives no
                    KEEP / REMOVE / REDUCE verdict for is dropped: nothing is bought unreviewed.
 11. final gate     position count, per-position size, 5% cash floor, day-trade guard (code, not LLM)
 12. execute        exits: cancel bracket legs, close.  entries: limit @ close*1.005 with attached
                    stop (2 x ATR) and take-profit (3R) - exits run broker-side even if this job dies
      PROTECTION    a bracket's legs inherit the entry's DAY tif: they go live when the entry fills and
                    EXPIRE AT THAT DAY'S CLOSE, leaving the position naked from the next session on
                    (found live 2026-09-16, three positions unprotected for a day and a half). So every
                    run re-checks each holding for a live stop at the broker and re-places any missing one
                    as a GTC OCO (stop + target). Two traps this hides: a resting stop sits in status
                    "held", which a status=OPEN query does NOT return, and a market sell carries neither
                    stop_price nor limit_price - a checker filtering on those reports a false "unprotected".
                    A holding already below its stop with nothing resting is closed outright. A holding
                    with no stop on record (adopted from the broker) or whose re-placement is rejected is a
                    MANUAL ACTION note in every brief and a SEV1 monitor alarm until it is fixed.
      FAILURES      one rejected order (a stop, a close, an entry) is noted and the rest still go in; a
                    rejected close re-places the stop it had cancelled; state is saved after every
                    submitted entry. If bars, the universe or the signals fail, the run makes no exits or
                    entries but still re-checks every stop and writes the brief, then fails (SEV1).
      GAPS          a holding with no complete bar today (trading halt, feed gap) is not sold as signal
                    decay; that check waits for data. Time, regime and earnings exits still apply.
      HALTED        every halted run re-sends the close for anything the kill switch failed to flatten -
                    liquidation cancels the stops first, so a survivor is otherwise open and unprotected.
 13. brief          printed, saved to state/briefs/, POSTed to a webhook if configured

audit.py (Saturday)   deterministic trade stats incl. what each stock did 20 days AFTER we exited, then Claude
                      hunts for process failures and proposes <= 3 config changes with a required backtest check.
                      It never edits config.
```

### The governing rule

**Promotion authority requires backtestability; everything else gets veto authority only.** External content
(headlines, social feeds, web pages) is written by people who want to move retail money, and any LLM reading it
is a prompt-injection surface. So:

- Fact-sheet items can *block* a candidate or *shrink* a size. None can add a name or enlarge a position.
- The red team and analyst can only say REMOVE or REDUCE. `test_verdicts_can_only_reduce_risk` pins that down.
- The only fact that can *tilt ranking upward* is insider cluster buying, and its weight defaults to `0.0`.
  Enable it only after `scripts/build_insider_history.py` + a backtest show it helps (see below).
- Every risk limit is enforced in `desk/risk.py` / `desk/portfolio.py`, which the LLM roles never touch.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env     # Alpaca paper keys, Anthropic key, SEC_USER_AGENT, optional FINNHUB_API_KEY
.venv/bin/python -m pytest -q
```

- Alpaca paper keys: app.alpaca.markets -> Paper Trading -> API Keys. Paper accounts are margin accounts by
  default; the code never spends more than settled cash, so it stays unlevered regardless.
- `SEC_USER_AGENT="Your Name you@example.com"` is required by the SEC's fair-access policy; without it filings and
  insider data are skipped.
- `FINNHUB_API_KEY` (free tier) gives the earnings calendar. Without it every candidate's earnings date is unknown,
  so every candidate is blocked and the desk makes no new entries (exits and stops still run).

## Commands

```bash
.venv/bin/python run_daily.py --dry-run       # full run incl. fact sheets + LLM roles; submits nothing, saves no
                                              # state, prints the brief instead of writing it (the email reads it)
.venv/bin/python run_daily.py                 # real run against the account in config.yaml (paper: true)
.venv/bin/python run_daily.py --resume        # clear a kill-switch halt (re-baselines the high-water mark)
.venv/bin/python run_daily.py --no-analyst    # skip red team + analyst (engine + facts + risk only)
.venv/bin/python audit.py                     # weekly process audit (add --stats for numbers only, no LLM)
.venv/bin/python report_daily.py              # daily activity email (14:00 job); --print to build without sending
.venv/bin/python backtest.py --start 2023-09-11 --end 2026-09-11              # frozen universe, reproducible
.venv/bin/python backtest.py --start 2023-09-11 --end 2026-09-11 --risk-per-trade 0.0125 --stagger 6
.venv/bin/python scripts/build_pit_universe.py   # point-in-time universe 2017-2026, delisted names included (once)
.venv/bin/python robustness.py                   # 216-run robustness study of config.yaml on it (~2 min)
```

### Backtesting the qualitative features

The regime gate is backtested automatically (SPY + FRED VIX history). Earnings exclusion and the insider tilt
need history files first:

```bash
.venv/bin/python scripts/build_earnings_history.py --start 2023-01-01   # -> state/history/earnings.csv
.venv/bin/python scripts/build_insider_history.py  --start 2023-01-01   # -> state/history/insider.csv (slow: EDGAR at 10 req/s)
```

Bar panels are cached under `state/cache` and keyed by fetch date, so yesterday's is never read again.
Each fetch deletes panels older than `data.cache_retention_days` (2); set it to 0 to keep everything and
expect roughly 20MB per run. Fact sheets, the earnings calendar, the EDGAR store and the VIX history sit
in the same directory and are never pruned.

`backtest.py` picks the files up if present and logs which features are active. To evaluate the insider tilt,
run the backtest with `facts.insider_tilt_weight: 0.0` and again with e.g. `0.1`, on a period you did not use
to pick the weight. Only then set it in config for live runs.

Sizing can be compared without editing config:

```bash
.venv/bin/python backtest.py --start 2024-03-01 --sizing risk_parity --tag rp
.venv/bin/python backtest.py --start 2024-03-01 --sizing notional   --tag nt
```

### Reproducible backtests

Until 2026-09-24 the backtest borrowed the live universe, which rebuilds weekly, and fetched prices through the
day it was run - so the same command drifted. The three-year baseline quoted everywhere as **+247%** re-ran at
**+124.6%** after the 2026-09-21 universe rebuild, then at **+143.8%** once one stock (TEVA) that a download had
silently returned empty was restored. **Every figure dated 2026-09-12 in this README and in `config.yaml`
comments was measured on a universe that no longer exists and does not reproduce - treat those as directional.**

Now:
- The universe is a frozen snapshot, `state/history/universe_bt.json` (800 names, built 2026-09-21, id
  `12be6539`). Only `--rebuild-universe` changes it; a rebuild moves every result, so re-run the baseline
  before comparing anything. `--min-dollar-volume` / `--max-symbols` get their own snapshot.
- Prices stop at `--end` and are saved on first download to `state/history/prices/` (one file per
  universe and window, ~45MB for 800 names over 3 years, never pruned) and reused on every later run, so a re-run cannot pick up a
  feed gap the first run did not have. `--refetch-prices` downloads them again deliberately.
- A held name with no bar on some day is valued and, if forced out, sold at its last known price, and the
  signal-decay check waits for data - the same rule the live run follows.
- A stock the feed returns empty is fetched again once, then named in a `STILL no bars` warning (the same
  retry protects the live run, where a missing holding would otherwise be sold as signal decay).
- The first line of every result states the snapshot, its id, the end date and data coverage, e.g.
  `data for 800/800`. Results with a different id or coverage are not comparable.
- The same command gives byte-identical output - checked by deleting the price cache and re-downloading.

Baseline on the snapshot, 2023-09-11 to 2026-09-11: **+143.8%, max drawdown -19.2%, Sharpe 1.33**, with the kill
switch firing three times (Oct 2023, Mar 2025, Jul 2026). Across six start dates a month apart the baseline
ranges from +121% to +171%.

**Judge a setting with `--stagger`, never one window.** A single path is dominated by when the kill switch
happens to fire: 1.25% risk per trade looked like +76 points on one run mostly because it tripped the switch a
few days earlier in two sell-offs. `--stagger N` re-runs the window from N start dates a month apart, the
setting against the untouched `config.yaml`, and reports how often it wins on return, drawdown and Sharpe. The
staggered windows share most of their path (same end date, same crashes), so 6/6 is consistency, not six
independent confirmations.

### Evidence: what the numbers can and cannot tell you

Four published results set how far to trust a backtest of this strategy. They are the reason for the trial
log and the Deflated Sharpe line below every result.

- **A backtest's Sharpe barely predicts live results; its drawdown and volatility do.** Across 888 Quantopian
  strategies with at least 6 months of out-of-sample trading, backtest Sharpe explained under 2.5% of later
  performance, while volatility and maximum drawdown carried real predictive value
  ([Wiecki, Campbell, Lent & Stauth 2016](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2745220)).
  *Here:* judge variants and the go-live decision on drawdown and volatility staying in their backtest range,
  not on backtest return.
- **Published return patterns decay.** Across 97 predictors from academic studies, portfolio returns were
  about 26% lower out-of-sample and about 58% lower after publication
  ([McLean & Pontiff 2016](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2156623)). Momentum is among the
  most-published. *Here:* plan around well under half the backtest's return.
- **The best of many tries is inflated by the trying.** The Deflated Sharpe Ratio corrects a Sharpe ratio for
  the number of variants tested and for skewed, fat-tailed returns
  ([Bailey & Lopez de Prado 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)). *Here:*
  `desk/trials.py`, below.
- **Momentum crashes in panics.** Its worst losses cluster after market declines, with volatility high, during
  the rebound ([Daniel & Moskowitz 2016](https://www.nber.org/papers/w20439)). *Here:* the 2023-2026 window
  contains no such episode. The point-in-time study below covers 2018-2026 on a universe that includes later
  delistings: the regime gate and kill switch cut the 2020 and 2022 losses to about half the market's, and the
  same caution missed the 2020 rebound.

**Trial log and Deflated Sharpe.** Every `backtest.py` run appends a record to `state/history/trials.jsonl`
(committed): the variant's settings, window, universe id, data coverage, and the daily-return statistics the
deflation needs. A variant is identified by a hash of every setting that changes a result, so re-running it
never counts as a new trial, while any change to `config.yaml`'s strategy settings does. Each single run
prints its Deflated Sharpe: the probability its edge is real, given how many variants have been tried (logged,
plus `backtest.prior_untracked_trials` for any the log cannot see), how widely their Sharpes spread on the same
window and universe, and the run's own skew and kurtosis. 0.95 is the usual bar.

What the deflation does **not** fix: survivorship (the universe is today's survivors), a window with no bear
market, fills approximated from daily bars, and an AI layer the backtest does not simulate. A high Deflated
Sharpe here is a necessary condition, not a sufficient one.

**Results** - regenerate with `python scripts/trials_report.py --write-readme` after testing variants:

<!-- trials:start -->
_Generated from 18 logged runs (latest 2026-09-25 03:36 UTC) by `scripts/trials_report.py --write-readme`. Window 2023-09-11 to 2026-09-11, universe `12be6539`, N = 17 trials._

| Variant | Deflated Sharpe | Return | Max DD | Sharpe | Runs |
|---|---|---|---|---|---|
| sizing notional | 0.98 | +127% | -19.4% | 1.60 | 1 |
| earnings-block-days 0 | 0.98 | +177% | -18.6% | 1.52 | 1 |
| max-position-pct 0.15, target-positions 8 | 0.98 | +151% | -20.4% | 1.51 | 1 |
| ladder True | 0.97 | +144% | -18.9% | 1.46 | 1 |
| risk-per-trade 0.0125 | 0.97 | +181% | -18.6% | 1.47 | 1 |
| take-profit 0.0, trail-atr 3.0 | 0.96 | +154% | -19.8% | 1.36 | 1 |
| max-atr-pct 0.12 | 0.96 | +161% | -24.2% | 1.35 | 1 |
| pyramid-adds 2, take-profit 0.0, trail-atr 2.0 | 0.95 | +143% | -18.0% | 1.32 | 1 |
| **current: config.yaml** | 0.95 | +144% | -19.2% | 1.33 | 3 |
| max-new-per-day 0 | 0.95 | +140% | -18.9% | 1.31 | 1 |
| take-profit 0.0, trail-atr 2.0 | 0.94 | +134% | -18.5% | 1.27 | 1 |
| target-positions 15 | 0.93 | +131% | -20.1% | 1.23 | 1 |
| ladder True, off-liquidate True | 0.93 | +96% | -20.2% | 1.19 | 1 |
| max-corr-peers 0 | 0.90 | +103% | -21.2% | 1.08 | 1 |
| earnings-exit-days 2 | 0.86 | +87% | -22.9% | 0.99 | 1 |
| earnings-block-days 21 | 0.86 | +87% | -20.0% | 0.97 | 1 |
<!-- trials:end -->

How to read it (as of 2026-09-24, N = 17; re-read the table above against this):
- **The core edge survives the trial penalty; the choice among settings does not matter much.** Twelve of 16
  variants sit between 0.93 and 0.98, and maximum drawdown clusters at -18% to -21% across nearly all of them -
  the stable quantity, as the Quantopian study predicts. Differences inside that band are within the ~40-point
  path noise measured with `--stagger`.
- **Several 2026-09-12 conclusions do not reproduce.** On the old universe the current config was best; here the
  0-day earnings block, 8 names at 15%, the ladder and notional sizing match or beat it. The earnings-window
  calibration note above (5 days best) rests on the old-universe numbers.
- **Do not switch to the top row on this evidence.** Picking the best of 17 is the selection bias the deflation
  charges for; any change should first win a `--stagger` comparison and ideally a bear-market window.
- **The clear losers are robust:** a 21-day earnings block, a forced pre-earnings exit, and zero tolerance for
  correlated peers lose on every measure, as they did on 2026-09-12.


### Point-in-time robustness study (2018-2026)

`backtest.py` runs on today's 800 most liquid names, so it only ever buys stocks that survived and grew.
`robustness.py` re-tests `config.yaml` on the names that were liquid *at the time*: each week, the live rule
(top 800 by 20-day dollar volume, traded price >= $5) applied to every ticker the SEC listed since 2017 (94
monthly snapshots via the Wayback Machine) plus Alpaca's active and inactive assets. 2,618 names qualify at some
point; 734 have since delisted or changed ticker (SVB and First Republic among them). The definition is committed
as `state/history/pit/members.csv`; `scripts/build_pit_universe.py` rebuilds it and the ~300MB price panel.
Same engine, three additions: weekly membership, a holding with no bar for 5 sessions closes at its last price,
and random-entry / analyst-veto modes. Nothing it runs goes into the trial log.

Results as of 2026-09-24 (full report: `state/robustness/<date>/report.md`):

| | Return | CAGR | Max DD | Sharpe |
|---|---|---|---|---|
| Published baseline, today's 800, 2023-09 to 2026-09 | +144% | +34.6% | -19.2% | 1.33 |
| Same window, point-in-time names | +68% | +18.9% | -18.7% | 0.91 |
| Point-in-time, 2018-01 to 2023-09 (never used for tuning) | +20% | +3.2% | -27.7% | 0.27 |
| Point-in-time, 2018-01 to 2026-09 | +85% | +7.4% | -27.7% | 0.46 |
| Same, median of 4 start dates a quarter apart | | +4.9% | -32.1% | 0.34 |
| SPY, 2018-01 to 2026-09 (no dividends) | +184% | +12.8% | -34.2% | 0.74 |

How to read it:
- **Most of the published return is survivor bias.** Same code, prices and window; only the universe differs.
  The baseline row reproduces +143.8% exactly on the new price panel. Over 8.7 years, trades in names still on
  today's list made $11,611, while trades in names that later dropped off lost $3,692.
- **The long-run config trails SPY:** it beat SPY in 3 of 23 rolling 3-year windows and lost money in 6. The
  ranking beats 88% of 40 random-entry runs that use the same rules (p ~ 0.15), and the Deflated Sharpe is 0.72.
  Neither clears the usual bar.
- **It limits crash losses:** -12% in the COVID crash (SPY -34%) and -10% in the 2022 bear market (SPY -25%),
  with a beta of 0.41. The kill switch fired 7 times and the hard floor twice in 8.7 years.
- **One rule helps on nearly every start date:** cash below the SPY 200-day MA (`regime.off_liquidate`) gave
  7.6% a year against 4.9% (better on 3 of 4 start dates) with a max drawdown of -22.5% against -32% (better on
  all 4). That is the revisit `config.yaml` asked for
  once a bear market was available. Lower-turnover neighbors (15 positions, decay rank 60, min hold 20) also
  beat the config in 3-4 of 4 starts. Treat these as leads to test, not as results: choosing among them is a
  new trial.
- **Costs dominate.** At 25 bps of slippage per side it earns 1.7% a year; at 50 bps it loses money. Analyst
  vetoes at the live rates, applied at random, cut the 8.7-year return from +85% to +33%.
- **A single path is noise.** Removing seven exchange-traded notes that traded twice for a net $3 moved the
  8.7-year total from +72% to +85%. The study therefore judges variants from four start dates.

Caveats: earnings dates start in June 2023 (before then the earnings block is inert); prices are not
dividend-adjusted and cash earns nothing; the analyst and red team are not simulated.

### Aggression levers

Tested 2026-09-24 on the frozen snapshot, `--stagger 6`:

| Setting | Won on return | Median vs base | Won on drawdown | Won on Sharpe | Verdict |
|---|---|---|---|---|---|
| `risk_per_trade_pct` 1% -> 1.25% | 6/6 | +34.8 pts (+9.7 to +53.6) | 4/6 | 4/6 | candidate, not adopted |
| `target_positions` 12 -> 15 | 2/6 | -15.9 pts (-30.7 to +40.9) | 2/6 | 1/6 | rejected |

15 positions spreads the same cash across more, smaller positions and adds names from whatever theme is already
leading. 1.25% is consistent but has three reasons for caution before it goes live: the average position only
grows from 7.6% to 7.8% of equity (the 10% cap and cash bind first, so the gain is not simply "more invested");
a good part of the edge comes from the kill switch firing sooner, and the backtest auto-resumes 20 trading days
after a halt while live waits for a manual `--resume`, so halt-timing effects may not transfer; and full-book
open risk becomes 12 x 1.25% = 15%, the kill-switch level itself.

Tested 2026-09-12 on the old universe (baseline then +247% / DD -20.5% / Sharpe 1.78) - **not comparable to the
table above; re-test with `--stagger` before relying on any of them**: trailing stop instead of the 3R target
(+130% at 2xATR, +141% at 3x - the target limit fills the intraday spike, a trail exits the pullback),
pyramiding (+115%), regime ladder (+192%), cash below the 200-day (+181%, DD -16.3%: the one rule whose payoff a
bull-market window cannot show), no entry pacing (+104%, DD -27.7%, six hard-floor hits - batch entries share one
day's gap risk), ATR filter 12% (+155%), 8 names at 15% (+203%), $5M-volume universe (+53%). Within one universe
a single window swings by roughly 40 points on path alone (15 positions ranged -31 to +41 across staggered
starts), so the large gaps - pacing, pyramiding, trailing stops, the ATR filter, the $5M universe - probably
survive a re-test, and the small ones - 8 names at 15%, the ladder, cash below the 200-day - may not. Each
result sits in the comment next to its key. The features stay wired so live data can re-open any of them:

```bash
.venv/bin/python backtest.py --start 2023-09-11 --trail-atr 2 --take-profit 0 --pyramid-adds 2 --tag x
.venv/bin/python backtest.py --start 2023-09-11 --ladder --off-liquidate --tag y
.venv/bin/python backtest.py --start 2023-09-11 --min-dollar-volume 5000000 --max-symbols 1500 --tag z
```

### Two calibration notes worth keeping

**Earnings window.** A 15-60 day hold will usually span an earnings report - that is not a bug, it is where
the money is. Over three years of history (state/history/earnings.csv), the trades that held through a report
produced 75% of all profit (median +5.9%); trades that avoided one had a negative median. The entry block
exists only to keep a day-old stop from being gapped over: backtested at 0/5/21 days it returned
+213% / +247% / +84% with Sharpe 1.65 / 1.78 / 0.96, so it is set to 5. (2026-09-12, old universe: the 21-day
shortfall is far outside path noise; 0 vs 5 days is not, and deserves a `--stagger` re-test.) A forced exit
before the report was tested and rejected (+71%, worst drawdown, hit the hard floor twice); it stays wired in at
`exits.earnings_exit_days_before: 0`. The two rules count days differently, on purpose: the pre-report *exit*
counts trading days (calendar days would miss a Monday report), while the *entry block* counts calendar days in
both the live run and the backtest, which is how the value 5 was calibrated. Five calendar days is three to four
trading days, so on a Tuesday a report the following Monday (six calendar days out) is not blocked. Switching the
block to trading days would make it stricter and needs its own `--stagger` test first.

**Finnhub's bulk calendar truncates at 1500 rows** and returns the far end of the requested range, so a wide
date query silently reports no upcoming earnings for every symbol. `desk/facts.py` queries one symbol at a
time for this reason. Do not "optimize" it back into a single range request.

## Schedule it

```bash
for p in daily report weekly-audit; do
  sed "s|__PROJECT_DIR__|$(pwd)|g" launchd/com.alpaca-desk.$p.plist > ~/Library/LaunchAgents/com.alpaca-desk.$p.plist
  launchctl load ~/Library/LaunchAgents/com.alpaca-desk.$p.plist
done
```

**launchd schedules use the Mac's local time.** This Mac is on Pacific time, so the desk job's plist says 13:45
to run at 16:45 ET, 45 minutes after the close. (From 2026-09-17 to 09-24 it said 16:45, which meant 19:45 ET:
the desk ran after the 14:00 email, and every email in that stretch described the previous evening's run
without its notes. On a machine in another zone, convert before editing.)

The 14:00 local job runs the agent monitor first (below), then `report_daily.py`, which emails the day's activity:
results, orders working for the next open, notable moves and news, performance against SPY (since
inception, week, month, YTD), and any change to config or code since the last report (`state/fingerprint.json`).
It never submits or cancels an order. Mail needs `SMTP_HOST`/`SMTP_USER`/`SMTP_PASSWORD` (+ optional
`SMTP_PORT`, `REPORT_FROM`) or `RESEND_API_KEY` in `.env`; with neither it still writes `state/reports/`.

Each buy and sell carries a one- or two-sentence plain-English explanation. `report.summary_model`
paraphrases the analyst's own wording into it; the call has no trading authority and its output is
display-only text. Any failure - no key, API down, a refusal - falls back to sentences built from the
structured fields (`fallback_summary`), so a trade is never shown without a reason under it. Blank the key
to use those templates always.

**The balance in the email will not match Alpaca exactly.** At 14:00 local the machine is an hour past the
16:00 ET close, and after-hours trading keeps moving the marks, so the emailed equity is a live snapshot
rather than the official close - a few dollars either way on seven positions. The email states its own
snapshot time for this reason. Cash matches to the penny; only position marks drift.

**launchd does not wake a sleeping Mac**, and this one sleeps after a minute idle. A single
`pmset repeat wakeorpoweron` event covers 13:40, so the desk job holds the machine awake for 20 minutes
after it finishes (`caffeinate -i -t 1200` in its plist) to carry it through the 14:00 report. The Saturday
audit has no wake of its own and runs whenever the Mac is next awake - harmless, it only reads.

The Mac must be awake at 16:45 ET. Orders placed after the close queue for the next open; time-based exits and
the kill switch only evaluate when the job runs, so a missed day delays them by a day (stops and targets sit at
the broker and are unaffected). launchd starts a missed job when the Mac next wakes: if that is before 16:20 ET
on a session day, the run is protect-only (sync, kill switch, stop check; no exits or entries) and the evening
run decides. For a machine-independent schedule, run it from a cloud scheduler instead.

### Agent reliability monitor

Every job run and every Claude call is recorded as one line of JSON in `state/telemetry/<date>.jsonl`
(`desk/telemetry.py`): run start and end with status and outcome fields, each call's role, model, tokens,
cost, latency and outcome, what each AI role returned against what it was asked (symbols answered, symbols
skipped or invented, contract breaks such as an over-long brief), and every WARNING or ERROR log line. Secrets
are redacted before anything is written; a telemetry failure never affects a run.

`monitor.py` runs first in the 14:00 weekday job and reviews everything since its previous run:

- **Code computes, Claude interprets.** `desk/monitor.py` works out expected against actual runs from the
  installed launchd schedules and the market calendar, golden signals per AI role against a 7-day baseline
  (median, p95, MAD; days with a SEV1/SEV2 excluded), error and brief-note signatures with new ones flagged,
  and config/code/schedule changes. Claude receives those numbers with `agent-monitor-prompt.md` as its
  instructions (its Configuration block is filled from `config.yaml`'s `monitor` section) and writes the report.
- **Hard alarms do not depend on the model.** SEV1 in code: a desk run missed on a trading day, crashed, or
  started and never finished; a holding the desk reported without a verified broker-side stop; a kill-switch
  halt; a daily report that did not run, crashed, or whose email failed. Each scheduled slot is judged once, by
  the first monitor run after it is 10 minutes old (desk) or 60 minutes old (report), so a missed 13:45 desk run
  is escalated at 14:00 the same day. If the Claude call fails, a deterministic fallback report is still
  written and the alarms still escalate.
- **Output.** The full report goes to `state/monitor/reports/<date>.md`; the daily email carries a "System
  health" section (status, findings, open recommendations), and says so when the monitor did not run. SEV1 is
  also emailed at once, one email per underlying problem. Open findings, recommendations, baselines' excluded
  days and escalations persist in `state/monitor/state.json`.
- **Read-only.** It never touches orders, positions, config or code; recommendations are text for you.
- Runs scheduled before telemetry existed (2026-09-24) are reported as unverifiable, not as missed.

```bash
.venv/bin/python monitor.py --print     # analyze and print, save and send nothing
.venv/bin/python monitor.py --no-llm    # signals and the fallback report only, no Claude call
```

## Your intervention

- Read the daily brief (2 minutes). Nothing needs a reply.
- Glance at the email's System health section; open `state/monitor/reports/<date>.md` when it is not healthy.
- Read the Saturday audit; accept a proposed change only after running its backtest check.
- If the kill switch fires you get a message; nothing trades again until you run `--resume`.
- To stop everything: `launchctl unload` both plists, then close positions in the Alpaca UI if desired.

## Costs

Alpaca: $0 commissions; real cost is spread and slippage, roughly 5-15 bps per side on liquid names. Market data,
news, and options snapshots are on Alpaca's free tier. EDGAR, FRED, and StockTwits are free. Finnhub free tier.
Claude: three calls per trading day (headline classifier, red team, analyst) plus one weekly audit -
roughly $1-3/day on `claude-opus-5`; set the `*_model` keys to `claude-sonnet-5` to cut that by ~60%.
The report's plain-English trade summaries add a fourth call, but only on days something traded, and it is
a short paraphrase on `claude-sonnet-5` - around a cent. The agent monitor adds one `claude-opus-5` call per
weekday: the first run read 15.6k tokens and wrote 17.4k (reasoning included), $0.51 - about $11 a month. The
monitor's own telemetry reports the real figure every day.

## Known limitations

- **`backtest.py` is an upper bound, by about half.** Its universe is a frozen snapshot of names that were liquid
  on 2026-09-21. On names that were liquid at the time, the same 3-year baseline falls from +144% to +68%, and
  8.7 years return 5-7% a year (see "Point-in-time robustness study"). Fills are approximated from daily OHLC and
  the red team / analyst vetoes are not simulated. Live, the analyst cut over half the entries in the first two
  weeks, so live trading runs at well under the risk the backtest assumes.
- **The backtest's kill switch resumes by itself.** After a halt it sits out 20 trading days
  (`--halt-cooldown-days`) and starts again; live waits for you to run `--resume`. It fired three times in
  three years on the baseline, so expect roughly one halt a year, and treat any result that depends on halt
  timing as optimistic.
- **An analyst REDUCE on a name already held is a note, not a trade or an action item.** The holding keeps
  its stop, target and decay rule, as in the backtest. The concern goes into the brief once, the first run
  it is raised (`analyst concern on X (held; no trade ...)`), and is kept on the position as `analyst_flag`;
  a later KEEP clears it. Until 2026-09-25 it was a MANUAL ACTION every run, which put the same advisories
  under "Needs your action" daily. Entries are still cut before purchase as normal. Trimming holdings
  automatically would need a partial-exit path that resizes the broker-side stop, and a backtest first:
  the robustness study found no evidence the analyst's vetoes add return.
- **Filing and news screens only look back 90 days.** A dilution filing older than that stops blocking the
  name: DELL was blocked 2026-09-14 on an 8-K 3.02 dated 2026-06-17 and bought on 09-16 when it aged out.
  Intentional (a June raise should not veto a September entry forever), but the block is a window, not a memory.
- **FRED is unreachable from some networks**, and the VIX half of the regime gate fails silently when it is
  (it did for the first three live sessions). `desk/regime.py` falls back to Yahoo; check the brief says
  `VIX <number>` and not `VIX None`.
- **StockTwits is an undocumented public endpoint** and may rate-limit or change; the z-score needs ~15 days of
  the system's own history per name before it activates. Failures degrade to "no social data".
- **Options IV needs a listed chain** 20-50 days out; names without options get no IV adjustment.
- **Whole shares only.** Alpaca bracket orders do not support fractional shares, so names above ~$800 are
  skipped at $10k equity.
- **Short-term capital gains.** 15-60 day holds are taxed as ordinary income; wash sales are not tracked.
- **Pattern day trader rule.** Under $25k equity, same-day round trips are limited. The scheduled run sells
  after the close, so its exits fill next session and cannot complete one. A run launched while the market is
  open defers the exit of anything whose fill it first saw that day once the day-trade count is used up. The
  kill switch ignores the guard, and a stop firing on entry day counts against you.
- **A run launched before the day's bars are final only protects.** launchd starts a missed job when the Mac
  next wakes. If the market is open, or it is before 16:20 ET, the run syncs with the broker, applies the kill
  switch and checks every stop, and makes no exits or entries (`PROTECT-ONLY RUN` in the brief): signals would
  read a partial day and exits would fill at once instead of at the next open. `--force` runs everything anyway
  and the brief warns that the bar is partial. The bar cache never hands a partial panel to the evening run: a
  cached panel is reused only if fetched after the latest 16:20 ET.
- **A config.yaml typo stops the run.** Every risk and sizing key is required at load (`desk/config.py`), and
  `account.paper` must be true: no code path builds a live trading client.

### Live against the backtest

Every rule that differs between `run_daily.py` and `run_backtest`. None is fixed in code, because fixing any of
them changes either live trading or the frozen baseline; each is a decision for the owner.

| Rule | Live | Backtest | Effect |
|---|---|---|---|
| Kill switch recovery | halts until `--resume` | resumes after 20 sessions (hard floor: 60, then re-baselines) | halt timing is optimistic in the backtest |
| Analyst, red team, implied-vol half size, offering / headline / social blocks | applied | not simulated (robustness.py's `frictions` samples them at random) | live runs smaller and more selective |
| Exit fill | decided at 16:45 ET, market sell fills at the next open | time, decay, regime and switch exits fill at that day's close | one overnight gap per exit |
| Hold clock (min 15 / max 60 days, decay eligibility) | counts from the day the entry was staged, the evening before the fill | counts from the fill | live reaches each limit one session early (three days for a Friday order); `hold_days` in trades.csv is one day long |
| Stop and target | set from the signal close when the order is staged | recomputed from the actual fill | after a gap at the open, live risk per share differs from the 1% budget |
| Entry pacing (4/day) | applied to sized entries: a pick that cannot be sized is replaced by the next | applied to picks before sizing | the backtest can buy fewer names on a day |
| Earnings block boundary | report 0-5 calendar days out (a report today blocks) | 1-5 days out | live is one day stricter |
| Fact-checked pool | only the top 30 non-held eligible names can be bought | earnings checked for the top 60 eligible names; anything eligible can be bought | the backtest can reach further down the ranking |
| VIX | FRED's series, usually one session behind | that day's close | the backtest's regime flips a day sooner |
| Holding with no bars | kept (broker decides) | sold at its last close after 5 sessions | backtest flatters delistings (robustness.py lists them) |
| Final gate, cash | `enforce_entry_limits`, non-marginable buying power | neither | cannot bind at the configured target |
| Guards | stale-data, day-trade, trading-day, protect-only before 16:20 ET, no entries without fact sheets or an earnings calendar, risk-off target when the regime feed fails | none | live skips or softens some days |
| Earnings dates | Finnhub, per symbol | yfinance history from June 2023 | block inert before mid-2023 in the backtest |

# Robocash

Systematic, long-only, no-leverage US equity strategy for a small Alpaca paper account. It runs once a day after
the close, holds up to 20 names for 21-126 days with broker-side stops, and has a hard-coded drawdown kill switch.
An intelligence layer turns SEC filings, news, options and social data into structured facts; Claude plays a red
team and a veto-only analyst on top of that; a weekly auditor attacks the process itself.

**Current release: v2.1 "slow and wide"** (from the 2026-10-30 close) - see [Releases](#releases).

**This is not investment advice and the strategy is not proven.** Run it in paper mode and expect a year of data
before you can distinguish it from noise.

Contents: [How it works](#how-it-works) · [Releases](#releases) · [Setup](#setup) · [Commands](#commands) ·
[Schedule](#schedule) · [Your intervention](#your-intervention) · [Costs](#costs) ·
[Research and backtests](#research-and-backtests) · [Known limitations](#known-limitations)

## How it works

```
run_daily.py (13:40 PT = 16:40 ET, weekdays)
  1. sync           state with the broker; log any positions closed by a stop
  2. RISK GOVERNOR  drawdown from episode high-water mark >= 15% -> liquidate, halt, resume on its own after
                      20 sessions (risk.halt_cooldown_days, as the backtest does); --resume restarts sooner
                    drawdown from ALL-TIME high >= 25%            -> liquidate, halt until --resume --reset-hwm
                    day down >= 5%                                -> no new entries today
                    (a resume re-baselines the episode HWM, so each episode is capped at 15%; the all-time
                    floor is the cumulative backstop and needs an explicit acknowledgement)
  3. universe       NYSE/Nasdaq common stock, price >= $5, 20d avg dollar volume >= $20M, top 800 (rebuilt weekly)
  4. signals        12-1 month momentum (60%), 20-day return (20%), low realized vol (20%);
                    must be above its 50-day MA, positive momentum, 14-day ATR <= 8% of price
  5. REGIME GATE    SPY below its 200-day MA or VIX > 30 -> target position count halved (code, Alpaca + FRED).
                    If the SPY/regime data fails, the halved target applies until it is back; nothing is sold for it.
  6. FACT SHEETS    for the top 30 candidates + holdings (desk/facts.py). Each source degrades on its own, but if
                    the sheets cannot be built at all, or the earnings calendar is unknown, no new entries today:
                      SEC EDGAR   dilution filings (8-K 3.02, S-1), shelf filings (424B, S-3), 8-K items, Form 4
                      Finnhub     next earnings date (one query per symbol - see calibration notes)
                      Alpaca      Benzinga headlines (Claude-classified: tone + offering/guidance/M&A/legal flags)
                      Alpaca      options chain -> ATM implied vol vs 20d realized vol, put skew
                      StockTwits  message rate vs the name's own history (z-score), bullish share
                    HARD BLOCKS (candidates only): earnings within 5 calendar days (a report today included; a
                      failed lookup blocks too), headline offering or acquisition target, social-volume z >= 3.
                    HALF SIZE: implied/realized vol >= 1.5 (floored at 1 share).
                    ANALYST FLAGS, not blocks: dilution filings (facts.offering_block: false since 2026-10-17)
                      and shelf registrations - the analyst reads the filing instead.
  7. exits          held >= 126 days -> close; held >= 21 days and rank fell past 80 -> close;
                    broker-side stop 4x ATR below the fill, no take-profit leg
                    Wired but off: trailing stop (exits.trail_atr_multiple), pyramiding (portfolio.pyramid_max_adds),
                      regime ladder / cash below the 200-day (regime.ladder, regime.off_liquidate).
  8. entries        fill open slots from the re-ranked list, at most 4 new names per session, skipping any
                    candidate correlated > 0.75 with more than one holding. Each position risks 1% of equity,
                    the stop distance setting the share count (portfolio.sizing: risk_parity), capped at 7% of
                    equity, 20 names targeted. Only fact-checked, analyst-reviewed names are eligible; no
                    entries at all if the last bar is not the latest session.
  9. RED TEAM       Claude writes the strongest bear case per buy candidate (trap type + severity)
 10. ANALYST        Claude rules KEEP / REMOVE / REDUCE per name with facts + bear case in hand. Veto-only.
                    REDUCE is recorded but no longer shrinks an entry (analyst.reduce_factor 1.0 since 2026-10-17;
                    capped at 1.0 in code). On a stock already held it is a note, not a trade.
                    If the call fails: no new entries today, holdings untouched. An entry with no verdict is dropped.
 11. final gate     position count, per-position size, 5% cash floor, day-trade guard (code, not LLM)
     trim           a holding that drifts to 15% of equity is sold back to 10% at the next open, its stop
                    re-placed for the rest. A safety net: it never fired in v2.0's backtests.
 12. execute        exits: cancel the stop, market sell at the next open.
                    entries: limit @ close x 1.005 with an attached stop (OTO; a bracket when a target is set).
      PROTECTION    attached legs inherit the entry's DAY time-in-force and expire at that day's close, so every
                    run re-checks each holding for a live stop at the broker and re-places any missing one as GTC
                    (OCO with the target when there is one). On the run that first sees a fill, the stop moves to
                    the same distance from the fill price, as the backtest measures it. A holding already below
                    its stop with nothing resting is closed outright; one with no stop on record, or whose
                    re-placement is rejected, is a MANUAL ACTION in every brief and a SEV1 monitor alarm.
                    Two traps: a resting stop has status "held", which a status=OPEN query does NOT return, and a
                    market sell carries neither stop_price nor limit_price.
      FAILURES      one rejected order is noted and the rest still go in; a rejected close re-places the stop it
                    cancelled; state is saved after every submitted entry. If bars, the universe or signals fail,
                    the run makes no exits or entries but still re-checks every stop and writes the brief (SEV1).
      GAPS          a holding with no complete bar today (halt, feed gap) is not sold as signal decay.
      HALTED        every halted run re-sends the close for anything the kill switch failed to flatten.
 13. brief          printed, saved to state/briefs/, POSTed to a webhook if configured

report_daily.py (14:00 PT)  the day's email: results, holdings, orders for the next open, moves, news,
                            performance against SPY, system health, and any config/code change since the last email
monitor.py      (14:00 PT)  agent reliability monitor, just before the email - see Schedule
audit.py        (Sat 09:00) deterministic trade stats, then Claude hunts for process failures and proposes <= 3
                            config changes, each with a required backtest check. It never edits config.
```

### The governing rule

**Promotion authority requires backtestability; everything else gets veto authority only.** External content
(headlines, social feeds, web pages) is written by people who want to move retail money, and any LLM reading it
is a prompt-injection surface. So:

- Fact-sheet items can *block* a candidate or *shrink* a size. None can add a name or enlarge a position.
- The red team and analyst can only say REMOVE or REDUCE. `test_verdicts_can_only_reduce_risk` pins that down.
- The only fact that can *tilt ranking upward* is insider cluster buying, and its weight defaults to `0.0`.
  Enable it only after `scripts/build_insider_history.py` + a backtest show it helps.
- Every risk limit is enforced in `desk/risk.py` / `desk/portfolio.py`, which the LLM roles never touch.

## Releases

A release is a named version of the strategy with results reported from its own start (`config.yaml` `release`,
git tags, GitHub Releases).

| Release | Counted from | What changed |
|---|---|---|
| v1.0 | inception, 2026-09-15 | the original desk |
| v2.0 | 2026-10-16 close ($9,361) | engineering review 2026-10-17: cost cuts, REDUCE and the dilution block retired |
| v2.1 "slow and wide" | 2026-10-30 close | quant review 2026-11-01: 12-month momentum, 4x ATR stop with no target, decay rank 80, holds of 21-126 days, 20 names at 7% |

v2.1 is the first change to clear the owner's bar in both backtest windows (under live frictions, and in every
start with the SPY sweep); the numbers sit in `config.yaml` above `release`. Positions opened under v2.0 keep their
2x-ATR stops and 3R targets at the broker and their `v2.0` label; v2.1's decay and hold rules apply to them.

- The daily email leads its performance table with "Since v2.1" beside since inception, both against SPY.
- New positions, trade-log rows (`release` column) and shadow-book entries carry the release. Positions held at
  a cut close as `<old release> (carried)`.
- The live-vs-backtest section and the weekly audit report the current release beside the whole history.
- Risk limits never reset at a release: the kill switch, high-water mark and cumulative floor are continuous.

**Stop rule** (`governance.stop_rule`). The alternative to the desk is a momentum ETF, not cash, so v2.1 is judged
against SPMO and SPY from its release date. The weekly audit reports the status:

- **REVIEW** on or after 2027-04-30 if the desk trails SPMO by more than 20 points - the owner looks again.
- **STOP** on or after 2027-10-29 if it trails SPMO by more than 15 points *and* trails SPY - replace the stock
  picker with SPMO; the owner decides.

Calibrated on v2.1's own backtest: it trails SPMO by those margins by chance 8% (7 months) and 19% (12 months) of
the time.

To cut a new release: set `release.name` and `release.since` (the last close under the old version), merge, tag the
merge, and publish a GitHub Release from the tag.

### The shadow book: what the layers on top of the engine are worth

The quant engine is backtested; the analyst's REMOVE, the red team behind it, the headline and earnings blocks and
the implied-vol half size are not. Each desk run appends every decision that changed what was bought to
`state/shadow/book.jsonl` (local, not committed), with the day's buys as the control group. The weekly audit
replays each one with the desk's own exit rules and against SPY, and shows the groups side by side
(`desk/shadow.py`). The retirement rule was set before any data (`shadow.min_closed: 30`): a group whose kept-out
names did at least as well as the names bought, over 30 closed trades, is flagged for retirement - the owner
decides. Blocked names count once per four weeks.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env     # Alpaca paper keys, Anthropic key, SEC_USER_AGENT, FINNHUB_API_KEY, mail settings
.venv/bin/python -m pytest -q
```

- **Alpaca paper keys:** app.alpaca.markets -> Paper Trading -> API Keys. Paper accounts are margin accounts by
  default; the code never spends more than settled cash, so it stays unlevered.
- **`SEC_USER_AGENT="Your Name you@example.com"`** is required by the SEC's fair-access policy; without it filings
  and insider data are skipped.
- **`FINNHUB_API_KEY`** (free tier) gives the earnings calendar. Without it every candidate's earnings date is
  unknown, so every candidate is blocked and the desk makes no new entries (exits and stops still run).
- **Mail:** `SMTP_HOST`/`SMTP_USER`/`SMTP_PASSWORD` (+ optional `SMTP_PORT`, `REPORT_FROM`) or `RESEND_API_KEY`.
  With neither, emails are still written to `state/reports/`.

## Commands

```bash
.venv/bin/python run_daily.py --dry-run       # full run incl. fact sheets + LLM roles; submits nothing, saves nothing
.venv/bin/python run_daily.py                 # real run against the account in config.yaml (paper: true)
.venv/bin/python run_daily.py --resume        # clear a kill-switch halt now instead of on its resume date
.venv/bin/python run_daily.py --no-analyst    # skip red team + analyst (engine + facts + risk only)
.venv/bin/python run_daily.py --force --max-new-entries 20   # owner override: a daytime run buying up to 20
.venv/bin/python report_daily.py --print      # build the daily email without sending it
.venv/bin/python monitor.py --print           # health check: analyze and print, save and send nothing
.venv/bin/python monitor.py --no-llm          # signals and the rules-only report, no Claude call
.venv/bin/python audit.py                     # weekly process audit (--stats for numbers only, no LLM)
.venv/bin/python audit.py --email-saved       # email the latest saved audit again (--date, --resend)

.venv/bin/python backtest.py --start 2023-09-11 --end 2026-09-11   # frozen universe, reproducible
.venv/bin/python backtest.py --start 2023-09-11 --stagger 6        # the setting from 6 start dates vs config.yaml
.venv/bin/python scripts/build_pit_universe.py                     # point-in-time universe 2017-2026 (once)
.venv/bin/python robustness.py                                     # robustness study of config.yaml on it
.venv/bin/python scripts/trials_report.py --write-readme           # refresh the trials table below
```

## Schedule

```bash
for p in daily report weekly-audit; do
  sed "s|__PROJECT_DIR__|$(pwd)|g" launchd/com.alpaca-desk.$p.plist > ~/Library/LaunchAgents/com.alpaca-desk.$p.plist
  launchctl load ~/Library/LaunchAgents/com.alpaca-desk.$p.plist
done
launchctl list | grep alpaca-desk      # should show all three, after any restart too
```

| Job | When (Mac local, Pacific) | Runs |
|---|---|---|
| `com.alpaca-desk.daily` | weekdays 13:40 (16:40 ET) | `launchd/teamrobo-desk` -> `run_daily.py` |
| `com.alpaca-desk.report` | weekdays 14:00 | `launchd/teamrobo-email` -> `monitor.py`, then `report_daily.py` |
| `com.alpaca-desk.weekly-audit` | Saturday 09:00 | `audit.py` |

- **launchd uses the Mac's local time.** On a machine in another zone, convert before editing the plists.
- **Allow them in the background.** macOS loads these jobs at login only if they are switched on under System
  Settings > General > Login Items & Extensions > Allow in the Background, where they appear as `teamrobo-desk`,
  `teamrobo-email` and `caffeinate`. Named scripts are used for that reason: inline `/bin/sh -c` commands were
  listed as "sh", got switched off, and did not load after a restart (2026-09-27).
- **Waking the Mac.** launchd does not wake a sleeping Mac. A `pmset repeat wakeorpoweron` event at 13:40 starts the
  desk job, which keeps the machine awake (`caffeinate -i`) for 20 minutes after it finishes and at least until
  14:50, carrying it through the email and the 14:30 health review (a Claude app scheduled task). The Saturday audit
  has no wake of its own and runs whenever the Mac is next awake - harmless, it only reads.
- **When the schedule slips.** launchd starts a missed job when the Mac next wakes. Before 16:20 ET on a session day
  that run is protect-only (sync, kill switch, stop check; no exits or entries). A desk run that finishes after
  14:00 sends the day's email itself; `report_daily.py` sends at most one email a day (`--resend` overrides). Every
  job checks that all three jobs are loaded and the 13:40 wake is scheduled - a problem is a MANUAL ACTION in the
  brief and email, a SEV1 in the monitor, and an alert from the Saturday audit. Stops sit at the broker and are
  unaffected by a missed run; time exits and the kill switch wait for the next one.
- **How a run knows it was scheduled.** Each plist sets `TEAMROBO_JOB` to its label; telemetry records a run as
  `launchd` only when that is present, and every `run_start` records the process chain (`Python<sh<launchd`).

### The daily email

Results, current holdings, orders working for the next open, notable moves and news, performance against SPY
(since the current release, inception, week, month, YTD), live drawdown and volatility against their backtest
range, system health, and any change to config or code since the last email (`state/fingerprint.json`). It never
submits or cancels an order.

- Each buy and sell carries a one- or two-sentence explanation: `report.summary_model` paraphrases the analyst's
  own wording, display-only. Any failure falls back to sentences built from the structured fields.
- **The balance will not match Alpaca exactly.** At 14:00 PT after-hours trading is still moving the marks, so the
  equity is a live snapshot, and the email states its time. Cash matches to the penny. The equity history is dated
  by the New York session (Alpaca stamps daily points at 00:00 UTC, which is the previous evening in New York).

### Agent reliability monitor

Every job run and every Claude call is one line of JSON in `state/telemetry/<date>.jsonl` (`desk/telemetry.py`):
run start and end, each call's role, model, tokens, cost, latency and outcome, what each AI role returned against
what it was asked, and every WARNING or ERROR log line. Secrets are redacted; a telemetry failure never affects a run.

`monitor.py` runs just before the 14:00 email and reviews everything since its previous run:

- **Code computes, Claude interprets.** `desk/monitor.py` works out expected against actual runs from the installed
  schedules and the market calendar, golden signals per AI role against a 7-day baseline, new error signatures, and
  config/code/schedule changes. The Claude review (`agent-monitor-prompt.md`) runs on Fridays, on any day a hard
  alarm fires, and at least weekly (`monitor.model_review_weekdays`); other days write a rules-only report.
- **Hard alarms do not depend on the model.** SEV1 in code: a desk run missed, crashed or never finished; a holding
  without a verified broker-side stop; a kill-switch halt; a daily report that did not run or whose email failed;
  API spend over `budget.monthly_api_usd` (SEV2 at 80%). If the Claude call fails, a fallback report is still
  written and the alarms still escalate.
- **Output.** `state/monitor/reports/<date>.md`; the email's System health section; SEV1 is also emailed at once,
  one email per underlying problem.
- **Read-only.** It never touches orders, positions, config or code; recommendations are text for you.

## Your intervention

- Read the daily email (2 minutes). Nothing needs a reply.
- Open `state/monitor/reports/<date>.md` when System health is not healthy.
- Read the Saturday audit email; accept a proposed change only after running its backtest check.
- If the kill switch fires you get an alarm. The 15% switch resumes by itself after 20 sessions (the email names the
  date); `--resume` restarts sooner. The 25% hard floor needs `--resume --reset-hwm`.
- To stop everything: `launchctl unload` the three plists, then close positions in the Alpaca UI if desired.

### Moving to a new paper account

1. Create the paper account in the Alpaca dashboard, generate its keys, and replace `ALPACA_API_KEY` /
   `ALPACA_SECRET_KEY` in `.env`. Until step 2 runs, the desk refuses to trade: `state.json` records which account
   it belongs to, and keys for another account stop the run with an "ACCOUNT CHANGED" alarm.
2. Run `python scripts/switch_account.py` to see the plan, then `python scripts/switch_account.py --apply`. It needs a
   flat new account (no positions, no open orders).
3. Commit the `config.yaml` change it makes.

The switch archives the old `state.json` (high-water marks, halt, positions) to `state/accounts/<old account>/` and
starts a fresh one, so the kill switch measures from the new balance. It sets `account.since` to the first desk run
on the new account: the email's since-inception rows (Alpaca's equity history), the parity replay and the stop rule
measure against SPY and SPMO from there. The stop rule's checkpoints move by the same number of days. What the desk
has learned stays: the trade log, the shadow book, the trial log, monitor findings and owner resolutions, audits and
briefs.

## Costs

Alpaca: $0 commissions; the real cost is spread and slippage, roughly 5-15 bps per side on liquid names. Market
data, news and options snapshots are on Alpaca's free tier; EDGAR, FRED, StockTwits and Finnhub are free.

Claude API, measured from telemetry, 2026-10-20 to 10-31 (10 sessions, after the 2026-10-17 cost cuts): **$9.87,
about $20 a month.**

| Job | Calls | Share |
|---|---|---|
| Monitor | one `claude-opus-5` review a week plus hard-alarm days | 46% |
| Desk | headline classifier (`claude-sonnet-5`), red team on buy candidates, analyst | 35% |
| Audit | one `claude-opus-5` call a week | 18% |
| Email | trade summaries on `claude-sonnet-5`, only on days something traded | <1% |

The monitor raises an alarm at `budget.monthly_api_usd` ($35 since 2026-11-07); it does not cap spending.

## Research and backtests

### Evidence: what the numbers can and cannot tell you

Four published results set how far to trust a backtest of this strategy. They are the reason for the trial log and
the Deflated Sharpe line below every result.

- **A backtest's Sharpe barely predicts live results; its drawdown and volatility do.** Across 888 Quantopian
  strategies, backtest Sharpe explained under 2.5% of later performance, while volatility and maximum drawdown carried
  real predictive value ([Wiecki, Campbell, Lent & Stauth 2016](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2745220)).
  *Here:* judge variants on drawdown and volatility staying in their backtest range, not on backtest return.
- **Published return patterns decay.** Across 97 predictors, returns were about 26% lower out-of-sample and about 58%
  lower after publication ([McLean & Pontiff 2016](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2156623)).
  Momentum is among the most-published. *Here:* plan around well under half the backtest's return.
- **The best of many tries is inflated by the trying.** The Deflated Sharpe Ratio corrects a Sharpe ratio for the
  number of variants tested and for skewed, fat-tailed returns
  ([Bailey & Lopez de Prado 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)). *Here:* `desk/trials.py`.
- **Momentum crashes in panics.** Its worst losses cluster after market declines, during the rebound
  ([Daniel & Moskowitz 2016](https://www.nber.org/papers/w20439)). *Here:* the point-in-time study covers 2018-2026,
  including the 2020 crash and the 2022 bear market.

### Point-in-time robustness study (2018-2026)

`backtest.py` runs on a frozen list of today's 800 most liquid names, so it only ever buys stocks that survived.
`robustness.py` re-tests `config.yaml` on the names that were liquid *at the time*: each week, the live rule applied
to every ticker the SEC listed since 2017 (94 monthly snapshots via the Wayback Machine) plus Alpaca's active and
inactive assets. 2,618 names qualify at some point; 734 have since delisted or changed ticker (SVB and First Republic
among them). The definition is committed as `state/history/pit/members.csv`; `scripts/build_pit_universe.py` rebuilds
it and the ~300MB price panel. Variants are judged from four start dates a quarter apart, and a `frictions` mode
samples the analyst's vetoes and the half-size rule at their live rates. Reports go to `state/robustness/<date>/`.

**The owner's bar** for a change: median six-month excess return over SPY better in at least 3 of 4 starts in *both*
windows (2018-2026 and 2023-2026), the 10th percentile no worse in 3 of 4, and median max drawdown no more than 1.5x
today's.

What the study found under v2.0 (2026-09-24, still the frame for reading every result):

| v2.0 config | Return | CAGR | Max DD | Sharpe |
|---|---|---|---|---|
| Frozen survivors' list, 2023-09 to 2026-09 | +144% | +34.6% | -19.2% | 1.33 |
| Same window, point-in-time names | +68% | +18.9% | -18.7% | 0.91 |
| Point-in-time, 2018-01 to 2026-09 | +85% | +7.4% | -27.7% | 0.46 |
| SPY, 2018-01 to 2026-09 (no dividends) | +184% | +12.8% | -34.2% | 0.74 |

- **About half the frozen-list return is survivor bias.** Same code, prices and window; only the universe differs.
- **v2.0 trailed SPY over the long run** (it beat SPY in 3 of 23 rolling 3-year windows) but limited crash losses:
  -12% in the COVID crash (SPY -34%) and -10% in 2022 (SPY -25%), with a beta of 0.41.
- **Costs dominate.** At 25 bps of slippage per side it earned 1.7% a year; at 50 bps it lost money.
- **A single path is noise.** Removing seven exchange-traded notes that traded twice for a net $3 moved the 8.7-year
  total from +72% to +85%.

The 2026-11-01 quant review found that adding risk within the v2.0 design (bigger positions, wider drawdown limits)
lost, and that slower, wider holding (v2.1) passed the bar under live frictions: 4/4 starts on median and p10 in
2018-2026, 3/4 and 3/4 in 2023-2026. Its scripts are `scripts/check_quant_review_1101.py` and
`scripts/check_cash_sweep_live.py`.

### Reproducible backtests and the trial log

`backtest.py` reads a frozen universe, `state/history/universe_bt.json` (800 names, built 2026-09-21, id `12be6539`),
and prices saved on first download to `state/history/prices/`. The same command gives byte-identical output, and the
first line of every result states the snapshot, its id, end date and data coverage - results with a different id or
coverage are not comparable. Only `--rebuild-universe` / `--refetch-prices` change the inputs. Figures dated before
2026-09-24 (anywhere in this repo) were measured on a universe that no longer exists; treat them as directional.

**Judge a setting with `--stagger` or `robustness.py`, never one window.** A single path is dominated by when the kill
switch happens to fire. The staggered windows share most of their path, so 6/6 is consistency, not six independent
confirmations.

**Trial log and Deflated Sharpe.** Every `backtest.py` run appends a record to `state/history/trials.jsonl`
(committed): the variant's settings, window, universe, coverage and the return statistics the deflation needs. A
variant is identified by a hash of every setting that changes a result, so re-running it never counts as a new trial.
Each run prints its Deflated Sharpe: the probability its edge is real given how many variants have been tried, how
widely their Sharpes spread, and the run's own skew and kurtosis. 0.95 is the usual bar. It does **not** fix fills
approximated from daily bars or an AI layer the backtest does not simulate: a high Deflated Sharpe is necessary, not
sufficient.

<!-- trials:start -->
_Generated from 123 logged runs (latest 2026-11-06 22:39 UTC) by `scripts/trials_report.py --write-readme`. Window 2023-09-11 to 2026-09-11, universe `pit`, N = 110 trials._

| Variant | Deflated Sharpe | Return | Max DD | Sharpe | Runs |
|---|---|---|---|---|---|
| exits.max-hold-days 126, exits.min-hold-days 21, exits.signal-decay-rank 80, exits.stop-atr-multiple 4.0, exits.take-profit-r-multiple 0.0, portfolio.cash-sweep {'enabled': True, 'band_pct': 0.02}, portfolio.max-position-pct 0.07, portfolio.max-positions 25, portfolio.target-positions 20, signals.momentum-lookback-days 252 | 0.90 | +140% | -22.2% | 1.31 | 2 |
| exits.max-hold-days 126, exits.min-hold-days 21, exits.signal-decay-rank 80, exits.stop-atr-multiple 4.0, exits.take-profit-r-multiple 0.0, portfolio.max-position-pct 0.07, portfolio.max-positions 25, portfolio.target-positions 20, signals.momentum-lookback-days 252 | 0.87 | +120% | -18.1% | 1.22 | 3 |
| portfolio.min-entry-fraction 0 | 0.80 | +97% | -21.0% | 1.06 | 2 |
| portfolio.min-entry-fraction 0.25 | 0.75 | +85% | -20.8% | 0.96 | 2 |
| portfolio.correlation-days 120 | 0.73 | +70% | -18.7% | 0.92 | 1 |
| facts.offering-block-backtest False | 0.73 | +68% | -18.7% | 0.91 | 1 |
| portfolio.cash-sweep {'enabled': False} | 0.73 | +68% | -18.7% | 0.91 | 2 |
| **current: config.yaml** | 0.73 | +68% | -18.7% | 0.91 | 1 |
| regime.off-liquidate True | 0.71 | +64% | -16.0% | 0.89 | 1 |
| exits.max-hold-days 126, exits.min-hold-days 21, exits.signal-decay-rank 80, exits.stop-atr-multiple 4.0, exits.take-profit-r-multiple 0.0, portfolio.max-position-pct 0.07, portfolio.max-positions 25, portfolio.target-positions 20 | 0.71 | +65% | -20.2% | 0.89 | 1 |
| portfolio.max-position-pct 0.15, portfolio.risk-per-trade-pct 0.015, portfolio.trim-above-pct 0.2, portfolio.trim-to-pct 0.15 | 0.69 | +68% | -19.8% | 0.86 | 1 |
| facts.offering-block-backtest True | 0.69 | +62% | -18.9% | 0.84 | 1 |
| regime.risk-off-scale 1.0 | 0.68 | +62% | -20.1% | 0.84 | 1 |
| portfolio.min-entry-fraction 0.5 | 0.68 | +71% | -22.7% | 0.84 | 2 |
| portfolio.cash-sweep {'enabled': True, 'band_pct': 0.02} | 0.62 | +51% | -22.2% | 0.73 | 5 |
| portfolio.cash-sweep {'enabled': True, 'band_pct': 0.05} | 0.61 | +50% | -21.7% | 0.72 | 2 |
| risk.max-cumulative-drawdown-pct 0.35, risk.max-drawdown-pct 0.25 | 0.59 | +47% | -25.8% | 0.69 | 1 |
| portfolio.target-positions 15 | 0.59 | +50% | -20.9% | 0.69 | 1 |
| portfolio.max-position-pct 0.15, portfolio.risk-per-trade-pct 0.0125, portfolio.trim-above-pct 0.2, portfolio.trim-to-pct 0.15 | 0.59 | +45% | -19.5% | 0.69 | 1 |
| risk.max-drawdown-pct 0.99 | 0.58 | +43% | -29.2% | 0.67 | 1 |
| portfolio.target-positions 15, regime.off-liquidate True | 0.57 | +44% | -16.6% | 0.65 | 1 |
| exits.signal-decay-rank 80 | 0.54 | +39% | -18.3% | 0.61 | 1 |
| exits.max-hold-days 252, exits.min-hold-days 63, exits.signal-decay-rank 100, exits.stop-atr-multiple 10.0, exits.take-profit-r-multiple 0.0, portfolio.max-position-pct 0.06, portfolio.max-positions 30, portfolio.sizing notional, portfolio.target-positions 25, portfolio.vol-weighting False, signals.max-atr-pct 0.15, signals.momentum-lookback-days 252, signals.weights {'momentum': 1.0, 'short': 0.0, 'low_vol': 0.0} | 0.54 | +52% | -31.4% | 0.61 | 1 |
| signals.weights {'momentum': 0.6, 'short': 0.1, 'low_vol': 0.3} | 0.42 | +17% | -15.5% | 0.45 | 1 |
| signals.weights {'momentum': 0.6, 'short': 0.15, 'low_vol': 0.25} | 0.32 | +11% | -19.3% | 0.29 | 1 |
<!-- trials:end -->

How to read it: **nothing clears 0.95.** v2.1 (second row) leads the configurations that run live, at 0.86 against
v2.0's 0.72; the top row adds the SPY cash sweep, which is backtest-only so far (decision deferred to 2026-12-07). The
"current: config.yaml" row is the last plain run of the config, logged under v2.0. Drawdown clusters at -16%
to -22% across nearly every variant - the stable quantity, as the Quantopian study predicts. Choosing the best row is
the selection bias the deflation charges for, so a change still has to pass the owner's bar on the point-in-time study.

### Earlier experiments

Wired in code and off, each with its result in the comment next to its key in `config.yaml`. Most were measured on the
old universe (2026-09-12) and need a `--stagger` re-test before relying on them: trailing stop instead of the 3R target,
pyramiding, the regime ladder, cash below the 200-day MA (`regime.off_liquidate` - the one rule that helped on nearly
every point-in-time start, 7.6% a year against 4.9% under v2.0), no entry pacing (deeper drawdowns: batch entries share
one day's gap risk), a looser ATR filter, 8 names at 15%, and a $5M-volume universe (thin names win the ranking and
lose the trade). Higher risk per trade (1.25%) won on return in 6/6 staggered starts but mostly by tripping the kill
switch earlier; it was not adopted.

```bash
.venv/bin/python backtest.py --start 2023-09-11 --trail-atr 2 --take-profit 0 --pyramid-adds 2 --tag x
.venv/bin/python backtest.py --start 2023-09-11 --ladder --off-liquidate --tag y
.venv/bin/python backtest.py --start 2024-03-01 --sizing notional --tag nt
```

The regime gate is backtested automatically (SPY + FRED VIX history). The earnings block and the insider tilt need
history files first (`backtest.py` logs which features are active):

```bash
.venv/bin/python scripts/build_earnings_history.py --start 2023-01-01   # -> state/history/earnings.csv
.venv/bin/python scripts/build_insider_history.py  --start 2023-01-01   # -> state/history/insider.csv (slow: EDGAR at 10 req/s)
```

Bar panels are cached under `state/cache`, keyed by fetch date, and pruned after `data.cache_retention_days` (2).

### Calibration notes worth keeping

**Earnings window.** A long hold will usually span an earnings report - that is where much of the money is: trades
that held through a report produced 75% of all profit. The entry block exists only to keep a fresh stop from being
gapped over, and is set to 5 calendar days (a 21-day block lost badly; 0 vs 5 is within noise). A forced exit before
the report was tested and rejected; it stays wired at `exits.earnings_exit_days_before: 0`. The pre-report *exit*
counts trading days; the *entry block* counts calendar days, which is how 5 was calibrated - so on a Tuesday, a report
the following Monday is not blocked.

**Finnhub's bulk calendar truncates at 1500 rows** and returns the far end of the requested range, so a wide date
query silently reports no upcoming earnings for every symbol. `desk/facts.py` queries one symbol at a time for this
reason. Do not "optimize" it back into a single range request.

## Known limitations

- **`backtest.py` is an upper bound, by about half.** Its universe is a frozen list of survivors; use
  `robustness.py` for anything that informs a decision. Fills are approximated from daily OHLC.
- **The AI layer is not backtested.** The red team and analyst vetoes are only sampled at random in
  `robustness.py`'s frictions mode; the shadow book is collecting the live evidence.
- **Filing and news screens only look back 90 days.** An older dilution filing no longer flags the name.
- **FRED is unreachable from some networks.** `desk/regime.py` falls back to Yahoo for VIX; check the brief says
  `VIX <number>` and not `VIX None`.
- **StockTwits is an undocumented public endpoint** and may rate-limit or change; the z-score needs ~15 days of the
  system's own history per name before it activates. Failures degrade to "no social data".
- **Options IV needs a listed chain** 20-50 days out; names without options get no IV adjustment.
- **Whole shares only.** Alpaca bracket orders do not support fractional shares, so at a 7% cap on ~$9k equity,
  names above ~$630 cannot be bought.
- **Taxes.** 21-126 day holds are short-term gains; wash sales are not tracked.
- **Pattern day trader rule.** Under $25k equity, same-day round trips are limited. The scheduled run sells after the
  close, so its exits cannot complete one; a run launched during market hours defers such exits once the day-trade
  count is used. The kill switch ignores the guard.
- **A run before the day's bars are final only protects** (`PROTECT-ONLY RUN` in the brief); `--force` runs
  everything anyway and the brief warns that the bar is partial.
- **A config.yaml typo stops the run.** Every risk and sizing key is required at load (`desk/config.py`), and
  `account.paper` must be true: no code path builds a live trading client.

### Live against the backtest

Every rule that differs between `run_daily.py` and the backtest. None is fixed in code, because fixing any of them
changes either live trading or the frozen baseline; each is a decision for the owner.

| Rule | Live | Backtest | Effect |
|---|---|---|---|
| Kill switch, 15% | resumes after 20 sessions, or sooner with `--resume` | resumes after 20 sessions | the same unless you resume early |
| Kill switch, 25% floor | halts until `--resume --reset-hwm` | resumes after 60 sessions, then re-baselines | the backtest is optimistic after a floor hit |
| Analyst, red team, implied-vol half size, headline / social blocks | applied | not simulated (`robustness.py --frictions` samples them) | live runs smaller and more selective |
| Exit fill | decided at 16:40 ET, market sell at the next open | time, decay, regime and switch exits fill at that day's close | one overnight gap per exit |
| Hold clock (min 21 / max 126 days) | counts from the day the entry was staged | counts from the fill | live reaches each limit one session early |
| Stop | set from the signal close, moved to the fill on the next run | computed from the fill | one evening at the signal-close distance |
| Entry pacing (4/day) | applied to sized entries | applied to picks before sizing | the backtest can buy fewer names on a day |
| Earnings block boundary | report 0-5 calendar days out | 1-5 days out | live is one day stricter |
| Fact-checked pool | only the top 30 ranked candidates get fact sheets and can be bought | earnings checked for the top 60; anything eligible can be bought | the backtest can reach further down the ranking |
| VIX | FRED, usually one session behind | that day's close | the backtest's regime flips a day sooner |
| Holding with no bars | kept (the broker decides) | sold at its last close after 5 sessions | the backtest flatters delistings |
| Final gate, cash | `enforce_entry_limits`, non-marginable buying power | neither | cannot bind at the configured target |
| Guards | stale data, day trades, trading day, protect-only before 16:20 ET, no entries without fact sheets or an earnings calendar | none | live skips or softens some days |
| Earnings dates | Finnhub, per symbol | yfinance history from June 2023 | block inert before mid-2023 in the backtest |

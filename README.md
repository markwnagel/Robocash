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
  6. FACT SHEETS    for the top 30 candidates + holdings (desk/facts.py, all sources optional):
                      SEC EDGAR   dilution filings (8-K 3.02, S-1), shelf filings (424B, S-3) flagged
                                  separately, 8-K items, Form 4 insider buys/sells
                      Finnhub     next earnings date (one query per symbol - see the note below)
                      Alpaca      Benzinga headlines (Claude-classified: tone + offering/guidance/M&A/legal flags)
                      Alpaca      options chain -> ATM implied vol vs 20d realized vol, put skew
                      StockTwits  message rate vs the name's own history (z-score), bullish share
                    HARD BLOCKS (candidates only): earnings within 21 days, unambiguous dilution filing,
                      headline offering or acquisition target, social-volume z >= 3.
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
                    has not reviewed is dropped; no entries at all if the last bar is not the latest session.
  9. RED TEAM       Claude writes the strongest bear case per name (trap type + severity) from the fact sheet
 10. ANALYST        Claude rules KEEP / REMOVE / REDUCE per name with facts + bear case in hand. Veto-only.
                    REDUCE keeps `analyst.reduce_factor` (0.75) of an entry's shares, floored, min 1 share;
                    capped at 1.0 in code so no config value can make a veto enlarge a position.
                    If the API call fails: no new entries today, holdings untouched.
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
                    A holding already below its stop with nothing resting is closed outright.
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
- `FINNHUB_API_KEY` (free tier) gives the earnings calendar. Without it the earnings-in-window rule falls back to
  the analyst's soft check.

## Commands

```bash
.venv/bin/python run_daily.py --dry-run       # full run incl. fact sheets + LLM roles; submits nothing, saves no state
.venv/bin/python run_daily.py                 # real run against the account in config.yaml (paper: true)
.venv/bin/python run_daily.py --resume        # clear a kill-switch halt (re-baselines the high-water mark)
.venv/bin/python run_daily.py --no-analyst    # skip red team + analyst (engine + facts + risk only)
.venv/bin/python audit.py                     # weekly process audit (add --stats for numbers only, no LLM)
.venv/bin/python report_daily.py              # daily activity email (14:00 job); --print to build without sending
.venv/bin/python backtest.py --start 2023-09-11 --end 2026-09-11              # frozen universe, reproducible
.venv/bin/python backtest.py --start 2023-09-11 --end 2026-09-11 --risk-per-trade 0.0125 --stagger 6
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
the broker and are unaffected). For a machine-independent schedule, run it from a cloud scheduler instead.

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
- **Hard alarms do not depend on the model.** A desk run that was missed on a trading day, crashed, or started
  and never finished is SEV1 in code. If the Claude call fails, a deterministic fallback report is still written
  and the alarms still escalate.
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

- **Backtest is an upper bound.** The universe is a frozen snapshot of names that were liquid on 2026-09-21
  (survivorship bias), fills are approximated from daily OHLC, and the red team / analyst vetoes are not
  simulated - live, the analyst halved over half the entries in the first two weeks, so live runs at well
  under the risk the backtest assumes.
- **The backtest's kill switch resumes by itself.** After a halt it sits out 20 trading days
  (`--halt-cooldown-days`) and starts again; live waits for you to run `--resume`. It fired three times in
  three years on the baseline, so expect roughly one halt a year, and treat any result that depends on halt
  timing as optimistic.
- **An analyst REDUCE on a name already held is reported, not acted on.** Trimming part of a position
  would leave the broker-side stop covering more shares than remain, so the verdict lands in the brief
  (`flagged REDUCE ... review manually`) and on the position as `analyst_flag`. Entries are still cut
  before purchase as normal. Acting on it automatically needs a partial-exit path and a backtest first.
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
- **Pattern day trader rule.** Under $25k equity, same-day round trips are limited. The job never sells
  something it bought that day (except via the kill switch), and a stop firing on entry day counts against you.

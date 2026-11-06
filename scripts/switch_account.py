#!/usr/bin/env python
"""Move the desk to a new Alpaca paper account.

Put the new account's PAPER keys in .env first (the old keys stop being used), then:

  python scripts/switch_account.py            show what would change; changes nothing
  python scripts/switch_account.py --apply    do it

Kept, because it is what the desk has learned: the trade log (trades.csv), the AI decision log (shadow book),
backtest trials, monitor findings and owner resolutions, audits, briefs and caches.

Archived to state/accounts/<old account>/: state.json - its high-water marks, halt and positions belong to the old
account. A fresh state.json is written for the new one, so the kill switch starts from the new balance.

Reset: config account.since becomes the first desk run on the new account. The email's "since inception" already
follows the account (Alpaca's equity history); the parity replay, its since-release block and the stop rule read
account.since, so results against SPY and SPMO start there. The stop rule's two checkpoints move by the same
number of days, keeping the spacing the owner set (2026-11-01: warn 6 months, decide 52 weeks after the start).

Refuses unless the new account is a different, flat (no positions, no open orders) paper account.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desk import state as S  # noqa: E402
from desk.config import ROOT, alpaca_keys, load_config, measured_from, require_paper  # noqa: E402

ET = ZoneInfo("America/New_York")
DESK_RUN = time(16, 40)          # the evening desk job (launchd 13:40 PT)
CONFIG = ROOT / "config.yaml"


def first_run_day(now_et: datetime, sessions: list) -> date:
    """The first evening desk run the new account will get: today's if it has not started yet, else the next
    session's. Its brief carries the flat balance - the baseline every "since" figure is measured from."""
    today = now_et.date()
    upcoming = sorted(d for d in sessions if d >= today)
    if upcoming and upcoming[0] == today and now_et.time() < DESK_RUN:
        return today
    later = [d for d in upcoming if d > today]
    if not later:
        raise RuntimeError("no upcoming trading session in the calendar")
    return later[0]


def shifted_checkpoints(rule: dict, old_from: date | None, new_from: date) -> dict:
    """The stop rule's warn_on / decide_on moved by (new_from - old_from) days. {} when there is nothing to move."""
    if not rule or not old_from or new_from <= old_from:
        return {}
    shift = new_from - old_from
    return {k: (date.fromisoformat(str(rule[k])) + shift).isoformat() for k in ("warn_on", "decide_on") if rule.get(k)}


def edit_config(text: str, since: date, checkpoints: dict) -> str:
    """Set account.since and the stop-rule dates in place, so config.yaml keeps its comments."""
    acct = re.search(r"^account:\n((?:[ \t]+.*\n)+)", text, re.M)
    if not acct:
        raise ValueError("config.yaml: no account section")
    body = acct.group(1)
    if re.search(r"^  since:", body, re.M):
        body = re.sub(r"^(  since:\s*)\S+", rf"\g<1>{since.isoformat()}", body, count=1, flags=re.M)
    else:
        body += f"  since: {since.isoformat()}    # first desk run on this paper account (scripts/switch_account.py)\n"
    text = text[:acct.start(1)] + body + text[acct.end(1):]
    for k, v in checkpoints.items():
        text, n = re.subn(rf"^(    {k}:\s*)\S+", rf"\g<1>{v}", text, count=1, flags=re.M)
        if n != 1:
            raise ValueError(f"config.yaml: governance.stop_rule.{k} not found")
    return text


def fresh_state(number: str, since: date) -> dict:
    st = S._default_state()
    st.update(account_number=number, account_since=since.isoformat(), resume_on="", manual_open={})
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="archive the old state, start the new one, edit config.yaml")
    args = ap.parse_args()

    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetCalendarRequest, GetOrdersRequest

    cfg = load_config()
    t = TradingClient(*alpaca_keys(), paper=require_paper(cfg))
    a = t.get_account()
    new = str(a.account_number)
    st = S.load_state()
    old = st.get("account_number") or ""
    problems = []
    if old and old == new:
        problems.append(f"the keys in .env still reach account {S.masked(new)}, the one state.json belongs to - "
                        "put the new account's keys in .env first")
    if a.trading_blocked or a.account_blocked:
        problems.append("the new account is blocked by the broker")
    held = t.get_all_positions()
    if held:
        problems.append(f"the new account already holds {len(held)} position(s); start from a flat account")
    pending = t.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    if pending:
        problems.append(f"the new account has {len(pending)} open order(s); cancel them first")

    now = datetime.now(ET)
    cal = t.get_calendar(GetCalendarRequest(start=now.date(), end=now.date() + timedelta(days=10)))
    since = first_run_day(now, [c.date for c in cal])
    old_from = measured_from(cfg)
    rule = (cfg.get("governance") or {}).get("stop_rule") or {}
    moves = shifted_checkpoints(rule, old_from, since)
    equity = float(a.equity)

    print(f"old account   {S.masked(old)}  (state.json: high-water ${st.get('high_water_mark', 0):,.2f}, "
          f"all-time ${st.get('all_time_hwm', 0):,.2f}, {len(st.get('positions') or {})} position(s)"
          f"{', HALTED' if st.get('halted') else ''})")
    print(f"new account   {S.masked(new)}  equity ${equity:,.2f}, cash ${float(a.cash):,.2f}, paper")
    print(f"account.since {since}  (the first desk run on the new account; results vs SPY and SPMO start here)")
    for k, v in moves.items():
        print(f"stop rule     {k}: {rule[k]} -> {v}")
    start = float(cfg["account"]["starting_capital"])
    if abs(equity - start) > 1:
        print(f"note          account.starting_capital is ${start:,.0f} (backtests, trials); the new account has "
              f"${equity:,.0f}. Left as is - live sizing reads the account, not this setting.")
    if st.get("positions"):
        print(f"note          state.json lists {len(st['positions'])} position(s) on the old account; they are no "
              "longer managed after the switch")
    if problems:
        print("\nREFUSED:\n  - " + "\n  - ".join(problems))
        sys.exit(1)
    if not args.apply:
        print("\nNothing changed. Re-run with --apply to switch.")
        return

    archive = S.STATE_DIR / "accounts" / (old or "previous")
    archive.mkdir(parents=True, exist_ok=True)
    stamp = now.date().isoformat()
    if S.STATE_FILE.exists():
        shutil.copy2(S.STATE_FILE, archive / f"state-{stamp}.json")
    S.atomic_write_text(CONFIG, edit_config(CONFIG.read_text(), since, moves))
    S.save_state(fresh_state(new, since))
    with open(S.STATE_DIR / "accounts" / "switches.jsonl", "a") as f:
        f.write(json.dumps({"date": stamp, "from": old, "to": new, "since": since.isoformat(),
                            "old_high_water_mark": st.get("high_water_mark"), "old_all_time_hwm": st.get("all_time_hwm"),
                            "new_equity": equity, "stop_rule_moves": moves}) + "\n")
    print(f"\nSwitched. Old state archived to {archive.relative_to(ROOT)}/state-{stamp}.json; "
          "config.yaml edited - commit it.")


if __name__ == "__main__":
    main()

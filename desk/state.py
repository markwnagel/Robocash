import csv
import json
from datetime import date
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.csv"
BRIEFS_DIR = STATE_DIR / "briefs"

TRADE_FIELDS = [
    "symbol", "entry_date", "entry_price", "exit_date", "exit_price",
    "qty", "pnl", "pnl_pct", "hold_days", "exit_reason", "thesis", "stop", "target",
]


def _default_state() -> dict:
    return {
        "high_water_mark": 0.0,
        "all_time_hwm": 0.0,
        "halted": False,
        "hard_halted": False,
        "halt_reason": "",
        "halt_date": "",
        "positions": {},
        "last_run": "",
    }


def load_state() -> dict:
    STATE_DIR.mkdir(exist_ok=True)
    if not STATE_FILE.exists():
        return _default_state()
    with open(STATE_FILE) as f:
        st = json.load(f)
    for k, v in _default_state().items():
        st.setdefault(k, v)
    return st


def save_state(st: dict) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2, sort_keys=True)
    tmp.replace(STATE_FILE)


def record_trade(row: dict) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    new = not TRADES_FILE.exists()
    with open(TRADES_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def write_brief(text: str, day: date) -> Path:
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    p = BRIEFS_DIR / f"{day.isoformat()}.md"
    p.write_text(text)
    return p

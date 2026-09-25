import csv
import json
import os
import pickle
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


def atomic_write_text(path: Path, text: str) -> None:
    """Write-then-rename, so a run killed mid-write leaves the previous file rather than half a file. Every
    file a later run parses (state, universe, caches, monitor state) goes through here or atomic_pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def atomic_pickle(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


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
    atomic_write_text(STATE_FILE, json.dumps(st, indent=2, sort_keys=True))


def record_trade(row: dict) -> bool:
    """Append one closed trade. Idempotent: a run that dies after logging a close but before saving state
    sees the position again next time and would log it twice, so a row already present for the same symbol,
    entry date and exit date is not written again. Returns whether a row was written."""
    STATE_DIR.mkdir(exist_ok=True)
    key = (str(row.get("symbol")), str(row.get("entry_date")), str(row.get("exit_date")))
    new = not TRADES_FILE.exists()
    if not new:
        with open(TRADES_FILE, newline="") as f:
            if any((r.get("symbol"), r.get("entry_date"), r.get("exit_date")) == key for r in csv.DictReader(f)):
                return False
    with open(TRADES_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
    return True


def write_brief(text: str, day: date) -> Path:
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    p = BRIEFS_DIR / f"{day.isoformat()}.md"
    atomic_write_text(p, text)
    return p

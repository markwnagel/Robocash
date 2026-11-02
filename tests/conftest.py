import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desk import config as config_mod  # noqa: E402
from desk.config import load_config as _load_live_config  # noqa: E402

# The suite tests mechanics (brackets, sizing, halts, exits) on a fixed strategy: the v2.0 geometry these tests were
# written for. A strategy release changes config.yaml, not what the mechanics must do, so it should not rewrite
# half the suite. Tests of the live strategy itself use the live_cfg fixture.
MECHANICS_BASELINE = {
    "signals": {"momentum_lookback_days": 126},
    "portfolio": {"target_positions": 12, "max_positions": 15, "max_position_pct": 0.10},
    "exits": {"stop_atr_multiple": 2.0, "take_profit_r_multiple": 3.0, "min_hold_days": 15, "max_hold_days": 60,
              "signal_decay_rank": 40},
}


def mechanics_config(*a, **k):
    cfg = _load_live_config(*a, **k)
    for section, values in MECHANICS_BASELINE.items():
        cfg[section].update(values)
    return cfg


@pytest.fixture(autouse=True)
def _mechanics_baseline(monkeypatch):
    import audit, backtest, monitor, report_daily, robustness, run_daily
    for mod in (config_mod, audit, backtest, monitor, report_daily, robustness, run_daily):
        monkeypatch.setattr(mod, "load_config", mechanics_config)


@pytest.fixture
def cfg():
    return mechanics_config()


@pytest.fixture
def live_cfg():
    """config.yaml as the desk runs it today."""
    return _load_live_config()


@pytest.fixture
def panel():
    """30 synthetic symbols x 320 days. S00 trends up strongly, S01 is a copy of S00, S29 trends down."""
    rng = np.random.default_rng(7)
    n, days = 30, 320
    # Ends on the latest weekday so the runner's stale-data guard sees fresh bars.
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    drift = np.linspace(0.0015, -0.0015, n)
    rets = rng.normal(0, 0.012, size=(days, n)) + drift
    rets[:, 1] = rets[:, 0]
    close = 50 * np.exp(np.cumsum(rets, axis=0))
    syms = [f"S{i:02d}" for i in range(n)]
    close = pd.DataFrame(close, index=dates, columns=syms)
    high = close * (1 + rng.uniform(0.002, 0.02, size=close.shape))
    low = close * (1 - rng.uniform(0.002, 0.02, size=close.shape))
    opn = (high + low) / 2
    vol = pd.DataFrame(rng.integers(1_000_000, 5_000_000, size=close.shape), index=dates, columns=syms)
    return {"open": opn, "high": high, "low": low, "close": close, "volume": vol}


@pytest.fixture(autouse=True)
def _telemetry_to_tmp(tmp_path, monkeypatch):
    """Tests exercise the real LLM-call paths; their telemetry must never land in state/telemetry, where
    the monitor would read fake calls as production traffic."""
    from desk import telemetry
    monkeypatch.setattr(telemetry, "DIR", tmp_path / "telemetry")


@pytest.fixture(autouse=True)
def _healthy_schedule(monkeypatch):
    """launchctl and pmset describe this Mac, not the code under test: every test sees a loaded, woken
    schedule and no installed slots unless it sets its own."""
    from desk import schedule
    monkeypatch.setattr(schedule, "loaded_jobs", lambda run=None: {j: True for j in schedule.JOBS})
    monkeypatch.setattr(schedule, "repeating_wake", lambda run=None: "wakepoweron at 1:40PM weekdays only")
    monkeypatch.setattr(schedule, "installed_slots", lambda job: [])

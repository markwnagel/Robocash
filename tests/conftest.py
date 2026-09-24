import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desk.config import load_config  # noqa: E402


@pytest.fixture
def cfg():
    return load_config()


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

from __future__ import annotations

import io
import logging
import time
import urllib.request
from pathlib import Path

import pandas as pd

log = logging.getLogger("desk.regime")
FRED_VIX_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
LABELS = {"off": "risk-off", "neutral": "risk-on", "strong": "risk-on (strong)"}


def _vix_from_fred(p: Path) -> pd.Series:
    req = urllib.request.Request(FRED_VIX_CSV, headers={"User-Agent": "alpaca-desk"})
    for attempt in range(3):
        try:
            raw = urllib.request.urlopen(req, timeout=20).read()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "vix"]
    s = df.assign(date=pd.to_datetime(df["date"]), vix=pd.to_numeric(df["vix"], errors="coerce")).set_index("date")["vix"].dropna()
    p.write_bytes(raw)
    return s


def _vix_from_yahoo(p: Path) -> pd.Series:
    import yfinance as yf
    s = yf.download("^VIX", period="5y", interval="1d", progress=False, auto_adjust=False)["Close"].squeeze().dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s.astype(float).rename("vix")
    s.rename_axis("date").to_frame().to_csv(p, index=True, header=["vix"])
    return s


def vix_series(cache_dir: Path) -> pd.Series | None:
    """FRED first, Yahoo second. FRED is unreachable from some networks and a silent None here
    disables half the regime gate (the volatility half) while the SPY trend half keeps reporting."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / "vixcls.csv"
    fresh = p.exists() and time.time() - p.stat().st_mtime < 12 * 3600
    if fresh:
        try:
            df = pd.read_csv(p)
            df.columns = ["date", "vix"]
            return df.assign(date=pd.to_datetime(df["date"]), vix=pd.to_numeric(df["vix"], errors="coerce")).set_index("date")["vix"].dropna()
        except Exception as e:
            log.warning("VIX cache unreadable, refetching: %s", e)
    errors = []
    for name, fn in (("FRED", _vix_from_fred), ("Yahoo", _vix_from_yahoo)):
        try:
            s = fn(p)
            if len(s):
                if name != "FRED":
                    log.info("VIX from %s (FRED unavailable)", name)
                return s
            errors.append(f"{name}: empty")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__} {e}")
    log.warning("VIX unavailable from any source (regime gate running on SPY trend only): %s", "; ".join(errors))
    return None


def regime_state_series(spy_close: pd.Series, vix: pd.Series | None, rcfg: dict) -> pd.Series:
    """'off' | 'neutral' | 'strong' per date, using only information available at each date.
    'strong' exists only when the ladder is enabled: SPY comfortably above its MA with VIX subdued."""
    ma = spy_close.rolling(rcfg["spy_ma_days"]).mean()
    off = spy_close < ma
    v = vix.reindex(spy_close.index, method="ffill") if vix is not None else None
    if v is not None:
        off = off | (v > rcfg["vix_risk_off"])
    state = pd.Series("neutral", index=spy_close.index, dtype=object)
    state[off] = "off"
    if rcfg.get("ladder"):
        strong = spy_close >= ma * (1 + float(rcfg.get("strong_spy_above_ma_pct") or 0))
        if v is not None:
            strong = strong & (v < float(rcfg.get("strong_vix_below") or float("inf")))
        state[strong & ~off] = "strong"
    state[ma.isna()] = "neutral"
    return state


def regime_params(state: str, pcfg: dict, rcfg: dict) -> dict:
    """Position count, per-position cap and whether to be flat, for one regime state."""
    if state == "strong":
        return {"target_positions": int(rcfg.get("strong_target_positions") or pcfg["target_positions"]),
                "max_position_pct": float(rcfg.get("strong_max_position_pct") or pcfg["max_position_pct"]),
                "liquidate": False}
    if state == "off":
        liq = bool(rcfg.get("off_liquidate"))
        return {"target_positions": 0 if liq else max(1, int(round(pcfg["target_positions"] * rcfg["risk_off_scale"]))),
                "max_position_pct": float(pcfg["max_position_pct"]), "liquidate": liq}
    return {"target_positions": int(pcfg["target_positions"]), "max_position_pct": float(pcfg["max_position_pct"]),
            "liquidate": False}


def regime_scale_series(spy_close: pd.Series, vix: pd.Series | None, rcfg: dict) -> pd.Series:
    """Per-date multiplier on target position count (two-state view of the regime)."""
    state = regime_state_series(spy_close, vix, rcfg)
    scale = pd.Series(1.0, index=spy_close.index)
    scale[state == "off"] = rcfg["risk_off_scale"]
    return scale


def current_regime(spy_close: pd.Series, vix: pd.Series | None, rcfg: dict) -> dict:
    state = str(regime_state_series(spy_close, vix, rcfg).iloc[-1])
    ma = spy_close.rolling(rcfg["spy_ma_days"]).mean().iloc[-1]
    vix_last = float(vix.iloc[-1]) if vix is not None and len(vix) else None
    return {
        "state": state,
        "scale": float(rcfg["risk_off_scale"]) if state == "off" else 1.0,
        "spy": float(spy_close.iloc[-1]),
        "spy_ma": float(ma) if pd.notna(ma) else None,
        "vix": vix_last,
        "label": LABELS[state],
    }

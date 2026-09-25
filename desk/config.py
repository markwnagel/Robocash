import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# Keys every job reads without a fallback. A typo in config.yaml must stop the run at load time: code that
# read one of these with .get() would otherwise run with the limit silently switched off (a misspelt
# max_cumulative_drawdown_pct used to disable the hard floor without a word).
REQUIRED = {
    "account": ("paper", "starting_capital"),
    "data": ("feed", "history_days", "cache_dir"),
    "universe": ("exchanges", "min_price", "min_avg_dollar_volume", "max_symbols", "refresh_days",
                 "exclude_name_keywords"),
    "signals": ("momentum_lookback_days", "momentum_skip_days", "short_lookback_days", "trend_ma_days",
                "atr_days", "vol_days", "weights", "max_atr_pct"),
    "portfolio": ("sizing", "risk_per_trade_pct", "min_shares_per_position", "target_positions", "max_positions",
                  "max_position_pct", "cash_floor_pct", "max_new_entries_per_day", "correlation_days",
                  "correlation_threshold", "max_correlated_peers"),
    "exits": ("stop_atr_multiple", "take_profit_r_multiple", "trail_atr_multiple", "min_hold_days",
              "max_hold_days", "signal_decay_rank", "earnings_exit_days_before"),
    "risk": ("max_drawdown_pct", "max_cumulative_drawdown_pct", "daily_loss_halt_pct", "max_daytrades_per_5d"),
    "regime": ("enabled", "spy_ma_days", "vix_risk_off", "risk_off_scale"),
    "facts": ("enabled", "candidate_pool", "exclude_earnings_within_days", "iv_over_rv_reduce_at"),
    "analyst": ("enabled", "candidates_to_review", "reduce_factor"),
}
FRACTIONS = {("risk", "max_drawdown_pct"), ("risk", "max_cumulative_drawdown_pct"), ("risk", "daily_loss_halt_pct"),
             ("portfolio", "risk_per_trade_pct"), ("portfolio", "max_position_pct"), ("portfolio", "cash_floor_pct"),
             ("analyst", "reduce_factor")}


def validate(cfg: dict) -> None:
    """Raise ValueError naming every missing or out-of-range key. Checks presence and sanity only; the values
    themselves are the owner's to choose."""
    problems = []
    for section, keys in REQUIRED.items():
        body = cfg.get(section)
        if not isinstance(body, dict):
            problems.append(f"{section}: section missing")
            continue
        problems += [f"{section}.{k}: missing" for k in keys if k not in body]
    for section, key in FRACTIONS:
        v = (cfg.get(section) or {}).get(key)
        if v is not None and not (isinstance(v, (int, float)) and 0 <= v <= 1):
            problems.append(f"{section}.{key}: {v!r} is not a fraction in [0, 1]")
    if (cfg.get("account") or {}).get("paper") is not True:
        problems.append("account.paper: must be true - this desk only trades a paper account")
    if problems:
        raise ValueError("config.yaml is invalid:\n  " + "\n  ".join(problems))


def load_config(path=None) -> dict:
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    validate(cfg)
    cfg["_root"] = str(ROOT)
    return cfg


def require_paper(cfg: dict) -> bool:
    """The only value any TradingClient may be built with. Raises rather than ever returning False, so a
    config edited after load (or a test double) cannot hand a live account to the broker client."""
    if cfg.get("account", {}).get("paper") is not True:
        raise RuntimeError("refusing to build a trading client: account.paper is not true")
    return True


def alpaca_keys() -> tuple:
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set (see .env.example)")
    return key, secret

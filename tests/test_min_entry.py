"""portfolio.min_entry_fraction: a pick the remaining cash can fund only as a sliver is skipped (owner, 2026-11-06)."""
from desk.portfolio import too_small
from tests.test_run_daily import FakeBroker, _patched_cfg, run, wired  # noqa: F401  (wired is a fixture)


def test_a_sliver_is_too_small_and_off_means_never():
    assert too_small(2, 280, 0.5) and too_small(1, 4, 0.5)
    assert not too_small(2, 4, 0.5) and not too_small(280, 280, 0.5)
    assert not too_small(2, 280, 0) and not too_small(2, 280, None) and not too_small(0, 0, 0.5)


def test_the_desk_skips_slivers_when_the_rule_is_on(wired, cfg, monkeypatch):
    holder, sent = wired
    holder["broker"] = FakeBroker(equity=10000.0, last_equity=10000.0, cash=1200.0)
    run(["--no-analyst"])
    assert "cash-limited:" in sent[-1]                                    # off: the last pick takes what is left
    _patched_cfg(monkeypatch, portfolio={"min_entry_fraction": 0.99})
    holder["broker"] = FakeBroker(equity=10000.0, last_equity=10000.0, cash=1200.0)
    run(["--no-analyst"])
    assert "cash-limited:" not in sent[-1] and "skipped " in sent[-1] and "picked again when cash frees up" in sent[-1]

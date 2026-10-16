

def test_the_brief_warns_when_the_desk_is_not_running_main():
    """2026-10-15 22:53: the checkout was switched back to the merged r38-counters branch a minute after main
    took two merges; the next desk run would have used three-day-old code without a word in the brief."""
    from desk.telemetry import checkout_warning
    assert checkout_warning({"branch": "main", "behind_main": 0}) is None
    assert checkout_warning({}) is None and checkout_warning({"branch": None}) is None
    w = checkout_warning({"branch": "r38-counters", "behind_main": 5})
    assert w.startswith("WARNING:") and "branch 'r38-counters', not main, 5 commits behind origin/main" in w
    assert "main 1 commit behind origin/main" in checkout_warning({"branch": "main", "behind_main": 1})
    assert "a detached commit, not main" in checkout_warning({"branch": "HEAD", "behind_main": None})


from datetime import date

from desk.portfolio import pace_entries

TODAY = date(2026, 11, 6)
E = [{"symbol": s} for s in "ABCDEF"]


def test_the_daily_cap_counts_entries_an_earlier_run_placed_today():
    held = {"X": {"entry_date": "2026-11-06"}, "Y": {"entry_date": "2026-11-06"}, "Z": {"entry_date": "2026-11-05"}}
    kept, note = pace_entries(E, 4, held, TODAY)
    assert [e["symbol"] for e in kept] == ["A", "B"] and "2 already placed today" in note


def test_a_day_already_at_the_cap_buys_nothing_more():
    held = {s: {"entry_date": "2026-11-06"} for s in "PQRSTUVWXYZ"}
    kept, note = pace_entries(E, 4, held, TODAY)
    assert kept == [] and "6 deferred" in note


def test_a_fresh_day_keeps_the_usual_cap_and_none_means_no_cap():
    kept, note = pace_entries(E, 4, {"Z": {"entry_date": "2026-11-05"}}, TODAY)
    assert len(kept) == 4 and note == "entry pacing: 2 deferred to later sessions (max 4/day)"
    assert pace_entries(E, None, {}, TODAY) == (E, "")

import sys
from datetime import date, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import switch_account as W  # noqa: E402
from desk.config import account_since, measured_from, validate  # noqa: E402
from desk.state import account_change  # noqa: E402

FRI, MON = date(2026, 11, 6), date(2026, 11, 9)


def test_the_first_run_is_tonight_before_the_desk_job_and_the_next_session_after_it():
    sessions = [FRI, MON]
    assert W.first_run_day(datetime(2026, 11, 6, 9, 0, tzinfo=W.ET), sessions) == FRI
    assert W.first_run_day(datetime(2026, 11, 6, 16, 45, tzinfo=W.ET), sessions) == MON
    assert W.first_run_day(datetime(2026, 11, 7, 10, 0, tzinfo=W.ET), [MON]) == MON     # a Saturday


def test_the_stop_rule_checkpoints_keep_their_spacing():
    rule = {"warn_on": "2027-04-30", "decide_on": "2027-10-29"}
    assert W.shifted_checkpoints(rule, date(2026, 10, 30), FRI) == {"warn_on": "2027-05-07", "decide_on": "2027-11-05"}
    assert W.shifted_checkpoints(rule, FRI, date(2026, 10, 30)) == {}     # never earlier


def test_the_config_edit_sets_the_dates_and_keeps_the_comments():
    text = (ROOT / "config.yaml").read_text()
    out = W.edit_config(text, FRI, {"warn_on": "2027-05-07", "decide_on": "2027-11-05"})
    cfg = yaml.safe_load(out)
    assert cfg["account"]["since"] == FRI and cfg["account"]["paper"] is True
    assert str(cfg["governance"]["stop_rule"]["warn_on"]) == "2027-05-07"
    assert str(cfg["governance"]["stop_rule"]["decide_on"]) == "2027-11-05"
    assert str(cfg["release"]["since"]) == str(yaml.safe_load(text)["release"]["since"])      # release untouched
    assert out.count("\n") == text.count("\n") and out.count("#") == text.count("#")


def test_results_are_measured_from_the_later_of_the_release_and_the_account():
    cfg = {"release": {"name": "v2.1", "since": "2026-10-30"}, "account": {"since": None}}
    assert account_since(cfg) is None and measured_from(cfg) == date(2026, 10, 30)
    cfg["account"]["since"] = "2026-11-06"
    assert measured_from(cfg) == FRI


def test_a_misspelt_account_start_stops_the_config_load():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["account"]["since"] = "2026-13-01"
    try:
        validate(cfg)
    except ValueError as e:
        assert "account.since" in str(e)
    else:
        raise AssertionError("validate accepted a bad account.since")


def test_the_desk_refuses_keys_for_another_account():
    assert account_change({}, "PA111") == ""                                   # nothing recorded yet
    assert account_change({"account_number": "PA111"}, "PA111") == ""
    assert account_change({"account_number": "PA111"}, "") == ""               # a broker that does not say
    msg = account_change({"account_number": "PA1111"}, "PA2222")
    assert msg.startswith("ACCOUNT CHANGED") and "...1111" in msg and "...2222" in msg and "switch_account" in msg


def test_a_fresh_state_starts_flat_with_no_high_water_mark():
    st = W.fresh_state("PA2222", MON)
    assert st["account_number"] == "PA2222" and st["account_since"] == "2026-11-09"
    assert st["high_water_mark"] == 0 and st["all_time_hwm"] == 0 and not st["halted"] and st["positions"] == {}

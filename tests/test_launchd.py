"""The scheduled jobs: launchd files point at named scripts, and the scripts run the right programs from the
project root. macOS lists a background job by the program it starts; as inline /bin/sh commands both jobs showed
as "sh", were switched off under Login Items & Extensions, and after a restart never loaded (2026-09-28)."""
import os
import plistlib
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS = {"com.alpaca-desk.daily": ("teamrobo-desk", 13, 45), "com.alpaca-desk.report": ("teamrobo-email", 14, 0)}


def test_each_job_starts_its_named_script_on_the_same_schedule():
    for label, (script, hour, minute) in JOBS.items():
        plist = plistlib.loads((ROOT / "launchd" / f"{label}.plist").read_bytes())
        assert plist["Label"] == label                                  # the monitor and README key on it
        assert plist["ProgramArguments"] == [f"__PROJECT_DIR__/launchd/{script}"]
        path = ROOT / "launchd" / script
        assert path.stat().st_mode & stat.S_IXUSR, f"{script} must be executable"
        slots = {(d["Weekday"], d["Hour"], d["Minute"]) for d in plist["StartCalendarInterval"]}
        assert slots == {(d, hour, minute) for d in range(1, 6)}


def test_each_job_names_itself_to_the_runs_it_starts():
    """launchd's XPC_SERVICE_NAME did not reach Python under the named scripts, so on 2026-09-30 every scheduled
    run was recorded as manual (monitor F-17). The job's label now travels as an ordinary variable."""
    for p in sorted((ROOT / "launchd").glob("com.alpaca-desk.*.plist")):
        plist = plistlib.loads(p.read_bytes())
        assert plist["EnvironmentVariables"] == {"TEAMROBO_JOB": plist["Label"]}, p.name


def test_the_job_name_reaches_python_through_the_scripts(tmp_path):
    root = _fake_project(tmp_path)
    (root / ".venv" / "bin" / "python").write_text(
        f'#!/bin/sh\necho "$*=$TEAMROBO_JOB" >> "{tmp_path}/calls.txt"\n')
    env = {**os.environ, "TEAMROBO_JOB": "com.alpaca-desk.report"}
    subprocess.run([str(root / "launchd" / "teamrobo-email")], cwd="/", check=True, timeout=60, env=env)
    assert (root / "calls.txt").read_text().splitlines() == ["monitor.py=com.alpaca-desk.report",
                                                              "report_daily.py=com.alpaca-desk.report"]


def _fake_project(tmp_path: Path, exit_code: int = 0) -> Path:
    """A project root whose python only records how it was called."""
    (tmp_path / "launchd").mkdir()
    for s in ("teamrobo-desk", "teamrobo-email"):
        shutil.copy2(ROOT / "launchd" / s, tmp_path / "launchd" / s)
    py = tmp_path / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text(f'#!/bin/sh\necho "$(pwd) $*${{TEAMROBO_TRIGGER:+ [$TEAMROBO_TRIGGER]}}" >> "{tmp_path}/calls.txt"\n'
                  f'exit {exit_code}\n')
    py.chmod(0o755)
    return tmp_path


def test_the_email_script_runs_the_monitor_then_the_report_from_the_project_root(tmp_path):
    root = _fake_project(tmp_path)
    subprocess.run([str(root / "launchd" / "teamrobo-email")], cwd="/", check=True, timeout=60)
    calls = (root / "calls.txt").read_text().splitlines()
    assert calls == [f"{root.resolve()} monitor.py", f"{root.resolve()} report_daily.py"]


def test_the_desk_script_runs_the_desk_and_reports_its_exit_status(tmp_path):
    root = _fake_project(tmp_path, exit_code=3)
    r = subprocess.run([str(root / "launchd" / "teamrobo-desk")], cwd="/", timeout=60,
                       env={**os.environ, "TEAMROBO_HOLD_SECONDS": "1", "TEAMROBO_EMAIL_AT": "2400"})
    assert (root / "calls.txt").read_text().splitlines() == [f"{root.resolve()} run_daily.py"]
    assert r.returncode == 3                         # a failed desk run shows as one in launchctl


def _desk_after_email_slot(root: Path):
    env = {**os.environ, "TEAMROBO_HOLD_SECONDS": "1", "TEAMROBO_EMAIL_AT": "0000"}     # always "after 14:00"
    subprocess.run([str(root / "launchd" / "teamrobo-desk")], cwd="/", timeout=60, env=env)
    return (root / "calls.txt").read_text().splitlines()


def test_a_desk_run_that_finishes_after_the_email_slot_sends_the_day_s_email(tmp_path):
    """Monitor R-13: 2026-09-28's desk ran at 15:31 and no email followed until someone sent it."""
    root = _fake_project(tmp_path)
    r = root.resolve()
    assert _desk_after_email_slot(root) == [f"{r} run_daily.py", f"{r} monitor.py [chained]",
                                            f"{r} report_daily.py [chained]"]
    assert not (root / "state" / ".email-job.lock").exists()                           # lock released


def test_no_follow_up_email_once_today_s_has_gone_out(tmp_path):
    from datetime import date
    root = _fake_project(tmp_path)
    (root / "state" / "reports").mkdir(parents=True)
    (root / "state" / "reports" / f"{date.today().isoformat()}.sent").write_text("emailed x")
    assert _desk_after_email_slot(root) == [f"{root.resolve()} run_daily.py"]


def test_a_second_email_job_steps_aside_while_one_runs_and_a_stale_lock_is_cleared(tmp_path):
    root = _fake_project(tmp_path)
    lock = root / "state" / ".email-job.lock"
    lock.mkdir(parents=True)
    (lock / "pid").write_text(str(os.getpid()))                   # a live process holds it
    subprocess.run([str(root / "launchd" / "teamrobo-email")], cwd="/", check=True, timeout=60)
    assert not (root / "calls.txt").exists()
    (lock / "pid").write_text("999999")                           # its owner was killed
    subprocess.run([str(root / "launchd" / "teamrobo-email")], cwd="/", check=True, timeout=60)
    assert len((root / "calls.txt").read_text().splitlines()) == 2 and not lock.exists()

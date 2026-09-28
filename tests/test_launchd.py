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


def _fake_project(tmp_path: Path, exit_code: int = 0) -> Path:
    """A project root whose python only records how it was called."""
    (tmp_path / "launchd").mkdir()
    for s in ("teamrobo-desk", "teamrobo-email"):
        shutil.copy2(ROOT / "launchd" / s, tmp_path / "launchd" / s)
    py = tmp_path / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text(f'#!/bin/sh\necho "$(pwd) $*" >> "{tmp_path}/calls.txt"\nexit {exit_code}\n')
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
                       env={**os.environ, "TEAMROBO_HOLD_SECONDS": "1"})
    assert (root / "calls.txt").read_text().splitlines() == [f"{root.resolve()} run_daily.py"]
    assert r.returncode == 3                         # a failed desk run shows as one in launchctl

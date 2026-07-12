from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

WEB_UI = Path(__file__).parents[2]
PROCESS_MANAGER = WEB_UI / "scripts" / "web_ui_process.py"


@pytest.mark.parametrize("name", ("start_web_ui.sh", "stop_web_ui.sh"))
def test_web_ui_shell_wrapper_is_executable_and_valid(name: str) -> None:
    wrapper = WEB_UI / "scripts" / name

    assert os.access(wrapper, os.X_OK)
    subprocess.run(["bash", "-n", wrapper], check=True)


def test_web_ui_process_manager_reports_stopped_state(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            PROCESS_MANAGER,
            "status",
            "--runtime-dir",
            tmp_path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout.strip() == "Calc Flow Studio is stopped."


def test_web_ui_stop_is_idempotent(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            PROCESS_MANAGER,
            "stop",
            "--runtime-dir",
            tmp_path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "already stopped" in result.stdout


def test_web_ui_process_manager_launches_workspace_backend() -> None:
    source = PROCESS_MANAGER.read_text(encoding="utf-8")

    assert 'WEB_UI = ROOT / "web-ui"' not in source
    assert '"--package", "calc-flow-studio"' in source
    assert '"--extra", "web"' not in source


@pytest.mark.parametrize("name", ("export_openapi.py", "run_e2e_server.py"))
def test_web_ui_python_script_imports_studio_package(name: str) -> None:
    source = (WEB_UI / "scripts" / name).read_text(encoding="utf-8")

    assert "calc_flow_studio" in source
    assert "calc_flow.web" not in source

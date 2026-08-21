from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

WEB_UI = Path(__file__).parents[2]
PROCESS_MANAGER = WEB_UI / "scripts" / "web_ui_process.py"
PLAYWRIGHT_CONFIG = WEB_UI / "playwright.config.ts"


def _load_process_manager() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "calc_flow_web_ui_process", PROCESS_MANAGER
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROCESS_MANAGER_MODULE = _load_process_manager()


def _wait_for_pid_file(path: Path, timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return int(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            time.sleep(0.05)
    raise AssertionError(f"timed out waiting for child PID at {path}")


@pytest.mark.parametrize("name", ("start_web_ui.sh", "stop_web_ui.sh"))
def test_web_ui_shell_wrapper_is_executable_and_valid(name: str) -> None:
    wrapper = WEB_UI / "scripts" / name

    assert os.access(wrapper, os.X_OK)
    subprocess.run(["bash", "-n", wrapper], check=True)


@pytest.mark.parametrize(
    ("name", "action"),
    (("start_web_ui.ps1", "start"), ("stop_web_ui.ps1", "stop")),
)
def test_web_ui_powershell_wrapper_delegates_to_process_manager(
    name: str, action: str, tmp_path: Path
) -> None:
    if os.name != "nt":
        pytest.skip("Windows PowerShell wrapper")
    wrapper = WEB_UI / "scripts" / name
    assert wrapper.is_file()
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    assert powershell is not None
    capture = tmp_path / "capture.txt"
    fake_uv = tmp_path / "uv.cmd"
    fake_uv.write_text(
        "@echo off\r\n"
        '> "%CALC_FLOW_CAPTURE%" echo cwd=%CD%\r\n'
        '>> "%CALC_FLOW_CAPTURE%" echo args=%*\r\n'
        "exit /b 23\r\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PATH"] = f"{tmp_path}{os.pathsep}{environment['PATH']}"
    environment["CALC_FLOW_CAPTURE"] = str(capture)

    result = subprocess.run(
        [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            wrapper,
            "--timeout",
            "12",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    values = dict(
        line.split("=", maxsplit=1)
        for line in capture.read_text(encoding="utf-8").splitlines()
    )
    assert result.returncode == 23
    assert Path(values["cwd"]) == WEB_UI.parent
    assert "run --no-sync python" in values["args"]
    assert str(PROCESS_MANAGER) in values["args"]
    assert f" {action} " in f" {values['args']} "
    assert "--timeout 12" in values["args"]


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


@pytest.mark.skipif(os.name != "nt", reason="Windows process behavior")
def test_windows_process_identity_is_stable_and_non_destructive() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        first = PROCESS_MANAGER_MODULE._process_identity(process.pid)
        second = PROCESS_MANAGER_MODULE._process_identity(process.pid)

        assert first[0] == "R"
        assert first[1]
        assert second == first
        assert process.poll() is None
    finally:
        process.kill()
        process.wait(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Windows process behavior")
def test_windows_stop_terminates_the_service_process_tree(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    source = (
        "import subprocess, sys, time; from pathlib import Path; "
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(30)']); "
        "Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8'); "
        "time.sleep(30)"
    )
    root = subprocess.Popen(
        [sys.executable, "-c", source, str(child_pid_path)],
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        ),
    )
    try:
        child_pid = _wait_for_pid_file(child_pid_path)
        _, token = PROCESS_MANAGER_MODULE._process_identity(root.pid)
        service = PROCESS_MANAGER_MODULE.ServiceRecord(
            name="test",
            pid=root.pid,
            process_group=root.pid,
            start_token=token,
            command=(sys.executable,),
            url="http://127.0.0.1:1",
            log_path=tmp_path / "test.log",
        )

        PROCESS_MANAGER_MODULE._stop_services({"test": service}, timeout=1.0)

        root.wait(timeout=5)
        assert not PROCESS_MANAGER_MODULE._pid_exists(child_pid)
    finally:
        subprocess.run(
            ["taskkill", "/PID", str(root.pid), "/T", "/F"],
            check=False,
            capture_output=True,
        )


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
    assert re.search(
        r'command=\[\s*uv,\s*"run",\s*"--no-sync",\s*"--package",\s*'
        r'"calc-flow-studio",\s*"calc-flow-web",?\s*\]',
        source,
    )
    assert '"--extra", "web"' not in source
    assert "/api/v3/catalog" in source
    assert "/api/v2/catalog" not in source


def test_web_ui_documentation_includes_native_windows_commands() -> None:
    for path in (WEB_UI.parent / "README.md", WEB_UI / "README.md"):
        source = path.read_text(encoding="utf-8")
        assert r".\web-ui\scripts\start_web_ui.ps1" in source
        assert r".\web-ui\scripts\stop_web_ui.ps1" in source


@pytest.mark.parametrize("name", ("export_openapi.py", "run_e2e_server.py"))
def test_web_ui_python_script_imports_studio_package(name: str) -> None:
    source = (WEB_UI / "scripts" / name).read_text(encoding="utf-8")

    assert "calc_flow_studio" in source
    assert "calc_flow.web" not in source


def test_e2e_server_registers_udf_on_the_v2_runtime() -> None:
    source = (WEB_UI / "scripts" / "run_e2e_server.py").read_text(encoding="utf-8")

    assert "Runtime()" in source
    assert "runtime.register_scalar_udf(" in source
    assert 'provider="python"' in source
    assert "create_app(" in source
    assert "runtime=runtime" in source
    assert "udf_registry=" not in source


def test_playwright_reuses_the_prepared_python_environment() -> None:
    source = PLAYWRIGHT_CONFIG.read_text(encoding="utf-8")

    assert "UV_CACHE_DIR=../target/playwright-uv-cache" in source
    assert "uv run --no-sync --package calc-flow-studio" in source
    assert "UV_CACHE_DIR=/tmp" not in source

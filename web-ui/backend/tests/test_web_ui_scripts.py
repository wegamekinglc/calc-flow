from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
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
    assert "/api/v2/catalog" in source


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

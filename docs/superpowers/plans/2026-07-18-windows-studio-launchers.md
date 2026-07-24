# Windows Studio Launchers Implementation Plan

> **Historical status:** Implemented and merged in PR #15. Unchecked boxes
> preserve the original execution plan; they are not current pending work.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native PowerShell start and stop commands that manage the Calc Flow Studio API and Vite process trees with the same lifecycle guarantees as the Unix launchers.

**Architecture:** Keep `web_ui_process.py` as the shared lifecycle manager. Add small Windows process primitives for Win32 creation-time identities, non-destructive liveness checks, background process groups, and process-tree termination; PowerShell wrappers only resolve `uv`, enter the repository root, delegate to the manager, forward arguments, and propagate its exit code.

**Tech Stack:** Python 3.13 standard library, Win32 APIs through `ctypes`, Windows `taskkill`, PowerShell 7/Windows PowerShell, pytest, Ruff.

## Global Constraints

- Preserve the current Bash wrappers and POSIX process-group behavior.
- Keep state and logs beneath `.calc-flow-web/` or the supplied `--runtime-dir`.
- Use no new runtime dependency for Windows process inspection.
- Start must wait for `http://127.0.0.1:8765/api/v2/catalog` and `http://127.0.0.1:5173`.
- Stop must terminate the descendant trees rooted at the recorded `uv` and `npm` processes.
- Keep start and stop idempotent and preserve explicit nonzero errors.
- Target Python 3.13 or newer and retain `from __future__ import annotations`.
- Every behavior change starts with a focused failing test whose expected failure is observed.

---

## File Structure

- Modify `web-ui/scripts/web_ui_process.py`: add the Windows process primitives while retaining shared lifecycle logic.
- Create `web-ui/scripts/start_web_ui.ps1`: delegate the `start` action through the repository `uv` environment.
- Create `web-ui/scripts/stop_web_ui.ps1`: delegate the `stop` action through the repository `uv` environment.
- Modify `web-ui/backend/tests/test_web_ui_scripts.py`: cover wrapper delegation and Windows process identity/lifecycle behavior.
- Modify `README.md`: document native PowerShell start and stop commands.
- Modify `web-ui/README.md`: document managed Windows usage alongside Bash usage.

### Task 1: Non-destructive Windows process identity

**Files:**
- Modify: `web-ui/backend/tests/test_web_ui_scripts.py`
- Modify: `web-ui/scripts/web_ui_process.py`

**Interfaces:**
- Produces: `_windows_process_start_token(pid: int) -> str | None`.
- Produces: `_process_identity(pid: int) -> tuple[str | None, str | None]` returning `("R", token)` for a live Windows process.
- Produces: `_pid_exists(pid: int) -> bool` that never signals a Windows process.
- Consumes: the existing `ServiceRecord.start_token` PID-reuse guard.

- [ ] **Step 1: Write the failing Windows creation-token test**

Load `web_ui_process.py` as a module in the focused test file, start a sleeping
child with `CREATE_NO_WINDOW`, and assert repeated identity reads return the
same nonempty token without ending the child:

```python
import importlib.util


def _load_process_manager() -> object:
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
```

- [ ] **Step 2: Run the test and record the expected failure**

Run:

```powershell
uv run --project web-ui/backend --extra dev pytest web-ui/backend/tests/test_web_ui_scripts.py::test_windows_process_identity_is_stable_and_non_destructive -q
```

Expected: FAIL because the current `/proc` implementation returns `(None,
None)` on Windows.

- [ ] **Step 3: Implement Win32 creation-time inspection**

In `web_ui_process.py`, conditionally initialize `kernel32` signatures for
`OpenProcess`, `GetExitCodeProcess`, `GetProcessTimes`, and `CloseHandle`.
Implement `_windows_process_start_token()` with
`PROCESS_QUERY_LIMITED_INFORMATION`, return `None` for a missing/exited PID,
and serialize the `FILETIME` creation value as a decimal token. Update
`_process_identity()` and `_pid_exists()` to use this helper on Windows while
leaving their POSIX branches unchanged:

```python
def _windows_process_start_token(pid: int) -> str | None:
    handle = _KERNEL32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == ERROR_INVALID_PARAMETER:
            return None
        raise ctypes.WinError(error)
    try:
        exit_code = wintypes.DWORD()
        if not _KERNEL32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())
        if exit_code.value != STILL_ACTIVE:
            return None
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not _KERNEL32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        value = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        return str(value)
    finally:
        _KERNEL32.CloseHandle(handle)
```

- [ ] **Step 4: Run the focused test and existing stopped-state tests**

Run:

```powershell
uv run --project web-ui/backend --extra dev pytest web-ui/backend/tests/test_web_ui_scripts.py -q
```

Expected: PASS, with Windows-only tests skipped on non-Windows hosts.

- [ ] **Step 5: Commit the identity primitive**

```powershell
git add web-ui/scripts/web_ui_process.py web-ui/backend/tests/test_web_ui_scripts.py
git commit -m "fix: inspect Windows Studio processes safely"
```

### Task 2: Windows process groups and tree termination

**Files:**
- Modify: `web-ui/backend/tests/test_web_ui_scripts.py`
- Modify: `web-ui/scripts/web_ui_process.py`

**Interfaces:**
- Consumes: `_process_identity(pid)` and `ServiceRecord` from Task 1.
- Produces: `_popen_group_options() -> dict[str, Any]`.
- Produces: `_process_group(pid: int) -> int`.
- Produces: `_terminate_windows_process_tree(service: ServiceRecord) -> None`.
- Updates: `_spawn_service()` and `_signal_service()` to select platform primitives.

- [ ] **Step 1: Write the failing process-tree test**

On Windows, start a root Python process that starts its own sleeping child and
writes the child PID to a temporary file. Construct a `ServiceRecord` from the
root process, call `_stop_services()`, and assert both PIDs are gone. Always use
`taskkill /T /F` in test cleanup so the red run cannot leak children:

```python
import time


def _wait_for_pid_file(path: Path, timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return int(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            time.sleep(0.05)
    raise AssertionError(f"timed out waiting for child PID at {path}")


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
```

- [ ] **Step 2: Run the test and record the expected failure**

Run:

```powershell
uv run --project web-ui/backend --extra dev pytest web-ui/backend/tests/test_web_ui_scripts.py::test_windows_stop_terminates_the_service_process_tree -q
```

Expected: FAIL because `_signal_service()` calls unavailable POSIX
`os.getpgid()`/`os.killpg()` functions on Windows.

- [ ] **Step 3: Implement Windows launch and stop primitives**

Return `CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW` from
`_popen_group_options()` on Windows and `start_new_session=True` elsewhere.
Store the root PID as `process_group` on Windows. Terminate Windows descendants
with `taskkill /PID <pid> /T /F`; accept a nonzero result only when the recorded
service identity is already gone, otherwise raise `WebUiProcessError` with the
captured command output. Route `_spawn_service()` and `_signal_service()`
through these helpers:

```python
def _terminate_windows_process_tree(service: ServiceRecord) -> None:
    result = subprocess.run(
        ["taskkill", "/PID", str(service.pid), "/T", "/F"],
        check=False,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode == 0 or not _service_is_running(service):
        return
    detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
    raise WebUiProcessError(
        f"could not stop {service.name} (PID {service.pid}): {detail}"
    )
```

- [ ] **Step 4: Run the focused process tests**

Run:

```powershell
uv run --project web-ui/backend --extra dev pytest web-ui/backend/tests/test_web_ui_scripts.py -q
```

Expected: PASS with no orphan test child processes.

- [ ] **Step 5: Commit Windows lifecycle support**

```powershell
git add web-ui/scripts/web_ui_process.py web-ui/backend/tests/test_web_ui_scripts.py
git commit -m "feat: manage Studio process trees on Windows"
```

### Task 3: PowerShell start and stop wrappers

**Files:**
- Create: `web-ui/scripts/start_web_ui.ps1`
- Create: `web-ui/scripts/stop_web_ui.ps1`
- Modify: `web-ui/backend/tests/test_web_ui_scripts.py`

**Interfaces:**
- Consumes: `web_ui_process.py start|stop [--runtime-dir PATH] [--timeout SECONDS]`.
- Produces: `start_web_ui.ps1 [manager arguments]`.
- Produces: `stop_web_ui.ps1 [manager arguments]`.

- [ ] **Step 1: Write failing wrapper-delegation tests**

Create a temporary `uv.cmd` that records `%CD%` and `%*`, put it first on
`PATH`, invoke each missing wrapper with `--timeout 12`, and assert the wrapper
uses the repository root, selects the expected action, forwards the option,
and returns the fake command's exit code:

```python
import shutil


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
```

- [ ] **Step 2: Run the wrapper tests and record the expected failure**

Run:

```powershell
uv run --project web-ui/backend --extra dev pytest web-ui/backend/tests/test_web_ui_scripts.py -k powershell_wrapper -q
```

Expected: FAIL because `start_web_ui.ps1` and `stop_web_ui.ps1` do not exist.

- [ ] **Step 3: Add minimal PowerShell wrappers**

Each script resolves `uv`, resolves the repository root relative to
`$PSScriptRoot`, temporarily enters the root, calls Python through the synced
environment, and preserves the manager exit code across `Pop-Location`:

```powershell
$ErrorActionPreference = "Stop"

$uv = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
if ($null -eq $uv) {
    Write-Error "'uv' is required; install it and ensure it is on PATH"
    exit 1
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$manager = Join-Path $PSScriptRoot "web_ui_process.py"
$exitCode = 1
Push-Location $repositoryRoot
try {
    & $uv.Source run --no-sync python $manager ACTION @args
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $exitCode
```

Replace `ACTION` with `start` or `stop` in the corresponding file.

- [ ] **Step 4: Run wrapper and lifecycle tests**

Run:

```powershell
uv run --project web-ui/backend --extra dev pytest web-ui/backend/tests/test_web_ui_scripts.py -q
```

Expected: PASS, including syntax/delegation because executing the scripts also
exercises PowerShell parsing.

- [ ] **Step 5: Commit the wrappers**

```powershell
git add web-ui/scripts/start_web_ui.ps1 web-ui/scripts/stop_web_ui.ps1 web-ui/backend/tests/test_web_ui_scripts.py
git commit -m "feat: add Windows Studio launchers"
```

### Task 4: Documentation and verification

**Files:**
- Modify: `README.md`
- Modify: `web-ui/README.md`

**Interfaces:**
- Documents: native Windows start and stop commands from the repository root.
- Preserves: Bash commands for macOS, Linux, and WSL.

- [ ] **Step 1: Write the failing documentation assertion**

Add a focused test that reads both README files and requires the two native
PowerShell commands:

```python
def test_web_ui_documentation_includes_native_windows_commands() -> None:
    for path in (WEB_UI.parent / "README.md", WEB_UI / "README.md"):
        source = path.read_text(encoding="utf-8")
        assert ".\\web-ui\\scripts\\start_web_ui.ps1" in source
        assert ".\\web-ui\\scripts\\stop_web_ui.ps1" in source
```

For `web-ui/README.md`, account for commands being shown from the repository
root, matching its existing launcher instructions.

- [ ] **Step 2: Run the documentation test and record the expected failure**

Run:

```powershell
uv run --project web-ui/backend --extra dev pytest web-ui/backend/tests/test_web_ui_scripts.py::test_web_ui_documentation_includes_native_windows_commands -q
```

Expected: FAIL because neither README mentions the PowerShell wrappers.

- [ ] **Step 3: Document platform-specific commands**

In both README files, label the existing `.sh` commands for macOS/Linux/WSL
and add this native PowerShell block:

```powershell
.\web-ui\scripts\start_web_ui.ps1
# Open http://127.0.0.1:5173
.\web-ui\scripts\stop_web_ui.ps1
```

State that managed logs and PID state remain in `.calc-flow-web/`.

- [ ] **Step 4: Run focused verification**

Run:

```powershell
uv run --project web-ui/backend --extra dev pytest web-ui/backend/tests/test_web_ui_scripts.py -q
uv run ruff check web-ui/scripts/web_ui_process.py web-ui/backend/tests/test_web_ui_scripts.py
uv run ruff format --check web-ui/scripts/web_ui_process.py web-ui/backend/tests/test_web_ui_scripts.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 5: Run repository-required verification**

Run the command groups from `AGENTS.md`, followed by:

```powershell
git diff --exit-code -- schemas/project-v2.schema.json web-ui/openapi.json web-ui/src/api/schema.d.ts
git diff --check
```

Expected: all commands exit 0 and generated contracts remain unchanged.

- [ ] **Step 6: Commit documentation**

```powershell
git add README.md web-ui/README.md web-ui/backend/tests/test_web_ui_scripts.py
git commit -m "docs: explain Windows Studio commands"
```

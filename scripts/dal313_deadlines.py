from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from urllib.parse import unquote, urlparse

SOURCE_SHA = "beb0bebccad4fc42cac39e6cb055d60018a55860"
MAIN_SHA = "d5906260f2518ba4544a7d717fced83bd49bd26f"
EXECUTION_REF = "refs/heads/diagnostic/DAL-313-windows-deadlines"
RUST_SOURCE = "crates/calc-flow-connectors/tests/late_output_file.rs"
RUST_SOURCE_SHA256 = "48c665c55f03eef482dd3af835f27369a920a3a22c1f57305de373ffacba277e"
TOOLS = Path(__file__).resolve().parent
CONTROL_FILES = frozenset(
    {
        ".github/workflows/dal313-windows-deadlines.yml",
        ".codex/artifacts/analysis/dal-313-windows-deadlines-plan.md",
        "scripts/dal313_deadlines.py",
        "scripts/dal313_deadline_inputs.json",
        "scripts/dal313_pytest_observer.py",
        "scripts/dal313_child_observer.py",
        "scripts/dal313_rust_observer.rs",
        "scripts/test_dal313_deadlines.py",
    }
)
INPUTS = json.loads((TOOLS / "dal313_deadline_inputs.json").read_text(encoding="utf-8"))


def validate_environment(environment: Mapping[str, str], surface: str) -> None:
    required = {
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": EXECUTION_REF,
        "GITHUB_RUN_ATTEMPT": "1",
        "RUNNER_OS": "Windows",
        "RUNNER_ARCH": "X64",
        "ImageOS": "win25-vs2026",
        "ImageVersion": "20260925.250.1",
        "CARGO_PROFILE_DEV_DEBUG": "0",
    }
    if surface == "python":
        required |= {"JAX_PLATFORMS": "cpu", "PYTEST_XDIST_AUTO_NUM_WORKERS": "2"}
    for key, value in required.items():
        if environment.get(key) != value:
            raise ValueError(
                f"input mismatch: {key}={environment.get(key)!r}; need {value!r}"
            )
    for key in (
        "RUST_TEST_THREADS",
        "CARGO_BUILD_JOBS",
        "RUSTFLAGS",
        "PYTHONTRACEMALLOC",
        "PYTEST_ADDOPTS",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "PYO3_PYTHON",
    ):
        if environment.get(key):
            raise ValueError(f"unexpected concurrency/build/profiling override: {key}")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


class ObservationHealth:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.errors: list[dict[str, str]] = []
        self.lock = threading.Lock()

    def record(self, operation: str, error: Exception) -> None:
        with self.lock:
            if len(self.errors) < 64:
                self.errors.append({"operation": operation, "error": repr(error)})

    def save_json(self, name: str, value: object) -> None:
        try:
            write_json(self.output / name, value)
        except (OSError, ValueError) as error:
            self.record(name, error)

    def finish(self, name: str = "observation-health.json", **outcome: object) -> None:
        record = {"healthy": not self.errors, "errors": self.errors, **outcome}
        try:
            write_json(self.output / name, record)
        except (OSError, ValueError) as error:
            self.record(name, error)
            record["healthy"] = False
        if self.errors:
            with suppress(OSError, ValueError):
                print(
                    "DAL313 observation unhealthy: " + json.dumps(record),
                    file=sys.stderr,
                )


class WindowsProcessTree:
    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_int64),
                ("job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD),
                ("minimum_working_set", ctypes.c_size_t),
                ("maximum_working_set", ctypes.c_size_t),
                ("active_process_limit", wintypes.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("basic", BasicLimits),
                ("io_counters", ctypes.c_uint64 * 6),
                ("memory_limits", ctypes.c_size_t * 4),
            ]

        class ThreadEntry(ctypes.Structure):
            _fields_ = [
                ("size", wintypes.DWORD),
                ("usage", wintypes.DWORD),
                ("thread", wintypes.DWORD),
                ("owner", wintypes.DWORD),
                ("priority", wintypes.LONG),
                ("delta", wintypes.LONG),
                ("flags", wintypes.DWORD),
            ]

        self.ctypes, self.ThreadEntry = ctypes, ThreadEntry
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        handle, dword, boolean = wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], handle),
            "SetInformationJobObject": (
                [handle, ctypes.c_int, ctypes.c_void_p, dword],
                boolean,
            ),
            "AssignProcessToJobObject": ([handle, handle], boolean),
            "OpenProcess": ([dword, boolean, dword], handle),
            "OpenThread": ([dword, boolean, dword], handle),
            "ResumeThread": ([handle], dword),
            "CreateToolhelp32Snapshot": ([dword, dword], handle),
            "Thread32First": ([handle, ctypes.POINTER(ThreadEntry)], boolean),
            "Thread32Next": ([handle, ctypes.POINTER(ThreadEntry)], boolean),
            "CloseHandle": ([handle], boolean),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = arguments, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE only.
        if not self.api.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def attach_and_resume(self, pid: int) -> None:
        ctypes, api = self.ctypes, self.api
        process = api.OpenProcess(0x101, False, pid)  # SET_QUOTA | TERMINATE.
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not api.AssignProcessToJobObject(self.handle, process):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            api.CloseHandle(process)
        snapshot = api.CreateToolhelp32Snapshot(4, 0)  # TH32CS_SNAPTHREAD.
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = self.ThreadEntry()
            entry.size = ctypes.sizeof(entry)
            found = api.Thread32First(snapshot, ctypes.byref(entry))
            while found:
                if entry.owner == pid:
                    thread = api.OpenThread(2, False, entry.thread)
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        if api.ResumeThread(thread) != 1:
                            raise OSError(
                                "owned suspended primary thread did not resume once"
                            )
                        return
                    finally:
                        api.CloseHandle(thread)
                found = api.Thread32Next(snapshot, ctypes.byref(entry))
            raise OSError("owned suspended primary thread not found")
        finally:
            api.CloseHandle(snapshot)

    def close(self) -> None:
        if self.handle:
            handle, self.handle = self.handle, None
            if not self.api.CloseHandle(handle):
                raise self.ctypes.WinError(self.ctypes.get_last_error())


def git(source: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *arguments], text=True
    ).strip()


def verify_source(source: Path) -> dict[str, object]:
    if git(source, "rev-parse", "HEAD") != SOURCE_SHA:
        raise ValueError("source checkout must be the pinned final UI SHA")
    if git(source, "diff", "--name-only") or git(
        source, "diff", "--cached", "--name-only"
    ):
        raise ValueError("source checkout must have no tracked changes before overlay")
    for name, expected in INPUTS["source_hashes"].items():
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"source hash mismatch: {name}")
    for name in INPUTS["comparison_paths"]:
        if git(source, "diff", "--name-only", MAIN_SHA, SOURCE_SHA, "--", name):
            raise ValueError(f"main/UI input differs: {name}")
    return {
        "source_sha": SOURCE_SHA,
        "main_sha": MAIN_SHA,
        "source_tree": git(source, "rev-parse", "HEAD^{tree}"),
        "crates_tree": git(source, "rev-parse", "HEAD:crates"),
        "python_tree": git(source, "rev-parse", "HEAD:python"),
        "source_hashes": INPUTS["source_hashes"],
    }


def verify_runner(source: Path, surface: str, output: Path) -> None:
    snapshot = {
        "environment": {
            key: os.environ.get(key)
            for key in (
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_SHA",
                "GITHUB_REF",
                "GITHUB_EVENT_NAME",
                "RUNNER_OS",
                "RUNNER_ARCH",
                "RUNNER_NAME",
                "ImageOS",
                "ImageVersion",
                "CARGO_PROFILE_DEV_DEBUG",
                "PYTEST_XDIST_AUTO_NUM_WORKERS",
                "JAX_PLATFORMS",
                "RUST_TEST_THREADS",
            )
        },
        "python": sys.version,
        "executable": sys.executable,
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "surface": surface,
        "original_run": INPUTS["original_run"],
    }
    write_json(output / "runner.json", snapshot)
    validate_environment(os.environ, surface)
    if sys.platform != "win32" or sys.version_info[:3] != (3, 13, 15):
        raise ValueError("need native Windows CPython 3.13.15")
    if (
        "rustc 1.88.0" not in snapshot["rustc"]
        or "host: x86_64-pc-windows-msvc" not in snapshot["rustc"]
    ):
        raise ValueError("need Rust 1.88.0 x86_64-pc-windows-msvc")
    volume = subprocess.check_output(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"(Get-Volume -DriveLetter '{source.drive[0]}').FileSystemType",
        ],
        text=True,
    ).strip()
    if volume != "NTFS":
        raise ValueError(f"source filesystem is {volume!r}; need NTFS")
    control = TOOLS.parent
    if git(control, "rev-parse", "HEAD") != os.environ.get("GITHUB_SHA"):
        raise ValueError("control checkout does not match workflow SHA")
    changed = frozenset(
        git(control, "diff", "--name-only", SOURCE_SHA, "HEAD").splitlines()
    )
    if not changed.issubset(CONTROL_FILES):
        raise ValueError("control branch contains changes outside diagnostic scope")
    subprocess.run(
        ["git", "-C", str(control), "merge-base", "--is-ancestor", SOURCE_SHA, "HEAD"],
        check=True,
    )
    write_json(output / "source.json", verify_source(source))


def verify_dependencies(source: Path, surface: str, output: Path) -> None:
    expected = (
        INPUTS["python_versions"]
        if surface == "python"
        else {"numpy": "2.5.3", "pyarrow": "24.0.0"}
    )
    versions = {name: importlib.metadata.version(name) for name in expected}
    write_json(output / "dependencies.json", versions)
    if versions != expected:
        raise ValueError(
            "resolved versions differ from original attempt; see dependencies.json"
        )
    if surface != "python":
        return
    installed = {
        re.sub(
            r"[-_.]+", "-", distribution.metadata["Name"]
        ).lower(): distribution.version
        for distribution in importlib.metadata.distributions()
    }
    write_json(output / "installed-distributions.json", installed)
    if installed != expected:
        raise ValueError("uv environment has unexpected packages or versions")
    uv_version = subprocess.check_output(["uv", "--version"], text=True)
    if not uv_version.startswith("uv 0.12.21 "):
        raise ValueError("need original uv 0.12.21")
    if Path(sys.prefix).resolve() != (source / ".venv").resolve():
        raise ValueError("Python tests must use this checkout's uv venv")
    direct = json.loads(
        importlib.metadata.distribution("calc-flow-python").read_text("direct_url.json")
        or "{}"
    )
    url = urlparse(direct.get("url", ""))
    origin = unquote(url.path).lstrip("/")
    if url.scheme != "file" or Path(origin).resolve() != source.resolve():
        raise ValueError("native package must come from uv sync in the pinned checkout")
    package = importlib.machinery.PathFinder.find_spec("calc_flow")
    spec = (
        importlib.machinery.PathFinder.find_spec(
            "calc_flow._native", package.submodule_search_locations
        )
        if package
        else None
    )
    if spec is None or spec.origin is None:
        raise ValueError("missing installed native extension")
    native = Path(spec.origin).resolve()
    if not native.is_relative_to(source.resolve()) or native.suffix != ".pyd":
        raise ValueError("native extension comes from another environment")
    write_json(
        output / "native-origin.json",
        {
            "direct_url": direct,
            "native_path": str(native),
            "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
            "build_command": ["uv", "sync", "--extra", "dev"],
            "original_binary_hash": None,
            "gap": (
                "original installed wheel/pyd bytes were not uploaded; "
                "build-source equality only"
            ),
        },
    )


def capture_command(
    command: Sequence[str], output: Path, source: Path, environment: Mapping[str, str]
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    started = time.time_ns()
    health = ObservationHealth(output)
    health.save_json(
        "invocation.json",
        {"command": list(command), "cwd": str(source), "started_ns": started},
    )
    read_failed = threading.Event()

    def copy_stream(stream: object, path: Path, console: object) -> None:
        log = None
        try:
            try:
                log = path.open("wb")
            except (OSError, ValueError) as error:
                health.record(path.name, error)
            while chunk := stream.read1(65536):
                if log is not None:
                    try:
                        log.write(chunk)
                        log.flush()
                    except (OSError, ValueError) as error:
                        health.record(path.name, error)
                        try:
                            log.close()
                        except (OSError, ValueError) as close_error:
                            health.record(path.name + ".close", close_error)
                        log = None
                if console is not None:
                    try:
                        console.write(chunk)
                        console.flush()
                    except (OSError, ValueError) as error:
                        health.record(path.name + ".console", error)
                        console = None
        except Exception as error:
            health.record(path.name + ".reader", error)
            read_failed.set()
        finally:
            for handle in (log, stream):
                if handle is not None:
                    try:
                        handle.close()
                    except (OSError, ValueError) as error:
                        health.record(path.name + ".close", error)

    tree = WindowsProcessTree() if sys.platform == "win32" else None
    process = None
    readers = []
    code = None
    interrupted = None
    previous_signal = None
    manage_signal = threading.current_thread() is threading.main_thread()

    def cancel(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    try:
        if manage_signal:
            previous_signal = signal.signal(signal.SIGTERM, cancel)
        # Attach Windows children while suspended, before they can spawn descendants.
        process = subprocess.Popen(
            command,
            cwd=source,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=4 if tree is not None else 0,  # CREATE_SUSPENDED.
            start_new_session=tree is None,
        )
        if tree is not None:
            tree.attach_and_resume(process.pid)
        for stream, name, console in (
            (process.stdout, "stdout.log", sys.stdout.buffer),
            (process.stderr, "stderr.log", sys.stderr.buffer),
        ):
            reader = threading.Thread(
                target=copy_stream,
                args=(stream, output / name, console),
                name="dal313-" + name,
                daemon=True,
            )
            readers.append(reader)
            reader.start()
        while True:
            try:
                code = process.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if read_failed.is_set():
                    raise OSError(
                        "diagnostic pipe read failed; original result unavailable"
                    ) from None
    except BaseException as error:
        interrupted = type(error).__name__
        raise
    finally:
        # This bound covers teardown only; the original command/test deadlines remain.
        deadline = time.monotonic() + 4
        if tree is not None:
            try:
                tree.close()
            except OSError as error:
                health.record("process_tree.close", error)
        elif process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                health.record("process_tree.kill", error)
        if process is not None:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError as error:
                    health.record("process.kill", error)
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as error:
                health.record("process_tree.reap", error)
            for reader in readers:
                reader.join(timeout=max(0, deadline - time.monotonic()))
                if reader.is_alive():
                    health.record(
                        reader.name + ".join", TimeoutError("cleanup bound reached")
                    )
            if not readers:
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
        if manage_signal:
            signal.signal(signal.SIGTERM, previous_signal)
        record = {
            "command": list(command),
            "exit_code": code,
            "elapsed_ns": time.time_ns() - started,
            "interrupted": interrupted,
            "reader_threads_alive": [
                reader.name for reader in readers if reader.is_alive()
            ],
        }
        health.save_json("exit.json", record)
        health.finish(**record)
    return code


def rust_assertions(text: str) -> list[str]:
    tokens = list(
        re.finditer(r'"(?:\\.|[^"\\])*"|//[^\n]*|\bassert(?:_eq|_ne)?!|[()]', text)
    )
    expressions = []
    for index, token in enumerate(tokens):
        if not token.group().startswith("assert"):
            continue
        depth = 0
        for following in tokens[index + 1 :]:
            if following.group() == "(":
                depth += 1
            elif following.group() == ")":
                depth -= 1
                if depth == 0:
                    expressions.append(text[token.start() : following.end()])
                    break
    return expressions


def instrument_rust(original: str) -> tuple[str, dict[str, object]]:
    if hashlib.sha256(original.encode()).hexdigest() != RUST_SOURCE_SHA256:
        raise ValueError(
            "unexpected Rust fixture input; refusing heuristic instrumentation"
        )
    text = original.replace(
        "struct Probe {\n",
        "struct Probe {\n    diagnostic: Option<Arc<DiagCase>>,\n",
        1,
    )
    text = text.replace(
        ".push((self.name, event, value));",
        """.push((self.name, event, value));
        if let Some(case) = &self.probe.diagnostic {
            diag_emit("callback_enter", json!({
                "case": case.label(), "sink": self.name,
                "phase": event, "value": value
            }));
        }""",
        1,
    )
    target_start = text.index("#[tokio::test]\nasync fn test_late_files_empty_mixed")
    target_end = text.index("#[tokio::test]\nasync fn test_late_files_later_oversize")
    prefix, cases, suffix = (
        text[:target_start],
        text[target_start:target_end],
        text[target_end:],
    )
    names = (
        "test_late_files_empty_mixed_and_all_late_epochs_recover_and_restart_terminal",
        "test_late_files_each_sink_write_prepare_and_commit_failure_settles_and_recovers",
    )
    for name in names:
        anchor = (
            f"async fn {name}() {{\n"
            "    tokio::time::timeout(Duration::from_secs(30), async {"
        )
        replacement = (
            anchor + f'\n        let case = Arc::new(DiagCase::new("{name}"));'
        )
        if cases.count(anchor) != 1:
            raise ValueError(f"missing unique deadline anchor: {name}")
        cases = cases.replace(anchor, replacement, 1)
    cases = (
        cases.replace(
            "for times in [&[20][..], &[5, 20][..], &[5, 7][..]] {",
            "for times in [&[20][..], &[5, 20][..], &[5, 7][..]] {\n"
            '                case.set_combo(format!("cross_section='
            '{cross_section},times={times:?}"));',
        )
        .replace(
            'for (phase, occurrence) in [("write", 0), ("write", 1), '
            '("prepare", 0), ("commit", 0)] {',
            'for (phase, occurrence) in [("write", 0), ("write", 1), '
            '("prepare", 0), ("commit", 0)] {\n'
            '                case.set_combo(format!("sink={name},phase={phase},'
            'occurrence={occurrence}"));',
        )
        .replace(
            "                    cross_section,\n"
            "                    ..Probe::default()",
            "                    cross_section,\n"
            "                    diagnostic: Some(case.clone()),\n"
            "                    ..Probe::default()",
        )
        .replace(
            "let probe = Arc::new(Probe::default());",
            "let probe = Arc::new(Probe { diagnostic: Some(case.clone()), "
            "..Probe::default() });",
        )
    )
    starts = re.compile(
        r"let (job|first|second|third|recovered) = "
        r"(runner\(.*?\.start\(\))\n\s*\.await\n\s*\.unwrap\(\);",
        re.S,
    )
    cases, count = starts.subn(
        lambda match: (
            f'let {match[1]} = diag_start(&case, "{match[1]}.start", '
            f"{match[2]}).await.unwrap();"
        ),
        cases,
    )
    if count != 6:
        raise ValueError("unexpected start topology")
    watched = (
        ("job", "wait"),
        ("first", "trigger_checkpoint"),
        ("first", "cancel"),
        ("second", "wait"),
        ("third", "wait"),
        ("first", "wait"),
        ("recovered", "wait"),
    )
    for variable, method in watched:
        cases = cases.replace(
            f"{variable}.{method}().await",
            f'diag_stage(&case, "{variable}.{method}", '
            f"{variable}.{method}(), Some(&{variable})).await",
        )
    cases = cases.replace(
        "probe.paused.notified().await",
        'diag_stage(&case, "source.paused", '
        "probe.paused.notified(), Some(&first)).await",
    )
    text = prefix + cases + suffix
    pattern = re.compile(r"(async fn (test_\w+)\(\) \{)")
    text, case_count = pattern.subn(
        lambda match: (
            match[1] + f'\n    let _activity = DiagActivity::new("{match[2]}");'
        ),
        text,
    )
    for expression, phase in (
        ("self.sink.open().await", "open"),
        ("self.sink.begin_epoch(epoch).await", "begin"),
        ("self.sink.write(batch).await", "write"),
        ("self.sink.pre_commit(epoch).await", "prepare"),
        ("self.sink.commit(epoch, evidence).await", "commit"),
        ("self.sink.abort(epoch, evidence).await", "abort"),
        ("self.sink.recover(recovery).await", "recover"),
        ("self.sink.close().await", "close"),
    ):
        if text.count(expression) != 1:
            raise ValueError(f"unexpected callback topology: {phase}")
        text = text.replace(
            expression,
            f'''let result = {expression};
        if let Some(case) = &self.probe.diagnostic {{
            diag_emit("callback_result", json!({{
                "case": case.label(), "sink": self.name,
                "phase": "{phase}", "result": format!("{{result:?}}")
            }}));
        }}
        result''',
            1,
        )
    normalized = text
    for variable, method in watched:
        normalized = normalized.replace(
            f'diag_stage(&case, "{variable}.{method}", '
            f"{variable}.{method}(), Some(&{variable})).await",
            f"{variable}.{method}().await",
        )
    original_assertions, observed_assertions = (
        rust_assertions(original),
        rust_assertions(normalized),
    )
    original_timeouts = re.findall(r"Duration::from_secs\(\d+\)", original)
    observed_timeouts = re.findall(r"Duration::from_secs\(\d+\)", text)
    if (
        original_assertions != observed_assertions
        or original_timeouts != observed_timeouts
        or case_count != 7
    ):
        raise ValueError("assertion/deadline/case topology changed")
    observed = text + (TOOLS / "dal313_rust_observer.rs").read_text(encoding="utf-8")
    return observed, {
        "original_sha256": RUST_SOURCE_SHA256,
        "observed_sha256": hashlib.sha256(observed.encode()).hexdigest(),
        "case_count": case_count,
        "original_assertions": original_assertions,
        "observed_assertions": observed_assertions,
        "original_timeouts": original_timeouts,
        "observed_timeouts": observed_timeouts,
        "harness_threads": (
            "unmodified native default; environment and actual overlap recorded"
        ),
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pinned, observation-only DAL-313 Windows diagnostic"
    )
    parser.add_argument("mode", choices=("preflight", "python", "rust"))
    parser.add_argument("--surface", choices=("python", "rust"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(arguments)
    source, output = options.source.resolve(), options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    surface = options.surface if options.mode == "preflight" else options.mode
    if surface is None:
        parser.error("preflight requires --surface")
    try:
        verify_runner(source, surface, output)
        if options.mode == "preflight":
            return 0
        verify_dependencies(source, surface, output)
        environment = dict(os.environ)
        environment["DAL313_DIAGNOSTIC_OUTPUT"] = str(output)
        if surface == "python":
            environment["PYTHONPATH"] = str(TOOLS) + (
                os.pathsep + environment["PYTHONPATH"]
                if environment.get("PYTHONPATH")
                else ""
            )
            return capture_command(
                [
                    "uv",
                    "run",
                    "pytest",
                    "-q",
                    "-n",
                    "auto",
                    "--dist",
                    "load",
                    "-p",
                    "no:benchmark",
                    "-p",
                    "dal313_pytest_observer",
                    "python/tests",
                ],
                output,
                source,
                environment,
            )
        fixture = source / RUST_SOURCE
        original = fixture.read_bytes()
        transformed, manifest = instrument_rust(original.decode())
        write_json(output / "rust-overlay.json", manifest)
        (output / "late_output_file.original.rs").write_bytes(original)
        (output / "late_output_file.observed.rs").write_text(
            transformed, encoding="utf-8"
        )
        fixture.write_text(transformed, encoding="utf-8")
        try:
            return capture_command(
                [
                    sys.executable,
                    "scripts/run_rust_tests.py",
                    "--python-stress-runs",
                    "1",
                    "--lib-skip",
                    "checkpoint_restart_soak_smoke",
                ],
                output,
                source,
                environment,
            )
        finally:
            fixture.write_bytes(original)
            health = ObservationHealth(output)
            health.save_json(
                "restored.json",
                {
                    "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                    "tracked_diff": git(source, "diff", "--name-only"),
                },
            )
            health.finish("restore-health.json")
    except (
        ValueError,
        subprocess.CalledProcessError,
        OSError,
        importlib.metadata.PackageNotFoundError,
    ) as error:
        write_json(
            output / "precondition-failure.json",
            {"error": repr(error), "tests_are_not_evidence_of_original_failure": True},
        )
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import asyncio
import hashlib
import inspect
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import dal313_deadlines, dal313_pytest_observer


def run_synthetic_probe(root: Path, program: str) -> subprocess.CompletedProcess:
    path = root / "probe.py"
    path.write_text(program, encoding="utf-8")
    tree = dal313_deadlines.WindowsProcessTree() if sys.platform == "win32" else None
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, str(path)],
            cwd=dal313_deadlines.TOOLS.parent,
            env={**os.environ, "PYTHONPATH": str(dal313_deadlines.TOOLS.parent)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=4 if tree is not None else 0,
            start_new_session=tree is None,
        )
        if tree is not None:
            tree.attach_and_resume(process.pid)
        out, err = process.communicate(timeout=12)
        return subprocess.CompletedProcess(process.args, process.returncode, out, err)
    finally:
        if tree is not None:
            tree.close()
        elif process is not None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=2)


def process_is_running(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.OpenProcess.restype = wintypes.HANDLE
        api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        api.WaitForSingleObject.restype = wintypes.DWORD
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        api.CloseHandle.restype = wintypes.BOOL
        handle = api.OpenProcess(0x100000, False, pid)
        if not handle:
            if ctypes.get_last_error() != 87:
                raise ctypes.WinError(ctypes.get_last_error())
            return False
        try:
            return api.WaitForSingleObject(handle, 0) == 258
        finally:
            api.CloseHandle(handle)
    path = Path(f"/proc/{pid}/stat")
    if not path.exists():
        return False
    return path.read_text().split(")", 1)[1].split()[0] != "Z"


class DeadlineDiagnosticTests(unittest.TestCase):
    def test_drains_large_output_after_log_and_console_write_failures(self) -> None:
        for failure_at in ("open", "write"):
            with (
                self.subTest(failure_at=failure_at),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                child = root / "child.py"
                child.write_text(
                    "import os,sys\n"
                    "for _ in range(32):\n"
                    " os.write(1,b'x'*65536); os.write(2,b'y'*65536)\n"
                    "sys.exit(7)\n",
                    encoding="utf-8",
                )
                result = run_synthetic_probe(
                    root,
                    f"""
import io, os, sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from scripts import dal313_deadlines as d
root = Path({str(root)!r})
real_open = Path.open
bad_log = mock.Mock()
bad_log.write.side_effect = OSError('synthetic disk write failure')
def open_log(path, *args, **kwargs):
    if path.name == 'stdout.log':
        if {failure_at!r} == 'open': raise OSError('synthetic disk open failure')
        return bad_log
    return real_open(path, *args, **kwargs)
bad_console = mock.Mock()
bad_console.write.side_effect = BrokenPipeError('synthetic closed console')
with (
    mock.patch.object(Path, 'open', open_log),
    mock.patch.object(sys, 'stdout', SimpleNamespace(buffer=bad_console)),
    mock.patch.object(sys, 'stderr', SimpleNamespace(
        buffer=io.BytesIO(), write=lambda s: len(s), flush=lambda: None)),
):
    code = d.capture_command(
        [sys.executable, str(root/'child.py')], root, root, os.environ)
raise SystemExit(code)
""",
                )
                self.assertEqual(result.returncode, 7, result.stderr.decode())
                self.assertEqual((root / "stderr.log").stat().st_size, 32 * 65536)
                record = json.loads((root / "exit.json").read_text())
                self.assertEqual(record["exit_code"], 7)
                self.assertEqual(record["reader_threads_alive"], [])
                health = json.loads((root / "observation-health.json").read_text())
                self.assertFalse(health["healthy"])
                self.assertEqual(
                    {item["operation"] for item in health["errors"]},
                    {"stdout.log", "stdout.log.console"},
                )

    def test_owned_descendants_are_cleaned_after_exit_and_cancellation(self) -> None:
        for cancel in (False, True):
            with (
                self.subTest(cancel=cancel),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                grandchild = root / "grandchild.py"
                grandchild.write_text(
                    "import os,time\nfrom pathlib import Path\n"
                    f"pid_file = Path({str(root / 'grandchild.pid')!r})\n"
                    "pid_file.write_text(str(os.getpid()))\n"
                    "time.sleep(60)\n",
                    encoding="utf-8",
                )
                child = root / "child.py"
                child.write_text(
                    "import os,subprocess,sys,time\nfrom pathlib import Path\n"
                    f"Path({str(root / 'child.pid')!r}).write_text(str(os.getpid()))\n"
                    f"subprocess.Popen([sys.executable,{str(grandchild)!r}])\n"
                    f"pid_file = Path({str(root / 'grandchild.pid')!r})\n"
                    "while not pid_file.exists(): time.sleep(.01)\n"
                    + ("time.sleep(60)\n" if cancel else "sys.exit(7)\n"),
                    encoding="utf-8",
                )
                before = time.monotonic()
                result = run_synthetic_probe(
                    root,
                    f"""
import os, subprocess, sys
from pathlib import Path
from unittest import mock
from scripts import dal313_deadlines as d
root = Path({str(root)!r})
original = subprocess.Popen.wait
interruption = KeyboardInterrupt('synthetic owner cancellation')
def wait(process, *args, **kwargs):
    if ({cancel!r} and (root/'grandchild.pid').exists()
            and not hasattr(process, '_guard_cancelled')):
        process._guard_cancelled = True
        raise interruption
    return original(process, *args, **kwargs)
try:
    with mock.patch.object(subprocess.Popen, 'wait', wait):
        code = d.capture_command(
            [sys.executable, str(root/'child.py')], root, root, os.environ)
except KeyboardInterrupt as error:
    assert error is interruption
    raise SystemExit(23)
raise SystemExit(code)
""",
                )
                self.assertEqual(
                    result.returncode, 23 if cancel else 7, result.stderr.decode()
                )
                self.assertLess(time.monotonic() - before, 8)
                record = json.loads((root / "exit.json").read_text())
                self.assertEqual(record["reader_threads_alive"], [])
                self.assertEqual(
                    record["interrupted"], "KeyboardInterrupt" if cancel else None
                )
                self.assertEqual(record["exit_code"], None if cancel else 7)
                for name in ("child.pid", "grandchild.pid"):
                    self.assertFalse(
                        process_is_running(int((root / name).read_text())), name
                    )

    def test_exit_artifact_failure_does_not_replace_command_result(self) -> None:
        real_write = dal313_deadlines.write_json

        def save(path: Path, value: object) -> None:
            if path.name == "exit.json":
                raise OSError("synthetic artifact failure")
            real_write(path, value)

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(dal313_deadlines, "write_json", side_effect=save),
        ):
            root = Path(directory)
            command = [sys.executable, "-c", "raise SystemExit(7)"]
            self.assertEqual(
                dal313_deadlines.capture_command(command, root, root, os.environ), 7
            )
            health = json.loads((root / "observation-health.json").read_text())
            self.assertFalse(health["healthy"])
            self.assertEqual(health["command"], command)
            self.assertEqual(health["exit_code"], 7)

    def test_child_artifact_failure_retains_original_timeout_and_result(self) -> None:
        command = [
            sys.executable,
            "-O",
            "-c",
            "original-code",
            "examples/01_datafusion_pipeline.py",
            "quantity",
        ]
        for result in (
            subprocess.TimeoutExpired(command, 60, output=b"partial", stderr=b"cause"),
            subprocess.CompletedProcess(command, 7, "partial", "cause"),
        ):
            with (
                self.subTest(result=type(result).__name__),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                observer = dal313_pytest_observer.Observer(
                    "synthetic-wrapper-guard", root
                )
                run = (
                    mock.Mock(side_effect=result)
                    if isinstance(result, Exception)
                    else mock.Mock(return_value=result)
                )
                with (
                    mock.patch.object(subprocess, "run", run),
                    mock.patch.object(
                        observer,
                        "save_output",
                        side_effect=OSError("synthetic artifact failure"),
                    ),
                    mock.patch.object(
                        dal313_pytest_observer.faulthandler,
                        "cancel_dump_traceback_later",
                        wraps=dal313_pytest_observer.faulthandler.cancel_dump_traceback_later,
                    ) as timer_cancel,
                    mock.patch.object(
                        dal313_pytest_observer.faulthandler,
                        "dump_traceback_later",
                        wraps=dal313_pytest_observer.faulthandler.dump_traceback_later,
                    ) as timer_start,
                    dal313_pytest_observer.pytest.MonkeyPatch.context() as patch,
                ):
                    observer.install_child(patch)
                    if isinstance(result, subprocess.TimeoutExpired):
                        with self.assertRaises(subprocess.TimeoutExpired) as raised:
                            subprocess.run(
                                command,
                                capture_output=True,
                                text=True,
                                timeout=60,
                                check=False,
                            )
                        self.assertIs(raised.exception, result)
                        self.assertEqual(result.cmd, command)
                        self.assertEqual(result.output, b"partial")
                        self.assertEqual(result.stderr, b"cause")
                        self.assertEqual(result.timeout, 60)
                    else:
                        self.assertIs(
                            subprocess.run(
                                command,
                                capture_output=True,
                                text=True,
                                timeout=60,
                                check=False,
                            ),
                            result,
                        )
                    timer_cancel.assert_called_once_with()
                    self.assertTrue(timer_start.call_args.kwargs["file"].closed)
                observer.save()
                health = json.loads((root / "observation-health.json").read_text())
                self.assertFalse(health["healthy"])
                self.assertEqual(health["errors"][0]["operation"], "child-output")
                self.assertEqual(run.call_args.kwargs["timeout"], 60)
                with (root / "example-parent-stacks.log").open("a"):
                    pass

    def test_observer_save_failure_does_not_raise_and_reports_health(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observer = dal313_pytest_observer.Observer("synthetic-wrapper-guard", root)
            original = Path.write_text

            def write(path: Path, *args: object, **kwargs: object) -> int:
                if path.name == "python-events.json":
                    raise OSError("synthetic events artifact failure")
                return original(path, *args, **kwargs)

            with mock.patch.object(Path, "write_text", write):
                observer.save()
            health = json.loads((root / "observation-health.json").read_text())
            self.assertFalse(health["healthy"])
            self.assertEqual(health["errors"][0]["operation"], "python-events.json")

    def test_hook_artifact_failure_preserves_original_test_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node = next(
                node
                for node in dal313_pytest_observer.TARGETS
                if "test_example" in node
            )
            observer = dal313_pytest_observer.Observer(node, root)
            failure = RuntimeError("synthetic original test failure")
            fallback = io.StringIO()
            with (
                mock.patch.dict(os.environ, DAL313_DIAGNOSTIC_OUTPUT=str(root)),
                mock.patch.object(
                    dal313_pytest_observer, "Observer", return_value=observer
                ),
                mock.patch.object(
                    Path,
                    "write_text",
                    side_effect=OSError("synthetic all artifact writes fail"),
                ),
                mock.patch.object(sys, "stderr", fallback),
            ):
                hook = dal313_pytest_observer.pytest_runtest_call(
                    SimpleNamespace(nodeid=node, name="synthetic")
                )
                next(hook)
                with self.assertRaises(RuntimeError) as raised:
                    hook.throw(failure)
            self.assertIs(raised.exception, failure)
            self.assertEqual(
                {item["operation"] for item in observer.health_errors},
                {"python-events.json", "observation-health.json"},
            )
            self.assertIn('"healthy": false', fallback.getvalue())

    def test_restore_artifact_failure_preserves_command_exit_and_fixture(self) -> None:
        original = (
            dal313_deadlines.TOOLS.parent / dal313_deadlines.RUST_SOURCE
        ).read_bytes()
        save = dal313_deadlines.write_json

        def write(path: Path, value: object) -> None:
            if path.name == "restored.json":
                raise OSError("synthetic restore evidence failure")
            save(path, value)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / dal313_deadlines.RUST_SOURCE
            fixture.parent.mkdir(parents=True)
            fixture.write_bytes(original)
            with (
                mock.patch.object(dal313_deadlines, "verify_runner"),
                mock.patch.object(dal313_deadlines, "verify_dependencies"),
                mock.patch.object(dal313_deadlines, "git", return_value=""),
                mock.patch.object(dal313_deadlines, "capture_command", return_value=7),
                mock.patch.object(dal313_deadlines, "write_json", side_effect=write),
            ):
                code = dal313_deadlines.main(
                    ["rust", "--source", str(root), "--output", str(root / "evidence")]
                )
            self.assertEqual(code, 7)
            self.assertEqual(fixture.read_bytes(), original)
            self.assertFalse(
                json.loads((root / "evidence/restore-health.json").read_text())[
                    "healthy"
                ]
            )

    def test_rejects_environment_drift_before_test_execution(self) -> None:
        environment = {
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": dal313_deadlines.EXECUTION_REF,
            "GITHUB_RUN_ATTEMPT": "1",
            "RUNNER_OS": "Windows",
            "RUNNER_ARCH": "X64",
            "ImageOS": "win25-vs2026",
            "ImageVersion": "20260925.250.1",
            "CARGO_PROFILE_DEV_DEBUG": "0",
        }
        dal313_deadlines.validate_environment(environment, "rust")
        for key, value in (
            ("GITHUB_RUN_ATTEMPT", "2"),
            ("ImageVersion", "20261001.1"),
            ("GITHUB_EVENT_NAME", "workflow_dispatch"),
            ("GITHUB_REF", "refs/heads/main"),
            ("RUST_TEST_THREADS", "1"),
            ("PYTHONTRACEMALLOC", "16"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                dal313_deadlines.validate_environment(
                    {**environment, key: value}, "rust"
                )

    def test_retains_failing_exit_and_both_output_streams(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [
                sys.executable,
                "-c",
                "import sys; print('partial'); "
                "print('cause', file=sys.stderr); sys.exit(7)",
            ]
            code = dal313_deadlines.capture_command(
                command, root, root, dict(os.environ)
            )
            self.assertEqual(code, 7)
            self.assertIn("partial", (root / "stdout.log").read_text())
            self.assertIn("cause", (root / "stderr.log").read_text())
            record = json.loads((root / "exit.json").read_text())
            self.assertEqual(record["command"], command)
            self.assertEqual(record["exit_code"], 7)

    def test_rust_overlay_preserves_all_cases_assertions_and_deadlines(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "crates/calc-flow-connectors/tests/late_output_file.rs"
        original = path.read_text()
        transformed, manifest = dal313_deadlines.instrument_rust(original)
        self.assertEqual(manifest["case_count"], 7)
        self.assertEqual(
            manifest["original_assertions"], manifest["observed_assertions"]
        )
        self.assertEqual(manifest["original_timeouts"], manifest["observed_timeouts"])
        self.assertEqual(
            manifest["original_sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
        )
        self.assertIn("diag_before_deadline", transformed)
        with self.assertRaises(ValueError):
            dal313_deadlines.instrument_rust(
                original.replace("from_secs(30)", "from_secs(31)")
            )

    def test_close_factory_returns_the_same_coroutine_and_records_creation(
        self,
    ) -> None:
        async def close() -> int:
            return 17

        coroutine = close()
        with tempfile.TemporaryDirectory() as directory:
            observer = dal313_pytest_observer.Observer(
                "synthetic-wrapper-guard", Path(directory)
            )
            factory = observer.callback_factory(
                lambda _: coroutine, "SinkBinding._native_close"
            )
            self.assertIs(factory(object()), coroutine)
            self.assertEqual(inspect.getcoroutinestate(coroutine), inspect.CORO_CREATED)
            self.assertEqual(asyncio.run(coroutine), 17)
            self.assertEqual(inspect.getcoroutinestate(coroutine), inspect.CORO_CLOSED)
            self.assertTrue(observer.events[0]["data"]["creation_stack"])

    def test_rust_fixture_is_restored_after_child_failure(self) -> None:
        original = (
            dal313_deadlines.TOOLS.parent / dal313_deadlines.RUST_SOURCE
        ).read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / dal313_deadlines.RUST_SOURCE
            fixture.parent.mkdir(parents=True)
            fixture.write_bytes(original)
            with (
                mock.patch.object(dal313_deadlines, "verify_runner"),
                mock.patch.object(dal313_deadlines, "verify_dependencies"),
                mock.patch.object(dal313_deadlines, "git", return_value=""),
                mock.patch.object(dal313_deadlines, "capture_command", return_value=7),
            ):
                code = dal313_deadlines.main(
                    ["rust", "--source", str(root), "--output", str(root / "evidence")]
                )
            self.assertEqual(code, 7)
            self.assertEqual(fixture.read_bytes(), original)
            self.assertTrue((root / "evidence/restored.json").is_file())

    def test_child_timeout_retains_original_exception_deadline_and_partial_output(
        self,
    ) -> None:
        command = [
            sys.executable,
            "-O",
            "-c",
            "original-code",
            "examples/01_datafusion_pipeline.py",
            "quantity",
        ]
        failure = subprocess.TimeoutExpired(
            command, 60, output=b"partial-out", stderr=b"partial-err"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observer = dal313_pytest_observer.Observer("synthetic-wrapper-guard", root)
            with (
                mock.patch.object(subprocess, "run", side_effect=failure) as original,
                dal313_pytest_observer.pytest.MonkeyPatch.context() as patch,
            ):
                observer.install_child(patch)
                with self.assertRaises(subprocess.TimeoutExpired) as raised:
                    subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=False,
                    )
            self.assertIs(raised.exception, failure)
            (observed,) = original.call_args.args
            self.assertEqual(observed[:3], command[:3])
            self.assertTrue(observed[3].endswith(command[3]))
            self.assertEqual(observed[4:], command[4:])
            self.assertEqual(original.call_args.kwargs["timeout"], 60)
            self.assertEqual(
                (root / "example-partial-stdout.txt").read_text(), "partial-out"
            )
            self.assertEqual(
                (root / "example-partial-stderr.txt").read_text(), "partial-err"
            )

    def test_observer_ignores_unselected_child_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            observer = dal313_pytest_observer.Observer(
                "synthetic-wrapper-guard", Path(directory)
            )
            command = [sys.executable, "-c", "print(1)"]
            result = subprocess.CompletedProcess(command, 0)
            with (
                mock.patch.object(subprocess, "run", return_value=result) as original,
                dal313_pytest_observer.pytest.MonkeyPatch.context() as patch,
            ):
                observer.install_child(patch)
                self.assertIs(subprocess.run(command, timeout=60), result)
            original.assert_called_once_with(command, timeout=60)

    def test_child_stage_wrapper_preserves_original_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            example = root / "synthetic.py"
            example.write_text(
                "print('partial-child', flush=True)\n"
                "raise RuntimeError('guard-cause')\n"
            )
            environment = {
                **os.environ,
                "PYTHONPATH": str(dal313_deadlines.TOOLS),
                "DAL313_CHILD_OUTPUT": str(root),
            }
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import dal313_child_observer as o; o.start(); "
                    "import runpy; runpy.run_path(" + repr(str(example)) + ")",
                ],
                capture_output=True,
                text=True,
                env=environment,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("partial-child", result.stdout)
            self.assertIn("guard-cause", result.stderr)
            stages = (root / "example-child-stages.log").read_text()
            self.assertIn("child_started", stages)
            self.assertIn("example_run_path_enter", stages)
            self.assertIn("example_run_path_error=RuntimeError", stages)


if __name__ == "__main__":
    unittest.main()

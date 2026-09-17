from __future__ import annotations

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
from unittest import mock

from scripts import run_rust_tests

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts/run_rust_tests.py"
WINDOWS_WORKFLOW = ROOT / ".github/workflows/ci-windows.yml"
LINUX_WORKFLOW = ROOT / ".github/workflows/ci-linux.yml"
SOAK_SMOKE_FILTER = "checkpoint_restart_soak_smoke"


def _write_python_command(directory: Path, name: str, source: str) -> Path:
    if os.name == "nt":
        script = directory / f"{name}.py"
        script.write_text(source, encoding="utf-8")
        command = directory / f"{name}.cmd"
        command.write_text(
            f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n',
            encoding="utf-8",
        )
        return command
    command = directory / name
    command.write_text(source, encoding="utf-8")
    command.chmod(0o755)
    return command


def _process_is_running(pid: int) -> bool:
    if os.name == "posix":
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        except (OSError, UnicodeDecodeError):
            return False
        if len(fields) >= 3 and fields[2] == "Z":
            return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_until_stopped(pid: int) -> bool:
    deadline = time.monotonic() + 5
    while _process_is_running(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    return not _process_is_running(pid)


class RustTestHarnessTests(unittest.TestCase):
    def _fake_cargo(self, directory: Path) -> tuple[Path, Path, Path]:
        cargo_log = directory / "cargo-calls.jsonl"
        test_log = directory / "python-test-calls.jsonl"
        _write_python_command(
            directory,
            "calc-flow-python-test",
            """#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

with Path(os.environ["FAKE_PYTHON_TEST_LOG"]).open("a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")

if pid_file := os.environ.get("FAKE_PYTHON_TEST_PID"):
    Path(pid_file).write_text(str(os.getpid()))
if child_pid_file := os.environ.get("FAKE_PYTHON_TEST_CHILD_PID"):
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    Path(child_pid_file).write_text(str(child.pid))
time.sleep(float(os.environ.get("FAKE_PYTHON_TEST_SLEEP", "0")))
raise SystemExit(int(os.environ.get("FAKE_PYTHON_TEST_EXIT", "0")))
""",
        )
        cargo = _write_python_command(
            directory,
            "fake-cargo",
            """#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

arguments = sys.argv[1:]
with Path(os.environ["FAKE_CARGO_LOG"]).open("a") as log:
    log.write(json.dumps(arguments) + "\\n")

if status := int(os.environ.get("FAKE_CARGO_EXIT", "0")):
    calls = Path(os.environ["FAKE_CARGO_LOG"]).read_text().splitlines()
    if len(calls) == int(os.environ.get("FAKE_CARGO_FAILURE_CALL", "1")):
        raise SystemExit(status)

if arguments[:3] == ["test", "-p", "calc-flow-python"]:
    if "--no-run" not in arguments:
        raise SystemExit(97)
    time.sleep(float(os.environ.get("FAKE_CARGO_COMPILE_SLEEP", "0")))
    print(json.dumps({
        "reason": "compiler-artifact",
        "target": {"name": "calc_flow_python"},
        "profile": {"test": True},
        "executable": os.environ["FAKE_PYTHON_TEST_EXECUTABLE"],
    }))
""",
        )
        return cargo, cargo_log, test_log

    def _harness_environment(
        self,
        cargo: Path,
        log: Path,
        environment: dict[str, str] | None = None,
    ) -> dict[str, str]:
        test_executable_name = (
            "calc-flow-python-test.cmd" if os.name == "nt" else "calc-flow-python-test"
        )
        return {
            **os.environ,
            "FAKE_CARGO_LOG": str(log),
            "FAKE_PYTHON_TEST_EXECUTABLE": str(cargo.parent / test_executable_name),
            "FAKE_PYTHON_TEST_LOG": str(cargo.parent / "python-test-calls.jsonl"),
            **(environment or {}),
        }

    def _run_harness(
        self,
        cargo: Path,
        log: Path,
        *arguments: str,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(HARNESS), "--cargo", str(cargo), *arguments],
            cwd=ROOT,
            env=self._harness_environment(cargo, log, environment),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_isolates_serial_python_lib_test_and_repeats_stress_runs(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            cargo, cargo_log, test_log = self._fake_cargo(Path(raw_directory))

            result = self._run_harness(
                cargo,
                cargo_log,
                "--python-stress-runs",
                "2",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [
                json.loads(line)
                for line in cargo_log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                calls,
                [
                    [
                        "test",
                        "-p",
                        "calc-flow",
                        "--lib",
                        "--bins",
                        "--tests",
                        "--examples",
                        "--all-features",
                    ],
                    [
                        "test",
                        "--locked",
                        "-p",
                        "calc-flow-connectors",
                        "--all-features",
                    ],
                    [
                        "test",
                        "--locked",
                        "-p",
                        "calc-flow",
                        "--bench",
                        "core",
                        "--all-features",
                    ],
                    [
                        "test",
                        "-p",
                        "calc-flow-python",
                        "--lib",
                        "--all-features",
                        "--no-run",
                        "--message-format=json",
                    ],
                ],
            )
            test_calls = [
                json.loads(line)
                for line in test_log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                test_calls,
                [["--test-threads=1"], ["--test-threads=1"]],
            )

    def test_lib_skip_appends_a_skip_filter_to_the_calc_flow_leg_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            cargo, cargo_log, _ = self._fake_cargo(Path(raw_directory))

            result = self._run_harness(
                cargo,
                cargo_log,
                "--lib-skip",
                "checkpoint_restart_soak_smoke",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [
                json.loads(line)
                for line in cargo_log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                calls,
                [
                    [
                        "test",
                        "-p",
                        "calc-flow",
                        "--lib",
                        "--bins",
                        "--tests",
                        "--examples",
                        "--all-features",
                        "--",
                        "--skip",
                        "checkpoint_restart_soak_smoke",
                    ],
                    [
                        "test",
                        "--locked",
                        "-p",
                        "calc-flow-connectors",
                        "--all-features",
                    ],
                    [
                        "test",
                        "--locked",
                        "-p",
                        "calc-flow",
                        "--bench",
                        "core",
                        "--all-features",
                    ],
                    [
                        "test",
                        "-p",
                        "calc-flow-python",
                        "--lib",
                        "--all-features",
                        "--no-run",
                        "--message-format=json",
                    ],
                ],
            )

    def test_lib_skip_is_ignored_by_compile_only_runs(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            cargo, cargo_log, test_log = self._fake_cargo(Path(raw_directory))

            result = self._run_harness(
                cargo,
                cargo_log,
                "--no-run",
                "--lib-skip",
                "checkpoint_restart_soak_smoke",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [
                json.loads(line)
                for line in cargo_log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                calls,
                [
                    [
                        "test",
                        "-p",
                        "calc-flow",
                        "--lib",
                        "--bins",
                        "--tests",
                        "--examples",
                        "--all-features",
                        "--no-run",
                    ],
                    [
                        "test",
                        "--locked",
                        "-p",
                        "calc-flow-connectors",
                        "--all-features",
                        "--no-run",
                    ],
                    [
                        "test",
                        "--locked",
                        "-p",
                        "calc-flow",
                        "--bench",
                        "core",
                        "--all-features",
                        "--no-run",
                    ],
                    [
                        "test",
                        "-p",
                        "calc-flow-python",
                        "--lib",
                        "--all-features",
                        "--no-run",
                        "--message-format=json",
                    ],
                ],
            )
            self.assertFalse(test_log.exists())

    def test_lib_skip_rejects_an_empty_filter(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            cargo, cargo_log, _ = self._fake_cargo(Path(raw_directory))

            result = self._run_harness(cargo, cargo_log, "--lib-skip", "")

            self.assertEqual(result.returncode, 2)
            self.assertIn("argument --lib-skip: must not be empty", result.stderr)

    def test_no_run_compiles_the_same_matrix_without_executing_tests(self) -> None:
        with (
            tempfile.TemporaryDirectory() as run_directory,
            tempfile.TemporaryDirectory() as compile_directory,
        ):
            cargo, cargo_log, _ = self._fake_cargo(Path(run_directory))
            baseline = self._run_harness(cargo, cargo_log)
            self.assertEqual(baseline.returncode, 0, baseline.stderr)
            baseline_calls = [
                json.loads(line)
                for line in cargo_log.read_text(encoding="utf-8").splitlines()
            ]
            cargo, cargo_log, test_log = self._fake_cargo(Path(compile_directory))

            result = self._run_harness(cargo, cargo_log, "--no-run")

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [
                json.loads(line)
                for line in cargo_log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                calls,
                [
                    call if "--no-run" in call else [*call, "--no-run"]
                    for call in baseline_calls
                ],
            )
            self.assertFalse(test_log.exists())

    def test_no_run_propagates_each_compile_failure(self) -> None:
        for failing_call in range(1, 5):
            with (
                self.subTest(failing_call=failing_call),
                tempfile.TemporaryDirectory() as raw_directory,
            ):
                cargo, cargo_log, test_log = self._fake_cargo(Path(raw_directory))

                result = self._run_harness(
                    cargo,
                    cargo_log,
                    "--no-run",
                    environment={
                        "FAKE_CARGO_EXIT": "17",
                        "FAKE_CARGO_FAILURE_CALL": str(failing_call),
                    },
                )

                self.assertEqual(result.returncode, 17, result.stderr)
                self.assertEqual(
                    len(cargo_log.read_text(encoding="utf-8").splitlines()),
                    failing_call,
                )
                self.assertFalse(test_log.exists())

    def test_compile_time_is_excluded_from_python_test_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            cargo, cargo_log, _ = self._fake_cargo(Path(raw_directory))
            started = time.monotonic()

            result = self._run_harness(
                cargo,
                cargo_log,
                "--python-timeout-seconds",
                "0.5",
                environment={
                    "FAKE_CARGO_COMPILE_SLEEP": "0.75",
                    "FAKE_PYTHON_TEST_SLEEP": "0.05",
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreaterEqual(time.monotonic() - started, 0.8)

    def test_times_out_the_already_compiled_python_test_binary(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            cargo, cargo_log, _ = self._fake_cargo(Path(raw_directory))
            started = time.monotonic()

            result = self._run_harness(
                cargo,
                cargo_log,
                "--python-timeout-seconds",
                "0.1",
                environment={"FAKE_PYTHON_TEST_SLEEP": "60"},
            )

            self.assertEqual(result.returncode, 124)
            self.assertLess(time.monotonic() - started, 5)
            self.assertIn("timed out after 0.1 seconds", result.stderr)

    def test_python_test_environment_includes_managed_python_libdir(self) -> None:
        with (
            mock.patch.object(run_rust_tests.os, "name", "posix"),
            mock.patch.object(run_rust_tests.sys, "platform", "linux"),
            mock.patch.dict(
                run_rust_tests.os.environ,
                {"LD_LIBRARY_PATH": "/existing/lib"},
                clear=True,
            ),
            mock.patch.object(
                run_rust_tests.sysconfig,
                "get_config_var",
                return_value="/managed/python/lib",
            ),
        ):
            environment = run_rust_tests._python_test_environment()

        self.assertEqual(
            environment["LD_LIBRARY_PATH"],
            f"/managed/python/lib{os.pathsep}/existing/lib",
        )

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    def test_terminate_ignores_process_lookup_race_before_sigterm(self) -> None:
        process = mock.Mock(pid=123)
        process.wait.return_value = 0

        with mock.patch.object(
            run_rust_tests.os,
            "killpg",
            side_effect=ProcessLookupError,
        ):
            run_rust_tests._terminate(process)

        process.wait.assert_called_once_with(timeout=5)

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    def test_terminate_ignores_process_lookup_race_before_sigkill(self) -> None:
        process = mock.Mock(pid=123)
        process.wait.side_effect = [
            subprocess.TimeoutExpired("test", 5),
            0,
        ]

        with mock.patch.object(
            run_rust_tests.os,
            "killpg",
            side_effect=[None, ProcessLookupError],
        ):
            run_rust_tests._terminate(process)

        self.assertEqual(process.wait.call_count, 2)

    def test_windows_termination_uses_taskkill_for_the_process_tree(self) -> None:
        process = mock.Mock(pid=123)
        process.wait.return_value = 0

        with (
            mock.patch.object(run_rust_tests.os, "name", "nt"),
            mock.patch.object(run_rust_tests.subprocess, "run") as taskkill,
        ):
            run_rust_tests._terminate(process)

        taskkill.assert_called_once_with(
            ["taskkill", "/PID", "123", "/T", "/F"],
            capture_output=True,
            check=False,
        )
        process.wait.assert_called_once()

    def test_windows_processes_start_in_a_new_process_group(self) -> None:
        with (
            mock.patch.object(run_rust_tests.os, "name", "nt"),
            mock.patch.object(
                run_rust_tests.subprocess,
                "CREATE_NEW_PROCESS_GROUP",
                512,
                create=True,
            ),
        ):
            options = run_rust_tests._popen_group_options()

        self.assertEqual(options, {"creationflags": 512})

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    def test_timeout_cleans_up_the_test_binary_process_tree_on_posix(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            cargo, cargo_log, _ = self._fake_cargo(directory)
            parent_pid_file = directory / "python-test.pid"
            child_pid_file = directory / "python-test-child.pid"

            result = self._run_harness(
                cargo,
                cargo_log,
                "--python-timeout-seconds",
                "1",
                environment={
                    "FAKE_PYTHON_TEST_PID": str(parent_pid_file),
                    "FAKE_PYTHON_TEST_CHILD_PID": str(child_pid_file),
                    "FAKE_PYTHON_TEST_SLEEP": "60",
                },
            )

            self.assertEqual(result.returncode, 124, result.stderr)
            self.assertTrue(parent_pid_file.exists())
            self.assertTrue(child_pid_file.exists())
            self.assertTrue(_wait_until_stopped(int(parent_pid_file.read_text())))
            self.assertTrue(_wait_until_stopped(int(child_pid_file.read_text())))

    @unittest.skipUnless(os.name == "nt", "requires Windows process trees")
    def test_timeout_cleans_up_the_test_binary_process_tree_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            cargo, cargo_log, _ = self._fake_cargo(directory)
            parent_pid_file = directory / "python-test.pid"
            child_pid_file = directory / "python-test-child.pid"

            result = self._run_harness(
                cargo,
                cargo_log,
                "--python-timeout-seconds",
                "1",
                environment={
                    "FAKE_PYTHON_TEST_PID": str(parent_pid_file),
                    "FAKE_PYTHON_TEST_CHILD_PID": str(child_pid_file),
                    "FAKE_PYTHON_TEST_SLEEP": "60",
                },
            )

            self.assertEqual(result.returncode, 124, result.stderr)
            self.assertTrue(parent_pid_file.exists())
            self.assertTrue(child_pid_file.exists())
            parent_stopped = _wait_until_stopped(int(parent_pid_file.read_text()))
            child_stopped = _wait_until_stopped(int(child_pid_file.read_text()))
            print(
                "WINDOWS_PROCESS_TREE_EVIDENCE "
                f"exit_code={result.returncode} "
                f"parent_stopped={str(parent_stopped).lower()} "
                f"child_stopped={str(child_stopped).lower()}",
                flush=True,
            )
            self.assertTrue(parent_stopped)
            self.assertTrue(child_stopped)

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    def test_terminating_harness_cleans_up_the_active_test_process_tree(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            cargo, log, _ = self._fake_cargo(directory)
            parent_pid_file = directory / "python-test.pid"
            child_pid_file = directory / "python-test-child.pid"
            environment = self._harness_environment(
                cargo,
                log,
                {
                    "FAKE_PYTHON_TEST_PID": str(parent_pid_file),
                    "FAKE_PYTHON_TEST_CHILD_PID": str(child_pid_file),
                    "FAKE_PYTHON_TEST_SLEEP": "60",
                },
            )
            process = subprocess.Popen(
                [sys.executable, str(HARNESS), "--cargo", str(cargo)],
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            parent_pid: int | None = None
            child_pid: int | None = None
            try:
                deadline = time.monotonic() + 5
                while (
                    not parent_pid_file.exists() or not child_pid_file.exists()
                ) and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(
                    parent_pid_file.exists(), "the fake Python test never started"
                )
                self.assertTrue(
                    child_pid_file.exists(), "the child process never started"
                )
                parent_pid = int(parent_pid_file.read_text())
                child_pid = int(child_pid_file.read_text())

                process.terminate()
                process.wait(timeout=5)

                self.assertEqual(process.returncode, 143)
                self.assertTrue(_wait_until_stopped(parent_pid))
                self.assertTrue(_wait_until_stopped(child_pid))
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                for pid in (parent_pid, child_pid):
                    if pid is not None and _process_is_running(pid):
                        with suppress(ProcessLookupError):
                            os.kill(pid, signal.SIGKILL)
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()


class WindowsSoakSmokeIsolationTests(unittest.TestCase):
    # The soak smoke tests fork generation child processes; on the 4 vCPU
    # Windows runner those forks amplify contention for every time-sensitive
    # guardrail (DAL-255/DAL-261), so the workflow must keep them isolated
    # from the parallel lib suite instead of relaxing the guardrails.
    def _rust_tests_job(self) -> str:
        workflow = WINDOWS_WORKFLOW.read_text(encoding="utf-8")
        return workflow.split("  rust-tests:\n", 1)[1].split("  python-tests:\n", 1)[0]

    def test_windows_main_lib_suite_skips_the_soak_smoke_tests(self) -> None:
        self.assertIn(f"--lib-skip {SOAK_SMOKE_FILTER}", self._rust_tests_job())

    def test_windows_runs_the_soak_smoke_tests_serially_in_a_dedicated_step(
        self,
    ) -> None:
        soak_step = self._rust_tests_job().split(
            "name: Run soak smoke tests serially\n", 1
        )[1]
        for fragment in (
            "cargo test -p calc-flow",
            "--lib",
            "--all-features",
            SOAK_SMOKE_FILTER,
            "--test-threads=1",
        ):
            self.assertIn(fragment, soak_step)

    def test_linux_ci_keeps_the_full_suite_without_soak_smoke_isolation(
        self,
    ) -> None:
        workflow = LINUX_WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("--lib-skip", workflow)


if __name__ == "__main__":
    unittest.main()

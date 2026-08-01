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

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts/run_rust_tests.py"


class RustTestHarnessTests(unittest.TestCase):
    def _fake_cargo(self, directory: Path) -> tuple[Path, Path]:
        log = directory / "cargo-calls.jsonl"
        cargo = directory / "fake-cargo"
        cargo.write_text(
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

if (
    arguments[:3] == ["test", "-p", "calc-flow-python"]
    and "--no-run" not in arguments
):
    if pid_file := os.environ.get("FAKE_PYTHON_TEST_PID"):
        Path(pid_file).write_text(str(os.getpid()))
    time.sleep(float(os.environ.get("FAKE_PYTHON_TEST_SLEEP", "0")))
    raise SystemExit(int(os.environ.get("FAKE_PYTHON_TEST_EXIT", "0")))
""",
            encoding="utf-8",
        )
        cargo.chmod(0o755)
        return cargo, log

    def _run_harness(
        self,
        cargo: Path,
        log: Path,
        *arguments: str,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        child_environment = {
            **os.environ,
            "FAKE_CARGO_LOG": str(log),
            **(environment or {}),
        }
        return subprocess.run(
            [sys.executable, str(HARNESS), "--cargo", str(cargo), *arguments],
            cwd=ROOT,
            env=child_environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_isolates_serial_python_lib_test_and_repeats_stress_runs(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            cargo, log = self._fake_cargo(Path(raw_directory))

            result = self._run_harness(
                cargo,
                log,
                "--python-stress-runs",
                "2",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                calls,
                [
                    [
                        "test",
                        "-p",
                        "calc-flow",
                        "--all-targets",
                        "--all-features",
                    ],
                    [
                        "test",
                        "-p",
                        "calc-flow-python",
                        "--lib",
                        "--all-features",
                        "--no-run",
                    ],
                    [
                        "test",
                        "-p",
                        "calc-flow-python",
                        "--lib",
                        "--all-features",
                        "--",
                        "--test-threads=1",
                    ],
                    [
                        "test",
                        "-p",
                        "calc-flow-python",
                        "--lib",
                        "--all-features",
                        "--",
                        "--test-threads=1",
                    ],
                ],
            )

    def test_times_out_only_the_already_compiled_python_test_run(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            cargo, log = self._fake_cargo(Path(raw_directory))
            started = time.monotonic()

            result = self._run_harness(
                cargo,
                log,
                "--python-timeout-seconds",
                "0.1",
                environment={"FAKE_PYTHON_TEST_SLEEP": "60"},
            )

            self.assertEqual(result.returncode, 124)
            self.assertLess(time.monotonic() - started, 5)
            self.assertIn("timed out after 0.1 seconds", result.stderr)

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    def test_terminating_harness_cleans_up_the_active_cargo_process(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            cargo, log = self._fake_cargo(directory)
            pid_file = directory / "python-test.pid"
            environment = {
                **os.environ,
                "FAKE_CARGO_LOG": str(log),
                "FAKE_PYTHON_TEST_PID": str(pid_file),
                "FAKE_PYTHON_TEST_SLEEP": "60",
            }
            process = subprocess.Popen(
                [sys.executable, str(HARNESS), "--cargo", str(cargo)],
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            child_pid: int | None = None
            try:
                deadline = time.monotonic() + 5
                while not pid_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(pid_file.exists(), "the fake Python test never started")
                child_pid = int(pid_file.read_text())

                process.terminate()
                process.wait(timeout=5)

                self.assertEqual(process.returncode, 143)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if child_pid is not None:
                    with suppress(ProcessLookupError):
                        os.kill(child_pid, signal.SIGKILL)
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()


if __name__ == "__main__":
    unittest.main()

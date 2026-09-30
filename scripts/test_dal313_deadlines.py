from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import dal313_deadlines, dal313_pytest_observer


class DeadlineDiagnosticTests(unittest.TestCase):
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

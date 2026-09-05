from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_suite.process import child_environment, command


class BenchmarkProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_executable_is_resolved_without_a_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "scripts.benchmark_suite.process.asyncio.create_subprocess_exec"
            ) as spawn:
                spawn.return_value.wait.return_value = 0
                await command(["git", "--version"], cwd=root, log=root / "run.log")
            self.assertTrue(Path(spawn.call_args.args[0]).is_absolute())
            self.assertIs(spawn.call_args.kwargs["shell"], False)

    async def test_failed_command_retains_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "run.log"
            with self.assertRaisesRegex(RuntimeError, "exited 2"):
                await command(
                    [
                        sys.executable,
                        "-c",
                        "print('failure evidence'); raise SystemExit(2)",
                    ],
                    cwd=root,
                    log=log,
                )
            self.assertIn("failure evidence", log.read_text(encoding="utf-8"))

    async def test_command_timeout_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(
                    command(
                        [sys.executable, "-c", "import time; time.sleep(60)"],
                        cwd=root,
                        log=root / "run.log",
                        timeout=0.05,
                    ),
                    timeout=5,
                )

    def test_worker_environment_has_explicit_thread_and_cache_boundaries(self):
        environment = child_environment(Path(__file__).resolve().parent / "test-site")
        self.assertEqual(environment["TOKIO_WORKER_THREADS"], "32")
        self.assertEqual(environment["POLARS_MAX_THREADS"], "32")
        self.assertEqual(environment["OPENBLAS_NUM_THREADS"], "1")
        self.assertTrue(environment["npm_config_cache"].endswith("target/npm-cache"))


if __name__ == "__main__":
    unittest.main()

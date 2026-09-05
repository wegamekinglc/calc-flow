from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_suite.process import child_environment, command


class BenchmarkProcessTests(unittest.IsolatedAsyncioTestCase):
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
        environment = child_environment(Path("/tmp/test-site"))
        self.assertEqual(environment["TOKIO_WORKER_THREADS"], "32")
        self.assertEqual(environment["POLARS_MAX_THREADS"], "32")
        self.assertEqual(environment["OPENBLAS_NUM_THREADS"], "1")
        self.assertTrue(environment["npm_config_cache"].endswith("target/npm-cache"))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_suite.legacy import _frontend_run, measure_legacy


class BenchmarkFrontendTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_legacy_blocks_retain_each_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("scripts.benchmark_suite.legacy.validate_sources"),
                patch("scripts.benchmark_suite.legacy._setup", return_value={}),
                patch(
                    "scripts.benchmark_suite.legacy.observe_workers", return_value={}
                ),
                patch(
                    "scripts.benchmark_suite.legacy._frontend_run",
                    side_effect=RuntimeError("missing benchmark dependency"),
                ),
            ):
                report = await measure_legacy(
                    {"id": "frontend", "family": "frontend"}, {}, root, root
                )
            self.assertEqual(report["cases"], [])
            self.assertEqual(report["expected_case_ids"], [])
            self.assertEqual(
                sum(
                    "missing benchmark dependency" in error
                    for error in report["errors"]
                ),
                4,
            )

    async def test_uses_checkout_local_static_runner_and_archives_its_report(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            frontend = source / "web-ui"
            generated = source / "target/benchmark-suite/vitest.json"
            generated.parent.mkdir(parents=True)
            frontend.mkdir()
            generated.write_text('{"files": []}', encoding="utf-8")
            (frontend / "package-lock.json").write_text("{}", encoding="utf-8")
            output = source / "archived"
            output.mkdir()
            with (
                patch("scripts.benchmark_suite.legacy.command") as run,
                patch("scripts.benchmark_suite.legacy.vitest_rows", return_value={}),
            ):
                await _frontend_run(source, output)
            self.assertEqual(run.call_args.kwargs["cwd"], frontend)
            argv = run.call_args.args[0]
            self.assertEqual(len(argv), 2)
            runner = Path(argv[1])
            self.assertTrue(runner.is_relative_to(frontend / "node_modules"))
            self.assertIn('from "vitest/node"', runner.read_text(encoding="utf-8"))
            self.assertEqual(
                (output / "vitest.json").read_bytes(), generated.read_bytes()
            )

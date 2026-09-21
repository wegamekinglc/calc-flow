from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_suite.process import ROOT, command


class NoSitePackagesTests(unittest.IsolatedAsyncioTestCase):
    async def test_diagnostic_checks_run_without_site_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "checks.log"
            try:
                await command(
                    [
                        sys.executable,
                        "-E",
                        "-S",
                        "-m",
                        "unittest",
                        "scripts.test_dal301_groupby",
                        "scripts.test_benchmark_release",
                    ],
                    cwd=ROOT,
                    log=log,
                    timeout=60,
                )
            except RuntimeError as error:
                self.fail(f"{error}\n{log.read_text()}")
            self.assertIn("OK", log.read_text())
            self.assertNotIn("skipped", log.read_text())

    async def test_cli_plan_and_runtime_failure_without_site_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "report 空格"
            argv = [sys.executable, "-E", "-S", "-m", "scripts.dal301_groupby"]
            try:
                await command(
                    [*argv, "plan", "--output", str(root)],
                    cwd=ROOT,
                    log=root / "plan.log",
                    timeout=60,
                )
            except RuntimeError as error:
                self.fail(f"{error}\n{(root / 'plan.log').read_text()}")
            plan = json.loads((root / "plan.json").read_text())
            self.assertEqual([case["rows"] for case in plan["cases"]], [10000, 100000])
            with self.assertRaisesRegex(RuntimeError, "command exited 1"):
                await command(
                    [*argv, "run", "--output", str(root), "--releases", str(root)],
                    cwd=ROOT,
                    log=root / "run.log",
                    timeout=60,
                )
            outcome = json.loads((root / "outcome.json").read_text())
            self.assertEqual(outcome["status"], "failed")
            self.assertIn("No module named 'psutil'", outcome["error"])

    async def test_missing_build_dependency_keeps_failure_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "command exited 1"):
                await command(
                    [
                        sys.executable,
                        "-E",
                        "-S",
                        "-m",
                        "scripts.dal301_groupby",
                        "build",
                        "--side",
                        "A",
                        "--source",
                        str(root),
                        "--output",
                        str(root),
                    ],
                    cwd=ROOT,
                    log=root / "build.log",
                    timeout=60,
                )
            outcome = json.loads((root / "outcome.json").read_text())
            self.assertEqual(outcome["status"], "failed")
            self.assertIn("No module named 'numpy'", outcome["error"])


class ScalarValidationTests(unittest.TestCase):
    def test_finite_positive_samples_and_correctness_remain_required(self):
        from scripts.benchmark_suite.measure import validate_sample

        for value in (1, 1e-9, 1e308):
            sample = {"seconds": value, "correctness": {"passed": True}}
            self.assertEqual(validate_sample(sample), value)
        for value in (0, -1, float("inf"), float("nan"), True, "1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_sample({"seconds": value, "correctness": {"passed": True}})
        with self.assertRaises(ValueError):
            validate_sample({"seconds": 1.0, "correctness": {"passed": False}})


if __name__ == "__main__":
    unittest.main()

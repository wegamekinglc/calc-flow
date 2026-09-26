from __future__ import annotations

import re
import tomllib
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkWorkflowTests(unittest.TestCase):
    def test_dependency_lock_excludes_the_current_workspace_distribution(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]["name"]
        suite = (ROOT / ".github/workflows/benchmark-suite.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn(f"--no-emit-package {project} \\", suite)
        lock = (ROOT / "benchmarks/requirements.lock").read_text(encoding="utf-8")
        self.assertIn(f"--no-emit-package {project} ", lock)
        self.assertFalse(any(line.startswith("-e ") for line in lock.splitlines()))

    def test_finance_python_dependencies_use_a_python_39_hash_lock(self):
        suite = (ROOT / ".github/workflows/benchmark-suite.yml").read_text(
            encoding="utf-8"
        )
        finance_install = suite.split(
            "- name: Install pinned Finance-Python in Python 3.9\n", 1
        )[1].split("      - uses: actions/download-artifact@", 1)[0]
        self.assertIn(
            "uv pip sync --python target/finance-python-venv/bin/python",
            finance_install,
        )
        self.assertIn(
            "--require-hashes benchmarks/finance-python-requirements.lock",
            finance_install,
        )
        self.assertNotIn("uv pip install", finance_install.split("git clone", 1)[0])
        lock = (ROOT / "benchmarks/finance-python-requirements.lock").read_text(
            encoding="utf-8"
        )
        self.assertIn("--hash=sha256:", lock)

    def test_paired_outputs_anchor_at_workspace_root(self):
        workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(
            encoding="utf-8"
        )
        paired = workflow.split("  sql-datafusion-paired:\n", 1)[1].split(
            "  sql-datafusion-matrix:\n", 1
        )[0]
        # `cargo bench -p calc-flow` runs the harness from the package root, so
        # a relative --output escapes the workspace-root report directory.
        self.assertNotIn('--output "benchmark-results/', paired)
        self.assertIn(
            '--output "${{ github.workspace }}/benchmark-results/sql-datafusion/',
            paired,
        )

    def test_paired_sql_datafusion_reports_upload_even_after_failure(self):
        workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(
            encoding="utf-8"
        )
        paired = workflow.split("  sql-datafusion-paired:\n", 1)[1].split(
            "  sql-datafusion-matrix:\n", 1
        )[0]
        upload = paired.split("- name: Upload SQL/DataFusion reports\n", 1)[1]
        # P1 gate failures must still leave the measured JSON evidence behind.
        self.assertIn("if: always()", upload)

    def test_matrix_sql_datafusion_reports_upload_even_after_failure(self):
        workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(
            encoding="utf-8"
        )
        matrix = workflow.split("  sql-datafusion-matrix:\n", 1)[1].split(
            "  dal301-profile-build:\n", 1
        )[0]
        upload = matrix.split("- name: Upload SQL/DataFusion matrix reports\n", 1)[1]
        # Stability-gate failures must still leave the measured screening and
        # candidate JSON behind for tolerance recalibration.
        self.assertIn("if: always()", upload)

    def test_sql_datafusion_paired_runs_use_two_warmups(self):
        workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(
            encoding="utf-8"
        )
        paired = workflow.split("  sql-datafusion-paired:\n", 1)[1].split(
            "  sql-datafusion-matrix:\n", 1
        )[0]
        self.assertIn("--warmups 2", paired)

    def test_complete_suite_runs_twice_daily_outside_regular_ci(self):
        linux = (ROOT / ".github/workflows/ci-linux.yml").read_text(encoding="utf-8")
        self.assertNotIn("benchmark-smoke:", linux)
        self.assertNotIn("BENCHMARK_RESULT", linux)
        self.assertNotIn("uses: ./.github/workflows/benchmark-suite.yml", linux)
        self.assertNotIn("cargo bench", linux)

        benchmarks = (ROOT / ".github/workflows/benchmarks.yml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("uses: ./.github/workflows/benchmark-suite.yml", benchmarks)

        suite = (ROOT / ".github/workflows/benchmark-suite.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn('cron: "0 22 * * *"', suite)
        self.assertIn('cron: "0 10 * * *"', suite)
        self.assertIn("workflow_call:", suite)
        self.assertIn("scripts.benchmark_suite catalog", suite)
        self.assertIn("fromJSON(needs.prepare.outputs.matrix)", suite)
        self.assertIn("fail-fast: false", suite)

    def test_regular_ci_does_not_build_or_run_benchmark_targets(self):
        for name in ("ci-linux.yml", "ci-windows.yml"):
            workflow = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
            with self.subTest(workflow=name):
                self.assertNotIn("benchmarks/test_warm_stream.py", workflow)
                self.assertNotIn("Verify streaming performance controllers", workflow)
                self.assertNotIn("--all-targets", workflow)

        rust_harness = (ROOT / "scripts/run_rust_tests.py").read_text(encoding="utf-8")
        self.assertNotIn('"--bench"', rust_harness)
        suite = (ROOT / ".github/workflows/benchmark-suite.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("matrix.id == 'warm-10'", suite)
        self.assertIn("benchmarks/test_warm_stream.py -q", suite)
        for module in (
            "scripts.test_profile_warm_stream",
            "scripts.test_performance_plan",
            "scripts.test_entity_parallel_inventory",
        ):
            self.assertIn(module, suite)

    def test_script_tests_run_in_the_matching_workflow(self):
        linux = (ROOT / ".github/workflows/ci-linux.yml").read_text(encoding="utf-8")
        suite = (ROOT / ".github/workflows/benchmark-suite.yml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("unittest discover", linux)
        self.assertNotIn("scripts.test_benchmark_", linux)
        for module in (
            "scripts.test_run_rust_tests",
            "scripts.test_verify_python_release",
            "scripts.test_verify_security_gates",
        ):
            self.assertIn(module, linux)

        expected = {
            f"scripts.{path.stem}" for path in (ROOT / "scripts").glob("test_*.py")
        }
        counts = Counter(re.findall(r"scripts\.test_[a-z0-9_]+", linux + suite))
        self.assertEqual(counts, dict.fromkeys(expected, 1))

    def test_complete_suite_is_the_only_scheduled_benchmark_workflow(self):
        diagnostics = (ROOT / ".github/workflows/benchmarks.yml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("  schedule:\n", diagnostics)
        suite = (ROOT / ".github/workflows/benchmark-suite.yml").read_text(
            encoding="utf-8"
        )
        schedule = suite.split("  schedule:\n", 1)[1].split("  workflow_call:\n", 1)[0]
        self.assertEqual(
            [line.strip() for line in schedule.splitlines() if "cron:" in line],
            ['- cron: "0 22 * * *"', '- cron: "0 10 * * *"'],
        )

    def test_final_tables_and_artifacts_are_emitted_on_failure(self):
        suite = (ROOT / ".github/workflows/benchmark-suite.yml").read_text(
            encoding="utf-8"
        )
        summary = suite.split("  summary:\n", 1)[1]
        self.assertIn("if: always()", summary)
        self.assertIn("needs: [prepare, build, suite]", summary)
        self.assertIn("--github-summary", summary)
        self.assertIn("--expected-base", summary)
        self.assertIn("--expected-head", summary)
        self.assertIn("retention-days: 30", suite)
        self.assertIn("name: Retain every measured result", suite)

    def test_measurements_use_release_wheels_and_exact_base_head(self):
        suite = (ROOT / ".github/workflows/benchmark-suite.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("side: [baseline, candidate]", suite)
        self.assertIn("scripts.benchmark_suite build", suite)
        self.assertIn("--baseline", suite)
        self.assertIn("--candidate", suite)
        self.assertIn("--require-hashes benchmarks/requirements.lock", suite)
        self.assertNotIn("--benchmark-disable", suite)

    def test_rust_shard_prints_the_informational_decode_benchmark(self):
        suite = (ROOT / ".github/workflows/benchmark-suite.yml").read_text(
            encoding="utf-8"
        )
        headers = suite.split(
            "- name: Ensure libcurl headers for vendored librdkafka\n", 1
        )[1]
        self.assertIn("if: always() && matrix.id == 'rust'", headers)
        step = suite.split(
            "- name: Run informational connector decode throughput benchmark\n", 1
        )[1]
        # The protobuf/JSON decode comparison prints its ns/op table in the
        # rust shard log even when the gating measurement finds a regression,
        # and the log uploads with the shard's measured results.
        self.assertIn("if: always() && matrix.id == 'rust'", step)
        self.assertIn("CARGO_TARGET_DIR: target/benchmark-rust-build", step)
        self.assertIn(
            "cargo test -p calc-flow-connectors --features kafka --lib perf::",
            step,
        )
        self.assertIn("--release -- --ignored --nocapture", step)
        self.assertIn("tee target/benchmark-results/decode-throughput/run.log", step)


if __name__ == "__main__":
    unittest.main()

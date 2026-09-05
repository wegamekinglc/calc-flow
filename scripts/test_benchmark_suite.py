from __future__ import annotations

import math
import unittest

from scripts.benchmark_suite.catalog import engine_cases, shards
from scripts.benchmark_suite.report import comparison, render_report, validate_shards


def measured_case(**changes):
    return {
        "id": "engines/100/calc-flow-sql/sma20",
        "family": "engines",
        "backend": "calc-flow-sql",
        "scenario": "sma20",
        "rows": 100,
        "scope": "execute-to-arrow",
        "status": "ok",
        "baseline": [[1.0] * 10, [1.0] * 10],
        "candidate": [[1.01] * 10, [1.01] * 10],
        "comparison": "interleaved",
        "correctness": True,
        **changes,
    }


class BenchmarkSuiteTests(unittest.TestCase):
    def test_engine_matrix_covers_every_decade_and_requested_library(self):
        cases = engine_cases()
        self.assertEqual({c["rows"] for c in cases}, {10**n for n in range(1, 8)})
        self.assertEqual(
            {c["backend"] for c in cases},
            {"calc-flow-stream", "calc-flow-sql", "datafusion", "polars", "ta-lib"},
        )
        self.assertEqual(len({c["id"] for c in cases}), len(cases))
        self.assertTrue(all(c["rows"] > 0 for c in cases))

    def test_shards_preserve_existing_scales_and_all_benchmark_families(self):
        matrix = shards()
        self.assertEqual(
            {s["scale"] for s in matrix if s["family"] == "python"},
            {"overhead", "small", "standard", "nightly"},
        )
        self.assertTrue(
            {"python", "engines", "warm", "rust", "studio", "frontend"}
            <= {s["family"] for s in matrix}
        )
        self.assertEqual(len(matrix), len({s["id"] for s in matrix}))

    def test_regression_requires_both_confirmation_rounds(self):
        failed = comparison(measured_case(candidate=[[1.1] * 10] * 2))
        self.assertEqual(failed["verdict"], "regression")
        noisy = comparison(measured_case(candidate=[[1.1] * 10, [1.0] * 10]))
        self.assertEqual(noisy["verdict"], "inconclusive")

    def test_external_comparison_is_not_a_version_regression(self):
        row = measured_case(backend="ta-lib", baseline=[], comparison="external")
        self.assertEqual(comparison(row)["verdict"], "external-reference")

    def test_exact_threshold_is_not_a_roundoff_regression(self):
        result = comparison(measured_case(candidate=[[1.05] * 10] * 2))
        self.assertEqual(result["verdict"], "no-confirmed-regression")

    def test_overflowed_derived_statistics_fail_closed(self):
        for base, head in ((1e-300, 1e300), (1.0, 1e308)):
            with self.subTest(base=base, head=head), self.assertRaises(ValueError):
                comparison(
                    measured_case(
                        baseline=[[base] * 10] * 2, candidate=[[head] * 10] * 2
                    )
                )

    def test_informational_blocks_are_not_called_paired(self):
        row = measured_case(comparison="suite-blocks", candidate=[[1.2] * 10] * 2)
        self.assertEqual(comparison(row)["verdict"], "informational-slowdown")

    def test_missing_baseline_is_explicit_new_coverage(self):
        self.assertEqual(
            comparison(measured_case(baseline=[], comparison="new"))["verdict"],
            "new-coverage",
        )

    def test_invalid_samples_fail_closed(self):
        for value in (0, -1, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                comparison(measured_case(candidate=[[value] * 10] * 2))
        with self.assertRaises(ValueError):
            comparison(measured_case(candidate=[[1.0]]))
        with self.assertRaises(ValueError):
            comparison(measured_case(correctness=False))

    def test_full_summary_keeps_every_row_and_regression_columns(self):
        rows = [measured_case(id=f"case-{n}") for n in range(250)]
        report = render_report(rows, [])
        self.assertEqual(report.count("| case-"), 250)
        for heading in ("Base P50", "Head P50", "P95", "Rows/s", "Change", "Result"):
            self.assertIn(heading, report)
        self.assertIn("250", report)

    def test_failed_cases_remain_visible(self):
        report = render_report([measured_case(status="error", error="timed out")], [])
        self.assertIn("timed out", report)
        self.assertIn("error", report)

    def test_cross_library_table_keeps_unsupported_combinations_explicit(self):
        rows = [
            measured_case(**case)
            for case in engine_cases(100)
            if case["backend"] == "calc-flow-sql"
        ]
        report = render_report(rows, [])
        self.assertIn("Cross-library comparison", report)
        self.assertIn("unsupported", report)
        self.assertIn("Polars", report)
        self.assertIn("missing", report)

    def test_allocation_metrics_are_not_mislabeled_as_milliseconds(self):
        row = measured_case(
            kind="metric",
            metric="calls_per_dispatch",
            baseline_value=7,
            candidate_value=1,
        )
        report = render_report([row], [])
        self.assertIn("Allocation metrics", report)
        self.assertIn("calls_per_dispatch", report)
        self.assertNotIn("error:", report)

    def test_long_errors_do_not_pad_every_result_row(self):
        report = render_report(
            [
                measured_case(id="good"),
                measured_case(id="bad", status="error", error="x" * 2000),
            ],
            [],
        )
        self.assertIn("x" * 2000, report)
        self.assertLess(
            len(
                next(line for line in report.splitlines() if line.startswith("| good"))
            ),
            500,
        )

    def test_missing_duplicate_or_unexpected_shards_fail(self):
        expected = ["a", "b"]
        for received in (["a"], ["a", "a", "b"], ["a", "b", "c"]):
            with self.subTest(received=received), self.assertRaises(ValueError):
                validate_shards(expected, received)
        validate_shards(expected, ["b", "a"])


if __name__ == "__main__":
    unittest.main()

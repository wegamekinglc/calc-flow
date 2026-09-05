from __future__ import annotations

import math
import unittest
from copy import deepcopy
from pathlib import Path

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
    def test_documented_tables_align_full_cell_separator_widths(self):
        root = Path(__file__).resolve().parents[1]
        documents = {
            "docs/benchmark-suite.md": (0, 1, 2),
            "benchmarks/README.md": (0,),
        }
        for name, indexes in documents.items():
            blocks = (root / name).read_text(encoding="utf-8").split("\n\n")
            tables = [block for block in blocks if block.startswith("|")]
            for index in indexes:
                with self.subTest(document=name, table=index):
                    self._assert_aligned_table(tables[index])

    def _assert_aligned_table(self, text):
        rows = [line.split("|")[1:-1] for line in text.splitlines()]
        widths = [len(cell) for cell in rows[0]]
        self.assertEqual(rows[1], ["-" * width for width in widths])
        for row in rows:
            self.assertEqual([len(cell) for cell in row], widths)

    def test_engine_matrix_covers_every_decade_and_requested_library(self):
        cases = engine_cases()
        self.assertEqual({c["rows"] for c in cases}, {10**n for n in range(1, 8)})
        self.assertEqual(
            {c["backend"] for c in cases},
            {"calc-flow-stream", "calc-flow-sql", "datafusion", "polars", "ta-lib"},
        )
        self.assertEqual(len({c["id"] for c in cases}), len(cases))
        self.assertTrue(all(c["rows"] > 0 for c in cases))

    def test_shards_exclude_slow_nightly_scale_and_keep_all_families(self):
        matrix = shards()
        self.assertEqual(
            {s["scale"] for s in matrix if s["family"] == "python"},
            {"overhead", "small", "standard"},
        )
        self.assertTrue(
            {"python", "engines", "warm", "rust", "studio", "frontend"}
            <= {s["family"] for s in matrix}
        )
        self.assertEqual(len(matrix), len({s["id"] for s in matrix}))
        self.assertEqual(len(matrix), 21)

    def test_native_stream_matrix_excludes_runner_startup(self):
        cases = [c for c in engine_cases() if c["backend"] == "calc-flow-stream"]
        self.assertEqual({c["scope"] for c in cases}, {"ready-enqueue-to-arrow"})

    def test_regression_requires_both_confirmation_rounds(self):
        failed = comparison(measured_case(candidate=[[1.1] * 10] * 2))
        self.assertEqual(failed["verdict"], "regression")
        noisy = comparison(measured_case(candidate=[[1.1] * 10, [1.0] * 10]))
        self.assertEqual(noisy["verdict"], "inconclusive")

    def test_paired_changes_keep_common_mode_host_drift(self):
        baseline = list(range(1, 11))
        candidate = [value * 1.1 for value in baseline]
        row = measured_case(baseline=[baseline] * 2, candidate=[candidate] * 2)
        original = deepcopy(row)
        result = comparison(row)
        self.assertEqual(row, original)
        self.assertEqual(result["verdict"], "regression")
        self.assertAlmostEqual(result["round_changes"][0], 10)
        self.assertAlmostEqual(result["round_intervals"][0]["low"], 10)
        unpaired = comparison({**row, "candidate": [candidate[::-1]] * 2})
        self.assertEqual(unpaired["verdict"], "inconclusive")

    def test_minimum_outlier_is_diagnostic_not_a_regression(self):
        result = comparison(measured_case(baseline=[[0.5, *([1.0] * 9)]] * 2))
        self.assertEqual(result["verdict"], "no-confirmed-regression")
        self.assertGreater(result["round_min_changes"][0], 100)
        self.assertAlmostEqual(result["round_changes"][0], 1)

    def test_round_interval_uses_conservative_exact_order_statistics(self):
        candidate = [1 + value / 100 for value in range(10)]
        result = comparison(measured_case(candidate=[candidate] * 2))
        interval = result["round_intervals"][0]
        self.assertAlmostEqual(result["round_changes"][0], 4.5)
        self.assertAlmostEqual(interval["low"], 1)
        self.assertAlmostEqual(interval["high"], 8)
        self.assertEqual(interval["coverage"], 1 - 22 / 1024)
        self.assertGreaterEqual(interval["coverage"], 0.95)
        self.assertEqual(result["verdict"], "inconclusive")

    def test_confidence_confirmed_improvement_is_distinct(self):
        self.assertEqual(
            comparison(measured_case(candidate=[[0.9] * 10] * 2))["verdict"],
            "improved",
        )

    def test_interval_rank_tracks_the_sample_count(self):
        candidate = [1 + value / 100 for value in range(20)]
        result = comparison(
            measured_case(baseline=[[1.0] * 20] * 2, candidate=[candidate] * 2)
        )
        interval = result["round_intervals"][0]
        self.assertAlmostEqual(interval["low"], 5)
        self.assertAlmostEqual(interval["high"], 14)
        self.assertEqual(interval["coverage"], 1 - 43400 / 1048576)

    def test_nonfinite_paired_statistics_fail_even_when_not_the_minimum(self):
        for candidate in ([1e308, *([1.0] * 9)], [1e306] * 10):
            with (
                self.subTest(candidate=candidate),
                self.assertRaisesRegex(ValueError, "nonfinite"),
            ):
                comparison(measured_case(candidate=[candidate] * 2))

    def test_report_shows_paired_uncertainty_and_original_minima(self):
        report = render_report([measured_case()], [])
        self.assertIn("Paired median [CI]", report)
        self.assertIn("Round min changes", report)
        self.assertIn("+1.00% [+1.00%, +1.00%]", report)
        self.assertIn("97.85%", report)

    def test_hosted_warm_samples_do_not_confirm_minimum_only_slowdowns(self):
        # Unmodified seconds from run 33980036723, head 7d79fed, identical native
        # hashes on both sides. Preserve all pairs, not only the winning minima.
        fixtures = [
            (
                [
                    [
                        0.000834632,
                        0.000611983,
                        0.000601026,
                        0.000614776,
                        0.000655702,
                        0.000688851,
                        0.000687474,
                        0.000708615,
                        0.000625392,
                        0.00066844,
                    ],
                    [
                        0.000801208,
                        0.000590956,
                        0.000552128,
                        0.000591861,
                        0.000542284,
                        0.000628185,
                        0.00069511,
                        0.000619403,
                        0.000654415,
                        0.000663018,
                    ],
                ],
                [
                    [
                        0.000637691,
                        0.000641491,
                        0.000692922,
                        0.000665302,
                        0.000686693,
                        0.000666188,
                        0.000693614,
                        0.000652211,
                        0.000736917,
                        0.000736933,
                    ],
                    [
                        0.000616804,
                        0.000669563,
                        0.000684765,
                        0.000612393,
                        0.000729195,
                        0.000576464,
                        0.000639198,
                        0.000651511,
                        0.000744228,
                        0.000695181,
                    ],
                ],
            ),
            (
                [
                    [
                        0.000828516,
                        0.00077526,
                        0.000655489,
                        0.000533343,
                        0.000589528,
                        0.000575238,
                        0.000575045,
                        0.000664482,
                        0.000615912,
                        0.00064654,
                    ],
                    [
                        0.000887266,
                        0.000653964,
                        0.000662931,
                        0.000627284,
                        0.000628888,
                        0.000658046,
                        0.000617793,
                        0.000676106,
                        0.000542814,
                        0.000677755,
                    ],
                ],
                [
                    [
                        0.000677313,
                        0.000673748,
                        0.000725492,
                        0.000663151,
                        0.000660619,
                        0.000576434,
                        0.000615753,
                        0.000630328,
                        0.00065082,
                        0.000650959,
                    ],
                    [
                        0.000679059,
                        0.000617977,
                        0.000711817,
                        0.000619823,
                        0.000655059,
                        0.000661222,
                        0.000599594,
                        0.0006466,
                        0.000630907,
                        0.00058105,
                    ],
                ],
            ),
            (
                [
                    [
                        0.000863417,
                        0.000745512,
                        0.000651253,
                        0.000595508,
                        0.000702348,
                        0.000683986,
                        0.000647896,
                        0.000672762,
                        0.000676189,
                        0.000642566,
                    ],
                    [
                        0.000893984,
                        0.000608563,
                        0.000790922,
                        0.000665388,
                        0.000633459,
                        0.000677041,
                        0.000627818,
                        0.000640973,
                        0.000699562,
                        0.000699582,
                    ],
                ],
                [
                    [
                        0.000731642,
                        0.000684454,
                        0.000737015,
                        0.000710169,
                        0.000693902,
                        0.00070393,
                        0.000672272,
                        0.000626987,
                        0.000680566,
                        0.000669206,
                    ],
                    [
                        0.000654498,
                        0.000639361,
                        0.000831487,
                        0.000640963,
                        0.000739317,
                        0.000662564,
                        0.000688511,
                        0.000688502,
                        0.000651984,
                        0.000728068,
                    ],
                ],
            ),
        ]
        for baseline, candidate in fixtures:
            with self.subTest(baseline=baseline[0][0]):
                result = comparison(
                    measured_case(baseline=baseline, candidate=candidate)
                )
                self.assertEqual(result["verdict"], "inconclusive")
                self.assertTrue(all(value > 5 for value in result["round_min_changes"]))

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
        self.assertEqual(comparison(row)["round_intervals"], [])

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

    def test_cross_library_table_declares_ready_runner_measurement(self):
        report = render_report(
            [measured_case(**case) for case in engine_cases(100)], []
        )
        self.assertIn("Native stream (ready)", report)
        self.assertIn("excludes runner startup", report)
        self.assertNotIn("includes runner startup/drain", report)

    def test_cross_library_table_rejects_startup_inclusive_stream_samples(self):
        case = next(c for c in engine_cases(100) if c["backend"] == "calc-flow-stream")
        report = render_report(
            [measured_case(**{**case, "scope": "runner-start-to-drain-arrow"})], []
        )
        self.assertIn("invalid scope", report.split("## Cross-library comparison")[1])

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

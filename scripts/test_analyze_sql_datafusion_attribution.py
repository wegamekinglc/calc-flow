"""Tests for same-binary SQL/DataFusion attribution decisions."""

from __future__ import annotations

import unittest

from scripts.analyze_sql_datafusion_attribution import analyze_report
from scripts.test_verify_sql_datafusion_performance import _report


class TestSqlDataFusionAttribution(unittest.TestCase):
    def test_rejects_diagnostic_or_dirty_evidence(self) -> None:
        report = _report()
        report["profile"] = "attribution"
        report["environment"]["git_dirty"] = True

        with self.assertRaisesRegex(ValueError, "clean Git worktree"):
            analyze_report(report)

    def test_attributes_gap_and_makes_independent_gate_decisions(self) -> None:
        report = _report()
        report["profile"] = "attribution"
        case = report["cases"][0]
        calc = case["calc_flow"]
        raw = case["raw_datafusion"]
        calc["phase_medians_ms"].update(
            {
                "runtime_acquire": 0.10,
                "session_state_create": 0.50,
                "input_adapter": 0.50,
                "table_register": 0.50,
                "sql_parse": 0.20,
                "logical_optimize": 0.80,
                "physical_plan": 0.80,
                "execution_to_first_batch": 2.00,
                "execution_remaining": 59.00,
                "output_arrow_wrap": 0.30,
                "metrics_traversal": 0.20,
                "batch_envelope": 0.10,
                "run_result": 0.20,
            }
        )
        raw["phase_medians_ms"].update(
            {
                "runtime_acquire": 0.05,
                "session_state_create": 0.20,
                "input_adapter": 0.20,
                "table_register": 0.20,
                "sql_parse": 0.10,
                "logical_optimize": 0.30,
                "physical_plan": 0.30,
                "execution_to_first_batch": 1.00,
                "execution_remaining": 49.00,
                "output_arrow_wrap": 0.10,
                "metrics_traversal": 0.05,
                "batch_envelope": 0.05,
                "run_result": 0.00,
            }
        )
        for engine in (calc, raw):
            engine["phase_samples_ms"] = {
                name: [value] * 20 for name, value in engine["phase_medians_ms"].items()
            }
        calc["window_compute_ms"] = 50.0
        raw["window_compute_ms"] = 45.0

        analysis = analyze_report(report)

        result = analysis["cases"][0]
        self.assertGreaterEqual(result["explained_fraction"], 0.90)
        self.assertEqual(result["gates"]["p5"]["decision"], "no-go")
        self.assertEqual(result["gates"]["p6"]["decision"], "go")
        self.assertEqual(result["gates"]["p7"]["decision"], "go")

    def test_rejects_non_attribution_or_incomparable_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "attribution"):
            analyze_report(_report())

        report = _report()
        report["profile"] = "attribution"
        report["cases"][0]["comparability"]["comparable"] = False
        with self.assertRaisesRegex(ValueError, "comparable"):
            analyze_report(report)

    def test_fail_closed_when_less_than_ninety_percent_is_explained(self) -> None:
        report = _report()
        report["profile"] = "attribution"
        for engine in ("calc_flow", "raw_datafusion"):
            report["cases"][0][engine]["phase_medians_ms"] = {
                name: 0.0 for name in report["cases"][0][engine]["phase_medians_ms"]
            }
            report["cases"][0][engine]["phase_samples_ms"] = {
                name: [0.0] * 20
                for name in report["cases"][0][engine]["phase_medians_ms"]
            }

        with self.assertRaisesRegex(ValueError, "90%"):
            analyze_report(report)


if __name__ == "__main__":
    unittest.main()

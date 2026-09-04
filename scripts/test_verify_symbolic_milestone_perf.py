"""Unit tests for the symbolic milestone paired-performance verifier."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.verify_symbolic_milestone_perf import compare_reports

_COMMIT = "1" * 40
_MACHINE = {
    "node": "benchmark-host",
    "machine": "x86_64",
    "system": "Linux",
    "release": "test-kernel",
    "python_implementation": "CPython",
    "python_version": "3.13.9",
    "cpu": {
        "arch": "X86_64",
        "bits": 64,
        "brand_raw": "Test CPU",
        "count": 8,
    },
}


def _pairs(hand_built: list[float], symbolic: list[float]) -> list[dict[str, object]]:
    return [
        {
            "order": "hand-built-first" if index % 2 == 0 else "symbolic-first",
            "hand_built_seconds": baseline,
            "symbolic_seconds": candidate,
        }
        for index, (baseline, candidate) in enumerate(
            zip(hand_built, symbolic, strict=True)
        )
    ]


def _benchmark(paired_samples: list[dict[str, object]]) -> dict[str, object]:
    return {
        "name": "test_row_local",
        "stats": {"mean": 2.0, "rounds": 20, "data": [2.0] * 20},
        "extra_info": {
            "scenario": "sce05_row_local_20_columns",
            "comparison_contract": "same-process-alternating-v1",
            "workload_contract": "sce05-row-local-v1",
            "scale": "standard",
            "input_rows": 100_000,
            "output_rows": 100_000,
            "paired_samples": paired_samples,
        },
    }


def _rolling_benchmark(
    paired_samples: list[dict[str, object]], *, finite_rows: int = 100
) -> dict[str, object]:
    return {
        "name": "test_rolling_kernel_sma20",
        "stats": {
            "mean": 2.0,
            "rounds": len(paired_samples),
            "data": [2.0] * len(paired_samples),
        },
        "extra_info": {
            "scenario": "rolling_kernel_sma20",
            "comparison_contract": "same-process-alternating-v1",
            "workload_contract": "rolling-kernel-paired-v1",
            "scale": "overhead",
            "input_rows": 100,
            "output_rows": 100,
            "machine_fingerprint": "a" * 64,
            "dependency_fingerprint": "b" * 64,
            "workload_fingerprint": "c" * 64,
            "oracle": "independent_direct_window_v1",
            "oracle_checked_rows": 100,
            "oracle_finite_rows": finite_rows,
            "oracle_rtol": 1e-10,
            "oracle_atol": 1e-10,
            "optimized_kernel": "ordered_primitive",
            "optimized_shared_state_groups": 1,
            "reference_rolling_rewrites": 0,
            "fast_window": None,
            "slow_window": 20,
            "paired_samples": paired_samples,
        },
    }


def _write_report(
    path: Path,
    paired_samples: list[dict[str, object]] | None,
    *,
    commit: str = _COMMIT,
    dirty: bool = False,
    machine: dict[str, object] | None = None,
) -> None:
    benchmarks = [] if paired_samples is None else [_benchmark(paired_samples)]
    path.write_text(
        json.dumps(
            {
                "commit_info": {"id": commit, "dirty": dirty},
                "machine_info": machine or _MACHINE,
                "benchmarks": benchmarks,
            }
        ),
        encoding="utf-8",
    )


class TestCompareReports(unittest.TestCase):
    def test_bootstrap_summary_is_deterministic(self) -> None:
        with TemporaryDirectory() as raw:
            report = Path(raw) / "report.json"
            _write_report(report, _pairs([1.0] * 20, [0.98, 1.04] * 10))

            first = compare_reports(
                (report,),
                scenarios=("sce05_row_local_20_columns",),
                bootstrap_resamples=2_000,
            )
            second = compare_reports(
                (report,),
                scenarios=("sce05_row_local_20_columns",),
                bootstrap_resamples=2_000,
            )

        self.assertEqual(first, second)

    def test_upper_interval_below_gate_passes(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "first.json"
            second = root / "second.json"
            _write_report(first, _pairs([1.00] * 20, [1.02] * 20))
            _write_report(second, _pairs([1.01] * 20, [1.03] * 20))

            summary = compare_reports(
                (first, second),
                scenarios=("sce05_row_local_20_columns",),
                bootstrap_resamples=2_000,
            )

        self.assertEqual(summary["decision"], "pass")
        result = summary["scenarios"][0]
        self.assertEqual(result["decision"], "pass")
        self.assertLessEqual(result["regression_interval_percent"][1], 5.0)

    def test_lower_interval_above_gate_fails(self) -> None:
        with TemporaryDirectory() as raw:
            report = Path(raw) / "report.json"
            _write_report(report, _pairs([1.0] * 20, [1.08] * 20))

            summary = compare_reports(
                (report,),
                scenarios=("sce05_row_local_20_columns",),
                bootstrap_resamples=2_000,
            )

        self.assertEqual(summary["decision"], "fail")
        self.assertGreater(
            summary["scenarios"][0]["regression_interval_percent"][0], 5.0
        )

    def test_interval_crossing_gate_is_inconclusive(self) -> None:
        with TemporaryDirectory() as raw:
            report = Path(raw) / "report.json"
            _write_report(report, _pairs([1.0] * 20, [0.9, 1.2] * 10))

            summary = compare_reports(
                (report,),
                scenarios=("sce05_row_local_20_columns",),
                bootstrap_resamples=2_000,
            )

        self.assertEqual(summary["decision"], "inconclusive")
        interval = summary["scenarios"][0]["regression_interval_percent"]
        self.assertLessEqual(interval[0], 5.0)
        self.assertGreater(interval[1], 5.0)

    def test_missing_same_process_pair_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            report = Path(raw) / "report.json"
            _write_report(report, None)

            with self.assertRaisesRegex(ValueError, "same-process pair"):
                compare_reports((report,), scenarios=("sce05_row_local_20_columns",))

    def test_invalid_alternating_order_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            report = Path(raw) / "report.json"
            samples = _pairs([1.0] * 20, [1.0] * 20)
            samples[1]["order"] = "hand-built-first"
            _write_report(report, samples)

            with self.assertRaisesRegex(ValueError, "alternating order"):
                compare_reports((report,), scenarios=("sce05_row_local_20_columns",))

    def test_dirty_or_mismatched_provenance_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            clean = root / "clean.json"
            dirty = root / "dirty.json"
            samples = _pairs([1.0] * 20, [1.0] * 20)
            _write_report(clean, samples)
            _write_report(dirty, samples, dirty=True)

            with self.assertRaisesRegex(ValueError, "dirty"):
                compare_reports(
                    (clean, dirty), scenarios=("sce05_row_local_20_columns",)
                )

            _write_report(dirty, samples, commit="2" * 40)
            with self.assertRaisesRegex(ValueError, "commit"):
                compare_reports(
                    (clean, dirty), scenarios=("sce05_row_local_20_columns",)
                )

    def test_expected_commit_mismatch_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            report = Path(raw) / "report.json"
            _write_report(report, _pairs([1.0] * 20, [1.0] * 20))

            with self.assertRaisesRegex(ValueError, "expected commit"):
                compare_reports(
                    (report,),
                    scenarios=("sce05_row_local_20_columns",),
                    expected_commit="2" * 40,
                )

    def test_rolling_oracle_and_non_vacuity_are_required(self) -> None:
        with TemporaryDirectory() as raw:
            report = Path(raw) / "report.json"
            pairs = _pairs([1.0] * 60, [0.5] * 60)
            report.write_text(
                json.dumps(
                    {
                        "commit_info": {"id": _COMMIT, "dirty": False},
                        "machine_info": _MACHINE,
                        "benchmarks": [_rolling_benchmark(pairs)],
                    }
                ),
                encoding="utf-8",
            )
            summary = compare_reports(
                (report,),
                scenarios=("rolling_kernel_sma20",),
                expected_commit=_COMMIT,
                bootstrap_resamples=2_000,
            )
            self.assertEqual(summary["decision"], "pass")

            report.write_text(
                json.dumps(
                    {
                        "commit_info": {"id": _COMMIT, "dirty": False},
                        "machine_info": _MACHINE,
                        "benchmarks": [_rolling_benchmark(pairs, finite_rows=0)],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "no finite oracle outputs"):
                compare_reports((report,), scenarios=("rolling_kernel_sma20",))

    def test_rolling_gate_requires_sixty_pairs(self) -> None:
        with TemporaryDirectory() as raw:
            report = Path(raw) / "report.json"
            pairs = _pairs([1.0] * 20, [0.5] * 20)
            report.write_text(
                json.dumps(
                    {
                        "commit_info": {"id": _COMMIT, "dirty": False},
                        "machine_info": _MACHINE,
                        "benchmarks": [_rolling_benchmark(pairs)],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "at least 60 paired samples"):
                compare_reports((report,), scenarios=("rolling_kernel_sma20",))

    def test_machine_and_workload_mismatch_fail_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "first.json"
            second = root / "second.json"
            samples = _pairs([1.0] * 20, [1.0] * 20)
            _write_report(first, samples)
            _write_report(
                second,
                samples,
                machine={**_MACHINE, "node": "different-host"},
            )

            with self.assertRaisesRegex(ValueError, "machine"):
                compare_reports(
                    (first, second), scenarios=("sce05_row_local_20_columns",)
                )

            _write_report(second, samples)
            data = json.loads(second.read_text(encoding="utf-8"))
            data["benchmarks"][0]["extra_info"]["input_rows"] = 99_999
            second.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "workload"):
                compare_reports(
                    (first, second), scenarios=("sce05_row_local_20_columns",)
                )


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the M7-01 performance gate runner."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.verify_perf_gates import (
    check_criterion_regression,
    check_regression,
    check_stream_lifecycle_regression,
    load_baseline,
    load_criterion,
    load_criterion_provenance,
    load_provenance,
    load_stream_lifecycle,
)


def _write_benchmark(
    directory: Path,
    name: str,
    mean: float,
    stddev: float,
    *,
    rounds: int = 100,
) -> None:
    data = {
        "benchmarks": [
            {
                "name": name,
                "stats": {"mean": mean, "stddev": stddev, "rounds": rounds},
            }
        ]
    }
    (directory / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


def _comparable_lifecycle() -> dict[str, object]:
    return {
        "checkpoint_bytes": 100,
        "checkpoint_bytes_p50": 100,
        "checkpoint_bytes_p95": 101,
        "checkpoint_duration_p50_seconds": 0.01,
        "checkpoint_duration_p95_seconds": 0.02,
        "recovery_duration_p50_seconds": 0.03,
        "recovery_duration_p95_seconds": 0.04,
        "machine_fingerprint": "a" * 64,
        "dependency_fingerprint": "b" * 64,
        "workload_fingerprint": "c" * 64,
    }


class TestLoadBaseline(unittest.TestCase):
    def test_loads_json_files_from_directory(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            _write_benchmark(directory, "bench_a", 1.0, 0.01)
            _write_benchmark(directory, "bench_b", 2.0, 0.02)
            results = load_baseline(directory)
            self.assertEqual(set(results), {"bench_a", "bench_b"})
            self.assertEqual(results["bench_a"]["mean_seconds"], 1.0)

    def test_empty_directory_returns_empty(self) -> None:
        with TemporaryDirectory() as raw:
            results = load_baseline(Path(raw))
            self.assertEqual(results, {})

    def test_excludes_isolated_stream_lifecycle_case(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            _write_benchmark(directory, "generic", 1.0, 0.01)
            directory.joinpath("stream-lifecycle.json").write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {
                                "name": "stream_lifecycle",
                                "stats": {
                                    "mean": 2.0,
                                    "stddev": 0.02,
                                    "rounds": 20,
                                },
                                "extra_info": {
                                    "scenario": "symbolic_stream_window_checkpoint"
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            results = load_baseline(directory)

        self.assertEqual(set(results), {"generic"})


class TestExactRefProvenance(unittest.TestCase):
    def test_loads_exact_python_and_criterion_commits(self) -> None:
        baseline_sha = "1" * 40
        candidate_sha = "2" * 40
        with TemporaryDirectory() as raw:
            root = Path(raw)
            root.joinpath("provenance.json").write_text(
                json.dumps({"role": "baseline", "git_sha": baseline_sha}),
                encoding="utf-8",
            )
            root.joinpath("criterion-provenance.json").write_text(
                json.dumps(
                    {"exact-baseline": baseline_sha, "exact-candidate": candidate_sha}
                ),
                encoding="utf-8",
            )

            provenance = load_provenance(root, "baseline")
            criterion = load_criterion_provenance(
                root, "exact-baseline", "exact-candidate"
            )

        self.assertEqual(provenance["git_sha"], baseline_sha)
        self.assertEqual(criterion, (baseline_sha, candidate_sha))

    def test_rejects_abbreviated_or_mislabeled_provenance(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            root.joinpath("provenance.json").write_text(
                json.dumps({"role": "candidate", "git_sha": "abc123"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "role"):
                load_provenance(root, "baseline")

            root.joinpath("provenance.json").write_text(
                json.dumps({"role": "baseline", "git_sha": "abc123"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "full git SHA"):
                load_provenance(root, "baseline")


class TestCheckRegression(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.baseline = {
            "bench_a": {
                "name": "bench_a",
                "mean_seconds": 1.0,
                "std_dev": 0.01,
                "rounds": 100,
            },
            "bench_b": {
                "name": "bench_b",
                "mean_seconds": 2.0,
                "std_dev": 0.02,
                "rounds": 100,
            },
        }

    def test_within_threshold_passes(self) -> None:
        candidate = {
            "bench_a": {
                "name": "bench_a",
                "mean_seconds": 1.03,
                "std_dev": 0.01,
                "rounds": 100,
            },
            "bench_b": {
                "name": "bench_b",
                "mean_seconds": 2.04,
                "std_dev": 0.02,
                "rounds": 100,
            },
        }
        self.assertEqual(check_regression(self.baseline, candidate), [])

    def test_improvement_passes(self) -> None:
        candidate = {
            "bench_a": {
                "name": "bench_a",
                "mean_seconds": 0.5,
                "std_dev": 0.01,
                "rounds": 100,
            },
            "bench_b": {
                "name": "bench_b",
                "mean_seconds": 1.0,
                "std_dev": 0.02,
                "rounds": 100,
            },
        }
        self.assertEqual(check_regression(self.baseline, candidate), [])

    def test_exceeding_threshold_fails(self) -> None:
        candidate = {
            "bench_a": {
                "name": "bench_a",
                "mean_seconds": 1.06,
                "std_dev": 0.01,
                "rounds": 100,
            },
            "bench_b": {
                "name": "bench_b",
                "mean_seconds": 2.0,
                "std_dev": 0.02,
                "rounds": 100,
            },
        }
        regressions = check_regression(self.baseline, candidate)
        self.assertEqual(len(regressions), 1)
        self.assertEqual(regressions[0][0], "bench_a")
        self.assertGreater(regressions[0][1], 0.05)

    def test_missing_candidate_fails_closed(self) -> None:
        candidate = {
            "bench_a": {
                "name": "bench_a",
                "mean_seconds": 1.0,
                "std_dev": 0.01,
                "rounds": 100,
            },
        }
        with self.assertRaisesRegex(
            ValueError, "missing candidate benchmarks: bench_b"
        ):
            check_regression(self.baseline, candidate)

    def test_noisy_mean_regression_without_confidence_support_passes(self) -> None:
        candidate = {
            "bench_a": {
                "name": "bench_a",
                "mean_seconds": 1.10,
                "std_dev": 1.0,
                "rounds": 4,
            },
            "bench_b": self.baseline["bench_b"],
        }
        self.assertEqual(check_regression(self.baseline, candidate), [])

    def test_zero_baseline_mean_fails_closed(self) -> None:
        baseline = {
            "bench": {
                "name": "bench",
                "mean_seconds": 0.0,
                "std_dev": 0.0,
                "rounds": 100,
            }
        }
        with self.assertRaisesRegex(ValueError, "positive mean"):
            check_regression(baseline, baseline)


class TestCriterionGate(unittest.TestCase):
    def test_loads_named_baseline_and_uses_reported_interval(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            report = root / "stream" / "channel" / "exact-base"
            report.mkdir(parents=True)
            report.joinpath("estimates.json").write_text(
                json.dumps(
                    {
                        "mean": {
                            "point_estimate": 1_000_000_000,
                            "confidence_interval": {
                                "lower_bound": 990_000_000,
                                "upper_bound": 1_010_000_000,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            results = load_criterion(root, "exact-base")

        self.assertEqual(set(results), {"stream/channel"})
        self.assertEqual(results["stream/channel"]["mean_seconds"], 1.0)

    def test_statistically_supported_criterion_regression_fails(self) -> None:
        baseline = {
            "stream/channel": {
                "name": "stream/channel",
                "mean_seconds": 1.0,
                "lower_seconds": 0.99,
                "upper_seconds": 1.01,
            }
        }
        candidate = {
            "stream/channel": {
                "name": "stream/channel",
                "mean_seconds": 1.10,
                "lower_seconds": 1.09,
                "upper_seconds": 1.11,
            }
        }
        self.assertEqual(
            check_criterion_regression(baseline, candidate)[0][0],
            "stream/channel",
        )

    def test_missing_candidate_criterion_case_fails_closed(self) -> None:
        baseline = {
            "stream/channel": {
                "name": "stream/channel",
                "mean_seconds": 1.0,
                "lower_seconds": 0.99,
                "upper_seconds": 1.01,
            }
        }
        with self.assertRaisesRegex(ValueError, "missing candidate Criterion"):
            check_criterion_regression(baseline, {})


class TestStreamLifecycleGate(unittest.TestCase):
    def test_loads_phase_quantiles_and_checkpoint_bytes(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            root.joinpath("stream.json").write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {
                                "extra_info": {
                                    "scenario": "symbolic_stream_window_checkpoint",
                                    "checkpoint_bytes": 100,
                                    "checkpoint_bytes_p50": 100,
                                    "checkpoint_bytes_p95": 101,
                                    "checkpoint_duration_p50_seconds": 0.01,
                                    "checkpoint_duration_p95_seconds": 0.02,
                                    "recovery_duration_p50_seconds": 0.03,
                                    "recovery_duration_p95_seconds": 0.04,
                                    "machine_fingerprint": "a" * 64,
                                    "dependency_fingerprint": "b" * 64,
                                    "workload_fingerprint": "c" * 64,
                                }
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = load_stream_lifecycle(root)

        self.assertEqual(result["checkpoint_bytes"], 100)

    def test_fails_only_on_supported_phase_or_size_regression(self) -> None:
        baseline = {
            "checkpoint_bytes": 100,
            "checkpoint_bytes_p50": 100,
            "checkpoint_bytes_p95": 101,
            "checkpoint_duration_p50_seconds": 0.01,
            "checkpoint_duration_p95_seconds": 0.02,
            "recovery_duration_p50_seconds": 0.03,
            "recovery_duration_p95_seconds": 0.04,
            "machine_fingerprint": "a" * 64,
            "dependency_fingerprint": "b" * 64,
            "workload_fingerprint": "c" * 64,
        }
        stable = {
            **baseline,
            "checkpoint_duration_p50_seconds": 0.02,
            "recovery_duration_p50_seconds": 0.04,
        }
        regressed = {
            **baseline,
            "checkpoint_bytes": 107,
            "checkpoint_bytes_p50": 107,
            "checkpoint_bytes_p95": 108,
            "checkpoint_duration_p50_seconds": 0.022,
            "recovery_duration_p50_seconds": 0.045,
        }

        self.assertEqual(check_stream_lifecycle_regression(baseline, stable), [])
        self.assertEqual(
            {
                name
                for name, _delta in check_stream_lifecycle_regression(
                    baseline, regressed
                )
            },
            {"checkpoint_bytes", "checkpoint_duration", "recovery_duration"},
        )

    def test_rejects_incomparable_stream_lifecycle_identity(self) -> None:
        baseline = {
            "checkpoint_bytes": 100,
            "checkpoint_bytes_p50": 100,
            "checkpoint_bytes_p95": 101,
            "checkpoint_duration_p50_seconds": 0.01,
            "checkpoint_duration_p95_seconds": 0.02,
            "recovery_duration_p50_seconds": 0.03,
            "recovery_duration_p95_seconds": 0.04,
            "machine_fingerprint": "a" * 64,
            "dependency_fingerprint": "b" * 64,
            "workload_fingerprint": "c" * 64,
        }
        candidate = {**baseline, "machine_fingerprint": "d" * 64}

        with self.assertRaisesRegex(ValueError, "machine_fingerprint"):
            check_stream_lifecycle_regression(baseline, candidate)

    def test_dependency_drift_fails_closed_with_an_escape_hint(self) -> None:
        baseline = _comparable_lifecycle()
        candidate = {**baseline, "dependency_fingerprint": "e" * 64}

        with self.assertRaisesRegex(ValueError, "dependency-drift acknowledgement"):
            check_stream_lifecycle_regression(baseline, candidate)

    def test_acknowledged_dependency_drift_still_gates_regressions(self) -> None:
        baseline = _comparable_lifecycle()
        drifted_stable = {
            **baseline,
            "dependency_fingerprint": "e" * 64,
        }
        drifted_regressed = {
            **drifted_stable,
            "checkpoint_bytes": 107,
            "checkpoint_bytes_p50": 107,
            "checkpoint_bytes_p95": 108,
        }

        self.assertEqual(
            check_stream_lifecycle_regression(
                baseline, drifted_stable, allow_dependency_drift=True
            ),
            [],
        )
        self.assertEqual(
            [
                name
                for name, _delta in check_stream_lifecycle_regression(
                    baseline, drifted_regressed, allow_dependency_drift=True
                )
            ],
            ["checkpoint_bytes"],
        )
        with self.assertRaisesRegex(ValueError, "machine_fingerprint"):
            check_stream_lifecycle_regression(
                baseline,
                {**drifted_stable, "machine_fingerprint": "f" * 64},
                allow_dependency_drift=True,
            )

    def test_malformed_benchmark_entries_fail_with_file_context(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            directory.joinpath("malformed.json").write_text(
                json.dumps({"benchmarks": [{"stats": {"mean": 1.0}}]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "malformed.json"):
                load_baseline(directory)

        with TemporaryDirectory() as raw:
            directory = Path(raw)
            directory.joinpath("not_a_list.json").write_text(
                json.dumps({"benchmarks": "nope"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "benchmark entry list"):
                load_baseline(directory)


if __name__ == "__main__":
    unittest.main()

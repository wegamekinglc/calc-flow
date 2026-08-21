"""Unit tests for the M7-01 performance gate runner."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.verify_perf_gates import (
    check_criterion_regression,
    check_regression,
    load_baseline,
    load_criterion,
    load_criterion_provenance,
    load_provenance,
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


if __name__ == "__main__":
    unittest.main()

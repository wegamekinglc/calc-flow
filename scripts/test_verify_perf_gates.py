"""Unit tests for the M7-01 performance gate runner."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.verify_perf_gates import check_regression, load_baseline


def _write_benchmark(directory: Path, name: str, mean: float, stddev: float) -> None:
    data = {
        "benchmarks": [
            {
                "name": name,
                "stats": {"mean": mean, "stddev": stddev},
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


class TestCheckRegression(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.baseline = {
            "bench_a": {"name": "bench_a", "mean_seconds": 1.0, "std_dev": 0.01},
            "bench_b": {"name": "bench_b", "mean_seconds": 2.0, "std_dev": 0.02},
        }

    def test_within_threshold_passes(self) -> None:
        candidate = {
            "bench_a": {"name": "bench_a", "mean_seconds": 1.03, "std_dev": 0.01},
            "bench_b": {"name": "bench_b", "mean_seconds": 2.04, "std_dev": 0.02},
        }
        self.assertEqual(check_regression(self.baseline, candidate), [])

    def test_improvement_passes(self) -> None:
        candidate = {
            "bench_a": {"name": "bench_a", "mean_seconds": 0.5, "std_dev": 0.01},
            "bench_b": {"name": "bench_b", "mean_seconds": 1.0, "std_dev": 0.02},
        }
        self.assertEqual(check_regression(self.baseline, candidate), [])

    def test_exceeding_threshold_fails(self) -> None:
        candidate = {
            "bench_a": {"name": "bench_a", "mean_seconds": 1.06, "std_dev": 0.01},
            "bench_b": {"name": "bench_b", "mean_seconds": 2.0, "std_dev": 0.02},
        }
        regressions = check_regression(self.baseline, candidate)
        self.assertEqual(len(regressions), 1)
        self.assertEqual(regressions[0][0], "bench_a")
        self.assertGreater(regressions[0][1], 0.05)

    def test_extra_baseline_is_ignored(self) -> None:
        candidate = {
            "bench_a": {"name": "bench_a", "mean_seconds": 1.0, "std_dev": 0.01},
        }
        # bench_b has no candidate; it is not counted as a regression.
        self.assertEqual(check_regression(self.baseline, candidate), [])


if __name__ == "__main__":
    unittest.main()

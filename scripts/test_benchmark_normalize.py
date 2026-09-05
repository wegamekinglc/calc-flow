from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_suite.legacy import combine_blocks
from scripts.benchmark_suite.normalize import (
    criterion_rows,
    pytest_rows,
    read_json,
    vitest_rows,
)


class BenchmarkNormalizeTests(unittest.TestCase):
    def test_json_overflow_is_not_a_finite_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"seconds": 1e309}', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_json(path)

    def test_pytest_retains_every_case_and_saved_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pytest.json"
            path.write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {
                                "fullname": "a",
                                "stats": {"data": [1.0, 2.0], "median": 1.5},
                                "extra_info": {"input_rows": 10},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = pytest_rows(path)
        self.assertEqual(rows["a"]["samples"], [1.0, 2.0])
        self.assertEqual(rows["a"]["rows"], 10)

    def test_criterion_normalizes_elapsed_time_by_iterations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "case/new"
            output.mkdir(parents=True)
            (output / "benchmark.json").write_text(
                json.dumps({"full_id": "case"}), encoding="utf-8"
            )
            (output / "sample.json").write_text(
                json.dumps({"iters": [10, 20], "times": [100, 400]}), encoding="utf-8"
            )
            rows = criterion_rows(root)
        self.assertEqual(rows["case"]["samples"], [1e-8, 2e-8])

    def test_empty_criterion_results_are_not_success(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            criterion_rows(Path(directory))

    def test_vitest_converts_milliseconds_to_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vitest.json"
            path.write_text(
                json.dumps(
                    {
                        "files": [
                            {
                                "groups": [
                                    {
                                        "fullName": "group",
                                        "benchmarks": [
                                            {"name": "case", "samples": [1, 2]}
                                        ],
                                    }
                                ]
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = vitest_rows(path)
        self.assertEqual(rows["case"]["samples"], [0.001, 0.002])

    def test_dropped_legacy_case_is_a_visible_error(self):
        value = {"samples": [1.0], "rows": 10, "scope": "test", "metadata": {}}
        blocks = {"baseline": [{"case": value}] * 2, "candidate": [{}, {}]}
        rows = combine_blocks({"id": "python-small", "family": "python"}, blocks)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "error")

    def test_vitest_without_raw_samples_is_not_a_fake_percentile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vitest.json"
            path.write_text(
                json.dumps(
                    {
                        "files": [
                            {
                                "groups": [
                                    {
                                        "fullName": "group",
                                        "benchmarks": [
                                            {
                                                "name": "case",
                                                "samples": [],
                                                "median": 1,
                                                "sampleCount": 10,
                                            }
                                        ],
                                    }
                                ]
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                vitest_rows(path)

    def test_changed_workload_fingerprint_is_not_compared(self):
        value = {
            "samples": [1.0],
            "rows": 10,
            "scope": "test",
            "metadata": {"workload_fingerprint": "a"},
        }
        changed = {**value, "metadata": {"workload_fingerprint": "b"}}
        blocks = {
            "baseline": [{"case": value}] * 2,
            "candidate": [{"case": changed}] * 2,
        }
        rows = combine_blocks({"id": "python-small", "family": "python"}, blocks)
        self.assertEqual(rows[0]["status"], "error")

    def test_empty_blocks_cannot_pass(self):
        with self.assertRaises(ValueError):
            combine_blocks(
                {"id": "python-small", "family": "python"},
                {"baseline": [{}, {}], "candidate": [{}, {}]},
            )


if __name__ == "__main__":
    unittest.main()

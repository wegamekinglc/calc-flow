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

    def test_dropped_legacy_case_is_a_visible_error(self):
        value = {"samples": [1.0], "rows": 10, "scope": "test", "metadata": {}}
        blocks = {"baseline": [{"case": value}] * 2, "candidate": [{}, {}]}
        rows = combine_blocks({"id": "python-small", "family": "python"}, blocks)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "error")

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

    def test_only_reviewed_rolling_diagnostics_compare_across_releases(self):
        original = "9abc1fc9a800ee545bc70c83b7c5550e0323f49eb34959b2d3f699a0d0b6fd4f"
        current = "947f95f5b2093dd36a83a2f12cbcfeff76f56b72a992cabdcb57515dc847ed9a"
        shard = {"id": "python-overhead", "family": "python"}

        def row(digest, workload):
            return {
                "samples": [1.0] * 20,
                "rows": 990,
                "scope": "pytest-native-boundary",
                "metadata": {
                    "benchmark_source_sha256": digest,
                    "workload_fingerprint": workload,
                    "dependency_fingerprint": "b" * 64,
                    "machine_fingerprint": "c" * 64,
                },
            }

        def compare(case, base, head, workload="a" * 64, *, last=None, support=None):
            blocks = {
                "baseline": [{case: row(base, "a" * 64)}] * 2,
                "candidate": [{case: row(head, workload)}] * 2,
            }
            if last is not None:
                blocks["candidate"][1] = {case: row(last, workload)}
            if support is not None:
                blocks["candidate"][0][case]["metadata"]["benchmark_support_sha256"] = (
                    support
                )
            return combine_blocks(shard, blocks)[0]

        for case in ("sma20", "dual_sma"):
            name = "benchmarks/test_rolling_kernel.py::" + (
                f"test_rolling_kernel_{case}[overhead]"
            )
            result = compare(name, original, current)
            self.assertEqual(result["status"], "ok", result.get("error"))
            self.assertEqual(
                result["metadata"]["rolling_explain_source_sha256"],
                {"baseline": original, "candidate": current},
            )
            for result in (
                compare(name, current, original),
                compare(name, original, current, last=original),
                compare(name, original, current, support="f" * 64),
                compare(name, original, "d" * 64),
                compare(name, original, current, "f" * 64),
                compare(
                    name.replace("test_rolling_kernel.py", "test_other.py"),
                    original,
                    current,
                ),
            ):
                self.assertEqual(result["status"], "error")

    def test_empty_blocks_cannot_pass(self):
        with self.assertRaises(ValueError):
            combine_blocks(
                {"id": "python-small", "family": "python"},
                {"baseline": [{}, {}], "candidate": [{}, {}]},
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_suite.aggregate import case_failures, collect, validate_fragment
from scripts.benchmark_suite.catalog import CONTRACT, get_shard, shard_cases
from scripts.benchmark_suite.provenance import harness_sha256
from scripts.benchmark_suite.validation import _validate_catalog_cases
from scripts.test_benchmark_suite import measured_case

BASE, HEAD = "a" * 40, "b" * 40


def fragment():
    shard = get_shard("engines-10")
    return {
        "contract": CONTRACT,
        "harness_sha256": harness_sha256(),
        "shard": shard,
        "releases": {
            side: {
                "git_sha": sha,
                "git_clean": True,
                "build_profile": "release",
                "native_sha256": "a" * 64,
            }
            for side, sha in (("baseline", BASE), ("candidate", HEAD))
        },
        "cases": [
            {**case, "status": "error", "error": "fixture failure"}
            for case in shard_cases(shard)
        ],
        "errors": [],
    }


class BenchmarkAggregateTests(unittest.TestCase):
    def collect_one(self, report):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results.json").write_text(json.dumps(report), encoding="utf-8")
            return collect(root, BASE, HEAD)

    def test_malformed_artifact_still_renders_missing_catalog_cases(self):
        for report in ([], {"shard": None}, {"cases": [None]}):
            with self.subTest(report=report):
                cases, errors, _ = self.collect_one(report)
                self.assertTrue(errors)
                self.assertTrue(
                    any(
                        case["id"] == "engines/10/calc-flow-sql/sma20" for case in cases
                    )
                )

    def test_present_but_empty_shard_does_not_hide_missing_cases(self):
        report = fragment()
        report["cases"] = []
        cases, errors, _ = self.collect_one(report)
        self.assertTrue(errors)
        names = {case["id"] for case in cases}
        self.assertTrue({case["id"] for case in shard_cases(report["shard"])} <= names)

    def test_partial_shard_preserves_the_results_that_were_returned(self):
        report = fragment()
        retained = report["cases"][0]
        report["cases"] = [retained]
        cases, errors, _ = self.collect_one(report)
        self.assertTrue(errors)
        actual = next(case for case in cases if case["id"] == retained["id"])
        self.assertEqual(actual, retained)

    def test_nonfinite_artifact_does_not_poison_summary_json(self):
        report = fragment()
        report["cases"][0]["candidate"] = [[float("nan")]]
        cases, errors, _ = self.collect_one(report)
        self.assertTrue(errors)
        json.dumps(cases, allow_nan=False)

    def test_dirty_or_wrong_baseline_is_not_comparable(self):
        for key, value in (("git_clean", False), ("git_sha", HEAD)):
            report = fragment()
            report["releases"]["baseline"][key] = value
            with self.assertRaises(ValueError):
                validate_fragment(report, BASE, HEAD)

    def test_invalid_allocation_values_fail_the_gate(self):
        for value in (-1, float("nan"), True, "0"):
            row = measured_case(
                kind="metric",
                metric="calls_per_dispatch",
                baseline_value=0,
                candidate_value=value,
            )
            self.assertTrue(case_failures([row]))

    def test_wrong_comparison_kind_is_not_a_historical_pass(self):
        report = fragment()
        row = measured_case(
            **shard_cases(report["shard"])[0], comparison="external", baseline=[]
        )
        report["cases"][0] = row
        with self.assertRaisesRegex(ValueError, "comparison"):
            validate_fragment(report, BASE, HEAD)

    def test_duplicate_case_cannot_pass_inventory_validation(self):
        report = fragment()
        report["cases"].append(copy.deepcopy(report["cases"][0]))
        with self.assertRaises(ValueError):
            validate_fragment(report, BASE, HEAD)

    def test_raw_samples_and_release_hashes_are_revalidated(self):
        report = fragment()
        row = measured_case(**shard_cases(report["shard"])[0])
        row["evidence"] = [
            {
                "environment": {"polars_threads": 32, "tokio_worker_threads": "32"},
                "native_sha256": {side: "a" * 64 for side in ("baseline", "candidate")},
                "completion": {
                    side: {"state": "completed"} for side in ("baseline", "candidate")
                },
                "samples": {
                    side: [
                        {"seconds": seconds, "correctness": {"passed": True}}
                        for seconds in row[side][index]
                    ]
                    for side in ("baseline", "candidate")
                },
            }
            for index in range(2)
        ]
        report["cases"][0] = row
        _validate_catalog_cases(report, report["shard"])
        row["evidence"][0]["native_sha256"]["baseline"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "hash"):
            _validate_catalog_cases(report, report["shard"])
        row["evidence"][0]["native_sha256"]["baseline"] = "a" * 64
        row["evidence"][1]["samples"]["candidate"][0]["seconds"] = 9.0
        with self.assertRaisesRegex(ValueError, "samples"):
            _validate_catalog_cases(report, report["shard"])


if __name__ == "__main__":
    unittest.main()

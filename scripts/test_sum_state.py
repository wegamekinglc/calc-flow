"""Reject incomplete or mismatched focused SUM measurement evidence."""

from __future__ import annotations

import unittest
from copy import deepcopy

from scripts.benchmark_suite.sum_state import (
    THREAD_ENV,
    compare_builds,
    compare_oracles,
    validate_sample,
)
from scripts.benchmark_suite.sum_state_profile import clock_alignment


class SumStateEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.case = {"mode": "stream", "partitions": 1, "cpus": [0]}
        self.fixture = {
            "input_sha256": "a" * 64,
            "schema_sha256": "b" * 64,
            "input_rows": 12,
            "batch_rows": [10, 2],
            "snapshot_sha256": ["c" * 64, "d" * 64],
            "snapshot_schema_sha256": "e" * 64,
            "snapshot_rows": [10, 10],
        }
        self.row = {
            "process_observations": {
                "observed_cpu_affinity": [0],
                "observed_thread_environment": {name: "1" for name in THREAD_ENV},
            },
            "schema": "calc-flow.sum-state.v1",
            "mode": "stream",
            "partitions": 1,
            "seconds": 0.01,
            "process_cpu_ns": 9_000_000,
            "start_ns": 100,
            "end_ns": 10_000_100,
            "validated_all_snapshots": True,
            "input_sha256": "a" * 64,
            "input_schema_sha256": "b" * 64,
            "input_rows": 12,
            "batch_rows": [10, 2],
            "snapshot_rows": [10, 10],
            "snapshot_sha256": ["c" * 64, "d" * 64],
            "snapshot_schema_sha256": ["e" * 64, "e" * 64],
            "diagnostics": {
                "state": "completed",
                "terminal_cause": "natural_end",
                "task_count": 0,
                "task_errors": 0,
                "metrics_overflowed": False,
                "checkpoint": {"current_epoch": None, "last_completed_epoch": None},
                "sources": {"s": {"errors": 0, "ended": True}},
                "operators": {"o": {"errors": 0, "ended": True}},
                "sinks": {"s": {"errors": 0, "ended": True}},
            },
        }

    def test_accepts_complete_per_snapshot_evidence(self):
        validate_sample(self.row, self.case, self.fixture)
        compare_oracles(self.row, deepcopy(self.row))

    def test_rejects_corrupt_intermediate_snapshot_with_correct_final(self):
        self.row["snapshot_sha256"][0] = "f" * 64
        with self.assertRaisesRegex(ValueError, "snapshot_sha256"):
            validate_sample(self.row, self.case, self.fixture)

    def test_rejects_different_partition_policy(self):
        self.row["partitions"] = 8
        with self.assertRaisesRegex(ValueError, "partitions"):
            validate_sample(self.row, self.case, self.fixture)

    def test_rejects_unsettled_job(self):
        self.row["diagnostics"]["task_count"] = 1
        with self.assertRaisesRegex(ValueError, "lifecycle"):
            validate_sample(self.row, self.case, self.fixture)

    def test_rejects_changed_intermediate_digest_between_versions(self):
        other = deepcopy(self.row)
        other["snapshot_sha256"][0] = "f" * 64
        with self.assertRaisesRegex(ValueError, "snapshot_sha256"):
            compare_oracles(self.row, other)

    def test_rejects_actual_affinity_mismatch(self):
        self.row["process_observations"]["observed_cpu_affinity"] = [1]
        with self.assertRaisesRegex(ValueError, "affinity"):
            validate_sample(self.row, self.case, self.fixture)

    def test_rejects_nonfinite_timing(self):
        self.row["seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "timing"):
            validate_sample(self.row, self.case, self.fixture)


class BuildCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "cargo_lock_sha256": "lock",
            "harness_sha256": "harness",
            "rustc": "compiler",
            "features": ["benchmark-internals"],
            "source_sha256": {
                "crates/calc-flow/examples/support/mod.rs": "shared",
                "crates/calc-flow/examples/support/stock_account_data.rs": "fixture",
                "vendor/arrow-ipc/src/lib.rs": "vendor",
                "crates/calc-flow/src/operator/sql/incremental.rs": "old",
            },
        }

    def test_accepts_product_change_with_shared_harness_dependencies(self):
        candidate = deepcopy(self.base)
        candidate["source_sha256"][
            "crates/calc-flow/src/operator/sql/incremental.rs"
        ] = "new"
        compare_builds({"baseline": self.base, "candidate": candidate})

    def test_rejects_different_shared_fixture_module(self):
        candidate = deepcopy(self.base)
        candidate["source_sha256"][
            "crates/calc-flow/examples/support/stock_account_data.rs"
        ] = "other"
        with self.assertRaisesRegex(ValueError, "shared"):
            compare_builds({"baseline": self.base, "candidate": candidate})

    def test_rejects_changed_vendored_dependency(self):
        candidate = deepcopy(self.base)
        candidate["source_sha256"]["vendor/arrow-ipc/src/lib.rs"] = "other"
        with self.assertRaisesRegex(ValueError, "vendor"):
            compare_builds({"baseline": self.base, "candidate": candidate})

    def test_rejects_different_release_optimization_flags(self):
        candidate = deepcopy(self.base)
        candidate["build_environment"] = {"RUSTFLAGS": "-C target-cpu=native"}
        with self.assertRaisesRegex(ValueError, "RUSTFLAGS"):
            compare_builds({"baseline": self.base, "candidate": candidate})


class ProfileClockTests(unittest.TestCase):
    def test_matches_every_thread_sample_to_one_clock_offset(self):
        raw = " 11/12 100.001000000: \n 11/12 100.002000000: \n"
        profile = {"threads": [{"pid": 11, "tid": 12, "samples": {"time": [0.0, 1.0]}}]}
        result = clock_alignment(raw, profile, 11)
        self.assertEqual(result["matched_sample_count"], 2)
        self.assertEqual(result["offset_ms"], 100001.0)

    def test_rejects_changed_intermediate_profile_timestamp(self):
        raw = " 11/12 100.001000000: \n 11/12 100.002000000: \n"
        profile = {"threads": [{"pid": 11, "tid": 12, "samples": {"time": [0.0, 1.2]}}]}
        with self.assertRaisesRegex(ValueError, "alignment"):
            clock_alignment(raw, profile, 11)


if __name__ == "__main__":
    unittest.main()

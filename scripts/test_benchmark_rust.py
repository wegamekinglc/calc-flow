from __future__ import annotations

import unittest

from scripts.benchmark_suite.legacy import combine_blocks
from scripts.benchmark_suite.rust import (
    _with_fingerprints,
    _workload_fingerprints,
    allocation_rows,
)


def reports():
    case = {
        "name": "one",
        "valid": True,
        "repetitions": [
            {"normalized": {"calls_per_dispatch": 0, "bytes_per_dispatch": 0}}
        ],
    }
    return {
        side: {"role": side, "valid": True, "cases": [case]}
        for side in ("baseline", "candidate")
    }


def provenance_side(scoped: dict[str, str]) -> dict:
    return {
        "machine_fingerprint": "machine",
        "compiled_dependency_fingerprint": "compiled-dependency",
        "scoped_workload_fingerprints": scoped,
    }


def suite_block(workload_fingerprint: str, migration: str | None = None) -> dict:
    metadata = {
        "machine_fingerprint": "machine",
        "dependency_fingerprint": "dependency",
        "workload_fingerprint": workload_fingerprint,
    }
    if migration is not None:
        metadata["workload_migration"] = migration
    return {
        "rows": 10,
        "scope": "native-sql-paired-boundary",
        "metadata": metadata,
        "samples": [1.0],
    }


class BenchmarkRustTests(unittest.TestCase):
    def test_zero_allocation_counts_remain_valid_metric_rows(self):
        rows = allocation_rows(reports())
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["candidate_value"] == 0 for row in rows))

    def test_wrong_role_cannot_be_a_version_comparison(self):
        inputs = reports()
        inputs["candidate"]["role"] = "baseline"
        with self.assertRaises(ValueError):
            allocation_rows(inputs)

    def test_duplicate_allocation_cases_cannot_disappear_in_a_mapping(self):
        inputs = reports()
        inputs["candidate"]["cases"] *= 2
        with self.assertRaises(ValueError):
            allocation_rows(inputs)

    def test_rows_carry_their_own_targets_workload_fingerprint(self):
        identity = provenance_side({"core": "a" * 64, "join": "b" * 64})
        rows = _with_fingerprints({"core/one": {"metadata": {}}}, identity, "a" * 64)
        metadata = rows["core/one"]["metadata"]
        self.assertEqual(metadata["workload_fingerprint"], "a" * 64)
        self.assertEqual(metadata["dependency_fingerprint"], "compiled-dependency")
        self.assertEqual(metadata["machine_fingerprint"], "machine")
        self.assertNotIn("workload_migration", metadata)

    def test_declared_migration_marks_rows_with_their_reference(self):
        identity = provenance_side({"core": "a" * 64})
        rows = _with_fingerprints(
            {"core/one": {"metadata": {}}}, identity, "c" * 64, "DAL-258"
        )
        metadata = rows["core/one"]["metadata"]
        self.assertEqual(metadata["workload_fingerprint"], "c" * 64)
        self.assertEqual(metadata["workload_migration"], "DAL-258")

    def test_declared_migrations_rebaseline_only_their_declared_target(self):
        provenance = {
            "baseline": provenance_side({"core": "a" * 64, "join": "b" * 64}),
            "candidate": provenance_side({"core": "c" * 64, "join": "b" * 64}),
        }
        applied = {"core": {"reference": "DAL-258"}}
        plan = _workload_fingerprints(provenance, applied)
        self.assertEqual(plan["baseline"]["core"], "c" * 64)
        self.assertEqual(plan["candidate"]["core"], "c" * 64)
        self.assertEqual(plan["baseline"]["join"], "b" * 64)
        self.assertEqual(plan["candidate"]["join"], "b" * 64)

    def test_one_changed_bench_source_only_invalidates_its_own_cases(self):
        shard = {"id": "rust", "family": "rust"}
        blocks = {"baseline": [], "candidate": []}
        for side in ("baseline", "candidate", "candidate", "baseline"):
            core_fingerprint = "a" * 64 if side == "baseline" else "c" * 64
            blocks[side].append(
                {
                    "core/case": suite_block(core_fingerprint),
                    "join/case": suite_block("b" * 64),
                }
            )
        cases = {case["scenario"]: case for case in combine_blocks(shard, blocks)}
        self.assertEqual(cases["core/case"]["status"], "error")
        self.assertIn("workload_fingerprint changed", cases["core/case"]["error"])
        self.assertEqual(cases["join/case"]["status"], "ok")

    def test_declared_migration_rows_remain_comparable(self):
        shard = {"id": "rust", "family": "rust"}
        blocks = {"baseline": [], "candidate": []}
        for side in ("baseline", "candidate", "candidate", "baseline"):
            blocks[side].append(
                {
                    "core/case": suite_block("c" * 64, migration="DAL-258"),
                }
            )
        cases = {case["scenario"]: case for case in combine_blocks(shard, blocks)}
        self.assertEqual(cases["core/case"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()

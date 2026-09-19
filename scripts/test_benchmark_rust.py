from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.benchmark_suite.legacy import combine_blocks
from scripts.benchmark_suite.rust import (
    _stamp_fingerprints,
    _with_fingerprints,
    allocation_rows,
    clear_stale_bench_binary,
    sql_rows,
)


def legacy_sql_report() -> dict:
    """A baseline report shape from before the output_rows contract."""

    samples = [70.0] * 20
    engine = {"samples_ms": samples}
    return {
        "schema_version": 1,
        "cases": [
            {
                "name": "sma_20",
                "rows": 1_000_000,
                "window": 20,
                "correctness": {"values": True},
                "calc_flow": engine,
                "raw_datafusion": engine,
            }
        ],
        "environment": {"machine_fingerprint": "machine"},
    }


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
    def test_stale_bench_binaries_are_removed_before_each_build(self):
        with TemporaryDirectory() as directory:
            shared = Path(directory) / "release/deps"
            shared.mkdir(parents=True)
            stale = shared / "sql_datafusion_performance-deadbeef"
            stale.write_bytes(b"stale")
            keep = shared / "core-other"
            keep.write_bytes(b"keep")
            clear_stale_bench_binary(shared.parent.parent, "sql_datafusion_performance")
            self.assertFalse(stale.exists())
            self.assertTrue(keep.exists())

    def test_baseline_sql_rows_read_the_frozen_legacy_contract(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sql.json"
            path.write_text(json.dumps(legacy_sql_report()), encoding="utf-8")
            rows = sql_rows(path, side="baseline")
        self.assertEqual(
            sorted(rows),
            [
                "sql_datafusion_performance/sma_20/calc_flow",
                "sql_datafusion_performance/sma_20/raw_datafusion",
            ],
        )
        row = rows["sql_datafusion_performance/sma_20/calc_flow"]
        self.assertEqual(row["rows"], 1_000_000)
        self.assertEqual(len(row["samples"]), 20)

    def test_candidate_sql_rows_keep_the_strict_verifier_contract(self):
        report = legacy_sql_report()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sql.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                sql_rows(path, side="candidate")

    def test_baseline_sql_rows_reject_incomplete_samples(self):
        report = legacy_sql_report()
        report["cases"][0]["calc_flow"]["samples_ms"] = [70.0] * 19
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sql.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                sql_rows(path, side="baseline")

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
        provenance = {
            "baseline": provenance_side({"core": "a" * 64, "join": "b" * 64}),
            "candidate": provenance_side({"core": "a" * 64, "join": "b" * 64}),
        }
        stamps = _stamp_fingerprints(provenance, {})
        rows = _with_fingerprints(
            {"core/one": {"metadata": {}}}, stamps["baseline"]["core"]
        )
        metadata = rows["core/one"]["metadata"]
        self.assertEqual(metadata["workload_fingerprint"], "a" * 64)
        self.assertEqual(metadata["dependency_fingerprint"], "compiled-dependency")
        self.assertEqual(metadata["machine_fingerprint"], "machine")
        self.assertNotIn("workload_migration", metadata)

    def test_declared_migration_marks_rows_with_their_reference(self):
        provenance = {
            "baseline": provenance_side({"core": "a" * 64}),
            "candidate": provenance_side({"core": "c" * 64}),
        }
        stamps = _stamp_fingerprints(provenance, {"core": {"reference": "DAL-258"}})
        rows = _with_fingerprints(
            {"core/one": {"metadata": {}}}, stamps["baseline"]["core"]
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
        stamps = _stamp_fingerprints(provenance, applied)
        for side in ("baseline", "candidate"):
            self.assertEqual(stamps[side]["core"]["workload_fingerprint"], "c" * 64)
            self.assertEqual(stamps[side]["core"]["workload_migration"], "DAL-258")
            self.assertEqual(stamps[side]["join"]["workload_fingerprint"], "b" * 64)
            self.assertNotIn("workload_migration", stamps[side]["join"])

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

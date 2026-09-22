from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from scripts.benchmark_suite.legacy import combine_blocks
from scripts.benchmark_suite.rust import (
    _stamp_fingerprints,
    _with_fingerprints,
    allocation_rows,
    build_binaries,
    clear_stale_bench_binary,
    measure_rust,
    run_binary,
    sql_rows,
)


class AddedRustTargetTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_source_rebuilds_its_product_library_before_linking(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            shared = root / "target"
            deps = shared / "release/deps"
            deps.mkdir(parents=True)
            library = deps / "libcalc_flow-samehash.rlib"
            metadata = deps / "libcalc_flow-samehash.rmeta"
            dependency = deps / "libdatafusion-dependency.rlib"
            library.write_bytes(b"baseline implementation")
            metadata.write_bytes(b"baseline metadata")
            dependency.write_bytes(b"reusable dependency")
            compiled = []

            async def command(argv, **kwargs):
                if not library.exists():
                    compiled.append(kwargs["cwd"])
                    self.assertFalse(metadata.exists())
                    library.write_bytes(b"candidate implementation")
                target = argv[argv.index("--bench") + 1]
                executable = deps / target
                executable.write_bytes(library.read_bytes())
                kwargs["log"].write_text(
                    json.dumps(
                        {
                            "reason": "compiler-artifact",
                            "target": {"name": target},
                            "executable": str(executable),
                        }
                    )
                )

            with (
                patch(
                    "scripts.benchmark_suite.rust.bench_targets",
                    return_value=["core", "stream_join_materialization"],
                ),
                patch("scripts.benchmark_suite.rust.command", side_effect=command),
            ):
                binaries = await build_binaries(root, root, shared)
            self.assertEqual(compiled, [root])
            self.assertEqual(dependency.read_bytes(), b"reusable dependency")
            for binary in binaries.values():
                self.assertEqual(binary.read_bytes(), b"candidate implementation")

    async def test_removed_target_still_requires_an_explicit_migration(self):
        binaries = [
            {"core": Path("core"), "previous": Path("previous")},
            {"core": Path("core")},
        ]
        with (
            patch(
                "scripts.benchmark_suite.rust.build_binaries",
                AsyncMock(side_effect=binaries),
            ),
            patch("scripts.benchmark_suite.rust._rust_provenance", return_value={}),
            self.assertRaisesRegex(ValueError, "targets removed"),
        ):
            await measure_rust(
                {"id": "rust", "family": "rust"},
                {},
                {"baseline": Path("base"), "candidate": Path("head")},
                Path("target/test"),
            )

    async def test_materialization_collector_retains_memory_and_backpressure_samples(
        self,
    ):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            samples = [
                {
                    **dict.fromkeys(
                        (
                            "allocation_total_bytes",
                            "allocation_count",
                            "rss_before_bytes",
                            "rss_first_emit_bytes",
                            "chunks",
                            "max_chunk_bytes",
                            "queue_high_water_bytes",
                        ),
                        1,
                    ),
                    "rss_available": True,
                    "blocked_seconds": 0.0,
                    "seconds": 0.1,
                    "output_rows": 100_000,
                    "allocation_peak_bytes": 123,
                    "rss_peak_bytes": 456,
                    "blocked_sends": 9,
                }
                for _ in range(20)
            ]
            report = {
                "schema": "calc-flow.join-materialization.v1",
                "scope": "operator-bounded-edge",
                "cases": [
                    {
                        "name": "wide_f100_slow",
                        "config": {"incoming": 1000, "fan": 100},
                        "oracle": {"validated_all_rows": True, "output_rows": 100_000},
                        "samples": samples,
                    }
                ],
            }

            async def command(argv, **_kwargs):
                self.assertIn("--output", argv)
                Path(argv[argv.index("--output") + 1]).write_text(json.dumps(report))

            with patch("scripts.benchmark_suite.rust.command", side_effect=command):
                rows = await run_binary(
                    "stream_join_materialization",
                    root / "binary",
                    root,
                    root,
                    "candidate",
                )
            row = rows["stream_join_materialization/wide_f100_slow"]
            self.assertEqual(row["rows"], 100_000)
            self.assertEqual(row["samples"], [0.1] * 20)
            self.assertEqual(row["scope"], "operator-bounded-edge")
            self.assertEqual(row["metadata"]["observations"], samples)

    async def test_added_target_is_new_coverage_without_fabricating_baseline(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            binary = root / "binary"
            binary.write_bytes(b"fixture")
            binaries = {
                "baseline": {"core": binary},
                "candidate": {"core": binary, "stream_join_materialization": binary},
            }
            row = {
                "rows": 100_000,
                "scope": "operator-bounded-edge",
                "metadata": {},
                "samples": [0.1],
            }

            async def block(_binaries, _source, _output, _stamps, side):
                return (
                    {"stream_join_materialization/wide_f100_slow": row}
                    if side == "candidate"
                    else {},
                    [],
                )

            with (
                patch(
                    "scripts.benchmark_suite.rust.build_binaries",
                    AsyncMock(side_effect=list(binaries.values())),
                ),
                patch("scripts.benchmark_suite.rust._rust_provenance", return_value={}),
                patch(
                    "scripts.benchmark_suite.rust.declared_migrations", return_value={}
                ),
                patch(
                    "scripts.benchmark_suite.rust._stamp_fingerprints",
                    return_value={"baseline": {}, "candidate": {}},
                ),
                patch("scripts.benchmark_suite.rust._rust_block", side_effect=block),
                patch(
                    "scripts.benchmark_suite.rust._allocation_reports",
                    AsyncMock(return_value={}),
                ),
                patch("scripts.benchmark_suite.rust.allocation_rows", return_value=[]),
            ):
                result = await measure_rust(
                    {"id": "rust", "family": "rust"},
                    {},
                    {"baseline": root, "candidate": root},
                    root,
                )
            self.assertEqual(result["errors"], [])
            case = result["cases"][0]
            self.assertEqual(case["comparison"], "new")
            self.assertEqual(case["result"]["verdict"], "new-coverage")
            self.assertEqual(case["baseline"], [])
            self.assertEqual(case["candidate"], [[0.1], [0.1]])


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
        "compiled_dependency_identity": {
            "schema": "calc-flow.compiled-benchmark-dependencies.v1",
            "rustc": "rustc 1.88",
            "cargo": "cargo 1.88",
            "builds": {
                name: [
                    {
                        "target": {"kind": ["bench"], "name": name},
                        "features": [],
                        "profile": {"opt_level": "3"},
                        "package": {"workspace_package": "crates/calc-flow"},
                    },
                    {
                        "target": {"kind": ["lib"], "name": "arrow"},
                        "features": [],
                        "profile": {"opt_level": "3"},
                        "package": {"source": "registry"},
                    },
                ]
                for name in scoped
            },
        },
        "scoped_workload_fingerprints": scoped,
    }


def suite_block(workload_fingerprint: str, migration: str | None = None) -> dict:
    metadata = {
        "machine_fingerprint": "d" * 64,
        "dependency_fingerprint": "e" * 64,
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
        self.assertEqual(len(metadata["dependency_fingerprint"]), 64)
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

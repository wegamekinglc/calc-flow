from __future__ import annotations

import copy
import json
import unittest
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from scripts.benchmark_suite.rust import measure_rust, run_binary
from scripts.benchmark_suite.sql_stream import (
    CASES,
    QUERY,
    expected_output_rows,
    expected_snapshot_digest,
    sql_stream_rows,
)


def report() -> dict:
    cases = []
    for name, (batches, checkpoint, count, unique) in CASES.items():
        sample = {
            "seconds": 0.1,
            "process_seconds": 0.09,
            "prepare_seconds": 0.001 if checkpoint else 0.0,
            "capture_seconds": 0.0001 if checkpoint else 0.0,
            "checkpoint_bytes": 100 * (batches // checkpoint) if checkpoint else 0,
            "checkpoint_count": batches // checkpoint if checkpoint else 0,
            "final_checkpoint_bytes": 100,
            "final_checkpoint_sha256": "0" * 64,
            "output_rows": expected_output_rows(batches, count, unique),
            "snapshot_count": batches,
            "input_rows": count,
            "all_snapshots_sha256": expected_snapshot_digest(batches, count, unique),
            "validated_all_snapshots": True,
            "validated_recovery": False,
        }
        cases.append(
            {
                "name": name,
                "rows": count,
                "batches": batches,
                "maximum_groups": count - (count + 100) // 101 + 1 if unique else 65,
                "unique_keys": unique,
                "checkpoint_every": checkpoint,
                "query": QUERY,
                "oracle": {**sample, "validated_recovery": True},
                "samples": [sample] * 20,
            }
        )
    return {
        "schema": "calc-flow.sql-stream-aggregate.v1",
        "scope": "warm-native-operator-cumulative-snapshots",
        "cases": cases,
    }


@lru_cache(maxsize=2)
def integer_recovery_digest(rows: int, unique: bool) -> str:
    import hashlib

    groups = {}
    combined = hashlib.sha256()
    for row in range(rows + 1):
        key = (row if unique else row % 64) if row % 101 else None
        value = row % 257 - 128 if row % 13 and key != 63 else None
        groups[key] = integer_aggregate(groups.get(key, (0, 0, None, None)), value)
        if row + 1 in (rows, rows + 1):
            combined.update(integer_snapshot_digest(groups))
    return combined.hexdigest()


def integer_aggregate(aggregate, value):
    count, total, low, high = aggregate
    if value is not None:
        count += 1
        total += value
        low = value if low is None else min(low, value)
        high = value if high is None else max(high, value)
    return count, total, low, high


def integer_snapshot_digest(groups):
    import hashlib
    import struct

    digest = hashlib.sha256()
    for key in sorted(groups, key=lambda key: (key is not None, key or 0)):
        count, total, low, high = groups[key]
        digest.update(struct.pack("<Bq", key is not None, key or 0))
        digest.update(struct.pack("<q", count))
        for value in (total if count else None, low, high):
            digest.update(struct.pack("<Bq", value is not None, value or 0))
    return digest.digest()


def input_logical_bytes(batches: int, rows: int, bits: int = 64) -> int:
    chunk = rows // batches
    bitmap_bytes = (chunk + 7) // 8
    return rows * (8 + bits // 8) + sum(
        bitmap_bytes
        * sum(
            (start + divisor - 1) // divisor * divisor < start + chunk
            for divisor in (101, 13)
        )
        for start in range(0, rows, chunk)
    )


def current_report(layout: int = 3) -> dict:
    evidence = copy.deepcopy(report())
    evidence["schema"] = "calc-flow.sql-stream-aggregate.v3"
    for case in evidence["cases"]:
        batches, _, rows, unique = CASES[case["name"]]
        state = current_checkpoint_state(case, layout, batches, rows)
        case["samples"] = [copy.deepcopy(sample) for sample in case["samples"]]
        for sample in [case["oracle"], *case["samples"]]:
            attach_current_checkpoint(sample, state, case["checkpoint_every"])
        case["oracle"]["recovery_snapshots_sha256"] = integer_recovery_digest(
            rows, unique
        )
    return evidence


def current_checkpoint_state(case, layout, batches, rows):
    return {
        "layout": layout,
        "accounting": layout,
        "segment": "group-state" if layout == 3 else "input-retained",
        "rows": case["maximum_groups"] if layout == 3 else rows,
        "columns": 5 if layout == 3 else 2,
        "logical_rows": rows,
        "logical_bytes": input_logical_bytes(batches, rows),
        "segment_ids": [
            "batch-metadata",
            "control",
            "group-state" if layout == 3 else "input-retained",
            "logical-schema",
        ],
        "total_bytes": 200,
        "snapshot_sha256": "1" * 64,
    }


def attach_current_checkpoint(sample, state, checkpoint_every):
    sample["checkpoint_state"] = copy.deepcopy(state)
    if checkpoint_every:
        sample["checkpoint_bytes"] *= 2
    sample["recovery_snapshots_sha256"] = None


class SqlStreamEvidenceTests(unittest.IsolatedAsyncioTestCase):
    def test_current_checkpoint_shapes_preserve_all_prefixes_and_integer_recovery(self):
        for layout in (3, 4):
            evidence = current_report(layout)
            with TemporaryDirectory() as raw, self.subTest(layout=layout):
                path = Path(raw) / "current.json"
                path.write_text(json.dumps(evidence))
                rows = sql_stream_rows(path)
                row = rows["stream_sql_aggregate/fixed_groups_100_batches"]
                self.assertEqual(
                    row["metadata"]["oracle"], evidence["cases"][2]["oracle"]
                )
                self.assertEqual(
                    row["metadata"]["observations"], evidence["cases"][2]["samples"]
                )

    def test_current_decimal_widths_and_signed_scales_keep_typed_recovery(self):
        from scripts.test_benchmark_sql_decimal import decimal_report

        for bits in (32, 64, 128, 256):
            for scale in (-2, 2):
                evidence = decimal_report(bits, scale)
                state_evidence = current_report(4 if scale < 0 else 3)
                evidence["schema"] = state_evidence["schema"]
                for case, state_case in zip(
                    evidence["cases"], state_evidence["cases"], strict=False
                ):
                    batches, _, rows, _ = CASES[case["name"]]
                    state = copy.deepcopy(state_case["oracle"]["checkpoint_state"])
                    state["logical_bytes"] = input_logical_bytes(batches, rows, bits)
                    for sample in [case["oracle"], *case["samples"]]:
                        sample["checkpoint_state"] = copy.deepcopy(state)
                        if case["checkpoint_every"]:
                            sample["checkpoint_bytes"] *= 2
                with TemporaryDirectory() as raw, self.subTest(bits=bits, scale=scale):
                    path = Path(raw) / "decimal-current.json"
                    path.write_text(json.dumps(evidence))
                    rows = sql_stream_rows(path)
                    key = (
                        f"stream_sql_aggregate/decimal{bits}_scale{scale}"
                        "/fixed_groups_100_batches"
                    )
                    self.assertEqual(
                        rows[key]["metadata"]["value_type"], evidence["value_type"]
                    )
                    self.assertEqual(
                        rows[key]["metadata"]["oracle"], evidence["cases"][2]["oracle"]
                    )

    def test_current_checkpoint_census_layout_ledger_inventory_and_bytes_are_strict(
        self,
    ):
        invalid = []
        for field, value in (
            ("layout", 2),
            ("accounting", 4),
            ("segment", "input-retained"),
            ("rows", 64),
            ("columns", 2),
            ("logical_rows", 1),
            ("logical_bytes", True),
            ("logical_bytes", 16 * 100_000),
            ("total_bytes", 100),
            ("snapshot_sha256", "z" * 64),
            ("layout", True),
            ("logical_bytes", -1),
            ("segment_ids", ["input"]),
        ):
            item = current_report()
            item["cases"][0]["oracle"]["checkpoint_state"][field] = value
            invalid.append(item)
        item = current_report(4)
        item["cases"][0]["oracle"]["checkpoint_state"]["rows"] = 65
        invalid.append(item)
        item = current_report()
        del item["cases"][0]["oracle"]["checkpoint_state"]
        invalid.append(item)
        item = current_report()
        item["cases"][3]["oracle"]["checkpoint_bytes"] = 150
        invalid.append(item)
        self.assert_invalid(invalid)

    def test_current_selected_payload_size_is_stable_with_same_arm_checkpoint(self):
        evidence = current_report()
        evidence["cases"][0]["samples"][0]["final_checkpoint_bytes"] += 1
        self.assert_invalid([evidence])

    def test_current_integer_recovery_digest_and_same_arm_checkpoint_census_are_strict(
        self,
    ):
        invalid = []
        for recovery in (None, "0" * 64):
            item = current_report()
            item["cases"][0]["oracle"]["recovery_snapshots_sha256"] = recovery
            invalid.append(item)
        item = current_report()
        item["cases"][0]["samples"][0]["recovery_snapshots_sha256"] = "0" * 64
        invalid.append(item)
        item = current_report()
        item["cases"][0]["samples"][0]["checkpoint_state"]["logical_bytes"] += 1
        invalid.append(item)
        self.assert_invalid(invalid)

    def test_inventory_covers_batch_scaling_and_growing_snapshot_cost(self):
        self.assertEqual(
            set(CASES),
            {
                "fixed_groups_1_batch",
                "fixed_groups_10_batches",
                "fixed_groups_100_batches",
                "fixed_groups_100_batches_checkpoint_10",
                "fixed_groups_100_batches_checkpoint_1",
                "fixed_groups_1000_batches",
                "growing_groups_10k_100_batches",
            },
        )

    def test_original_fixed_input_digests_are_unchanged(self):
        expected = {
            1: "2122b4893bd2e27d5ed9d23394f0caacb016fcd425dcd87fdf74afb57060d843",
            10: "dca2a3268e49d1d52cb4b0dc4883a2d8239f0f3a4affb7ceb9d67d0feb0e142d",
            100: "0f463633face3d426442584df335c8f9679766144626ae0e0d6daa75d04571d5",
        }
        for batches, digest in expected.items():
            self.assertEqual(expected_snapshot_digest(batches), digest)

    def test_growing_output_row_count_matches_all_prefix_group_sets(self):
        groups = set()
        total = 0
        for row in range(10_000):
            groups.add(row if row % 101 else None)
            if (row + 1) % 100 == 0:
                total += len(groups)
        self.assertEqual(expected_output_rows(100, 10_000, True), total)
        self.assertEqual(len(groups), 9901)

    def test_full_inventory_preserves_snapshot_and_checkpoint_evidence(self):
        with TemporaryDirectory() as raw:
            path = Path(raw) / "sql.json"
            evidence = report()
            path.write_text(json.dumps(evidence))
            rows = sql_stream_rows(path)
            self.assertEqual(len(rows), 7)
            row = rows["stream_sql_aggregate/fixed_groups_100_batches_checkpoint_10"]
            self.assertEqual(row["samples"], [0.1] * 20)
            self.assertEqual(row["rows"], 100_000)
            self.assertEqual(
                row["metadata"]["observations"], evidence["cases"][3]["samples"]
            )

    def test_incomplete_duplicate_and_wrong_contract_reports_fail(self):
        invalid = []
        for field, value in (("schema", "bad"), ("scope", "bad"), ("cases", [])):
            invalid.append({**report(), field: value})
        item = report()
        item["cases"].append(item["cases"][0])
        invalid.append(item)
        item = report()
        item["cases"][0]["samples"] = item["cases"][0]["samples"][:19]
        invalid.append(item)
        self.assert_invalid(invalid)

    def test_malformed_timing_snapshot_counts_and_digest_fail(self):
        invalid = []
        for field, value in (
            ("seconds", float("nan")),
            ("seconds", 0),
            ("process_seconds", True),
            ("prepare_seconds", -1),
            ("capture_seconds", float("inf")),
            ("output_rows", 64),
            ("snapshot_count", True),
            ("input_rows", 1),
            ("all_snapshots_sha256", "0" * 64),
            ("validated_all_snapshots", False),
            ("validated_recovery", False),
            ("final_checkpoint_sha256", "z" * 64),
            ("checkpoint_count", 1),
            ("checkpoint_bytes", 100),
        ):
            item = copy.deepcopy(report())
            item["cases"][0]["oracle"][field] = value
            invalid.append(item)
        item = copy.deepcopy(report())
        item["cases"][3]["oracle"]["checkpoint_count"] = 100
        invalid.append(item)
        item = copy.deepcopy(report())
        item["cases"][0]["samples"][0]["process_seconds"] = 0.2
        invalid.append(item)
        self.assert_invalid(invalid)

    def assert_invalid(self, reports):
        with TemporaryDirectory() as raw:
            path = Path(raw) / "sql.json"
            for index, evidence in enumerate(reports):
                with self.subTest(index=index):
                    path.write_text(json.dumps(evidence))
                    with self.assertRaises(ValueError):
                        sql_stream_rows(path)

    async def test_native_dispatch_requests_twenty_samples_and_uses_strict_loader(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)

            async def command(argv, **_kwargs):
                self.assertEqual(argv[argv.index("--samples") + 1], "20")
                Path(argv[argv.index("--output") + 1]).write_text(json.dumps(report()))

            with patch("scripts.benchmark_suite.rust.command", side_effect=command):
                rows = await run_binary(
                    "stream_sql_aggregate", root / "binary", root, root, "candidate"
                )
            self.assertEqual(len(rows), 7)

    async def test_added_target_is_new_coverage_without_fabricating_baseline(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            binary = root / "binary"
            binary.write_bytes(b"fixture")
            binaries = {
                "baseline": {"core": binary},
                "candidate": {"core": binary, "stream_sql_aggregate": binary},
            }
            row = {
                "rows": 100_000,
                "scope": "warm-native-operator-cumulative-snapshots",
                "metadata": {},
                "samples": [0.1],
            }

            async def block(_binaries, _source, _output, _stamps, side):
                return (
                    {"stream_sql_aggregate/fixed_groups_100_batches": row}
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

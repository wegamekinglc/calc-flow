from __future__ import annotations

import copy
import hashlib
import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.benchmark_suite.sql_stream import CASES, sql_stream_rows
from scripts.test_benchmark_sql_stream import report


def decimal_type(bits: int, scale: int = 2) -> dict:
    precision = {32: 4, 64: 12, 128: 28, 256: 60}[bits]
    maximum = {32: 9, 64: 18, 128: 38, 256: 76}[bits]
    return {
        "name": f"decimal{bits}",
        "input": {"bits": bits, "precision": precision, "scale": scale},
        "total": {
            "bits": bits,
            "precision": min(precision + 10, maximum),
            "scale": scale,
        },
    }


def typed_digest(groups: dict, value_type: dict) -> bytes:
    digest = hashlib.sha256()
    for key in sorted(groups, key=lambda item: (item is not None, item or 0)):
        count, total, minimum, maximum = groups[key]
        digest.update(struct.pack("<Bq", key is not None, key or 0))
        digest.update(struct.pack("<q", count))
        for value, field in (
            (total if count else None, "total"),
            (minimum, "input"),
            (maximum, "input"),
        ):
            dtype = value_type[field]
            digest.update(
                struct.pack(
                    "<HBbB",
                    dtype["bits"],
                    dtype["precision"],
                    dtype["scale"],
                    value is not None,
                )
            )
            digest.update(
                (value or 0).to_bytes(dtype["bits"] // 8, "little", signed=True)
            )
    return digest.digest()


def fixture_row(index: int, unique: bool) -> tuple[int | None, int | None]:
    key = (index if unique else index % 64) if index % 101 else None
    value = index % 257 - 128 if index % 13 and key != 63 else None
    return key, value


def fixture_accumulate(state: tuple, value: int | None) -> tuple:
    count, total, minimum, maximum = state
    if value is None:
        return state
    return (
        count + 1,
        total + value,
        value if minimum is None else min(minimum, value),
        value if maximum is None else max(maximum, value),
    )


def decimal_digests(
    batches: int, rows: int, unique: bool, value_type: dict
) -> tuple[str, str]:
    groups = {}
    prefix = hashlib.sha256()
    recovery = hashlib.sha256()
    chunk = rows // batches
    for index in range(rows + 1):
        key, value = fixture_row(index, unique)
        groups[key] = fixture_accumulate(groups.get(key, (0, 0, None, None)), value)
        if index < rows and (index + 1) % chunk == 0:
            prefix.update(typed_digest(groups, value_type))
        if index + 1 in (rows, rows + 1):
            recovery.update(typed_digest(groups, value_type))
    return prefix.hexdigest(), recovery.hexdigest()


def decimal_report(bits: int, scale: int = 2) -> dict:
    evidence = report()
    value_type = decimal_type(bits, scale)
    evidence.update(schema="calc-flow.sql-stream-aggregate.v2", value_type=value_type)
    for case in evidence["cases"]:
        batches, _, rows, unique = CASES[case["name"]]
        prefix, recovery = decimal_digests(batches, rows, unique, value_type)
        case["oracle"].update(
            all_snapshots_sha256=prefix, recovery_snapshots_sha256=recovery
        )
        case["samples"] = [
            {
                **sample,
                "all_snapshots_sha256": prefix,
                "recovery_snapshots_sha256": None,
            }
            for sample in case["samples"]
        ]
    return evidence


class SqlDecimalEvidenceTests(unittest.TestCase):
    def load(self, evidence: dict) -> dict:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "decimal.json"
            path.write_text(json.dumps(evidence))
            return sql_stream_rows(path)

    def test_typed_decimal_inventory_preserves_all_prefixes_and_recovery(self):
        for bits in (32, 64, 128, 256):
            evidence = decimal_report(bits)
            with self.subTest(bits=bits):
                rows = self.load(evidence)
                self.assertEqual(len(rows), 7)
                row = rows[
                    f"stream_sql_aggregate/decimal{bits}_scale2/fixed_groups_100_batches"
                ]
                self.assertEqual(row["metadata"]["value_type"], evidence["value_type"])
                self.assertEqual(
                    row["metadata"]["oracle"], evidence["cases"][2]["oracle"]
                )

    def test_decimal_width_precision_scale_promotion_and_recovery_are_strict(self):
        evidence = decimal_report(256)
        invalid = []
        for field, value in (("bits", True), ("precision", 59), ("scale", 3)):
            item = copy.deepcopy(evidence)
            item["value_type"]["input"][field] = value
            invalid.append(item)
        for field, value in (("bits", 128), ("precision", 76), ("scale", -2)):
            item = copy.deepcopy(evidence)
            item["value_type"]["total"][field] = value
            invalid.append(item)
        item = copy.deepcopy(evidence)
        item["value_type"]["name"] = "decimal128"
        invalid.append(item)
        item = copy.deepcopy(evidence)
        del item["value_type"]
        invalid.append(item)
        for field in ("all_snapshots_sha256", "recovery_snapshots_sha256"):
            item = copy.deepcopy(evidence)
            item["cases"][0]["oracle"][field] = "0" * 64
            invalid.append(item)
        item = copy.deepcopy(evidence)
        item["cases"][0]["samples"][0]["recovery_snapshots_sha256"] = "0" * 64
        invalid.append(item)
        for index, item in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.load(item)

    def test_scale_changes_typed_oracle_without_changing_mathematical_rows(self):
        positive = decimal_report(32)
        negative = decimal_report(32, -2)
        self.assertNotEqual(
            positive["cases"][0]["oracle"]["all_snapshots_sha256"],
            negative["cases"][0]["oracle"]["all_snapshots_sha256"],
        )
        self.assertEqual(len(self.load(negative)), 7)

    def test_valid_type_changes_cannot_reuse_another_typed_digest(self):
        evidence = decimal_report(32)
        for bits, scale in ((64, 2), (128, 2), (256, 2), (32, -2)):
            item = copy.deepcopy(evidence)
            item["value_type"] = decimal_type(bits, scale)
            with self.subTest(bits=bits, scale=scale), self.assertRaises(ValueError):
                self.load(item)

    def test_integer_v1_rejects_undeclared_decimal_type(self):
        evidence = report()
        evidence["value_type"] = decimal_type(32)
        with self.assertRaises(ValueError):
            self.load(evidence)

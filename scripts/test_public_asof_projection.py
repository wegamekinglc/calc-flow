from __future__ import annotations

import asyncio
import unittest

import pyarrow as pa


class AsofProjectionTest(unittest.TestCase):
    def test_public_result_checks_null_tie_tolerance_and_all_columns(self):
        from benchmarks.public_asof_projection import fixture, observe, validate

        rows, probes, history = fixture()
        _, table, status = asyncio.run(observe(rows, probes, history))
        proof = validate(table, status)
        self.assertEqual((proof["matched"], proof["unmatched"]), (764, 260))
        for index, value in [(4, 4), (6, 4), (7, None), (1022, None), (1023, None)]:
            self.assertEqual(
                table["right__time"].cast(pa.int64())[index].as_py(), value
            )
        self.assertEqual(table["right__seq"][4].as_py(), 2)
        broken = table.set_column(5, table.schema.field(5), pa.array([1] * 1024))
        with self.assertRaisesRegex(ValueError, "projection"):
            validate(broken, status)

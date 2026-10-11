from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pyarrow as pa

from benchmarks.sustained_join import (
    BoundedJoinOracle,
    JoinInventory,
    SustainedJoinSink,
    build_input_blocks,
    checkpoint_windows,
)


def expected_block_rows(inventory, block):
    def price(index):
        return 100 + index % 257 / 8 + index % 64 / 8

    rows = []
    for left in range(inventory.block_rows):
        for delta in range(-5, 6):
            right = left + delta * 64
            if 0 <= right < inventory.block_rows:
                rows.append(
                    (
                        block * inventory.block_rows + left,
                        block * inventory.block_rows + right,
                        price(left) * price(right),
                    )
                )
    return rows


def expected_table(inventory, *, blocks=None):
    selected = range(inventory.blocks) if blocks is None else blocks
    rows = []
    for block in selected:
        rows.extend(expected_block_rows(inventory, block))
    return pa.table(
        {
            "sequence": pa.array([row[0] for row in rows], type=pa.uint64()),
            "right_sequence": pa.array([row[1] for row in rows], type=pa.uint64()),
            "value": pa.array([row[2] for row in rows], type=pa.float64()),
        }
    )


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.inventory = JoinInventory(block_rows=640, blocks=2, warmup_blocks=1)
        self.expected = expected_table(self.inventory)

    def test_exact_boundary_multiplicity_and_bounded_allocation(self):
        oracle = BoundedJoinOracle(self.inventory)
        order = np.random.default_rng(406).permutation(self.expected.num_rows)
        table = self.expected.take(pa.array(order))
        for batch in table.to_batches(max_chunksize=173):
            oracle.accept(pa.Table.from_batches([batch]))
        self.assertEqual(oracle.finish()["rows"], self.expected.num_rows)
        self.assertEqual(oracle.bitmap_bytes, 2 * 640 * 11)
        self.assertEqual(oracle.block_counts.tolist(), [5120, 5120])
        self.assertEqual(JoinInventory().expected_rows_per_block, 2_198_080)
        self.assertEqual(JoinInventory().bitmap_bytes, 176_000_000)
        self.assertFalse(hasattr(oracle, "tables"))

    def test_duplicates_within_and_across_delivered_batches_fail(self):
        first = self.expected.slice(0, 1)
        for chunks in ((pa.concat_tables([first, first]),), (first, first)):
            with self.subTest(chunk_count=len(chunks)):
                oracle = BoundedJoinOracle(self.inventory)
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    for chunk in chunks:
                        oracle.accept(chunk)

    def test_missing_output_and_empty_output_do_not_complete(self):
        oracle = BoundedJoinOracle(self.inventory)
        oracle.accept(self.expected.slice(0, 0))
        oracle.accept(self.expected.slice(1))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            oracle.finish()

    def test_invalid_pair_and_value_leave_oracle_unchanged(self):
        valid = self.expected.slice(0, 1)
        mutations = (
            ("right_sequence", 1, "key"),
            ("right_sequence", 384, "time"),
            ("right_sequence", 640, "block"),
            ("sequence", 1280, "range"),
            ("value", float("nan"), "value"),
            ("value", 1.0, "value"),
        )
        for column, value, message in mutations:
            with self.subTest(column=column, value=value):
                oracle = BoundedJoinOracle(self.inventory)
                index = valid.column_names.index(column)
                changed = valid.set_column(
                    index,
                    valid.schema.field(index),
                    pa.array([value], type=valid[column].type),
                )
                with self.assertRaisesRegex(ValueError, message):
                    oracle.accept(changed)
                self.assertEqual(oracle.rows, 0)
                self.assertFalse(oracle.seen.any())

    def test_shifted_block_uses_block_local_price(self):
        table = expected_table(self.inventory, blocks=(1,))
        oracle = BoundedJoinOracle(self.inventory)
        oracle.accept(table)
        first = table.slice(0, 1)
        wrong_global_price = (100 + 640 % 257 / 8 + 640 % 64 / 8) ** 2
        wrong = first.set_column(2, "value", pa.array([wrong_global_price]))
        with self.assertRaisesRegex(ValueError, "value"):
            BoundedJoinOracle(self.inventory).accept(wrong)


class FixtureTests(unittest.TestCase):
    def test_input_blocks_advance_ids_and_time_without_changing_base(self):
        inventory = JoinInventory(block_rows=640, blocks=2, warmup_blocks=1)
        start = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
        base = pa.table(
            {
                "sequence": pa.array(range(640), type=pa.uint64()),
                "event_time": pa.array(
                    [start + timedelta(seconds=index // 64) for index in range(640)],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "symbol": [f"S{index % 64:03}" for index in range(640)],
                "price": [
                    100 + index % 257 / 8 + index % 64 / 8 for index in range(640)
                ],
            }
        )
        original = base.to_pydict()
        first, second = build_input_blocks(base, inventory)
        self.assertEqual(first.to_pydict(), original)
        self.assertEqual(base.to_pydict(), original)
        self.assertEqual(second["sequence"][0].as_py(), 640)
        self.assertEqual(second["sequence"][-1].as_py(), 1279)
        self.assertEqual(second["price"].to_pylist(), base["price"].to_pylist())
        gap = second["event_time"][0].as_py() - first["event_time"][-1].as_py()
        self.assertEqual(gap, timedelta(seconds=7))


class SinkTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_boundary_counts_include_overlapping_later_block_work(self):
        inventory = JoinInventory(block_rows=640, blocks=3, warmup_blocks=1)
        clock = iter(range(0, 100_000, 100))
        sink = SustainedJoinSink(
            inventory,
            snapshot=lambda: {"source_rows": 999},
            clock=lambda: next(clock),
            cpu_clock=lambda: next(clock),
        )
        await sink.open()
        later = expected_table(inventory, blocks=(1,)).slice(0, 10)
        prefix = expected_table(inventory, blocks=(0,))
        for table in (later, prefix):
            await sink.write(SimpleNamespace(to_pyarrow=lambda table=table: table))
        self.assertEqual(sink.steady_start["rows"], 5130)
        self.assertEqual(sink.steady_start["status"], {"source_rows": 999})
        remaining = pa.concat_tables(
            [
                expected_table(inventory, blocks=(1,)).slice(10),
                expected_table(inventory, blocks=(2,)),
            ]
        )
        await sink.write(SimpleNamespace(to_pyarrow=lambda: remaining))
        self.assertTrue(sink.complete.is_set())
        self.assertEqual(sink.steady_end["rows"] - sink.steady_start["rows"], 10230)
        self.assertGreater(sink.validation_wall_ns, 0)
        self.assertGreater(sink.validation_cpu_ns, 0)
        self.assertFalse(hasattr(sink, "tables"))
        await sink.close()


class CheckpointWindowTests(unittest.TestCase):
    def test_sink_commit_does_not_prove_retirement_before_steady_end(self):
        observations = [
            {
                "before_ns": before,
                "after_ns": after,
                "checkpoint": {
                    "current_epoch": current,
                    "elapsed_micros": elapsed,
                    "last_completed_epoch": completed,
                    "terminal": False,
                    "phase": "sinks_committed" if completed == 1 else "requested",
                },
            }
            for before, after, current, elapsed, completed in (
                (50_000, 51_000, 1, 5, None),
                (70_000, 71_000, 1, 25, 1),
                (99_000, 100_000, 1, 54, 1),
                (110_000, 111_000, None, None, 1),
            )
        ]
        window = checkpoint_windows(observations, start_ns=0, end_ns=105_000)[0]
        self.assertFalse(window["inside_steady"])
        self.assertEqual(window["completion_upper_ns"], 111_000)

    def test_missing_active_epoch_cannot_supply_a_duration_or_coverage(self):
        observations = [
            {
                "before_ns": 20_000,
                "after_ns": 21_000,
                "checkpoint": {
                    "current_epoch": None,
                    "elapsed_micros": None,
                    "last_completed_epoch": 3,
                    "terminal": False,
                    "phase": None,
                },
            }
        ]
        windows = checkpoint_windows(observations, start_ns=0, end_ns=100_000)
        self.assertEqual([item["epoch"] for item in windows], [3])
        self.assertFalse(windows[0]["inside_steady"])
        self.assertIsNone(windows[0]["start_lower_ns"])
        self.assertNotIn("duration_upper_ns", windows[0])
        with self.assertRaisesRegex(ValueError, "monotonic"):
            checkpoint_windows(
                [{**observations[0], "before_ns": 22_000}], start_ns=0, end_ns=100_000
            )

    def test_only_wholly_observed_nonterminal_epochs_qualify(self):
        def sample(before, after, checkpoint, terminal=False):
            current, elapsed, completed = checkpoint
            return {
                "before_ns": before,
                "after_ns": after,
                "checkpoint": {
                    "current_epoch": current,
                    "elapsed_micros": elapsed,
                    "last_completed_epoch": completed,
                    "terminal": terminal,
                    "phase": "requested" if current is not None else None,
                },
            }

        observations = [
            sample(20_000, 21_000, (1, 15, None)),
            sample(30_000, 31_000, (None, None, 1)),
            sample(50_000, 51_000, (2, 5, 1)),
            sample(60_000, 61_000, (None, None, 2)),
            sample(70_000, 71_000, (3, 5, 2)),
            sample(99_000, 101_000, (None, None, 3)),
            sample(110_000, 111_000, (4, 5, 3), terminal=True),
            sample(120_000, 121_000, (None, None, 4)),
        ]
        windows = checkpoint_windows(observations, start_ns=40_000, end_ns=100_000)
        self.assertEqual(
            [item["epoch"] for item in windows if item["inside_steady"]], [2]
        )
        accepted = next(item for item in windows if item["epoch"] == 2)
        self.assertLessEqual(accepted["start_lower_ns"], 45_000)
        self.assertGreaterEqual(accepted["start_upper_ns"], 45_000)
        self.assertEqual(accepted["completion_upper_ns"], 61_000)
        self.assertFalse(
            next(item for item in windows if item["epoch"] == 4)["inside_steady"]
        )


if __name__ == "__main__":
    unittest.main()

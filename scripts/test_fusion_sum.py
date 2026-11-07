"""Focused P1b fixture and complete evidence-contract guards."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import pyarrow as pa

from scripts.benchmark_suite import fusion_sum, sum_state


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


class FusionSumTests(unittest.TestCase):
    def test_fixed_selection_contains_only_two_p4_batch_shapes(self):
        self.assertEqual(len(fusion_sum.CASES), 2)
        self.assertEqual(
            [case["batch_size"] for case in fusion_sum.CASES], [1000, 64000]
        )
        for case in fusion_sum.CASES:
            self.assertEqual(case["cpus"], [0, 2, 4, 6])
            self.assertEqual(case["partitions"], 4)
            self.assertEqual(case["mode"], "stream")

    def fixture(self, path):
        return fusion_sum.write_fixture(
            path, business_rows=4, accounts=1, stocks=2, batch_size=3
        )

    def test_fixture_keeps_every_snapshot_and_real_arrow_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.arrow"
            fixture = self.fixture(path)
            self.assertEqual(fixture["input_rows"], 6)
            self.assertEqual(fixture["batch_rows"], [2, 3, 1])
            self.assertEqual(fixture["snapshot_rows"], [2, 2, 2])
            self.assertEqual(
                fixture["snapshot_sha256"],
                [
                    digest(
                        [
                            ["account-000", "stock-000", 0],
                            ["account-000", "stock-001", 0],
                        ]
                    ),
                    digest(
                        [
                            ["account-000", "stock-000", 100],
                            ["account-000", "stock-001", 200],
                        ]
                    ),
                    digest(
                        [
                            ["account-000", "stock-000", 100],
                            ["account-000", "stock-001", 100],
                        ]
                    ),
                ],
            )
            self.assertEqual(fixture["input_sha256"], sum_state.sha(path))
            with pa.ipc.open_stream(path) as reader:
                self.assertEqual(reader.schema, fusion_sum.TRADE_SCHEMA)
                self.assertEqual(
                    [batch.num_rows for batch in reader], fixture["batch_rows"]
                )
            self.assertEqual(
                fixture["schema_sha256"],
                digest(
                    [
                        ["account", "Utf8", True],
                        ["stock", "Utf8", True],
                        ["quantity", "Int64", True],
                        ["side", "Utf8", True],
                        ["price", "Float64", True],
                        ["flow_type", "Utf8", True],
                    ]
                ),
            )

    def test_complete_descriptor_resource_and_status_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(Path(directory) / "tiny.arrow")
        case = fusion_sum.CASES[0]
        row = {
            **sum_state.expected_sample(case, fixture),
            "seconds": 0.01,
            "process_cpu_ns": 9_000_000,
            "start_ns": 100,
            "end_ns": 10_000_100,
            "process_observations": {
                "observed_cpu_affinity": [0, 2, 4, 6],
                "observed_thread_environment": {
                    **{name: "1" for name in sum_state.THREAD_ENV},
                    "TOKIO_WORKER_THREADS": "4",
                },
            },
            "diagnostics": {
                "state": "completed",
                "terminal_cause": "natural_end",
                "task_count": 0,
                "task_errors": 0,
                "metrics_overflowed": False,
                "checkpoint": {"current_epoch": None, "last_completed_epoch": None},
                "sources": {"source": {"errors": 0, "ended": True}},
                "operators": {"sql": {"errors": 0, "ended": True}},
                "sinks": {"sink": {"errors": 0, "ended": True}},
            },
        }
        sum_state.validate_sample(row, case, fixture)
        for field, changed in (
            ("input_rows", 7),
            ("snapshot_rows", [2, 2]),
            ("partitions", 1),
        ):
            broken = {**row, field: changed}
            with self.assertRaises(ValueError):
                sum_state.validate_sample(broken, case, fixture)
        broken = deepcopy(row)
        broken["diagnostics"]["task_count"] = 1
        with self.assertRaisesRegex(ValueError, "lifecycle"):
            sum_state.validate_sample(broken, case, fixture)

    def test_fixture_batch_boundaries_cross_the_example_template_period(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = fusion_sum.write_fixture(
                Path(directory) / "period.arrow",
                business_rows=28,
                accounts=1,
                stocks=2,
                batch_size=16,
            )
        self.assertEqual(fixture["batch_rows"], [2, 16, 12])

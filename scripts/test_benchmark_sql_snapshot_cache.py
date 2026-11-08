from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.benchmark_suite import sql_snapshot_cache as selected


def report(mode="sparse"):
    warm, inputs, outputs, recovery = selected.expected_digests(mode)
    return {
        "schema": selected.SCHEMA,
        "scope": selected.SCOPE,
        "selection": mode,
        "query": selected.QUERY,
        "counts": selected.counts(),
        "warm": {"input_sha256": warm, "sha256": warm, "chunks": [8192, 1808]},
        "appends": [
            {
                "index": index,
                "input_sha256": incoming,
                "output_sha256": output,
                "held_warm_sha256": warm,
                "seconds": 0.001,
            }
            for index, (incoming, output) in enumerate(
                zip(inputs, outputs, strict=True)
            )
        ],
        "process_seconds": 0.02,
        "recovery": {
            "input_sha256": [
                selected.row_digest((), ()),
                selected.row_digest((0,), (1,)),
            ],
            "snapshot_sha256": list(recovery),
            "validated": True,
        },
        "metadata": selected.DEFAULT_METADATA,
        "checkpoint": {
            "layout": 3,
            "groups": 10000,
            "columns": 3,
            "logical_rows": 30000,
            "logical_bytes": 1000000,
            "sha256": "a" * 64,
        },
        "engine_reserved": {"status": "unmeasured"},
        "engine_limit_basis": "default-runtime-known-infinite",
        "warm_seconds": 0.01,
        "validation_seconds": 0.01,
        "checkpoint_seconds": 0.01,
        "recovery_seconds": 0.01,
        "cleanup_seconds": 0.01,
        "total_seconds": 0.08,
        "resources": {
            "schema": "calc-flow.phase-resources.v2",
            "status": "measured",
            "scope": "c1-invocation-before-json",
            "process_cpu_seconds": 0.07,
            "process_cpu_resolution_seconds": 1e-9,
        },
        "observed_cpu_affinity": [0, 2],
        "process_id": 12345,
        "observed_thread_environment": selected.THREAD_ENVIRONMENT,
        "cleanup": True,
    }


class InvocationContractTests(unittest.TestCase):
    def test_one_invocation_has_one_primary_observation_not_twenty_samples(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            path.write_text(json.dumps(report()))
            row = selected.selected_snapshot_row(path)
        self.assertEqual(row["samples"], [0.02])
        self.assertEqual(row["rows"], 20000)
        self.assertEqual(row["metadata"]["input_rows"], 20000)
        self.assertEqual(row["metadata"]["output_rows"], 200000)
        self.assertEqual(row["metadata"]["workload_identity"]["output_rows"], 200000)

    def test_dense_input_reaches_both_chunks(self):
        values = selected.slots("dense")
        self.assertEqual(sum(slot < 8192 for slot in values), 853)
        self.assertEqual(sum(slot >= 8192 for slot in values), 147)

    def test_dense_invocation_is_also_one_observation(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            path.write_text(json.dumps(report("dense")))
            self.assertEqual(selected.selected_snapshot_row(path)["samples"], [0.02])

    def test_full_row_metadata_counts_and_unit_cannot_be_weakened(self):
        golden = report()
        for change in (
            lambda row: row["counts"]["total"].update(input_rows=30000),
            lambda row: row["counts"]["recovery"].update(output_rows=0),
            lambda row: row["metadata"].update(sequence=1),
            lambda row: row.update(appends=row["appends"][:19]),
            lambda row: row["appends"][2].update(input_sha256="0" * 64),
            lambda row: row["appends"][2].update(output_sha256="0" * 64),
            lambda row: row["appends"][2].update(held_warm_sha256="0" * 64),
            lambda row: row["appends"][2].update(index=3),
            lambda row: row["recovery"].update(validated=False),
            lambda row: row["checkpoint"].update(layout=4),
            lambda row: row.update(process_seconds=0.001),
            lambda row: row.update(samples=[0.02] * 20),
            lambda row: row.update(cleanup=False),
            lambda row: row.update(engine_reserved={"bytes": 0}),
            lambda row: row["metadata"].update(sequence=False),
            lambda row: row["counts"]["warm"].update(snapshots=True),
            lambda row: row["recovery"].update(validated=1),
            lambda row: row.update(observed_cpu_affinity=[0, 1]),
        ):
            with self.subTest(change=change):
                row = copy.deepcopy(golden)
                change(row)
                with TemporaryDirectory() as temporary:
                    path = Path(temporary) / "report.json"
                    path.write_text(json.dumps(row))
                    with self.assertRaises(ValueError):
                        selected.selected_snapshot_row(path)

    def test_selected_format_does_not_relax_scheduled_inventory_or_twenty_samples(self):
        from scripts.benchmark_suite.sql_stream import sql_stream_rows
        from scripts.test_benchmark_sql_stream import report as scheduled_report

        for value in (report(), scheduled_report()):
            if value.get("schema") != selected.SCHEMA:
                value["cases"][0]["samples"] = value["cases"][0]["samples"][:19]
            with TemporaryDirectory() as temporary:
                path = Path(temporary) / "report.json"
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    sql_stream_rows(path)


if __name__ == "__main__":
    unittest.main()

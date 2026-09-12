from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pyarrow as pa

from benchmarks.performance_diagnostics import (
    NativeDiagnosticCase,
    SqlDiagnosticCase,
    diagnostic_sql,
    save_output,
)
from scripts import measure_performance_plan as controller
from scripts.measure_performance_plan import (
    _validate_completion,
    _validate_optimized_path,
    compare_outputs,
    load_release,
    measure_round,
)


class PerformancePlanComparisonTests(unittest.TestCase):
    def compare(self, left, right):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                save_output(table, root / name, ordered=True)
                for table, name in ((left, "left.arrow"), (right, "right.arrow"))
            ]
            return compare_outputs(*paths)

    def test_null_is_not_silently_equated_with_nan(self):
        with self.assertRaisesRegex(ValueError, "validity"):
            self.compare(
                pa.table({"value": pa.array([None], type=pa.float64())}),
                pa.table({"value": [float("nan")]}),
            )

    def test_schema_metadata_and_delivered_order_are_compared(self):
        table = pa.table({"sequence": [1, 2], "value": [3.0, 4.0]})
        with self.assertRaisesRegex(ValueError, "schema"):
            self.compare(table, table.replace_schema_metadata({"changed": "yes"}))
        with self.assertRaisesRegex(ValueError, "payload"):
            self.compare(table, table.take([1, 0]))

    def test_finite_tolerance_does_not_relax_infinity_sign(self):
        result = self.compare(
            pa.table({"value": [1.0, None, float("nan"), float("inf")]}),
            pa.table({"value": [1.0 + 1e-12, None, float("nan"), float("inf")]}),
        )
        self.assertTrue(result["passed"])
        with self.assertRaisesRegex(ValueError, "classification"):
            self.compare(
                pa.table({"value": [float("inf")]}),
                pa.table({"value": [float("-inf")]}),
            )

    def test_warm_outputs_use_tolerance_while_price_payload_stays_exact(self):
        for name in ("moving_average", "dual_sma_spread"):
            with self.subTest(name=name):
                result = self.compare(
                    pa.table({name: [1.0], "price": [2.0]}),
                    pa.table({name: [1.0 + 1e-12], "price": [2.0]}),
                )
                self.assertTrue(result["passed"])
        with self.assertRaisesRegex(ValueError, "payload"):
            self.compare(
                pa.table({"moving_average": [1.0], "price": [2.0]}),
                pa.table({"moving_average": [1.0], "price": [2.0 + 1e-12]}),
            )


def read_records(root):
    return [json.loads(line) for line in (root / "raw.jsonl").read_text().splitlines()]


def sample_failure_workers(case, sample):
    workers = {}
    for side in ("baseline", "candidate"):
        worker = AsyncMock()
        worker.request.side_effect = [
            {"native_sha256": side, "tokio_worker_threads": "32"},
            {"case": case, "warmup": sample},
            sample,
            RuntimeError("late sample failure"),
        ]
        workers[side] = worker
    return workers


class PerformancePlanFailureEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_later_sample_retains_earlier_raw_responses(self):
        case = {"id": "sample-fixture"}
        sample = {"seconds": 1.0, "correctness": {"passed": True}}
        workers = sample_failure_workers(case, sample)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    "scripts.measure_performance_plan.Worker.start",
                    side_effect=[workers["baseline"], workers["candidate"]],
                ),
                patch(
                    "scripts.measure_performance_plan._compare_latest",
                    return_value={"passed": True},
                ),
                self.assertRaisesRegex(RuntimeError, "late sample failure"),
            ):
                await measure_round(
                    case,
                    controller.ReleasePair(
                        {side: root / side for side in workers},
                        {side: {"native_sha256": side} for side in workers},
                    ),
                    root,
                    2,
                )
            records = read_records(root)
            measured = [
                entry for entry in records if entry.get("operation") == "sample"
            ]
            self.assertEqual(
                [entry["side"] for entry in measured], ["baseline", "candidate"]
            )
            self.assertEqual(
                [entry["response"]["seconds"] for entry in measured], [1.0, 1.0]
            )
            self.assertIn("late sample failure", records[-1]["error"])
            for worker in workers.values():
                worker.close.assert_awaited_once()

    async def test_terminal_diagnostic_overflow_is_rejected(self):
        case = {"id": "sample-fixture"}
        sample = {"seconds": 1.0, "correctness": {"passed": True}}
        for status in (
            {"metrics_overflowed": True},
            {"rolling_metrics": {"rolling": {"overflowed": True}}},
        ):
            with (
                self.subTest(status=status),
                tempfile.TemporaryDirectory() as directory,
            ):
                workers = []
                for _ in range(2):
                    worker = AsyncMock()
                    worker.request.side_effect = [
                        {"native_sha256": "native", "tokio_worker_threads": "32"},
                        {"case": case, "warmup": sample},
                        {"state": "completed", "after_status": status},
                    ]
                    workers.append(worker)
                with (
                    patch(
                        "scripts.measure_performance_plan.Worker.start",
                        side_effect=workers,
                    ),
                    patch(
                        "scripts.measure_performance_plan._compare_latest",
                        return_value={"passed": True},
                    ),
                    self.assertRaisesRegex(ValueError, "metrics"),
                ):
                    await measure_round(
                        case,
                        controller.ReleasePair(
                            {
                                side: Path(directory) / side
                                for side in ("baseline", "candidate")
                            },
                            {
                                side: {"native_sha256": "native"}
                                for side in ("baseline", "candidate")
                            },
                        ),
                        Path(directory),
                        0,
                    )


class NativeDiagnosticFinalityTests(unittest.TestCase):
    def test_warm_eof_cannot_hide_an_extra_output(self):
        queue = asyncio.Queue()

        async def finish():
            await queue.put(pa.table({"moving_average": [1.0]}))
            return {"state": "completed"}

        diagnostic = NativeDiagnosticCase.__new__(NativeDiagnosticCase)
        diagnostic.loop = asyncio.new_event_loop()
        diagnostic.finished = False
        diagnostic.scenario = SimpleNamespace(
            finish=finish, sink=SimpleNamespace(_tables=queue)
        )
        try:
            with self.assertRaisesRegex(ValueError, "extra output"):
                diagnostic.finish()
        finally:
            diagnostic.loop.close()

    def test_warm_completion_checks_total_sink_delivery(self):
        for rows, expected_valid in ((48, True), (49, False)):
            with self.subTest(rows=rows):
                result = {
                    "state": "completed",
                    "after_status": {
                        "sinks": {
                            "profile": {
                                "delivered_rows": rows,
                                "errors": 0,
                                "ended": True,
                            }
                        }
                    },
                }
                diagnostic = NativeDiagnosticCase.__new__(NativeDiagnosticCase)
                diagnostic.loop = asyncio.new_event_loop()
                diagnostic.finished = False
                diagnostic.scenario = SimpleNamespace(
                    finish=AsyncMock(return_value=result),
                    sink=SimpleNamespace(_tables=asyncio.Queue()),
                    position=48,
                )
                try:
                    if expected_valid:
                        self.assertTrue(
                            diagnostic.finish()["delivery_validation"]["queue_empty"]
                        )
                    else:
                        with self.assertRaisesRegex(ValueError, "delivery"):
                            diagnostic.finish()
                finally:
                    diagnostic.loop.close()

    def test_completed_callbacks_require_settled_outcomes_and_disjoint_stages(self):
        callback = {
            "started": 1,
            "succeeded": 1,
            "failed": 0,
            "cancelled": 0,
            "interrupted": 0,
            "callback_duration_ns": 10,
            "input_validation_duration_ns": 0,
            "ordering_proof_duration_ns": 0,
            "entity_resolution_duration_ns": 0,
            "state_preparation_duration_ns": 0,
            "numeric_update_duration_ns": 0,
            "history_maintenance_duration_ns": 0,
            "arrow_output_duration_ns": 0,
            "budget_preparation_duration_ns": 0,
            "send_wait_duration_ns": 0,
            "other_duration_ns": 10,
        }
        for change in ({}, {"succeeded": 0}, {"other_duration_ns": 9}, {"failed": -1}):
            with self.subTest(change=change):
                node = {
                    "overflowed": False,
                    **{
                        name: {**callback, **change}
                        for name in ("data", "watermark", "end")
                    },
                }
                completion = {
                    "state": "completed",
                    "after_status": {"rolling_metrics": {"r": node}},
                }
                if change:
                    with self.assertRaisesRegex(ValueError, "callback"):
                        _validate_completion(completion)
                else:
                    _validate_completion(completion)


class SqlDiagnosticPathTests(unittest.TestCase):
    def test_full_window_query_must_use_its_declared_rewrite(self):
        case = {"family": "sql-diagnostic", "scenario": "sma20"}
        query = {
            "rolling_rewritten_windows": 2,
            "rolling_fallback_reasons": [],
            "physical_plan": "CalcFlowRollingExec",
        }
        _validate_optimized_path(case, {"datafusion_metrics": [query]})
        with self.assertRaisesRegex(ValueError, "rolling path"):
            _validate_optimized_path(
                case,
                {"datafusion_metrics": [{**query, "rolling_rewritten_windows": 0}]},
            )
        with self.assertRaisesRegex(ValueError, "UInt64"):
            _validate_optimized_path(
                {**case, "scenario": "filter"},
                {
                    "datafusion_metrics": [
                        {"physical_plan": "CAST(sequence AS Decimal128) % 4 = 0"}
                    ]
                },
            )


def expected_core_ids():
    return (
        [
            f"calc-flow-sql/{scenario}/{rows}"
            for rows in (10, 1_000, 1_000_000, 10_000_000)
            for scenario in ("filter", "filter_uint64_modulo", "sma20", "dual_sma")
        ]
        + [
            f"calc-flow-sql/{scenario}/{rows}"
            for rows in (1_000_000, 10_000_000)
            for scenario in ("projection", "group_by", "join")
        ]
        + [
            f"calc-flow-stream/{scenario}/{rows}"
            for rows in (10, 1_000, 100_000, 1_000_000, 10_000_000)
            for scenario in ("sma20", "dual_sma")
        ]
    )


def fallback_case(variant="map_partition_single_batch", scenario="sma20", rows=1_000):
    return {
        "id": f"calc-flow-sql/fallback-{variant}/{scenario}/{rows}",
        "family": "sql-diagnostic",
        "backend": "calc-flow-sql",
        "scenario": scenario,
        "rows": rows,
        "scope": "execute-to-arrow",
        "diagnostic_variant": variant,
        "enable_rolling_rewrite": variant != "rewrite_disabled",
        **(
            {"input_layout": "single_batch", "batch_size": max(8192, rows)}
            if variant == "map_partition_single_batch"
            else {}
        ),
    }


def fallback_reasons(variant, side, count):
    reasons = []
    if variant != "rewrite_disabled":
        reasons = [
            "window_aggregate_is_not_avg"
            if side == "baseline"
            else (
                f"physical_window_shape_not_supported:0_of_{count}"
                if variant == "map_partition_single_batch"
                else "count_filter_distinct_or_null_treatment_is_not_supported"
            )
        ]
    return reasons


def fallback_sample_batches(case):
    rows = case["rows"]
    if case["diagnostic_variant"] == "map_partition_single_batch":
        return [rows]
    return [64_000] * (rows // 64_000) + ([rows % 64_000] if rows % 64_000 else [])


def fallback_sample(case, side="candidate"):
    variant = case["diagnostic_variant"]
    disabled = variant == "rewrite_disabled"
    count = 2 if case["scenario"] == "sma20" else 3
    count += variant == "unsupported_count"
    reasons = fallback_reasons(variant, side, count)
    entities = min(64, max(1, case["rows"] // 40))
    full = case["rows"] - 19 * entities
    return {
        "seconds": 1.0,
        "correctness": {"passed": True},
        "query": diagnostic_sql(case["scenario"], variant),
        "fixture": {
            "input_sha256": "a" * 64,
            "schema_sha256": "c" * 64,
            "rows": case["rows"],
            "entities": entities,
            "full_window_rows": full,
            "batch_rows": fallback_sample_batches(case),
            "native_batch_rows": fallback_sample_batches(case),
            "input_layout": (
                "single_batch"
                if variant == "map_partition_single_batch"
                else "engine_batches"
            ),
        },
        "full_window_validation": {
            "passed": True,
            "full_window_rows": full,
            "null_rows": case["rows"] - full,
        },
        "datafusion_metrics": [
            {
                "configured_batch_size": case.get("batch_size", 8192),
                "configured_target_partitions": 32,
                "requested_target_partitions": 32,
                "rolling_rewrite_enabled": not disabled,
                "rolling_candidate_windows": 0 if disabled else count,
                "rolling_rewritten_windows": 0,
                "rolling_fallback_reasons": reasons,
                "physical_plan": "BoundedWindowAggExec: fixture\n  MemoryExec",
                "logical_plan": "WindowAggr: fixture",
            }
        ],
    }


class SqlFallbackPathTests(unittest.TestCase):
    def test_v2_rejects_old_map_variant(self):
        for scenario in ("sma20", "dual_sma"):
            with self.subTest(scenario=scenario):
                with self.assertRaisesRegex(ValueError, "variant"):
                    diagnostic_sql(scenario, "map_partition")
                case = fallback_case("map_partition", scenario)
                with self.assertRaisesRegex(ValueError, "variant"):
                    _validate_optimized_path(case, fallback_sample(fallback_case()))

    def test_fallback_case_rejects_successful_rewrite(self):
        case = fallback_case()
        sample = fallback_sample(case)
        sample["datafusion_metrics"][0].update(
            rolling_rewritten_windows=2,
            rolling_fallback_reasons=[],
            physical_plan="CalcFlowRollingExec",
        )
        with self.assertRaisesRegex(ValueError, "fallback"):
            _validate_optimized_path(case, sample)

    def test_rewrite_disabled_accepts_zero_audit_and_datafusion_plan(self):
        case = fallback_case("rewrite_disabled")
        _validate_optimized_path(case, fallback_sample(case))
        for changes in (
            {"rolling_rewrite_enabled": True},
            {"rolling_candidate_windows": 2},
            {"rolling_fallback_reasons": ["unexpected"]},
            {"physical_plan": "ProjectionExec"},
        ):
            with self.subTest(changes=changes):
                sample = fallback_sample(case)
                sample["datafusion_metrics"][0].update(changes)
                with self.assertRaisesRegex(ValueError, "fallback"):
                    _validate_optimized_path(case, sample)

    def test_fallback_rejects_wrong_counts_reasons_and_unknown_variants(self):
        for variant in ("map_partition_single_batch", "unsupported_count"):
            case = fallback_case(variant, "dual_sma")
            _validate_optimized_path(case, fallback_sample(case))
            for changes in (
                {"rolling_candidate_windows": 1},
                {"rolling_fallback_reasons": []},
                {"rolling_fallback_reasons": ["unrelated_reason"]},
                {"rolling_rewrite_enabled": False},
            ):
                with self.subTest(variant=variant, changes=changes):
                    sample = fallback_sample(case)
                    sample["datafusion_metrics"][0].update(changes)
                    with self.assertRaisesRegex(ValueError, "fallback"):
                        _validate_optimized_path(case, sample)
        with self.assertRaisesRegex(ValueError, "variant"):
            _validate_optimized_path(
                fallback_case("unknown"), fallback_sample(fallback_case())
            )


class SqlFallbackFactoryTests(unittest.TestCase):
    def setUp(self):
        sequence = list(range(42))
        self.table = pa.table(
            {
                "event_time": pa.array(sequence, type=pa.timestamp("us")),
                "sequence": pa.array(sequence, type=pa.uint64()),
                "symbol": ["S000", "S001"] * 21,
                "price": [2.0] * 42,
            },
            metadata={"fixture": "preserved"},
        )
        self.table = pa.Table.from_batches(self.table.to_batches(max_chunksize=14))

    def make_case(
        self, root, variant, scenario="sma20", *, result=None, roundtrip=None
    ):
        engine = Mock()
        engine.data = SimpleNamespace(table=self.table, entities=2)
        engine.validate.return_value = {"passed": True}
        builder = Mock()
        builder.with_datafusion_config.return_value = builder
        builder.sql.return_value = builder
        if result is not None:
            builder.compile_batch.return_value.execute.return_value = SimpleNamespace(
                outputs={"output": SimpleNamespace(to_pyarrow=lambda: result)},
                datafusion_metrics=[],
            )
        with (
            patch("benchmarks.engine_comparison.EngineCase", return_value=engine),
            patch("calc_flow.PipelineBuilder", return_value=builder),
            patch(
                "calc_flow.Batch.from_pyarrow",
                side_effect=lambda table: SimpleNamespace(
                    to_pyarrow=lambda: table if roundtrip is None else roundtrip(table)
                ),
            ),
        ):
            active = SqlDiagnosticCase(
                fallback_case(variant, scenario, self.table.num_rows), root
            )
        return active, builder

    def test_map_variant_uses_bijective_actual_batch_input_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            active, builder = self.make_case(
                Path(directory), "map_partition_single_batch", "dual_sma"
            )
            actual = active.inputs["input"].to_pyarrow()
            self.assertIn("entity_map", actual.column_names)
            self.assertTrue(pa.types.is_map(actual["entity_map"].type))
            self.assertEqual(
                actual["entity_map"].to_pylist(),
                [[("entity", index % 2)] for index in range(42)],
            )
            self.assertTrue(actual.drop(["entity_map"]).equals(self.table))
            self.assertNotIn("entity_map", self.table.column_names)
            self.assertEqual([batch.num_rows for batch in actual.to_batches()], [42])
            query = builder.sql.call_args.args[1]
            self.assertEqual(query.count("PARTITION BY entity_map"), 2)
            self.assertNotIn("entity_map", query.split("FROM input")[0])
            output = io.BytesIO()
            with pa.ipc.new_stream(output, actual.schema) as writer:
                writer.write_table(actual)
            self.assertEqual(
                active.fixture["input_sha256"],
                hashlib.sha256(output.getvalue()).hexdigest(),
            )
            self.assertEqual(
                active.fixture["schema_sha256"],
                hashlib.sha256(actual.schema.serialize()).hexdigest(),
            )
            self.assertEqual(active.fixture["batch_rows"], [42])
            self.assertEqual(active.fixture["native_batch_rows"], [42])
            self.assertEqual(active.fixture["input_layout"], "single_batch")
            self.assertEqual(
                [batch.num_rows for batch in self.table.to_batches()], [14] * 3
            )
            self.assertTrue(
                actual.schema.equals(
                    self.table.append_column(
                        actual.schema.field("entity_map"), actual["entity_map"]
                    ).schema,
                    check_metadata=True,
                )
            )
            self.assertEqual(active.fixture["entities"], 2)

    def test_distinct_count_variant_preserves_original_complete_window_query(self):
        from benchmarks.engine_comparison import sql_query

        with tempfile.TemporaryDirectory() as directory:
            for scenario in ("sma20", "dual_sma"):
                with self.subTest(scenario=scenario):
                    active, builder = self.make_case(
                        Path(directory), "unsupported_count", scenario
                    )
                    query = builder.sql.call_args.args[1]
                    expected = sql_query(scenario).replace(
                        "COUNT(price) OVER slow = 20 ",
                        "COUNT(price) OVER slow = 20 "
                        "AND COUNT(DISTINCT sequence) OVER slow = 20 ",
                    )
                    self.assertEqual(query, expected)
                    self.assertTrue(
                        active.inputs["input"]
                        .to_pyarrow()
                        .equals(self.table, check_metadata=True)
                    )
                    disabled, _ = self.make_case(
                        Path(directory), "rewrite_disabled", scenario
                    )
                    self.assertEqual(disabled.query, sql_query(scenario))

    def test_only_map_changes_actual_layout_and_requested_batch_size(self):
        import numpy as np

        for rows in (1_000, 1_000_000):
            sequence = np.arange(rows, dtype=np.uint64)
            base = pa.table(
                {
                    "event_time": pa.array(
                        sequence.astype(np.int64), type=pa.timestamp("us")
                    ),
                    "sequence": sequence,
                    "symbol": pa.repeat("S000", rows),
                    "price": pa.repeat(2.0, rows),
                },
                metadata={"fixture": "preserved"},
            )
            self.table = pa.Table.from_batches(base.to_batches(max_chunksize=64_000))
            original_chunks = [batch.num_rows for batch in self.table.to_batches()]
            for variant in (
                "map_partition_single_batch",
                "unsupported_count",
                "rewrite_disabled",
            ):
                with (
                    self.subTest(rows=rows, variant=variant),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    active, builder = self.make_case(Path(directory), variant)
                    is_map = variant == "map_partition_single_batch"
                    expected_chunks = [rows] if is_map else original_chunks
                    actual = active.inputs["input"].to_pyarrow()
                    self.assertEqual(
                        [batch.num_rows for batch in actual.to_batches()],
                        expected_chunks,
                    )
                    self.assertEqual(active.fixture["batch_rows"], expected_chunks)
                    self.assertEqual(
                        active.fixture["native_batch_rows"], expected_chunks
                    )
                    self.assertEqual(
                        builder.with_datafusion_config.call_args.kwargs,
                        {
                            "target_partitions": 32,
                            "batch_size": max(8192, rows) if is_map else 8192,
                            "enable_rolling_rewrite": variant != "rewrite_disabled",
                        },
                    )
                    self.assertTrue(
                        actual.select(base.column_names).equals(
                            base, check_metadata=True
                        )
                    )
                    self.assertEqual(
                        [batch.num_rows for batch in self.table.to_batches()],
                        original_chunks,
                    )
                    self.assertNotIn("entity_map", self.table.column_names)

    def test_native_input_roundtrip_must_preserve_payload_schema_and_layout(self):
        for corruption in ("chunks", "payload", "metadata"):

            def roundtrip(table, change=corruption):
                if change == "chunks":
                    return pa.Table.from_batches(table.to_batches(max_chunksize=14))
                if change == "payload":
                    return table.set_column(
                        3, table.schema.field(3), pa.repeat(99.0, 42)
                    )
                return table.replace_schema_metadata({"corrupted": "yes"})

            with (
                self.subTest(corruption=corruption),
                tempfile.TemporaryDirectory() as directory,
                self.assertRaisesRegex(ValueError, "roundtrip|native input"),
            ):
                self.make_case(
                    Path(directory),
                    "map_partition_single_batch",
                    roundtrip=roundtrip,
                )

    def test_full_window_bitmap_checks_each_entity_nineteenth_and_twentieth_rows(self):
        for variant, scenario, corruption in product(
            ("map_partition_single_batch", "unsupported_count", "rewrite_disabled"),
            ("sma20", "dual_sma"),
            (None, "nan", "global_position", "late_null"),
        ):
            with (
                self.subTest(variant=variant, scenario=scenario, corruption=corruption),
                tempfile.TemporaryDirectory() as directory,
            ):
                values = [None] * 38 + [0.0 if scenario == "dual_sma" else 2.0] * 4
                if corruption == "nan":
                    values[36] = float("nan")  # Entity 0, local row 19.
                elif corruption == "global_position":
                    values[19] = 2.0  # Global row 20 is local row 10.
                elif corruption == "late_null":
                    values[39] = None  # Entity 1, local row 20.
                result = self.table.append_column(
                    "value", pa.array(values, type=pa.float64())
                ).take(list(reversed(range(42))))
                active, _ = self.make_case(
                    Path(directory), variant, scenario, result=result
                )
                with patch(
                    "benchmarks.performance_diagnostics.time.perf_counter_ns",
                    side_effect=[1, 2],
                ):
                    if corruption:
                        with self.assertRaisesRegex(ValueError, "null bitmap"):
                            active.sample()
                    else:
                        sample = active.sample()
                        self.assertEqual(
                            sample["full_window_validation"]["full_window_rows"],
                            4,
                        )
                        self.assertEqual(
                            sample["full_window_validation"]["null_rows"], 38
                        )


class SqlFallbackControllerTests(unittest.IsolatedAsyncioTestCase):
    async def run_round(self, root, case, samples, *, declaration=None):
        workers = []
        for side in ("baseline", "candidate"):
            warmup, sample = samples[side]
            worker = AsyncMock()
            worker.request.side_effect = [
                {"native_sha256": side, "tokio_worker_threads": "32"},
                {"case": case, "warmup": warmup},
                sample,
                {"state": "completed"},
            ]
            workers.append(worker)
        if declaration is None:
            reference = fallback_sample(case)
            declaration = {
                "query": reference["query"],
                "fixture": reference["fixture"],
                "environment": {"tokio_worker_threads": "32"},
                "paths": {
                    side: controller._fallback_path(
                        fallback_sample(case, side)["datafusion_metrics"][0]
                    )
                    for side in ("baseline", "candidate")
                },
            }
        options = {"fallback_declaration": declaration}
        with (
            patch("scripts.measure_performance_plan.Worker.start", side_effect=workers),
            patch(
                "scripts.measure_performance_plan._compare_latest",
                return_value={"passed": True},
            ),
        ):
            return await measure_round(
                case,
                controller.ReleasePair(
                    {side: root / side for side in ("baseline", "candidate")},
                    {
                        side: {"native_sha256": side}
                        for side in ("baseline", "candidate")
                    },
                ),
                root,
                1,
                **options,
            )

    async def test_baseline_fallback_path_failure_is_journaled(self):
        case = fallback_case()
        for phase in ("warmup", "sample"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                samples = {
                    side: [fallback_sample(case, side), fallback_sample(case, side)]
                    for side in ("baseline", "candidate")
                }
                bad = samples["baseline"][phase == "sample"]
                bad["datafusion_metrics"][0]["physical_plan"] = "CalcFlowRollingExec"
                root = Path(directory)
                with self.assertRaisesRegex(ValueError, "baseline.*fallback"):
                    await self.run_round(root, case, samples)
                records = [
                    json.loads(line)
                    for line in (root / "raw.jsonl").read_text().splitlines()
                ]
                record = records[-2]
                self.assertEqual(record["side"], "baseline")
                response = (
                    record["response"]["warmup"]
                    if phase == "warmup"
                    else record["response"]
                )
                self.assertEqual(
                    response["datafusion_metrics"][0]["physical_plan"],
                    "CalcFlowRollingExec",
                )
                self.assertEqual(records[-1]["operation"], "error")

    async def test_real_config_metrics_are_checked_on_both_sides_and_each_phase(self):
        for variant, side, phase, field, wrong in product(
            ("unsupported_count", "rewrite_disabled", "map_partition_single_batch"),
            ("baseline", "candidate"),
            ("warmup", "sample"),
            (
                "configured_batch_size",
                "configured_target_partitions",
                "requested_target_partitions",
            ),
            (None, 1),
        ):
            case = fallback_case(variant, rows=1_000_000)
            with (
                self.subTest(
                    variant=variant,
                    side=side,
                    phase=phase,
                    field=field,
                    wrong=wrong,
                ),
                tempfile.TemporaryDirectory() as directory,
            ):
                samples = {
                    item: [
                        fallback_sample(case, item),
                        fallback_sample(case, item),
                    ]
                    for item in ("baseline", "candidate")
                }
                metric = samples[side][phase == "sample"]["datafusion_metrics"][0]
                if wrong is None:
                    metric.pop(field)
                else:
                    metric[field] = wrong
                root = Path(directory)
                with self.assertRaisesRegex(
                    ValueError, "fallback.*config|config.*fallback"
                ):
                    await self.run_round(root, case, samples)
                records = [
                    json.loads(line)
                    for line in (root / "raw.jsonl").read_text().splitlines()
                ]
                record = records[-2]
                self.assertEqual(record["side"], side)
                response = (
                    record["response"]["warmup"]
                    if phase == "warmup"
                    else record["response"]
                )
                self.assertEqual(response["datafusion_metrics"][0], metric)
                self.assertEqual(records[-1]["operation"], "error")

    async def test_fixture_layout_must_be_complete_before_it_can_be_frozen(self):
        for variant in (
            "unsupported_count",
            "rewrite_disabled",
            "map_partition_single_batch",
        ):
            case = fallback_case(variant, rows=1_000_000)
            for field, changed in (
                ("input_layout", None),
                ("input_layout", "unspecified"),
                ("native_batch_rows", None),
                ("native_batch_rows", [500_000, 500_000]),
                ("batch_rows", [500_000, 500_000]),
            ):
                with self.subTest(variant=variant, field=field, changed=changed):
                    sample = fallback_sample(case)
                    if changed is None:
                        sample["fixture"].pop(field)
                    else:
                        sample["fixture"][field] = changed
                    with self.assertRaisesRegex(ValueError, "fixture|layout"):
                        controller._fallback_fixture(case, sample)

    async def test_fixture_and_query_must_match_peer_and_frozen_declaration(self):
        case = fallback_case("rewrite_disabled")
        for changed in ("peer_input", "later_input", "query", "declaration"):
            with (
                self.subTest(changed=changed),
                tempfile.TemporaryDirectory() as directory,
            ):
                samples = {
                    side: [fallback_sample(case, side), fallback_sample(case, side)]
                    for side in ("baseline", "candidate")
                }
                declaration = None
                if changed == "peer_input":
                    samples["candidate"][0]["fixture"]["input_sha256"] = "b" * 64
                elif changed == "later_input":
                    samples["baseline"][1]["fixture"]["input_sha256"] = "b" * 64
                elif changed == "query":
                    samples["candidate"][1]["query"] = "different query"
                else:
                    original = fallback_sample(case)
                    declaration = {
                        "query": original["query"],
                        "fixture": {**original["fixture"], "input_sha256": "b" * 64},
                        "paths": {},
                    }
                with self.assertRaisesRegex(ValueError, "fixture|query|declaration"):
                    await self.run_round(
                        Path(directory), case, samples, declaration=declaration
                    )

    async def test_allowed_path_attempt_cannot_change_after_preflight(self):
        case = fallback_case()
        samples = {
            side: [fallback_sample(case, side), fallback_sample(case, side)]
            for side in ("baseline", "candidate")
        }
        samples["candidate"][1]["datafusion_metrics"][0]["rolling_fallback_reasons"] = [
            "physical_window_shape_not_supported:1_of_2"
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "candidate.*frozen declaration"),
        ):
            await self.run_round(Path(directory), case, samples)

    async def test_plan_text_may_change_without_changing_structured_path(self):
        case = fallback_case()
        samples = {
            side: [fallback_sample(case, side), fallback_sample(case, side)]
            for side in ("baseline", "candidate")
        }
        for side in samples:
            samples[side][1]["datafusion_metrics"][0]["physical_plan"] += (
                "\n elapsed=42"
            )
        with tempfile.TemporaryDirectory() as directory:
            result = await self.run_round(Path(directory), case, samples)
            self.assertEqual(len(result["samples"]["candidate"]), 1)

    async def test_fallback_selection_preserves_existing_inventory(self):
        expected_core = expected_core_ids()
        expected_sensitivity = [
            "warm/h64000/a1024/e1/b1024/w20/activeNone",
            "warm/h64000/a8192/e64/b8192/w20/activeNone",
            "warm/h64000/a64000/e1000/b64000/w20/activeNone",
            "warm/h1024000/a256000/e64/b256000/w20/activeNone",
            "warm/h64000/a64000/e64/b64000/w1/activeNone",
            "warm/h64000/a64000/e64/b64000/w5/activeNone",
            "warm/h64000/a64000/e64/b64000/w20/activeNone",
            "warm/h64000/a64000/e64/b64000/w512/activeNone",
            "warm/h64000/a1/e64/b64000/w20/active1",
            "warm/h64000/a64/e64/b64000/w20/active1",
            "warm/h1024000/a1/e64/b64000/w20/active1",
            "warm/h1024000/a64/e64/b64000/w20/active1",
            "warm/h1024000/a64000/e64/b64000/w20/activeNone",
            "warm/h64000/a1/e1/b64000/w20/activeNone",
            "warm/h1024000/a64/e64/b64000/w20/activeNone",
        ]
        expected = {
            "core": expected_core,
            "sensitivity": expected_sensitivity,
            "tail": expected_sensitivity[-2:],
            "all": expected_core + expected_sensitivity + expected_sensitivity[-2:],
            "fallback-cost": [
                fallback_case(variant, scenario, rows)["id"]
                for variant, scenario, rows in product(
                    (
                        "map_partition_single_batch",
                        "unsupported_count",
                        "rewrite_disabled",
                    ),
                    ("sma20", "dual_sma"),
                    (1_000, 1_000_000),
                )
            ],
        }
        for group, ids in expected.items():
            with (
                self.subTest(group=group),
                patch("sys.stdout", new_callable=io.StringIO) as output,
            ):
                await controller.run(SimpleNamespace(group=group, case=None, list=True))
                selected = json.loads(output.getvalue())
                self.assertEqual([case["id"] for _, case in selected], ids)
                if group == "fallback-cost":
                    self.assertEqual(
                        [case for _, case in selected],
                        [
                            fallback_case(variant, scenario, rows)
                            for variant, scenario, rows in product(
                                (
                                    "map_partition_single_batch",
                                    "unsupported_count",
                                    "rewrite_disabled",
                                ),
                                ("sma20", "dual_sma"),
                                (1_000, 1_000_000),
                            )
                        ],
                    )


def legacy_map_declaration(declaration):
    altered = copy.deepcopy(declaration)
    for _, old_case in altered["inventory"][:4]:
        old_case["diagnostic_variant"] = "map_partition"
        old_case["id"] = old_case["id"].replace(
            "map_partition_single_batch", "map_partition"
        )
        old_case.pop("input_layout", None)
        old_case.pop("batch_size", None)
    for row in altered["cases"][:4]:
        row["id"] = row["id"].replace("map_partition_single_batch", "map_partition")
    return altered


def altered_declaration(original, change):
    altered = copy.deepcopy(original)
    if change == "incomplete":
        altered["status"] = "incomplete"
    elif change == "inventory":
        altered["inventory"][0][1]["rows"] = 10
    elif change == "instrument":
        altered["diagnostic_harness_sha256"] = {}
    elif change in ("source", "native"):
        altered["releases"]["candidate"][
            {"source": "source_sha", "native": "native_sha256"}[change]
        ] = "changed"
    elif change == "partial":
        altered["cases"].pop()
    elif change == "old_contract":
        altered["contract"] = "calc-flow-sql-fallback-preflight-v1"
    elif change == "old_map":
        return legacy_map_declaration(altered)
    else:
        altered["cases"][1] = altered["cases"][0]
    return altered


class SqlFallbackDeclarationTests(unittest.IsolatedAsyncioTestCase):
    def arguments(self, root, *, preflight=False, declaration=None):
        return SimpleNamespace(
            group="fallback-cost",
            case=None,
            list=False,
            baseline_build=root / "baseline-build.json",
            candidate_build=root / "candidate-build.json",
            root=root,
            preflight=preflight,
            fallback_declaration=declaration,
        )

    def releases(self, root):
        return [
            (
                {
                    "source_sha": side * 40,
                    "native_sha256": side * 64,
                    "wheel_sha256": side * 64,
                    "cargo_lock_sha256": "dependencies",
                    "rustc_verbose": "rustc",
                    "features": "default",
                    "profile": "release",
                },
                root / side,
            )
            for side in ("a", "b")
        ]

    async def preflight(self, root):
        async def prepared(case, _pair, _root, count):
            self.assertEqual(count, 0)
            sample = fallback_sample(case)
            return {
                "fallback_preflight": {
                    "query": sample["query"],
                    "fixture": sample["fixture"],
                    "environment": {"machine": "fixture"},
                    "paths": {
                        side: controller._fallback_path(
                            fallback_sample(case, side)["datafusion_metrics"][0]
                        )
                        for side in ("baseline", "candidate")
                    },
                }
            }

        with (
            patch(
                "scripts.measure_performance_plan.load_release",
                side_effect=self.releases(root),
            ),
            patch(
                "scripts.measure_performance_plan.measure_round", side_effect=prepared
            ) as rounds,
            patch("scripts.measure_performance_plan.measure_case") as measured,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            await controller.run(self.arguments(root, preflight=True))
            self.assertEqual(rounds.await_count, 12)
            measured.assert_not_called()
        return root / "preflight.json"

    async def test_preflight_has_no_measured_pairs_and_binds_formal_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            declaration = await self.preflight(root / "preflight")
            saved = json.loads(declaration.read_text())
            self.assertEqual(saved["status"], "complete")
            self.assertEqual(saved["contract"], "calc-flow-sql-fallback-preflight-v2")
            self.assertNotIn("result", saved["cases"][0])
            self.assertEqual(len(saved["cases"]), 12)
            # Sites can move without changing the sealed release identities.
            with (
                patch(
                    "scripts.measure_performance_plan.load_release",
                    side_effect=self.releases(root / "preflight"),
                ),
                patch(
                    "scripts.measure_performance_plan.measure_case",
                    return_value={"status": "ok"},
                ) as measured,
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                await controller.run(
                    self.arguments(root / "formal", declaration=declaration)
                )
                self.assertEqual(measured.await_count, 12)
                for call, row in zip(
                    measured.call_args_list, saved["cases"], strict=True
                ):
                    self.assertFalse(call.kwargs["tail"])
                    self.assertEqual(
                        call.kwargs["fallback_declaration"], row["fallback_preflight"]
                    )

    async def test_formal_run_rejects_missing_partial_or_changed_declarations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("scripts.measure_performance_plan.load_release") as release:
                with self.assertRaisesRegex(ValueError, "declaration"):
                    await controller.run(self.arguments(root / "missing"))
                release.assert_not_called()
            declaration = await self.preflight(root / "preflight")
            original = json.loads(declaration.read_text())
            for change in (
                "incomplete",
                "inventory",
                "instrument",
                "source",
                "native",
                "partial",
                "duplicate",
                "old_contract",
                "old_map",
            ):
                with self.subTest(change=change):
                    altered = altered_declaration(original, change)
                    changed = root / f"{change}.json"
                    changed.write_text(json.dumps(altered))
                    with (
                        patch(
                            "scripts.measure_performance_plan.load_release",
                            side_effect=self.releases(root / "preflight"),
                        ),
                        patch(
                            "scripts.measure_performance_plan.measure_case"
                        ) as measured,
                        self.assertRaisesRegex(ValueError, "declaration"),
                    ):
                        await controller.run(
                            self.arguments(root / change, declaration=changed)
                        )
                    measured.assert_not_called()

    async def test_formal_run_rejects_missing_case_facts_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            declaration = await self.preflight(root / "preflight")
            original = json.loads(declaration.read_text())
            for change in (
                "null",
                "empty",
                "query",
                "fixture",
                "paths",
                "environment",
                "lane_path",
                "configured_batch_size",
                "configured_target_partitions",
                "requested_target_partitions",
                "input_layout",
                "native_batch_rows",
            ):
                with self.subTest(change=change):
                    altered = copy.deepcopy(original)
                    row = altered["cases"][0]
                    if change == "null":
                        row["fallback_preflight"] = None
                    elif change == "empty":
                        row["fallback_preflight"] = {}
                    elif change == "lane_path":
                        row["fallback_preflight"]["paths"]["baseline"] = None
                    elif change in (
                        "configured_batch_size",
                        "configured_target_partitions",
                        "requested_target_partitions",
                    ):
                        row["fallback_preflight"]["paths"]["baseline"].pop(change, None)
                    elif change in ("input_layout", "native_batch_rows"):
                        row["fallback_preflight"]["fixture"].pop(change)
                    else:
                        row["fallback_preflight"].pop(change)
                    changed = root / f"{change}.json"
                    changed.write_text(json.dumps(altered))
                    with (
                        patch(
                            "scripts.measure_performance_plan.load_release",
                            side_effect=self.releases(root / "preflight"),
                        ),
                        patch(
                            "scripts.measure_performance_plan.measure_case",
                            return_value={"status": "ok"},
                        ) as measured,
                        patch("sys.stdout", new_callable=io.StringIO),
                        self.assertRaisesRegex(ValueError, "declaration"),
                    ):
                        await controller.run(
                            self.arguments(root / change, declaration=changed)
                        )
                    measured.assert_not_called()


class PerformancePlanReleaseTests(unittest.TestCase):
    def manifest(self, root, *, native=b"native", source_sha="a" * 40):
        wheel = root / "test.whl"
        with zipfile.ZipFile(wheel, "w") as output:
            output.writestr("calc_flow/_native.abi3.so", b"native")
        module = root / "site/calc_flow/_native.abi3.so"
        module.parent.mkdir(parents=True)
        module.write_bytes(native)
        manifest = {
            "profile": "release",
            "tracked_source_clean": True,
            "source_sha": source_sha,
            "source_tree": "b" * 40,
            "wheel": str(wheel),
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "native": str(module),
            "native_sha256": hashlib.sha256(native).hexdigest(),
        }
        path = root / "build.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_release_requires_exact_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.manifest(Path(directory), source_sha="HEAD")
            with self.assertRaisesRegex(ValueError, "source"):
                load_release(path)

    def test_extracted_native_must_match_the_recorded_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.manifest(Path(directory), native=b"different build")
            with self.assertRaisesRegex(ValueError, "wheel"):
                load_release(path)

    def test_verified_release_selects_its_private_site(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, site = load_release(self.manifest(root))
            self.assertEqual(manifest["source_sha"], "a" * 40)
            self.assertEqual(site, root / "site")

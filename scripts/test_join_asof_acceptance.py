from __future__ import annotations

import copy
import json
import unittest
from datetime import UTC
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pyarrow as pa


class AcceptanceBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_final_cut_waits_for_output_projection_dispatch_completion(self):
        from benchmarks.join_asof_acceptance import await_asof_frontier

        job = SimpleNamespace(
            status=lambda: {
                "state": "running",
                "stream_asof_joins": {"asof": {"output_watermark_micros": 10}},
            },
            _inner=SimpleNamespace(
                _benchmark_dispatch_frontier=Mock(side_effect=(None, 10))
            ),
        )
        with patch(
            "benchmarks.join_asof_acceptance.asyncio.sleep", new=AsyncMock()
        ) as sleep:
            await await_asof_frontier(job, 10, output_node_id="result")
        sleep.assert_awaited_once_with(0.001)
        self.assertEqual(job._inner._benchmark_dispatch_frontier.call_count, 2)

    async def test_final_frontier_wait_requires_callback_publication(self):
        from benchmarks.join_asof_acceptance import await_asof_frontier

        observations = iter(
            (
                {
                    "state": "running",
                    "stream_asof_joins": {"join": {"output_watermark_micros": 9}},
                },
                {
                    "state": "running",
                    "stream_asof_joins": {"join": {"output_watermark_micros": 10}},
                },
            )
        )
        job = SimpleNamespace(status=lambda: next(observations))
        with patch(
            "benchmarks.join_asof_acceptance.asyncio.sleep", new=AsyncMock()
        ) as sleep:
            status = await await_asof_frontier(job, 10)
        self.assertEqual(
            status["stream_asof_joins"]["join"]["output_watermark_micros"], 10
        )
        sleep.assert_awaited_once_with(0.001)

    async def test_failed_job_cannot_become_a_frontier_wait_timeout(self):
        from benchmarks.join_asof_acceptance import await_asof_frontier

        job = SimpleNamespace(
            status=lambda: {"state": "failed", "stream_asof_joins": {}}
        )
        with self.assertRaisesRegex(RuntimeError, "terminated"):
            await await_asof_frontier(job, 10)

    def test_retention_requires_all_rows_and_no_sealed_idle_or_evicted_side(self):
        from benchmarks.join_asof_acceptance import validate_retained_status

        side = {
            "retained_rows": 100,
            "retained_bytes": 12_800,
            "evicted_rows": 0,
            "watermark_micros": None,
            "idle": False,
            "ended": False,
        }
        status = {
            "left": side,
            "right": {**side, "retained_rows": 64},
            "emitted_match_rows": 100,
        }
        validate_retained_status(status, 100, 64)
        self.assertEqual(status["left"], side)
        for field, value in (
            ("retained_rows", 99),
            ("evicted_rows", 1),
            ("idle", True),
            ("ended", True),
        ):
            changed = copy.deepcopy(status)
            changed["left"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_retained_status(changed, 100, 64)
        changed = copy.deepcopy(status)
        changed["right"]["watermark_micros"] = 10
        with self.assertRaises(ValueError):
            validate_retained_status(changed, 100, 64)


class AcceptanceFixtureTests(unittest.TestCase):
    def test_oracle_rejects_duplicate_missing_and_wrong_values(self):
        from benchmarks.join_asof_acceptance import validate_one_to_one_output

        expected = pa.table(
            {
                "sequence": pa.array([0, 1, 2], type=pa.uint64()),
                "value": [1.0, 2.0, 3.0],
            }
        )
        actual = expected.take(pa.array([2, 0, 1]))
        self.assertEqual(validate_one_to_one_output(actual, expected)["rows"], 3)
        for invalid in (
            expected.take(pa.array([0, 1, 1])),
            expected.slice(0, 2),
            expected.set_column(1, "value", pa.array([1.0, 2.0, 4.0])),
        ):
            with self.assertRaises(ValueError):
                validate_one_to_one_output(invalid, expected)

    def test_oracle_rejects_wrong_arrow_types(self):
        from benchmarks.join_asof_acceptance import validate_one_to_one_output

        expected = pa.table(
            {
                "sequence": pa.array([0, 1], type=pa.uint64()),
                "value": pa.array([1.0, 2.0], type=pa.float64()),
            }
        )
        for actual in (
            expected.set_column(0, "sequence", pa.array([0, 1], type=pa.int64())),
            expected.set_column(1, "value", pa.array([1.0, 2.0], type=pa.float32())),
        ):
            with self.assertRaisesRegex(ValueError, "type"):
                validate_one_to_one_output(actual, expected)

    @staticmethod
    def fixture():
        schema = pa.schema(
            [pa.field("sequence", pa.uint64(), nullable=False)],
            metadata={b"fixture": b"unchanged"},
        )
        batches = [
            pa.record_batch([pa.array(values, type=pa.uint64())], schema=schema)
            for values in ([0, 1], [2], [3])
        ]
        inputs = {name: batches for name in ("reference.input", "quotes.input")}
        events = {"preload": [], "processing": [], "terminal": [], "barriers": []}
        for chunk, indices in enumerate(([0, 1], [2])):
            ids = []
            for kind in ("data", "watermark"):
                for name in inputs:
                    event_id = f"processing/{len(events['processing'])}"
                    events["processing"].append(
                        {
                            "kind": kind,
                            "input": name,
                            "event_id": event_id,
                            "chunk_id": chunk,
                            **(
                                {"batches": indices}
                                if kind == "data"
                                else {"micros": chunk + 1}
                            ),
                        }
                    )
                    ids.append(event_id)
            events["barriers"].append(
                {
                    "chunk_id": chunk,
                    "after_events": ids,
                    "output_rows": 3 if chunk == 0 else 4,
                }
            )
        events["terminal"] = [
            {"kind": "end", "input": name, "event_id": f"terminal/{index}"}
            for index, name in enumerate(inputs)
        ]
        return {
            "project": {"version": 3},
            "inputs": inputs,
            "events": events,
            "expected": pa.Table.from_batches(batches),
            "output_binding": "output",
            "plan_fingerprint": "f" * 64,
            "rows": 4,
            "batch_rows": 3,
            "checkpoint_enabled": False,
        }

    def test_export_preserves_record_batch_groups_and_input_metadata(self):
        from benchmarks.join_asof_acceptance import export_matched_fixture

        fixture = self.fixture()
        untouched = copy.deepcopy(fixture["events"])
        with TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = export_matched_fixture(root, **fixture)
            stored = json.loads((root / manifest["events"]["path"]).read_text())
            self.assertEqual(stored, untouched)
            reader = pa.ipc.open_file(root / manifest["inputs"]["quotes.input"]["path"])
            self.assertEqual(reader.num_record_batches, 3)
            self.assertEqual(reader.schema, fixture["expected"].schema)
            self.assertEqual(manifest["inputs"]["quotes.input"]["rows"], 4)
        self.assertEqual(fixture["events"], untouched)

    def test_fixture_rejects_mislabelled_counts_phase_and_barriers_before_writing(self):
        from benchmarks.join_asof_acceptance import export_matched_fixture

        changes = (
            lambda item: item.update(rows=5),
            lambda item: item["events"]["preload"].append(
                item["events"]["processing"][0]
            ),
            lambda item: item["events"]["processing"][0].update(kind="end"),
            lambda item: item["events"]["barriers"][0].update(output_rows=2),
            lambda item: item["events"]["barriers"].clear(),
            lambda item: item["events"]["processing"][4].update(batches=[0]),
        )
        for change in changes:
            fixture = self.fixture()
            change(fixture)
            with self.subTest(change=change), TemporaryDirectory() as raw:
                root = Path(raw)
                with self.assertRaises(ValueError):
                    export_matched_fixture(root, **fixture)
                self.assertEqual(list(root.iterdir()), [])


class SealedJoinFixtureTests(unittest.TestCase):
    @staticmethod
    def fixture():
        left = [
            pa.record_batch({"sequence": pa.array(values, type=pa.uint64())})
            for values in ([0, 1], [2])
        ]
        right = [pa.record_batch({"factor": [1.0]})]
        events = {
            "preload": [
                {
                    "kind": "data",
                    "input": "right",
                    "event_id": "preload/0",
                    "chunk_id": 0,
                    "batches": [0],
                },
                {
                    "kind": "watermark",
                    "input": "right",
                    "event_id": "preload/1",
                    "chunk_id": 0,
                    "micros": 3,
                },
            ],
            "processing": [
                {
                    "kind": "data",
                    "input": "left",
                    "event_id": "processing/0",
                    "chunk_id": 0,
                    "batches": [0],
                },
                {
                    "kind": "watermark",
                    "input": "left",
                    "event_id": "processing/1",
                    "chunk_id": 0,
                    "micros": 1,
                },
                {
                    "kind": "data",
                    "input": "left",
                    "event_id": "processing/2",
                    "chunk_id": 1,
                    "batches": [1],
                },
                {
                    "kind": "watermark",
                    "input": "left",
                    "event_id": "processing/3",
                    "chunk_id": 1,
                    "micros": 2,
                },
            ],
            "terminal": [
                {"kind": "end", "input": "right", "event_id": "terminal/0"},
                {"kind": "end", "input": "left", "event_id": "terminal/1"},
            ],
            "barriers": [
                {
                    "chunk_id": 0,
                    "after_events": ["processing/0", "processing/1"],
                    "output_rows": 2,
                },
                {
                    "chunk_id": 1,
                    "after_events": ["processing/2", "processing/3"],
                    "output_rows": 3,
                },
            ],
        }
        return {
            "project": {"version": 3},
            "inputs": {"left": left, "right": right},
            "events": events,
            "expected": pa.Table.from_batches(left),
            "output_binding": "output",
            "plan_fingerprint": "f" * 64,
            "rows": 3,
            "batch_rows": 2,
            "checkpoint_enabled": False,
            "scenario": "join_sealed",
        }

    def test_real_arrow_export_preserves_preload_groups_and_terminal_order(self):
        from benchmarks.join_asof_acceptance import export_matched_fixture

        fixture = self.fixture()
        with TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = export_matched_fixture(root, **fixture)
            self.assertEqual(manifest["rows"], 3)
            self.assertEqual(manifest["dimension_rows"], 1)
            self.assertNotIn("rows_per_side", manifest)
            reader = pa.ipc.open_file(root / manifest["inputs"]["left"]["path"])
            self.assertEqual(
                [
                    reader.get_batch(index).num_rows
                    for index in range(reader.num_record_batches)
                ],
                [2, 1],
            )
            events = json.loads((root / "events.json").read_text())
            self.assertEqual(events, fixture["events"])

    def test_malformed_preload_counts_and_barriers_fail_before_writing(self):
        from benchmarks.join_asof_acceptance import export_matched_fixture

        for change in (
            lambda item: item["events"]["preload"].clear(),
            lambda item: item.update(rows=4),
            lambda item: item["events"]["preload"][0].update(batches=[1]),
            lambda item: item["events"]["barriers"][0].update(output_rows=1),
        ):
            fixture = self.fixture()
            change(fixture)
            with TemporaryDirectory() as raw:
                root = Path(raw)
                with self.assertRaises(ValueError):
                    export_matched_fixture(root, **fixture)
                self.assertEqual(list(root.iterdir()), [])


class AcceptanceRuntimePolicyTests(unittest.TestCase):
    def test_ordering_fixture_worker_route_uses_fixed_partial_tail_exporter(self):
        from types import ModuleType

        from benchmarks.join_asof_acceptance import prepare_acceptance_fixture

        comparison = ModuleType("benchmarks.engine_comparison")
        comparison.expected_output = Mock()
        comparison.workload = Mock(side_effect=AssertionError("ordinary workload"))
        root = Path("/tmp/issue406-ordering-route-test")
        case = {
            "workload": "asof_operator",
            "rows": 2049,
            "batch_rows": 1024,
            "mode": "off",
            "ordering_proof": True,
            "fixture_root": str(root),
        }
        with (
            patch.dict(
                "sys.modules",
                {
                    "benchmarks.engine_comparison": comparison,
                    "benchmarks.engine_stream": ModuleType("benchmarks.engine_stream"),
                },
            ),
            patch(
                "benchmarks.join_asof_acceptance.export_ordering_fixture",
                return_value={"rows_per_side": 2049},
            ) as exporter,
            patch("benchmarks.join_asof_acceptance.sha256_file", return_value="a" * 64),
        ):
            result = prepare_acceptance_fixture(case, root)
        exporter.assert_called_once_with(root, checkpoint_enabled=False)
        comparison.workload.assert_not_called()
        self.assertEqual(result["fixture_path"], str(root / "fixture.json"))
        self.assertEqual(result["manifest"], {"rows_per_side": 2049})

    def test_legacy_enabled_modes_do_not_receive_new_disable_keyword(self):
        from datetime import timedelta

        from benchmarks.join_asof_acceptance_runtime import runtime_config_arguments

        self.assertEqual(
            runtime_config_arguments("off"),
            {"checkpoint_interval": timedelta(hours=24), "checkpointing": False},
        )
        self.assertEqual(
            runtime_config_arguments("low"),
            {"checkpoint_interval": timedelta(hours=24)},
        )
        self.assertEqual(
            runtime_config_arguments("on"),
            {"checkpoint_interval": timedelta(milliseconds=100)},
        )


class AcceptanceLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, *, observer_fails=False, asof=False):
        import asyncio
        from datetime import datetime, timedelta
        from types import ModuleType

        from benchmarks.join_asof_acceptance_runtime import measure_acceptance_job

        done = asyncio.Event()
        checkpoint = {
            "current_epoch": None,
            "last_completed_epoch": None,
            "terminal": False,
            "phase": "idle",
            "elapsed_micros": None,
        }
        operator = {
            "input_batches": 2,
            "input_rows": 2,
            "fanned_out_batches": 1,
            "fanned_out_rows": 1,
            "errors": 0,
            "ended": False,
        }
        trace = {
            "schema": "calc-flow.asof-dispatch-trace.v1",
            "job_id": "job",
            "node_id": "asof",
            "compiled_plan_fingerprint": "a" * 64,
            "capacity": 8192,
            "record_count": 4,
            "overflowed": False,
            "allow_barriers": False,
            "terminal_excluded": True,
            "completed_frontiers": {"result": 9},
            "events": [
                {
                    "kind": "data",
                    "ingress": ingress,
                    "ordinal": 0,
                    "rows": 1,
                    "first_sequence": 0,
                    "last_sequence": 0,
                    "record_batch_rows": [1],
                }
                for ingress in ("left", "right")
            ]
            + [
                {"kind": "watermark", "ingress": ingress, "micros": 10}
                for ingress in ("left", "right")
            ],
            "callback_counts": {"data": 2, "ingress_progress": 2, "watermark": 1},
        }
        job = SimpleNamespace(
            status=lambda: {
                "checkpoint": dict(checkpoint),
                "operators": {"asof": operator},
                "stream_asof_joins": {"asof": {"output_watermark_micros": 9}},
            },
            cancel_async=AsyncMock(),
            _inner=SimpleNamespace(
                _benchmark_start_dispatch_trace=Mock(),
                _benchmark_take_dispatch_trace=Mock(return_value=json.dumps(trace)),
            ),
        )

        async def wait():
            await done.wait()
            return SimpleNamespace(state="completed")

        async def push(_):
            done.set()

        async def observe(_job, _observations, stop):
            if observer_fails:
                raise RuntimeError("observer fault")
            await stop.wait()

        async def frontier(_job, micros, *, output_node_id=None):
            self.assertEqual(micros, 9)
            self.assertEqual(output_node_id, "result")
            checkpoint["current_epoch"] = 7
            return {"frontier": micros}

        job.wait_async = wait
        engine = ModuleType("benchmarks.engine_stream")
        engine.BASE = datetime(2000, 1, 1, tzinfo=UTC)
        engine.BASE_MICROS = 0
        engine._require_ready_sources = Mock()
        engine._measure_ready_clock = AsyncMock(
            return_value=(
                object(),
                0.0000001,
                {"processing_started_ns": 100, "processing_ended_ns": 200},
            )
        )
        streams = {
            "quotes.input": (
                SimpleNamespace(at=engine.BASE + timedelta(microseconds=10)),
            )
        }
        ready = asyncio.Event()
        ready.set()
        with (
            patch.dict("sys.modules", {engine.__name__: engine}),
            patch(
                "benchmarks.join_asof_acceptance_runtime._observe_checkpoints", observe
            ),
            patch("benchmarks.join_asof_acceptance_runtime._require_terminal"),
            patch(
                "benchmarks.join_asof_acceptance_runtime.time.perf_counter_ns",
                return_value=350,
            ),
            patch(
                "benchmarks.join_asof_acceptance_runtime.time.process_time_ns",
                side_effect=(100, 200),
            ),
            patch("benchmarks.join_asof_acceptance.await_asof_frontier", frontier),
            patch(
                "benchmarks.join_asof_acceptance_runtime.checkpoint_windows",
                return_value=[],
            ) as windows,
        ):
            try:
                result = await measure_acceptance_job(
                    job,
                    {"source": SimpleNamespace(push=push, ready=ready)},
                    SimpleNamespace(
                        rows=1,
                        expected_rows=1,
                        tables=[pa.table({"value": [1.0]})],
                    ),
                    streams,
                    case={
                        "mode": "low",
                        "workload": "asof" if asof else "projection",
                        "batch_rows": 1024,
                        "side": "candidate",
                        "rows": 1,
                        "dispatch_output_node_id": "result",
                    },
                )
            finally:
                job.cancel_async.assert_awaited_once()
        self.assertEqual(result[1]["job_cpu_seconds"], 100 / 1e9)
        self.assertEqual(result[1]["full_cycle_seconds"], 250 / 1e9)
        if asof:
            job._inner._benchmark_start_dispatch_trace.assert_called_once_with(False)
            job._inner._benchmark_take_dispatch_trace.assert_called_once_with()
        return result[1], windows

    async def test_trace_is_taken_after_settled_timer_and_retains_runtime_data_counts(
        self,
    ):
        evidence, _windows = await self.exercise(asof=True)
        self.assertGreaterEqual(
            evidence["dispatch_trace_capture_started_ns"],
            evidence["settled_processing_ended_ns"],
        )
        self.assertEqual(evidence["dispatch_trace"]["record_count"], 4)
        self.assertEqual(
            evidence["dispatch_trace"]["asof_state"], {"output_watermark_micros": 9}
        )
        self.assertEqual(evidence["dispatch_trace"]["output_record_batch_rows"], [[1]])
        self.assertEqual(
            evidence["dispatch_trace"]["operator_data_counts"]["asof"]["input_rows"],
            2,
        )

    async def test_observer_failure_still_settles_owned_job(self):
        with self.assertRaisesRegex(
            RuntimeError, "observer failed after owned cleanup"
        ):
            await self.exercise(observer_fails=True)

    async def test_checkpoint_activity_during_final_settlement_invalidates_empty_window(
        self,
    ):
        evidence, windows = await self.exercise(asof=True)
        self.assertEqual(evidence["nonterminal_epochs"], [7])
        self.assertEqual(evidence["completed_processing_epochs"], [])
        self.assertFalse(evidence["processing_checkpoint_free"])
        self.assertEqual(windows.call_args.kwargs["end_ns"], 350)
        self.assertTrue(evidence["processing_settled"])


class AcceptanceSignalTests(unittest.IsolatedAsyncioTestCase):
    async def test_signal_cancels_active_sample_cooperatively(self):
        import asyncio
        import signal

        from benchmarks.join_asof_acceptance_runtime import AcceptanceEngineCase

        settled = asyncio.Event()
        started = asyncio.Event()

        async def sample():
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                settled.set()

        owner = object.__new__(AcceptanceEngineCase)
        owner.loop = asyncio.get_running_loop()
        owner.active_task = asyncio.create_task(sample())
        await started.wait()
        owner._signal(signal.SIGTERM, None)
        with self.assertRaises(asyncio.CancelledError):
            await owner.active_task
        self.assertTrue(settled.is_set())
        owner.active_task = None
        with self.assertRaises(SystemExit):
            owner._signal(signal.SIGTERM, None)


class AcceptanceCheckpointEvidenceTests(unittest.TestCase):
    def test_manifest_source_progress_is_retained_without_claiming_missing_epochs(self):
        from benchmarks.join_asof_acceptance_runtime import (
            acceptance_checkpoint_evidence,
        )

        with TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "manifest-1.json").write_text(
                json.dumps(
                    {
                        "sources": {
                            "quotes.input": {
                                "cursor": {"payload": {"rows": 1024}},
                                "ended": False,
                            }
                        }
                    }
                )
            )
            base = {
                "manifests": [
                    {"path": "manifest-1.json", "epoch": 1, "terminal": False}
                ]
            }
            with patch(
                "benchmarks.join_asof_acceptance_runtime.checkpoint_evidence",
                return_value=base,
            ):
                evidence = acceptance_checkpoint_evidence(root, {}, checkpointing=True)
            source = evidence["manifests"][0]["sources"]["quotes.input"]
            self.assertEqual(source["rows"], 1024)
            self.assertEqual(len(source["cursor_sha256"]), 64)
            self.assertNotIn("sources", base["manifests"][0])


if __name__ == "__main__":
    unittest.main()

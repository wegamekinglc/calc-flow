from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.checkpoint_cycles import (
    CycleCallbacks,
    CycleOptions,
    CycleValidationError,
    checkpoint_evidence,
    measure_cycle,
    run_cycle,
)


class Clock:
    def __init__(self):
        self.elapsed = 0

    def __call__(self):
        return self.elapsed

    def advance(self, seconds):
        self.elapsed += seconds * 1_000_000_000


class Sink:
    def __init__(self):
        self.opened = asyncio.Event()
        self.opened.set()
        self.complete = asyncio.Event()
        self.rows = 0
        self.expected_rows = 1
        self.tables = []


class Source:
    def __init__(self, clock, sink):
        self.clock = clock
        self.sink = sink
        self.opened = asyncio.Event()
        self.ready = asyncio.Event()
        self.ended = asyncio.Event()
        self.opened.set()
        self.ready.set()
        self.events = []

    async def push(self, event):
        self.events.append(event)
        if event is None:
            self.clock.advance(3)
            self.ended.set()
        else:
            self.clock.advance(2)
            self.sink.rows = 1
            self.sink.tables.append("batch")
            self.sink.complete.set()


class Job:
    def __init__(self, clock, source):
        self.clock = clock
        self.source = source
        self.cancelled = False
        self.finished = False
        self.extra_rows = 0
        self.state = "completed"
        self.cause = "natural_end"
        self.errors = ()
        self.task_errors = 0
        self.current_epoch = None
        self.failure_category = None

    async def wait_async(self):
        if self.state == "failed":
            return SimpleNamespace(
                state=self.state, cause=self.cause, errors=self.errors
            )
        await self.source.ended.wait()
        self.clock.advance(5)
        self.finished = True
        self.source.sink.rows += self.extra_rows
        return SimpleNamespace(state=self.state, cause=self.cause, errors=self.errors)

    async def cancel_async(self):
        self.clock.advance(11)
        self.cancelled = True

    def status(self):
        return {
            "state": self.state if self.finished else "running",
            "terminal_cause": self.cause if self.finished else None,
            "task_errors": self.task_errors,
            "metrics_overflowed": False,
            "checkpoint": {
                "current_epoch": self.current_epoch,
                "last_completed_epoch": 1 if self.finished else None,
                "failure_category": self.failure_category,
            },
        }


def fixture():
    clock = Clock()
    sink = Sink()
    source = Source(clock, sink)
    job = Job(clock, source)

    def concat(tables):
        if not job.finished:
            raise AssertionError("concat must follow job completion")
        clock.advance(7)
        return tuple(tables)

    return clock, sink, source, job, concat


class ControlledSource(Source):
    def __init__(self, clock, sink):
        super().__init__(clock, sink)
        self.push_entered = asyncio.Event()
        self.push_settled = asyncio.Event()
        self.push_release = asyncio.Event()
        self.push_release.set()

    async def push(self, event):
        self.push_entered.set()
        try:
            await self.push_release.wait()
            await super().push(event)
        finally:
            self.push_settled.set()


class OwnedJob(Job):
    def __init__(self, clock, source):
        super().__init__(clock, source)
        self.wait_entered = asyncio.Event()
        self.wait_settled = asyncio.Event()
        self.completion_release = asyncio.Event()
        self.completion_release.set()
        self.cleanup_entered = asyncio.Event()
        self.cleanup_release = asyncio.Event()
        self.cleanup_release.set()
        self.stop = asyncio.Event()
        self.runtime_task = asyncio.create_task(self.stop.wait())
        self.cleanup_error = False

    async def wait_async(self):
        self.wait_entered.set()
        try:
            await self.source.ended.wait()
            await self.completion_release.wait()
            return await super().wait_async()
        finally:
            self.wait_settled.set()

    async def cancel_async(self):
        self.cleanup_entered.set()
        await self.cleanup_release.wait()
        self.stop.set()
        await self.runtime_task
        await super().cancel_async()
        if self.cleanup_error:
            raise ValueError("cleanup rejected")


def owned_fixture():
    clock, sink = Clock(), Sink()
    source = ControlledSource(clock, sink)
    job = OwnedJob(clock, source)
    operation = measure_cycle(
        {"input": source},
        sink,
        {"input": ("data",)},
        job,
        callbacks=CycleCallbacks(concat=tuple, clock=clock, cpu_clock=clock),
    )
    return source, job, operation


class CycleInterruptionTests(unittest.IsolatedAsyncioTestCase):
    def assert_settled(self, source, job):
        self.assertTrue(job.wait_settled.is_set())
        self.assertTrue(job.cancelled)
        self.assertTrue(job.runtime_task.done())
        if source.push_entered.is_set():
            self.assertTrue(source.push_settled.is_set())

    async def test_cancel_during_feed_preserves_evidence_and_settles_owned_tasks(self):
        source, job, operation = owned_fixture()
        source.push_release.clear()
        task = asyncio.create_task(operation)
        await source.push_entered.wait()
        task.cancel("measurement budget")
        with self.assertRaises(asyncio.CancelledError) as error:
            await task
        self.assertTrue(task.cancelled())
        self.assert_settled(source, job)
        self.assertIn("before_data", error.exception.evidence)
        self.assertEqual(error.exception.evidence["seconds"], 11)
        self.assertTrue(error.exception.evidence["cleanup_completed"])

    async def test_external_timeout_during_completion_preserves_partial_result(self):
        source, job, operation = owned_fixture()
        job.completion_release.clear()
        with self.assertRaises(TimeoutError) as error:
            await asyncio.wait_for(operation, 0.01)
        self.assert_settled(source, job)
        evidence = error.exception.__cause__.evidence
        self.assertEqual(evidence["data_seconds"], 2)
        self.assertEqual(evidence["seconds"], 16)
        self.assertTrue(evidence["cleanup_completed"])

    async def test_ready_failure_keeps_unstarted_timing_and_settles_job(self):
        source, job, operation = owned_fixture()
        source.opened.clear()
        with self.assertRaisesRegex(CycleValidationError, "ready") as error:
            await operation
        self.assert_settled(source, job)
        self.assertIsNone(error.exception.evidence["seconds"])
        self.assertIsNone(error.exception.evidence["cpu_seconds"])
        self.assertTrue(error.exception.evidence["cleanup_completed"])

    async def test_cleanup_error_retains_completed_window_and_invalidates_sample(self):
        source, job, operation = owned_fixture()
        job.cleanup_error = True
        with self.assertRaisesRegex(CycleValidationError, "cleanup rejected") as error:
            await operation
        self.assert_settled(source, job)
        self.assertEqual(error.exception.evidence["completion_seconds"], 10)
        self.assertEqual(error.exception.evidence["seconds"], 21)
        self.assertEqual(error.exception.evidence["cleanup_error"], "cleanup rejected")
        self.assertFalse(error.exception.evidence["cleanup_completed"])

    async def test_cancel_during_cleanup_waits_for_owned_cleanup(self):
        source, job, operation = owned_fixture()
        job.cleanup_release.clear()
        task = asyncio.create_task(operation)
        await job.cleanup_entered.wait()
        task.cancel("measurement budget")
        job.cleanup_release.set()
        with self.assertRaises(asyncio.CancelledError) as error:
            await task
        self.assert_settled(source, job)
        self.assertEqual(error.exception.evidence["seconds"], 21)
        self.assertTrue(error.exception.evidence["cleanup_completed"])


class CheckpointCycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_timer_includes_eof_completion_concat_and_cleanup(self):
        clock, sink, source, job, concat = fixture()

        table, seconds, evidence = await measure_cycle(
            {"input": source},
            sink,
            {"input": ("data",)},
            job,
            callbacks=CycleCallbacks(concat=concat, clock=clock, cpu_clock=clock),
        )

        self.assertEqual(table, ("batch",))
        self.assertEqual(seconds, 28)
        self.assertEqual(evidence["data_seconds"], 2)
        self.assertEqual(evidence["completion_seconds"], 10)
        self.assertEqual(evidence["cpu_seconds"], 28)
        self.assertEqual(source.events, ["data", None])
        self.assertTrue(job.cancelled)

    async def test_rejects_unsettled_or_failed_terminal_status(self):
        for field, value, message in (
            ("cause", "cancelled", "natural_end"),
            ("errors", ("task failed",), "errors"),
            ("task_errors", 1, "task errors"),
            ("current_epoch", 2, "in-flight"),
            ("failure_category", "internal", "checkpoint failure"),
        ):
            with self.subTest(field=field):
                clock, sink, source, job, concat = fixture()
                setattr(job, field, value)
                with self.assertRaisesRegex(CycleValidationError, message) as error:
                    await measure_cycle(
                        {"input": source},
                        sink,
                        {"input": ("data",)},
                        job,
                        callbacks=CycleCallbacks(
                            concat=concat, clock=clock, cpu_clock=clock
                        ),
                    )
                self.assertTrue(job.cancelled)
                self.assertIn("after_eof", error.exception.evidence)
                self.assertGreater(error.exception.evidence["seconds"], 0)

    async def test_rejects_extra_output_delivered_during_eof(self):
        clock, sink, source, job, concat = fixture()
        job.extra_rows = 1
        with self.assertRaisesRegex(CycleValidationError, "row count"):
            await measure_cycle(
                {"input": source},
                sink,
                {"input": ("data",)},
                job,
                callbacks=CycleCallbacks(concat=concat, clock=clock, cpu_clock=clock),
            )
        self.assertTrue(job.cancelled)

    async def test_failed_job_interrupts_blocked_data_and_cleans_up(self):
        clock, sink, source, job, concat = fixture()
        job.state = "failed"
        job.errors = ("checkpoint rejected",)
        with self.assertRaisesRegex(CycleValidationError, "checkpoint rejected"):
            await asyncio.wait_for(
                measure_cycle(
                    {"input": source},
                    sink,
                    {"input": ()},
                    job,
                    callbacks=CycleCallbacks(
                        concat=concat, clock=clock, cpu_clock=clock
                    ),
                ),
                1,
            )
        self.assertTrue(job.cancelled)


class ManifestEvidenceTests(unittest.TestCase):
    def test_committed_nonterminal_nonempty_evidence_excludes_prepared_and_eof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for epoch, ended, rows in ((5, False, 12), (6, True, 0), (7, False, 99)):
                (root / f"manifest-{epoch:020}.json").write_text(
                    json.dumps(
                        {
                            "format_version": 3,
                            "epoch": epoch,
                            "sources": {
                                "left": {"ended": ended},
                                "right": {"ended": ended},
                            },
                            "operators": {
                                "join": {
                                    "inline_metadata": {
                                        "spec": {
                                            "left_keys": ["symbol"],
                                            "right_keys": ["symbol"],
                                        },
                                        "metrics": {
                                            "left": {"retained_rows": rows},
                                            "right": {"retained_rows": rows},
                                        },
                                    }
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            evidence = checkpoint_evidence(
                root, {"last_completed_epoch": 6}, checkpointing=True
            )

        self.assertEqual(evidence["nonterminal_epochs"], [5])
        self.assertEqual(evidence["nonempty_nonterminal_epochs"], [5])
        self.assertEqual(evidence["terminal_epochs"], [6])
        self.assertEqual(evidence["coverage"], "nonempty_nonterminal_observed")
        self.assertEqual(evidence["count_scope"], "retained_committed_manifests_only")
        self.assertEqual(evidence["manifests"][0]["join_retained_rows"], 24)
        self.assertEqual(len(evidence["manifests"][0]["sha256"]), 64)

    def test_disabled_requires_no_epoch_or_checkpoint_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = checkpoint_evidence(
                root, {"last_completed_epoch": None}, checkpointing=False
            )
            self.assertEqual(evidence["coverage"], "disabled")
            with self.assertRaisesRegex(ValueError, "disabled"):
                checkpoint_evidence(
                    root, {"last_completed_epoch": 1}, checkpointing=False
                )
            (root / "unexpected-state").write_bytes(b"state")
            with self.assertRaisesRegex(ValueError, "disabled"):
                checkpoint_evidence(
                    root, {"last_completed_epoch": None}, checkpointing=False
                )

    def test_terminal_only_is_insufficient_checkpoint_on_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest-00000000000000000001.json").write_text(
                json.dumps(
                    {
                        "format_version": 3,
                        "epoch": 1,
                        "sources": {"input": {"ended": True}},
                        "operators": {},
                    }
                ),
                encoding="utf-8",
            )
            evidence = checkpoint_evidence(
                root, {"last_completed_epoch": 1}, checkpointing=True
            )
        self.assertEqual(evidence["coverage"], "insufficient_nonterminal_coverage")


def adapter_fixture():
    _, sink, source, job, concat = fixture()
    calls = {}

    def config(**kwargs):
        calls["config"] = kwargs
        return kwargs

    def managed(root):
        calls["managed"] = root
        return "managed"

    async def start():
        return job

    def runner(plan, sources, sinks, checkpoints, *, config):
        calls["checkpoints"] = checkpoints
        return SimpleNamespace(start_async=start)

    async def inspect_in_test(function, *args, **kwargs):
        return function(*args, **kwargs)

    fake = SimpleNamespace(
        _validated_timed_streams=lambda plan, streams: {
            name: events[:-1] for name, events in streams.items()
        },
        _ReplaySource=lambda events, *, batch_rows: source,
        _CollectSink=lambda expected_rows: sink,
        SourceBinding=lambda *args, **kwargs: "source",
        SourceProvidedWatermarks=lambda: "watermarks",
        SinkBinding=SimpleNamespace(ordinary=lambda *args: "sink"),
        ManagedCheckpointRuntime=managed,
        StreamRuntimeConfig=config,
        EdgeBudget=lambda **kwargs: kwargs,
        StreamingRunner=runner,
        pa=SimpleNamespace(concat_tables=concat),
    )
    return calls, fake, inspect_in_test


class CycleAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_during_manifest_inspection_waits_and_preserves_evidence(self):
        _, fake, _ = adapter_fixture()
        entered = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()

        async def inspect_in_test(function, *args, **kwargs):
            entered.set()
            await release.wait()
            finished.set()
            return {"coverage": "test"}

        with (
            patch.dict(sys.modules, {"benchmarks.engine_stream": fake}),
            patch(
                "benchmarks.checkpoint_cycles.asyncio.to_thread", new=inspect_in_test
            ),
        ):
            task = asyncio.create_task(
                run_cycle(
                    "plan",
                    {"input": ("data", None)},
                    Path("unused"),
                    1,
                    options=CycleOptions(mode="on", batch_rows=8192),
                )
            )
            await entered.wait()
            task.cancel("measurement budget")
            release.set()
            with self.assertRaises(asyncio.CancelledError) as error:
                await task
        self.assertTrue(finished.is_set())
        self.assertGreater(error.exception.evidence["seconds"], 0)
        self.assertTrue(error.exception.evidence["cleanup_completed"])

    async def test_manifest_failure_retains_sample_timing(self):
        _, fake, inspect_in_test = adapter_fixture()
        with (
            patch.dict(sys.modules, {"benchmarks.engine_stream": fake}),
            patch(
                "benchmarks.checkpoint_cycles.checkpoint_evidence",
                side_effect=ValueError("missing terminal manifest"),
            ),
            patch(
                "benchmarks.checkpoint_cycles.asyncio.to_thread", new=inspect_in_test
            ),
            self.assertRaisesRegex(CycleValidationError, "missing terminal") as error,
        ):
            await run_cycle(
                "plan",
                {"input": ("data", None)},
                Path("unused"),
                1,
                options=CycleOptions(mode="on", batch_rows=8192),
            )
        self.assertGreater(error.exception.evidence["seconds"], 0)
        self.assertEqual(error.exception.evidence["after_eof"]["state"], "completed")

    async def test_modes_select_real_runtime_policy_without_recovery(self):
        for mode, interval in (
            ("low", timedelta(hours=24)),
            ("on", timedelta(milliseconds=100)),
            ("off", timedelta(hours=24)),
        ):
            with self.subTest(mode=mode):
                calls, fake, inspect_in_test = adapter_fixture()
                with (
                    patch.dict(sys.modules, {"benchmarks.engine_stream": fake}),
                    patch(
                        "benchmarks.checkpoint_cycles.checkpoint_evidence",
                        return_value={"coverage": "test"},
                    ),
                    patch(
                        "benchmarks.checkpoint_cycles.asyncio.to_thread",
                        new=inspect_in_test,
                    ),
                ):
                    table, seconds, evidence = await run_cycle(
                        "plan",
                        {"input": ("data", None)},
                        Path("unused"),
                        1,
                        options=CycleOptions(mode=mode, batch_rows=8192),
                    )
                self.assertEqual(table, ("batch",))
                self.assertEqual(calls["config"]["checkpoint_interval"], interval)
                self.assertEqual(calls["config"]["edge_budget"]["max_rows"], 8192)
                if mode == "off":
                    self.assertIs(calls["config"]["checkpointing"], False)
                    self.assertIsNone(calls["checkpoints"])
                    self.assertNotIn("managed", calls)
                else:
                    self.assertNotIn("checkpointing", calls["config"])
                    self.assertEqual(calls["checkpoints"], "managed")

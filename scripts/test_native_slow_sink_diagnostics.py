"""Lifecycle and evidence guards for the independent slow-sink instrument."""

from __future__ import annotations

import asyncio
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from benchmarks.native_slow_sink_diagnostics import (
    BoundedSource,
    DelayedSink,
    _StreamHandles,
    _wait_drained,
    backpressure_proof,
    check_terminal,
    finish,
    memory_summary,
    release_after_backpressure,
    release_pair,
    require_release,
    thread_settings,
)


def _edge(**changed) -> dict:
    return {
        "current_envelopes": 0,
        "current_rows": 0,
        "current_bytes": 0,
        "high_water_envelopes": 1,
        "high_water_rows": 64,
        "high_water_bytes": 2048,
        "blocked_sends": 0,
        "blocked_duration_micros": 0,
        "envelope_limit": 256,
        "row_limit": 256,
        "byte_limit": 1 << 20,
        **changed,
    }


def _status(**changed) -> dict:
    return {
        "state": "completed",
        "task_errors": 0,
        "metrics_overflowed": False,
        "watermark_micros": 100,
        "edges": {"sink/output/profile": _edge()},
        "sources": {
            "input": {"data_rows": 64, "data_batches": 1, "errors": 0, "ended": True}
        },
        "operators": {
            "rolling": {"input_rows": 64, "errors": 0, "ended": True},
            "projection": {"input_rows": 64, "errors": 0, "ended": True},
        },
        "sinks": {"profile": {"delivered_rows": 64, "errors": 0, "ended": True}},
        "rolling_metrics": {
            "rolling": {
                "overflowed": False,
                **{
                    name: {
                        "started": 1,
                        "succeeded": 1,
                        "failed": 0,
                        "cancelled": 0,
                        "interrupted": 0,
                        "callback_duration_ns": 10,
                        **{
                            f"{stage}_duration_ns": 1
                            for stage in (
                                "input_validation",
                                "ordering_proof",
                                "entity_resolution",
                                "state_preparation",
                                "numeric_update",
                                "history_maintenance",
                                "arrow_output",
                                "budget_preparation",
                                "send_wait",
                                "other",
                            )
                        },
                    }
                    for name in ("data", "watermark", "end")
                },
            }
        },
        **changed,
    }


class SlowSinkEvidenceTests(unittest.TestCase):
    def test_pair_binds_new_builds_and_rejects_identical_native_or_unmatched_flags(
        self,
    ) -> None:
        candidate = {
            "profile": "release",
            "tracked_source_clean": True,
            "source_sha": "1" * 40,
            "native_sha256": "a" * 64,
            "wheel_sha256": "c" * 64,
            "cargo_lock_sha256": "e" * 64,
            "features": "default",
            "rustc_verbose": "rustc pinned",
            "nested_build_record": {"values": [1]},
        }
        control = {
            **candidate,
            "source_sha": "2" * 40,
            "native_sha256": "b" * 64,
            "wheel_sha256": "d" * 64,
        }
        expected = copy.deepcopy({"candidate": candidate, "control": control})
        self.assertEqual(release_pair("candidate", candidate, control), expected)
        pair = release_pair("control", control, candidate)
        self.assertEqual(pair, expected)
        pair["candidate"]["nested_build_record"]["values"].append(2)
        self.assertEqual({"candidate": candidate, "control": control}, expected)
        for changed in (
            {**control, "native_sha256": candidate["native_sha256"]},
            {**control, "source_sha": "unknown"},
            {**control, "profile": "dev"},
            {**control, "tracked_source_clean": False},
            {**control, "features": "different"},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                release_pair("candidate", candidate, changed)

    def test_thread_settings_preserve_formal_worker_limits(self) -> None:
        expected = {
            "TOKIO_WORKER_THREADS": "32",
            "POLARS_MAX_THREADS": "32",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        original = {**expected, "UNRELATED_VALUE": "keep"}
        self.assertEqual(thread_settings(original), expected)
        self.assertEqual(original, {**expected, "UNRELATED_VALUE": "keep"})
        self.assertEqual(thread_settings({}), expected)
        for changed in (
            {**expected, "TOKIO_WORKER_THREADS": "1"},
            {**expected, "OPENBLAS_NUM_THREADS": "32"},
        ):
            with self.assertRaisesRegex(ValueError, "predeclared diagnostic"):
                thread_settings(changed)

    def test_only_the_declared_distinct_native_releases_are_accepted(self) -> None:
        declared = {
            "candidate": {
                "source_sha": "1" * 40,
                "native_sha256": "a" * 64,
                "wheel_sha256": "c" * 64,
            },
            "control": {
                "source_sha": "2" * 40,
                "native_sha256": "b" * 64,
                "wheel_sha256": "d" * 64,
            },
        }
        original = copy.deepcopy(declared)
        for role, expected in declared.items():
            identity = {
                "native_sha256": expected["native_sha256"],
                "build": dict(expected),
            }
            require_release(role, identity, declared)
            changed = {**identity, "native_sha256": "0" * 64}
            with self.assertRaises(ValueError):
                require_release(role, changed, declared)
            changed = {
                **identity,
                "build": {**identity["build"], "wheel_sha256": "0" * 64},
            }
            with self.assertRaises(ValueError):
                require_release(role, changed, declared)
        stale_control = {
            "native_sha256": declared["candidate"]["native_sha256"],
            "build": dict(declared["control"]),
        }
        with self.assertRaisesRegex(ValueError, "declared release"):
            require_release("control", stale_control, declared)
        self.assertEqual(declared, original)

    def test_native_sink_backpressure_requires_a_new_wait_and_live_charge(self) -> None:
        before = _status()
        held = _status(
            edges={
                "sink/output/profile": _edge(
                    current_envelopes=1,
                    current_rows=256,
                    current_bytes=8192,
                    high_water_rows=256,
                    high_water_bytes=8192,
                    blocked_sends=1,
                )
            }
        )
        original = copy.deepcopy((before, held))
        proof = backpressure_proof(before, held)
        self.assertEqual(proof["sink/output/profile"]["new_blocked_sends"], 1)
        self.assertEqual((before, held), original)
        self.assertFalse(backpressure_proof(held, held))
        self.assertFalse(backpressure_proof(before, _status()))
        source_only = _status(
            edges={"source/input/rolling": held["edges"]["sink/output/profile"]}
        )
        self.assertFalse(backpressure_proof(before, source_only))

    def test_terminal_validation_rejects_budget_delivery_and_error_loss(self) -> None:
        status = _status()
        original = copy.deepcopy(status)
        check_terminal(status, rows=64, batches=1)
        self.assertEqual(status, original)
        mutations = (
            ("edges", "sink/output/profile", "current_rows", 1),
            ("edges", "sink/output/profile", "high_water_rows", 257),
            ("sinks", "profile", "delivered_rows", 63),
            ("sinks", "profile", "ended", False),
            ("sources", "input", "data_batches", 2),
            ("sources", "input", "errors", 1),
            ("operators", "rolling", "input_rows", 63),
            ("operators", "projection", "errors", 1),
        )
        for group, key, field, value in mutations:
            changed = copy.deepcopy(status)
            changed[group][key][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_terminal(changed, rows=64, batches=1)
        for changed in (
            _status(metrics_overflowed=True),
            _status(task_errors=1),
            _status(state="failed"),
        ):
            with self.assertRaises(ValueError):
                check_terminal(changed, rows=64, batches=1)

    def test_sampled_rss_peak_is_distinct_from_lifetime_high_water(self) -> None:
        samples = [
            {"monotonic_ns": 1, "rss_bytes": 100, "lifetime_peak_rss_bytes": 900},
            {"monotonic_ns": 2, "rss_bytes": 200, "lifetime_peak_rss_bytes": 900},
            {"monotonic_ns": 3, "rss_bytes": 120, "lifetime_peak_rss_bytes": 900},
        ]
        original = copy.deepcopy(samples)
        result = memory_summary(samples)
        self.assertEqual(result["sampled_peak_rss_bytes"], 200)
        self.assertEqual(result["process_lifetime_high_water_bytes"], 900)
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(samples, original)
        with self.assertRaises(ValueError):
            memory_summary([])

    def test_terminal_rolling_metrics_require_exact_stage_accounting(self) -> None:
        for field, value in (
            ("started", 2),
            ("succeeded", 2),
            ("failed", 1),
            ("cancelled", 1),
            ("interrupted", 1),
            ("callback_duration_ns", 11),
            ("send_wait_duration_ns", -1),
            ("started", True),
        ):
            status = _status()
            status["rolling_metrics"]["rolling"]["watermark"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_terminal(status, rows=64, batches=1)
        status = _status()
        status["rolling_metrics"]["rolling"]["overflowed"] = True
        with self.assertRaisesRegex(ValueError, "rolling metrics overflowed"):
            check_terminal(status, rows=64, batches=1)


class SlowSinkLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_drain_waits_for_rolling_commit_metric_publication(self) -> None:
        settled = _status(state="running")
        pending = copy.deepcopy(settled)
        callback = pending["rolling_metrics"]["rolling"]["watermark"]
        callback["started"] = 2
        callback["arrow_output_duration_ns"] = 2
        job = SimpleNamespace(status=Mock(side_effect=[pending, settled]))
        source, sink = BoundedSource(), DelayedSink()
        source.waiting.set()
        actual = await _wait_drained(job, source, sink, rows=64, watermark=100)
        self.assertEqual(actual, settled)
        self.assertEqual(job.status.call_count, 2)

    async def test_gate_blocks_inside_write_before_arrow_materialization(self) -> None:
        sink = DelayedSink()
        sink.arm()
        table = SimpleNamespace(num_rows=64, nbytes=2048)
        batch = Mock(to_pyarrow=Mock(return_value=table))
        write = asyncio.create_task(sink.write(batch))
        try:
            await asyncio.wait_for(sink.entered.wait(), 1)
            batch.to_pyarrow.assert_not_called()
            self.assertFalse(write.done())
            self.assertTrue(sink.tables.empty())
            sink.release()
            await asyncio.wait_for(write, 1)
            delivery = await sink.receive()
            self.assertIs(delivery.table, table)
            batch.to_pyarrow.assert_called_once_with()
            self.assertLessEqual(delivery.entered_ns, sink.released_ns)
            self.assertLessEqual(sink.released_ns, delivery.materialized_ns)
        finally:
            sink.release()
            write.cancel()
            await asyncio.gather(write, return_exceptions=True)

    async def test_output_queue_waits_until_prior_table_is_consumed(self) -> None:
        sink = DelayedSink()
        first = SimpleNamespace(num_rows=1, nbytes=8)
        second = SimpleNamespace(num_rows=2, nbytes=16)
        await sink.write(Mock(to_pyarrow=Mock(return_value=first)))
        write = asyncio.create_task(
            sink.write(Mock(to_pyarrow=Mock(return_value=second)))
        )
        try:
            await asyncio.sleep(0)
            self.assertFalse(write.done())
            self.assertEqual(sink.tables.maxsize, 1)
            self.assertIs((await sink.receive()).table, first)
            await asyncio.wait_for(write, 1)
            self.assertIs((await sink.receive()).table, second)
        finally:
            write.cancel()
            await asyncio.gather(write, return_exceptions=True)

    async def test_cancelled_held_write_never_materializes_output(self) -> None:
        sink = DelayedSink()
        sink.arm()
        batch = Mock()
        write = asyncio.create_task(sink.write(batch))
        try:
            await asyncio.wait_for(sink.entered.wait(), 1)
            write.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await write
            batch.to_pyarrow.assert_not_called()
            self.assertEqual(sink.inflight, 0)
            self.assertTrue(sink.tables.empty())
        finally:
            sink.release()
            write.cancel()
            await asyncio.gather(write, return_exceptions=True)

    async def test_source_queue_is_bounded_and_readiness_waits_for_next(self) -> None:
        source = BoundedSource()
        self.assertEqual(source.events.maxsize, 2)
        await source.open(None)
        self.assertFalse(source.waiting.is_set())
        next_event = asyncio.create_task(source.next())
        try:
            await asyncio.wait_for(source.waiting.wait(), 1)
            value = object()
            await source.push(value)
            self.assertIs(await next_event, value)
            self.assertFalse(source.waiting.is_set())
        finally:
            next_event.cancel()
            await asyncio.gather(next_event, return_exceptions=True)

    async def test_backpressure_timeout_opens_gate_and_fails(self) -> None:
        sink = DelayedSink()
        sink.arm()
        batch = Mock(
            to_pyarrow=Mock(return_value=SimpleNamespace(num_rows=1, nbytes=8))
        )
        write = asyncio.create_task(sink.write(batch))
        before = _status(state="running")
        job = SimpleNamespace(status=lambda: before)
        try:
            with self.assertRaises(TimeoutError):
                await release_after_backpressure(
                    _StreamHandles(BoundedSource(), sink, job),
                    before,
                    hold_seconds=0,
                    timeout=0.01,
                )
            self.assertTrue(sink.gate.is_set())
            await asyncio.wait_for(write, 1)
        finally:
            sink.release()
            write.cancel()
            await asyncio.gather(write, return_exceptions=True)

    async def test_gate_release_retains_native_backpressure_evidence(self) -> None:
        sink = DelayedSink()
        sink.arm()
        batch = Mock(
            to_pyarrow=Mock(return_value=SimpleNamespace(num_rows=1, nbytes=8))
        )
        write = asyncio.create_task(sink.write(batch))
        before = _status(state="running")
        held = _status(
            state="running",
            edges={
                "sink/output/profile": _edge(
                    current_envelopes=1,
                    current_rows=256,
                    current_bytes=8192,
                    high_water_rows=256,
                    high_water_bytes=8192,
                    blocked_sends=1,
                )
            },
        )
        job = SimpleNamespace(status=lambda: held)
        try:
            result = await release_after_backpressure(
                _StreamHandles(BoundedSource(), sink, job),
                before,
                hold_seconds=0,
                timeout=1,
            )
            await asyncio.wait_for(write, 1)
            self.assertEqual(result["held_status"], held)
            self.assertEqual(
                result["edges"]["sink/output/profile"]["new_blocked_sends"], 1
            )
            self.assertLessEqual(sink.entered_ns, result["observed_ns"])
            self.assertLessEqual(result["observed_ns"], sink.released_ns)
        finally:
            sink.release()
            write.cancel()
            await asyncio.gather(write, return_exceptions=True)

    async def test_eof_latency_starts_at_enqueue_and_reports_source_poll(self) -> None:
        ticks = iter((10, 20, 30, 40))

        def clock():
            return next(ticks)

        source, sink = BoundedSource(clock=clock), DelayedSink()

        async def wait():
            self.assertIsNone(await source.next())
            await sink.close()
            return SimpleNamespace(state="completed", errors=())

        job = SimpleNamespace(wait_async=wait, status=_status)
        result = await finish(
            _StreamHandles(source, sink, job), rows=64, batches=1, clock=clock
        )
        self.assertEqual(result["eof_enqueue_started_ns"], 10)
        self.assertEqual(result["eof_enqueue_completed_ns"], 20)
        self.assertEqual(result["eof_source_polled_ns"], 30)
        self.assertEqual(result["job_completed_ns"], 40)
        self.assertEqual(result["eof_enqueue_to_completed_ns"], 30)
        self.assertEqual(result["eof_poll_to_completed_ns"], 10)

    async def test_eof_drains_bounded_sink_and_rejects_extra_rows(self) -> None:
        source, sink = BoundedSource(), DelayedSink()

        async def wait():
            self.assertIsNone(await source.next())
            table = SimpleNamespace(num_rows=1, nbytes=8)
            for _ in range(3):
                await sink.write(Mock(to_pyarrow=Mock(return_value=table)))
            await sink.close()
            return SimpleNamespace(state="completed", errors=())

        job = SimpleNamespace(wait_async=wait, status=_status)
        with self.assertRaisesRegex(ValueError, "unexpected terminal output"):
            await asyncio.wait_for(
                finish(_StreamHandles(source, sink, job), rows=64, batches=1), 1
            )
        self.assertTrue(sink.tables.empty())


if __name__ == "__main__":
    unittest.main()

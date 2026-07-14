from __future__ import annotations

import asyncio
import gc
import subprocess
import sys
import textwrap
import weakref

import numpy as np
import pyarrow as pa
import pytest

from calc_flow import (
    Batch,
    CheckpointError,
    ConfigError,
    ExecutionError,
    FileCheckpointStore,
    MicroBatchRunner,
    PipelineBuilder,
    ProviderError,
    Runtime,
    StreamingRunner,
    register_numpy,
)


def _batch(value: int) -> Batch:
    return Batch.from_pyarrow(pa.table({"value": [value]}))


def _plan(name: str = "stream"):
    return PipelineBuilder(name).expression("calc", "result = value + 1").compile()


def test_execution_plan_lifecycle_is_async_defensive_and_guarded() -> None:
    plan = _plan("lifecycle")

    async def exercise() -> None:
        state = await plan.snapshot_async()
        state["calc"] = {"changed": True}
        assert await plan.snapshot_async() == {"calc": None}
        await plan.restore_async({"calc": None})
        await plan.reset_async()
        with pytest.raises(RuntimeError, match="await snapshot_async"):
            plan.snapshot()

    asyncio.run(exercise())
    assert plan.snapshot() == {"calc": None}
    plan.restore({"calc": None})
    plan.reset()


def test_streaming_runner_supports_sync_and_async_sinks_and_checkpointing(
    tmp_path,
) -> None:
    async def exercise() -> None:
        plan = _plan()
        checkpoints = FileCheckpointStore(tmp_path)
        runner = StreamingRunner(plan, checkpoints)
        calls: list[tuple[str, int]] = []

        async def async_sink(batch: Batch) -> None:
            await asyncio.sleep(0)
            calls.append(("async", batch.to_pyarrow()["result"][0].as_py()))

        def sync_sink(batch: Batch) -> None:
            calls.append(("sync", batch.to_pyarrow()["result"][0].as_py()))

        result = await runner.step_async(
            _batch(2), sinks={"output": [sync_sink, async_sink]}
        )
        assert result.outputs["output"].to_pyarrow()["result"].to_pylist() == [3]
        assert calls == [("sync", 3), ("async", 3)]
        checkpoint = await checkpoints.load("stream")
        assert checkpoint is not None
        assert checkpoint["sequence"] == 0
        assert checkpoint["source_cursor"] is None
        assert await runner.plan_snapshot_async() == checkpoint["state"]
        with pytest.raises(ExecutionError, match="exclusively leased"):
            await plan.snapshot_async()

        await runner.reset_async()
        assert await checkpoints.load("stream") is None

    asyncio.run(exercise())


def test_streaming_failure_rolls_back_and_retries(tmp_path) -> None:
    async def exercise() -> None:
        runner = StreamingRunner(_plan("retry"), FileCheckpointStore(tmp_path))
        delivered: list[int] = []

        async def fail(_: Batch) -> None:
            raise RuntimeError("sink failed")

        with pytest.raises(ProviderError, match="sink failed"):
            await runner.step_async(_batch(1), sinks={"output": [fail]})
        assert await runner.plan_snapshot_async() == {"calc": None}

        await runner.step_async(
            _batch(1),
            sinks={
                "output": [
                    lambda batch: delivered.append(
                        batch.to_pyarrow()["result"][0].as_py()
                    )
                ]
            },
        )
        assert delivered == [2]

    asyncio.run(exercise())


def test_streaming_validates_all_sinks_before_delivery(tmp_path) -> None:
    async def exercise() -> None:
        runner = StreamingRunner(_plan("validation"), FileCheckpointStore(tmp_path))
        calls: list[str] = []
        with pytest.raises((TypeError, ConfigError)):
            await runner.step_async(
                _batch(1),
                sinks={"output": [lambda _: calls.append("called"), object()]},
            )
        assert calls == []

        with pytest.raises(ConfigError, match="unknown graph output"):
            await runner.step_async(
                _batch(1), sinks={"missing": [lambda _: calls.append("called")]}
            )
        assert calls == []

    asyncio.run(exercise())


@pytest.mark.parametrize("invalid_output", ["missing", ""])
def test_streaming_validates_empty_sink_routes_before_advancing(
    tmp_path, invalid_output: str
) -> None:
    async def exercise() -> None:
        pipeline_name = f"empty-stream-route-{invalid_output or 'blank'}"
        plan = _plan(pipeline_name)
        checkpoints = FileCheckpointStore(tmp_path)
        runner = StreamingRunner(plan, checkpoints)
        calls: list[str] = []

        with pytest.raises(ConfigError):
            await runner.step_async(
                _batch(1),
                sinks={
                    "output": [lambda _: calls.append("called")],
                    invalid_output: [],
                },
            )

        assert calls == []
        assert await checkpoints.load(pipeline_name) is None
        await runner.step_async(_batch(1), sinks={"output": []})
        checkpoint = await checkpoints.load(pipeline_name)
        assert checkpoint is not None
        assert checkpoint["sequence"] == 0

    asyncio.run(exercise())


def test_streaming_delivers_sorted_outputs_then_insertion_order(tmp_path) -> None:
    async def exercise() -> None:
        plan = (
            PipelineBuilder("ordered")
            .expression("source", "mid = value")
            .expression("zeta", "result = mid + 2")
            .expression("alpha", "result = mid + 1")
            .connect("source", "zeta")
            .connect("source", "alpha")
            .compile()
        )
        calls: list[str] = []
        runner = StreamingRunner(plan, FileCheckpointStore(tmp_path))
        await runner.step_async(
            _batch(1),
            sinks={
                "zeta.output": [lambda _: calls.append("zeta")],
                "alpha.output": [
                    lambda _: calls.append("alpha-1"),
                    lambda _: calls.append("alpha-2"),
                ],
            },
        )
        assert calls == ["alpha-1", "alpha-2", "zeta"]

    asyncio.run(exercise())


class _Source:
    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.index = 0
        self.opened: list[object] = []

    async def open(self, cursor: object) -> None:
        await asyncio.sleep(0)
        self.opened.append(cursor)
        self.index = 0 if cursor is None else int(cursor["offset"])

    def next(self):
        if self.index == len(self.values):
            return None
        position = self.index
        self.index += 1
        return _batch(self.values[position]), {"offset": self.index}, self.index


@pytest.mark.parametrize("invalid_output", ["missing", ""])
def test_micro_validates_empty_sink_routes_before_leasing_or_reading(
    tmp_path, invalid_output: str
) -> None:
    pipeline_name = f"empty-micro-route-{invalid_output or 'blank'}"
    plan = _plan(pipeline_name)
    checkpoints = FileCheckpointStore(tmp_path)
    source = _Source([1])
    calls: list[str] = []

    with pytest.raises(ConfigError):
        MicroBatchRunner(
            plan,
            source,
            checkpoints,
            sinks={
                "output": [lambda _: calls.append("called")],
                invalid_output: [],
            },
            checkpoint_every=1,
        )

    assert calls == []
    assert source.opened == []
    assert checkpoints.load_blocking(pipeline_name) is None
    assert plan.snapshot() == {"calc": None}

    runner = MicroBatchRunner(
        plan,
        source,
        checkpoints,
        sinks={"output": []},
        checkpoint_every=1,
    )
    assert runner.next() is not None


def test_micro_batch_runner_recovers_flushes_retries_and_releases_lease(
    tmp_path,
) -> None:
    async def exercise() -> None:
        checkpoints = FileCheckpointStore(tmp_path)
        first_plan = _plan("micro")
        source = _Source([1, 2])
        delivered: list[int] = []
        runner = MicroBatchRunner(
            first_plan,
            source,
            checkpoints,
            sinks={
                "output": [
                    lambda batch: delivered.append(
                        batch.to_pyarrow()["result"][0].as_py()
                    )
                ]
            },
            checkpoint_every=3,
        )
        assert await runner.next_async() is not None
        assert await checkpoints.load("micro") is None
        assert await runner.next_async() is not None
        assert await runner.next_async() is None
        checkpoint = await checkpoints.load("micro")
        assert checkpoint is not None
        assert checkpoint["source_cursor"] == {"offset": 2}
        assert delivered == [2, 3]

        del runner
        gc.collect()
        recovered_source = _Source([1, 2, 3])
        recovered = MicroBatchRunner(
            first_plan,
            recovered_source,
            checkpoints,
            checkpoint_every=1,
        )
        result = await recovered.next_async()
        assert result is not None
        assert recovered_source.opened == [{"offset": 2}]
        assert result.outputs["output"].to_pyarrow()["result"].to_pylist() == [4]
        await recovered.reset_async()
        assert await checkpoints.load("micro") is None

    asyncio.run(exercise())


def test_micro_batch_failure_retries_the_buffered_item(tmp_path) -> None:
    class OnceSource(_Source):
        def __init__(self) -> None:
            super().__init__([4])
            self.next_calls = 0

        def next(self):
            self.next_calls += 1
            return super().next()

    async def exercise() -> None:
        source = OnceSource()
        attempts = 0

        def sink(_: Batch) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("retry me")

        runner = MicroBatchRunner(
            _plan("micro-retry"),
            source,
            FileCheckpointStore(tmp_path),
            sinks={"output": [sink]},
            checkpoint_every=1,
        )
        with pytest.raises(ProviderError, match="retry me"):
            await runner.next_async()
        assert await runner.next_async() is not None
        assert source.next_calls == 1
        assert attempts == 2

    asyncio.run(exercise())


def test_source_validation_and_callback_failure_are_provider_errors(tmp_path) -> None:
    class InvalidSource:
        def open(self, cursor: object) -> None:
            del cursor

        def next(self):
            return object(), {"offset": 1}, 1

    async def exercise() -> None:
        runner = MicroBatchRunner(
            _plan("invalid-source"),
            InvalidSource(),
            FileCheckpointStore(tmp_path),
            checkpoint_every=1,
        )
        with pytest.raises(ProviderError, match="Batch"):
            await runner.next_async()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "item, message",
    [
        ((_batch(1), None), "exactly"),
        ((_batch(1), {1: "invalid"}, 1), "keys must be strings"),
        ((_batch(1), None, True), "u64"),
        ((_batch(1), None, -1), "u64"),
        ((_batch(1), None, 2**64), "u64"),
    ],
)
def test_source_items_are_strictly_validated(
    tmp_path, item: object, message: str
) -> None:
    class Source:
        def open(self, cursor: object) -> None:
            del cursor

        def next(self) -> object:
            return item

    async def exercise() -> None:
        runner = MicroBatchRunner(
            _plan(f"source-{len(message)}-{id(item) % 1_000_000}"),
            Source(),
            FileCheckpointStore(tmp_path / str(id(item))),
            checkpoint_every=1,
        )
        with pytest.raises(ProviderError, match=message):
            await runner.next_async()

    asyncio.run(exercise())


def test_streaming_rehomes_python_owned_array_batches(tmp_path) -> None:
    async def exercise() -> None:
        runtime = Runtime()
        register_numpy(runtime)
        plan = (
            PipelineBuilder("arrays")
            .external(
                "calc",
                "numpy",
                "expression",
                "1",
                {"expression": "x + 1"},
            )
            .compile(runtime)
        )
        seen: list[np.ndarray] = []
        runner = StreamingRunner(plan, FileCheckpointStore(tmp_path))
        result = await runner.step_async(
            Batch.from_array(np.array([1, 2]), backend="numpy"),
            sinks={"output": [lambda batch: seen.append(batch.array)]},
        )
        np.testing.assert_array_equal(seen[0], np.array([2, 3]))
        np.testing.assert_array_equal(result.outputs["output"].array, np.array([2, 3]))

    asyncio.run(exercise())


def test_runner_blocking_methods_and_loop_guards(tmp_path) -> None:
    runner = StreamingRunner(_plan("blocking"), FileCheckpointStore(tmp_path))
    result = runner.step(_batch(3))
    assert result.outputs["output"].to_pyarrow()["result"].to_pylist() == [4]
    assert runner.plan_snapshot() == {"calc": None}
    runner.reset()

    async def reject() -> None:
        with pytest.raises(RuntimeError, match="await step_async"):
            runner.step(_batch(1))
        with pytest.raises(RuntimeError, match="await plan_snapshot_async"):
            runner.plan_snapshot()

    asyncio.run(reject())


def test_micro_blocking_methods_and_constructor_guards(tmp_path) -> None:
    plan = _plan("micro-blocking")
    checkpoints = FileCheckpointStore(tmp_path)
    runner = MicroBatchRunner(plan, _Source([]), checkpoints, checkpoint_every=1)
    assert runner.next() is None
    assert runner.plan_snapshot() == {"calc": None}
    runner.reset()

    with pytest.raises(TypeError, match="ExecutionPlan"):
        MicroBatchRunner(object(), _Source([]), checkpoints)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="FileCheckpointStore"):
        MicroBatchRunner(plan, _Source([]), object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        MicroBatchRunner(plan, _Source([]), checkpoints, checkpoint_every=0)
    with pytest.raises(TypeError, match="open"):
        MicroBatchRunner(plan, object(), checkpoints)  # type: ignore[arg-type]

    async def reject() -> None:
        with pytest.raises(RuntimeError, match="await next_async"):
            runner.next()
        with pytest.raises(RuntimeError, match="await reset_async"):
            runner.reset()

    asyncio.run(reject())


def test_runner_is_cyclic_gc_visible_and_pending_task_retains_callbacks(
    tmp_path,
) -> None:
    async def exercise() -> None:
        released = asyncio.Event()
        entered = asyncio.Event()

        class Sink:
            async def __call__(self, batch: Batch) -> None:
                del batch
                entered.set()
                await released.wait()

        sink = Sink()
        reference = weakref.ref(sink)
        runner = StreamingRunner(_plan("lifetime"), FileCheckpointStore(tmp_path))
        task = asyncio.create_task(
            runner.step_async(_batch(1), sinks={"output": [sink]})
        )
        await entered.wait()
        del sink
        gc.collect()
        assert gc.is_tracked(runner._inner)
        assert reference() is not None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        result = await runner.step_async(_batch(1))
        assert result.outputs["output"].to_pyarrow()["result"].to_pylist() == [2]

    asyncio.run(exercise())


def test_micro_source_cycle_is_collectable_and_releases_plan_lease(tmp_path) -> None:
    class Source(_Source):
        runner: MicroBatchRunner | None = None

    plan = _plan("cycle")
    source = Source([])
    runner = MicroBatchRunner(
        plan, source, FileCheckpointStore(tmp_path), checkpoint_every=1
    )
    source.runner = runner
    runner_ref = weakref.ref(runner)
    source_ref = weakref.ref(source)
    del runner
    del source
    gc.collect()

    assert runner_ref() is None
    assert source_ref() is None
    assert plan.snapshot() == {"calc": None}


def test_micro_buffered_array_payload_cycle_is_collectable_after_failure(
    tmp_path,
) -> None:
    class Payload:
        runner: MicroBatchRunner | None = None

    class Source:
        def __init__(self, item: Batch) -> None:
            self.item: Batch | None = item

        def open(self, cursor: object) -> None:
            assert cursor is None

        def next(self) -> tuple[Batch, None, int] | None:
            if self.item is None:
                return None
            item = self.item
            self.item = None
            return item, None, 1

    def identity(batch: Batch, _options: dict[str, object]) -> Batch:
        return batch

    def fail(_: Batch) -> None:
        raise RuntimeError("retain buffered payload")

    runtime = Runtime()
    runtime.register_provider("test", "identity", "1", identity)
    plan = (
        PipelineBuilder("buffered-array-cycle")
        .external("calc", "test", "identity", "1", {})
        .compile(runtime)
    )
    payload = Payload()
    batch = Batch._from_external(payload, "test", 1, {})
    source = Source(batch)
    runner = MicroBatchRunner(
        plan,
        source,
        FileCheckpointStore(tmp_path),
        sinks={"output": [fail]},
        checkpoint_every=1,
    )
    payload.runner = runner
    payload_ref = weakref.ref(payload)
    runner_ref = weakref.ref(runner)
    source_ref = weakref.ref(source)

    with pytest.raises(ProviderError, match="retain buffered payload"):
        asyncio.run(runner.next_async())
    assert source.item is None

    del batch
    del payload
    del runner
    del source
    gc.collect()

    assert payload_ref() is None
    assert runner_ref() is None
    assert source_ref() is None
    assert plan.snapshot() == {"calc": None}


def test_streaming_cancellation_cancels_pending_sink_without_late_delivery(
    tmp_path,
) -> None:
    async def exercise() -> None:
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        cancellation_release = asyncio.Event()
        gate = asyncio.Event()
        late_deliveries: list[str] = []

        class Sink:
            async def __call__(self, batch: Batch) -> None:
                del batch
                entered.set()
                try:
                    await gate.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await cancellation_release.wait()
                    raise
                late_deliveries.append("delivered")

        runner = StreamingRunner(
            _plan("cancel-pending-sink"), FileCheckpointStore(tmp_path)
        )
        task = asyncio.create_task(
            runner.step_async(_batch(1), sinks={"output": [Sink()]})
        )
        await entered.wait()
        task.cancel()

        async def observe_cancellation() -> None:
            await task

        observer = asyncio.create_task(observe_cancellation())
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        task.cancel()
        completed, _ = await asyncio.wait({observer}, timeout=0.05)
        assert completed == set()
        cancellation_release.set()
        with pytest.raises(asyncio.CancelledError):
            await observer

        gate.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert late_deliveries == []

        result = await runner.step_async(_batch(2))
        assert result.outputs["output"].to_pyarrow()["result"].to_pylist() == [3]

    asyncio.run(exercise())


def test_micro_cancellation_cancels_pending_source_and_runner_recovers(
    tmp_path,
) -> None:
    async def exercise() -> None:
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        cancellation_release = asyncio.Event()
        gate = asyncio.Event()
        late_reads: list[str] = []

        class Source:
            def __init__(self) -> None:
                self.next_calls = 0

            def open(self, cursor: object) -> None:
                assert cursor is None

            async def next(self) -> tuple[Batch, None, int]:
                self.next_calls += 1
                if self.next_calls == 1:
                    entered.set()
                    try:
                        await gate.wait()
                    except asyncio.CancelledError:
                        cancelled.set()
                        await cancellation_release.wait()
                        raise
                    late_reads.append("read")
                return _batch(4), None, 1

        source = Source()
        runner = MicroBatchRunner(
            _plan("cancel-pending-source"),
            source,
            FileCheckpointStore(tmp_path),
            checkpoint_every=1,
        )
        task = asyncio.create_task(runner.next_async())
        await entered.wait()
        task.cancel()

        async def observe_cancellation() -> None:
            await task

        observer = asyncio.create_task(observe_cancellation())
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        task.cancel()
        completed, _ = await asyncio.wait({observer}, timeout=0.05)
        assert completed == set()
        cancellation_release.set()
        with pytest.raises(asyncio.CancelledError):
            await observer

        gate.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert late_reads == []

        result = await runner.next_async()
        assert result is not None
        assert result.outputs["output"].to_pyarrow()["result"].to_pylist() == [5]
        assert source.next_calls == 2

    asyncio.run(exercise())


def test_native_idle_waiters_all_complete_when_streaming_operation_finishes(
    tmp_path,
) -> None:
    script = textwrap.dedent(
        """
        import asyncio
        import sys

        import pyarrow as pa

        from calc_flow import (
            Batch,
            FileCheckpointStore,
            PipelineBuilder,
            StreamingRunner,
        )


        def batch(value):
            return Batch.from_pyarrow(pa.table({"value": [value]}))


        async def exercise():
            entered = asyncio.Event()
            release = asyncio.Event()

            async def sink(_):
                entered.set()
                await release.wait()

            plan = (
                PipelineBuilder("multiple-idle-waiters")
                .expression("calc", "result = value + 1")
                .compile()
            )
            runner = StreamingRunner(plan, FileCheckpointStore(sys.argv[1]))
            operation = asyncio.create_task(
                runner.step_async(batch(1), sinks={"output": [sink]})
            )
            await entered.wait()
            waiters = [
                asyncio.ensure_future(runner._inner.wait_idle_async()) for _ in range(3)
            ]
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            release.set()
            await operation
            await asyncio.gather(*waiters)
            result = await runner.step_async(batch(2))
            assert result.outputs["output"].to_pyarrow()["result"].to_pylist() == [3]


        asyncio.run(exercise())
        """
    )
    completed = subprocess.run(
        [sys.executable, "-W", "error", "-c", script, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr


def test_cancelled_sink_retaining_array_batch_is_collectable_and_releases_lease(
    tmp_path,
) -> None:
    class Payload:
        runner: StreamingRunner | None = None

    class Sink:
        def __init__(
            self,
            entered: asyncio.Event,
            cancelled: asyncio.Event,
            gate: asyncio.Event,
            late_deliveries: list[str],
        ) -> None:
            self.entered = entered
            self.cancelled = cancelled
            self.gate = gate
            self.late_deliveries = late_deliveries
            self.batch: Batch | None = None

        async def __call__(self, batch: Batch) -> None:
            self.batch = batch
            self.entered.set()
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            self.late_deliveries.append("delivered")

    def identity(batch: Batch, _options: dict[str, object]) -> Batch:
        return batch

    async def exercise() -> tuple[
        object,
        weakref.ReferenceType[Payload],
        weakref.ReferenceType[StreamingRunner],
        weakref.ReferenceType[Sink],
    ]:
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        gate = asyncio.Event()
        late_deliveries: list[str] = []
        runtime = Runtime()
        runtime.register_provider("test", "identity", "1", identity)
        plan = (
            PipelineBuilder("cancelled-array-cycle")
            .external("calc", "test", "identity", "1", {})
            .compile(runtime)
        )
        payload = Payload()
        batch = Batch._from_external(payload, "test", 1, {})
        sink = Sink(entered, cancelled, gate, late_deliveries)
        runner = StreamingRunner(plan, FileCheckpointStore(tmp_path))
        payload.runner = runner
        payload_ref = weakref.ref(payload)
        runner_ref = weakref.ref(runner)
        sink_ref = weakref.ref(sink)

        task = asyncio.create_task(runner.step_async(batch, sinks={"output": [sink]}))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        gate.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert late_deliveries == []

        del task
        del batch
        del payload
        del runner
        del sink
        return plan, payload_ref, runner_ref, sink_ref

    plan, payload_ref, runner_ref, sink_ref = asyncio.run(exercise())
    gc.collect()

    assert payload_ref() is None
    assert runner_ref() is None
    assert sink_ref() is None
    with pytest.raises(CheckpointError, match="requires recovery"):
        plan.snapshot()

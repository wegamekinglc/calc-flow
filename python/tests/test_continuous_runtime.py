from __future__ import annotations

import asyncio
import gc
import inspect
import json
import weakref
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pytest

import calc_flow
import calc_flow.runtime as runtime_module
from calc_flow import (
    Batch,
    BatchExecutionPlan,
    Cursor,
    Data,
    DeliveryGuarantee,
    DisabledWatermarks,
    Idle,
    ManagedCheckpointRuntime,
    NativeWatermarkCapability,
    PipelineBuilder,
    ReplayPositioning,
    SinkBinding,
    SinkRecovery,
    SourceBinding,
    SourceCapabilities,
    SourceDeliveryCapability,
    StreamExecutionPlan,
    StreamingRunner,
    StreamRequirements,
    _native,
)


def test_public_continuous_type_aliases_are_exported() -> None:
    for name in ("JSONValue", "SinkDelivery", "SourceEvent", "WatermarkPolicy"):
        assert name in calc_flow.__all__
        assert getattr(calc_flow, name) is not None


def test_legacy_continuous_runtime_symbols_are_removed() -> None:
    for name in (
        "FileCheckpointStore",
        "LegacyStreamingRunner",
        "MicroBatchRunner",
    ):
        assert name not in calc_flow.__all__
        assert not hasattr(calc_flow, name)

    assert not hasattr(runtime_module, "Source")
    for name in (
        "_ContinuousStreamingRunner",
        "_FileCheckpointStore",
        "_MicroBatchRunner",
    ):
        assert not hasattr(_native, name)

    assert hasattr(_native, "_StreamingRunner")
    assert hasattr(_native._StreamingRunner, "start_async")
    for method in ("plan_snapshot_async", "reset_async", "step_async"):
        assert not hasattr(_native._StreamingRunner, method)


def test_streaming_exceptions_are_exported_from_errors_module() -> None:
    from calc_flow import errors

    for name in ("StreamingRuntimeError", "CheckpointPublicationUnknownError"):
        assert name in errors.__all__
        assert getattr(errors, name) is getattr(calc_flow, name)


def test_compile_methods_return_distinct_plan_types() -> None:
    builder = PipelineBuilder("typed-plans").expression("calc", "result = value + 1")

    batch = builder.compile_batch()
    stream = builder.compile_stream()

    assert isinstance(batch, BatchExecutionPlan)
    assert isinstance(stream, StreamExecutionPlan)
    assert not isinstance(batch, StreamExecutionPlan)
    assert not isinstance(stream, BatchExecutionPlan)
    assert batch.name == stream.name == "typed-plans"


def test_stream_plan_exposes_immutable_binding_metadata() -> None:
    delivery = {"output": DeliveryGuarantee.EXACTLY_ONCE}
    requirements = StreamRequirements(delivery)
    delivery["output"] = DeliveryGuarantee.AT_LEAST_ONCE
    plan = (
        PipelineBuilder("stream-metadata")
        .expression("calc", "result = value + 1")
        .compile_stream(requirements=requirements)
    )

    assert plan.name == "stream-metadata"
    assert plan.fingerprint
    assert plan.source_binding_ids == ("input",)
    assert plan.sink_binding_ids == ("output",)
    assert plan.requirements.delivery == {"output": DeliveryGuarantee.EXACTLY_ONCE}


@pytest.mark.parametrize("delivery", ([], 1))
def test_stream_requirements_reject_non_mapping_delivery(delivery: object) -> None:
    with pytest.raises(TypeError, match="delivery must be a mapping"):
        StreamRequirements(delivery)  # type: ignore[arg-type]


def test_public_continuous_signatures_are_async_first() -> None:
    assert str(inspect.signature(ManagedCheckpointRuntime)) == (
        "(directory: 'os.PathLike[str] | str', /) -> 'None'"
    )
    assert str(inspect.signature(StreamingRunner.start_async)) == (
        "(self) -> 'StreamingJob'"
    )
    assert str(inspect.signature(StreamingRunner.start)) == ("(self) -> 'StreamingJob'")


def test_connector_bindings_require_declared_coroutine_methods() -> None:
    class InvalidSource:
        def capabilities(self) -> object:
            raise AssertionError("validation must not invoke the connector")

        def open(self, cursor: object) -> None:
            raise AssertionError("validation must not invoke the connector")

        async def next(self) -> None:
            return None

        async def close(self) -> None:
            return None

    class InvalidSink:
        async def open(self) -> None:
            return None

        def write(self, batch: object) -> None:
            raise AssertionError("validation must not invoke the connector")

        async def close(self) -> None:
            return None

    with pytest.raises(
        TypeError, match=r"source\.open must be declared with async def"
    ):
        SourceBinding(InvalidSource())
    with pytest.raises(TypeError, match=r"sink\.write must be declared with async def"):
        SinkBinding.ordinary("archive", InvalidSink())


def test_runner_is_consumed_once_and_blocking_calls_reject_active_loop(
    tmp_path: Path,
) -> None:
    class Source:
        def capabilities(self) -> object:
            from calc_flow import (
                ReplayPositioning,
                SourceCapabilities,
                SourceDeliveryCapability,
            )

            return SourceCapabilities(
                ReplayPositioning.UNSUPPORTED,
                SourceDeliveryCapability.LOSSY,
                max_batch_rows=1,
                max_batch_bytes=1,
            )

        async def open(self, cursor: object) -> None:
            return None

        async def next(self) -> None:
            return None

        async def close(self) -> None:
            return None

    class Sink:
        async def open(self) -> None:
            return None

        async def write(self, batch: object) -> None:
            return None

        async def close(self) -> None:
            return None

    plan = PipelineBuilder("demo").expression("plus_one", "b = a + 1").compile_stream()
    runner = StreamingRunner(
        plan,
        {"input": SourceBinding(Source())},
        {"output": [SinkBinding.ordinary("archive", Sink())]},
        ManagedCheckpointRuntime(tmp_path),
    )

    async def exercise() -> None:
        with pytest.raises(
            RuntimeError,
            match=r"start\(\) cannot run inside an event loop; use start_async\(\)",
        ):
            runner.start()
        job = await runner.start_async()
        waiter = job.wait_async()
        assert inspect.isawaitable(waiter)
        assert job.status()["job_id"] == job.id
        with pytest.raises(
            RuntimeError,
            match=r"wait\(\) cannot run inside an event loop; use wait_async\(\)",
        ):
            job.wait()
        await waiter
        with pytest.raises(
            RuntimeError,
            match=r"streaming runner has already been consumed by start\(\)",
        ):
            await runner.start_async()

    asyncio.run(exercise())


def test_continuous_runner_processes_data_and_closes_connectors(tmp_path: Path) -> None:
    events: list[str] = []
    written: list[int] = []

    class Source:
        def capabilities(self) -> object:
            from calc_flow import (
                NativeWatermarkCapability,
                ReplayPositioning,
                SourceCapabilities,
                SourceDeliveryCapability,
            )

            return SourceCapabilities(
                ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
                SourceDeliveryCapability.LOSSLESS,
                max_batch_rows=1,
                max_batch_bytes=1024,
                native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
            )

        async def open(self, cursor: Cursor | None) -> None:
            events.append("source.open")
            self.offset = 0 if cursor is None else int(cursor.payload["offset"])

        async def next(self) -> Data | None:
            if self.offset:
                return None
            self.offset = 1
            return Data(
                Batch.from_pyarrow(pa.table({"a": [1]})),
                Cursor(b"1", {"offset": 1}),
            )

        async def close(self) -> None:
            events.append("source.close")

    class Sink:
        async def open(self) -> None:
            events.append("sink.open")

        async def write(self, batch: Batch) -> None:
            written.extend(batch.to_pyarrow()["b"].to_pylist())

        async def close(self) -> None:
            events.append("sink.close")

    async def exercise() -> None:
        plan = (
            PipelineBuilder("data").expression("plus_one", "b = a + 1").compile_stream()
        )
        job = await StreamingRunner(
            plan,
            {"input": SourceBinding(Source(), watermark_policy=DisabledWatermarks())},
            {"output": [SinkBinding.ordinary("archive", Sink())]},
            ManagedCheckpointRuntime(tmp_path),
        ).start_async()
        outcome = await job.wait_async()
        assert outcome.state == "completed"

    asyncio.run(exercise())
    assert written == [2]
    assert events == ["source.open", "sink.open", "source.close", "sink.close"]


def test_blocking_lifecycle_keeps_connector_event_loop_alive(tmp_path: Path) -> None:
    written: list[int] = []

    class Source:
        def capabilities(self) -> object:
            from calc_flow import (
                NativeWatermarkCapability,
                ReplayPositioning,
                SourceCapabilities,
                SourceDeliveryCapability,
            )

            return SourceCapabilities(
                ReplayPositioning.UNSUPPORTED,
                SourceDeliveryCapability.LOSSY,
                max_batch_rows=1,
                max_batch_bytes=1024,
                native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
            )

        async def open(self, cursor: Cursor | None) -> None:
            self.emitted = False

        async def next(self) -> Data | None:
            if self.emitted:
                return None
            self.emitted = True
            return Data(Batch.from_pyarrow(pa.table({"a": [3]})), Cursor(b"1", {}))

        async def close(self) -> None:
            return None

    class Sink:
        async def open(self) -> None:
            return None

        async def write(self, batch: Batch) -> None:
            written.extend(batch.to_pyarrow()["b"].to_pylist())

        async def close(self) -> None:
            return None

    job = StreamingRunner(
        PipelineBuilder("blocking-continuous")
        .expression("plus_one", "b = a + 1")
        .compile_stream(),
        {"input": SourceBinding(Source(), watermark_policy=DisabledWatermarks())},
        {"output": [SinkBinding.ordinary("archive", Sink())]},
        ManagedCheckpointRuntime(tmp_path),
    ).start()
    assert job.wait().state == "completed"
    assert written == [4]


def test_cancelling_waiter_does_not_cancel_job(tmp_path: Path) -> None:
    class Source:
        def capabilities(self) -> object:
            from calc_flow import (
                NativeWatermarkCapability,
                ReplayPositioning,
                SourceCapabilities,
                SourceDeliveryCapability,
            )

            return SourceCapabilities(
                ReplayPositioning.UNSUPPORTED,
                SourceDeliveryCapability.LOSSY,
                max_batch_rows=1,
                max_batch_bytes=1,
                native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
            )

        async def open(self, cursor: Cursor | None) -> None:
            return None

        async def next(self) -> None:
            await asyncio.Event().wait()

        async def close(self) -> None:
            return None

    class Sink:
        async def open(self) -> None:
            return None

        async def write(self, batch: Batch) -> None:
            return None

        async def close(self) -> None:
            return None

    async def exercise() -> None:
        plan = PipelineBuilder("cancel").expression("copy", "b = a").compile_stream()
        job = await StreamingRunner(
            plan,
            {"input": SourceBinding(Source(), watermark_policy=DisabledWatermarks())},
            {"output": [SinkBinding.ordinary("archive", Sink())]},
            ManagedCheckpointRuntime(tmp_path),
        ).start_async()
        waiter = asyncio.create_task(job.wait_async())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert job.status()["state"] == "running"
        status = job.status()
        status["state"] = "failed"
        assert job.status()["state"] == "running"
        outcome = await job.cancel_async()
        assert outcome.state == "cancelled"

    asyncio.run(exercise())


def test_cancelling_start_reaps_launch_and_consumes_runner(tmp_path: Path) -> None:
    class Source:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.closed = asyncio.Event()

        def capabilities(self) -> object:
            from calc_flow import (
                NativeWatermarkCapability,
                ReplayPositioning,
                SourceCapabilities,
                SourceDeliveryCapability,
            )

            return SourceCapabilities(
                ReplayPositioning.UNSUPPORTED,
                SourceDeliveryCapability.LOSSY,
                max_batch_rows=1,
                max_batch_bytes=1,
                native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
            )

        async def open(self, cursor: Cursor | None) -> None:
            self.entered.set()
            await asyncio.Event().wait()

        async def next(self) -> None:
            return None

        async def close(self) -> None:
            self.closed.set()

    class Sink:
        async def open(self) -> None:
            return None

        async def write(self, batch: Batch) -> None:
            return None

        async def close(self) -> None:
            return None

    async def exercise() -> None:
        source = Source()
        runner = StreamingRunner(
            PipelineBuilder("cancel-start")
            .expression("copy", "b = a")
            .compile_stream(),
            {"input": SourceBinding(source, watermark_policy=DisabledWatermarks())},
            {"output": [SinkBinding.ordinary("archive", Sink())]},
            ManagedCheckpointRuntime(tmp_path),
        )
        launch = asyncio.create_task(runner.start_async())
        await asyncio.wait_for(source.entered.wait(), timeout=1)
        launch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await launch
        assert source.closed.is_set()
        with pytest.raises(
            RuntimeError,
            match=r"streaming runner has already been consumed by start\(\)",
        ):
            await runner.start_async()

    asyncio.run(exercise())


def test_checkpoint_restart_restores_owned_source_cursor(tmp_path: Path) -> None:
    opened: list[int] = []
    written: list[int] = []

    class Source:
        def __init__(self, pause_after_first: bool) -> None:
            self.pause_after_first = pause_after_first
            self.offset = 0
            self.written = asyncio.Event()

        def capabilities(self) -> object:
            from calc_flow import (
                NativeWatermarkCapability,
                ReplayPositioning,
                SourceCapabilities,
                SourceDeliveryCapability,
            )

            return SourceCapabilities(
                ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
                SourceDeliveryCapability.LOSSLESS,
                max_batch_rows=1,
                max_batch_bytes=1024,
                native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
            )

        async def open(self, cursor: Cursor | None) -> None:
            self.offset = 0 if cursor is None else int(cursor.payload["offset"])
            opened.append(self.offset)

        async def next(self) -> Data | Idle | None:
            if self.pause_after_first and self.offset == 1:
                await asyncio.sleep(0)
                return Idle()
            if self.offset >= 2:
                return None
            value = self.offset
            self.offset += 1
            return Data(
                Batch.from_pyarrow(pa.table({"a": [value]})),
                Cursor(self.offset.to_bytes(1), {"offset": self.offset}),
            )

        async def close(self) -> None:
            return None

    class Sink:
        def __init__(self, source: Source) -> None:
            self.source = source

        async def open(self) -> None:
            return None

        async def write(self, batch: Batch) -> None:
            written.extend(batch.to_pyarrow()["b"].to_pylist())
            self.source.written.set()

        async def close(self) -> None:
            return None

    def plan() -> StreamExecutionPlan:
        return PipelineBuilder("restart").expression("copy", "b = a").compile_stream()

    async def exercise() -> None:
        first_source = Source(pause_after_first=True)
        first = await StreamingRunner(
            plan(),
            {
                "input": SourceBinding(
                    first_source, watermark_policy=DisabledWatermarks()
                )
            },
            {"output": [SinkBinding.ordinary("archive", Sink(first_source))]},
            ManagedCheckpointRuntime(tmp_path),
        ).start_async()
        await first_source.written.wait()
        assert await first.trigger_checkpoint_async() == 1
        await first.cancel_async()

        second_source = Source(pause_after_first=False)
        second = await StreamingRunner(
            plan(),
            {
                "input": SourceBinding(
                    second_source, watermark_policy=DisabledWatermarks()
                )
            },
            {"output": [SinkBinding.ordinary("archive", Sink(second_source))]},
            ManagedCheckpointRuntime(tmp_path),
        ).start_async()
        assert (await second.wait_async()).state == "completed"

    asyncio.run(exercise())
    assert opened == [0, 1]
    assert written == [0, 1]


def test_checkpoint_completion_does_not_wait_for_next_source_poll(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        polling = asyncio.Event()
        written = asyncio.Event()

        class Source:
            def capabilities(self) -> object:
                from calc_flow import (
                    NativeWatermarkCapability,
                    ReplayPositioning,
                    SourceCapabilities,
                    SourceDeliveryCapability,
                )

                return SourceCapabilities(
                    ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
                    SourceDeliveryCapability.LOSSLESS,
                    max_batch_rows=1,
                    max_batch_bytes=1024,
                    native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
                )

            async def open(self, cursor: Cursor | None) -> None:
                self.emitted = False

            async def next(self) -> Data | None:
                if not self.emitted:
                    self.emitted = True
                    return Data(
                        Batch.from_pyarrow(pa.table({"a": [1]})),
                        Cursor(b"1", {"offset": 1}),
                    )
                polling.set()
                await asyncio.Event().wait()

            async def close(self) -> None:
                return None

        class Sink:
            async def open(self) -> None:
                return None

            async def write(self, batch: Batch) -> None:
                written.set()

            async def close(self) -> None:
                return None

        job = await StreamingRunner(
            PipelineBuilder("checkpoint-long-poll")
            .expression("copy", "b = a")
            .compile_stream(),
            {"input": SourceBinding(Source(), watermark_policy=DisabledWatermarks())},
            {"output": [SinkBinding.ordinary("archive", Sink())]},
            ManagedCheckpointRuntime(tmp_path),
        ).start_async()
        await asyncio.gather(written.wait(), polling.wait())
        try:
            assert (
                await asyncio.wait_for(job.trigger_checkpoint_async(), timeout=1) == 1
            )
        finally:
            await job.cancel_async()

    asyncio.run(exercise())


def test_terminal_job_releases_cyclic_connector_roots(tmp_path: Path) -> None:
    class Source:
        owner: object | None = None

        def capabilities(self) -> object:
            from calc_flow import (
                NativeWatermarkCapability,
                ReplayPositioning,
                SourceCapabilities,
                SourceDeliveryCapability,
            )

            return SourceCapabilities(
                ReplayPositioning.UNSUPPORTED,
                SourceDeliveryCapability.LOSSY,
                max_batch_rows=1,
                max_batch_bytes=1,
                native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
            )

        async def open(self, cursor: Cursor | None) -> None:
            return None

        async def next(self) -> None:
            return None

        async def close(self) -> None:
            return None

    class Sink:
        owner: object | None = None

        async def open(self) -> None:
            return None

        async def write(self, batch: Batch) -> None:
            return None

        async def close(self) -> None:
            return None

    async def exercise() -> tuple[weakref.ReferenceType[object], ...]:
        source = Source()
        sink = Sink()
        runner = StreamingRunner(
            PipelineBuilder("gc").expression("copy", "b = a").compile_stream(),
            {"input": SourceBinding(source, watermark_policy=DisabledWatermarks())},
            {"output": [SinkBinding.ordinary("archive", sink)]},
            ManagedCheckpointRuntime(tmp_path),
        )
        source.owner = runner
        job = await runner.start_async()
        sink.owner = job
        await job.wait_async()
        return tuple(weakref.ref(value) for value in (source, sink, runner, job))

    references = asyncio.run(exercise())
    gc.collect()
    assert [reference() for reference in references] == [None, None, None, None]


def _blocking_started_job(tmp_path: Path, *, finite: bool) -> object:
    class Source:
        def capabilities(self) -> SourceCapabilities:
            return SourceCapabilities(
                ReplayPositioning.UNSUPPORTED,
                SourceDeliveryCapability.LOSSY,
                max_batch_rows=1,
                max_batch_bytes=1,
                native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
            )

        async def open(self, cursor: Cursor | None) -> None:
            return None

        async def next(self) -> None:
            if finite:
                return None
            await asyncio.Event().wait()

        async def close(self) -> None:
            return None

    class Sink:
        async def open(self) -> None:
            return None

        async def write(self, batch: Batch) -> None:
            return None

        async def close(self) -> None:
            return None

    return StreamingRunner(
        PipelineBuilder(f"blocking-loop-{finite}")
        .expression("copy", "b = a")
        .compile_stream(),
        {"input": SourceBinding(Source(), watermark_policy=DisabledWatermarks())},
        {"output": [SinkBinding.ordinary("archive", Sink())]},
        ManagedCheckpointRuntime(tmp_path),
    ).start()


@pytest.mark.parametrize(
    ("method", "finite", "expected_state"),
    (
        ("shutdown_async", False, "completed"),
        ("cancel_async", False, "cancelled"),
        ("wait_async", True, "completed"),
    ),
)
def test_blocking_start_async_terminal_closes_owned_event_loop(
    tmp_path: Path, method: str, finite: bool, expected_state: str
) -> None:
    job = _blocking_started_job(tmp_path / method, finite=finite)
    blocking_loop = job._blocking_loop
    thread = blocking_loop._thread

    outcome = asyncio.run(getattr(job, method)())

    thread.join(timeout=2)
    assert outcome.state == expected_state
    assert not thread.is_alive()
    assert job._blocking_loop is None


@pytest.mark.parametrize("cancel_count", (1, 3), ids=("single", "multiple"))
def test_blocking_start_native_terminal_outcome_wins_cleanup_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_count: int
) -> None:
    job = _blocking_started_job(tmp_path, finite=True)
    blocking_loop = job._blocking_loop
    thread = blocking_loop._thread
    original_close_async = runtime_module._BlockingEventLoop.close_async
    cleanup_completions = 0

    async def exercise() -> tuple[runtime_module.JobOutcome, int]:
        cleanup_entered = asyncio.Event()
        cleanup_release = asyncio.Event()

        async def controlled_close_async(
            self: runtime_module._BlockingEventLoop,
        ) -> None:
            nonlocal cleanup_completions
            cleanup_entered.set()
            await cleanup_release.wait()
            await original_close_async(self)
            cleanup_completions += 1

        async def observe_event_loop_progress() -> bool:
            await asyncio.sleep(0)
            return True

        monkeypatch.setattr(
            runtime_module._BlockingEventLoop,
            "close_async",
            controlled_close_async,
        )
        observer = asyncio.create_task(job.wait_async())
        await cleanup_entered.wait()
        for _ in range(cancel_count):
            observer.cancel()
        progress = asyncio.create_task(observe_event_loop_progress())
        assert await asyncio.wait_for(progress, timeout=1)
        cleanup_release.set()
        outcome = await asyncio.wait_for(observer, timeout=1)
        return outcome, observer.cancelling()

    outcome, observer_cancellation_count = asyncio.run(exercise())

    thread.join(timeout=2)
    assert outcome.state == "completed"
    assert cleanup_completions == 1
    assert observer_cancellation_count == 0
    assert not thread.is_alive()
    assert job._blocking_loop is None


def test_blocking_started_job_gc_reaps_owned_event_loop(tmp_path: Path) -> None:
    job = _blocking_started_job(tmp_path, finite=False)
    blocking_loop = job._blocking_loop
    thread = blocking_loop._thread
    reference = weakref.ref(job)

    del job
    gc.collect()
    thread.join(timeout=2)

    assert reference() is None
    assert not thread.is_alive()


def test_abandoned_job_reaps_python_connector(tmp_path: Path) -> None:
    async def exercise() -> None:
        source_closed = asyncio.Event()
        sink_closed = asyncio.Event()

        class Source:
            def capabilities(self) -> object:
                from calc_flow import (
                    NativeWatermarkCapability,
                    ReplayPositioning,
                    SourceCapabilities,
                    SourceDeliveryCapability,
                )

                return SourceCapabilities(
                    ReplayPositioning.UNSUPPORTED,
                    SourceDeliveryCapability.LOSSY,
                    max_batch_rows=1,
                    max_batch_bytes=1,
                    native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
                )

            async def open(self, cursor: Cursor | None) -> None:
                return None

            async def next(self) -> None:
                await asyncio.Event().wait()

            async def close(self) -> None:
                source_closed.set()

        class Sink:
            async def open(self) -> None:
                return None

            async def write(self, batch: Batch) -> None:
                return None

            async def close(self) -> None:
                sink_closed.set()

        source = Source()
        job = await StreamingRunner(
            PipelineBuilder("abandoned-gc")
            .expression("copy", "b = a")
            .compile_stream(),
            {"input": SourceBinding(source, watermark_policy=DisabledWatermarks())},
            {"output": [SinkBinding.ordinary("archive", Sink())]},
            ManagedCheckpointRuntime(tmp_path),
        ).start_async()
        reference = weakref.ref(job)
        del job
        gc.collect()
        await asyncio.wait_for(
            asyncio.gather(source_closed.wait(), sink_closed.wait()), timeout=1
        )
        assert reference() is None

    asyncio.run(exercise())


def test_shared_restart_vector_is_exactly_once_across_checkpoint_recovery(
    tmp_path: Path,
) -> None:
    fixture = (
        Path(__file__).parents[2]
        / "tests"
        / "fixtures"
        / "a6"
        / "continuous_restart_vectors.json"
    )
    vector = json.loads(fixture.read_text(encoding="utf-8"))
    original = json.dumps(vector, sort_keys=True)
    plan_vector = vector["plan"]
    records = vector["records"]
    expected = vector["expected"]
    managed_root = tmp_path / "managed"
    sink_root = tmp_path / "sink"
    opened_offsets: list[int] = []
    lifecycle = {"source_closes": 0, "sink_closes": 0, "writes": 0}

    def commit_epoch(epoch: int, values: list[int]) -> None:
        sink_root.mkdir(parents=True, exist_ok=True)
        target = sink_root / f"visible-{epoch:020}.json"
        if target.exists():
            assert json.loads(target.read_text(encoding="utf-8")) == values
            return
        temporary = sink_root / f".tmp-{epoch:020}.json"
        temporary.write_text(json.dumps(values), encoding="utf-8")
        temporary.replace(target)

    class Source:
        def __init__(self, pause_at: int | None) -> None:
            self.pause_at = pause_at
            self.offset = 0

        def capabilities(self) -> SourceCapabilities:
            return SourceCapabilities(
                ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
                SourceDeliveryCapability.LOSSLESS,
                max_batch_rows=1,
                max_batch_bytes=1024,
                native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
            )

        async def open(self, cursor: Cursor | None) -> None:
            self.offset = 0 if cursor is None else int(cursor.payload["offset"])
            opened_offsets.append(self.offset)

        async def next(self) -> Data | Idle | None:
            if self.pause_at == self.offset:
                await asyncio.sleep(0)
                return Idle()
            if self.offset >= len(records):
                return None
            record = records[self.offset]
            assert record["offset"] == self.offset
            self.offset += 1
            return Data(
                Batch.from_pyarrow(pa.table({"value": [record["value"]]})),
                Cursor(
                    self.offset.to_bytes(8, "big"),
                    {"offset": self.offset},
                ),
            )

        async def close(self) -> None:
            lifecycle["source_closes"] += 1

    class Sink:
        def __init__(self, changed: asyncio.Event) -> None:
            self.changed = changed
            self.pending: list[int] = []

        async def open(self) -> None:
            return None

        async def begin_epoch(self, epoch: int) -> None:
            self.pending.clear()

        async def write(self, batch: Batch) -> None:
            values = batch.to_pyarrow()["doubled"].to_pylist()
            self.pending.extend(values)
            lifecycle["writes"] += len(values)
            self.changed.set()

        async def pre_commit(self, epoch: int) -> dict[str, object]:
            return {"values": list(self.pending)}

        async def commit(self, epoch: int, pre_commit: Mapping[str, object]) -> None:
            raw_values = pre_commit["values"]
            assert isinstance(raw_values, list)
            values = [int(value) for value in raw_values]
            await asyncio.to_thread(commit_epoch, epoch, values)

        async def abort(
            self, epoch: int, pre_commit: Mapping[str, object] | None
        ) -> None:
            self.pending.clear()

        async def recover(self, recovery: SinkRecovery) -> None:
            raw_values = recovery.pre_commit["values"]
            assert isinstance(raw_values, list)
            await asyncio.to_thread(
                commit_epoch,
                recovery.epoch,
                [int(value) for value in raw_values],
            )

        async def close(self) -> None:
            lifecycle["sink_closes"] += 1

    def plan() -> StreamExecutionPlan:
        return (
            PipelineBuilder(plan_vector["name"])
            .expression(plan_vector["operator_id"], plan_vector["expression"])
            .compile_stream(
                requirements=StreamRequirements(
                    {
                        plan_vector["output_id"]: DeliveryGuarantee.EXACTLY_ONCE,
                    }
                )
            )
        )

    def runner(pause_at: int | None, changed: asyncio.Event) -> StreamingRunner:
        return StreamingRunner(
            plan(),
            {
                plan_vector["source_id"]: SourceBinding(
                    Source(pause_at), watermark_policy=DisabledWatermarks()
                )
            },
            {
                plan_vector["output_id"]: [
                    SinkBinding.transactional(plan_vector["sink_id"], Sink(changed))
                ]
            },
            ManagedCheckpointRuntime(managed_root),
        )

    def charged_edges(status: Mapping[str, object]) -> int:
        edges = status["edges"]
        assert isinstance(edges, dict)
        return sum(
            isinstance(edge, dict)
            and (
                edge["current_envelopes"] != 0
                or edge["current_rows"] != 0
                or edge["current_bytes"] != 0
            )
            for edge in edges.values()
        )

    async def exercise() -> None:
        changed = asyncio.Event()
        first = await runner(vector["checkpoint_after"], changed).start_async()
        while lifecycle["writes"] < vector["checkpoint_after"]:
            await changed.wait()
            changed.clear()
        epoch = await first.trigger_checkpoint_async()
        assert epoch == expected["checkpoint_epoch"]
        assert (await first.cancel_async()).state == "cancelled"
        first_status = first.status()
        assert first_status["task_count"] == expected["terminal_tasks"]
        assert charged_edges(first_status) == expected["terminal_charged_edges"]

        second = await runner(None, asyncio.Event()).start_async()
        outcome = await second.wait_async()
        assert outcome.state == "completed"
        assert outcome.completed_epoch == expected["terminal_epoch"]
        second_status = second.status()
        assert second_status["task_count"] == expected["terminal_tasks"]
        assert charged_edges(second_status) == expected["terminal_charged_edges"]

    asyncio.run(exercise())

    visible: list[int] = []
    for path in sorted(sink_root.glob("visible-*.json")):
        visible.extend(json.loads(path.read_text(encoding="utf-8")))
    assert visible == expected["values"]
    assert len(visible) - len(set(visible)) == expected["duplicates"]
    assert len(set(expected["values"]) - set(visible)) == expected["missing"]
    assert opened_offsets == expected["opened_offsets"]
    assert lifecycle["source_closes"] == 2
    assert lifecycle["sink_closes"] == 2
    assert (
        sum(
            path.is_file() and (".tmp" in path.name or path.name.startswith("tmp"))
            for path in tmp_path.rglob("*")
        )
        == expected["temporary_artifacts"]
    )
    assert json.dumps(vector, sort_keys=True) == original


@pytest.mark.parametrize(
    ("method", "expected_state"),
    (("shutdown_async", "completed"), ("cancel_async", "cancelled")),
    ids=("shutdown", "cancel"),
)
def test_cancelling_terminal_observer_preserves_owner_cleanup(
    tmp_path: Path, method: str, expected_state: str
) -> None:
    async def exercise() -> None:
        polled = asyncio.Event()
        close_entered = asyncio.Event()
        close_release = asyncio.Event()

        class Source:
            def capabilities(self) -> SourceCapabilities:
                return SourceCapabilities(
                    ReplayPositioning.UNSUPPORTED,
                    SourceDeliveryCapability.LOSSY,
                    max_batch_rows=1,
                    max_batch_bytes=1,
                    native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
                )

            async def open(self, cursor: Cursor | None) -> None:
                return None

            async def next(self) -> None:
                polled.set()
                await asyncio.Event().wait()

            async def close(self) -> None:
                close_entered.set()
                await close_release.wait()

        class Sink:
            async def open(self) -> None:
                return None

            async def write(self, batch: Batch) -> None:
                return None

            async def close(self) -> None:
                return None

        job = await StreamingRunner(
            PipelineBuilder(f"cancel-{method}")
            .expression("copy", "b = a")
            .compile_stream(),
            {"input": SourceBinding(Source(), watermark_policy=DisabledWatermarks())},
            {"output": [SinkBinding.ordinary("archive", Sink())]},
            ManagedCheckpointRuntime(tmp_path / method),
        ).start_async()
        await polled.wait()
        observer = asyncio.create_task(getattr(job, method)())
        await close_entered.wait()
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        close_release.set()
        outcome = await asyncio.wait_for(job.wait_async(), timeout=1)
        assert outcome.state == expected_state
        assert job.status()["task_count"] == 0

    asyncio.run(exercise())


def test_cancelling_checkpoint_observer_does_not_cancel_checkpoint_or_job(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        polled = asyncio.Event()
        pre_commit_entered = asyncio.Event()
        pre_commit_release = asyncio.Event()
        commit_completed = asyncio.Event()
        commits: list[int] = []

        class Source:
            def capabilities(self) -> SourceCapabilities:
                return SourceCapabilities(
                    ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
                    SourceDeliveryCapability.LOSSLESS,
                    max_batch_rows=1,
                    max_batch_bytes=1,
                    native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
                )

            async def open(self, cursor: Cursor | None) -> None:
                return None

            async def next(self) -> None:
                polled.set()
                await asyncio.Event().wait()

            async def close(self) -> None:
                return None

        class Sink:
            async def open(self) -> None:
                return None

            async def begin_epoch(self, epoch: int) -> None:
                return None

            async def write(self, batch: Batch) -> None:
                return None

            async def pre_commit(self, epoch: int) -> dict[str, object]:
                pre_commit_entered.set()
                await pre_commit_release.wait()
                return {"epoch": epoch}

            async def commit(
                self, epoch: int, pre_commit: Mapping[str, object]
            ) -> None:
                commits.append(epoch)
                commit_completed.set()

            async def abort(
                self, epoch: int, pre_commit: Mapping[str, object] | None
            ) -> None:
                return None

            async def recover(self, recovery: SinkRecovery) -> None:
                return None

            async def close(self) -> None:
                return None

        job = await StreamingRunner(
            PipelineBuilder("cancel-checkpoint-observer")
            .expression("copy", "b = a")
            .compile_stream(
                requirements=StreamRequirements(
                    {"output": DeliveryGuarantee.EXACTLY_ONCE}
                )
            ),
            {"input": SourceBinding(Source(), watermark_policy=DisabledWatermarks())},
            {"output": [SinkBinding.transactional("archive", Sink())]},
            ManagedCheckpointRuntime(tmp_path),
        ).start_async()
        await polled.wait()
        observer = asyncio.create_task(job.trigger_checkpoint_async())
        await pre_commit_entered.wait()
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert job.status()["state"] == "running"
        pre_commit_release.set()
        await asyncio.wait_for(commit_completed.wait(), timeout=1)
        assert await job.trigger_checkpoint_async() == 2
        assert job.status()["checkpoint"]["last_completed_epoch"] == 2
        assert commits == [1, 2]
        assert (await job.cancel_async()).state == "cancelled"
        assert job.status()["task_count"] == 0

    asyncio.run(exercise())

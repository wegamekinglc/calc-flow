from __future__ import annotations

import asyncio
import gc
import inspect
import weakref
from pathlib import Path

import pyarrow as pa
import pytest

import calc_flow
from calc_flow import (
    Batch,
    BatchExecutionPlan,
    Cursor,
    Data,
    DeliveryGuarantee,
    DisabledWatermarks,
    Idle,
    ManagedCheckpointRuntime,
    PipelineBuilder,
    SinkBinding,
    SourceBinding,
    StreamExecutionPlan,
    StreamingRunner,
    StreamRequirements,
)


def test_public_continuous_type_aliases_are_exported() -> None:
    for name in ("JSONValue", "SinkDelivery", "SourceEvent", "WatermarkPolicy"):
        assert name in calc_flow.__all__
        assert getattr(calc_flow, name) is not None


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

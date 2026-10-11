from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pytest

import calc_flow as cf
import calc_flow.stream as stream_module


def test_checkpointing_defaults_on_and_preserves_positional_config() -> None:
    config = cf.StreamRuntimeConfig(
        timedelta(seconds=1), timedelta(seconds=2), cf.EdgeBudget(), 7
    )
    assert config.checkpointing is True
    assert config.retained_epochs == 7
    assert config._native()["checkpointing"] is True
    assert (
        cf.StreamRuntimeConfig(checkpointing=False)._native()["checkpointing"] is False
    )


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_checkpointing_requires_a_boolean(value: object) -> None:
    with pytest.raises(TypeError, match="checkpointing must be a bool"):
        cf.StreamRuntimeConfig(checkpointing=value)._native()


@pytest.mark.parametrize("entrypoint", ["expression", "program", "execute"])
def test_checkpointing_off_streams_without_a_temporary_directory(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str
) -> None:
    def forbidden_root(**kwargs: object) -> str:
        raise AssertionError("disabled streams must not create checkpoint storage")

    monkeypatch.setattr(stream_module.tempfile, "mkdtemp", forbidden_root)
    closed: list[bool] = []

    async def feed():
        try:
            yield pa.table({"value": [1, 2]})
            yield pa.table({"value": [3]})
        finally:
            closed.append(True)

    source = cf.table_input("events", schema=pa.schema([("value", pa.int64())]))
    output = source.select(doubled=source["value"] * 2)
    config = cf.StreamRuntimeConfig(checkpointing=False)
    if entrypoint == "expression":
        results = output.stream(feed(), config=config)
    else:
        program = cf.Program("off", engine="streaming", outputs={"result": output})
        results = getattr(program, "stream" if entrypoint == "program" else "execute")(
            {"events": feed()}, config=config
        )

    async def run() -> None:
        async with results:
            values = [value async for value in results]
        tables = (
            values if entrypoint == "expression" else [value.table for value in values]
        )
        assert pa.concat_tables(tables).to_pydict() == {"doubled": [2, 4, 6]}
        outcome = await results.job.wait_async()
        assert (outcome.state, outcome.cause, outcome.completed_epoch) == (
            "completed",
            "natural_end",
            None,
        )
        assert not outcome.errors
        status = results.job.status()
        assert status["task_count"] == 0
        assert status["checkpoint"]["current_epoch"] is None
        assert status["checkpoint"]["failure_category"] is None

    asyncio.run(run())
    assert closed == [True]


def test_checkpointing_off_keeps_sql_state_budget() -> None:
    source = cf.table_input("events", schema=pa.schema([("value", pa.int64())]))
    query = source.sql("SELECT SUM(value) AS total FROM input")

    async def feed():
        yield pa.table({"value": [1]})
        yield pa.table({"value": [2]})

    async def run() -> None:
        config = cf.StreamRuntimeConfig(
            checkpointing=False, sql_state_budget=cf.StateBudget(1, 1_024)
        )
        with pytest.raises(
            cf.StreamingRuntimeError, match="operator .* execution failed"
        ):
            async with query.stream(feed(), config=config) as results:
                async for _ in results:
                    pass
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())


def test_checkpointing_off_early_exit_closes_source() -> None:
    closed: list[bool] = []

    async def feed():
        try:
            yield pa.table({"value": [1]})
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    source = cf.table_input("events", schema=pa.schema([("value", pa.int64())]))

    async def run() -> None:
        results = source.stream(
            feed(), config=cf.StreamRuntimeConfig(checkpointing=False)
        )
        async with results:
            assert (await results.__anext__()).to_pydict() == {"value": [1]}
        assert results.job.status()["task_count"] == 0
        outcome = await results.job.wait_async()
        assert outcome.state == "cancelled"
        assert outcome.completed_epoch is None

    asyncio.run(run())
    assert closed == [True]


@pytest.mark.parametrize("checkpointing", [True, False])
def test_checkpointing_rejects_conflicting_storage(
    tmp_path: Path, checkpointing: bool
) -> None:
    plan = cf.PipelineBuilder("mode").expression("calc", "b = a + 1").compile_stream()
    backend = None if checkpointing else cf.ManagedCheckpointRuntime(tmp_path)
    config = cf.StreamRuntimeConfig(checkpointing=checkpointing)
    with pytest.raises((TypeError, ValueError), match="checkpoints"):
        cf.StreamingRunner(plan, {}, {}, backend, config=config)
    with pytest.raises(ValueError, match="checkpointing"):
        cf._native._StreamingRunner(
            plan._inner,
            {},
            {},
            None if backend is None else backend._inner,
            config._native(),
            {},
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_native_checkpointing_requires_a_boolean(value: object) -> None:
    plan = cf.PipelineBuilder("mode").expression("calc", "b = a + 1").compile_stream()
    config = cf.StreamRuntimeConfig()._native()
    config["checkpointing"] = value
    with pytest.raises(TypeError, match="checkpointing must be a bool"):
        cf._native._StreamingRunner(plan._inner, {}, {}, None, config, {})


def test_checkpointing_off_explicit_runner_drains_and_closes() -> None:
    events: list[str] = []
    output: list[int] = []

    class Source:
        def capabilities(self) -> cf.SourceCapabilities:
            return cf.SourceCapabilities(
                cf.ReplayPositioning.UNSUPPORTED,
                cf.SourceDeliveryCapability.LOSSY,
                max_batch_rows=1,
                max_batch_bytes=1_024,
                native_watermarks=cf.NativeWatermarkCapability.NEVER_EMITS,
            )

        async def open(self, cursor: cf.Cursor | None) -> None:
            assert cursor is None
            events.append("source.open")
            self.sent = False

        async def next(self) -> cf.Data | None:
            if self.sent:
                return None
            self.sent = True
            return cf.Data(
                cf.Batch.from_pyarrow(pa.table({"a": [1]})), cf.Cursor(b"1", {})
            )

        async def close(self) -> None:
            events.append("source.close")

    class Sink:
        async def open(self) -> None:
            events.append("sink.open")

        async def write(self, batch: cf.Batch) -> None:
            output.extend(batch.to_pyarrow()["b"].to_pylist())

        async def close(self) -> None:
            events.append("sink.close")

    async def run() -> None:
        plan = (
            cf.PipelineBuilder("off-explicit")
            .expression("calc", "b = a + 1")
            .compile_stream()
        )
        job = await cf.StreamingRunner(
            plan,
            {
                "input": cf.SourceBinding(
                    Source(), watermark_policy=cf.DisabledWatermarks()
                )
            },
            {"output": [cf.SinkBinding.ordinary("sink", Sink())]},
            config=cf.StreamRuntimeConfig(checkpointing=False),
        ).start_async()
        outcome = await job.wait_async()
        assert (outcome.state, outcome.cause, outcome.completed_epoch) == (
            "completed",
            "natural_end",
            None,
        )
        assert not outcome.errors
        assert job.status()["task_count"] == 0

    asyncio.run(run())
    assert output == [2]
    assert events.count("source.open") == events.count("source.close") == 1
    assert events.count("sink.open") == events.count("sink.close") == 1

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pytest

import calc_flow as cf
import calc_flow.stream as stream_module


def test_checkpointing_defaults_to_mode_and_preserves_positional_config() -> None:
    config = cf.StreamRuntimeConfig(
        timedelta(seconds=1), timedelta(seconds=2), cf.EdgeBudget(), 7
    )
    assert config.checkpointing is None
    assert config.retained_epochs == 7
    # Unresolved lowering keeps the explicit-runner default; native sees a bool.
    assert config._native()["checkpointing"] is True
    for value in (True, False):
        native = cf.StreamRuntimeConfig(checkpointing=value)._native()
        assert native["checkpointing"] is value


def test_checkpoint_inflight_max_bytes_defaults_and_validates() -> None:
    native = cf.StreamRuntimeConfig()._native()
    assert native["checkpoint_inflight_max_bytes"] == 1 << 30
    assert (
        cf.StreamRuntimeConfig(checkpoint_inflight_max_bytes=4096)._native()[
            "checkpoint_inflight_max_bytes"
        ]
        == 4096
    )
    with pytest.raises(ValueError, match="checkpoint_inflight_max_bytes"):
        cf.StreamRuntimeConfig(checkpoint_inflight_max_bytes=0)._native()
    with pytest.raises(ValueError, match="checkpoint_inflight_max_bytes"):
        cf.StreamRuntimeConfig(checkpoint_inflight_max_bytes=1 << 53)._native()


@pytest.mark.parametrize("value", [0, 1, "false"])
def test_checkpointing_requires_a_boolean_or_none(value: object) -> None:
    with pytest.raises(TypeError, match="checkpointing must be a bool or None"):
        cf.StreamRuntimeConfig(checkpointing=value)._native()


@pytest.mark.parametrize("value", [0, 1, "false"])
def test_runners_reject_non_boolean_checkpointing(value: object) -> None:
    plan = cf.PipelineBuilder("mode").expression("calc", "b = a + 1").compile_stream()
    config = cf.StreamRuntimeConfig(checkpointing=value)
    with pytest.raises(TypeError, match="checkpointing must be a bool or None"):
        cf.StreamingRunner(plan, {}, {}, config=config)

    source = cf.table_input("events", schema=pa.schema([("value", pa.int64())]))

    async def run() -> None:
        with pytest.raises(TypeError, match="checkpointing must be a bool or None"):
            await source.stream(pa.table({"value": [1]}), config=config).__aenter__()

    asyncio.run(run())


@pytest.mark.parametrize("entrypoint", ["expression", "program", "execute"])
@pytest.mark.parametrize(
    "config",
    [
        None,
        cf.StreamRuntimeConfig(edge_budget=cf.EdgeBudget(8, 1 << 20)),
        cf.StreamRuntimeConfig(checkpointing=False),
    ],
    ids=["default", "explicit-mode-default", "off"],
)
def test_checkpointing_off_streams_without_a_temporary_directory(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    config: cf.StreamRuntimeConfig | None,
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


def test_explicit_checkpointing_stream_creates_and_reclaims_a_temporary_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stream_module.tempfile, "tempdir", str(tmp_path))
    original = stream_module.tempfile.mkdtemp
    created: list[str] = []

    def recorded_root(**kwargs: object) -> str:
        created.append(original(**kwargs))
        return created[-1]

    monkeypatch.setattr(stream_module.tempfile, "mkdtemp", recorded_root)

    async def feed():
        yield pa.table({"value": [1]})

    source = cf.table_input("events", schema=pa.schema([("value", pa.int64())]))
    config = cf.StreamRuntimeConfig(checkpointing=True)

    async def run() -> None:
        async with source.stream(feed(), config=config) as results:
            tables = [table async for table in results]
        assert pa.concat_tables(tables).to_pydict() == {"value": [1]}
        assert (await results.job.wait_async()).completed_epoch is not None

    asyncio.run(run())
    assert len(created) == 1
    assert Path(created[0]).parent == tmp_path
    assert list(tmp_path.iterdir()) == []


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


class _OneRowSource:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self._sent = False

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
        self._events.append("source.open")
        self._sent = False

    async def next(self) -> cf.Data | None:
        if self._sent:
            return None
        self._sent = True
        return cf.Data(cf.Batch.from_pyarrow(pa.table({"a": [1]})), cf.Cursor(b"1", {}))

    async def close(self) -> None:
        self._events.append("source.close")


class _ListSink:
    def __init__(self, events: list[str], output: list[int]) -> None:
        self._events = events
        self._output = output

    async def open(self) -> None:
        self._events.append("sink.open")

    async def write(self, batch: cf.Batch) -> None:
        self._output.extend(batch.to_pyarrow()["b"].to_pylist())

    async def close(self) -> None:
        self._events.append("sink.close")


def _run_explicit(
    checkpoints: cf.ManagedCheckpointRuntime | None,
    config: cf.StreamRuntimeConfig | None,
) -> tuple[cf.JobOutcome, list[str], list[int]]:
    events: list[str] = []
    output: list[int] = []

    async def run() -> cf.JobOutcome:
        plan = (
            cf.PipelineBuilder("explicit")
            .expression("calc", "b = a + 1")
            .compile_stream()
        )
        job = await cf.StreamingRunner(
            plan,
            {
                "input": cf.SourceBinding(
                    _OneRowSource(events), watermark_policy=cf.DisabledWatermarks()
                )
            },
            {"output": [cf.SinkBinding.ordinary("sink", _ListSink(events, output))]},
            checkpoints,
            config=config,
        ).start_async()
        outcome = await job.wait_async()
        assert job.status()["task_count"] == 0
        return outcome

    return asyncio.run(run()), events, output


def test_checkpointing_off_explicit_runner_drains_and_closes() -> None:
    outcome, events, output = _run_explicit(
        None, cf.StreamRuntimeConfig(checkpointing=False)
    )
    assert (outcome.state, outcome.cause, outcome.completed_epoch) == (
        "completed",
        "natural_end",
        None,
    )
    assert not outcome.errors
    assert output == [2]
    assert events.count("source.open") == events.count("source.close") == 1
    assert events.count("sink.open") == events.count("sink.close") == 1


@pytest.mark.parametrize(
    "config",
    [None, cf.StreamRuntimeConfig(retained_epochs=3)],
    ids=["no-config", "mode-default"],
)
def test_explicit_runner_mode_default_still_requires_storage(
    config: cf.StreamRuntimeConfig | None,
) -> None:
    plan = cf.PipelineBuilder("mode").expression("calc", "b = a + 1").compile_stream()
    with pytest.raises(TypeError, match="checkpoints must be a calc_flow"):
        cf.StreamingRunner(plan, {}, {}, config=config)


@pytest.mark.parametrize(
    "config",
    [None, cf.StreamRuntimeConfig(retained_epochs=3)],
    ids=["no-config", "mode-default"],
)
def test_explicit_runner_mode_default_checkpoints_with_storage(
    tmp_path: Path, config: cf.StreamRuntimeConfig | None
) -> None:
    outcome, _, output = _run_explicit(cf.ManagedCheckpointRuntime(tmp_path), config)
    assert (outcome.state, outcome.cause) == ("completed", "natural_end")
    assert outcome.completed_epoch is not None
    assert output == [2]

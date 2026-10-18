from __future__ import annotations

import asyncio
import gc
import weakref
from datetime import timedelta

import pyarrow as pa
import pytest

import calc_flow as cf
from calc_flow._stream_inputs import _IterableSource


def test_arrow_source_binding_streams_bounded_materialized_input() -> None:
    data = pa.table({"value": list(range(9))})
    source = cf.table_input("events", schema=data.schema)
    binding = cf.SourceBinding.from_arrow(data, budget=cf.EdgeBudget(2, 1024))

    async def run() -> None:
        async with source.stream(binding) as results:
            tables = [table async for table in results]
        assert pa.concat_tables(tables).equals(data)
        assert all(table.num_rows <= 2 for table in tables)
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("kind", ["chunks", "slice", "record", "empty"])
def test_materialized_stream_matches_iterable_without_python_source_calls(
    monkeypatch, named, kind
) -> None:
    field = pa.field("value", pa.int64(), metadata={b"unit": b"count"})
    schema = pa.schema([field], metadata={b"origin": b"caller"})
    data = pa.Table.from_batches(
        [
            pa.record_batch([[1, 2]], schema=schema),
            pa.record_batch([[3]], schema=schema),
        ]
    )
    if kind == "slice":
        data = data.slice(1)
    elif kind == "record":
        data = data.to_batches()[0]
    elif kind == "empty":
        data = data.slice(0, 0)
    before = data.to_pydict(), data.schema
    source = cf.table_input("events", schema=data.schema)
    output = source.select(doubled=source["value"] * 2)
    owner = (
        cf.Program("arrow", engine="streaming", outputs={"result": output})
        if named
        else output
    )
    calls = 0
    original = _IterableSource.next

    async def counted(self):
        nonlocal calls
        calls += 1
        return await original(self)

    monkeypatch.setattr(_IterableSource, "next", counted)

    async def feed():
        yield data

    async def collect(value):
        inputs = {"events": value} if named else value
        async with owner.stream(inputs) as results:
            tables = [event.table if named else event async for event in results]
        assert results.job.status()["state"] == "completed"
        assert results.job.status()["task_count"] == 0
        return pa.concat_tables(tables)

    expected = asyncio.run(collect(feed()))
    assert calls == 2  # data and EOF
    calls = 0
    actual = asyncio.run(collect(data))
    assert calls == 0
    assert actual.equals(expected, check_metadata=True)
    assert (data.to_pydict(), data.schema) == before


def test_materialized_stream_obeys_visible_byte_budget() -> None:
    data = pa.table({"value": ["a" * 10, "b" * 10, None, "c" * 10]})
    source = cf.table_input("events", schema=data.schema)
    config = cf.StreamRuntimeConfig(edge_budget=cf.EdgeBudget(10, 32))

    async def run():
        async with source.stream(data, config=config) as results:
            tables = [table async for table in results]
        assert pa.concat_tables(tables).equals(data)
        assert len(tables) > 1
        assert all(table.nbytes <= 32 for table in tables)

    asyncio.run(run())


def test_materialized_stream_rejects_oversized_row_and_reaps_tasks() -> None:
    data = pa.table({"value": ["too large"]})
    source = cf.table_input("events", schema=data.schema)
    config = cf.StreamRuntimeConfig(edge_budget=cf.EdgeBudget(10, 4))

    async def run():
        results = source.stream(data, config=config)
        with pytest.raises(cf.StreamingRuntimeError):
            async with results:
                _ = [table async for table in results]
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())


def _temporal(data):
    return cf.table_input(
        "events",
        schema=data.schema,
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("ts",),
    )


@pytest.mark.parametrize("times", [[2, 1], [1, None]])
def test_materialized_default_time_policy_rejects_disorder_and_null(times) -> None:
    data = pa.table(
        {
            "symbol": ["a", "a"],
            "ts": pa.array(times, type=pa.timestamp("us", tz="UTC")),
            "price": [1.0, 2.0],
        }
    )
    source = _temporal(data)
    config = cf.StreamRuntimeConfig(edge_budget=cf.EdgeBudget(1, 1024))

    async def run():
        results = source.stream(data, config=config)
        with pytest.raises(cf.StreamingRuntimeError):
            async with results:
                _ = [table async for table in results]
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())


@pytest.mark.parametrize("ordered", [False, True])
def test_materialized_time_policy_matches_iterable_at_eof(ordered) -> None:
    data = pa.table(
        {
            "symbol": ["a", "a", "a"],
            "ts": pa.array(
                [1, 2, 3] if ordered else [2, 1, 3], type=pa.timestamp("us", tz="UTC")
            ),
            "price": [2.0, 1.0, 3.0],
        }
    )
    data = data.cast(
        pa.schema(
            [
                field.with_nullable(False) if field.name in ("symbol", "ts") else field
                for field in data.schema
            ]
        )
    )
    source = _temporal(data)
    output = source.select(previous=cf.ts.lag(source["price"]))
    policy = (
        None
        if ordered
        else cf.BoundedOutOfOrderness(
            "ts", timedelta(microseconds=2), timedelta(milliseconds=1)
        )
    )

    async def feed():
        yield data

    async def collect(value):
        async with output.stream(value, watermarks=policy) as results:
            return pa.concat_tables([table async for table in results])

    assert asyncio.run(collect(data)).equals(asyncio.run(collect(feed())))


def test_materialized_input_is_bounded_when_consumer_stops_and_cancelled() -> None:
    data = pa.table({"value": list(range(1000))})
    source = cf.table_input("events", schema=data.schema)
    config = cf.StreamRuntimeConfig(edge_budget=cf.EdgeBudget(1, 1024))

    async def run():
        results = source.stream(data, config=config)
        async with results:
            await results.__anext__()
            await asyncio.sleep(0.05)
            status = results.job.status()
            assert sum(item["data_batches"] for item in status["sources"].values()) < 20
        assert results.job.status()["state"] == "cancelled"
        assert results.job.status()["task_count"] == 0
        await results.aclose()

    asyncio.run(run())


def test_materialized_input_rejects_reader_and_source_watermarks() -> None:
    data = pa.table({"value": [1]})
    with pytest.raises(TypeError, match="Table, RecordBatch"):
        cf.SourceBinding.from_arrow(data.to_reader())
    with pytest.raises(ValueError, match="cannot emit native watermarks"):
        cf.SourceBinding.from_arrow(
            data, watermark_policy=cf.SourceProvidedWatermarks()
        )
    source = cf.table_input("events", schema=pa.schema([("other", pa.int64())]))

    async def run():
        with pytest.raises(ValueError, match="declared Arrow schema"):
            async with source.stream(data):
                pass

    asyncio.run(run())


@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_materialized_join_matches_full_oracle(side) -> None:
    from benchmarks.arrow_source import collect

    facts = asyncio.run(collect("join", side))
    assert facts["rows"] == 64
    assert facts["task_count"] == 0


def test_materialized_binding_preserves_batch_metadata_and_can_be_reused() -> None:
    data = cf.Batch.from_pyarrow(
        pa.table({"a": [1, 2, 3]}), metadata={"caller": [1, 2]}
    )
    binding = cf.SourceBinding.from_arrow(data, budget=cf.EdgeBudget(1, 1024))

    class Sink:
        def __init__(self):
            self.batches = []

        async def open(self):
            pass

        async def write(self, batch):
            self.batches.append(batch)

        async def close(self):
            pass

    async def run():
        for _ in range(2):
            sink = Sink()
            runner = cf.StreamingRunner(
                cf.PipelineBuilder("arrow")
                .expression("copy", "b = a")
                .compile_stream(),
                {"input": binding},
                {"output": [cf.SinkBinding.ordinary("sink", sink)]},
                config=cf.StreamRuntimeConfig(checkpointing=False),
            )
            job = await runner.start_async()
            outcome = await job.wait_async()
            assert outcome.state == "completed"
            assert [batch.metadata for batch in sink.batches] == [data.metadata] * 3
            assert [batch.to_pyarrow()["b"].to_pylist() for batch in sink.batches] == [
                [1],
                [2],
                [3],
            ]

    asyncio.run(run())


@pytest.mark.parametrize("ending", ["eof", "failure", "cancel"])
def test_materialized_job_releases_foreign_buffer_owner(ending) -> None:
    import numpy as np

    class Owner:
        def __init__(self):
            self.values = np.arange(128, dtype=np.int64)

    owner = Owner()
    reference = weakref.ref(owner)
    buffer = pa.foreign_buffer(
        owner.values.ctypes.data, owner.values.nbytes, base=owner
    )
    array = pa.Array.from_buffers(pa.int64(), 128, [None, buffer])
    binding = cf.SourceBinding.from_arrow(
        pa.table({"a": array}), budget=cf.EdgeBudget(1, 1024)
    )

    class Sink:
        def __init__(self):
            self.entered = asyncio.Event()

        async def open(self):
            pass

        async def write(self, batch):
            self.entered.set()
            if ending == "failure":
                raise RuntimeError("sink failure")
            if ending == "cancel":
                await asyncio.Event().wait()

        async def close(self):
            pass

    sink = Sink()
    runner = cf.StreamingRunner(
        cf.PipelineBuilder("owner").expression("copy", "b = a").compile_stream(),
        {"input": binding},
        {"output": [cf.SinkBinding.ordinary("sink", sink)]},
        config=cf.StreamRuntimeConfig(checkpointing=False),
    )
    del owner, buffer, array, binding
    assert reference() is not None

    async def run():
        job = await runner.start_async()
        if ending == "cancel":
            await asyncio.wait_for(sink.entered.wait(), 5)
            outcome = await job.cancel_async()
        else:
            outcome = await job.wait_async()
        assert (
            outcome.state
            == {
                "eof": "completed",
                "failure": "recovery_required",
                "cancel": "cancelled",
            }[ending]
        )
        assert job.status()["task_count"] == 0

    asyncio.run(run())
    gc.collect()
    assert reference() is None

from __future__ import annotations

import asyncio

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import calc_flow as cf


def test_ordered_iterable_validates_sliced_chunks_without_python_comparison(
    monkeypatch,
):
    calls = []
    compare = pc.less

    def record_comparison(left, right):
        calls.append(len(left))
        return compare(left, right)

    monkeypatch.setattr(pc, "less", record_comparison)
    dtype = pa.timestamp("us", tz="UTC")
    chunks = [
        pa.array([99, 20, 20, 98], type=dtype).slice(1, 2),
        pa.array([21], type=dtype),
    ]
    table = pa.table({"ts": pa.chunked_array(chunks)})
    source = cf.table_input("events", schema=table.schema, event_time="ts")

    async def feed():
        yield table
        yield table.slice(0, 0)
        yield table.slice(2, 1)

    async def collect():
        async with source.stream(feed()) as results:
            return pa.concat_tables([item async for item in results])

    assert asyncio.run(collect()).equals(pa.concat_tables([table, table.slice(2, 1)]))
    assert calls == []


@pytest.mark.parametrize(
    ("chunks", "prior", "message"),
    [
        ([[401, 400], [None]], None, "expected non-null event time"),
        ([[401], [], [400]], None, "expected nondecreasing event time"),
        ([[], [400]], 401, "expected nondecreasing event time"),
    ],
)
def test_ordered_time_checks_all_records_in_the_declared_unit(chunks, prior, message):
    from calc_flow._native import _ordered_event_time

    table = pa.table({"ts": pa.chunked_array(chunks, type=pa.timestamp("ns"))})
    with pytest.raises(ValueError, match=message):
        _ordered_event_time(cf.Batch.from_pyarrow(table), "ts", prior)


def test_timestamp_error_precedes_table_import_error(monkeypatch):
    table = pa.table(
        {
            "ts": pa.array([2, 1], type=pa.timestamp("us", tz="UTC")),
            "value": [10, 20],
        }
    )
    original = cf.Batch.from_pyarrow
    rejected = []

    def fail_full_import(value, *args, **kwargs):
        if value is table:
            rejected.append(value)
            raise RuntimeError("cannot import table")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(cf.Batch, "from_pyarrow", fail_full_import)
    source = cf.table_input("events", schema=table.schema, event_time="ts")

    async def feed():
        yield table

    async def run():
        with pytest.raises(
            cf.StreamingRuntimeError, match="events.event_time: expected nondecreasing"
        ):
            async with source.stream(feed()) as results:
                await results.__anext__()

    asyncio.run(run())
    assert len(rejected) == 1


def test_empty_iterable_batch_keeps_previous_timestamp_for_next_batch():
    dtype = pa.timestamp("us", tz="UTC")
    first = pa.table({"ts": pa.array([20], type=dtype)})
    source = cf.table_input("events", schema=first.schema, event_time="ts")

    async def feed():
        yield first
        yield first.slice(0, 0)
        yield pa.record_batch({"ts": pa.array([19], type=dtype)})

    async def run():
        with pytest.raises(cf.StreamingRuntimeError, match="expected nondecreasing"):
            async with source.stream(feed()) as results:
                async for _ in results:
                    pass

    asyncio.run(run())

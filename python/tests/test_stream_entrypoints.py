"""Explicit progress policies work through both public stream entry points."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pyarrow as pa
import pytest

import calc_flow as cf


def _data() -> pa.Table:
    schema = pa.schema(
        [
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("price", pa.float64(), nullable=False),
        ]
    )
    return pa.table(
        {"symbol": ["a", "a"], "ts": [1, 2], "price": [10.0, 12.0]}, schema=schema
    )


async def _read_previous(results) -> list[float | None]:
    values = []
    async for result in results:
        table = result.table if isinstance(result, cf.StreamOutput) else result
        values.extend(table["previous"].to_pylist())
        if len(values) == 2:
            return values
    raise AssertionError("stream ended before yielding its declared rows")


@pytest.mark.parametrize("named", [False, True], ids=["table", "program"])
def test_stream_entrypoints_forward_explicit_progress_before_eof(named):
    data = _data()
    source = cf.table_input(
        "quotes",
        schema=data.schema,
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("ts",),
    )
    output = source.select(previous=cf.ts.lag(source["price"]))
    owner = cf.Program("lag", outputs={"lag": output}) if named else output
    policy = cf.SourceProvidedWatermarks()

    async def run():
        paused, closed = asyncio.Event(), asyncio.Event()

        async def feed():
            try:
                yield data
                yield cf.Watermark(datetime(1970, 1, 1, microsecond=3, tzinfo=UTC))
                paused.set()
                await asyncio.Event().wait()
            finally:
                closed.set()

        inputs = {"quotes": feed()} if named else feed()
        watermarks = {"quotes": policy} if named else policy
        results = owner.stream(inputs, watermarks=watermarks)
        async with asyncio.timeout(5), results:
            await paused.wait()
            assert await _read_previous(results) == [None, 10.0]
            assert not closed.is_set()
        assert closed.is_set()
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())

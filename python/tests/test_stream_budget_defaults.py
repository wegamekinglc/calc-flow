from __future__ import annotations

import asyncio
import sys

import pyarrow as pa

import calc_flow as cf


def test_default_stream_budget_allows_large_batches_without_byte_ceiling() -> None:
    budget = cf.EdgeBudget()
    assert budget.max_rows == 1_000_000
    assert budget.max_bytes == 2 * sys.maxsize + 1
    native = cf.StreamRuntimeConfig()._native()
    assert native["edge_max_rows"] == 1_000_000
    assert native["edge_max_bytes"] == 2 * sys.maxsize + 1


def test_explicit_stream_budget_remains_finite() -> None:
    native = cf.StreamRuntimeConfig(edge_budget=cf.EdgeBudget(7, 1 << 30))._native()
    assert native["edge_max_rows"] == 7
    assert native["edge_max_bytes"] == 1 << 30


def test_materialized_input_preserves_one_large_arrow_batch_by_default() -> None:
    data = pa.table({"value": range(10_001)})
    source = cf.table_input("events", schema=data.schema)

    async def run() -> list[pa.Table]:
        async with source.stream(data) as results:
            return [table async for table in results]

    output = asyncio.run(run())
    assert [table.num_rows for table in output] == [10_001]
    assert output[0].equals(data)


def test_iterable_large_batches_preserve_caller_boundaries() -> None:
    data = pa.table({"value": range(10_001)})
    tail = pa.table({"value": [10_001, 10_002]})
    source = cf.table_input("events", schema=data.schema)

    async def batches():
        yield data
        yield tail

    async def run() -> list[pa.Table]:
        async with source.stream(batches()) as results:
            return [table async for table in results]

    output = asyncio.run(run())
    assert [table.num_rows for table in output] == [10_001, 2]
    assert pa.concat_tables(output).equals(pa.concat_tables([data, tail]))

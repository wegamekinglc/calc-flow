"""Finalize fixed UTC windows over complete finite input or real stream EOF."""

from __future__ import annotations

import asyncio

import pyarrow as pa

import calc_flow as cf


def values(table):
    return list(
        zip(
            table["window_start"].cast(pa.int64()).to_pylist(),
            table["n"].to_pylist(),
            table["volume"].to_pylist(),
        )
    )


async def main():
    schema = pa.schema([pa.field("ts", pa.timestamp("us")), pa.field("q", pa.int8())])
    source = cf.table_input("trades", schema=schema)
    df = cf.window.tumbling(
        source,
        event_time="ts",
        size_micros=10,
        aggregates=[
            cf.window.count("q", output="n"),
            cf.window.sum("q", output="volume"),
        ],
    )
    sql = cf.sql(
        "SELECT window_start, window_end, COUNT(q) AS n, SUM(q) AS volume "
        "FROM CF_TUMBLE('a', 'ts', 10) GROUP BY window_start, window_end",
        a=source,
    )
    program = cf.Program("bars", mode="batch", outputs={"df": df, "sql": sql})
    table = pa.Table.from_arrays(
        [
            pa.array([-1, 0, 9, 10], type=pa.timestamp("us")),
            pa.array([2, 3, None, 5], type=pa.int8()),
        ],
        schema=schema,
    )
    expected = [(-10, 1, 2), (0, 1, 3), (10, 1, 5)]
    for result in (await program.collect_async({"trades": table})).values():
        if values(result) != expected:
            raise RuntimeError("finite window oracle mismatch")
    observed = {"df": [], "sql": []}
    async with program.for_mode("stream").stream(
        {"trades": [table.slice(0, 2), table.slice(2)]},
        watermarks=cf.DisabledWatermarks(),
    ) as events:
        async for event in events:
            observed[event.name].extend(values(event.table))
    if any(actual != expected for actual in observed.values()):
        raise RuntimeError("EOF window oracle mismatch")
    print("DataFrame and SQL window oracles passed in batch and at real EOF")


if __name__ == "__main__":
    asyncio.run(main())

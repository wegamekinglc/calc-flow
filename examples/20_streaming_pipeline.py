from __future__ import annotations

import asyncio

import pyarrow as pa

import calc_flow as cf

SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
    ]
)


async def batches():
    for ts, prices in [([1, 2], [10.0, 12.0]), ([3, 4], [15.0, 14.0])]:
        yield pa.table({"ts": ts, "symbol": ["a", "a"], "price": prices}, schema=SCHEMA)
    await asyncio.Event().wait()  # Keep the input open: output must not wait for EOF.


def features(t: cf.TableExpr) -> cf.TableExpr:
    delta = cf.ts.delta(t["price"])
    return t.select(delta=delta, mean_delta=cf.ts.mean(delta, window=cf.rows(2)))


async def main() -> None:
    source = cf.table_input(
        "quotes",
        schema=SCHEMA,
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("ts",),
    )
    output = source.pipe(features).sql("SELECT delta, mean_delta FROM input")
    tables = []
    async with asyncio.timeout(5), output.stream(batches()) as results:
        async for table in results:
            tables.append(table)
            if sum(batch.num_rows for batch in tables) >= 3:
                break
    actual = pa.concat_tables(tables).to_pydict()
    if actual != {"delta": [None, 2.0, 3.0], "mean_delta": [None, 2.0, 2.5]}:
        raise RuntimeError(actual)


if __name__ == "__main__":
    asyncio.run(main())

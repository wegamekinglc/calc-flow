"""Reuse table and column pipelines around a native SQL stage."""

from __future__ import annotations

import asyncio

import pyarrow as pa

import calc_flow as cf


def add_gross(t: cf.TableExpr) -> cf.TableExpr:
    return t.with_columns(gross=cf.row.cast(t["quantity"], "float64") * t["price"])


def discounted(t: cf.TableExpr, rate: float) -> cf.TableExpr:
    return t.select("order_id", net=t["gross"] * (1.0 - rate))


def pipeline(t: cf.TableExpr) -> cf.TableExpr:
    return (
        t.pipe(add_gross)
        .sql("SELECT order_id, gross FROM input WHERE gross >= 20 ORDER BY order_id")
        .pipe(discounted, rate=0.1)
    )


async def run() -> None:
    orders = pa.table(
        {
            "order_id": [1, 2, 3],
            "quantity": [2, 1, 3],
            "price": [10.0, 5.0, 10.0],
        }
    )
    source = cf.table_input("orders", schema=orders.schema)
    result_expr = pipeline(source).sql("SELECT order_id, net FROM input")
    batch = cf.Program(
        "sql-expression-pipeline",
        mode="batch",
        outputs={"gross": add_gross(source), "result": result_expr},
    )
    stream = batch.for_mode("stream")
    assert stream.fingerprint == batch.fingerprint
    schemas = await asyncio.to_thread(batch.collect_schema)
    assert schemas["result"].names == ["order_id", "net"]
    result = (await batch.collect_async({"orders": orders}))["result"]
    expected = {"order_id": [1, 3], "net": [18.0, 27.0]}
    if result.to_pydict() != expected:
        raise RuntimeError(result.to_pydict())
    streamed = []
    async with stream.stream({"orders": [orders]}) as events:
        async for event in events:
            if event.name == "result":
                streamed.extend(event.table.to_pylist())
    assert streamed == result.to_pylist()
    print(result.to_pydict())


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()

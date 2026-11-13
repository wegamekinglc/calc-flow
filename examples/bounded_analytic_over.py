"""One native bounded analytic declaration, executed in both lifecycles."""

from __future__ import annotations

import asyncio

import pyarrow as pa

import calc_flow as cf


def declaration():
    schema = pa.schema(
        [pa.field("v", pa.int64()), pa.field("seq", pa.uint64(), nullable=False)]
    )
    events = cf.table_input("events", schema=schema)
    order = cf.OverOrder([], [cf.SortKey("v")], sequence_by=["seq"])
    out = events.over(
        [
            cf.AnalyticExpr.row_number("rn"),
            cf.AnalyticExpr.lag("v", name="prev"),
            cf.AnalyticExpr.sum("v", frame=cf.RowsFrame.cumulative(), name="total"),
        ],
        spec=cf.OverSpec.bounded_source(order, state_budget=cf.StateBudget(9, 8 << 20)),
    )
    return cf.Program("features", mode="batch", outputs={"out": out}), schema


async def main():
    program, schema = declaration()
    data = pa.Table.from_pydict({"v": [3, 1, 2], "seq": [30, 10, 20]}, schema)
    batch = (await program.collect_async({"events": data}))["out"]
    assert batch.to_pydict() == dict(
        v=[1, 2, 3],
        seq=[10, 20, 30],
        rn=[1, 2, 3],
        prev=[None, 1, 2],
        total=[1, 3, 6],
    )
    source = cf.SourceBinding.from_arrow(
        data, budget=cf.EdgeBudget(max_rows=2, max_bytes=4096)
    )
    finals = []
    async with program.for_mode("stream").stream(
        {"events": source}, config=cf.StreamRuntimeConfig(checkpointing=False)
    ) as results:
        async for event in results:
            finals.append(event.table)
    assert len(finals) == 1
    assert finals[0].equals(batch)
    print(batch.to_pydict())


if __name__ == "__main__":
    asyncio.run(main())

"""Share DataFrame/SQL union declarations across finite and stream execution."""

from __future__ import annotations

import asyncio
from collections import Counter

import pyarrow as pa

import calc_flow as cf


async def main() -> None:
    schema = pa.schema([pa.field("x", pa.int64(), nullable=True)])
    a = cf.table_input("a", schema=schema)
    b = cf.table_input("b", schema=schema)
    batch = cf.Program(
        "union",
        mode="batch",
        outputs={
            "df": a.union_all(b),
            "sql": cf.sql("SELECT x FROM a UNION ALL SELECT x FROM b", a=a, b=b),
        },
    )
    av = pa.Table.from_arrays([pa.array([1, None, 1], type=pa.int64())], schema=schema)
    bv = pa.Table.from_arrays([pa.array([1, 2], type=pa.int64())], schema=schema)
    expected = Counter([1, None, 1, 1, 2])
    for result in (await batch.collect_async({"a": av, "b": bv})).values():
        assert Counter(result["x"].to_pylist()) == expected
    observed = {"df": [], "sql": []}
    async with batch.for_mode("stream").stream({"a": [av], "b": [bv]}) as events:
        async for event in events:
            observed[event.name].extend(event.table["x"].to_pylist())
    assert all(Counter(values) == expected for values in observed.values())
    print("DataFrame and SQL: five rows in batch and completed stream")


if __name__ == "__main__":
    asyncio.run(main())

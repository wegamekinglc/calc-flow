from __future__ import annotations

import asyncio

import pyarrow as pa

import calc_flow as cf


async def batches():
    yield pa.table({"value": [1, 2]})
    yield pa.table({"value": [3]})


async def main() -> None:
    source = cf.table_input("events", schema=pa.schema([("value", pa.int64())]))
    program = cf.Program(
        "branches",
        outputs={
            "double": source.select(value2=source["value"] * 2),
            "large": source.filter(source["value"] >= 2).select("value"),
        },
    )
    values = {"double": [], "large": []}
    async with program.stream({"events": batches()}) as results:
        async for output in results:
            values[output.name].extend(output.table.column(0).to_pylist())
    if values != {"double": [2, 4, 6], "large": [2, 3]}:
        raise RuntimeError(values)
    print(values)


if __name__ == "__main__":
    asyncio.run(main())

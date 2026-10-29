from __future__ import annotations

import asyncio
import inspect

import pyarrow as pa

import calc_flow as cf


def test_native_coroutines_skip_general_awaitable_inspection(monkeypatch) -> None:
    inspected: list[str] = []
    original_awaitable = inspect.isawaitable
    original_coroutine = inspect.iscoroutine

    def observe(original, value):
        code = getattr(value, "cr_code", None)
        if code is not None and code.co_name in ("_native_next", "_native_write"):
            inspected.append(code.co_name)
        return original(value)

    monkeypatch.setattr(
        inspect, "isawaitable", lambda v: observe(original_awaitable, v)
    )
    monkeypatch.setattr(
        inspect, "iscoroutine", lambda v: observe(original_coroutine, v)
    )

    async def run():
        table = pa.table({"value": [1, 2]})
        source = cf.table_input("events", schema=table.schema)

        async def batches():
            for _ in range(3):
                yield table

        async with source.select(output=source["value"] + 1).stream(
            batches(), config=cf.StreamRuntimeConfig(checkpointing=False)
        ) as results:
            output = [table async for table in results]
        assert [t["output"].to_pylist() for t in output] == [[2, 3]] * 3
        assert table["value"].to_pylist() == [1, 2]

    asyncio.run(run())
    assert inspected == []

from __future__ import annotations

import asyncio
import inspect
from contextvars import ContextVar

import pyarrow as pa
import pytest

import calc_flow as cf
import calc_flow.stream as stream_module


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


def test_coroutine_dispatch_preserves_task_factory_and_context() -> None:
    value = ContextVar("callback-context", default="caller")
    created: list[str] = []

    def factory(loop, coroutine, **kwargs):
        created.append(coroutine.cr_code.co_name)
        return asyncio.Task(coroutine, loop=loop, **kwargs)

    async def batches():
        for item in range(3):
            assert value.get() == "caller"
            value.set("callback")
            yield pa.table({"value": [item]})

    async def run():
        loop = asyncio.get_running_loop()
        loop.set_task_factory(factory)
        try:
            source = cf.table_input("events", schema=pa.schema([("value", pa.int64())]))
            async with source.stream(
                batches(), config=cf.StreamRuntimeConfig(checkpointing=False)
            ) as results:
                output = [table["value"].to_pylist() async for table in results]
            assert output == [[0], [1], [2]]
            assert value.get() == "caller"
        finally:
            loop.set_task_factory(None)

    asyncio.run(run())
    assert "_native_next" in created
    assert "_native_write" in created


def test_sink_callback_error_releases_stream_resources(monkeypatch) -> None:
    closed: list[bool] = []

    async def batches():
        try:
            while True:
                yield pa.table({"value": [1]})
        finally:
            closed.append(True)

    async def fail(sink, batch):
        raise ValueError("sink dispatch failed")

    monkeypatch.setattr(stream_module._QueueSink, "write", fail)

    async def run():
        before = asyncio.all_tasks()
        source = cf.table_input("events", schema=pa.schema([("value", pa.int64())]))
        results = source.stream(
            batches(), config=cf.StreamRuntimeConfig(checkpointing=False)
        )
        with pytest.raises(cf.StreamingRuntimeError):
            async with results:
                await results.__anext__()
        assert closed == [True]
        assert results.job.status()["task_count"] == 0
        assert asyncio.all_tasks() == before

    asyncio.run(run())

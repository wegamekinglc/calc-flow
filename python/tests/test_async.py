from __future__ import annotations

import asyncio
from threading import Event

import pyarrow as pa
import pytest

from calc_flow import Batch, PipelineBuilder, Runtime


def _batch(**columns: list[int]) -> Batch:
    return Batch.from_pyarrow(pa.table(columns))


def test_execute_async_does_not_block_the_python_loop() -> None:
    async def exercise() -> None:
        events: list[str] = []
        plan = (
            PipelineBuilder("totals")
            .expression("calc", "total = a + b")
            .compile_batch()
        )

        async def heartbeat() -> None:
            await asyncio.sleep(0)
            events.append("heartbeat")

        pulse = asyncio.create_task(heartbeat())
        execution = asyncio.create_task(
            plan.execute_async({"input": _batch(a=list(range(20_000)), b=[2] * 20_000)})
        )
        result = await execution
        events.append("execution")
        await pulse

        assert events == ["heartbeat", "execution"]
        assert result.metadata["pipeline_name"] == "totals"

    asyncio.run(exercise())


def test_blocking_execute_rejects_running_loop_before_native_entry() -> None:
    async def exercise() -> None:
        plan = (
            PipelineBuilder("totals")
            .expression("calc", "total = a + b")
            .compile_batch()
        )
        with pytest.raises(
            RuntimeError,
            match=r"execute\(\) cannot run inside an event loop; use execute_async\(\)",
        ):
            plan.execute({"input": _batch(a=[1], b=[2])})

        result = await plan.execute_async({"input": _batch(a=[1], b=[2])})
        assert result.outputs["output"].to_pyarrow()["total"].to_pylist() == [3]

    asyncio.run(exercise())


def test_concurrent_async_calls_serialize_and_both_complete() -> None:
    async def exercise() -> None:
        plan = (
            PipelineBuilder("concurrent")
            .expression("calc", "b = a + 1")
            .compile_batch()
        )
        first, second = await asyncio.gather(
            plan.execute_async({"input": _batch(a=[1])}),
            plan.execute_async({"input": _batch(a=[10])}),
        )
        assert first.outputs["output"].to_pyarrow()["b"].to_pylist() == [2]
        assert second.outputs["output"].to_pyarrow()["b"].to_pylist() == [11]
        assert first.metadata["run_id"] != second.metadata["run_id"]

    asyncio.run(exercise())


def test_cancelling_async_execution_preserves_plan_recovery() -> None:
    async def exercise() -> None:
        entered = asyncio.Event()
        release = Event()
        loop = asyncio.get_running_loop()

        def await_release(values: pa.Array) -> pa.Array:
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(timeout=10):
                raise TimeoutError("cancellation test did not release the query")
            return values

        runtime = Runtime()
        runtime.register_scalar_udf(
            provider="python",
            name="await_release",
            version="1",
            input_types=("int64",),
            return_type="int64",
            volatility="volatile",
            function=await_release,
        )
        plan = (
            PipelineBuilder("cancel")
            .sql(
                "query",
                "SELECT sum(await_release(left.value) * right.value) AS total "
                "FROM left CROSS JOIN right",
                aliases=("left", "right"),
                udfs=(("python", "await_release", "1"),),
            )
            .compile_batch(runtime)
        )
        value = _batch(value=[1])
        running = asyncio.create_task(
            plan.execute_async({"left": value, "right": value})
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=10)
            assert running.cancel()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await running

        recovered = await plan.execute_async(
            {"left": _batch(value=[2]), "right": _batch(value=[3])}
        )
        assert recovered.outputs["output"].to_pyarrow()["total"].to_pylist() == [6]

    asyncio.run(exercise())

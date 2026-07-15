"""Execute a compiled v2 plan without blocking an asyncio event loop."""

from __future__ import annotations

import asyncio

import pyarrow as pa

from calc_flow import Batch, PipelineBuilder


async def run() -> None:
    plan = (
        PipelineBuilder("async-example").expression("calc", "total = a + b").compile()
    )
    heartbeat = asyncio.create_task(asyncio.sleep(0, result="event loop remained live"))
    execution = asyncio.create_task(
        plan.execute_async(
            {"input": Batch.from_pyarrow(pa.table({"a": [1, 3], "b": [2, 4]}))}
        )
    )
    print(await heartbeat)
    result = await execution
    print(result.outputs["output"].to_pyarrow().to_pylist())


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()

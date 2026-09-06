"""Execute a compiled batch plan without blocking an asyncio event loop."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pyarrow as pa

from calc_flow import Batch, ExecutionOptions, PipelineBuilder


async def run() -> None:
    plan = (
        PipelineBuilder("async-example")
        .expression("calc", "total = a + b")
        .compile_batch()
    )
    options = ExecutionOptions(
        settings={"request": {"source": "async-example"}},
        deadline=datetime.now(UTC) + timedelta(seconds=30),
    )
    heartbeat = asyncio.create_task(asyncio.sleep(0, result="event loop remained live"))
    execution = asyncio.create_task(
        plan.execute_async(
            {"input": Batch.from_pyarrow(pa.table({"a": [1, 3], "b": [2, 4]}))},
            options=options,
        )
    )
    print(await heartbeat)
    result = await execution
    output = result.outputs["output"].to_pyarrow()
    assert output["total"].to_pylist() == [3, 7]
    print(output.to_pylist())


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()

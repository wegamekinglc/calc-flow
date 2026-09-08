"""Compute Python expressions without blocking an asyncio event loop."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pyarrow as pa

import calc_flow as cf


async def run() -> None:
    options = cf.ExecutionOptions(
        settings={"request": {"source": "async-example"}},
        deadline=datetime.now(UTC) + timedelta(seconds=30),
    )
    heartbeat = asyncio.create_task(asyncio.sleep(0, result="event loop remained live"))
    execution = asyncio.create_task(
        cf.compute_async(
            pa.table({"a": [1, 3], "b": [2, 4]}),
            lambda t: t.select(total=t["a"] + t["b"]),
            options=options,
        )
    )
    print(await heartbeat)
    output = await execution
    if output["total"].to_pylist() != [3, 7]:
        raise RuntimeError(f"unexpected async totals: {output.to_pylist()}")
    print(output.to_pylist())


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()

"""Observe publication of ASOF and optional projection watermarks."""

from __future__ import annotations

import asyncio

FINAL_FRONTIER_POLL_SECONDS = 0.001


async def await_asof_frontier(
    job, micros: int, *, output_node_id: str | None = None
) -> dict:
    """Observe ASOF publication and the optional completed output projection cut."""

    while True:
        status = job.status()
        if status["state"] not in ("running", "draining"):
            raise RuntimeError(f"ASOF terminated before final frontier: {status}")
        joins = tuple(status["stream_asof_joins"].values())
        if len(joins) != 1:
            raise ValueError("ASOF acceptance requires exactly one operator status")
        frontier = joins[0].get("output_watermark_micros")
        if (
            frontier is not None
            and frontier >= micros
            and _projection_completed(job, output_node_id, micros)
        ):
            return status
        await asyncio.sleep(FINAL_FRONTIER_POLL_SECONDS)


def _projection_completed(job, output_node_id: str | None, micros: int) -> bool:
    completed = (
        micros
        if output_node_id is None
        else job._inner._benchmark_dispatch_frontier(output_node_id)
    )
    return completed is not None and completed >= micros

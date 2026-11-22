"""Time an eligible rolling result on an open, default Python event-time stream.

Run with `python -m benchmarks.open_stream_latency --build release.json
--output samples.json` from the checkout with the sealed wheel on PYTHONPATH.
See the command help for the sampling defaults.
"""

from __future__ import annotations

import argparse
import asyncio
from asyncio import sleep
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pyarrow as pa

import calc_flow as cf
from benchmarks.native_release_identity import file_hash, release_identity
from benchmarks.warm_stream import SCHEMA
from scripts.toolkit import linear_percentile, worker_environment, write_json

OUTPUT = pa.schema(
    [pa.field("sequence", pa.uint64(), nullable=False), ("mean", pa.float64())]
)


def workload() -> tuple[cf.TableExpr, tuple[pa.Table, ...]]:
    data = pa.table(
        {
            "event_time": [1_000_000, 1_000_001, 1_000_002],
            "sequence": [0, 1, 2],
            "symbol": ["A"] * 3,
            "price": [1.0, 2.0, 3.0],
        },
        schema=SCHEMA,
    )
    source = cf.table_input(
        "ticks",
        schema=SCHEMA,
        entity_by=("symbol",),
        event_time="event_time",
        sequence_by=("sequence",),
    )
    query = source.select(
        "sequence", mean=cf.ts.mean(source["price"], window=cf.duration(60))
    )
    return query, tuple(data.slice(i, 1) for i in range(3))


def check_output(result: pa.Table, sequence: int) -> None:
    expected = {"sequence": [sequence], "mean": [(1.0, 1.5)[sequence]]}
    if not result.schema.equals(OUTPUT, check_metadata=True) or (
        result.to_pydict() != expected
    ):
        raise ValueError(f"expected {expected} with schema {OUTPUT}, got {result}")


def arrival_schedule(samples: int, seed: int) -> list[tuple[str, int]]:
    if samples < 1:
        raise ValueError("samples per phase must be positive")
    generator = np.random.default_rng(seed)
    plan = []
    for index in range(samples):
        offset = int(generator.integers(1, 100_000_001))
        pair = [("immediate", 0), ("seeded", offset)]
        plan.extend(reversed(pair) if index % 2 else pair)
    return plan


async def _open_source(queue: asyncio.Queue[pa.Table]):
    while True:
        yield await queue.get()


async def verify_eligibility(query: cf.TableExpr, rows: tuple[pa.Table, ...]) -> None:
    queue: asyncio.Queue[pa.Table] = asyncio.Queue()
    async with query.stream(_open_source(queue)) as stream:
        queue.put_nowait(rows[0])
        queue.put_nowait(rows[1])
        check_output(await asyncio.wait_for(stream.__anext__(), 5), 0)
        pending = asyncio.ensure_future(stream.__anext__())
        try:
            await asyncio.wait([pending], timeout=0.05)
            if pending.done():
                pending.result()
                raise ValueError("second result arrived before the third input")
            queue.put_nowait(rows[2])
            check_output(await asyncio.wait_for(pending, 5), 1)
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


async def observe(
    query: cf.TableExpr, rows: tuple[pa.Table, ...], offset_ns: int
) -> dict[str, int]:
    queue: asyncio.Queue[pa.Table] = asyncio.Queue()
    async with query.stream(_open_source(queue)) as stream:
        queue.put_nowait(rows[0])
        input_at = perf_counter_ns()
        queue.put_nowait(rows[1])
        first = await asyncio.wait_for(stream.__anext__(), 5)
        warm_delivered = perf_counter_ns()
        if offset_ns:
            await sleep(offset_ns / 1e9)
        eligible_at = perf_counter_ns()
        queue.put_nowait(rows[2])
        second = await asyncio.wait_for(stream.__anext__(), 5)
        delivered = perf_counter_ns()
        check_output(first, 0)
        check_output(second, 1)
    return {
        "requested_offset_ns": offset_ns,
        "actual_offset_ns": eligible_at - warm_delivered,
        "eligible_to_arrow_ns": delivered - eligible_at,
        "input_to_arrow_ns": delivered - input_at,
    }


async def collect(samples: int, seed: int) -> dict:
    plan = arrival_schedule(samples, seed)
    query, rows = workload()
    await observe(query, rows, 0)
    await verify_eligibility(query, rows)
    observations = []
    for phase, offset in plan:
        sample = await observe(query, rows, offset)
        observations.append({"phase": phase, **sample})
    summary = {}
    for phase in ("immediate", "seeded"):
        elapsed = [
            row["eligible_to_arrow_ns"] for row in observations if row["phase"] == phase
        ]
        summary[phase] = {
            "p50_ns": linear_percentile(elapsed, 0.50),
            "p95_ns": linear_percentile(elapsed, 0.95),
        }
    return {"seed": seed, "observations": observations, "eligible_to_arrow": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--samples", type=int, default=20, help="observations per phase"
    )
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    identity = release_identity(args.build)
    environment = worker_environment()
    report = asyncio.run(collect(args.samples, args.seed))
    write_json(
        args.output,
        {
            "contract": "calc-flow.open-stream-latency.v1",
            "build": identity,
            "harness_sha256": file_hash(Path(__file__)),
            "environment": environment,
            **report,
        },
        exclusive=True,
    )


if __name__ == "__main__":
    main()

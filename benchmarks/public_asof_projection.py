"""Compare public stream ASOF with one admitted run on each side."""

from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from pathlib import Path
from time import perf_counter_ns

import pyarrow as pa

import calc_flow as cf
from scripts.toolkit import worker_environment, write_json

ROWS = 1024
SCHEMA = pa.schema(
    [
        pa.field("key", pa.int64(), False),
        pa.field("time", pa.timestamp("us", tz="UTC"), False),
        pa.field("seq", pa.int64(), False),
    ]
)
OUTPUT = pa.schema(
    [
        pa.field(f"{side}__{field.name}", field.type, nullable=side == "right")
        for side in ("left", "right")
        for field in SCHEMA
    ]
)


def fixture():
    probes = pa.table(
        [[int(t != 1022) for t in range(ROWS)], list(range(ROWS)), list(range(ROWS))],
        schema=SCHEMA,
    )
    times = [t for t in range(4, ROWS, 4) for _ in (1, 2)]
    history = pa.table(
        [[1] * len(times), times, [1, 2] * (len(times) // 2)], schema=SCHEMA
    )
    left, right = (
        cf.table_input(
            name,
            schema=SCHEMA,
            entity_by=["key"],
            event_time="time",
            sequence_by=["seq"],
        )
        for name in ("left", "right")
    )
    joined = left.stream_asof_join(
        right,
        tolerance=timedelta(microseconds=2),
        limits=cf.AsofStateLimits(4096, 8 << 20),
    )
    return joined.select(*OUTPUT.names), probes, history


async def one(table):
    yield table


async def observe(declaration, probes, history):
    streams = {"left": one(probes), "right": one(history)}
    policies = {side: cf.SourceProvidedWatermarks() for side in streams}
    config = cf.StreamRuntimeConfig(checkpointing=False)
    start = perf_counter_ns()
    stream = declaration.stream(streams, watermarks=policies, config=config)
    async with stream:
        outputs = [part async for part in stream]
    table = pa.concat_tables(outputs)
    elapsed = perf_counter_ns() - start
    return elapsed, table, stream.job.status()


def validate(table, status) -> dict:
    frontier = [
        (t - t % 4) if t >= 4 and t % 4 <= 2 and t != 1022 else None
        for t in range(ROWS)
    ]
    expected = pa.table(
        [
            [int(t != 1022) for t in range(ROWS)],
            list(range(ROWS)),
            list(range(ROWS)),
            [1 if t is not None else None for t in frontier],
            frontier,
            [2 if t is not None else None for t in frontier],
        ],
        schema=OUTPUT,
    )
    if not table.equals(expected, check_metadata=True):
        raise ValueError("ASOF projection differs from the ordered six-column oracle")
    (join,) = status["stream_asof_joins"].values()
    if (
        status["state"] != "completed"
        or status["task_count"] != 0
        or (join["left"]["accepted_rows"], join["right"]["accepted_rows"])
        != (1024, 510)
        or (join["matched_rows"], join["unmatched_rows"], join["emitted_left_rows"])
        != (764, 260, 1024)
        or join["pending_left_rows"]
    ):
        raise ValueError(f"unexpected final ASOF status: {join}")
    return {
        "rows": table.num_rows,
        "matched": 764,
        "unmatched": 260,
        "columns": table.column_names,
    }


def main() -> None:
    elapsed, table, status = asyncio.run(observe(*fixture()))
    proof = validate(table, status)
    write_json(
        Path(sys.argv[1]),
        {"stream_ns": elapsed, "proof": proof, "environment": worker_environment()},
        exclusive=True,
    )


if __name__ == "__main__":
    main()

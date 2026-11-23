"""Time one public left anti-join release while both inputs remain active."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone

import pyarrow as pa

import calc_flow as cf
from calc_flow import _native

SCHEMA = pa.schema(
    [
        pa.field("key", pa.int64()),
        pa.field("time", pa.timestamp("us", tz="UTC"), False),
        pa.field("sequence", pa.uint64(), False),
    ]
)


def table(keys, times, sequence):
    return pa.Table.from_arrays(
        [
            pa.array(keys, type=SCHEMA.field("key").type),
            pa.array(times, type=SCHEMA.field("time").type),
            pa.array(sequence, type=SCHEMA.field("sequence").type),
        ],
        schema=SCHEMA,
    )


def declaration():
    left, right = (
        cf.table_input(
            name,
            schema=SCHEMA,
            entity_by=["sequence"],
            event_time="time",
            sequence_by=["sequence"],
        )
        for name in ("left", "right")
    )
    result = cf.table.stream_join(
        left,
        right,
        join_type="left_anti",
        left_keys=["key"],
        right_keys=["key"],
        left_event_time="time",
        right_event_time="time",
        bounds=cf.JoinTimeBounds(timedelta(0), timedelta(microseconds=2)),
        limits=cf.JoinStateLimits(512, 8 << 20, 32),
    )
    return cf.Program("active_anti_release", mode="stream", outputs={"absent": result})


async def feed(queue):
    while (event := await queue.get()) is not None:
        yield event


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(tables, expected):
    require(
        all(t.schema.equals(SCHEMA, check_metadata=True) for t in tables),
        "public output schema differs from the declared left schema",
    )
    require(
        pa.concat_tables(tables).equals(expected, check_metadata=True),
        "public output differs from the expected left rows",
    )


async def receive(outputs, count):
    tables = []
    received = 0
    while received < count:
        event = await outputs.__anext__()
        require(event.name == "absent", "unexpected public output name")
        tables.append(event.table)
        received += event.table.num_rows
    return tables


async def observe(measure):
    left = table([7] * 258 + [8, 7, None], [9] + [10] * 258 + [11, 11], range(261))
    right = table([8], [10], [0])
    equality_ack = table([None], [11], [261])
    left_queue, right_queue = asyncio.Queue(), asyncio.Queue()
    origin = datetime(1970, 1, 1, tzinfo=timezone.utc)
    cuts = [cf.Watermark(origin + timedelta(microseconds=v)) for v in (12, 13, 14)]
    config = cf.StreamRuntimeConfig(
        checkpointing=False, edge_budget=cf.EdgeBudget(max_rows=32, max_bytes=4096)
    )
    async with declaration().stream(
        {"left": feed(left_queue), "right": feed(right_queue)},
        watermarks={name: cf.SourceProvidedWatermarks() for name in ("left", "right")},
        config=config,
    ) as outputs:
        right_queue.put_nowait(right)
        for batch in left.to_batches(max_chunksize=32):
            left_queue.put_nowait(batch)

        verify(await receive(outputs, 1), left.slice(260, 1))
        right_queue.put_nowait(cuts[0])
        verify(await receive(outputs, 1), left.slice(0, 1))
        left_queue.put_nowait(equality_ack)
        verify(await receive(outputs, 1), equality_ack)
        started = time.perf_counter_ns() if measure else None
        right_queue.put_nowait(cuts[1])
        released = await receive(outputs, 257)
        elapsed = time.perf_counter_ns() - started if started is not None else None

        chunks = [item.num_rows for item in released]
        require(chunks == [32] * 8 + [1], f"unexpected chunk sizes {chunks}")
        verify(released, left.slice(1, 257))
        right_queue.put_nowait(cuts[2])
        verify(await receive(outputs, 1), left.slice(259, 1))
        right_queue.put_nowait(None)
        left_queue.put_nowait(None)
        require([event async for event in outputs] == [], "unexpected EOF output")

    return {
        "setup": [260, 0, 261],
        "chunks": chunks,
        "rows": sum(chunks),
        "tail": [259],
        "elapsed_ns": elapsed,
        "affinity": sorted(os.sched_getaffinity(0)),
        "native": _native.__file__,
        "tokio_threads": os.environ.get("TOKIO_WORKER_THREADS"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(asyncio.wait_for(observe(not args.verify_only), 15))))

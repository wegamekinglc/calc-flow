"""Measure public ASOF delivery after an untimed warm delivery on the same stream.

Run one worker with `python -m benchmarks.warm_asof_latency OUTPUT.json` in
a Python 3.11+ environment with the release wheel. Each observation times a
four-row right batch and its advancing watermark through receipt of eight eligible
left rows. One later probe remains pending until the next watermark.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter_ns

import pyarrow as pa

import calc_flow as cf
from benchmarks.asof_frontier import await_asof_frontier
from scripts.toolkit import worker_environment, write_json

INPUT = pa.schema(
    [
        pa.field("key", pa.int64(), nullable=False),
        pa.field("time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("seq", pa.int64(), nullable=False),
    ]
)
OUTPUT = pa.schema(
    [
        pa.field(f"{side}__{f.name}", f.type, nullable=side == "right")
        for side in ("left", "right")
        for f in INPUT
    ]
)


def prefix_oracle() -> pa.Table:
    # Explicit expected winners for ties at 8 and 12, the inclusive distance
    # of two, the unknown key at 12, and the expired right rows at 11 and 15.
    return pa.table(
        [
            [1, 1, 1, 1, 9, 1, 1, 1],
            range(8, 16),
            range(8, 16),
            [1, 1, 1, None, None, 1, 1, None],
            [8, 8, 8, None, None, 12, 12, None],
            [2, 2, 2, None, None, 4, 4, None],
        ],
        schema=OUTPUT,
    )


def check_rows(tables: list[pa.Table], expected: pa.Table) -> None:
    if not all(t.schema.equals(expected.schema, check_metadata=True) for t in tables):
        raise ValueError("ASOF output schema differs from the oracle")
    if not pa.concat_tables(tables).equals(expected, check_metadata=True):
        raise ValueError("ASOF output differs from the ordered row oracle")


async def _feed(queue: asyncio.Queue):
    while (event := await queue.get()) is not None:
        yield event


async def _receive(stream, rows: int) -> list[pa.Table]:
    tables = []
    while rows > 0:
        table = await stream.__anext__()
        tables.append(table)
        rows -= table.num_rows
    return tables


def _query() -> cf.TableExpr:
    left, right = (
        cf.table_input(
            name,
            schema=INPUT,
            entity_by=["key"],
            event_time="time",
            sequence_by=["seq"],
        )
        for name in ("left", "right")
    )
    return left.stream_asof_join(
        right,
        tolerance=timedelta(microseconds=2),
        limits=cf.AsofStateLimits(32, 8 << 20),
    )


def _watermark(micros: int) -> cf.Watermark:
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return cf.Watermark(epoch + timedelta(microseconds=micros))


async def _pending_at(stream, micros: int, expected: int) -> int:
    status = await await_asof_frontier(stream.job, micros)
    (join,) = status["stream_asof_joins"].values()
    pending = join["pending_left_rows"]
    if pending != expected or join["left"]["ended"] or join["right"]["ended"]:
        raise ValueError(
            f"ASOF frontier {micros} requires {expected} pending and open inputs"
        )
    return pending


async def _warm(stream, queues, end) -> None:
    probes = [0, *range(8, 17)]
    left = pa.table([[9 if t == 12 else 1 for t in probes], probes, probes], INPUT)
    warm = pa.table([[1], [0], [0]], INPUT)
    expected = pa.table([[1], [0], [0], [1], [0], [0]], OUTPUT)
    queues["left"].put_nowait(left)
    queues["left"].put_nowait(end)
    queues["right"].put_nowait(warm)
    queues["right"].put_nowait(_watermark(1))
    check_rows(await _receive(stream, 1), expected)
    await _pending_at(stream, 0, 9)


async def _finish(stream, queues, end) -> None:
    tail = pa.table([[1], [16], [16], [None], [None], [None]], OUTPUT)
    queues["right"].put_nowait(end)
    check_rows(await _receive(stream, 1), tail)
    for queue in queues.values():
        queue.put_nowait(None)
    async for _ in stream:
        raise ValueError("ASOF produced an unexpected output at EOF")


async def observe() -> dict:
    update = pa.table([[1] * 4, [8, 8, 12, 12], [1, 2, 3, 4]], INPUT)
    expected = prefix_oracle()
    cut, end = _watermark(16), _watermark(17)
    queues = {name: asyncio.Queue() for name in ("left", "right")}
    async with _query().stream(
        {name: _feed(q) for name, q in queues.items()},
        watermarks={name: cf.SourceProvidedWatermarks() for name in queues},
        config=cf.StreamRuntimeConfig(checkpointing=False),
    ) as stream:
        await _warm(stream, queues, end)
        started = perf_counter_ns()
        queues["right"].put_nowait(update)
        queues["right"].put_nowait(cut)
        prefix = await _receive(stream, expected.num_rows)
        elapsed = perf_counter_ns() - started

        check_rows(prefix, expected)
        pending = await _pending_at(stream, 15, 1)
        await _finish(stream, queues, end)

    nulls = expected["right__key"].null_count
    return {
        "delivery_ns": elapsed,
        "proof": {
            "rows": expected.num_rows,
            "matched": expected.num_rows - nulls,
            "unmatched": nulls,
            "pending_at_cut": pending,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    env = worker_environment()
    observation = asyncio.run(asyncio.wait_for(observe(), timeout=15))
    write_json(args.output, {**observation, "environment": env}, exclusive=True)


if __name__ == "__main__":
    main()

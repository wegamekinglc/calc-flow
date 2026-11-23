"""Measure a ready sparse update through its public Arrow result on a live graph."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from time import perf_counter_ns

import pyarrow as pa

from benchmarks.sparse_changelog import (
    OUTPUT,
    _positions_after,
    load_fixture,
    position_stream,
    validate,
)
from scripts.toolkit import worker_environment, write_json


async def observe_updates(batches: tuple[pa.Table, ...], mode: str):
    if mode not in ("snapshot", "changelog"):
        raise ValueError("expected snapshot or changelog")
    queue = asyncio.Queue(maxsize=1)

    async def source():
        while (batch := await queue.get()) is not None:
            yield batch

    positions, config = position_stream(
        batches[0].schema, changelog=mode == "changelog"
    )
    outputs = []
    clocks = []
    async with positions.stream(source(), config=config) as stream:
        for index, batch in enumerate(batches):
            started = perf_counter_ns()
            queue.put_nowait(batch)
            table = await stream.__anext__()
            elapsed = perf_counter_ns() - started
            outputs.append(table)
            if index:
                clocks.append(elapsed)
        queue.put_nowait(None)
        async for _unexpected in stream:
            raise ValueError("unexpected output after final sparse update")
    return clocks, outputs


def validate_snapshots(batches: tuple[pa.Table, ...], outputs: list[pa.Table]):
    schema = OUTPUT.remove(3)
    positions = {}
    for index, (batch, output) in enumerate(zip(batches, outputs, strict=True)):
        positions = _positions_after(positions, batch)
        if not output.schema.equals(schema, check_metadata=True):
            raise ValueError(f"incorrect snapshot schema at batch {index}")
        actual = zip(*output.to_pydict().values())
        if list(actual) != [(*key, value) for key, value in positions.items()]:
            raise ValueError(f"incorrect ordered snapshot at batch {index}")
    return [table.num_rows for table in outputs]


def main() -> None:
    mode, fixture, destination = sys.argv[1:]
    batches = load_fixture(Path(fixture))
    clocks, outputs = asyncio.run(observe_updates(batches, mode))
    rows = (validate if mode == "changelog" else validate_snapshots)(batches, outputs)
    expected = [10_000, *([160 if mode == "changelog" else 10_000] * 20)]
    if rows != expected or len(clocks) != 20 or min(clocks) <= 0:
        raise ValueError("incorrect output rows or update clock count")
    write_json(
        Path(destination),
        {
            "mode": mode,
            "update_ns": clocks,
            "output_rows": rows,
            "environment": worker_environment(),
        },
        exclusive=True,
    )


if __name__ == "__main__":
    main()

"""Drain both late-output branches with one owned iterator."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pyarrow as pa

import calc_flow as cf

BASE = datetime(2026, 1, 1, tzinfo=UTC)
schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("seq", pa.uint64(), nullable=False),
        pa.field("x", pa.float64()),
    ]
)
events = cf.table_input(
    "events", schema=schema, entity_by=["symbol"], event_time="ts", sequence_by=["seq"]
)
calculation = events.with_columns(avg=cf.ts.mean(events["x"], window=cf.rows(2)))
pair = cf.with_late_output(calculation)
program = cf.Program("late", outputs={"normal": pair.output, "late": pair.late})


async def batches():
    yield cf.Watermark(BASE + timedelta(microseconds=1))
    times = [BASE, BASE + timedelta(microseconds=2)]
    rows = {"ts": times, "symbol": ["a", "a"], "seq": [0, 1], "x": [10.0, 20.0]}
    yield pa.table(rows, schema=schema)
    yield cf.Watermark(BASE + timedelta(microseconds=2))


async def main():
    values = {"normal": [], "late": []}
    async with program.stream(
        {"events": batches()}, watermarks=cf.SourceProvidedWatermarks()
    ) as results:
        async for event in results:
            values[event.name].extend(event.table["x"].to_pylist())
    if values != {"normal": [20.0], "late": [10.0]}:
        raise RuntimeError(values)
    print(values)


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(main(), 30))

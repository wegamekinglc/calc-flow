"""Cold StreamingRunner adapter: prebuilt input, real tasks and finalization."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pyarrow as pa

from benchmarks.warm_stream import BASE, BASE_MICROS, _InteractiveSource
from calc_flow import (
    Batch,
    Cursor,
    Data,
    EdgeBudget,
    ManagedCheckpointRuntime,
    Runtime,
    SinkBinding,
    SourceBinding,
    SourceProvidedWatermarks,
    StreamingRunner,
    StreamRuntimeConfig,
    Watermark,
)
from calc_flow.symbolic import FeatureSet, Field, Program, rows, table_input, ts
from scripts.benchmark_suite.catalog import BATCH_ROWS


def stream_plan(scenario: str):
    quotes = table_input(
        "quotes",
        schema=(
            Field("event_time", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("price", "float64", nullable=False),
        ),
        entity_by=("symbol",),
        event_time="event_time",
        sequence_by=("sequence",),
    )
    slow = ts.mean(quotes["price"], window=rows(20), min_periods=20)
    value = (
        slow
        if scenario == "sma20"
        else ts.mean(quotes["price"], window=rows(5), min_periods=5) - slow
    )
    output = quotes.with_columns(FeatureSet((("value", value),)))
    return Program(
        "suite-stream", inputs=(quotes,), outputs=(("result", output),)
    ).compile_stream(Runtime())


def stream_events(table: pa.Table, entities: int) -> tuple:
    # Never finalize half an entity tick before the next data batch.
    size = max(entities, BATCH_ROWS // entities * entities)
    events = []
    for start in range(0, table.num_rows, size):
        part = table.slice(start, size)
        end = start + part.num_rows
        events.append(
            Data(
                Batch.from_pyarrow(part), Cursor(end.to_bytes(8, "big"), {"rows": end})
            )
        )
        micros = part["event_time"][-1].value - int(BASE_MICROS) + 1
        events.append(Watermark(BASE + timedelta(microseconds=micros)))
    return (*events, None)


class _CollectSink:
    def __init__(self) -> None:
        self.tables: list[pa.Table] = []

    async def open(self) -> None:
        return None

    async def write(self, batch: Batch) -> None:
        self.tables.append(batch.to_pyarrow())

    async def close(self) -> None:
        return None


async def run_stream(plan, events: tuple, root: Path) -> pa.Table:
    source = _InteractiveSource(max_batch_rows=BATCH_ROWS)
    sink = _CollectSink()
    job = await StreamingRunner(
        plan,
        {"input": SourceBinding(source, watermark_policy=SourceProvidedWatermarks())},
        {"output": [SinkBinding.ordinary("suite", sink)]},
        ManagedCheckpointRuntime(root),
        config=StreamRuntimeConfig(
            checkpoint_interval=timedelta(hours=24),
            edge_budget=EdgeBudget(max_rows=BATCH_ROWS, max_bytes=64 << 20),
        ),
    ).start_async()
    try:
        for event in events:
            await source.push(event)
        outcome = await asyncio.wait_for(job.wait_async(), timeout=600)
        if outcome.state != "completed":
            raise RuntimeError(f"stream failed: {outcome.errors}")
        return pa.concat_tables(sink.tables)
    finally:
        await job.cancel_async()

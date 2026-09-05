"""Ready StreamingRunner adapter: empty state, real tasks and finalization."""

from __future__ import annotations

import asyncio
import time
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


class _ReadySource(_InteractiveSource):
    def __init__(self) -> None:
        super().__init__(max_batch_rows=BATCH_ROWS)
        self.ready = asyncio.Event()

    async def next(self) -> Data | Watermark | None:
        # The first poll proves that the runtime's startup data gate opened.
        self.ready.set()
        return await super().next()


class _CollectSink:
    def __init__(self, expected_rows: int) -> None:
        self.expected_rows = expected_rows
        self.rows = 0
        self.tables: list[pa.Table] = []
        self.opened = asyncio.Event()
        self.complete = asyncio.Event()

    async def open(self) -> None:
        self.opened.set()

    async def write(self, batch: Batch) -> None:
        table = batch.to_pyarrow()
        self.tables.append(table)
        self.rows += table.num_rows
        if self.rows >= self.expected_rows:
            self.complete.set()

    async def close(self) -> None:
        return None


async def _measure_ready(
    source: _ReadySource, sink: _CollectSink, events: tuple
) -> tuple[pa.Table, float]:
    await asyncio.wait_for(source.ready.wait(), timeout=30)
    if not source.opened.is_set() or not sink.opened.is_set() or sink.rows:
        raise RuntimeError("stream must be ready with empty state before timing")
    started = time.perf_counter_ns()
    for event in events:
        await source.push(event)
    await asyncio.wait_for(sink.complete.wait(), timeout=600)
    if sink.rows != sink.expected_rows:
        raise RuntimeError("stream output row count differs from the timed workload")
    table = pa.concat_tables(sink.tables)
    return table, (time.perf_counter_ns() - started) / 1e9


async def run_stream(
    plan, events: tuple, root: Path, expected_rows: int
) -> tuple[pa.Table, float]:
    if not events or events[-1] is not None:
        raise ValueError("stream input must end with an EOF marker")
    timed_events = events[:-1]
    source = _ReadySource()
    sink = _CollectSink(expected_rows)
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
        table, seconds = await _measure_ready(source, sink, timed_events)
        # Complete and verify the job, but do not time EOF/shutdown bookkeeping.
        await source.push(None)
        outcome = await asyncio.wait_for(job.wait_async(), timeout=600)
        if outcome.state != "completed":
            raise RuntimeError(f"stream failed: {outcome.errors}")
        if sink.rows != expected_rows:
            raise RuntimeError("stream output row count changed after the timed result")
        return table, seconds
    finally:
        await job.cancel_async()

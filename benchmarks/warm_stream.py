"""Reproducible warm-state measurements through the actual StreamingRunner."""

from __future__ import annotations

import asyncio
import gc
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from benchmarks.rolling_indicator_comparison import (
    _native_program,
    expected_dual_sma_spread,
    expected_rolling_mean,
)
from calc_flow import (
    Batch,
    Cursor,
    Data,
    EdgeBudget,
    ManagedCheckpointRuntime,
    NativeWatermarkCapability,
    ReplayPositioning,
    Runtime,
    SinkBinding,
    SourceBinding,
    SourceCapabilities,
    SourceDeliveryCapability,
    SourceProvidedWatermarks,
    StreamingRunner,
    StreamRuntimeConfig,
    Watermark,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)
BASE_MICROS = np.int64(int(BASE.timestamp() * 1_000_000))
SCHEMA = pa.schema(
    [
        pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("sequence", pa.uint64(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
    ]
)
RTOL = 1e-10
ATOL = 1e-10


def _prices(sequence: np.ndarray, entities: int) -> np.ndarray:
    entity_index = sequence % np.uint64(entities)
    return (
        100.0
        + ((sequence * np.uint64(17)) % np.uint64(1_000)).astype(np.float64) / 100.0
        + entity_index.astype(np.float64) / 100.0
    )


def _segment(start: int, rows: int, entities: int) -> pa.Table:
    sequence = np.arange(start, start + rows, dtype=np.uint64)
    entity_index = sequence % np.uint64(entities)
    positions = sequence // np.uint64(entities)
    symbols = np.asarray([f"S{index:03d}" for index in range(entities)])
    return pa.table(
        {
            "event_time": pa.array(
                BASE_MICROS + positions.astype(np.int64) * np.int64(1_000_000),
                type=pa.timestamp("us", tz="UTC"),
            ),
            "sequence": sequence,
            "symbol": symbols[entity_index.astype(np.int64)],
            "price": _prices(sequence, entities),
        },
        schema=SCHEMA,
    )


class _InteractiveSource:
    def __init__(self, *, max_batch_rows: int) -> None:
        self._events: asyncio.Queue[Data | Watermark | None] = asyncio.Queue()
        self.opened = asyncio.Event()
        self.data_at = 0
        self.watermark_at = 0
        self._max_batch_rows = max_batch_rows

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.UNSUPPORTED,
            SourceDeliveryCapability.LOSSY,
            max_batch_rows=self._max_batch_rows,
            max_batch_bytes=32 * 1024 * 1024,
            native_watermarks=NativeWatermarkCapability.EMITS_NATIVE,
        )

    async def open(self, cursor: Cursor | None) -> None:
        if cursor is not None:
            raise RuntimeError("diagnostic source does not support replay")
        self.opened.set()

    async def next(self) -> Data | Watermark | None:
        event = await self._events.get()
        if isinstance(event, Data):
            self.data_at = time.perf_counter_ns()
        elif isinstance(event, Watermark):
            self.watermark_at = time.perf_counter_ns()
        return event

    async def close(self) -> None:
        return None

    async def push(self, event: Data | Watermark | None) -> None:
        await self._events.put(event)


class _QueueSink:
    def __init__(self) -> None:
        self._tables: asyncio.Queue[pa.Table] = asyncio.Queue()
        self.entered_at = 0
        self.convert_ns = 0

    async def open(self) -> None:
        return None

    async def write(self, batch: Batch) -> None:
        self.entered_at = time.perf_counter_ns()
        table = batch.to_pyarrow()
        self.convert_ns = time.perf_counter_ns() - self.entered_at
        if table.num_rows:
            await self._tables.put(table)

    async def close(self) -> None:
        return None

    async def receive(self, timeout: float = 120.0) -> pa.Table:
        return await asyncio.wait_for(self._tables.get(), timeout=timeout)


def _prepared_events(start: int, rows: int, entities: int) -> tuple[Data, Watermark]:
    end = start + rows
    cursor = Cursor(end.to_bytes(8, "big"), {"offset": end})
    data = Data(Batch.from_pyarrow(_segment(start, rows, entities)), cursor)
    watermark_position = (end - 1) // entities
    watermark = Watermark(BASE + timedelta(seconds=watermark_position))
    return data, watermark


async def _push_and_receive(
    source: _InteractiveSource,
    sink: _QueueSink,
    data: Data,
    watermark: Watermark,
) -> tuple[pa.Table, float]:
    started = time.perf_counter_ns()
    await source.push(data)
    await source.push(watermark)
    table = await sink.receive()
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
    return table, elapsed


def _expected(
    *,
    start: int,
    rows: int,
    entities: int,
    indicator: str,
    fast_window: int,
    window: int,
) -> np.ndarray:
    context_rows = min(start, (window - 1) * entities)
    context_start = start - context_rows
    sequence = np.arange(context_start, start + rows, dtype=np.uint64)
    prices = _prices(sequence, entities)
    if indicator == "dual_sma_spread":
        values = expected_dual_sma_spread(
            prices,
            entities=entities,
            fast_window=fast_window,
            slow_window=window,
        )
    else:
        values = expected_rolling_mean(prices, entities=entities, window=window)
    return values[context_rows:]


def _validate_output(
    table: pa.Table,
    *,
    start: int,
    rows: int,
    entities: int,
    indicator: str,
    fast_window: int,
    window: int,
) -> dict[str, Any]:
    if table.num_rows != rows:
        raise RuntimeError(f"expected {rows} output rows, observed {table.num_rows}")
    output_column = (
        "dual_sma_spread" if indicator == "dual_sma_spread" else "moving_average"
    )
    sequence = table["sequence"].combine_chunks().to_numpy(zero_copy_only=False)
    order = np.arange(rows)
    wanted_sequence = np.arange(start, start + rows, dtype=np.uint64)
    if not np.array_equal(sequence[order], wanted_sequence):
        raise RuntimeError("stream output sequence does not match appended rows")
    if not table.select(SCHEMA.names).equals(_segment(start, rows, entities)):
        raise RuntimeError("stream output identity columns changed")
    actual = table[output_column].combine_chunks().to_numpy(zero_copy_only=False)[order]
    expected = _expected(
        start=start,
        rows=rows,
        entities=entities,
        indicator=indicator,
        fast_window=fast_window,
        window=window,
    )
    close = np.isclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True)
    absolute = np.abs(actual - expected)
    result = {
        "passed": bool(np.all(close)),
        "mismatched_values": int(np.count_nonzero(~close)),
        "max_absolute_error": float(absolute.max(initial=0.0)),
    }
    if not result["passed"]:
        raise RuntimeError(f"strict incremental correctness failed: {result}")
    return result


def _percentile(samples: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(samples), percentile))


def _summary(samples: list[float], rows: int) -> dict[str, Any]:
    median = statistics.median(samples)
    mean = statistics.mean(samples)
    standard_deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "median_seconds": median,
        "p25_seconds": _percentile(samples, 25),
        "p75_seconds": _percentile(samples, 75),
        "p95_seconds": _percentile(samples, 95),
        "mad_seconds": statistics.median(abs(value - median) for value in samples),
        "cv": standard_deviation / mean if mean else 0.0,
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "throughput_rows_per_second": rows / median,
        "samples_seconds": samples,
    }


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    history_rows: int = 1_024_000
    append_rows: int = 64
    entities: int = 64
    indicator: str = "rolling_mean"
    window: int = 20
    fast_window: int = 5
    history_segment_rows: int = 64_000

    def __post_init__(self) -> None:
        sizes = (
            self.history_rows,
            self.append_rows,
            self.entities,
            self.window,
            self.fast_window,
            self.history_segment_rows,
        )
        if any(type(value) is not int or value <= 0 for value in sizes):
            raise ValueError("scenario sizes must be positive integers")
        if any(
            value % self.entities
            for value in (
                self.history_rows,
                self.append_rows,
                self.history_segment_rows,
            )
        ):
            raise ValueError(
                "history and increments must contain complete entity ticks"
            )
        if self.fast_window > self.window:
            raise ValueError("fast_window must not exceed window")
        if self.indicator not in ("rolling_mean", "dual_sma_spread"):
            raise ValueError("unsupported warm indicator")


class WarmScenario:
    """Own one continuously advancing runner; no startup or IPC is timed."""

    def __init__(
        self,
        config: ScenarioConfig,
        source: _InteractiveSource,
        sink: _QueueSink,
        job: Any,
    ) -> None:
        self.config = config
        self.source = source
        self.sink = sink
        self.job = job
        self.position = 0
        self.warm_seconds = 0.0
        self.before_status: dict[str, Any] = {}

    @classmethod
    async def start(cls, config: ScenarioConfig, state_root: Path) -> WarmScenario:
        plan = _native_program(
            config.window,
            indicator=config.indicator,
            fast_window=config.fast_window,
        ).compile_stream(Runtime())
        maximum = max(config.history_segment_rows, config.append_rows)
        source = _InteractiveSource(max_batch_rows=maximum)
        sink = _QueueSink()
        job = await StreamingRunner(
            plan,
            {
                "input": SourceBinding(
                    source, watermark_policy=SourceProvidedWatermarks()
                )
            },
            {"output": [SinkBinding.ordinary("profile", sink)]},
            ManagedCheckpointRuntime(state_root),
            config=StreamRuntimeConfig(
                checkpoint_interval=timedelta(hours=24),
                edge_budget=EdgeBudget(max_rows=maximum, max_bytes=64 << 20),
            ),
        ).start_async()
        scenario = cls(config, source, sink, job)
        try:
            await asyncio.wait_for(source.opened.wait(), timeout=30)
            started = time.perf_counter()
            while scenario.position < config.history_rows:
                count = min(
                    config.history_segment_rows, config.history_rows - scenario.position
                )
                data, watermark = _prepared_events(
                    scenario.position, count, config.entities
                )
                table, _elapsed = await _push_and_receive(source, sink, data, watermark)
                scenario.validate(table, count)
                scenario.position += count
            scenario.warm_seconds = time.perf_counter() - started
            # A loop turn outside timing lets prior completion callbacks retire.
            await asyncio.sleep(0)
            scenario.before_status = job.status()
            return scenario
        except BaseException:
            await job.cancel_async()
            raise

    def validate(self, table: pa.Table, rows: int) -> dict[str, Any]:
        return _validate_output(
            table,
            start=self.position,
            rows=rows,
            entities=self.config.entities,
            indicator=self.config.indicator,
            window=self.config.window,
            fast_window=self.config.fast_window,
        )

    async def sample(self, *, collect_gc: bool) -> dict[str, Any]:
        config = self.config
        data, watermark = _prepared_events(
            self.position, config.append_rows, config.entities
        )
        if collect_gc:
            gc.collect()
        started = time.perf_counter_ns()
        await self.source.push(data)
        await self.source.push(watermark)
        table = await self.sink.receive()
        finished = time.perf_counter_ns()
        correctness = self.validate(table, config.append_rows)
        sample = {
            "start_row": self.position,
            "seconds": (finished - started) / 1e9,
            "correctness": correctness,
            "phases_seconds": {
                "enqueue_to_source_data": (self.source.data_at - started) / 1e9,
                "source_data_to_source_watermark": (
                    self.source.watermark_at - self.source.data_at
                )
                / 1e9,
                "source_watermark_to_sink": (
                    self.sink.entered_at - self.source.watermark_at
                )
                / 1e9,
                "sink_to_receive": (finished - self.sink.entered_at) / 1e9,
                "to_pyarrow": self.sink.convert_ns / 1e9,
            },
        }
        self.position += config.append_rows
        return sample

    async def finish(self) -> dict[str, Any]:
        await self.source.push(None)
        outcome = await asyncio.wait_for(self.job.wait_async(), timeout=30)
        if outcome.state != "completed":
            raise RuntimeError(
                f"warm runner ended in {outcome.state}: {outcome.errors}"
            )
        return {
            "state": outcome.state,
            "warm_seconds": self.warm_seconds,
            "before_status": self.before_status,
            "after_status": self.job.status(),
        }

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


def _input_table(config: ScenarioConfig, start: int, rows: int) -> pa.Table:
    table = _segment(start, rows, config.entities)
    if config.append_entities is None or start + rows <= config.history_rows:
        return table
    sequence = np.arange(start, start + rows, dtype=np.uint64)
    appended = sequence >= config.history_rows
    # Sparse appends have distinct timestamps, so a partial entity tick cannot
    # close the timestamp of a later append before that append has arrived.
    positions = np.where(appended, sequence, sequence // config.entities)
    entity_index = np.where(
        appended,
        (sequence.astype(np.int64) - config.history_rows) % config.append_entities,
        sequence % config.entities,
    ).astype(np.int64)
    symbols = np.asarray([f"S{index:03d}" for index in range(config.entities)])
    table = table.set_column(
        0,
        SCHEMA.field(0),
        pa.array(
            BASE_MICROS + positions.astype(np.int64) * np.int64(1_000_000),
            type=SCHEMA.field(0).type,
        ),
    )
    return table.set_column(2, SCHEMA.field(2), pa.array(symbols[entity_index]))


def _prepared_events(
    config: ScenarioConfig, start: int, rows: int
) -> tuple[Data, Watermark]:
    end = start + rows
    cursor = Cursor(end.to_bytes(8, "big"), {"offset": end})
    table = _input_table(config, start, rows)
    data = Data(Batch.from_pyarrow(table), cursor)
    micros = table["event_time"][-1].value - int(BASE_MICROS)
    watermark = Watermark(BASE + timedelta(microseconds=micros))
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
    config: ScenarioConfig,
    *,
    start: int,
    rows: int,
) -> np.ndarray:
    if config.append_entities is not None and start >= config.history_rows:
        return _sparse_expected(config, start, rows)
    context_rows = min(start, (config.window - 1) * config.entities)
    context_start = start - context_rows
    sequence = np.arange(context_start, start + rows, dtype=np.uint64)
    prices = _prices(sequence, config.entities)
    if config.indicator == "dual_sma_spread":
        values = expected_dual_sma_spread(
            prices,
            entities=config.entities,
            fast_window=config.fast_window,
            slow_window=config.window,
        )
    else:
        values = expected_rolling_mean(
            prices, entities=config.entities, window=config.window
        )
    return values[context_rows:]


def _entity_context(config: ScenarioConfig, first: int, entity: int) -> np.ndarray:
    active = config.append_entities
    previous = (first - config.history_rows - entity) // active
    appended = min(config.window - 1, previous)
    warm = min(config.window - 1 - appended, config.history_rows // config.entities)
    last_warm = config.history_rows - config.entities + entity
    return np.concatenate(
        (
            np.arange(
                last_warm - (warm - 1) * config.entities,
                last_warm + 1,
                config.entities,
                dtype=np.uint64,
            ),
            np.arange(first - appended * active, first, active, dtype=np.uint64),
        )
    )


def _sparse_expected(config: ScenarioConfig, start: int, rows: int) -> np.ndarray:
    sequence = np.arange(start, start + rows, dtype=np.uint64)
    entities = (sequence - config.history_rows) % config.append_entities
    expected = np.empty(rows, dtype=np.float64)
    for entity in np.unique(entities):
        positions = np.flatnonzero(entities == entity)
        current = sequence[positions]
        context = _entity_context(config, int(current[0]), int(entity))
        prices = _prices(np.concatenate((context, current)), config.entities)
        slow = expected_rolling_mean(prices, entities=1, window=config.window)
        values = (
            expected_rolling_mean(prices, entities=1, window=config.fast_window) - slow
            if config.indicator == "dual_sma_spread"
            else slow
        )
        expected[positions] = values[len(context) :]
    return expected


def _validate_output(
    table: pa.Table,
    config: ScenarioConfig,
    *,
    start: int,
    rows: int,
) -> dict[str, Any]:
    if table.num_rows != rows:
        raise RuntimeError(f"expected {rows} output rows, observed {table.num_rows}")
    output_column = (
        "dual_sma_spread" if config.indicator == "dual_sma_spread" else "moving_average"
    )
    sequence = table["sequence"].combine_chunks().to_numpy(zero_copy_only=False)
    order = np.arange(rows)
    wanted_sequence = np.arange(start, start + rows, dtype=np.uint64)
    if not np.array_equal(sequence[order], wanted_sequence):
        raise RuntimeError("stream output sequence does not match appended rows")
    if not table.select(SCHEMA.names).equals(_input_table(config, start, rows)):
        raise RuntimeError("stream output identity columns changed")
    actual = table[output_column].combine_chunks().to_numpy(zero_copy_only=False)[order]
    expected = _expected(config, start=start, rows=rows)
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


def _validate_append_entities(active: int | None, entities: int) -> None:
    if active is not None and (type(active) is not int or not 1 <= active <= entities):
        raise ValueError("append_entities must be between one and entities")


def _validate_positive_sizes(sizes: tuple[int, ...]) -> None:
    if any(type(value) is not int or value <= 0 for value in sizes):
        raise ValueError("scenario sizes must be positive integers")


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    history_rows: int = 1_024_000
    append_rows: int = 64
    entities: int = 64
    indicator: str = "rolling_mean"
    window: int = 20
    fast_window: int = 5
    history_segment_rows: int = 64_000
    append_entities: int | None = None

    def __post_init__(self) -> None:
        sizes = (
            self.history_rows,
            self.append_rows,
            self.entities,
            self.window,
            self.fast_window,
            self.history_segment_rows,
        )
        _validate_positive_sizes(sizes)
        _validate_append_entities(self.append_entities, self.entities)
        ticks = (self.history_rows, self.history_segment_rows)
        if self.append_entities is None:
            ticks += (self.append_rows,)
        if any(value % self.entities for value in ticks):
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
                data, watermark = _prepared_events(config, scenario.position, count)
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
            self.config,
            start=self.position,
            rows=rows,
        )

    async def sample(self, *, collect_gc: bool) -> dict[str, Any]:
        config = self.config
        data, watermark = _prepared_events(config, self.position, config.append_rows)
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

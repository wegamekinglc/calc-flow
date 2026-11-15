"""Reproducible warm-state measurements through the StreamingRunner or a session."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import importlib.metadata
import json
import statistics
import struct
import sys
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import (
    BuiltinMethodType,
    CodeType,
    FunctionType,
    MethodDescriptorType,
    ModuleType,
)
from typing import Any

import numpy as np
import pyarrow as pa

from benchmarks.rolling_indicator_comparison import (
    TaLibMethod,
    _native_program,
    _timed_ta_lib,
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
from calc_flow.symbolic import Program, rows, ts

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
DECAY_INDICATORS = ("rolling_decay", "decay_mean")
# The logical input declared by ``_native_program``; sessions key inputs by it.
SESSION_INPUT = "quotes"


def expected_decay(prices: np.ndarray, *, entities: int, window: int) -> np.ndarray:
    """Independently fold finite fixture frames in the reference weight order."""
    expected = np.empty(len(prices), dtype=np.float64)
    for entity in range(entities):
        history = deque(maxlen=window)
        for position in range(entity, len(prices), entities):
            history.append(float(prices[position]))
            numerator = denominator = 0.0
            for weight, value in enumerate(history, window - len(history) + 1):
                numerator += float(weight) * value
                denominator += float(weight)
            expected[position] = numerator / denominator
    return expected


FLOAT64_DIGEST_FORMAT = "calc-flow-warm-float64-v1"


def _metadata_pairs(metadata: dict[bytes, bytes] | None) -> list[list[str]]:
    return [[key.hex(), value.hex()] for key, value in sorted((metadata or {}).items())]


def _canonical_schema(schema: pa.Schema) -> bytes:
    fields = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
            "metadata": _metadata_pairs(field.metadata),
        }
        for field in schema
    ]
    value = {"fields": fields, "metadata": _metadata_pairs(schema.metadata)}
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest_field(digest: Any, value: bytes) -> None:
    digest.update(struct.pack("<Q", len(value)))
    digest.update(value)


class _Float64Digest:
    """Hash logical validity and valid raw bits independently of Arrow chunks."""

    def __init__(self) -> None:
        self._schema: bytes | None = None
        self._columns: tuple[int, ...] = ()
        self._column_count = 0
        self._rows = 0
        self._validity: list[Any] = []
        self._values: list[Any] = []
        self._valid_counts: list[int] = []

    def _initialize(self, schema: pa.Schema) -> None:
        self._columns = tuple(
            index
            for index, field in enumerate(schema)
            if pa.types.is_float64(field.type)
        )
        if not self._columns:
            raise ValueError("output digest requires a Float64 column")
        self._schema = _canonical_schema(schema)
        self._column_count = len(schema)
        self._validity = [hashlib.sha256() for _ in self._columns]
        self._values = [hashlib.sha256() for _ in self._columns]
        self._valid_counts = [0 for _ in self._columns]

    def update(self, table: pa.Table) -> None:
        if self._schema is None:
            self._initialize(table.schema)
        if _canonical_schema(table.schema) != self._schema:
            raise ValueError("output digest schema changed")
        for position, column in enumerate(self._columns):
            for chunk in table.column(column).chunks:
                self._update_chunk(position, chunk)
        self._rows += table.num_rows

    def _update_chunk(self, position: int, chunk: pa.Array) -> None:
        valid = chunk.is_valid().to_numpy(zero_copy_only=False)
        self._validity[position].update(valid.astype(np.uint8).tobytes())
        count = int(np.count_nonzero(valid))
        if count:
            buffer = chunk.buffers()[1]
            if buffer is None:
                raise ValueError("valid Float64 output has no value buffer")
            bits = np.frombuffer(
                buffer, dtype=np.uint64, count=len(chunk), offset=chunk.offset * 8
            )
            self._values[position].update(
                np.asarray(bits[valid], dtype="<u8").tobytes()
            )
        self._valid_counts[position] += count

    def hexdigest(self) -> str:
        if self._schema is None:
            raise ValueError("output digest has no schema")
        digest = hashlib.sha256(FLOAT64_DIGEST_FORMAT.encode() + b"\0")
        for value in (
            self._schema,
            struct.pack("<Q", self._column_count),
            struct.pack("<Q", len(self._columns)),
            struct.pack("<Q", self._rows),
        ):
            _digest_field(digest, value)
        for position, column in enumerate(self._columns):
            for value in (
                struct.pack("<Q", column),
                struct.pack("<Q", self._rows),
                struct.pack("<Q", self._valid_counts[position]),
                self._validity[position].digest(),
                self._values[position].digest(),
            ):
                _digest_field(digest, value)
        return digest.hexdigest()


def _float64_output_digest(table: pa.Table) -> str:
    digest = _Float64Digest()
    digest.update(table)
    return digest.hexdigest()


def _prices(sequence: np.ndarray, entities: int) -> np.ndarray:
    entity_index = sequence % np.uint64(entities)
    return (
        100.0
        + ((sequence * np.uint64(17)) % np.uint64(1_000)).astype(np.float64) / 100.0
        + entity_index.astype(np.float64) / 100.0
    )


def _symbols(entities: int) -> np.ndarray:
    width = max(3, len(str(entities - 1)))
    return np.asarray([f"S{index:0{width}d}" for index in range(entities)])


def _segment(start: int, rows: int, entities: int) -> pa.Table:
    sequence = np.arange(start, start + rows, dtype=np.uint64)
    entity_index = sequence % np.uint64(entities)
    positions = sequence // np.uint64(entities)
    symbols = _symbols(entities)
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
    symbols = _symbols(config.entities)
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
    if config.indicator in DECAY_INDICATORS:
        values = expected_decay(prices, entities=config.entities, window=config.window)
    elif config.indicator == "dual_sma_spread":
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
        "decay"
        if config.indicator in DECAY_INDICATORS
        else "dual_sma_spread"
        if config.indicator == "dual_sma_spread"
        else "moving_average"
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
    tolerance = (1e-12, 1e-12) if config.indicator in DECAY_INDICATORS else (RTOL, ATOL)
    close = np.isclose(
        actual, expected, rtol=tolerance[0], atol=tolerance[1], equal_nan=True
    )
    absolute = np.abs(actual - expected)
    result = {
        "passed": bool(np.all(close)),
        "mismatched_values": int(np.count_nonzero(~close)),
        "max_absolute_error": float(absolute.max(initial=0.0)),
        "float64_digest": _float64_output_digest(table),
    }
    if not result["passed"]:
        raise RuntimeError(f"strict incremental correctness failed: {result}")
    if config.indicator == "decay_mean":
        _validate_output(
            table, replace(config, indicator="rolling_mean"), start=start, rows=rows
        )
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
    if active is not None and (
        active.__class__ is not int or not 1 <= active <= entities
    ):
        raise ValueError("append_entities must be between one and entities")


def _validate_positive_sizes(sizes: tuple[int, ...]) -> None:
    if any(value.__class__ is not int or value <= 0 for value in sizes):
        raise ValueError("scenario sizes must be positive integers")


def _expected_mean(size: int) -> dict[str, Any]:
    return {
        "kind": "mean",
        "primitive_version": 1,
        "input": "price",
        "frame": {"kind": "rows", "size": size},
        "min_periods": 1,
    }


def _expected_rolling_output(config: ScenarioConfig) -> dict[str, Any]:
    slow = _expected_mean(config.window)
    if config.indicator == "rolling_mean":
        return slow
    return {
        "kind": "difference",
        "primitive_version": 1,
        "left": _expected_mean(config.fast_window),
        "right": slow,
    }


def _expected_rolling_outputs(config: ScenarioConfig) -> list[dict[str, Any]]:
    if config.indicator not in DECAY_INDICATORS:
        return [_expected_rolling_output(config)]
    decay = {**_expected_mean(config.window), "kind": "decay"}
    if config.indicator == "decay_mean":
        return [decay, _expected_mean(config.window)]
    return [decay]


def _rolling_preflight(spec: dict[str, Any], config: ScenarioConfig) -> None:
    expected = {
        "configuration_version": 1,
        "state_layout_version": 1,
        "numerical_profile": "stable_v1",
        "partition_by": ["symbol"],
        "event_time": "event_time",
        "sequence_by": ["sequence"],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1",
    }
    actual = {key: spec.get(key) for key in expected}
    actual["numerical_profile"] = spec.get("numerical_profile", "stable_v1")
    if actual != expected:
        raise ValueError("warm plan preflight rolling semantics changed")
    outputs = [
        {key: value for key, value in output.items() if key != "output"}
        for output in spec["outputs"]
    ]
    if outputs != _expected_rolling_outputs(config):
        raise ValueError("warm plan preflight output frames or min_periods changed")


def _plan_preflight(project: dict[str, Any], config: ScenarioConfig) -> dict[str, Any]:
    if project["runtime"]["mode"] != "stream":
        raise ValueError("warm plan preflight requires streaming mode")
    graph = project["graph"]
    expected = {
        "batch_size": 8192,
        "target_partitions": 1,
        "parallelism_mode": "fixed",
        "enable_rolling_rewrite": True,
    }
    datafusion = {key: graph["datafusion"].get(key) for key in expected}
    if datafusion != expected:
        raise ValueError("warm plan preflight DataFusion policy changed")
    rolling = [
        node["operator"]["spec"]
        for node in graph["nodes"]
        if node["operator"]["kind"] == "rolling"
    ]
    if len(rolling) != 1:
        raise ValueError("warm plan preflight requires one rolling stage")
    _rolling_preflight(rolling[0], config)
    windows = (
        [config.fast_window, config.window]
        if config.indicator == "dual_sma_spread"
        else [config.window] * len(_expected_rolling_outputs(config))
    )
    return {
        "engine": "streaming",
        "numerical_profile": "stable_v1",
        "windows": windows,
        "min_periods": 1,
        "datafusion": datafusion,
        "exact_projection": config.exact_projection,
        "checkpointing": not config.exact_projection,
    }


def _validate_no_late_rows(status: dict[str, Any]) -> None:
    operators = status.get("operators")
    if not isinstance(operators, dict) or not operators:
        raise RuntimeError("warm stream requires measured zero late rows")
    if any(
        not isinstance(node, dict) or node.get("late_rows") != 0
        for node in operators.values()
    ):
        raise RuntimeError("warm stream requires measured zero late rows")


def _prepared_plan(config: ScenarioConfig) -> tuple[Any, dict[str, Any]]:
    runtime = Runtime()
    program = _warm_program(config)
    project = program.to_project(runtime).model_dump(mode="json")
    preflight = _plan_preflight(project, config)
    encoded = json.dumps(project, sort_keys=True, separators=(",", ":")).encode()
    plan = program.compile_stream(runtime)
    identity = {
        "program_fingerprint": program.fingerprint,
        "plan_fingerprint": plan.fingerprint,
        "project_sha256": hashlib.sha256(encoded).hexdigest(),
        "preflight": preflight,
    }
    return plan, identity


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
    exact_projection: bool = False

    def __post_init__(self) -> None:
        if type(self.exact_projection) is not bool:
            raise TypeError("exact_projection must be a bool")
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
        ticks = (self.history_rows,)
        if self.append_entities is None:
            ticks += (self.append_rows,)
        if any(value % self.entities for value in ticks):
            raise ValueError(
                "history and increments must contain complete entity ticks"
            )
        segment = self.history_segment_rows
        if segment % self.entities and self.entities % segment:
            raise ValueError(
                "history segments must contain or divide complete entity ticks"
            )
        if self.fast_window > self.window:
            raise ValueError("fast_window must not exceed window")
        if self.indicator not in ("rolling_mean", "dual_sma_spread", *DECAY_INDICATORS):
            raise ValueError("unsupported warm indicator")
        if self.indicator in DECAY_INDICATORS and self.append_entities is not None:
            raise ValueError("decay profiling requires full entity ticks")


def _warm_program(config: ScenarioConfig) -> Program:
    indicator = (
        "rolling_mean" if config.indicator in DECAY_INDICATORS else config.indicator
    )
    program = _native_program(
        config.window,
        indicator=indicator,
        fast_window=config.fast_window,
        engine="streaming",
    )
    if config.indicator in DECAY_INDICATORS:
        quotes = program.inputs[0]
        features = {"decay": ts.decay(quotes["price"], window=rows(config.window))}
        if config.indicator == "decay_mean":
            features["moving_average"] = ts.mean(
                quotes["price"], window=rows(config.window)
            )
        program = Program(
            f"warm-{config.indicator}",
            engine="streaming",
            inputs=program.inputs,
            outputs=(("indicators", quotes.with_columns(features)),),
        )
    if not config.exact_projection:
        return program
    value_columns = {
        "rolling_decay": ["decay"],
        "decay_mean": ["decay", "moving_average"],
        "dual_sma_spread": ["dual_sma_spread"],
        "rolling_mean": ["moving_average"],
    }[config.indicator]
    return Program(
        program.name,
        engine="streaming",
        inputs=program.inputs,
        outputs=tuple(
            (name, output.select(*SCHEMA.names, *value_columns))
            for name, output in program.outputs
        ),
    )


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
        self._warm_digest = _Float64Digest()
        self.plan_identity: dict[str, Any] = {}

    @property
    def warm_output_digest(self) -> str:
        return self._warm_digest.hexdigest()

    @classmethod
    async def start(cls, config: ScenarioConfig, state_root: Path) -> WarmScenario:
        plan, identity = _prepared_plan(config)
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
            None if config.exact_projection else ManagedCheckpointRuntime(state_root),
            config=StreamRuntimeConfig(
                checkpoint_interval=timedelta(hours=24),
                edge_budget=EdgeBudget(max_rows=maximum, max_bytes=64 << 20),
                checkpointing=not config.exact_projection,
            ),
        ).start_async()
        scenario = cls(config, source, sink, job)
        scenario.plan_identity = identity
        try:
            await asyncio.wait_for(source.opened.wait(), timeout=30)
            started = time.perf_counter()
            await scenario._warm_history()
            scenario.warm_seconds = time.perf_counter() - started
            # A loop turn outside timing lets prior completion callbacks retire.
            await asyncio.sleep(0)
            scenario.before_status = job.status()
            _validate_no_late_rows(scenario.before_status)
            return scenario
        except BaseException:
            await job.cancel_async()
            raise

    async def _warm_history(self) -> None:
        group_rows = max(self.config.history_segment_rows, self.config.entities)
        while self.position < self.config.history_rows:
            await self._warm_group(
                min(group_rows, self.config.history_rows - self.position)
            )

    async def _warm_group(self, rows: int) -> None:
        start = self.position
        end = start + rows
        watermark = None
        for offset in range(0, rows, self.config.history_segment_rows):
            count = min(self.config.history_segment_rows, rows - offset)
            data, watermark = _prepared_events(self.config, start + offset, count)
            await self.source.push(data)
        await self.source.push(watermark)
        while self.position < end:
            table = await self.sink.receive()
            if not 0 < table.num_rows <= end - self.position:
                raise RuntimeError(
                    "warm output must contain only the pending tick rows"
                )
            self.validate(table, table.num_rows)
            self._warm_digest.update(table)
            self.position += table.num_rows

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
        status = self.job.status()
        _validate_no_late_rows(status)
        return {
            "state": outcome.state,
            "warm_seconds": self.warm_seconds,
            "before_status": self.before_status,
            "after_status": status,
        }

    async def cancel(self) -> None:
        await self.job.cancel_async()


def _is_calc_flow_module(name: str, module: object) -> bool:
    in_package = name == "calc_flow" or name.startswith("calc_flow.")
    return in_package and isinstance(module, ModuleType)


def _module_members(module: ModuleType) -> list[object]:
    """Return a module's values and the attributes of classes it defines."""
    values = list(vars(module).values())
    for value in tuple(values):
        if isinstance(value, type) and value.__module__ == module.__name__:
            values.extend(vars(value).values())
    return values


def _member_functions(value: object) -> tuple[object, ...]:
    if isinstance(value, property):
        return (value.fget, value.fset, value.fdel)
    return (getattr(value, "__func__", value),)


def _defined_function(function: object, module: ModuleType) -> bool:
    return isinstance(function, FunctionType) and function.__module__ == module.__name__


def _module_codes(module: ModuleType) -> list[CodeType]:
    return [
        function.__code__
        for value in _module_members(module)
        for function in _member_functions(value)
        if _defined_function(function, module)
    ]


def _with_nested_codes(roots: list[CodeType]) -> tuple[CodeType, ...]:
    codes: set[CodeType] = set()
    pending = list(roots)
    while pending:
        code = pending.pop()
        if code not in codes:
            codes.add(code)
            pending.extend(
                value for value in code.co_consts if isinstance(value, CodeType)
            )
    return tuple(codes)


def _calc_flow_code_objects() -> tuple[CodeType, ...]:
    """Return the code of every loaded pure-Python ``calc_flow`` function."""
    return _with_nested_codes(
        [
            code
            for name, module in tuple(sys.modules.items())
            if _is_calc_flow_module(name, module)
            for code in _module_codes(module)
        ]
    )


def _native_method_matcher(
    native_type: type, name: str
) -> Callable[[object, object], bool]:
    """Match a call to one native method, bound or through its descriptor."""

    def matches(callable_: object, arg0: object) -> bool:
        if isinstance(callable_, MethodDescriptorType):
            owner = arg0
        elif isinstance(callable_, BuiltinMethodType):
            owner = callable_.__self__
        else:
            return False
        return callable_.__name__ == name and isinstance(owner, native_type)

    return matches


class _NativeCallProbe:
    """Split one sample at a native call observed through ``sys.monitoring``.

    The library is not instrumented: CALL and C_RETURN events on selected
    Python code objects timestamp the matching native call. Every other call
    site is disabled the first time it is seen, so warm-up leaves only the
    matching site armed and each timed sample pays two small callbacks.
    """

    def __init__(
        self,
        codes: tuple[CodeType, ...],
        matches: Callable[[object, object], bool],
    ) -> None:
        self._codes = codes
        self._matches = matches
        self._tool: int | None = None
        self._called: object = None
        self.calls = 0
        self.called_ns = 0
        self.returned_ns = 0

    def install(self) -> str | None:
        """Arm the probe, or return why phase attribution is unavailable."""
        monitoring = getattr(sys, "monitoring", None)
        if monitoring is None:
            return "sys.monitoring requires Python 3.12 or newer"
        tool = monitoring.PROFILER_ID
        try:
            monitoring.use_tool_id(tool, "calc-flow-warm-session")
        except ValueError:
            return "the sys.monitoring profiler tool ID is already in use"
        events = monitoring.events
        self._tool = tool
        try:
            monitoring.register_callback(tool, events.CALL, self._on_call)
            # C_RETURN follows the CALL event of the same site.
            monitoring.register_callback(tool, events.C_RETURN, self._on_return)
            for code in self._codes:
                monitoring.set_local_events(tool, code, events.CALL)
        except BaseException:
            self.close()
            raise
        return None

    def close(self) -> None:
        if self._tool is None:
            return
        monitoring = sys.monitoring
        for code in self._codes:
            monitoring.set_local_events(self._tool, code, 0)
        monitoring.register_callback(self._tool, monitoring.events.CALL, None)
        monitoring.register_callback(self._tool, monitoring.events.C_RETURN, None)
        monitoring.free_tool_id(self._tool)
        self._tool = None

    def _on_call(self, code: CodeType, offset: int, callable_: object, arg0: object):
        if self._matches(callable_, arg0):
            self.calls += 1
            self._called = callable_
            self.called_ns = time.perf_counter_ns()
            return None
        return sys.monitoring.DISABLE

    def _on_return(
        self, code: CodeType, offset: int, callable_: object, arg0: object
    ) -> None:
        returned = time.perf_counter_ns()
        if callable_ is self._called:
            self.returned_ns = returned
            self._called = None

    def reset(self) -> None:
        self.calls = 0
        self.called_ns = self.returned_ns = 0
        self._called = None

    def phases(self, started: int, finished: int) -> dict[str, float] | str:
        """Return contiguous phases of one sample, or why they are unavailable."""
        if self.calls != 1:
            return f"observed {self.calls} matching native calls"
        if not started <= self.called_ns <= self.returned_ns <= finished:
            return "native call timestamps fall outside the sample"
        return {
            "argument_handling": (self.called_ns - started) / 1e9,
            "native_call": (self.returned_ns - self.called_ns) / 1e9,
            "result_assembly": (finished - self.returned_ns) / 1e9,
        }


def _probe_calibration_target(mapping: dict[int, int]) -> int | None:
    return mapping.get(0)


def _probe_overhead_seconds(*, calls: int = 2_000, rounds: int = 5) -> float | None:
    """Estimate the wall time one matched CALL/C_RETURN callback pair adds.

    The calibration call is a C method descriptor matched the same way as the
    native session method, so it pays the same matcher and both callbacks.
    """
    mapping: dict[int, int] = {}

    def measure() -> int:
        started = time.perf_counter_ns()
        for _ in range(calls):
            _probe_calibration_target(mapping)
        return time.perf_counter_ns() - started

    plain = min(measure() for _ in range(rounds))
    probe = _NativeCallProbe(
        (_probe_calibration_target.__code__,), _native_method_matcher(dict, "get")
    )
    if probe.install() is not None:
        return None
    try:
        probed = min(measure() for _ in range(rounds))
    finally:
        probe.close()
    return max(0, probed - plain) / calls / 1e9


class WarmSessionScenario:
    """Own one caller-thread StreamSession on a dedicated thread.

    The session is built, warmed, sampled and finished on one thread without a
    running event loop, as a synchronous caller would use it. IPC, executor
    handoff, input preparation and validation are outside the timed region.
    """

    def __init__(
        self,
        config: ScenarioConfig,
        executor: ThreadPoolExecutor,
        session: Any,
        program: Program,
    ) -> None:
        self.config = config
        self.session = session
        self.position = 0
        self.warm_seconds = 0.0
        self.plan_identity: dict[str, Any] = {}
        self.phase_probe: dict[str, Any] = {}
        self._executor = executor
        self._output_name = program.outputs[0][0]
        self._probe: _NativeCallProbe | None = None
        self._probe_note = "phase probe was not installed"
        self._warm_digest = _Float64Digest()
        self._closed = False

    @property
    def warm_output_digest(self) -> str:
        return self._warm_digest.hexdigest()

    @classmethod
    async def start(
        cls, config: ScenarioConfig, state_root: Path | None = None
    ) -> WarmSessionScenario:
        """Start a session; it is in-memory, so ``state_root`` is unused."""
        if not config.exact_projection:
            raise ValueError(
                "the session driver measures the checkpoint-off exact projection"
            )
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="warm-session")
        try:
            return await asyncio.get_running_loop().run_in_executor(
                executor, cls._open, config, executor
            )
        except BaseException:
            executor.shutdown(wait=False)
            raise

    @classmethod
    def _open(
        cls, config: ScenarioConfig, executor: ThreadPoolExecutor
    ) -> WarmSessionScenario:
        _plan, identity = _prepared_plan(config)
        program = _warm_program(config)
        maximum = max(config.history_segment_rows, config.append_rows)
        session = program.session(
            watermarks={SESSION_INPUT: SourceProvidedWatermarks()},
            config=StreamRuntimeConfig(
                checkpointing=False,
                edge_budget=EdgeBudget(max_rows=maximum, max_bytes=64 << 20),
            ),
        )
        scenario = cls(config, executor, session, program)
        scenario.plan_identity = identity
        try:
            scenario._install_probe()
            started = time.perf_counter()
            scenario._warm_history()
            scenario.warm_seconds = time.perf_counter() - started
            return scenario
        except BaseException:
            scenario._close()
            raise

    def _install_probe(self) -> None:
        native_type = getattr(
            sys.modules.get("calc_flow._native"), "_StreamSession", None
        )
        overhead = _probe_overhead_seconds()
        if native_type is None:
            self._probe_note = "calc_flow._native exposes no _StreamSession"
        else:
            probe = _NativeCallProbe(
                _calc_flow_code_objects(), _native_method_matcher(native_type, "push")
            )
            failure = probe.install()
            if failure is None:
                self._probe = probe
            else:
                self._probe_note = failure
        self.phase_probe = {
            "installed": self._probe is not None,
            "note": None if self._probe is not None else self._probe_note,
            "overhead_seconds_estimate": overhead,
        }

    def _warm_history(self) -> None:
        group_rows = max(self.config.history_segment_rows, self.config.entities)
        while self.position < self.config.history_rows:
            self._warm_group(min(group_rows, self.config.history_rows - self.position))

    def _warm_group(self, rows: int) -> None:
        # Like the runner arm, only a group of complete ticks carries a watermark.
        start = self.position
        end = start + rows
        segment = self.config.history_segment_rows
        for offset in range(0, rows, segment):
            count = min(segment, rows - offset)
            data, watermark = _prepared_events(self.config, start + offset, count)
            final = offset + count >= rows
            outputs = self.session.push(
                {SESSION_INPUT: data.batch}, watermark=watermark if final else None
            )
            table = outputs[self._output_name]
            if table.num_rows > end - self.position:
                raise RuntimeError(
                    "warm output must contain only the pending tick rows"
                )
            if table.num_rows:
                self.validate(table, table.num_rows)
                self._warm_digest.update(table)
                self.position += table.num_rows
        if self.position != end:
            raise RuntimeError("warm watermark did not finalize every pending row")

    def validate(self, table: pa.Table, rows: int) -> dict[str, Any]:
        return _validate_output(table, self.config, start=self.position, rows=rows)

    def _sample(self, collect_gc: bool) -> dict[str, Any]:
        config = self.config
        data, watermark = _prepared_events(config, self.position, config.append_rows)
        inputs = {SESSION_INPUT: data.batch}
        session, output_name, probe = self.session, self._output_name, self._probe
        if probe is not None:
            probe.reset()
        if collect_gc:
            gc.collect()
        started = time.perf_counter_ns()
        table = session.push(inputs, watermark=watermark)[output_name]
        finished = time.perf_counter_ns()
        phases = self._probe_note if probe is None else probe.phases(started, finished)
        correctness = self.validate(table, config.append_rows)
        sample = {
            "start_row": self.position,
            "seconds": (finished - started) / 1e9,
            "correctness": correctness,
            "phases_seconds": phases if isinstance(phases, dict) else {},
            "phase_attribution": (
                "native_call_probe"
                if isinstance(phases, dict)
                else f"unavailable: {phases}"
            ),
        }
        self.position += config.append_rows
        return sample

    async def sample(self, *, collect_gc: bool) -> dict[str, Any]:
        return await self._run(self._sample, collect_gc)

    def _finish(self) -> dict[str, Any]:
        try:
            flushed = {
                name: table.num_rows for name, table in self.session.finish().items()
            }
        finally:
            self._close()
        if any(flushed.values()):
            raise RuntimeError(f"session finish emitted unexpected rows: {flushed}")
        return {
            "state": "completed",
            "warm_seconds": self.warm_seconds,
            "finish_rows": flushed,
            "late_rows_guard": "late_policy=error fails the session on any late row",
        }

    async def finish(self) -> dict[str, Any]:
        try:
            return await self._run(self._finish)
        finally:
            self._executor.shutdown(wait=False)

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.session.close()
        finally:
            if self._probe is not None:
                self._probe.close()
                self._probe = None

    async def cancel(self) -> None:
        try:
            if not self._closed:
                await self._run(self._close)
        finally:
            self._executor.shutdown(wait=False)

    async def _run(self, function: Callable[..., Any], *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, function, *args
        )


SCENARIO_DRIVERS: dict[str, Any] = {
    "runner": WarmScenario,
    "session": WarmSessionScenario,
}


def _require_ta_lib_layout(config: ScenarioConfig, samples: int) -> None:
    if config.indicator not in ("rolling_mean", "dual_sma_spread"):
        raise ValueError("the TA-Lib reference covers SMA indicators only")
    if config.append_entities is not None:
        raise ValueError("the TA-Lib reference requires complete entity ticks")
    if type(samples) is not int or samples < 2:
        raise ValueError("the TA-Lib reference requires at least two samples")


def _ta_lib_append(config: ScenarioConfig, start: int) -> TaLibMethod:
    """Return TA-Lib over one append and its ``window - 1`` rows per entity."""
    context = min(start, (config.window - 1) * config.entities)
    sequence = np.arange(start - context, start + config.append_rows, dtype=np.uint64)
    fast_window = config.fast_window if config.indicator == "dual_sma_spread" else None
    return TaLibMethod(
        _prices(sequence, config.entities),
        entities=config.entities,
        window=config.window,
        fast_window=fast_window,
        history_rows=context,
    )


def _timed_ta_lib_append(config: ScenarioConfig, index: int, collect_gc: bool) -> float:
    start = config.history_rows + index * config.append_rows
    method = _ta_lib_append(config, start)
    actual = method.run()
    expected = _expected(config, start=start, rows=config.append_rows)
    if not np.allclose(actual, expected, rtol=RTOL, atol=ATOL):
        raise RuntimeError("TA-Lib reference does not match the warm oracle")
    if collect_gc:
        gc.collect()
    return _timed_ta_lib(method, 1)


def ta_lib_reference(
    config: ScenarioConfig, *, samples: int, collect_gc: bool
) -> dict[str, Any]:
    """Time TA-Lib over each paired append's bounded window context.

    Sample ``i`` recalculates the ``window - 1`` rows per entity before the
    ``i``-th paired append plus that append, returns only the appended rows and
    matches them against the warm oracle outside the timed call.
    """
    _require_ta_lib_layout(config, samples)
    seconds = [
        _timed_ta_lib_append(config, index, collect_gc) for index in range(samples)
    ]
    return {
        "adapter": "TaLibMethod bounded-history append",
        "ta_lib_version": importlib.metadata.version("ta-lib"),
        "history_rows_per_entity": config.window - 1,
        "collect_gc": collect_gc,
        **_summary(seconds, config.append_rows),
    }

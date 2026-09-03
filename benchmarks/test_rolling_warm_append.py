"""Compare warm native rolling append with SQL full-history recomputation."""

from __future__ import annotations

import asyncio
import os
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from benchmarks.support import BenchmarkFixture, benchmark_group, selected_scale
from benchmarks.symbolic_support import (
    SYMBOLIC_SEED,
    quote_workload,
    record_symbolic_benchmark,
)
from calc_flow import (
    Batch,
    Cursor,
    Data,
    EdgeBudget,
    Idle,
    ManagedCheckpointRuntime,
    NativeWatermarkCapability,
    PipelineBuilder,
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
from calc_flow.symbolic import FeatureSet, Field, Program, rows, table_input, ts

STANDARD_HISTORY_ROWS = 99_968
ENTITY_COUNT = 64
WINDOW_ROWS = (20, 60, 252, 1024)
DELTA_ROWS = (64, 640, 6400)
WARM_UP_RUNS = 1
STANDARD_SAMPLE_COUNT = 5
OVERHEAD_SAMPLE_COUNT = 1
NULL_PERIODS = 29
RELATIVE_TOLERANCE = 1e-10
ABSOLUTE_TOLERANCE = 1e-12
SCENARIO = "warm_native_append_vs_sql_full_recompute"
WORKLOAD_CONTRACT = "dal-184-warm-append-v1"
STATISTICS = ("mean", "stddev", "sum", "min", "max")
EXACT_COLUMNS = ("event_time", "sequence", "symbol", "industry", "close", "volume")
STREAM_TIMEOUT_SECONDS = float(
    os.environ.get("CALC_FLOW_BENCHMARK_STREAM_TIMEOUT_SECONDS", "120")
)
SEED_STRATEGY = "new runner plus untimed seed, watermark, and empty probe per sample"
ROLLING_NODE_ID = "signals__cf_rolling"


@dataclass(frozen=True, slots=True)
class _BenchmarkProfile:
    name: str
    history_rows: int
    windows: tuple[int, ...]
    delta_rows: tuple[int, ...]
    measured_samples: int


@dataclass(frozen=True, slots=True)
class _CaseInputs:
    history: Batch
    empty_probe: Batch
    delta: Batch
    sql_full: Batch
    history_table: pa.Table
    delta_table: pa.Table
    sql_full_table: pa.Table
    entities: int
    null_rows: int


@dataclass(frozen=True, slots=True)
class _NativeObservation:
    output: pa.Table
    seconds: float | None
    history_rows: int
    delta_rows: int
    seed_emissions: int
    seed_probe_emissions: int
    delta_emissions: int
    seed_probe_operator_inputs: tuple[tuple[str, int], ...]
    seed_completed_ns: int
    append_started_ns: int | None


@dataclass(frozen=True, slots=True)
class _PairedObservation:
    order: str
    native: _NativeObservation
    sql_output: pa.Table
    sql_seconds: float | None
    sql_input_rows: int


def _benchmark_profile() -> _BenchmarkProfile:
    scale = selected_scale()
    history_rows = (
        STANDARD_HISTORY_ROWS
        if scale.name == "standard"
        else max(ENTITY_COUNT, scale.table_rows // ENTITY_COUNT * ENTITY_COUNT)
    )
    measured_samples = (
        OVERHEAD_SAMPLE_COUNT if scale.name == "overhead" else STANDARD_SAMPLE_COUNT
    )
    return _BenchmarkProfile(
        name=scale.name,
        history_rows=history_rows,
        windows=WINDOW_ROWS,
        delta_rows=DELTA_ROWS,
        measured_samples=measured_samples,
    )


PROFILE = _benchmark_profile()


def _sql_query(window: int) -> str:
    frame = (
        "PARTITION BY symbol ORDER BY event_time, sequence "
        f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
    )
    return f"""
SELECT
  event_time,
  sequence,
  symbol,
  industry,
  close,
  volume,
  avg(close) OVER ({frame}) AS mean_close_{window},
  stddev_samp(close) OVER ({frame}) AS stddev_close_{window},
  sum(close) OVER ({frame}) AS sum_close_{window},
  min(close) OVER ({frame}) AS min_close_{window},
  max(close) OVER ({frame}) AS max_close_{window}
FROM input
ORDER BY event_time, symbol, sequence
"""


def _incremental_program(window: int) -> Program:
    quotes = table_input(
        "quotes",
        schema=(
            Field("event_time", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("industry", "string", nullable=False),
            Field("close", "float64"),
            Field("volume", "float64"),
        ),
        entity_by=("symbol",),
        event_time="event_time",
        sequence_by=("sequence",),
    )
    frame = rows(window)
    signals = quotes.with_columns(
        FeatureSet(
            (
                (f"mean_close_{window}", ts.mean(quotes["close"], window=frame)),
                (
                    f"stddev_close_{window}",
                    ts.stddev(quotes["close"], window=frame, ddof=1),
                ),
                (f"sum_close_{window}", ts.sum(quotes["close"], window=frame)),
                (f"min_close_{window}", ts.min(quotes["close"], window=frame)),
                (f"max_close_{window}", ts.max(quotes["close"], window=frame)),
            )
        )
    )
    return Program(
        f"dal-184-warm-append-{window}",
        inputs=(quotes,),
        outputs=(("signals", signals),),
    )


@lru_cache(maxsize=4)
def _dataset(history_rows: int) -> tuple[pa.Table, int, int]:
    workload = quote_workload(rows=history_rows + max(DELTA_ROWS))
    if workload.entities != ENTITY_COUNT:
        raise AssertionError(
            f"warm append workload requires {ENTITY_COUNT} entities, "
            f"got {workload.entities}"
        )
    table = workload.batch.to_pyarrow()
    event_time_index = table.schema.get_field_index("event_time")
    close_index = table.schema.get_field_index("close")
    event_time = table["event_time"].cast(pa.timestamp("us", tz="UTC"))
    close = table["close"].to_numpy(zero_copy_only=False)
    positions = np.arange(table.num_rows) // workload.entities
    null_mask = positions % NULL_PERIODS == 0
    prepared = table.set_column(
        event_time_index,
        pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
        event_time,
    ).set_column(
        close_index,
        pa.field("close", pa.float64()),
        pa.array(close, mask=null_mask),
    )
    return prepared, workload.entities, int(null_mask.sum())


@lru_cache(maxsize=24)
def _case_inputs(history_rows: int, delta_rows: int) -> _CaseInputs:
    full_dataset, entities, _null_rows = _dataset(history_rows)
    history_table = full_dataset.slice(0, history_rows)
    delta_table = full_dataset.slice(history_rows, delta_rows)
    sql_full_table = full_dataset.slice(0, history_rows + delta_rows)
    return _CaseInputs(
        history=Batch.from_pyarrow(history_table),
        empty_probe=Batch.from_pyarrow(delta_table.slice(0, 0)),
        delta=Batch.from_pyarrow(delta_table),
        sql_full=Batch.from_pyarrow(sql_full_table),
        history_table=history_table,
        delta_table=delta_table,
        sql_full_table=sql_full_table,
        entities=entities,
        null_rows=sql_full_table["close"].null_count,
    )


def _feature_columns(window: int) -> tuple[str, ...]:
    return tuple(f"{statistic}_close_{window}" for statistic in STATISTICS)


def _assert_equivalent_delta(native: pa.Table, sql: pa.Table, window: int) -> None:
    if native.column_names != sql.column_names:
        raise AssertionError(
            f"warm append columns differ: {native.column_names} != {sql.column_names}"
        )
    if native.num_rows != sql.num_rows:
        raise AssertionError(
            f"warm append row counts differ: {native.num_rows} != {sql.num_rows}"
        )
    if not native.select(EXACT_COLUMNS).equals(sql.select(EXACT_COLUMNS)):
        raise AssertionError("warm append identity or input columns differ")
    for name in _feature_columns(window):
        native_values = native[name]
        sql_values = sql[name]
        if native_values.is_null().to_pylist() != sql_values.is_null().to_pylist():
            raise AssertionError(f"warm append null masks differ for {name}")
        np.testing.assert_allclose(
            native_values.to_numpy(zero_copy_only=False),
            sql_values.to_numpy(zero_copy_only=False),
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
            equal_nan=True,
            err_msg=f"warm append values differ for {name}",
        )


def _last_event_time(table: pa.Table) -> Any:
    return table["event_time"][table.num_rows - 1].as_py()


class _GatedAppendSource:
    """Emit seed + watermark, then block until one delta is released."""

    def __init__(self, inputs: _CaseInputs) -> None:
        self._inputs = inputs
        self._phase = 0
        self._delta_release = asyncio.Event()
        self.waiting_for_delta = asyncio.Event()
        self._hold_open = asyncio.Event()
        self.seed_emissions = 0
        self.seed_probe_emissions = 0
        self.delta_emissions = 0

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.UNSUPPORTED,
            SourceDeliveryCapability.LOSSY,
            max_batch_rows=max(
                self._inputs.history.num_rows,
                self._inputs.delta.num_rows,
            ),
            max_batch_bytes=max(
                self._inputs.history_table.nbytes,
                self._inputs.delta_table.nbytes,
            ),
            native_watermarks=NativeWatermarkCapability.EMITS_NATIVE,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self._phase = 0 if cursor is None else int(cursor.payload["phase"]) + 1

    async def next(self) -> Data | Watermark | Idle | None:
        if self._phase == 0:
            self._phase = 1
            self.seed_emissions += 1
            return Data(
                self._inputs.history,
                Cursor((1).to_bytes(8, "big"), {"phase": 1}),
            )
        if self._phase == 1:
            self._phase = 2
            return Watermark(_last_event_time(self._inputs.history_table))
        if self._phase == 2:
            self._phase = 3
            self.seed_probe_emissions += 1
            return Data(
                self._inputs.empty_probe,
                Cursor((2).to_bytes(8, "big"), {"phase": 2}),
            )
        if self._phase == 3:
            self.waiting_for_delta.set()
            await self._delta_release.wait()
            self._phase = 4
            self.delta_emissions += 1
            return Data(
                self._inputs.delta,
                Cursor((3).to_bytes(8, "big"), {"phase": 3}),
            )
        if self._phase == 4:
            self._phase = 5
            return Watermark(_last_event_time(self._inputs.delta_table))
        await self._hold_open.wait()
        return Idle()

    async def close(self) -> None:
        return None

    def release_delta(self) -> None:
        self._delta_release.set()


class _MaterializingSink:
    """Discard materialized seed output and retain only the measured delta."""

    def __init__(self, history_rows: int) -> None:
        self._expected_history_rows = history_rows
        self._expected_delta_rows: int | None = None
        self._capture_delta = False
        self._delta_tables: list[pa.Table] = []
        self.history_rows = 0
        self.delta_rows = 0
        self.history_done = asyncio.Event()
        self.delta_done = asyncio.Event()
        self.delta_output: pa.Table | None = None

    async def open(self) -> None:
        return None

    async def write(self, batch: Batch) -> None:
        table = batch.to_pyarrow()
        if not self._capture_delta:
            self.history_rows += table.num_rows
            if self.history_rows >= self._expected_history_rows:
                self.history_done.set()
            return
        self._delta_tables.append(table)
        self.delta_rows += table.num_rows
        if self.delta_rows >= self._expected_delta_rows:
            self.delta_output = pa.concat_tables(self._delta_tables)
            self.delta_done.set()

    async def close(self) -> None:
        return None

    def begin_delta_capture(self, delta_rows: int) -> None:
        if self.history_rows != self._expected_history_rows:
            raise AssertionError(
                "native seed output must finish before the append timer starts"
            )
        self._expected_delta_rows = delta_rows
        self._capture_delta = True


@dataclass(slots=True)
class _WarmNativeSession:
    job: Any
    source: _GatedAppendSource
    sink: _MaterializingSink
    seed_probe_operator_inputs: tuple[tuple[str, int], ...]
    seed_completed_ns: int


async def _wait_for_seed_probe(job: Any) -> tuple[tuple[str, int], ...]:
    deadline = asyncio.get_running_loop().time() + STREAM_TIMEOUT_SECONDS
    while True:
        status = job.status()
        if status["state"] == "failed":
            outcome = await job.wait_async()
            raise AssertionError(
                f"native seed probe failed before timing: {outcome.errors}"
            )
        rolling = status["operators"].get(ROLLING_NODE_ID)
        observed = (
            ()
            if rolling is None
            else ((ROLLING_NODE_ID, int(rolling["input_batches"])),)
        )
        if observed and observed[0][1] >= 2:
            return observed
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(
                "untimed empty probe did not pass the rolling operator before timing: "
                f"rolling_input_batches={observed}"
            )
        await asyncio.sleep(0.001)


async def _start_warm_native(
    stream_plan: object,
    inputs: _CaseInputs,
    state_root: Path,
) -> _WarmNativeSession:
    source = _GatedAppendSource(inputs)
    sink = _MaterializingSink(inputs.history.num_rows)
    max_rows = inputs.sql_full.num_rows
    runner = StreamingRunner(
        stream_plan,
        {"input": SourceBinding(source, watermark_policy=SourceProvidedWatermarks())},
        {"output": [SinkBinding.ordinary("warm-append", sink)]},
        ManagedCheckpointRuntime(state_root),
        config=StreamRuntimeConfig(
            checkpoint_interval=timedelta(hours=1),
            edge_budget=EdgeBudget(max_rows=max_rows, max_bytes=256 << 20),
        ),
    )
    job = await runner.start_async()
    try:
        await asyncio.wait_for(sink.history_done.wait(), timeout=STREAM_TIMEOUT_SECONDS)
    except TimeoutError as error:
        status = job.status()
        await job.cancel_async()
        raise AssertionError(
            "native seed did not materialize before the timeout: "
            f"source_phase={source._phase}, sink_history_rows={sink.history_rows}, "
            f"job_status={status}"
        ) from error
    except BaseException:
        await job.cancel_async()
        raise
    if sink.history_rows != inputs.history.num_rows:
        await job.cancel_async()
        raise AssertionError(
            f"native seed emitted {sink.history_rows} rows; "
            f"expected {inputs.history.num_rows}"
        )
    try:
        seed_probe_operator_inputs = await _wait_for_seed_probe(job)
        await asyncio.wait_for(
            source.waiting_for_delta.wait(), timeout=STREAM_TIMEOUT_SECONDS
        )
    except BaseException:
        await job.cancel_async()
        raise
    return _WarmNativeSession(
        job=job,
        source=source,
        sink=sink,
        seed_probe_operator_inputs=seed_probe_operator_inputs,
        seed_completed_ns=time.perf_counter_ns(),
    )


async def _append_native(
    session: _WarmNativeSession,
    delta_rows: int,
    *,
    measure: bool,
) -> _NativeObservation:
    session.sink.begin_delta_capture(delta_rows)
    append_started_ns = time.perf_counter_ns() if measure else None
    session.source.release_delta()
    await asyncio.wait_for(
        session.sink.delta_done.wait(), timeout=STREAM_TIMEOUT_SECONDS
    )
    seconds = (
        (time.perf_counter_ns() - append_started_ns) / 1_000_000_000
        if append_started_ns is not None
        else None
    )
    output = session.sink.delta_output
    if output is None:
        raise AssertionError("native delta completed without materialized output")
    observation = _NativeObservation(
        output=output,
        seconds=seconds,
        history_rows=session.sink.history_rows,
        delta_rows=session.sink.delta_rows,
        seed_emissions=session.source.seed_emissions,
        seed_probe_emissions=session.source.seed_probe_emissions,
        delta_emissions=session.source.delta_emissions,
        seed_probe_operator_inputs=session.seed_probe_operator_inputs,
        seed_completed_ns=session.seed_completed_ns,
        append_started_ns=append_started_ns,
    )
    _assert_native_measurement_contract(observation, delta_rows)
    return observation


def _assert_native_measurement_contract(
    observation: _NativeObservation, delta_rows: int
) -> None:
    if (
        observation.seed_emissions != 1
        or observation.seed_probe_emissions != 1
        or observation.delta_emissions != 1
    ):
        raise AssertionError(
            "each native sample must emit seed, empty probe, and delta exactly once"
        )
    if not observation.seed_probe_operator_inputs or any(
        input_batches < 2
        for _name, input_batches in observation.seed_probe_operator_inputs
    ):
        raise AssertionError("native seed probe must pass rolling before timing")
    if (
        observation.delta_rows != delta_rows
        or observation.output.num_rows != delta_rows
    ):
        raise AssertionError(
            "native timed output must contain exactly the appended rows"
        )
    if (
        observation.append_started_ns is not None
        and observation.seed_completed_ns > observation.append_started_ns
    ):
        raise AssertionError("native seed work entered the timed append region")


async def _execute_sql(
    sql_plan: object,
    inputs: _CaseInputs,
    *,
    measure: bool,
) -> tuple[pa.Table, float | None]:
    start_ns = time.perf_counter_ns() if measure else None
    result = await sql_plan.execute_async({"input": inputs.sql_full})
    output = result.outputs["output"].to_pyarrow()
    seconds = (
        (time.perf_counter_ns() - start_ns) / 1_000_000_000
        if start_ns is not None
        else None
    )
    if output.num_rows != inputs.sql_full.num_rows:
        raise AssertionError("SQL baseline must fully recompute history plus delta")
    return output, seconds


async def _paired_observation(
    window: int,
    sql_plan: object,
    inputs: _CaseInputs,
    state_root: Path,
    *,
    native_first: bool,
    measure: bool,
) -> _PairedObservation:
    stream_plan = _incremental_program(window).compile_stream(Runtime())
    session = await _start_warm_native(stream_plan, inputs, state_root)
    try:
        if native_first:
            native = await _append_native(
                session, inputs.delta.num_rows, measure=measure
            )
            sql_output, sql_seconds = await _execute_sql(
                sql_plan, inputs, measure=measure
            )
            order = "native-first"
        else:
            sql_output, sql_seconds = await _execute_sql(
                sql_plan, inputs, measure=measure
            )
            native = await _append_native(
                session, inputs.delta.num_rows, measure=measure
            )
            order = "sql-first"
    finally:
        outcome = await session.job.cancel_async()
        if outcome.state != "cancelled":
            raise AssertionError(f"warm native sample ended as {outcome.state!r}")
    return _PairedObservation(
        order=order,
        native=native,
        sql_output=sql_output,
        sql_seconds=sql_seconds,
        sql_input_rows=inputs.sql_full.num_rows,
    )


def _assert_paired_observation(
    observation: _PairedObservation,
    *,
    history_rows: int,
    delta_rows: int,
    window: int,
) -> None:
    if observation.native.history_rows != history_rows:
        raise AssertionError(
            "each native sample must begin from the same logical history"
        )
    if observation.sql_input_rows != history_rows + delta_rows:
        raise AssertionError("SQL baseline did not process history plus delta")
    sql_delta = observation.sql_output.slice(history_rows, delta_rows)
    _assert_equivalent_delta(observation.native.output, sql_delta, window)


def _summary(durations: list[float], denominator_rows: int) -> dict[str, float]:
    median_seconds = statistics.median(durations)
    return {
        "total_seconds": sum(durations),
        "mean_seconds": statistics.fmean(durations),
        "median_seconds": median_seconds,
        "throughput_rows_per_second": denominator_rows / median_seconds,
    }


def test_warm_append_profile_contract() -> None:
    profile = _benchmark_profile()

    assert profile.windows == WINDOW_ROWS
    assert profile.delta_rows == DELTA_ROWS
    assert profile.history_rows % ENTITY_COUNT == 0
    if profile.name == "standard":
        assert profile.history_rows == STANDARD_HISTORY_ROWS
        assert profile.measured_samples == STANDARD_SAMPLE_COUNT


@pytest.mark.benchmark(
    group=benchmark_group("rolling-warm-append"), min_rounds=1, max_time=0.1
)
@pytest.mark.parametrize("window", PROFILE.windows)
@pytest.mark.parametrize("delta_rows", PROFILE.delta_rows)
def test_warm_state_append_vs_sql_full_recompute(
    benchmark: BenchmarkFixture,
    tmp_path: Path,
    window: int,
    delta_rows: int,
) -> None:
    inputs = _case_inputs(PROFILE.history_rows, delta_rows)
    sql_plan = (
        PipelineBuilder(f"dal-184-warm-sql-{window}")
        .sql("features", _sql_query(window))
        .compile_batch()
    )

    warm = asyncio.run(
        _paired_observation(
            window,
            sql_plan,
            inputs,
            tmp_path / "warm-up",
            native_first=True,
            measure=False,
        )
    )
    _assert_paired_observation(
        warm,
        history_rows=PROFILE.history_rows,
        delta_rows=delta_rows,
        window=window,
    )

    observations: list[_PairedObservation] = []
    for index in range(PROFILE.measured_samples):
        observation = asyncio.run(
            _paired_observation(
                window,
                sql_plan,
                inputs,
                tmp_path / f"sample-{index}",
                native_first=index % 2 == 0,
                measure=True,
            )
        )
        _assert_paired_observation(
            observation,
            history_rows=PROFILE.history_rows,
            delta_rows=delta_rows,
            window=window,
        )
        observations.append(observation)

    native_durations = [
        float(observation.native.seconds) for observation in observations
    ]
    sql_durations = [float(observation.sql_seconds) for observation in observations]
    native_summary = _summary(native_durations, delta_rows)
    sql_summary = _summary(sql_durations, PROFILE.history_rows + delta_rows)
    speedup = sql_summary["median_seconds"] / native_summary["median_seconds"]
    samples = [
        {
            "order": observation.order,
            "native_append_seconds": observation.native.seconds,
            "sql_full_recompute_seconds": observation.sql_seconds,
            "native_seed_rows_before_timer": observation.native.history_rows,
            "native_timed_output_rows": observation.native.delta_rows,
            "native_seed_probe_operator_inputs": dict(
                observation.native.seed_probe_operator_inputs
            ),
            "sql_input_rows": observation.sql_input_rows,
            "seed_completed_before_timer": (
                observation.native.append_started_ns is not None
                and observation.native.seed_completed_ns
                <= observation.native.append_started_ns
            ),
        }
        for observation in observations
    ]
    record_symbolic_benchmark(
        benchmark,
        scenario=SCENARIO,
        input_rows=PROFILE.history_rows + delta_rows,
        output_rows=delta_rows,
        extra={
            "comparison_contract": "warm-append-v1",
            "workload_contract": WORKLOAD_CONTRACT,
            "build_profile": os.environ.get(
                "CALC_FLOW_BENCHMARK_PROFILE", "unspecified"
            ),
            "calc_flow_version": version("calc-flow"),
            "operating_system_release": platform.release(),
            "history_rows": PROFILE.history_rows,
            "delta_rows": delta_rows,
            "window_rows": window,
            "entities": inputs.entities,
            "statistics": list(STATISTICS),
            "input_seed": SYMBOLIC_SEED,
            "input_pattern": "deterministic interleaved per-symbol quotes",
            "null_rows_in_sql_input": inputs.null_rows,
            "null_period_rows_per_entity": NULL_PERIODS,
            "null_policy": "exclude null samples; null when no valid sample exists",
            "boundary_policy": "current row plus window_rows - 1 preceding entity rows",
            "seed_strategy": SEED_STRATEGY,
            "seed_state_proof": (
                "an empty probe is processed after the seed watermark; rolling reports "
                "at least two input batches before timing"
            ),
            "timing_boundary_native": (
                "release delta source gate through sink Arrow "
                "materialization of delta rows"
            ),
            "timing_boundary_sql": (
                "execute precompiled plan on history plus delta "
                "through Arrow collection"
            ),
            "native_scope": (
                "end-to-end public StreamingRunner append including "
                "Python source/sink, "
                "async scheduling, watermark, and Arrow materialization"
            ),
            "sql_scope": (
                "DataFusion full-history recompute with precompiled query plan"
            ),
            "warm_up_runs": WARM_UP_RUNS,
            "measured_sample_count": PROFILE.measured_samples,
            "summary_statistic": "median of alternating same-process samples",
            "relative_tolerance": RELATIVE_TOLERANCE,
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "native_append": native_summary,
            "native_throughput_denominator": "delta_rows",
            "sql_full_recompute": sql_summary,
            "sql_throughput_denominator": "history_rows + delta_rows",
            "speedup_sql_over_native": speedup,
            "paired_samples": samples,
            "pytest_benchmark_scope": (
                "one additional full harness lifecycle; authoritative paired latencies "
                "are stored in extra_info"
            ),
        },
    )
    print(
        "DAL-184 warm append: "
        f"history={PROFILE.history_rows}, delta={delta_rows}, W={window}, "
        f"native={native_summary['median_seconds']:.6f}s "
        f"({native_summary['throughput_rows_per_second']:.0f} delta rows/s), "
        f"sql={sql_summary['median_seconds']:.6f}s "
        f"({sql_summary['throughput_rows_per_second']:.0f} total rows/s), "
        f"speedup={speedup:.2f}x"
    )

    def exercise_once() -> None:
        integration = asyncio.run(
            _paired_observation(
                window,
                sql_plan,
                inputs,
                tmp_path / "pytest-benchmark",
                native_first=True,
                measure=True,
            )
        )
        _assert_paired_observation(
            integration,
            history_rows=PROFILE.history_rows,
            delta_rows=delta_rows,
            window=window,
        )

    benchmark.pedantic(exercise_once, rounds=1, iterations=1)

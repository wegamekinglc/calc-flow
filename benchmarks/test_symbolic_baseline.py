"""SCE-01 symbolic execution baselines over hand-built calc-flow plans.

Each scenario measures the existing runtime doing the shape of work the
symbolic layer will later compile: row-local projections, rolling
per-entity features, complete-group cross sections, provider-owned matrix
products, and stateful stream checkpoints. The plans are hand-built with
current public APIs (plus the documented private graph hook for the
stream-only window node) so no symbolic implementation can influence the
numbers.
"""

from __future__ import annotations

import asyncio
import time
from itertools import count
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from benchmarks.support import BenchmarkFixture, benchmark_group, selected_scale
from benchmarks.symbolic_support import (
    PROJECTION_COLUMN_COUNT,
    ROLLING_LONG_WINDOW,
    ROLLING_SHORT_WINDOW,
    STREAM_BATCH_ROWS,
    STREAM_ENTITIES,
    STREAM_WINDOW_SECONDS,
    arrow_column_bytes,
    counting_matmul_runtime,
    directory_bytes,
    matmul_workload,
    quote_workload,
    record_symbolic_benchmark,
    stream_batches,
    stream_graph_json,
)
from calc_flow import (
    Batch,
    Cursor,
    Data,
    DisabledWatermarks,
    Idle,
    ManagedCheckpointRuntime,
    NativeWatermarkCapability,
    PipelineBuilder,
    ReplayPositioning,
    SinkBinding,
    SourceBinding,
    SourceCapabilities,
    SourceDeliveryCapability,
    StreamingRunner,
)

PROJECTION_QUERY = """
SELECT
  sequence,
  ln(close) AS f01,
  sqrt(close) AS f02,
  close * volume / 10000.0 AS f03,
  close / volume * 1000.0 AS f04,
  power(close, 2) / 10000.0 AS f05,
  abs(close - 100.0) AS f06,
  ln(volume) AS f07,
  sqrt(volume) AS f08,
  close + volume / 10000.0 AS f09,
  close - volume / 100000.0 AS f10,
  exp(close / 100.0) AS f11,
  ln(close * volume) AS f12,
  sqrt(close * volume / 10000.0) AS f13,
  power(close / 100.0, 3) AS f14,
  abs(volume - 5000.0) AS f15,
  ln(abs(close - 100.0) + 1.0) AS f16,
  sqrt(abs(close - 100.0) + 1.0) AS f17,
  close / (volume + 1.0) AS f18,
  ln(volume + 1.0) AS f19,
  power(volume / 10000.0, 2) AS f20
FROM input
"""

_ROLLING_FRAME = (
    "PARTITION BY symbol ORDER BY event_time, sequence ROWS BETWEEN "
    "{preceding} PRECEDING AND CURRENT ROW"
)

ROLLING_QUERY = f"""
SELECT
  event_time,
  sequence,
  symbol,
  ln(close)
    - lag(ln(close)) OVER (PARTITION BY symbol ORDER BY event_time, sequence)
    AS return_1,
  ln(close)
    - lag(ln(close), 2) OVER (PARTITION BY symbol ORDER BY event_time, sequence)
    AS return_2,
  avg(ln(close)) OVER ({_ROLLING_FRAME.format(preceding=ROLLING_SHORT_WINDOW - 1)})
    AS mean_log_{ROLLING_SHORT_WINDOW},
  stddev_samp(ln(close)) OVER (
    {_ROLLING_FRAME.format(preceding=ROLLING_SHORT_WINDOW - 1)})
    AS std_log_{ROLLING_SHORT_WINDOW},
  sum(close) OVER ({_ROLLING_FRAME.format(preceding=ROLLING_SHORT_WINDOW - 1)})
    AS sum_close_{ROLLING_SHORT_WINDOW},
  min(close) OVER ({_ROLLING_FRAME.format(preceding=ROLLING_SHORT_WINDOW - 1)})
    AS min_close_{ROLLING_SHORT_WINDOW},
  max(close) OVER ({_ROLLING_FRAME.format(preceding=ROLLING_SHORT_WINDOW - 1)})
    AS max_close_{ROLLING_SHORT_WINDOW},
  count(close) OVER ({_ROLLING_FRAME.format(preceding=ROLLING_SHORT_WINDOW - 1)})
    AS count_close_{ROLLING_SHORT_WINDOW},
  avg(ln(close)) OVER ({_ROLLING_FRAME.format(preceding=ROLLING_LONG_WINDOW - 1)})
    AS mean_log_{ROLLING_LONG_WINDOW},
  stddev_samp(ln(close)) OVER (
    {_ROLLING_FRAME.format(preceding=ROLLING_LONG_WINDOW - 1)})
    AS std_log_{ROLLING_LONG_WINDOW},
  sum(close) OVER ({_ROLLING_FRAME.format(preceding=ROLLING_LONG_WINDOW - 1)})
    AS sum_close_{ROLLING_LONG_WINDOW},
  min(close) OVER ({_ROLLING_FRAME.format(preceding=ROLLING_LONG_WINDOW - 1)})
    AS min_close_{ROLLING_LONG_WINDOW},
  max(close) OVER ({_ROLLING_FRAME.format(preceding=ROLLING_LONG_WINDOW - 1)})
    AS max_close_{ROLLING_LONG_WINDOW},
  count(close) OVER ({_ROLLING_FRAME.format(preceding=ROLLING_LONG_WINDOW - 1)})
    AS count_close_{ROLLING_LONG_WINDOW}
FROM input
"""

CROSS_SECTION_QUERY = """
WITH bucketed AS (
  SELECT
    date_bin(interval '1 second', event_time, timestamp '1970-01-01T00:00:00')
      AS bucket,
    symbol,
    industry,
    close
  FROM input
)
SELECT
  symbol,
  industry,
  close,
  rank() OVER (PARTITION BY bucket, industry ORDER BY close DESC) AS rank_desc,
  rank() OVER (PARTITION BY bucket, industry ORDER BY close ASC) AS rank_asc,
  (close - avg(close) OVER (PARTITION BY bucket, industry))
    / nullif(stddev_samp(close) OVER (PARTITION BY bucket, industry), 0.0)
    AS zscore,
  avg(close) OVER (PARTITION BY bucket, industry) AS group_mean,
  count(close) OVER (PARTITION BY bucket, industry) AS group_size
FROM bucketed
"""

MATMUL_FEATURE_COUNT = 20
MATMUL_OUTPUT_WIDTH = 8
_MATMUL_EXPRESSIONS: tuple[tuple[str, str], ...] = (
    ("ln(close)", "f01"),
    ("sqrt(close)", "f02"),
    ("close * volume / 10000.0", "f03"),
    *(
        (f"close * {index / 10.0} + volume / {index * 1000.0}", f"f{index:02d}")
        for index in range(4, 21)
    ),
)
_MATMUL_FEATURES = tuple(alias for _expression, alias in _MATMUL_EXPRESSIONS)


def _matmul_feature_query(backend: str) -> str:
    cast = "CAST({expression} AS real) AS {alias}"
    plain = "{expression} AS {alias}"
    template = cast if backend == "jax" else plain
    projection = ",\n  ".join(
        template.format(expression=expression, alias=alias)
        for expression, alias in _MATMUL_EXPRESSIONS
    )
    return f"SELECT\n  {projection}\nFROM input"


@pytest.mark.benchmark(
    group=benchmark_group("symbolic-projection"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_projection_twenty_derived_columns(
    benchmark: BenchmarkFixture, _scale: str
) -> None:
    workload = quote_workload()
    plan = (
        PipelineBuilder("symbolic-projection-baseline")
        .sql("features", PROJECTION_QUERY)
        .compile_batch()
    )

    warm_result = plan.execute({"input": workload.batch})
    warm_output = warm_result.outputs["output"]
    record_symbolic_benchmark(
        benchmark,
        scenario="symbolic_projection_20_columns",
        input_rows=workload.rows,
        output_rows=warm_output.num_rows,
        metrics=warm_result.datafusion_metrics,
        extra={"derived_columns": PROJECTION_COLUMN_COUNT},
    )

    result = benchmark(plan.execute, {"input": workload.batch})
    output = result.outputs["output"].to_pyarrow()

    assert output.num_rows == workload.rows
    derived = [name for name in output.column_names if name.startswith("f")]
    assert len(derived) == PROJECTION_COLUMN_COUNT
    assert sum(output[name].null_count for name in derived) == 0


@pytest.mark.benchmark(
    group=benchmark_group("symbolic-rolling"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_rolling_temporal_features(benchmark: BenchmarkFixture, _scale: str) -> None:
    workload = quote_workload()
    plan = (
        PipelineBuilder("symbolic-rolling-baseline")
        .sql("features", ROLLING_QUERY)
        .compile_batch()
    )

    warm_result = plan.execute({"input": workload.batch})
    warm_output = warm_result.outputs["output"]
    record_symbolic_benchmark(
        benchmark,
        scenario="symbolic_rolling_20_60_row_features",
        input_rows=workload.rows,
        output_rows=warm_output.num_rows,
        metrics=warm_result.datafusion_metrics,
        extra={
            "rolling_short_window": ROLLING_SHORT_WINDOW,
            "rolling_long_window": ROLLING_LONG_WINDOW,
            "entities": workload.entities,
        },
    )

    result = benchmark(plan.execute, {"input": workload.batch})
    output = result.outputs["output"].to_pyarrow()

    assert output.num_rows == workload.rows
    # Window execution emits rows grouped per entity partition, so the first
    # output row is one entity's first observation and the last output row is
    # the trailing observation of the final partition. Every entity owns the
    # same number of rows, so both caps are deterministic.
    short_counts = output[f"count_close_{ROLLING_SHORT_WINDOW}"].to_pylist()
    assert short_counts[0] == 1
    assert short_counts[-1] == min(workload.rows_per_entity, ROLLING_SHORT_WINDOW)
    long_counts = output[f"count_close_{ROLLING_LONG_WINDOW}"].to_pylist()
    assert long_counts[-1] == min(workload.rows_per_entity, ROLLING_LONG_WINDOW)


@pytest.mark.benchmark(
    group=benchmark_group("symbolic-cross-section"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_cross_section_rank_and_zscore(
    benchmark: BenchmarkFixture, _scale: str
) -> None:
    workload = quote_workload()
    plan = (
        PipelineBuilder("symbolic-cross-section-baseline")
        .sql("cross_section", CROSS_SECTION_QUERY)
        .compile_batch()
    )

    warm_result = plan.execute({"input": workload.batch})
    warm_output = warm_result.outputs["output"]
    record_symbolic_benchmark(
        benchmark,
        scenario="symbolic_cross_section_rank_zscore",
        input_rows=workload.rows,
        output_rows=warm_output.num_rows,
        metrics=warm_result.datafusion_metrics,
        extra={
            "entities": workload.entities,
            "industries": workload.industries,
            "group_size": workload.entities // workload.industries,
        },
    )

    result = benchmark(plan.execute, {"input": workload.batch})
    output = result.outputs["output"].to_pyarrow()

    assert output.num_rows == workload.rows
    group_size = workload.entities // workload.industries
    sizes = output["group_size"].to_pylist()
    assert set(sizes) == {group_size}
    ranks_up = output["rank_asc"].to_pylist()
    ranks_down = output["rank_desc"].to_pylist()
    assert all(
        up + down == group_size + 1
        for up, down in zip(ranks_up, ranks_down, strict=True)
    )
    zscores = output["zscore"].to_pylist()
    assert all(value is not None for value in zscores)
    assert max(abs(value) for value in zscores) < group_size


@pytest.mark.parametrize("backend", ("numpy", "jax"))
@pytest.mark.benchmark(
    group=benchmark_group("symbolic-matmul"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_table_matmul(benchmark: BenchmarkFixture, backend: str, _scale: str) -> None:
    workload = matmul_workload()
    runtime, counting = counting_matmul_runtime(backend)
    feature_query = _matmul_feature_query(backend)
    plan = (
        PipelineBuilder(f"symbolic-matmul-{backend}-baseline")
        .sql("features", feature_query)
        .table_matmul("score", backend=backend, columns=_MATMUL_FEATURES)
        .connect("features", "score", target_port="table")
        .compile_batch(runtime)
    )
    feature_table = (
        PipelineBuilder(f"symbolic-matmul-{backend}-features")
        .sql("features", feature_query)
        .compile_batch()
        .execute({"input": workload.batch})
        .outputs["output"]
    )
    dtype = np.float32 if backend == "jax" else np.float64
    weights_source = (
        np.arange(MATMUL_FEATURE_COUNT * MATMUL_OUTPUT_WIDTH, dtype=dtype).reshape(
            MATMUL_FEATURE_COUNT, MATMUL_OUTPUT_WIDTH
        )
        / MATMUL_FEATURE_COUNT
    )
    weights = Batch.from_array(weights_source, backend=backend)

    inputs = {"input": workload.batch, "weights": weights}
    warm_result = plan.execute(inputs)
    warm_output = warm_result.outputs["output"]
    item_size = np.dtype(dtype).itemsize
    record_symbolic_benchmark(
        benchmark,
        scenario=f"symbolic_table_matmul_{backend}",
        input_rows=workload.rows,
        output_rows=warm_output.num_rows,
        backend=backend,
        extra={
            "provider_calls_per_execute": counting.calls,
            "matmul_feature_columns": MATMUL_FEATURE_COUNT,
            "matmul_output_width": MATMUL_OUTPUT_WIDTH,
            "arrow_column_bytes": arrow_column_bytes(feature_table, _MATMUL_FEATURES),
            "dense_matrix_bytes": workload.rows * MATMUL_FEATURE_COUNT * item_size,
        },
    )

    def synchronized_execute() -> Any:
        result = plan.execute(inputs)
        output = result.outputs["output"]
        block_until_ready = getattr(output.array, "block_until_ready", None)
        if callable(block_until_ready):
            block_until_ready()
        return result

    result = benchmark(synchronized_execute)
    output = result.outputs["output"]

    assert output.num_rows == workload.rows
    assert tuple(output.array.shape) == (
        workload.rows,
        MATMUL_OUTPUT_WIDTH,
    )
    assert counting.calls >= 1


class _BaselineSource:
    """Replayable source emitting interleaved-entity batches.

    Watermarks are disabled so the tumbling windows close during the drain
    phase; a cancelled job keeps a replayable checkpoint (only end-of-input
    marks the source ended) and recovery reopens at the exact cursor.
    """

    def __init__(
        self,
        batches: list[tuple[pa.Table, int]],
        pause_at: int | None = None,
    ) -> None:
        self._batches = batches
        self._pause_at = pause_at
        self._index = 0

    @property
    def delivered_batches(self) -> int:
        return self._index

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            SourceDeliveryCapability.LOSSLESS,
            max_batch_rows=STREAM_BATCH_ROWS,
            max_batch_bytes=8 * 1024 * 1024,
            native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self._index = 0 if cursor is None else int(cursor.payload["batches"])

    async def next(self) -> Data | Idle | None:
        if self._pause_at is not None and self._index >= self._pause_at:
            return Idle()
        if self._index >= len(self._batches):
            return None
        table, max_micros = self._batches[self._index]
        self._index += 1
        return Data(
            Batch.from_pyarrow(table),
            Cursor(
                self._index.to_bytes(8, "big"),
                {"batches": self._index, "max_micros": int(max_micros)},
            ),
        )

    async def close(self) -> None:
        return None


class _BaselineSink:
    def __init__(self) -> None:
        self.rows = 0

    async def open(self) -> None:
        return None

    async def write(self, batch: Batch) -> None:
        self.rows += batch.num_rows

    async def close(self) -> None:
        return None


def _expected_stream_window_rows(total_rows: int) -> int:
    rows_per_entity = total_rows // STREAM_ENTITIES
    windows_per_entity = (rows_per_entity + STREAM_WINDOW_SECONDS - 1) // (
        STREAM_WINDOW_SECONDS
    )
    return STREAM_ENTITIES * windows_per_entity


def _run_stream_lifecycle(
    state_root: Path, batches: list[tuple[pa.Table, int]]
) -> dict[str, Any]:
    """One measured run/checkpoint/cancel/recover lifecycle (own loop).

    The first run pauses after half the batches, takes one durable
    checkpoint, and is cancelled. Cancelling drops the unconsumed half by
    design, so the recovery run restores the checkpointed operator state
    (the active tumbling windows built from the consumed half) and flushes
    it at drain without replaying cancelled input.
    """
    state_root.mkdir(parents=True, exist_ok=True)
    graph_json = stream_graph_json()
    pause_at = max(1, len(batches) // 2)
    consumed_rows = sum(table.num_rows for table, _micros in batches[:pause_at])

    async def lifecycle() -> dict[str, Any]:
        source = _BaselineSource(batches, pause_at)
        sink = _BaselineSink()
        job = await StreamingRunner(
            PipelineBuilder._from_json(graph_json).compile_stream(),
            {"input": SourceBinding(source, watermark_policy=DisabledWatermarks())},
            {"output": [SinkBinding.ordinary("baseline", sink)]},
            ManagedCheckpointRuntime(state_root),
        ).start_async()
        status = job.status()
        deadline = asyncio.get_running_loop().time() + 30
        # Wait until the window operator itself reports consuming the paused
        # batches: checkpointing on the source attribute alone can race the
        # batch's edge delivery and snapshot operator state that is still
        # empty, which would strand those rows outside the checkpoint.
        while status["operators"]["windows"]["input_batches"] < pause_at:
            status = job.status()
            if status["state"] == "failed":
                outcome = await job.wait_async()
                raise AssertionError(f"stream job failed early: {outcome.errors}")
            if asyncio.get_running_loop().time() > deadline:
                outcome = await job.wait_async()
                raise AssertionError(
                    f"stream job stalled in state {status['state']!r} "
                    f"after {source.delivered_batches} batches: {outcome.errors}"
                )
            await asyncio.sleep(0.001)
        checkpoint_start = time.perf_counter()
        epoch = await job.trigger_checkpoint_async()
        checkpoint_seconds = time.perf_counter() - checkpoint_start
        await job.cancel_async()
        checkpoint_bytes = directory_bytes(state_root)

        recovery_source = _BaselineSource(batches)
        recovery_sink = _BaselineSink()
        recovery_start = time.perf_counter()
        recovery_job = await StreamingRunner(
            PipelineBuilder._from_json(graph_json).compile_stream(),
            {
                "input": SourceBinding(
                    recovery_source, watermark_policy=DisabledWatermarks()
                )
            },
            {"output": [SinkBinding.ordinary("baseline", recovery_sink)]},
            ManagedCheckpointRuntime(state_root),
        ).start_async()
        recovery_seconds = time.perf_counter() - recovery_start
        outcome = await recovery_job.shutdown_async()
        return {
            "state": outcome.state,
            "checkpoint_epoch": epoch,
            "checkpoint_duration_seconds": checkpoint_seconds,
            "checkpoint_bytes": checkpoint_bytes,
            "recovery_duration_seconds": recovery_seconds,
            "checkpoint_batches": pause_at,
            "consumed_rows": consumed_rows,
            "resumed_batches": recovery_source.delivered_batches,
            # Every window accumulated before the checkpoint is emitted
            # exactly once across the cancelled and recovered runs.
            "sink_rows": sink.rows + recovery_sink.rows,
        }

    return asyncio.run(lifecycle())


@pytest.mark.benchmark(
    group=benchmark_group("symbolic-stream"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_stream_window_checkpoint_and_recovery(
    benchmark: BenchmarkFixture, _scale: str, tmp_path: Path
) -> None:
    batches, total_rows = stream_batches()
    pause_at = max(1, len(batches) // 2)
    consumed_rows = sum(table.num_rows for table, _micros in batches[:pause_at])
    expected_rows = _expected_stream_window_rows(consumed_rows)

    measured = _run_stream_lifecycle(tmp_path / "measured", batches)
    assert measured["state"] == "completed"
    assert measured["sink_rows"] == expected_rows

    record_symbolic_benchmark(
        benchmark,
        scenario="symbolic_stream_window_checkpoint",
        input_rows=total_rows,
        output_rows=measured["sink_rows"],
        extra={
            "stream_batches": len(batches),
            "stream_batch_rows": STREAM_BATCH_ROWS,
            "stream_entities": STREAM_ENTITIES,
            "stream_window_seconds": STREAM_WINDOW_SECONDS,
            "checkpoint_batches": measured["checkpoint_batches"],
            "consumed_rows": measured["consumed_rows"],
            "checkpoint_epoch": measured["checkpoint_epoch"],
            "checkpoint_duration_seconds": measured["checkpoint_duration_seconds"],
            "checkpoint_bytes": measured["checkpoint_bytes"],
            "recovery_duration_seconds": measured["recovery_duration_seconds"],
            "recovery_resumed_batches": measured["resumed_batches"],
        },
    )

    rounds = count()

    def timed_lifecycle() -> dict[str, Any]:
        return _run_stream_lifecycle(tmp_path / f"round-{next(rounds)}", batches)

    result = benchmark(timed_lifecycle)

    assert result["state"] == "completed"
    assert result["sink_rows"] == expected_rows

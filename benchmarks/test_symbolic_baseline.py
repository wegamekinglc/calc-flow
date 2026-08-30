"""SCE-01 baselines and milestone pairs over calc-flow plans.

Each scenario measures the existing runtime doing the shape of work the
symbolic layer will later compile: row-local projections, rolling
per-entity features, complete-group cross sections, provider-owned matrix
products, and stateful stream checkpoints. Delivered milestone scenarios
pair equivalent hand-built and symbolic plans in one benchmark process;
scenarios awaiting their milestone pair remain hand-built baselines.
"""

from __future__ import annotations

import asyncio
import json
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
    Runtime,
    SinkBinding,
    SourceBinding,
    SourceCapabilities,
    SourceDeliveryCapability,
    StreamingRunner,
)
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    cs,
    duration,
    exact_time,
    row,
    rows,
    table,
    table_input,
    ts,
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

_ROW_LOCAL_SCENARIO = "sce05_row_local_20_columns"
_ROW_LOCAL_WORKLOAD_CONTRACT = "sce05-row-local-v1"
_ROW_LOCAL_PAIRED_SAMPLES = 60
_ROW_LOCAL_PAIR_PROJECTIONS = (
    "ln(close) AS f01",
    "sqrt(close) AS f02",
    "close * volume / 10000.0 AS f03",
    "close / volume * 1000.0 AS f04",
    "close * close / 10000.0 AS f05",
    "abs(close - 100.0) AS f06",
    "ln(volume) AS f07",
    "sqrt(volume) AS f08",
    "close + volume / 10000.0 AS f09",
    "close - volume / 100000.0 AS f10",
    "exp(close / 100.0) AS f11",
    "ln(close * (volume + 0.5)) AS f12",
    "sqrt(close * (volume / 9999.0)) AS f13",
    "(close + 3.0) / (volume + 4.0) AS f14",
    "abs(volume - 5000.0) AS f15",
    "ln(abs(close - 101.0) + 1.0) AS f16",
    "sqrt(abs(close - 102.0) + 1.0) AS f17",
    "close / (volume + 1.0) AS f18",
    "ln(volume + 2.0) AS f19",
    "volume * volume / 100000000.0 AS f20",
)
_ROW_LOCAL_PAIR_QUERY = (
    "SELECT\n  sequence,\n  "
    + ",\n  ".join(_ROW_LOCAL_PAIR_PROJECTIONS)
    + "\nFROM input"
)


def _projection_symbolic_program() -> Program:
    quotes = table_input(
        "quotes",
        schema=[
            Field("event_time", "timestamp[us]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("industry", "string", nullable=False),
            Field("close", "float64"),
            Field("volume", "float64"),
        ],
    )
    close = quotes["close"]
    volume = quotes["volume"]
    feature_names = tuple(f"f{index:02d}" for index in range(1, 21))
    derived = quotes.with_columns(
        FeatureSet(
            [
                ("f01", row.log(close)),
                ("f02", row.sqrt(close)),
                ("f03", close * volume / 10_000.0),
                ("f04", close / volume * 1_000.0),
                ("f05", close * close / 10_000.0),
                ("f06", row.abs(close - 100.0)),
                ("f07", row.log(volume)),
                ("f08", row.sqrt(volume)),
                ("f09", close + volume / 10_000.0),
                ("f10", close - volume / 100_000.0),
                ("f11", row.exp(close / 100.0)),
                ("f12", row.log(close * (volume + 0.5))),
                ("f13", row.sqrt(close * (volume / 9_999.0))),
                ("f14", (close + 3.0) / (volume + 4.0)),
                ("f15", row.abs(volume - 5_000.0)),
                ("f16", row.log(row.abs(close - 101.0) + 1.0)),
                ("f17", row.sqrt(row.abs(close - 102.0) + 1.0)),
                ("f18", close / (volume + 1.0)),
                ("f19", row.log(volume + 2.0)),
                ("f20", volume * volume / 100_000_000.0),
            ]
        )
    )
    projected = table.project(derived, ("sequence", *feature_names))
    return Program(
        "symbolic-projection-pair",
        inputs=[quotes],
        outputs=[("features", projected)],
    )


def _timed_execute(plan: object, inputs: dict[str, Batch]) -> tuple[Any, float]:
    start = time.perf_counter_ns()
    result = plan.execute(inputs)
    seconds = (time.perf_counter_ns() - start) / 1_000_000_000
    return result, seconds


def _alternating_plan_samples(
    hand_built_plan: object,
    symbolic_plan: object,
    inputs: dict[str, Batch],
    *,
    sample_count: int,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for index in range(sample_count):
        if index % 2 == 0:
            _hand_result, hand_seconds = _timed_execute(hand_built_plan, inputs)
            _symbolic_result, symbolic_seconds = _timed_execute(symbolic_plan, inputs)
            order = "hand-built-first"
        else:
            _symbolic_result, symbolic_seconds = _timed_execute(symbolic_plan, inputs)
            _hand_result, hand_seconds = _timed_execute(hand_built_plan, inputs)
            order = "symbolic-first"
        samples.append(
            {
                "order": order,
                "hand_built_seconds": hand_seconds,
                "symbolic_seconds": symbolic_seconds,
            }
        )
    return samples


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

_SCE08_SCENARIO = "sce08_temporal_catalog"
_SCE08_WORKLOAD_CONTRACT = "sce08-temporal-v1"
_SCE08_DURATION_MICROS = 60_000_000
_SCE08_PAIRED_SAMPLES = 30
_SCE14_SCENARIO = "sce14_cross_domain_sharing"
_SCE14_WORKLOAD_CONTRACT = "sce14-cross-domain-sharing-v1"
_SCE14_PAIRED_SAMPLES = 30


def _sce08_input_schema() -> list[dict[str, object]]:
    return [
        {"name": "event_time", "data_type": "timestamp[us, UTC]", "nullable": False},
        {"name": "sequence", "data_type": "uint64", "nullable": False},
        {"name": "symbol", "data_type": "string", "nullable": False},
        {"name": "industry", "data_type": "string", "nullable": False},
        {"name": "close", "data_type": "float64", "nullable": True},
        {"name": "volume", "data_type": "float64", "nullable": True},
    ]


def _utc_quote_batch() -> Batch:
    input_table = quote_workload().batch.to_pyarrow()
    event_time_index = input_table.schema.get_field_index("event_time")
    return Batch.from_pyarrow(
        input_table.set_column(
            event_time_index,
            pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
            input_table["event_time"].cast(pa.timestamp("us", tz="UTC")),
        )
    )


def _sce14_programs() -> tuple[Program, Program, Program]:
    quotes = table_input(
        "quotes",
        schema=[
            Field("event_time", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("industry", "string", nullable=False),
            Field("close", "float64"),
            Field("volume", "float64"),
        ],
        entity_by=("symbol",),
        event_time="event_time",
        sequence_by=("sequence",),
    )
    group = exact_time(quotes["event_time"], partition_by=(quotes["industry"],))
    first = quotes.with_columns(
        FeatureSet(
            (
                ("short_mean", ts.mean(quotes["close"], window=rows(20))),
                ("rank", cs.rank(quotes["close"], group=group)),
            )
        )
    )
    second = quotes.with_columns(
        FeatureSet(
            (
                ("long_max", ts.max(quotes["close"], window=rows(60))),
                ("zscore", cs.zscore(quotes["close"], group=group)),
            )
        )
    )
    first_program = Program(
        "sce14-first-reference", inputs=(quotes,), outputs=(("first", first),)
    )
    second_program = Program(
        "sce14-second-reference", inputs=(quotes,), outputs=(("second", second),)
    )
    optimized = Program(
        "sce14-optimized",
        inputs=(quotes,),
        outputs=(("first", first), ("second", second)),
    )
    return first_program, second_program, optimized


class _ReferencePlans:
    def __init__(self, first: Any, second: Any) -> None:
        self._first = first
        self._second = second

    def execute(self, inputs: dict[str, Batch]) -> tuple[Any, Any]:
        return self._first.execute(inputs), self._second.execute(inputs)


def _sce08_output_schema() -> list[dict[str, object]]:
    return [
        *_sce08_input_schema(),
        {
            "name": "duration_min_close",
            "data_type": "float64",
            "nullable": True,
        },
        {
            "name": "duration_max_close",
            "data_type": "float64",
            "nullable": True,
        },
        {
            "name": "duration_cov_close_volume",
            "data_type": "float64",
            "nullable": True,
        },
        {
            "name": "duration_corr_close_volume",
            "data_type": "float64",
            "nullable": True,
        },
    ]


def _sce08_output_spec(
    kind: str,
    output: str,
    *,
    input_name: str | None = None,
    right: str | None = None,
) -> dict[str, object]:
    spec: dict[str, object] = {
        "frame": {"kind": "duration", "micros": _SCE08_DURATION_MICROS},
        "kind": kind,
        "min_periods": 1,
        "output": output,
        "primitive_version": 1,
    }
    if input_name is not None:
        spec["input"] = input_name
    if right is not None:
        spec.update({"ddof": 1, "left": "close", "right": right})
    return spec


def _sce08_hand_built_project_json() -> str:
    input_schema = _sce08_input_schema()
    output_schema = _sce08_output_schema()
    selected = [f'"{field["name"]}"' for field in output_schema]
    project = {
        "data_sources": [
            {"data": [], "format": "inline_json", "id": "source_1", "input": "input"}
        ],
        "format_version": 3,
        "graph": {
            "edges": [
                {
                    "source_node": "features__cf_rolling",
                    "source_port": "output",
                    "target_node": "features",
                    "target_port": "input",
                }
            ],
            "name": "sce08-temporal-pair",
            "nodes": [
                {
                    "id": "features__cf_rolling",
                    "input_ports": [
                        {
                            "kind": "table",
                            "name": "input",
                            "required": True,
                            "schema": input_schema,
                        }
                    ],
                    "operator": {
                        "kind": "rolling",
                        "spec": {
                            "allowed_lateness_micros": 0,
                            "configuration_version": 1,
                            "event_time": "event_time",
                            "late_policy": {"kind": "error", "scope": "envelope"},
                            "outputs": [
                                _sce08_output_spec(
                                    "min", "duration_min_close", input_name="close"
                                ),
                                _sce08_output_spec(
                                    "max", "duration_max_close", input_name="close"
                                ),
                                _sce08_output_spec(
                                    "covariance",
                                    "duration_cov_close_volume",
                                    right="volume",
                                ),
                                _sce08_output_spec(
                                    "correlation",
                                    "duration_corr_close_volume",
                                    right="volume",
                                ),
                            ],
                            "partition_by": ["symbol"],
                            "sequence_by": ["sequence"],
                            "state_layout_version": 1,
                            "value_policy": "stateful_numeric_v1",
                        },
                    },
                    "output_ports": [
                        {
                            "kind": "table",
                            "name": "output",
                            "required": True,
                            "schema": output_schema,
                        }
                    ],
                },
                {
                    "id": "features",
                    "input_ports": [
                        {
                            "kind": "table",
                            "name": "input",
                            "required": True,
                            "schema": output_schema,
                        }
                    ],
                    "operator": {
                        "expression": "",
                        "filter": None,
                        "kind": "expression",
                        "select": selected,
                        "udfs": [],
                    },
                },
            ],
        },
        "id": "sce08-temporal-pair",
        "name": "sce08-temporal-pair",
        "runtime": {"mode": "batch", "options": {}},
    }
    return json.dumps(project, sort_keys=True, separators=(",", ":"))


def _sce08_symbolic_program() -> Program:
    quotes = table_input(
        "quotes",
        schema=[
            Field("event_time", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("industry", "string", nullable=False),
            Field("close", "float64"),
            Field("volume", "float64"),
        ],
        entity_by=["symbol"],
        event_time="event_time",
        sequence_by=["sequence"],
    )
    frame = duration(_SCE08_DURATION_MICROS)
    enriched = quotes.with_columns(
        FeatureSet(
            [
                ("duration_min_close", ts.min(quotes["close"], window=frame)),
                ("duration_max_close", ts.max(quotes["close"], window=frame)),
                (
                    "duration_cov_close_volume",
                    ts.covariance(
                        quotes["close"], quotes["volume"], window=frame, ddof=1
                    ),
                ),
                (
                    "duration_corr_close_volume",
                    ts.correlation(
                        quotes["close"], quotes["volume"], window=frame, ddof=1
                    ),
                ),
            ]
        )
    )
    return Program(
        "sce08-temporal-pair",
        inputs=[quotes],
        outputs=[("features", enriched)],
    )


def _sorted_temporal_output(value: pa.Table) -> pa.Table:
    return value.sort_by(
        [
            ("event_time", "ascending"),
            ("symbol", "ascending"),
            ("sequence", "ascending"),
        ]
    )


def _assert_sce08_outputs_equal(hand_built: pa.Table, symbolic: pa.Table) -> None:
    baseline = _sorted_temporal_output(hand_built)
    candidate = _sorted_temporal_output(symbolic)
    exact_columns = (
        "event_time",
        "sequence",
        "symbol",
        "industry",
        "close",
        "volume",
        "duration_min_close",
        "duration_max_close",
    )
    assert baseline.select(exact_columns).equals(candidate.select(exact_columns))
    for name in ("duration_cov_close_volume", "duration_corr_close_volume"):
        np.testing.assert_allclose(
            baseline[name].to_numpy(zero_copy_only=False),
            candidate[name].to_numpy(zero_copy_only=False),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )


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
    group=benchmark_group("sce05-row-local-pair"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_sce05_row_local_milestone_pair(
    benchmark: BenchmarkFixture, _scale: str
) -> None:
    workload = quote_workload()
    hand_built_plan = (
        PipelineBuilder("sce05-row-local-hand-built")
        .sql("features", _ROW_LOCAL_PAIR_QUERY)
        .compile_batch()
    )
    symbolic_plan = _projection_symbolic_program().compile_batch(Runtime())

    hand_built_output = (
        hand_built_plan.execute({"input": workload.batch})
        .outputs["output"]
        .to_pyarrow()
    )
    symbolic_output = (
        symbolic_plan.execute({"input": workload.batch}).outputs["output"].to_pyarrow()
    )
    assert hand_built_output.equals(symbolic_output)
    symbolic_warm_result = symbolic_plan.execute({"input": workload.batch})
    assert len(symbolic_warm_result.datafusion_metrics) == 1
    inputs = {"input": workload.batch}
    paired_samples = _alternating_plan_samples(
        hand_built_plan,
        symbolic_plan,
        inputs,
        sample_count=_ROW_LOCAL_PAIRED_SAMPLES,
    )

    record_symbolic_benchmark(
        benchmark,
        scenario=_ROW_LOCAL_SCENARIO,
        input_rows=workload.rows,
        output_rows=symbolic_output.num_rows,
        metrics=symbolic_warm_result.datafusion_metrics,
        extra={
            "comparison_contract": "same-process-alternating-v1",
            "workload_contract": _ROW_LOCAL_WORKLOAD_CONTRACT,
            "derived_columns": PROJECTION_COLUMN_COUNT,
            "paired_samples": paired_samples,
        },
    )

    def execute_pair() -> tuple[Any, Any]:
        return hand_built_plan.execute(inputs), symbolic_plan.execute(inputs)

    hand_built_result, symbolic_result = benchmark(execute_pair)
    assert hand_built_result.outputs["output"].num_rows == workload.rows
    output = symbolic_result.outputs["output"].to_pyarrow()
    assert output.column_names == [
        "sequence",
        *(f"f{index:02d}" for index in range(1, 21)),
    ]


@pytest.mark.benchmark(
    group=benchmark_group("sce08-temporal-pair"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_sce08_temporal_milestone_pair(
    benchmark: BenchmarkFixture, _scale: str
) -> None:
    workload = quote_workload()
    input_table = workload.batch.to_pyarrow()
    event_time_index = input_table.schema.get_field_index("event_time")
    input_table = input_table.set_column(
        event_time_index,
        pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
        input_table["event_time"].cast(pa.timestamp("us", tz="UTC")),
    )
    hand_built_plan = PipelineBuilder._from_json(
        _sce08_hand_built_project_json()
    ).compile_batch()
    symbolic_plan = _sce08_symbolic_program().compile_batch(Runtime())
    assert hand_built_plan.fingerprint == symbolic_plan.fingerprint
    inputs = {"input": Batch.from_pyarrow(input_table)}

    hand_built_output = hand_built_plan.execute(inputs).outputs["output"].to_pyarrow()
    symbolic_warm_result = symbolic_plan.execute(inputs)
    symbolic_output = symbolic_warm_result.outputs["output"].to_pyarrow()
    _assert_sce08_outputs_equal(hand_built_output, symbolic_output)
    assert len(symbolic_warm_result.datafusion_metrics) == 1
    paired_samples = _alternating_plan_samples(
        hand_built_plan,
        symbolic_plan,
        inputs,
        sample_count=_SCE08_PAIRED_SAMPLES,
    )

    record_symbolic_benchmark(
        benchmark,
        scenario=_SCE08_SCENARIO,
        input_rows=workload.rows,
        output_rows=symbolic_output.num_rows,
        metrics=symbolic_warm_result.datafusion_metrics,
        extra={
            "comparison_contract": "same-process-alternating-v1",
            "workload_contract": _SCE08_WORKLOAD_CONTRACT,
            "duration_micros": _SCE08_DURATION_MICROS,
            "entities": workload.entities,
            "maximum_retained_rows_per_entity": 60,
            "temporal_outputs": 4,
            "paired_samples": paired_samples,
        },
    )

    def execute_pair() -> tuple[Any, Any]:
        return hand_built_plan.execute(inputs), symbolic_plan.execute(inputs)

    hand_built_result, symbolic_result = benchmark(execute_pair)
    assert hand_built_result.outputs["output"].num_rows == workload.rows
    assert symbolic_result.outputs["output"].num_rows == workload.rows


@pytest.mark.benchmark(
    group=benchmark_group("sce14-cross-domain-pair"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_sce14_cross_domain_sharing_pair(
    benchmark: BenchmarkFixture, _scale: str
) -> None:
    workload = quote_workload()
    first_program, second_program, optimized_program = _sce14_programs()
    runtime = Runtime()
    reference = _ReferencePlans(
        first_program.compile_batch(runtime),
        second_program.compile_batch(runtime),
    )
    optimized = optimized_program.compile_batch(runtime)
    inputs = {"input": _utc_quote_batch()}

    reference_first, reference_second = reference.execute(inputs)
    optimized_warm = optimized.execute(inputs)
    assert (
        reference_first.outputs["output"]
        .to_pyarrow()
        .equals(optimized_warm.outputs["first.output"].to_pyarrow())
    )
    assert (
        reference_second.outputs["output"]
        .to_pyarrow()
        .equals(optimized_warm.outputs["second.output"].to_pyarrow())
    )
    paired_samples = _alternating_plan_samples(
        reference,
        optimized,
        inputs,
        sample_count=_SCE14_PAIRED_SAMPLES,
    )

    record_symbolic_benchmark(
        benchmark,
        scenario=_SCE14_SCENARIO,
        input_rows=workload.rows,
        output_rows=workload.rows * 2,
        metrics=optimized_warm.datafusion_metrics,
        extra={
            "comparison_contract": "same-process-alternating-v1",
            "workload_contract": _SCE14_WORKLOAD_CONTRACT,
            "entities": workload.entities,
            "industries": workload.industries,
            "reference_state_stages": 4,
            "optimized_state_stages": 2,
            "paired_samples": paired_samples,
        },
    )

    def execute_pair() -> tuple[Any, Any]:
        return reference.execute(inputs), optimized.execute(inputs)

    reference_result, optimized_result = benchmark(execute_pair)
    assert len(reference_result) == 2
    assert sum(batch.num_rows for batch in optimized_result.outputs.values()) == (
        workload.rows * 2
    )


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

        # Bound the recovery source at the pause point so it idles at the
        # restored cursor instead of replaying: graceful shutdown then drains
        # only the checkpointed window state, fixing replayed batches at zero.
        recovery_source = _BaselineSource(batches, pause_at)
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

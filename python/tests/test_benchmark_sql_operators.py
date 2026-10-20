from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from benchmarks.engine_comparison import (
    EngineCase,
    Workload,
    _calc_flow,
    _datafusion,
    sql_query,
    workload,
)
from calc_flow import Batch, PipelineBuilder
from scripts.benchmark_suite.catalog import (
    ROLLING_CASES,
    ROW_SCALES,
    SQL_CASES,
    STREAM_CASES,
    engine_cases,
)

SQL_GAPS = (
    "average",
    "argmax64",
    "argmax256",
    "unique64",
    "cs_mean",
    "window_sum",
    "asof_join",
)


def test_sql_catalog_covers_every_stream_scenario_at_original_scales():
    assert SQL_CASES == STREAM_CASES
    assert ROLLING_CASES == ("sma20", "dual_sma")
    for rows in ROW_SCALES:
        for backend in ("calc-flow-sql", "datafusion"):
            registered = {
                case["scenario"]
                for case in engine_cases(rows)
                if case["backend"] == backend
            }
            assert set(SQL_GAPS) <= registered


@pytest.mark.parametrize("backend", ("calc-flow-sql", "datafusion"))
@pytest.mark.parametrize("scenario", SQL_GAPS)
@pytest.mark.parametrize("rows", (10, 17_001))
def test_missing_sql_scenarios_execute_against_independent_oracles(
    tmp_path, backend, scenario, rows
):
    case = {
        "backend": backend,
        "scenario": scenario,
        "rows": rows,
    }
    runner = EngineCase(case, tmp_path)
    original = runner.data.table.combine_chunks()
    try:
        result = runner.sample()
        assert result["correctness"]["passed"]
        assert runner.data.table.combine_chunks().equals(original)
    finally:
        runner.close()


@pytest.mark.parametrize("scenario", SQL_GAPS)
def test_sql_scenarios_match_native_streams_across_small_batches(tmp_path, scenario):
    case = next(
        case
        for case in engine_cases(17_001)
        if case["backend"] == "calc-flow-stream" and case["scenario"] == scenario
    )
    runner = EngineCase(
        {
            **case,
            "batch_rows": 640,
            "checkpoint_interval_millis": None,
        },
        tmp_path,
    )
    try:
        streamed = runner._stream()[0]
        runner.validate(streamed)
        sql = _calc_flow(runner.data, scenario)()
        key = (
            [("window_start", "ascending"), ("symbol", "ascending")]
            if scenario == "window_sum"
            else "sequence"
        )
        expected, actual = streamed.sort_by(key), sql.sort_by(key)
        assert actual.column_names == expected.column_names
        for name in expected.column_names:
            if name == "value":
                np.testing.assert_allclose(
                    actual[name].to_numpy(), expected[name].to_numpy(), rtol=1e-10
                )
            else:
                assert actual[name].equals(expected[name])
    finally:
        runner.close()


@pytest.mark.parametrize("factory", (_calc_flow, _datafusion))
@pytest.mark.parametrize("window", (64, 256))
def test_sql_argmax_uses_oldest_tied_maximum_and_row_offsets(factory, window):
    original = workload(600)
    # Interleaved entities and repeated maxima distinguish row offsets from
    # global sequence distances, and oldest ties from most recent ties.
    entities = np.arange(600) % 2
    ticks = np.arange(600) // 2
    prices = np.where(ticks % 31 == 0, 10.0, (ticks % 7).astype(float))
    table = original.table.set_column(2, "symbol", pa.array(entities.astype(str)))
    table = table.set_column(3, "price", pa.array(prices))
    data = Workload(table, original.dimension, 2)
    result = factory(data, f"argmax{window}")().sort_by("sequence")
    expected = np.empty(600)
    for entity in range(2):
        series = prices[entity::2]
        for index in range(len(series)):
            frame = series[max(0, index - window + 1) : index + 1]
            expected[entity + index * 2] = len(frame) - 1 - np.argmax(frame)
    np.testing.assert_array_equal(result["value"].to_numpy(), expected)


@pytest.mark.parametrize("factory", (_calc_flow, _datafusion))
def test_sql_unique_counts_repeated_values_as_the_window_expires(factory):
    original = workload(150)
    prices = np.asarray([1.0] * 65 + [2.0] * 65 + [3.0] * 20)
    table = original.table.set_column(2, "symbol", pa.array(["S000"] * 150))
    table = table.set_column(3, "price", pa.array(prices))
    data = Workload(table, original.dimension, 1)
    result = factory(data, "unique64")().sort_by("sequence")
    expected = [len(set(prices[max(0, i - 63) : i + 1])) for i in range(150)]
    np.testing.assert_array_equal(result["value"].to_numpy(), expected)


@pytest.mark.parametrize("factory", (_calc_flow, _datafusion))
def test_sql_tumbling_sum_uses_epoch_anchored_half_open_windows(factory):
    original = workload(6)
    table = pa.table(
        {
            "event_time": [
                -1_000_000,
                0,
                9_000_000,
                10_000_000,
                19_000_000,
                20_000_000,
            ],
            "sequence": range(6),
            "symbol": ["S000"] * 6,
            "price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
        schema=original.table.schema,
    )
    result = factory(Workload(table, original.dimension, 1), "window_sum")()
    result = result.sort_by("window_start")
    assert result["window_start"].type == pa.timestamp("us", tz="UTC")
    assert result["window_start"].cast(pa.int64()).to_pylist() == [
        -10_000_000,
        0,
        10_000_000,
        20_000_000,
    ]
    assert result["value"].to_pylist() == [1.0, 5.0, 9.0, 6.0]


@pytest.mark.parametrize("backend", ("calc-flow-sql", "datafusion"))
def test_sql_zero_tolerance_asof_preserves_unmatched_rows_and_latest_ties(backend):
    schema = workload(5).table.schema
    left = pa.table(
        {
            "event_time": [0, 1, 2, 3, 4],
            "sequence": range(5),
            "symbol": ["a", "a", "a", "a", "b"],
            "price": [1.0] * 5,
        },
        schema=schema,
    )
    right = pa.table(
        {
            "event_time": [0, 0, 2, 3, 4],
            "sequence": [5, 8, 9, 10, 11],
            "symbol": ["a", "a", "a", "b", "a"],
            "price": [2.0, 3.0, 5.0, 7.0, 11.0],
        },
        schema=schema,
    )
    query = sql_query("asof_join")
    if backend == "calc-flow-sql":
        plan = (
            PipelineBuilder("sql-asof-boundaries")
            .sql("query", query, aliases=("input", "reference"))
            .compile_batch()
        )
        result = (
            plan.execute(
                {
                    "input": Batch.from_pyarrow(left),
                    "reference": Batch.from_pyarrow(right),
                }
            )
            .outputs["output"]
            .to_pyarrow()
        )
    else:
        from datafusion import SessionContext

        context = SessionContext()
        for name, table in (("input", left), ("reference", right)):
            context.register_record_batches(name, [table.to_batches()])
        result = context.sql(query).to_arrow_table()
    result = result.sort_by("sequence")
    assert result["sequence"].to_pylist() == list(range(5))
    assert result["value"].to_pylist() == [3.0, None, 5.0, None, None]

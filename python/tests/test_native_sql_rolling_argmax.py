from __future__ import annotations

import pyarrow as pa

import calc_flow as cf


def test_native_sql_rewrites_rolling_argmax_with_sql_null_and_tie_semantics():
    table = pa.table(
        {
            "sequence": pa.array(range(12), type=pa.int64()),
            "event_time": pa.array(range(12), type=pa.timestamp("us", tz="UTC")),
            "symbol": ["a", "b"] * 6,
            "price": pa.array(
                [3.0, 4.0, 5.0, 4.0, 5.0, None, 2.0, 1.0, 1.0, 8.0, None, 1.0]
            ),
        }
    )
    query = (
        "SELECT sequence, CAST(COUNT(*) OVER trailing - "
        "CAST(array_position(array_agg(price) OVER trailing, "
        "MAX(price) OVER trailing) AS BIGINT) AS DOUBLE) AS value FROM input "
        "WINDOW trailing AS (PARTITION BY symbol ORDER BY event_time, sequence "
        "ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)"
    )
    plan = (
        cf.PipelineBuilder("sql-rolling-argmax")
        .with_datafusion_config(target_partitions=4)
        .sql("query", query, aliases=("input",))
        .compile_batch()
    )

    result = plan.execute({"input": cf.Batch.from_pyarrow(table)})
    output = result.outputs["output"].to_pyarrow().sort_by("sequence")

    assert output["value"].to_pylist() == [
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        2.0,
        2.0,
        2.0,
        2.0,
        0.0,
        2.0,
        1.0,
    ]
    metric = result.datafusion_metrics[0]
    assert metric["rolling_rewritten_windows"] == 1
    assert "array_agg" not in metric["physical_plan"]


def test_native_sql_rolling_argmax_matches_fallback_for_special_floats():
    table = pa.table(
        {
            "sequence": pa.array(range(9), type=pa.int64()),
            "event_time": pa.array(range(9), type=pa.timestamp("us", tz="UTC")),
            "symbol": ["a"] * 9,
            "price": pa.array(
                [
                    None,
                    None,
                    float("nan"),
                    float("nan"),
                    0.0,
                    -0.0,
                    float("inf"),
                    float("-inf"),
                    3.0,
                ]
            ),
        }
    )
    query = (
        "SELECT sequence, CAST(COUNT(*) OVER trailing - "
        "CAST(array_position(array_agg(price) OVER trailing, "
        "MAX(price) OVER trailing) AS BIGINT) AS DOUBLE) AS value FROM input "
        "WINDOW trailing AS (PARTITION BY symbol ORDER BY event_time, sequence "
        "ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)"
    )

    def execute(rewrite: bool):
        plan = (
            cf.PipelineBuilder(f"sql-rolling-argmax-{rewrite}")
            .with_datafusion_config(target_partitions=4, enable_rolling_rewrite=rewrite)
            .sql("query", query, aliases=("input",))
            .compile_batch()
        )
        result = plan.execute({"input": cf.Batch.from_pyarrow(table)})
        values = result.outputs["output"].to_pyarrow().sort_by("sequence")["value"]
        return result, values.to_pylist()

    reference, expected = execute(False)
    optimized, actual = execute(True)

    def normalize(value):
        return "nan" if value is not None and value != value else value

    assert [normalize(value) for value in actual] == [
        normalize(value) for value in expected
    ]
    assert (
        reference.outputs["output"].to_pyarrow().schema
        == optimized.outputs["output"].to_pyarrow().schema
    )
    assert reference.datafusion_metrics[0]["rolling_rewritten_windows"] == 0
    assert optimized.datafusion_metrics[0]["rolling_rewritten_windows"] == 1


def test_native_sql_rolling_argmax_matches_fallback_for_signed_zero_order():
    query = (
        "SELECT sequence, CAST(COUNT(*) OVER trailing - "
        "CAST(array_position(array_agg(price) OVER trailing, "
        "MAX(price) OVER trailing) AS BIGINT) AS DOUBLE) AS value FROM input "
        "WINDOW trailing AS (PARTITION BY symbol ORDER BY event_time, sequence "
        "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)"
    )
    for prices, expected in [
        ([-0.0, 0.0, -1.0], [0.0, 0.0, 1.0]),
        ([0.0, -0.0, -1.0], [0.0, 1.0, 1.0]),
    ]:
        table = pa.table(
            {
                "sequence": pa.array(range(3), type=pa.int64()),
                "event_time": pa.array(range(3), type=pa.timestamp("us", tz="UTC")),
                "symbol": ["a"] * 3,
                "price": pa.array(prices),
            }
        )
        outputs = []
        for rewrite in (False, True):
            plan = (
                cf.PipelineBuilder(f"sql-rolling-argmax-zero-{rewrite}")
                .with_datafusion_config(
                    target_partitions=1, enable_rolling_rewrite=rewrite
                )
                .sql("query", query, aliases=("input",))
                .compile_batch()
            )
            result = plan.execute({"input": cf.Batch.from_pyarrow(table)})
            outputs.append(result.outputs["output"].to_pyarrow().sort_by("sequence"))
        assert outputs[0]["value"].to_pylist() == expected
        assert outputs[1]["value"].to_pylist() == expected


def test_native_sql_rolling_argmax_retains_non_float_fallback():
    table = pa.table(
        {
            "sequence": pa.array(range(4), type=pa.int64()),
            "event_time": pa.array(range(4), type=pa.timestamp("us", tz="UTC")),
            "symbol": ["a"] * 4,
            "price": pa.array([1, 3, 2, 2], type=pa.int64()),
        }
    )
    query = (
        "SELECT sequence, CAST(COUNT(*) OVER trailing - "
        "CAST(array_position(array_agg(price) OVER trailing, "
        "MAX(price) OVER trailing) AS BIGINT) AS DOUBLE) AS value FROM input "
        "WINDOW trailing AS (PARTITION BY symbol ORDER BY event_time, sequence "
        "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)"
    )
    plan = (
        cf.PipelineBuilder("sql-rolling-argmax-integer")
        .sql("query", query, aliases=("input",))
        .compile_batch()
    )

    result = plan.execute({"input": cf.Batch.from_pyarrow(table)})

    assert result.outputs["output"].to_pyarrow()["value"].to_pylist() == [
        0.0,
        0.0,
        1.0,
        1.0,
    ]
    assert result.datafusion_metrics[0]["rolling_rewritten_windows"] == 0

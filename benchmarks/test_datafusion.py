from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from benchmarks.support import (
    BenchmarkFixture,
    benchmark_group,
    record_benchmark,
    selected_scale,
    table_inputs,
)
from calc_flow import (
    Batch,
    ExpressionOperator,
    Pipeline,
    SqlOperator,
    UdfReference,
    UdfRegistry,
)
from calc_flow.engine.datafusion import DataFusionConfig, DataFusionRuntime


def _input() -> Batch:
    return table_inputs(selected_scale().table_rows).fact


@pytest.mark.benchmark(
    group=benchmark_group("datafusion-expression"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_projection_and_calculated_column(
    benchmark: BenchmarkFixture, _scale: str
) -> None:
    batch = _input()
    plan = (
        Pipeline("benchmark-projection")
        .then(
            ExpressionOperator(
                "calculate",
                select=("id", "amount * quantity AS gross"),
            )
        )
        .compile()
    )

    result = benchmark(plan.execute, {"input": batch})

    assert result.output.num_rows == batch.num_rows
    record_benchmark(
        benchmark,
        scenario="datafusion_projection",
        input_rows=batch.num_rows,
        output_rows=result.output.num_rows,
        metrics=result.datafusion_metrics,
    )


@pytest.mark.benchmark(
    group=benchmark_group("datafusion-filter"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_filter_selectivity(benchmark: BenchmarkFixture, _scale: str) -> None:
    batch = _input()
    plan = (
        Pipeline("benchmark-filter")
        .then(
            ExpressionOperator(
                "filter",
                select=("id", "amount"),
                filter_expression="selected",
            )
        )
        .compile()
    )

    result = benchmark(plan.execute, {"input": batch})

    assert 0 < result.output.num_rows < batch.num_rows
    record_benchmark(
        benchmark,
        scenario="datafusion_filter_35_percent",
        input_rows=batch.num_rows,
        output_rows=result.output.num_rows,
        metrics=result.datafusion_metrics,
    )


@pytest.mark.benchmark(
    group=benchmark_group("datafusion-aggregate"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_group_by_aggregation(benchmark: BenchmarkFixture, _scale: str) -> None:
    batch = _input()
    plan = (
        Pipeline("benchmark-group-by")
        .then(
            SqlOperator(
                "aggregate",
                "SELECT group_id, SUM(amount) AS total FROM fact GROUP BY group_id",
                inputs=("fact",),
            )
        )
        .compile()
    )

    result = benchmark(plan.execute, {"fact": batch})

    assert result.output.num_rows > 0
    record_benchmark(
        benchmark,
        scenario="datafusion_group_by",
        input_rows=batch.num_rows,
        output_rows=result.output.num_rows,
        metrics=result.datafusion_metrics,
    )


@pytest.mark.benchmark(
    group=benchmark_group("datafusion-join"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_join_cardinality(benchmark: BenchmarkFixture, _scale: str) -> None:
    inputs = table_inputs(selected_scale().table_rows)
    plan = (
        Pipeline("benchmark-join")
        .then(
            SqlOperator(
                "join",
                "SELECT f.id, f.amount * d.multiplier AS adjusted "
                "FROM fact f JOIN dimension d USING (group_id)",
                inputs=("fact", "dimension"),
            )
        )
        .compile()
    )

    result = benchmark(
        plan.execute,
        {"fact": inputs.fact, "dimension": inputs.dimension},
    )

    assert result.output.num_rows == inputs.fact.num_rows
    record_benchmark(
        benchmark,
        scenario="datafusion_inner_join",
        input_rows=inputs.fact.num_rows + inputs.dimension.num_rows,
        output_rows=result.output.num_rows,
        metrics=result.datafusion_metrics,
    )


@pytest.mark.benchmark(
    group=benchmark_group("datafusion-window"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_window_function(benchmark: BenchmarkFixture, _scale: str) -> None:
    batch = _input()
    plan = (
        Pipeline("benchmark-window")
        .then(
            SqlOperator(
                "window",
                "SELECT id, ROW_NUMBER() OVER ("
                "PARTITION BY group_id ORDER BY amount) AS position FROM fact",
                inputs=("fact",),
            )
        )
        .compile()
    )

    result = benchmark(plan.execute, {"fact": batch})

    assert result.output.num_rows == batch.num_rows
    record_benchmark(
        benchmark,
        scenario="datafusion_window",
        input_rows=batch.num_rows,
        output_rows=result.output.num_rows,
        metrics=result.datafusion_metrics,
    )


@pytest.mark.parametrize("implementation", ("builtin", "registered_udf"))
@pytest.mark.benchmark(
    group=benchmark_group("datafusion-udf"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_builtin_versus_registered_udf(
    benchmark: BenchmarkFixture, implementation: str, _scale: str
) -> None:
    batch = _input()
    registry = UdfRegistry()

    @registry.datafusion_scalar(
        name="double_amount",
        version="1",
        input_fields=(pa.int64(),),
        return_field=pa.int64(),
    )
    def double_amount(values: pa.Array) -> pa.Array:
        return pc.multiply(values, 2)

    expression = "doubled = amount * 2"
    references: tuple[UdfReference, ...] = ()
    if implementation == "registered_udf":
        expression = "doubled = double_amount(amount)"
        references = (UdfReference("double_amount", "1"),)
    plan = (
        Pipeline("benchmark-udf", udf_registry=registry)
        .then(ExpressionOperator("calculate", expression, udfs=references))
        .compile()
    )

    result = benchmark(plan.execute, {"input": batch})

    assert result.output.num_rows == batch.num_rows
    record_benchmark(
        benchmark,
        scenario=f"datafusion_{implementation}",
        input_rows=batch.num_rows,
        output_rows=result.output.num_rows,
        metrics=result.datafusion_metrics,
    )


@pytest.mark.parametrize(
    ("batch_size", "target_partitions"),
    ((1024, 1), (8192, 2)),
)
@pytest.mark.benchmark(
    group=benchmark_group("datafusion-config"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_execution_configuration(
    benchmark: BenchmarkFixture,
    batch_size: int,
    target_partitions: int,
    _scale: str,
) -> None:
    batch = _input()
    plan = (
        Pipeline(
            "benchmark-config",
            datafusion_config=DataFusionConfig(
                batch_size=batch_size,
                target_partitions=target_partitions,
            ),
        )
        .then(ExpressionOperator("calculate", "gross = amount * quantity"))
        .compile()
    )

    result = benchmark(plan.execute, {"input": batch})

    assert result.output.num_rows == batch.num_rows
    record_benchmark(
        benchmark,
        scenario=(f"datafusion_batch_{batch_size}_partitions_{target_partitions}"),
        input_rows=batch.num_rows,
        output_rows=result.output.num_rows,
        metrics=result.datafusion_metrics,
    )


@pytest.mark.benchmark(
    group=benchmark_group("datafusion-session"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_warm_session_context(benchmark: BenchmarkFixture, _scale: str) -> None:
    batch = _input()
    runtime = DataFusionRuntime()
    try:
        runtime.sql("SELECT amount + 1 AS value FROM fact", {"fact": batch})
        result = benchmark(
            runtime.sql,
            "SELECT amount + 1 AS value FROM fact",
            {"fact": batch},
        )
    finally:
        runtime.close()

    assert result.num_rows == batch.num_rows
    record_benchmark(
        benchmark,
        scenario="datafusion_warm_session",
        input_rows=batch.num_rows,
        output_rows=result.num_rows,
        metrics=runtime.metrics[-1:],
    )

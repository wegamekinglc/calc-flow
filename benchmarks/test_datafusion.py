from __future__ import annotations

import json
from collections.abc import Mapping

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
from calc_flow import Batch, ExecutionPlan, PipelineBuilder, RunResult, Runtime


def _input() -> Batch:
    return table_inputs(selected_scale().table_rows).fact


def _output(result: RunResult, name: str = "output") -> Batch:
    return result.outputs[name]


def _compile_with_datafusion(
    builder: PipelineBuilder, *, batch_size: int, target_partitions: int
) -> ExecutionPlan:
    project = builder.project
    configured = {
        **project,
        "pipeline": {
            **project["pipeline"],
            "datafusion": {
                "batch_size": batch_size,
                "target_partitions": target_partitions,
            },
        },
    }
    return Runtime().compile_project(
        json.dumps(configured, separators=(",", ":"), sort_keys=True)
    )


def _benchmark_plan(
    benchmark: BenchmarkFixture,
    plan: ExecutionPlan,
    inputs: Mapping[str, Batch],
    *,
    scenario: str,
    input_rows: int,
) -> RunResult:
    warm_result = plan.execute(inputs)
    record_benchmark(
        benchmark,
        scenario=scenario,
        input_rows=input_rows,
        output_rows=_output(warm_result).num_rows,
        metrics=warm_result.datafusion_metrics,
    )
    return benchmark(plan.execute, inputs)


@pytest.mark.benchmark(
    group=benchmark_group("datafusion-expression"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_projection_and_calculated_column(
    benchmark: BenchmarkFixture, _scale: str
) -> None:
    batch = _input()
    plan = (
        PipelineBuilder("benchmark-projection")
        .expression(
            "calculate",
            "",
            select=("id", "amount * quantity AS gross"),
        )
        .compile()
    )

    result = _benchmark_plan(
        benchmark,
        plan,
        {"input": batch},
        scenario="datafusion_projection",
        input_rows=batch.num_rows,
    )
    output = _output(result)

    assert output.num_rows == batch.num_rows


@pytest.mark.benchmark(
    group=benchmark_group("datafusion-filter"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_filter_selectivity(benchmark: BenchmarkFixture, _scale: str) -> None:
    batch = _input()
    plan = (
        PipelineBuilder("benchmark-filter")
        .expression(
            "filter",
            "",
            select=("id", "amount"),
            filter="selected",
        )
        .compile()
    )

    result = _benchmark_plan(
        benchmark,
        plan,
        {"input": batch},
        scenario="datafusion_filter_35_percent",
        input_rows=batch.num_rows,
    )
    output = _output(result)

    assert 0 < output.num_rows < batch.num_rows


@pytest.mark.benchmark(
    group=benchmark_group("datafusion-aggregate"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_group_by_aggregation(benchmark: BenchmarkFixture, _scale: str) -> None:
    batch = _input()
    plan = (
        PipelineBuilder("benchmark-group-by")
        .sql(
            "aggregate",
            "SELECT group_id, SUM(amount) AS total FROM fact GROUP BY group_id",
            aliases=("fact",),
        )
        .compile()
    )

    result = _benchmark_plan(
        benchmark,
        plan,
        {"fact": batch},
        scenario="datafusion_group_by",
        input_rows=batch.num_rows,
    )
    output = _output(result)

    assert output.num_rows > 0


@pytest.mark.benchmark(
    group=benchmark_group("datafusion-join"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_join_cardinality(benchmark: BenchmarkFixture, _scale: str) -> None:
    inputs = table_inputs(selected_scale().table_rows)
    plan = (
        PipelineBuilder("benchmark-join")
        .sql(
            "join",
            "SELECT f.id, f.amount * d.multiplier AS adjusted "
            "FROM fact f JOIN dimension d USING (group_id)",
            aliases=("fact", "dimension"),
        )
        .compile()
    )

    result = _benchmark_plan(
        benchmark,
        plan,
        {"fact": inputs.fact, "dimension": inputs.dimension},
        scenario="datafusion_inner_join",
        input_rows=inputs.fact.num_rows + inputs.dimension.num_rows,
    )
    output = _output(result)

    assert output.num_rows == inputs.fact.num_rows


@pytest.mark.benchmark(
    group=benchmark_group("datafusion-window"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_window_function(benchmark: BenchmarkFixture, _scale: str) -> None:
    batch = _input()
    plan = (
        PipelineBuilder("benchmark-window")
        .sql(
            "window",
            "SELECT id, ROW_NUMBER() OVER ("
            "PARTITION BY group_id ORDER BY amount) AS position FROM fact",
            aliases=("fact",),
        )
        .compile()
    )

    result = _benchmark_plan(
        benchmark,
        plan,
        {"fact": batch},
        scenario="datafusion_window",
        input_rows=batch.num_rows,
    )
    output = _output(result)

    assert output.num_rows == batch.num_rows


@pytest.mark.parametrize("implementation", ("builtin", "registered_udf"))
@pytest.mark.benchmark(
    group=benchmark_group("datafusion-udf"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_builtin_versus_registered_udf(
    benchmark: BenchmarkFixture, implementation: str, _scale: str
) -> None:
    batch = _input()
    runtime = Runtime()
    expression = "doubled = amount * 2"
    references: tuple[tuple[str, str, str], ...] = ()
    if implementation == "registered_udf":

        def double_amount(values: pa.Array) -> pa.Array:
            return pc.multiply(values, 2)

        runtime.register_scalar_udf(
            provider="python",
            name="double_amount",
            version="1",
            input_types=("int64",),
            return_type="int64",
            volatility="immutable",
            function=double_amount,
        )
        expression = "doubled = double_amount(amount)"
        references = (("python", "double_amount", "1"),)
    plan = (
        PipelineBuilder("benchmark-udf")
        .expression("calculate", expression, udfs=references)
        .compile(runtime)
    )

    result = _benchmark_plan(
        benchmark,
        plan,
        {"input": batch},
        scenario=f"datafusion_{implementation}",
        input_rows=batch.num_rows,
    )
    output = _output(result)

    assert output.num_rows == batch.num_rows


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
    plan = _compile_with_datafusion(
        PipelineBuilder("benchmark-config").expression(
            "calculate", "gross = amount * quantity"
        ),
        batch_size=batch_size,
        target_partitions=target_partitions,
    )

    result = _benchmark_plan(
        benchmark,
        plan,
        {"input": batch},
        scenario=f"datafusion_batch_{batch_size}_partitions_{target_partitions}",
        input_rows=batch.num_rows,
    )
    output = _output(result)

    assert output.num_rows == batch.num_rows


@pytest.mark.benchmark(
    group=benchmark_group("datafusion-session"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_repeated_compiled_plan_execution(
    benchmark: BenchmarkFixture, _scale: str
) -> None:
    batch = _input()
    plan = (
        PipelineBuilder("benchmark-repeated-plan")
        .expression("calculate", "value = amount + 1")
        .compile()
    )
    result = _benchmark_plan(
        benchmark,
        plan,
        {"input": batch},
        scenario="datafusion_repeated_compiled_plan",
        input_rows=batch.num_rows,
    )
    output = _output(result)

    assert output.num_rows == batch.num_rows

"""Same-input native and SQL declarations through the compiled batch engine."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyarrow as pa
import pytest

from benchmarks.engine_comparison import Workload, expected_output, sql_query, workload
from benchmarks.support import (
    BenchmarkFixture,
    benchmark_group,
    record_comparable_identity,
    selected_scale,
)
from calc_flow import Batch, Program, TableExpr, cs, exact_time, table_input, ts


def _native_expression(source: TableExpr, scenario: str) -> TableExpr:
    price = source["price"]
    operations = {
        "projection": lambda: price * 2.0 + 1.0,
        "average": lambda: ts.average(price),
        "cs_mean": lambda: cs.mean(price, group=exact_time(source["event_time"])),
    }
    return source.select("sequence", value=operations[scenario]())


def prepare_batch_case(
    data: Workload, scenario: str, declaration: str
) -> Callable[[], pa.Table]:
    source = table_input(
        "events",
        schema=data.table.schema,
        entity_by=("symbol",),
        event_time="event_time",
        sequence_by=("sequence",),
    )
    output = (
        _native_expression(source, scenario)
        if declaration == "native"
        else source.sql(sql_query(scenario))
    )
    plan = Program("batch-expression-parity", mode="batch").output("value", output)
    compiled = plan.compile_batch()
    inputs = {compiled.external_inputs[0]: Batch.from_pyarrow(data.table)}
    output_name = compiled.external_outputs[0]

    def execute() -> pa.Table:
        return compiled.execute(inputs).outputs[output_name].to_pyarrow()

    return execute


def validate_batch_output(actual: pa.Table, expected: pa.Table) -> None:
    assert actual.column_names == expected.column_names
    ordered = actual.sort_by("sequence")
    assert ordered["sequence"].equals(expected["sequence"])
    assert ordered["value"].type == expected["value"].type
    np.testing.assert_allclose(
        ordered["value"].to_numpy(),
        expected["value"].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.benchmark(
    group=benchmark_group("batch-expression-parity"), min_rounds=3, max_time=1.0
)
@pytest.mark.parametrize("scenario", ("projection", "average", "cs_mean"))
@pytest.mark.parametrize("declaration", ("native", "sql"))
def test_batch_expression_parity(
    benchmark: BenchmarkFixture, scenario: str, declaration: str
) -> None:
    data = workload(selected_scale().table_rows)
    execute = prepare_batch_case(data, scenario, declaration)
    expected = expected_output(data, scenario)
    validate_batch_output(execute(), expected)
    workload_identity = {
        "scenario": scenario,
        "scope": "compiled-batch-execute-to-arrow",
        "input_rows": data.table.num_rows,
        "output_rows": expected.num_rows,
        "entities": data.entities,
        "fixture": "engine-comparison-exact-eighths-v1",
    }
    benchmark.extra_info = {
        **benchmark.extra_info,
        **workload_identity,
        "backend": f"calc-flow-batch-{declaration}",
    }
    record_comparable_identity(
        benchmark,
        workload_identity=workload_identity,
        dependency_packages=("numpy", "pyarrow", "pytest", "pytest-benchmark"),
    )
    validate_batch_output(benchmark(execute), expected)

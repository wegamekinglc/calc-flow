from __future__ import annotations

import pytest

from benchmarks.array_support import (
    ACCEPTANCE_WORKLOADS,
    TRANSPOSE_DIAGNOSTIC,
    ArrayBackend,
    ArrayWorkload,
    batch_for,
    benchmark_calculation,
    dtype_for,
    output_shape_for,
    plan_for,
    runtime_for,
    shape_for,
    values_for,
)
from benchmarks.support import BenchmarkFixture, benchmark_group, selected_scale
from calc_flow import Batch

PLAN_WORKLOADS = (*ACCEPTANCE_WORKLOADS, TRANSPOSE_DIAGNOSTIC)


@pytest.mark.parametrize("backend", ("numpy", "jax"))
@pytest.mark.parametrize(
    "workload",
    PLAN_WORKLOADS,
    ids=(
        "array_elementwise",
        "array_mean",
        "array_matrix_multiplication",
        "array_transpose_reshape_diagnostic",
    ),
)
@pytest.mark.benchmark(
    group=benchmark_group("array-plan-end-to-end"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_plan_end_to_end(
    benchmark: BenchmarkFixture,
    backend: ArrayBackend,
    workload: ArrayWorkload,
    _scale: str,
) -> None:
    scale = selected_scale()
    runtime = runtime_for(backend)
    values = batch_for(values_for(workload, backend, scale), backend)
    plan = plan_for(runtime, backend, workload)

    result = benchmark_calculation(
        benchmark,
        backend=backend,
        scope="plan_end_to_end",
        scenario=workload.scenario,
        expression=workload.expression,
        input_value=values,
        calculate=lambda: plan.execute({"input": values}).outputs["output"],
    )

    assert isinstance(result, Batch)
    assert shape_for(result) == output_shape_for(workload, scale)
    assert dtype_for(result) == dtype_for(values)

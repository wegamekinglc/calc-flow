from __future__ import annotations

import pytest

from benchmarks.array_support import (
    ARRAY_WORKLOADS,
    ArrayBackend,
    ArrayWorkload,
    benchmark_calculation,
    dtype_for,
    namespace_for,
    output_shape_for,
    shape_for,
    values_for,
)
from benchmarks.support import BenchmarkFixture, benchmark_group, selected_scale


@pytest.mark.parametrize("backend", ("numpy", "jax"))
@pytest.mark.parametrize(
    "workload",
    ARRAY_WORKLOADS,
    ids=lambda workload: workload.scenario,
)
@pytest.mark.benchmark(
    group=benchmark_group("array-backend-kernel"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_backend_kernel(
    benchmark: BenchmarkFixture,
    backend: ArrayBackend,
    workload: ArrayWorkload,
    _scale: str,
) -> None:
    scale = selected_scale()
    namespace = namespace_for(backend)
    values = values_for(workload, backend, scale)
    input_dtype = dtype_for(values)

    result = benchmark_calculation(
        benchmark,
        backend=backend,
        scope="backend_kernel",
        scenario=workload.scenario,
        expression=workload.expression,
        input_value=values,
        calculate=lambda: workload.operation(values, namespace),
    )

    assert shape_for(result) == output_shape_for(workload, scale)
    assert dtype_for(result) == input_dtype

from __future__ import annotations

import pytest

from benchmarks.array_support import (
    ARRAY_WORKLOADS,
    ArrayBackend,
    ArrayWorkload,
    batch_for,
    benchmark_calculation,
    dtype_for,
    namespace_for,
    output_shape_for,
    shape_for,
    values_for,
)
from benchmarks.support import BenchmarkFixture, benchmark_group, selected_scale
from calc_flow import Batch
from calc_flow.array import _ArrayProvider


@pytest.mark.parametrize("backend", ("numpy", "jax"))
@pytest.mark.parametrize(
    "workload",
    ARRAY_WORKLOADS,
    ids=lambda workload: workload.scenario,
)
@pytest.mark.benchmark(
    group=benchmark_group("array-provider-boundary"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_provider_boundary(
    benchmark: BenchmarkFixture,
    backend: ArrayBackend,
    workload: ArrayWorkload,
    _scale: str,
) -> None:
    scale = selected_scale()
    values = values_for(workload, backend, scale)
    input_batch = batch_for(values, backend)
    provider = _ArrayProvider(backend, namespace_for(backend))

    result = benchmark_calculation(
        benchmark,
        backend=backend,
        scope="provider_boundary",
        scenario=workload.scenario,
        expression=workload.expression,
        input_value=input_batch,
        calculate=lambda: provider(
            input_batch,
            {"expression": workload.expression},
        ),
    )

    assert isinstance(result, Batch)
    assert shape_for(result) == output_shape_for(workload, scale)
    assert dtype_for(result) == dtype_for(input_batch)

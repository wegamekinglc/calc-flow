from __future__ import annotations

import numpy as np
import pytest

from benchmarks.array_support import (
    ARRAY_WORKLOADS,
    ArrayBackend,
    benchmark_calculation,
    dtype_for,
    shape_for,
    values_for,
)
from benchmarks.support import BenchmarkFixture, benchmark_group, selected_scale
from calc_flow import Batch


@pytest.mark.parametrize("backend", ("numpy", "jax"))
@pytest.mark.benchmark(
    group=benchmark_group("array-batch-ownership"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_batch_ownership(
    benchmark: BenchmarkFixture,
    backend: ArrayBackend,
    _scale: str,
) -> None:
    scale = selected_scale()
    source = values_for(ARRAY_WORKLOADS[0], backend, scale)
    expected = np.asarray(source).copy()
    input_dtype = dtype_for(source)

    result = benchmark_calculation(
        benchmark,
        backend=backend,
        scope="batch_ownership",
        scenario="array_batch_ownership",
        expression="Batch.from_array",
        input_value=source,
        calculate=lambda: Batch.from_array(source, backend=backend),
    )

    assert isinstance(result, Batch)
    assert shape_for(result) == (scale.array_elements,)
    assert dtype_for(result) == input_dtype
    if backend == "numpy":
        source[0] = -1
        assert np.array_equal(result.array, expected)
        current: object = result.array
        while isinstance(current, np.ndarray):
            array = current
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array.setflags(write=True)
            current = array.base

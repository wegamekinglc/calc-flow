from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from benchmarks.support import (
    BenchmarkFixture,
    benchmark_group,
    record_benchmark,
    selected_scale,
)
from calc_flow import (
    Batch,
    PipelineBuilder,
    Runtime,
    register_jax,
    register_numpy,
)


def _runtime(backend: str) -> Runtime:
    runtime = Runtime()
    if backend == "numpy":
        register_numpy(runtime)
    else:
        pytest.importorskip("jax")
        register_jax(runtime)
    return runtime


def _batch(value: object, backend: str) -> Batch:
    return Batch.from_array(value, backend=backend)


def _plan(runtime: Runtime, backend: str, expression: str):
    return (
        PipelineBuilder(f"benchmark-{backend}")
        .external(
            "calculate",
            backend,
            "expression",
            "1",
            {"expression": expression},
        )
        .compile(runtime)
    )


def _synchronize(value: Any) -> None:
    block_until_ready = getattr(value, "block_until_ready", None)
    if callable(block_until_ready):
        block_until_ready()


def _benchmark_array(
    benchmark: BenchmarkFixture,
    *,
    backend: str,
    scenario: str,
    input_rows: int,
    calculate: Callable[[], Batch],
) -> Batch:
    warm_result = calculate()
    _synchronize(warm_result.array)
    record_benchmark(
        benchmark,
        scenario=scenario,
        input_rows=input_rows,
        output_rows=warm_result.num_rows,
        backend=backend,
    )

    result = benchmark(calculate)

    return result


@pytest.mark.parametrize("backend", ("numpy", "jax"))
@pytest.mark.benchmark(
    group=benchmark_group("array-elementwise"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_elementwise_expression(
    benchmark: BenchmarkFixture, backend: str, _scale: str
) -> None:
    scale = selected_scale()
    runtime = _runtime(backend)
    values = _batch(np.arange(scale.array_elements, dtype=np.float64), backend)
    plan = _plan(runtime, backend, "(x * x + 1) ** 0.5")

    def calculate() -> Batch:
        result = plan.execute({"input": values}).outputs["output"]
        _synchronize(result.array)
        return result

    result = _benchmark_array(
        benchmark,
        backend=backend,
        scenario="array_elementwise",
        input_rows=scale.array_elements,
        calculate=calculate,
    )

    assert result.num_rows == scale.array_elements


@pytest.mark.parametrize("backend", ("numpy", "jax"))
@pytest.mark.benchmark(
    group=benchmark_group("array-reduction"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_reduction(benchmark: BenchmarkFixture, backend: str, _scale: str) -> None:
    scale = selected_scale()
    runtime = _runtime(backend)
    values = _batch(np.arange(scale.array_elements, dtype=np.float64), backend)
    plan = _plan(runtime, backend, "mean(x)")

    def calculate() -> Batch:
        result = plan.execute({"input": values}).outputs["output"]
        _synchronize(result.array)
        return result

    result = _benchmark_array(
        benchmark,
        backend=backend,
        scenario="array_mean",
        input_rows=scale.array_elements,
        calculate=calculate,
    )

    assert result.num_rows == 1


@pytest.mark.parametrize("backend", ("numpy", "jax"))
@pytest.mark.benchmark(
    group=benchmark_group("array-matmul"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_matrix_multiplication(
    benchmark: BenchmarkFixture, backend: str, _scale: str
) -> None:
    dimension = selected_scale().matrix_dimension
    runtime = _runtime(backend)
    matrix = _batch(
        np.arange(dimension**2, dtype=np.float64).reshape(dimension, dimension),
        backend,
    )
    plan = _plan(runtime, backend, "x @ x")

    def calculate() -> Batch:
        result = plan.execute({"input": matrix}).outputs["output"]
        _synchronize(result.array)
        return result

    result = _benchmark_array(
        benchmark,
        backend=backend,
        scenario="array_matrix_multiplication",
        input_rows=dimension,
        calculate=calculate,
    )

    assert result.array.shape == (dimension, dimension)


@pytest.mark.parametrize("backend", ("numpy", "jax"))
@pytest.mark.benchmark(
    group=benchmark_group("array-reshape"), min_rounds=3, max_time=0.5
)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_transpose_and_reshape(
    benchmark: BenchmarkFixture, backend: str, _scale: str
) -> None:
    dimension = selected_scale().matrix_dimension
    runtime = _runtime(backend)
    matrix = _batch(
        np.arange(dimension**2, dtype=np.float64).reshape(dimension, dimension),
        backend,
    )
    plan = _plan(
        runtime,
        backend,
        f"reshape(transpose(x), ({dimension**2},))",
    )

    def calculate() -> Batch:
        result = plan.execute({"input": matrix}).outputs["output"]
        _synchronize(result.array)
        return result

    result = _benchmark_array(
        benchmark,
        backend=backend,
        scenario="array_transpose_reshape",
        input_rows=dimension,
        calculate=calculate,
    )

    assert result.num_rows == dimension**2

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from benchmarks.support import BenchmarkFixture, record_benchmark, selected_scale
from calc_flow import Batch, UdfReference, UdfRegistry
from calc_flow.engine import JaxEngine, NumpyEngine
from calc_flow.engine.array import ArrayEngine


def _engine(backend: str) -> ArrayEngine:
    if backend == "numpy":
        return NumpyEngine()
    pytest.importorskip("jax")
    return JaxEngine()


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
    calculate: Any,
) -> Batch:
    warm_result = calculate()
    _synchronize(warm_result.array_payload)

    result = benchmark(calculate)

    record_benchmark(
        benchmark,
        scenario=scenario,
        input_rows=input_rows,
        output_rows=result.num_rows,
        backend=backend,
    )
    return result


@pytest.mark.parametrize("backend", ("numpy", "jax"))
@pytest.mark.benchmark(group="array-elementwise", min_rounds=3, max_time=0.5)
def test_elementwise_expression(benchmark: BenchmarkFixture, backend: str) -> None:
    scale = selected_scale()
    engine = _engine(backend)
    values = Batch.array(engine.xp.asarray(np.arange(scale.array_elements)))

    def calculate() -> Batch:
        result = engine.evaluate("xp.sqrt(x * x + 1)", values)
        _synchronize(result.array_payload)
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
@pytest.mark.benchmark(group="array-reduction", min_rounds=3, max_time=0.5)
def test_reduction(benchmark: BenchmarkFixture, backend: str) -> None:
    scale = selected_scale()
    engine = _engine(backend)
    values = Batch.array(engine.xp.asarray(np.arange(scale.array_elements)))

    def calculate() -> Batch:
        result = engine.mean(values)
        _synchronize(result.array_payload)
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
@pytest.mark.benchmark(group="array-matmul", min_rounds=3, max_time=0.5)
def test_matrix_multiplication(benchmark: BenchmarkFixture, backend: str) -> None:
    dimension = selected_scale().matrix_dimension
    engine = _engine(backend)
    matrix = Batch.array(
        engine.xp.asarray(np.arange(dimension**2).reshape(dimension, dimension))
    )

    def calculate() -> Batch:
        result = engine.matmul(matrix, matrix)
        _synchronize(result.array_payload)
        return result

    result = _benchmark_array(
        benchmark,
        backend=backend,
        scenario="array_matrix_multiplication",
        input_rows=dimension,
        calculate=calculate,
    )

    assert result.array_payload.shape == (dimension, dimension)


@pytest.mark.parametrize("backend", ("numpy", "jax"))
@pytest.mark.benchmark(group="array-udf", min_rounds=3, max_time=0.5)
def test_registered_array_udf(benchmark: BenchmarkFixture, backend: str) -> None:
    scale = selected_scale()
    registry = UdfRegistry()

    @registry.array(name="square", version="1", argument_count=1)
    def square(values: Any) -> Any:
        return values * values

    reference = UdfReference("square", "1")
    if backend == "numpy":
        engine: ArrayEngine = NumpyEngine(
            udf_registry=registry,
            udfs=(reference,),
        )
    else:
        pytest.importorskip("jax")
        engine = JaxEngine(udf_registry=registry, udfs=(reference,))
    values = Batch.array(engine.xp.asarray(np.arange(scale.array_elements)))

    def calculate() -> Batch:
        result = engine.evaluate("square(x)", values)
        _synchronize(result.array_payload)
        return result

    result = _benchmark_array(
        benchmark,
        backend=backend,
        scenario="array_registered_udf",
        input_rows=scale.array_elements,
        calculate=calculate,
    )

    assert result.num_rows == scale.array_elements

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from benchmarks.support import (
    ArrayBenchmarkRecord,
    ArrayBenchmarkScope,
    BenchmarkFixture,
    BenchmarkScale,
    record_array_benchmark,
)
from calc_flow import (
    Batch,
    ExecutionPlan,
    PipelineBuilder,
    Runtime,
    register_jax,
    register_numpy,
)

ArrayBackend = Literal["numpy", "jax"]


@dataclass(frozen=True, slots=True)
class ArrayWorkload:
    scenario: str
    expression: str
    operation: Callable[[object, object], object]
    input_factory: Callable[[BenchmarkScale], np.ndarray[Any, Any]]


def _vector_input(scale: BenchmarkScale) -> np.ndarray[Any, Any]:
    return np.arange(scale.array_elements, dtype=np.float64)


def _matrix_input(scale: BenchmarkScale) -> np.ndarray[Any, Any]:
    dimension = scale.matrix_dimension
    return np.arange(dimension**2, dtype=np.float64).reshape(dimension, dimension)


def _elementwise(value: object, namespace: object) -> object:
    return cast(Any, namespace).sqrt(value * value + 1)


def _mean(value: object, namespace: object) -> object:
    return cast(Any, namespace).mean(value)


def _matrix_multiplication(value: object, _namespace: object) -> object:
    return operator.matmul(value, value)


def _transpose_and_reshape(value: object, namespace: object) -> object:
    typed_namespace = cast(Any, namespace)
    transpose = typed_namespace.transpose(value)
    return typed_namespace.reshape(transpose, (-1,))


ARRAY_WORKLOADS = (
    ArrayWorkload(
        scenario="array_elementwise",
        expression="(x * x + 1) ** 0.5",
        operation=_elementwise,
        input_factory=_vector_input,
    ),
    ArrayWorkload(
        scenario="array_mean",
        expression="mean(x)",
        operation=_mean,
        input_factory=_vector_input,
    ),
    ArrayWorkload(
        scenario="array_matrix_multiplication",
        expression="x @ x",
        operation=_matrix_multiplication,
        input_factory=_matrix_input,
    ),
    ArrayWorkload(
        scenario="array_transpose_reshape",
        expression="reshape(transpose(x), (-1,))",
        operation=_transpose_and_reshape,
        input_factory=_matrix_input,
    ),
)

ACCEPTANCE_WORKLOADS = ARRAY_WORKLOADS[:3]
TRANSPOSE_DIAGNOSTIC = ARRAY_WORKLOADS[3]


def namespace_for(backend: ArrayBackend) -> object:
    if backend == "numpy":
        return np
    if backend == "jax":
        import jax.numpy as jnp

        return jnp
    raise ValueError(f"unsupported array benchmark backend: {backend!r}")


def values_for(
    workload: ArrayWorkload,
    backend: ArrayBackend,
    scale: BenchmarkScale,
) -> object:
    source = workload.input_factory(scale)
    if backend == "numpy":
        return source
    return cast(Any, namespace_for(backend)).asarray(source)


def runtime_for(backend: ArrayBackend) -> Runtime:
    runtime = Runtime()
    if backend == "numpy":
        register_numpy(runtime)
    elif backend == "jax":
        register_jax(runtime)
    else:
        raise ValueError(f"unsupported array benchmark backend: {backend!r}")
    return runtime


def batch_for(value: object, backend: ArrayBackend) -> Batch:
    return Batch.from_array(value, backend=backend)


def plan_for(
    runtime: Runtime,
    backend: ArrayBackend,
    workload: ArrayWorkload,
) -> ExecutionPlan:
    return (
        PipelineBuilder(f"benchmark-{backend}-{workload.scenario}")
        .external(
            "calculate",
            backend,
            "expression",
            "1",
            {"expression": workload.expression},
        )
        .compile_batch(runtime)
    )


def array_value(value: object) -> object:
    return value.array if isinstance(value, Batch) else value


def synchronize(value: object) -> None:
    block_until_ready = getattr(array_value(value), "block_until_ready", None)
    if callable(block_until_ready):
        block_until_ready()


def dtype_for(value: object) -> str:
    dtype = getattr(array_value(value), "dtype", None)
    if dtype is None:
        raise TypeError("array benchmark values must expose a dtype")
    return str(dtype)


def shape_for(value: object) -> tuple[int, ...]:
    shape = getattr(array_value(value), "shape", None)
    if shape is None:
        raise TypeError("array benchmark values must expose a shape")
    return tuple(int(dimension) for dimension in shape)


def output_shape_for(
    workload: ArrayWorkload,
    scale: BenchmarkScale,
) -> tuple[int, ...]:
    if workload.scenario == "array_elementwise":
        return (scale.array_elements,)
    if workload.scenario == "array_mean":
        return ()
    if workload.scenario == "array_matrix_multiplication":
        return (scale.matrix_dimension, scale.matrix_dimension)
    if workload.scenario == "array_transpose_reshape":
        return (scale.matrix_dimension**2,)
    raise ValueError(f"unsupported array benchmark workload: {workload.scenario!r}")


def rows_for(value: object) -> int:
    if isinstance(value, Batch):
        return value.num_rows
    shape = shape_for(value)
    return shape[0] if shape else 1


def benchmark_calculation(
    benchmark: BenchmarkFixture,
    *,
    backend: ArrayBackend,
    scope: ArrayBenchmarkScope,
    scenario: str,
    expression: str,
    input_value: object,
    calculate: Callable[[], object],
) -> object:
    def synchronized_calculation() -> object:
        result = calculate()
        synchronize(result)
        return result

    warm_result = synchronized_calculation()
    record_array_benchmark(
        benchmark,
        ArrayBenchmarkRecord(
            scenario=scenario,
            scope=scope,
            backend=backend,
            expression=expression,
            input_dtype=dtype_for(input_value),
            output_dtype=dtype_for(warm_result),
            input_rows=rows_for(input_value),
            output_rows=rows_for(warm_result),
        ),
    )
    return benchmark(synchronized_calculation)

"""Correctness and identity boundaries for the public native Array benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow as pa

import calc_flow.array as cfa
from benchmarks.support import (
    BenchmarkFixture,
    record_benchmark,
    record_comparable_identity,
    selected_scale,
)
from calc_flow import _native


class NativeBenchmarkFixture(BenchmarkFixture, Protocol):
    def pedantic(
        self, function: Callable, *, rounds: int, iterations: int, warmup_rounds: int
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class NativeCase:
    scenario: str
    scope: str
    calculate: Callable[[], object]
    expected: tuple[np.ndarray, ...] | pa.Table
    inputs: tuple[np.ndarray, ...]
    parameters: tuple[tuple[str, object], ...] = ()

    def validate(self, result: object) -> None:
        if isinstance(self.expected, pa.Table):
            _validate_table(result, self.expected)
            return
        components = _components(result)
        if len(components) != len(self.expected):
            raise AssertionError("native benchmark returned the wrong component count")
        for actual, expected in zip(components, self.expected):
            _validate_array(actual, expected)


def _components(result: object) -> tuple[cfa.NativeArray, ...]:
    if isinstance(result, cfa.NativeArray):
        return (result,)
    if isinstance(result, cfa.MacdOutputs):
        return result.line, result.signal, result.histogram
    raise AssertionError("native benchmark did not return immutable native arrays")


def _validate_array(actual: cfa.NativeArray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape or actual.dtype != str(expected.dtype):
        raise AssertionError("native benchmark returned the wrong shape or dtype")
    _validate_values(actual.to_arrow(), expected.reshape(-1))


def _validate_table(actual: object, expected: pa.Table) -> None:
    if not isinstance(actual, pa.Table) or actual.column_names != expected.column_names:
        raise AssertionError("native benchmark returned the wrong table columns")
    if actual.num_rows != expected.num_rows:
        raise AssertionError("native benchmark returned the wrong table length")
    if actual.schema.types != expected.schema.types:
        raise AssertionError("native benchmark returned the wrong table dtypes")
    actual = actual.sort_by("seq") if "seq" in actual.column_names else actual
    for name in expected.column_names:
        _validate_values(actual[name], expected[name].to_numpy())


def _validate_values(actual: pa.Array | pa.ChunkedArray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(
        actual.is_null().to_numpy(zero_copy_only=False), np.isnan(expected)
    )
    observed = actual.to_numpy(zero_copy_only=False)
    if np.issubdtype(expected.dtype, np.integer):
        np.testing.assert_array_equal(observed, expected)
    else:
        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def _input_digest(values: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        descriptor = json.dumps([str(value.dtype), list(value.shape)]).encode()
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _native_sha256() -> str:
    return hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest()


def _output_shapes(case: NativeCase) -> list[list[int]]:
    if isinstance(case.expected, pa.Table):
        return [[case.expected.num_rows, case.expected.num_columns]]
    return [list(value.shape) for value in case.expected]


def _record(benchmark: NativeBenchmarkFixture, case: NativeCase) -> None:
    scale = selected_scale()
    shapes = _output_shapes(case)
    input_rows = case.inputs[0].shape[0]
    output_rows = shapes[0][0] if shapes[0] else 1
    record_benchmark(
        benchmark,
        scenario=case.scenario,
        input_rows=input_rows,
        output_rows=output_rows,
        backend="native",
    )
    benchmark.extra_info = {
        **benchmark.extra_info,
        "scope": case.scope,
        "native_binary_sha256": _native_sha256(),
    }
    workload = {
        "contract": "native-array-benchmarks-v1",
        "scenario": case.scenario,
        "scope": case.scope,
        "backend": "native",
        "scale": scale.name,
        "table_rows": scale.table_rows,
        "array_elements": scale.array_elements,
        "matrix_dimension": scale.matrix_dimension,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "input_shapes": [list(value.shape) for value in case.inputs],
        "input_dtypes": [str(value.dtype) for value in case.inputs],
        "input_sha256": _input_digest(case.inputs),
        "output_shapes": shapes,
        "parameters": dict(case.parameters),
        "sampling": {"rounds": 10, "iterations": 1, "warmup_rounds": 1},
        "thread_configuration": {
            name: os.environ.get(name, "default")
            for name in (
                "TOKIO_WORKER_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
    }
    record_comparable_identity(
        benchmark, workload_identity=workload, dependency_packages=("numpy", "pyarrow")
    )


def measure_native(benchmark: NativeBenchmarkFixture, case: NativeCase) -> object:
    """Validate first, record identities, then measure ten fresh calculations."""
    case.validate(case.calculate())
    _record(benchmark, case)
    result = benchmark.pedantic(
        case.calculate, rounds=10, iterations=1, warmup_rounds=1
    )
    case.validate(result)
    return result

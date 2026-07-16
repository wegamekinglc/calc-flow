from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
from typing import Any, Literal, Protocol

import numpy as np
import psutil
import pyarrow as pa
from cpuinfo import get_cpu_info

from calc_flow import Batch

SEED = 20_260_711
BENCHMARK_CONTRACT_VERSION = 2
ARRAY_WORKLOAD_VERSION = 1

ArrayBenchmarkScope = Literal[
    "backend_kernel",
    "provider_boundary",
    "plan_end_to_end",
    "batch_ownership",
]
_ArrayBenchmarkBackend = Literal["numpy", "jax"]


class BenchmarkFixture(Protocol):
    extra_info: dict[str, Any]

    def __call__(self, function: Any, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class BenchmarkScale:
    name: str
    table_rows: int
    array_elements: int
    matrix_dimension: int


@dataclass(frozen=True, slots=True)
class ArrayBenchmarkRecord:
    scenario: str
    scope: ArrayBenchmarkScope
    backend: Literal["numpy", "jax"]
    expression: str
    input_dtype: str
    output_dtype: str
    input_rows: int
    output_rows: int


SCALES = {
    "overhead": BenchmarkScale("overhead", 1_000, 1_000, 16),
    "small": BenchmarkScale("small", 10_000, 10_000, 64),
    "standard": BenchmarkScale("standard", 100_000, 100_000, 256),
    "nightly": BenchmarkScale("nightly", 1_000_000, 1_000_000, 512),
}


@dataclass(frozen=True, slots=True)
class TableInputs:
    fact: Batch
    dimension: Batch


@lru_cache(maxsize=1)
def selected_scale() -> BenchmarkScale:
    name = os.environ.get("CALC_FLOW_BENCHMARK_SCALE", "overhead")
    try:
        return SCALES[name]
    except KeyError as error:
        choices = ", ".join(SCALES)
        raise ValueError(
            f"unknown CALC_FLOW_BENCHMARK_SCALE {name!r}; choose {choices}"
        ) from error


def scale_spec(scale: BenchmarkScale | None = None) -> str:
    """One-line description of the active problem scale.

    Embedded in benchmark group headers so each results section names the
    data size its timings were measured at.
    """
    scale = scale or selected_scale()
    return (
        f"{scale.name} rows={scale.table_rows} "
        f"array={scale.array_elements} matmul={scale.matrix_dimension}"
    )


def benchmark_group(base: str) -> str:
    """Tag a benchmark group with the active problem scale."""
    return f"{base} [{scale_spec()}]"


@lru_cache(maxsize=4)
def table_inputs(rows: int) -> TableInputs:
    rng = np.random.default_rng(SEED)
    group_count = max(1, min(10_000, int(rows**0.5)))
    group_ids = rng.integers(0, group_count, size=rows, dtype=np.int64)
    fact = Batch.from_pyarrow(
        pa.table(
            {
                "id": np.arange(rows, dtype=np.int64),
                "group_id": group_ids,
                "amount": rng.integers(1, 10_000, size=rows, dtype=np.int64),
                "quantity": rng.integers(1, 20, size=rows, dtype=np.int64),
                "selected": rng.random(rows) < 0.35,
            }
        )
    )
    dimension = Batch.from_pyarrow(
        pa.table(
            {
                "group_id": np.arange(group_count, dtype=np.int64),
                "multiplier": rng.integers(1, 5, size=group_count, dtype=np.int64),
            }
        )
    )
    return TableInputs(fact=fact, dimension=dimension)


def record_benchmark(
    benchmark: BenchmarkFixture,
    *,
    scenario: str,
    input_rows: int,
    output_rows: int,
    metrics: Sequence[Mapping[str, object]] = (),
    backend: str | None = None,
) -> None:
    scale = selected_scale()
    backend_info = {"backend": backend} if backend is not None else {}
    metric_info = (
        {
            "datafusion_planning_ns": sum(
                _metric_value(metric, "planning_ns") for metric in metrics
            ),
            "datafusion_execution_ns": sum(
                _metric_value(metric, "execution_ns") for metric in metrics
            ),
            "datafusion_query_count": len(metrics),
        }
        if metrics
        else {}
    )
    # pytest-benchmark snapshots this attribute when benchmark(...) starts.
    # Call this helper before that boundary, and replace the mapping so no
    # caller-owned dictionary is changed in place.
    benchmark.extra_info = {
        **benchmark.extra_info,
        "scenario": scenario,
        "scale": scale.name,
        "table_rows": scale.table_rows,
        "array_elements": scale.array_elements,
        "matrix_dimension": scale.matrix_dimension,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "process_rss_bytes": psutil.Process().memory_info().rss,
        **backend_info,
        **metric_info,
    }


def record_array_benchmark(
    benchmark: BenchmarkFixture,
    record: ArrayBenchmarkRecord,
) -> None:
    _validate_array_record(record)
    record_benchmark(
        benchmark,
        scenario=record.scenario,
        input_rows=record.input_rows,
        output_rows=record.output_rows,
        backend=record.backend,
    )
    scale = selected_scale()
    machine_identity = _machine_identity()
    dependency_identity = _dependency_identity(record.backend)
    backend_configuration = _backend_configuration(record.backend)
    workload_identity = {
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "scenario": record.scenario,
        "scope": record.scope,
        "workload_version": ARRAY_WORKLOAD_VERSION,
        "backend": record.backend,
        "scale": scale.name,
        "table_rows": scale.table_rows,
        "array_elements": scale.array_elements,
        "matrix_dimension": scale.matrix_dimension,
        "input_rows": record.input_rows,
        "output_rows": record.output_rows,
        "expression": record.expression,
        "input_dtype": record.input_dtype,
        "output_dtype": record.output_dtype,
        "backend_configuration": backend_configuration,
    }
    benchmark.extra_info = {
        **benchmark.extra_info,
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "workload_version": ARRAY_WORKLOAD_VERSION,
        "scope": record.scope,
        "expression": record.expression,
        "input_dtype": record.input_dtype,
        "output_dtype": record.output_dtype,
        "machine_identity": machine_identity,
        "dependency_identity": dependency_identity,
        "backend_configuration": backend_configuration,
        "workload_identity": workload_identity,
        "machine_fingerprint": _canonical_fingerprint(machine_identity),
        "dependency_fingerprint": _canonical_fingerprint(dependency_identity),
        "workload_fingerprint": _canonical_fingerprint(workload_identity),
        **dependency_identity,
        **backend_configuration,
    }


def _validate_array_record(record: ArrayBenchmarkRecord) -> None:
    if record.scope not in (
        "backend_kernel",
        "provider_boundary",
        "plan_end_to_end",
        "batch_ownership",
    ):
        raise ValueError(f"unsupported array benchmark scope: {record.scope!r}")
    if record.backend not in ("numpy", "jax"):
        raise ValueError(f"unsupported array benchmark backend: {record.backend!r}")
    if not record.scenario:
        raise ValueError("array benchmark scenario must not be empty")
    if not record.expression:
        raise ValueError("array benchmark expression must not be empty")


def _canonical_fingerprint(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalize_cpu_brand(value: str) -> str:
    return " ".join(value.casefold().split())


def _machine_identity() -> dict[str, object]:
    raw = get_cpu_info()
    brand = str(raw.get("brand_raw") or platform.processor() or platform.machine())
    logical_cpu_count = os.cpu_count()
    if logical_cpu_count is None:
        raise RuntimeError("logical CPU count is unavailable")
    return {
        "operating_system": platform.system().casefold(),
        "architecture": platform.machine().casefold(),
        "cpu_brand": _normalize_cpu_brand(brand),
        "logical_cpu_count": logical_cpu_count,
        "python_implementation": platform.python_implementation().casefold(),
    }


def _dependency_identity(backend: _ArrayBenchmarkBackend) -> dict[str, object]:
    identity = {
        "python_version": platform.python_version(),
        "numpy_version": version("numpy"),
    }
    if backend == "jax":
        identity.update(
            {
                "jax_version": version("jax"),
                "jaxlib_version": version("jaxlib"),
            }
        )
    return identity


def _backend_configuration(backend: _ArrayBenchmarkBackend) -> dict[str, object]:
    if backend == "numpy":
        return {}

    import jax

    return {
        "jax_platform": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def _metric_value(metric: Mapping[str, object], name: str) -> int:
    value = metric.get(name, 0)
    return value if type(value) is int and value >= 0 else 0

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import numpy as np
import psutil
import pyarrow as pa

from calc_flow import Batch
from calc_flow.engine.datafusion import DataFusionQueryMetrics

SEED = 20_260_711


class BenchmarkFixture(Protocol):
    extra_info: dict[str, Any]

    def __call__(self, function: Any, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class BenchmarkScale:
    name: str
    table_rows: int
    array_elements: int
    matrix_dimension: int


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


@lru_cache(maxsize=4)
def table_inputs(rows: int) -> TableInputs:
    rng = np.random.default_rng(SEED)
    group_count = max(1, min(10_000, int(rows**0.5)))
    group_ids = rng.integers(0, group_count, size=rows, dtype=np.int64)
    fact = Batch.table(
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
    dimension = Batch.table(
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
    metrics: tuple[DataFusionQueryMetrics, ...] = (),
    backend: str | None = None,
) -> None:
    scale = selected_scale()
    benchmark.extra_info.update(
        {
            "scenario": scenario,
            "scale": scale.name,
            "input_rows": input_rows,
            "output_rows": output_rows,
            "process_rss_bytes": psutil.Process().memory_info().rss,
        }
    )
    if backend is not None:
        benchmark.extra_info["backend"] = backend
    if metrics:
        benchmark.extra_info.update(
            {
                "datafusion_planning_ns": sum(metric.planning_ns for metric in metrics),
                "datafusion_execution_ns": sum(
                    metric.execution_ns for metric in metrics
                ),
                "datafusion_query_count": len(metrics),
            }
        )

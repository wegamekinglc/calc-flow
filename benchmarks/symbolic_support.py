"""Shared workloads and metric helpers for the symbolic execution baselines.

The scenarios in ``test_symbolic_baseline.py`` measure hand-built calc-flow
plans that compute what the future symbolic layer will compile: row-local
projections, rolling per-entity features, complete-group cross sections,
provider-owned matrix products, and stateful stream checkpoints. Every input
is deterministic (fixed seed, fixed interleave, complete groups) so a paired
re-run on the same process, input order, and machine is comparable.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pyarrow as pa

from benchmarks.support import (
    BenchmarkFixture,
    record_benchmark,
    record_comparable_identity,
    selected_scale,
)
from calc_flow import Batch, Runtime

SYMBOLIC_SEED = 20_260_822
PROJECTION_COLUMN_COUNT = 20
ROLLING_SHORT_WINDOW = 20
ROLLING_LONG_WINDOW = 60
STREAM_ENTITIES = 40
STREAM_WINDOW_SECONDS = 60
STREAM_ROW_MICROS = 1_000_000
STREAM_BATCH_ROWS = 2_500
STREAM_MAX_ROWS = 50_000
# Keeps the dense feature matrix (rows x 20 columns) under the owned-NumPy
# 10,000,000-element conversion cap in crates/calc-flow-python/src/batch.rs.
MATMUL_MAX_ROWS = 400_000

_QUOTE_SCHEMA = pa.schema(
    (
        pa.field("event_time", pa.timestamp("us"), nullable=False),
        pa.field("sequence", pa.uint64(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("industry", pa.string(), nullable=False),
        pa.field("close", pa.float64(), nullable=True),
        pa.field("volume", pa.float64(), nullable=True),
    )
)

_STREAM_SCHEMA = pa.schema(
    (
        pa.field("event_time", pa.timestamp("us"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("value", pa.float64(), nullable=True),
    )
)


@dataclass(frozen=True, slots=True)
class QuoteWorkload:
    """Deterministic interleaved entity quotes with complete time buckets."""

    batch: Batch
    entities: int
    industries: int
    rows: int

    @property
    def rows_per_entity(self) -> int:
        return self.rows // self.entities


def quote_entities(rows: int) -> int:
    """Entity count for a table size; every bucket stays a complete group."""
    return max(4, min(64, int(rows**0.5)))


def quote_workload(rows: int | None = None) -> QuoteWorkload:
    """Build the interleaved entity quote table for one benchmark scale.

    Rows are emitted in global round-robin entity order with one quote per
    entity per second, so each one-second bucket holds exactly one row per
    entity and every ``(bucket, industry)`` cross section is complete.
    """
    scale = selected_scale()
    total = scale.table_rows if rows is None else rows
    entities = quote_entities(total)
    industries = max(2, entities // 8)
    # Every industry must own the same number of entities so each
    # (bucket, industry) cross section is exactly entities/industries wide.
    entities = entities // industries * industries
    total = total // entities * entities
    rng = np.random.default_rng(SYMBOLIC_SEED)
    order = np.arange(total)
    entity_index = order % entities
    per_entity_position = order // entities
    base = np.datetime64("2026-01-01T00:00:00", "us").astype("int64")
    event_micros = base + per_entity_position * STREAM_ROW_MICROS
    table = pa.table(
        {
            "event_time": event_micros.astype("datetime64[us]"),
            "sequence": order.astype(np.uint64),
            "symbol": np.array([f"S{index:03d}" for index in entity_index]),
            "industry": np.array(
                [f"I{index % industries:02d}" for index in entity_index]
            ),
            "close": rng.uniform(50.0, 150.0, total),
            "volume": rng.uniform(1_000.0, 10_000.0, total),
        },
        schema=_QUOTE_SCHEMA,
    )
    return QuoteWorkload(
        batch=Batch.from_pyarrow(table),
        entities=entities,
        industries=industries,
        rows=total,
    )


def matmul_workload() -> QuoteWorkload:
    """Quote workload capped to the dense-matrix budget of the matmul scenario."""
    return quote_workload(rows=min(selected_scale().table_rows, MATMUL_MAX_ROWS))


def utc_event_time_batch(batch: Batch) -> Batch:
    """Return one batch whose event_time column carries a UTC timezone."""
    table = batch.to_pyarrow()
    event_time = table.schema.get_field_index("event_time")
    return Batch.from_pyarrow(
        table.set_column(
            event_time,
            pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
            table["event_time"].cast(pa.timestamp("us", tz="UTC")),
        )
    )


def stream_batches(
    rows: int | None = None,
) -> tuple[list[tuple[pa.Table, int]], int]:
    """Split one stream workload into bounded interleaved-entity batches.

    Each entity contributes one row per second and the tumbling window spans
    60 seconds, so active operator state per entity is bounded by the 60-row
    rolling-history shape the symbolic rolling operator will declare.
    """
    scale = selected_scale()
    total = min(
        STREAM_MAX_ROWS,
        scale.table_rows if rows is None else rows,
    )
    total = total // STREAM_ENTITIES * STREAM_ENTITIES
    rng = np.random.default_rng(SYMBOLIC_SEED)
    order = np.arange(total)
    entity_index = order % STREAM_ENTITIES
    per_entity_position = order // STREAM_ENTITIES
    base = np.datetime64("2026-01-01T00:00:00", "us").astype("int64")
    event_micros = base + per_entity_position * STREAM_ROW_MICROS
    batches: list[tuple[pa.Table, int]] = []
    for start in range(0, total, STREAM_BATCH_ROWS):
        stop = min(start + STREAM_BATCH_ROWS, total)
        batches.append(
            (
                pa.table(
                    {
                        "event_time": event_micros[start:stop].astype("datetime64[us]"),
                        "symbol": np.array(
                            [f"S{index:03d}" for index in entity_index[start:stop]]
                        ),
                        "value": rng.random(stop - start),
                    },
                    schema=_STREAM_SCHEMA,
                ),
                int(event_micros[start:stop].max()),
            )
        )
    return batches, total


def stream_graph_json(window_seconds: int = STREAM_WINDOW_SECONDS) -> str:
    """Hand-authored stream graph with one stateful window node.

    ``PipelineBuilder`` cannot yet declare window nodes, so the benchmark
    authors the canonical project graph directly (same private surface the
    existing suite already reaches for ``_ArrayProvider``) and compiles it
    through the public ``PipelineBuilder.compile_stream()`` path.
    """
    project = {
        "data_sources": [],
        "format_version": 3,
        "id": "symbolic-stream-baseline",
        "name": "symbolic-stream-baseline",
        "runtime": {"mode": "stream", "options": {}},
        "graph": {
            "edges": [],
            "name": "symbolic-stream-baseline",
            "nodes": [
                {
                    "id": "windows",
                    "input_ports": [
                        {
                            "name": "input",
                            "kind": "table",
                            "required": True,
                            "schema": [
                                {
                                    "name": field.name,
                                    "data_type": _arrow_type_name(field.type),
                                    "nullable": field.nullable,
                                }
                                for field in _STREAM_SCHEMA
                            ],
                        }
                    ],
                    "operator": {
                        "kind": "window",
                        "spec": {
                            "event_time_column": "event_time",
                            "group_by": ["symbol"],
                            "geometry": {
                                "kind": "tumbling",
                                "size_micros": window_seconds * 1_000_000,
                            },
                            "aggregates": [
                                {
                                    "function": "count",
                                    "column": "value",
                                    "output": "row_count",
                                },
                                {
                                    "function": "sum",
                                    "column": "value",
                                    "output": "total",
                                },
                                {
                                    "function": "avg",
                                    "column": "value",
                                    "output": "mean",
                                },
                            ],
                        },
                    },
                }
            ],
        },
    }
    return json.dumps(project, separators=(",", ":"), sort_keys=True)


def _arrow_type_name(data_type: pa.DataType) -> str:
    names = {
        pa.timestamp("us"): "timestamp[us]",
        pa.string(): "string",
        pa.float64(): "float64",
    }
    try:
        return names[data_type]
    except KeyError as error:
        raise ValueError(
            f"stream graph schema has no portable type name for {data_type}"
        ) from error


def peak_rss_bytes() -> int:
    """Process peak resident set size as reported by the operating system."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    if sys.platform == "win32":
        return int(psutil.Process().memory_info().peak_wset)
    # ru_maxrss is reported in bytes on macOS and kibibytes elsewhere.
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def directory_bytes(path: Path) -> int:
    """Total size of every regular file under one checkpoint directory."""
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


class CountingTableMatmul:
    """Delegate for the real table_matmul provider that counts invocations."""

    __slots__ = ("calls", "inner")

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.calls = 0

    def validate(self, options: Mapping[str, object]) -> None:
        self.inner.validate(options)

    def __call__(
        self,
        inputs: Mapping[str, Batch],
        options: Mapping[str, object],
    ) -> dict[str, Batch]:
        self.calls += 1
        return self.inner(inputs, options)


def counting_matmul_runtime(backend: str) -> tuple[Runtime, CountingTableMatmul]:
    """Register one counting table_matmul provider on a fresh runtime."""
    from calc_flow.array import _TableMatmulProvider

    namespace = _matmul_namespace(backend)
    counting = CountingTableMatmul(_TableMatmulProvider(backend, namespace))
    runtime = Runtime()
    runtime._register_mapping_provider(
        backend,
        "table_matmul",
        "1",
        counting,
        input_ports=(("table", "table"), ("weights", "array")),
        output_ports=(("output", "array"),),
    )
    return runtime, counting


def _matmul_namespace(backend: str) -> object:
    if backend == "numpy":
        return np
    if backend == "jax":
        import jax.numpy as jnp

        return jnp
    raise ValueError(f"unsupported table_matmul backend {backend!r}")


def arrow_column_bytes(batch: Batch, columns: Sequence[str]) -> int:
    """Bytes of the Arrow buffers backing the selected input columns."""
    table = batch.to_pyarrow()
    return sum(table[name].nbytes for name in columns)


def execute_compiled_plan(plan: Any, inputs: Mapping[str, Batch]) -> Any:
    """Execute a plan compiled only from static benchmark declarations."""

    return plan.execute(dict(inputs))  # type: ignore[attr-defined]  # nosemgrep


def timed_plan_execute(plan: Any, inputs: Mapping[str, Batch]) -> tuple[Any, float]:
    """Time one compiled-plan execution, returning the result and seconds."""
    started = time.perf_counter_ns()
    result = execute_compiled_plan(plan, inputs)
    seconds = (time.perf_counter_ns() - started) / 1_000_000_000
    return result, seconds


def alternating_plan_samples(
    hand_built_plan: Any,
    symbolic_plan: Any,
    inputs: Mapping[str, Batch],
    *,
    sample_count: int,
) -> list[dict[str, object]]:
    """Collect paired plan timings, alternating the execution order each round."""
    samples: list[dict[str, object]] = []
    for index in range(sample_count):
        if index % 2 == 0:
            _hand_result, hand_seconds = timed_plan_execute(hand_built_plan, inputs)
            _symbolic_result, symbolic_seconds = timed_plan_execute(
                symbolic_plan, inputs
            )
            order = "hand-built-first"
        else:
            _symbolic_result, symbolic_seconds = timed_plan_execute(
                symbolic_plan, inputs
            )
            _hand_result, hand_seconds = timed_plan_execute(hand_built_plan, inputs)
            order = "symbolic-first"
        samples.append(
            {
                "order": order,
                "hand_built_seconds": hand_seconds,
                "symbolic_seconds": symbolic_seconds,
            }
        )
    return samples


def record_symbolic_benchmark(
    benchmark: BenchmarkFixture,
    *,
    scenario: str,
    input_rows: int,
    output_rows: int,
    metrics: Sequence[Mapping[str, object]] = (),
    backend: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> None:
    """Snapshot symbolic baseline metrics into pytest-benchmark metadata."""
    record_benchmark(
        benchmark,
        scenario=scenario,
        input_rows=input_rows,
        output_rows=output_rows,
        metrics=metrics,
        backend=backend,
    )
    scale = selected_scale()
    # The package under test is intentionally excluded: its source revision is
    # the candidate variable, while this identity captures only the runtime
    # dependencies that must remain fixed across a paired comparison.
    dependency_packages = ["numpy", "pyarrow"]
    if backend == "jax":
        dependency_packages.extend(("jax", "jaxlib"))
    record_comparable_identity(
        benchmark,
        workload_identity={
            "suite": "symbolic",
            "workload_version": 1,
            "scenario": scenario,
            "scale": scale.name,
            "table_rows": scale.table_rows,
            "array_elements": scale.array_elements,
            "matrix_dimension": scale.matrix_dimension,
            "input_rows": input_rows,
            "output_rows": output_rows,
            "backend": backend,
            "stream_configuration": (
                {
                    name: extra[name]
                    for name in (
                        "stream_batches",
                        "stream_batch_rows",
                        "stream_entities",
                        "stream_window_seconds",
                        "checkpoint_batches",
                        "consumed_rows",
                    )
                    if extra is not None and name in extra
                }
                or None
            ),
        },
        dependency_packages=dependency_packages,
    )
    additions = {
        "peak_rss_bytes": peak_rss_bytes(),
        "process_rss_bytes": psutil.Process().memory_info().rss,
    }
    if extra:
        additions = {**additions, **dict(extra)}
    benchmark.extra_info = {**benchmark.extra_info, **additions}

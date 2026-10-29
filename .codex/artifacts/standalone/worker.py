"""Fixed DAL-475 cases using the maintained array and paired-statistics harness."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pyarrow as pa

import calc_flow as cf
from benchmarks.array_support import (
    ARRAY_WORKLOADS,
    benchmark_calculation,
    plan_for,
    runtime_for,
    synchronize,
)

case = sys.argv[1]
root = Path(cf.__file__).parent
with Path(cf._native.__file__).open("rb") as native_file:
    native_hash = hashlib.file_digest(native_file, "sha256").hexdigest()
source_hash = hashlib.sha256(
    b"".join(p.read_bytes() for p in sorted(root.rglob("*.py")))
).hexdigest()
versions = {
    dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
}

table = pa.table(
    {
        "a": pa.chunked_array([list(range(32)), list(range(32, 64))]),
        "b": pa.chunked_array([[1] * 32, [1] * 32]),
    }
)
expected = {"total": list(range(1, 65))}
source = None
metadata = {}

if case == "arrow":

    def calculate():
        return cf.Batch.from_pyarrow(table).to_pyarrow()

    def check(result):
        assert result.equals(table)
        for left, right in zip(result.columns, table.columns):
            assert [
                [None if b is None else b.address for b in c.buffers()]
                for c in left.chunks
            ] == [
                [None if b is None else b.address for b in c.buffers()]
                for c in right.chunks
            ]
elif case == "compute":

    def calculate():
        return cf.compute(table, lambda t: t.select(total=t["a"] + t["b"]))

    def check(result):
        assert result.to_pydict() == expected
elif case == "numpy":
    workload = ARRAY_WORKLOADS[0]
    runtime = runtime_for("numpy")
    plan = plan_for(runtime, "numpy", workload)
    source = np.arange(1000, dtype=np.float64)

    def calculate():
        return plan.execute(
            {"input": cf.Batch.from_array(source, backend="numpy")}
        ).outputs["output"]

    def check(result):
        np.testing.assert_allclose(result.array, np.sqrt(source * source + 1))
        assert np.array_equal(source, np.arange(1000, dtype=np.float64))
else:
    backend = "jax" if case == "jax_matrix" else "numpy"
    dtype = np.float32 if backend == "jax" else np.float64
    table = pa.table({"a": np.arange(64, dtype=dtype), "b": np.ones(64, dtype=dtype)})
    runtime = runtime_for(backend)
    values = np.array([[1, 2], [3, 4]], dtype=dtype)
    if backend == "jax":
        import jax.numpy as jnp

        values = jnp.asarray(values)
    weights = cf.Batch.from_array(values, backend=backend)
    plan = (
        cf.PipelineBuilder("matrix")
        .table_matmul("multiply", backend=backend, columns=("a", "b"))
        .compile_batch(runtime)
    )

    def calculate():
        result = plan.execute(
            {"table": cf.Batch.from_pyarrow(table), "weights": weights}
        ).outputs["output"]
        synchronize(result)
        if backend == "jax":
            # Complete the explicit device-to-host boundary inside the timing.
            np.asarray(result.array)
        return result

    def check(result):
        np.testing.assert_allclose(
            np.asarray(result.array),
            np.column_stack((np.arange(64) + 3, 2 * np.arange(64) + 4)),
        )
        assert table["a"].to_pylist() == list(range(64))
        if backend == "jax":
            assert result.array.device == weights.array.device


start = time.perf_counter()
first = calculate()
synchronize(first) if isinstance(first, cf.Batch) else None
cold = time.perf_counter() - start
check(first)


class Capture:
    extra_info: dict

    def __init__(self):
        self.extra_info = {}
        self.samples = []

    def __call__(self, function):
        result = None
        for _ in range(7):
            gc.collect()
            start = time.perf_counter_ns()
            result = function()
            self.samples.append((time.perf_counter_ns() - start) / 1e9)
        return result


capture = Capture()
if case == "numpy":
    result = benchmark_calculation(
        capture,
        backend="numpy",
        scope="plan_end_to_end",
        scenario=workload.scenario,
        expression=workload.expression,
        input_value=cf.Batch.from_array(source, backend="numpy"),
        calculate=calculate,
    )
else:
    result = capture(calculate)
check(result)
tracemalloc.start()
profiled = calculate()
_, python_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
check(profiled)
if isinstance(result, cf.Batch):
    metadata = result.metadata
print(
    json.dumps(
        {
            "case": case,
            "pid": os.getpid(),
            "native_sha256": native_hash,
            "source_sha256": source_hash,
            "seconds": statistics.median(capture.samples),
            "raw": capture.samples,
            "first_seconds": cold,
            "python_peak_bytes": python_peak,
            "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "metadata": metadata,
            "array_record": capture.extra_info,
            "correctness": True,
            "identity": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "cpu_model": next(
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "affinity": sorted(os.sched_getaffinity(0)),
                "versions": versions,
                "threads": {
                    k: os.environ.get(k)
                    for k in (
                        "TOKIO_WORKER_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "XLA_FLAGS",
                    )
                },
            },
        }
    )
)

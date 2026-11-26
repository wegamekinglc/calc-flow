"""Multiply selected Arrow table columns by NumPy and JAX weight matrices."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from calc_flow import (
    Batch,
    PipelineBuilder,
    RunResult,
    Runtime,
    register_jax,
    register_numpy,
)

EXPECTED = [[6.0, 10.0], [2.0, 12.0], [8.0, 10.0]]
COLUMNS = ("quantity", "unit_price")


def table() -> pa.Table:
    return pa.table(
        {
            "quantity": pa.array([3.0, 1.0, 4.0], type=pa.float32()),
            "unit_price": pa.array([10.0, 12.0, 10.0], type=pa.float32()),
        }
    )


def checked_output(
    run: RunResult,
    backend: str,
    source: pa.Table,
    source_before: dict[str, list[float]],
) -> list[list[float]]:
    output = run.outputs["output"]
    if (output.kind, output.backend) != ("array", backend):
        raise RuntimeError(f"expected an array result on {backend}")
    result = output.array.tolist()
    if output.array.shape != (3, 2) or result != EXPECTED:
        raise RuntimeError(f"unexpected {backend} matrix: {result}")
    if source.to_pydict() != source_before:
        raise RuntimeError("the input table changed during multiplication")
    if run.datafusion_metrics:
        raise RuntimeError("the array provider unexpectedly used DataFusion")
    return result


def run_numpy(source: pa.Table) -> list[list[float]]:
    runtime = Runtime()
    register_numpy(runtime)
    plan = (
        PipelineBuilder("numpy-table-matmul")
        .table_matmul("multiply", backend="numpy", columns=COLUMNS)
        .compile_batch(runtime)
    )
    weights = np.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    source_before = source.to_pydict()
    weights_before = weights.copy()
    run = plan.execute(
        {
            "table": Batch.from_pyarrow(source),
            "weights": Batch.from_array(weights, backend="numpy"),
        }
    )
    np.testing.assert_array_equal(weights, weights_before)
    return checked_output(run, "numpy", source, source_before)


def run_jax(source: pa.Table) -> list[list[float]] | None:
    try:
        import jax.numpy as jnp
    except ImportError:
        return None

    runtime = Runtime()
    register_jax(runtime)
    plan = (
        PipelineBuilder("jax-table-matmul")
        .table_matmul("multiply", backend="jax", columns=COLUMNS)
        .compile_batch(runtime)
    )
    weights = jnp.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    source_before = source.to_pydict()
    weights_before = weights.tolist()
    run = plan.execute(
        {
            "table": Batch.from_pyarrow(source),
            "weights": Batch.from_array(weights, backend="jax"),
        }
    )
    if weights.tolist() != weights_before:
        raise RuntimeError("the JAX weights changed during multiplication")
    return checked_output(run, "jax", source, source_before)


def main() -> None:
    source = table()
    print("NumPy result:", run_numpy(source))
    jax_result = run_jax(source)
    if jax_result is None:
        print("JAX result: skipped; install calc-flow-python[jax]")
    else:
        print("JAX result:", jax_result)


if __name__ == "__main__":
    main()

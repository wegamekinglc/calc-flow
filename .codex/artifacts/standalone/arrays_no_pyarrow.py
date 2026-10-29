"""Exercise both optional matrix providers in an environment without PyArrow."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

import calc_flow as cf

assert importlib.util.find_spec("pyarrow") is None
batch = cf._native._batch_from_ipc(
    Path("scripts/fixtures/standalone.arrow").read_bytes()
)
runtime = cf.Runtime()
cf.register_numpy(runtime)
cf.register_jax(runtime)
for backend in ("numpy", "jax"):
    import jax.numpy as jnp

    data = cf.compute(
        batch,
        lambda t: t.sql("SELECT CAST(a AS REAL) AS a, CAST(b AS REAL) AS b FROM input"),
        output_format="batch",
    )
    source = cf.table_input("input", schema=data._schema)
    dtype = "float32" if backend == "jax" else "float64"
    weights = cf.parameter(
        "weights", kind="array", backend=backend, dtype=dtype, shape=(2, 1)
    )
    matrix = cf.linalg.from_columns(source, columns=("a", "b"), backend=backend)
    output = cf.table.attach_columns(
        source, cf.linalg.matmul(matrix, weights), names=("score",)
    )
    program = cf.Program("matrix-standalone", engine="sql", outputs={"result": output})
    caller = np.array([[1.0], [2.0]], dtype=dtype)
    weights_batch = cf.Batch.from_array(
        jnp.asarray(caller) if backend == "jax" else caller, backend=backend
    )
    caller[:] = 99
    result = program.collect(
        {"input": data, "weights": weights_batch},
        runtime=runtime,
        output_format="batch",
    )["result"]
    assert result.num_rows == 2
    assert (
        cf.compute(
            result,
            lambda t: t.filter(
                cf.row.cast(t["score"], "float64")
                != cf.row.cast(t["a"], "float64") + 2.0 * cf.row.cast(t["b"], "float64")
            ),
            output_format="batch",
        ).num_rows
        == 0
    )
assert "pyarrow" not in sys.modules
print(
    "NumPy and JAX table assembly, attachment and input snapshots "
    "passed without PyArrow"
)

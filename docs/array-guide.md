# Arrays and matrices

[Documentation](README.md) / 2.2 Arrays and matrices

Calc Flow supports explicit NumPy and JAX array providers alongside Arrow table
calculations. Compose table-attached matrix expressions with the root `cf.linalg`,
`cf.parameter`, and `cf.table` namespaces. Install the matching optional dependency
and register it on the `Runtime` used for collection or compilation. Example 06
and the NumPy parts of examples 07 and 11 require NumPy; example 07 skips its
JAX part when JAX is unavailable.

## Reuse static weights in batch and stream modes

Run [11_symbolic_static_matrix.py](../examples/11_symbolic_static_matrix.py):

```bash
uv run --no-sync python examples/11_symbolic_static_matrix.py
```

The program declares table columns and a typed `weights` parameter, converts
the columns with `linalg.from_columns`, applies `linalg.matmul`, and attaches
the result with `table.attach_columns`. It asserts batch/stream result parity
and checks one-time weight placement for the stream job. It also demonstrates
the compilation error when the required provider has not been registered.

For batch use, the example calls `await program.collect_async` with logical
inputs `"prices"` and `"weights"` and reads logical output `"signals"`. The
weights value is an explicit `Batch.from_array(..., backend="numpy")`; the
selected runtime has `register_numpy(runtime)` installed. Default runtimes do
not register NumPy/JAX automatically.

Supply stream weights through `StreamingRunner(..., static_inputs={...})`.
The runner freezes and fingerprints their values before opening a source.
Recovery rejects changed weights against the saved checkpoint. JAX float64
declarations require JAX x64 support; types are not silently narrowed.

See [expression workflows](symbolic-workflows.md#use-static-matrices-with-numpy-or-jax)
for the complete workflow and [static inputs](streaming-guide.md#static-inputs)
for validation and recovery behavior. Implementation and copy boundaries are
described separately in [symbolic compiler design](symbolic-design.md).

## Advanced array and graph interfaces

Standalone `ArrayExpr` outputs are declaration-only. To return an array `Batch`
directly, use the explicit provider and graph methods below; `compute` and
`collect` return tables. The table-attached expression workflow above produces
Arrow results and requires the exact supported matrix shape.

## Center an array

Run [06_numpy_array.py](../examples/06_numpy_array.py):

```bash
uv run --no-sync python examples/06_numpy_array.py
```

The example calls `register_numpy(runtime)`, wraps `[1.0, 2.0, 4.0, 6.0]`
with `Batch.from_array(..., backend="numpy")`, and selects the external
`numpy:expression@1` provider. The expression `x - mean(x)` produces:

```text
[-2.25, -1.25, 0.75, 2.75]
```

Array expressions use a bounded allowlist of syntax and operations. Imports,
arbitrary attribute access, and arbitrary Python execution are unsupported.
This reduction is a batch operation. Stream array expressions must depend on
`x` and use the supported elementwise subset without calls or matrix
multiplication.

## Multiply table columns by weights

Run [07_array_and_dataframe.py](../examples/07_array_and_dataframe.py):

```bash
uv run --no-sync python examples/07_array_and_dataframe.py
```

Here a dataframe means a `pyarrow.Table`. `table_matmul` selects ordered
columns `("quantity", "unit_price")`; its named inputs are `table` and
`weights`. A diagonal weight matrix doubles quantity and retains unit price:

```text
[[6.0, 10.0], [2.0, 12.0], [8.0, 10.0]]
```

Both array operands must use the selected backend, and their dimensions and
numeric types must be compatible. The output is an array `Batch`, accessed
through `.array`, rather than a table. The example checks shape, backend,
values, and preservation of the original table and weights. Although it reads
Arrow columns, this provider operation does not produce DataFusion metrics.

`table_matmul@1` is batch-only. To reuse weights in a continuous calculation,
use the expression matrix workflow above.

Next: [continuous streaming](streaming-guide.md).

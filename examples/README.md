# Calc Flow 4.0 examples

These executable examples use the Rust-native 4.0 engine through the PyO3
Python package. Table batches contain PyArrow tables and every table expression
or query runs in DataFusion. NumPy remains an explicit optional provider.
DataFrame-style data means `pyarrow.Table`; pandas and Polars are not part of
this example surface.

Install the optional array providers as needed:

```bash
uv add "calc-flow[numpy]"
uv add "calc-flow[jax]"
```

After `uv sync --extra dev && uv run maturin develop`, run:

```bash
uv run python examples/01_datafusion_pipeline.py
uv run python examples/02_sql_join.py
uv run python examples/03_registered_udf.py
uv run python examples/04_continuous_runtime.py
uv run python examples/05_async_execution.py
JAX_PLATFORMS=cpu uv run python examples/06_numpy_array.py
JAX_PLATFORMS=cpu uv run python examples/07_array_and_dataframe.py
uv run python examples/08_streaming_recovery.py
uv run python examples/09_symbolic_financial_features.py
uv run python examples/10_symbolic_streaming_recovery.py
uv run python examples/11_symbolic_static_matrix.py
uv run python examples/12_symbolic_stream_join.py
```

Or run the entire Python and Rust user-example inventory:

```bash
JAX_PLATFORMS=cpu uv run python scripts/run_examples.py
```

The files cover:

- `01_datafusion_pipeline.py` — the functional `PipelineBuilder`, expression
  nodes, projection/filtering, immutable batches, outputs, and node timings.
- `02_sql_join.py` — named table inputs and one read-only multi-table
  DataFusion SQL query.
- `03_registered_udf.py` — a trusted, versioned Python scalar UDF registered on
  `Runtime` and explicitly selected by a node.
- `04_continuous_runtime.py` — the async source-driven continuous lifecycle,
  a replayable cursor, managed checkpoints, synchronous status, and explicit
  terminal wait.
- `05_async_execution.py` — non-blocking `execute_async()` use with copied
  settings and a timezone-aware deadline normalized to UTC inside an asyncio
  application.
- `06_numpy_array.py` — explicit NumPy provider registration and a restricted
  array expression over an immutable array batch.
- `07_array_and_dataframe.py` — explicit `pyarrow.Table`-to-array matrix
  multiplication using NumPy and, when installed, JAX.
- `08_streaming_recovery.py` — a second process-lifecycle run over the same
  managed checkpoint root, proving terminal recovery does not reopen an ended
  source or duplicate sink output.
- `09_symbolic_financial_features.py` — composed rolling momentum, Bollinger,
  RSI, and cross-section features with analysis and batch execution.
- `10_symbolic_streaming_recovery.py` — two-stage symbolic rolling with a
  replayable source, mid-stream checkpoint recovery, and terminal recovery.
- `11_symbolic_static_matrix.py` — an immutable NumPy weight matrix used by the
  same symbolic program in batch and stream modes, including provider failure
  and one-time static placement evidence.
- `12_symbolic_stream_join.py` — two ordered symbolic inputs lowered to one
  bounded native stream join and executed across independent segmentations.

`07_array_and_dataframe.py` selects ordered numeric `pyarrow.Table` columns
and multiplies their dense matrix by an array weight matrix. After input
`Batch` construction, execution makes no redundant copies: NumPy allocates one
dense table matrix and one result; JAX permits one host staging buffer, one
device table buffer, and one device result. Input `Batch` construction is
outside these operator-execution ceilings. JAX performs
no result-to-host round trip during operator execution.

The previous v1 notebook was removed because it taught the frozen Python v1
operator API. Historical v1 behavior is preserved in
[commit `c87324e`](https://github.com/wegamekinglc/calc-flow/tree/c87324ecaee30d8b883d3c30ae03704dee45f593)
and in `tests/fixtures/v1/`.

## Rust counterparts

The Rust crate ships parallel examples under
[`crates/calc-flow/examples/`](../crates/calc-flow/examples/README.md). The
expression (`01_datafusion_pipeline.py` ↔ `expression_pipeline.rs`) and SQL
join (`02_sql_join.py` ↔ `sql_join.rs`) examples share their datasets and
expressions. `04_continuous_runtime.py` and `continuous_runtime.rs` demonstrate
the same source-driven lifecycle on the Python and Rust surfaces. Rust's
`windowed_streaming.rs` adds event-time aggregation; Python's
`08_streaming_recovery.py` focuses on durable terminal recovery.

See the [executable example guide](../docs/examples.md) for the cross-language
matrix and the [continuous streaming guide](../docs/streaming-guide.md) for the
runtime contracts behind these programs.
The [symbolic workflow guide](../docs/symbolic-workflows.md) covers the
declaration-to-Studio path, NumPy/JAX providers, capability failures, and
performance interpretation.

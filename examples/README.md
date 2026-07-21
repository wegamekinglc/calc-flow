# Calc Flow v2 examples

These executable examples use the Rust-native v2 engine through the PyO3
Python package. Table batches contain PyArrow tables and every table expression
or query runs in DataFusion. NumPy remains an explicit optional provider.

After `uv sync --extra dev && uv run maturin develop`, run:

```bash
uv run python examples/01_datafusion_pipeline.py
uv run python examples/02_sql_join.py
uv run python examples/03_registered_udf.py
uv run python examples/04_micro_batch_recovery.py
uv run python examples/05_async_execution.py
JAX_PLATFORMS=cpu uv run python examples/06_numpy_array.py
```

The files cover:

- `01_datafusion_pipeline.py` — the functional `PipelineBuilder`, expression
  nodes, projection/filtering, immutable batches, outputs, and node timings.
- `02_sql_join.py` — named table inputs and one read-only multi-table
  DataFusion SQL query.
- `03_registered_udf.py` — a trusted, versioned Python scalar UDF registered on
  `Runtime` and explicitly selected by a node.
- `04_micro_batch_recovery.py` — a replayable source, checkpoint commit, plan
  lease release, and recovery from the stored source cursor.
- `05_async_execution.py` — non-blocking `execute_async()` use inside an
  asyncio application.
- `06_numpy_array.py` — explicit NumPy provider registration and a restricted
  array expression over an immutable array batch.

The previous v1 notebook was removed because it taught the frozen Python v1
operator API. Historical v1 behavior remains available at the
`v1-python-final` tag and in `tests/fixtures/v1/`.

## Rust counterparts

The Rust crate ships parallel examples under
[`crates/calc-flow/examples/`](../crates/calc-flow/examples/README.md). The
expression (`01_datafusion_pipeline.py` ↔ `expression_pipeline.rs`), SQL join
(`02_sql_join.py` ↔ `sql_join.rs`), and micro-batch recovery
(`04_micro_batch_recovery.py` ↔ `micro_batch_recovery.rs`) examples share their
datasets and expressions across surfaces so the Python binding and the Rust
crate read as one engine.

# Calc Flow Rust examples

These executable examples use the `calc-flow` crate directly. Table batches
contain Arrow `RecordBatch` values and every table expression or query runs in
DataFusion. Run them from a repository checkout with:

```bash
cargo run -p calc-flow --example expression_pipeline
cargo run -p calc-flow --example sql_join
cargo run -p calc-flow --example continuous_runtime
cargo run -p calc-flow --example windowed_streaming
cargo run -p calc-flow --example export_schema
cargo run -p calc-flow --example gen_v3_schema
```

The files cover:

- `expression_pipeline.rs` — the canonical first example. A one-node
  `ExpressionOperator` computes `total = a + b` over an Arrow `RecordBatch` and
  awaits `BatchExecutionPlan::execute`.
- `sql_join.rs` — a `SqlOperator` joins two named table inputs (`orders`,
  `fees`) with one read-only DataFusion `SELECT`.
- `continuous_runtime.rs` — a replayable `StreamSource`, an ordinary
  `StreamSink`, managed checkpoints, and the one-shot runner/job lifecycle.
- `windowed_streaming.rs` — source-provided event-time progress, a stateful
  tumbling-window sum, deterministic watermark/end output, and terminal state.
- `export_schema.rs` — prints the canonical v3 project JSON Schema.
- `gen_v3_schema.rs` — regenerates the canonical v3 project JSON Schema.

The Python binding ships parallel examples under
[`examples/`](../../../examples/README.md). The SQL examples share their dataset.
The Rust expression example uses the introduction's addition calculation;
Python 01 extends the same builder pattern with order totals and filtering.

Use `uv run python scripts/run_examples.py --surface rust` to run every
user-facing Rust example. Schema export and generation remain explicit tooling
commands because generation updates a checked-in artifact.

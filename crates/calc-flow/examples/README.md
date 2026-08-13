# Calc Flow Rust examples

These executable examples use the `calc-flow` crate directly. Table batches
contain Arrow `RecordBatch` values and every table expression or query runs in
DataFusion. Run them from a repository checkout with:

```bash
cargo run -p calc-flow --example expression_pipeline
cargo run -p calc-flow --example sql_join
cargo run -p calc-flow --example continuous_runtime
cargo run -p calc-flow --example export_schema
```

The files cover:

- `expression_pipeline.rs` — the canonical first example. A one-node
  `ExpressionOperator` computes `total = a + b` over an Arrow `RecordBatch` and
  awaits `ExecutionPlan::execute`.
- `sql_join.rs` — a `SqlOperator` joins two named table inputs (`orders`,
  `fees`) with one read-only DataFusion `SELECT`.
- `continuous_runtime.rs` — a replayable `StreamSource`, an ordinary
  `StreamSink`, managed checkpoints, and the one-shot runner/job lifecycle.
- `export_schema.rs` — prints the canonical v2 project JSON Schema.

The Python binding ships parallel examples under
[`examples/`](../../../examples/README.md). The expression and SQL examples use
the same datasets and expressions on both surfaces so the Rust crate and the
Python package read as one engine.

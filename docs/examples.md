# Executable examples

Calc Flow keeps examples executable, small, and aligned across Rust and Python.
They are integration smoke tests for the public API, not pseudocode. Start with
the batch examples, then use the continuous examples to learn source ownership,
watermarks, checkpoints, and recovery.

## Prepare a checkout

```bash
uv sync --extra dev
uv run maturin develop
```

The NumPy example needs the `numpy` extra. The development environment installs
it. JAX is optional; the mixed table/array example reports a skip when JAX is
not installed.

Run every user-facing example with one command:

```bash
JAX_PLATFORMS=cpu uv run python scripts/run_examples.py
```

Select one language surface when iterating:

```bash
uv run python scripts/run_examples.py --surface python
uv run python scripts/run_examples.py --surface rust
```

Schema exporters are intentionally excluded from this runner because
`gen_v3_schema` updates a checked-in generated file. Their commands remain in
the [Rust example inventory](../crates/calc-flow/examples/README.md).

## Python examples

| File                                                                                      | What it proves                                                        |
| ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [`01_datafusion_pipeline.py`](../examples/01_datafusion_pipeline.py)                      | Immutable Arrow input, expressions, projection, filtering, timings    |
| [`02_sql_join.py`](../examples/02_sql_join.py)                                            | Named inputs and one read-only DataFusion SQL join                    |
| [`03_registered_udf.py`](../examples/03_registered_udf.py)                                | Trusted, versioned, explicitly selected scalar UDF                    |
| [`04_continuous_runtime.py`](../examples/04_continuous_runtime.py)                        | One-shot runner, replay cursor, managed checkpoint, terminal wait     |
| [`05_async_execution.py`](../examples/05_async_execution.py)                              | Async batch execution, settings, absolute deadline                    |
| [`06_numpy_array.py`](../examples/06_numpy_array.py)                                      | Explicit NumPy registration and bounded array expression              |
| [`07_array_and_dataframe.py`](../examples/07_array_and_dataframe.py)                      | Arrow-table to NumPy/JAX matrix multiplication without input mutation |
| [`08_streaming_recovery.py`](../examples/08_streaming_recovery.py)                        | Durable terminal recovery without source reopen or duplicate output   |
| [`09_symbolic_financial_features.py`](../examples/09_symbolic_financial_features.py)      | Composed rolling, RSI, and cross-section financial features           |
| [`10_symbolic_streaming_recovery.py`](../examples/10_symbolic_streaming_recovery.py)      | Multi-stage symbolic checkpoint, recovery, and terminal restart       |
| [`11_symbolic_static_matrix.py`](../examples/11_symbolic_static_matrix.py)                | Batch/stream static weights, copy facts, and provider failure         |

The Python continuous examples use application-owned in-memory connectors.
Production transport configuration is data-only and covered by the
[connector guide](connectors.md).

## Rust examples

| File                                                                            | What it proves                                                     |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| [`expression_pipeline.rs`](../crates/calc-flow/examples/expression_pipeline.rs) | Native Arrow expression pipeline and async plan execution          |
| [`sql_join.rs`](../crates/calc-flow/examples/sql_join.rs)                       | Multi-input read-only SQL                                          |
| [`continuous_runtime.rs`](../crates/calc-flow/examples/continuous_runtime.rs)   | Public source/sink traits and owning continuous lifecycle          |
| [`windowed_streaming.rs`](../crates/calc-flow/examples/windowed_streaming.rs)   | Source-provided watermark and stateful tumbling-window aggregation |
| [`export_schema.rs`](../crates/calc-flow/examples/export_schema.rs)             | Print the canonical project-v3 JSON Schema                         |
| [`gen_v3_schema.rs`](../crates/calc-flow/examples/gen_v3_schema.rs)             | Regenerate the checked-in project-v3 JSON Schema                   |

The windowed example sends three rows followed by a one-minute watermark. The
watermark closes `[00:00, 00:01)`, and end-of-input closes `[00:01, 00:02)`.
The sink asserts both deterministic aggregates before printing them.

## Which example should I copy?

- For a synchronous Python calculation, start with `01_datafusion_pipeline.py`.
- For an asyncio service, start with `05_async_execution.py`.
- For an application-owned live source, start with
  `04_continuous_runtime.py` and then read `08_streaming_recovery.py`.
- For symbolic financial calculations, start with
  `09_symbolic_financial_features.py`, then use
  `10_symbolic_streaming_recovery.py` for continuous recovery or
  `11_symbolic_static_matrix.py` for immutable model weights.
- For event-time aggregation in Rust, start with `windowed_streaming.rs`.
- For Kafka, PostgreSQL, ClickHouse, HTTP, WebSocket, files, or Parquet, use a
  project-v3 connector binding from the [connector guide](connectors.md).
- For a browser-managed local job, use [Calc Flow Studio](../web-ui/README.md).

The [symbolic workflow guide](symbolic-workflows.md) connects the three
symbolic examples to Studio inspection, NumPy/JAX selection, failure handling,
and compile-time performance facts.

## Verification contract

The runner stops at the first failed example and preserves that process's exit
code. Its command inventory is covered by:

```bash
python -m unittest scripts.test_run_examples
```

Rust formatting, Clippy, and rustdoc also compile the Rust example targets.
Python linting covers the scripts themselves, while the example runner checks
their end-to-end behavior against the installed native extension.

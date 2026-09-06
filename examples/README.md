# Calc Flow examples

Start with the [documentation overview](../docs/introduction.md) and
[installation guide](../docs/getting-started.md). These programs use the public
Rust-native engine and check observable results. Each numbered Python program
is standalone; the [learning paths](../docs/examples.md) group them by task.

## Prepare and run

From a checkout with the core package built and installed, run one program:

```bash
uv run --no-sync python examples/01_datafusion_pipeline.py
```

Run the complete inventory, or choose a language:

```bash
uv run --no-sync python scripts/run_examples.py
uv run --no-sync python scripts/run_examples.py --surface python
uv run --no-sync python scripts/run_examples.py --surface rust
```

These commands work in Bash and PowerShell. The runner defaults JAX to CPU
when `JAX_PLATFORMS` is unset. NumPy is required by examples 06, 07, and 11;
JAX is optional in 07, which prints an explicit skip when it is unavailable.
The `dev` extra installs both providers. For individual application installs,
use `calc-flow[numpy]` or `calc-flow[jax]` as needed.

The [source installation](../docs/getting-started.md#build-and-install-from-source)
builds wheels. An editable developer environment can instead use
`uv sync --extra dev` and `uv run maturin develop`; avoid leaving a generated
`_native*.so` in the source tree when preparing release artifacts.
`--no-sync` runs against the environment you prepared without replacing its
native installation.

The stream examples use application-owned connectors, finite synthetic data,
and temporary checkpoint roots. They require no Kafka, PostgreSQL, MySQL,
ClickHouse, or network service. Example 14 also uses a temporary directory. For a
constrained checkout, set `TMPDIR` on Linux or `TEMP` and `TMP` on Windows to
an existing writable directory under `target/` before running.

On Windows, checkpoint segment paths can exceed the traditional 260-character
limit. Use an extended-length absolute temporary path so Python can also clean
up those directories when system-wide long-path support is disabled:

```powershell
New-Item -ItemType Directory -Path target/tmp -Force | Out-Null
$exampleTempPath = (Resolve-Path -LiteralPath target/tmp).Path
$exampleSavedTemp = $env:TEMP
$exampleSavedTmp = $env:TMP
try {
    $env:TEMP = '\\?\' + $exampleTempPath
    $env:TMP = $env:TEMP
    uv run --no-sync python scripts/run_examples.py --surface python
} finally {
    $env:TEMP = $exampleSavedTemp
    $env:TMP = $exampleSavedTmp
}
```

## Python inventory

1. [01_datafusion_pipeline.py](01_datafusion_pipeline.py) — calculate order
   gross amounts, project columns, and filter small orders. Checks orders
   `A-100` / `A-102` with gross values `30` / `40`; prints node timings.
   Guide: [batch calculations](../docs/batch-guide.md).
2. [02_sql_join.py](02_sql_join.py) — join named Arrow inputs `orders` and
   `fees` with read-only SQL. Checks net values `[70, 108, 36]` in order-ID
   order. Guide: [SQL joins](../docs/batch-guide.md#named-inputs-and-sql-joins).
3. [03_registered_udf.py](03_registered_udf.py) — register and explicitly
   select a typed vectorized `double_amount` UDF. Checks totals
   `[200, 500, 800]`; prints registration metadata.
   Guide: [scalar functions](../docs/batch-guide.md#registered-scalar-functions).
4. [04_continuous_runtime.py](04_continuous_runtime.py) — run an async source
   and sink with replay cursors, managed checkpoints, status, and terminal
   wait. Checks lifecycle completion.
   Guide: [continuous streaming](../docs/streaming-guide.md).
5. [05_async_execution.py](05_async_execution.py) — run a batch plan alongside
   an asyncio heartbeat with settings and a deadline. Checks totals `[3, 7]`.
   Guide: [async execution](../docs/batch-guide.md#async-execution-and-deadlines).
6. [06_numpy_array.py](06_numpy_array.py) — register NumPy and center an array
   with `x - mean(x)`. Checks `[-2.25, -1.25, 0.75, 2.75]`.
   Guide: [arrays](../docs/array-guide.md#center-an-array).
7. [07_array_and_dataframe.py](07_array_and_dataframe.py) — multiply ordered
   Arrow columns by NumPy/JAX weights. Checks
   `[[6.0, 10.0], [2.0, 12.0], [8.0, 10.0]]`, shape, backend, and unchanged
   inputs. Guide: [table matrices](../docs/array-guide.md#multiply-table-columns-by-weights).
8. [08_streaming_recovery.py](08_streaming_recovery.py) — run a completed
   lineage again from its checkpoint. Checks that ended sources stay closed
   and final output is not duplicated.
   Guide: [recovery](../docs/streaming-guide.md#checkpoints-and-recovery).
9. [09_symbolic_financial_features.py](09_symbolic_financial_features.py) —
   analyze and run momentum, Bollinger, RSI, EMA/MACD, and cross-section
   features. Prints analysis, explanation, and checked feature output.
   Guide: [financial features](../docs/symbolic-workflows.md#compose-and-run-financial-features).
10. [10_symbolic_streaming_recovery.py](10_symbolic_streaming_recovery.py) —
    checkpoint a two-stage rolling program mid-stream, cancel, and resume.
    Checks restored values and terminal recovery without duplicate output.
    Guide: [symbolic recovery](../docs/symbolic-workflows.md#run-continuously-and-recover).
11. [11_symbolic_static_matrix.py](11_symbolic_static_matrix.py) — reuse an
    immutable NumPy weight matrix in batch and stream modes. Checks parity,
    provider failure, and one-time static placement.
    Guide: [static matrices](../docs/array-guide.md#reuse-static-weights-in-batch-and-stream-modes).
12. [12_symbolic_stream_join.py](12_symbolic_stream_join.py) — match ordered
    authorization and payment streams with explicit time/state bounds. Checks
    results across independent batch segmentations.
    Guide: [symbolic joins](../docs/symbolic-workflows.md#join-two-symbolic-streams).
13. [13_symbolic_relational_dag.py](13_symbolic_relational_dag.py) — feed an
    ordered authorization/payment join into a settlement join. Checks nested
    join results.
    Guide: [symbolic joins](../docs/symbolic-workflows.md#join-two-symbolic-streams).
14. [14_project_persistence.py](14_project_persistence.py) — round-trip a
    project through JSON, YAML, and an async file store, then compile and run
    the loaded graph. Checks totals `[3, 7]` and unchanged builder input.
    Guide: [project persistence](../docs/projects-guide.md).

## Rust counterparts

Run a Rust example with `cargo run -p calc-flow --example NAME`:

- [expression_pipeline.rs](../crates/calc-flow/examples/expression_pipeline.rs)
  (`expression_pipeline`) calculates the introductory totals `[3, 7]`.
- [sql_join.rs](../crates/calc-flow/examples/sql_join.rs) (`sql_join`) uses the
  same order/fee dataset as Python 02 and checks net values `[70, 108, 36]`.
- [continuous_runtime.rs](../crates/calc-flow/examples/continuous_runtime.rs)
  (`continuous_runtime`) demonstrates native source/sink traits and the owning
  job lifecycle.
- [windowed_streaming.rs](../crates/calc-flow/examples/windowed_streaming.rs)
  (`windowed_streaming`) checks deterministic one-minute window sums closed
  by a source watermark and end-of-input.

The [Rust inventory](../crates/calc-flow/examples/README.md) also lists schema
export/generation tools. They are excluded from the user-example runner
because schema generation updates a tracked artifact.

## Reading results and adding examples

JAX matrix results remain on the selected backend: there is
no result-to-host round trip during operator execution. The example's later
`.tolist()` call transfers values for checking and printing, outside operator
execution. See [arrays and matrices](../docs/array-guide.md) for usage and
[symbolic compiler design](../docs/symbolic-design.md) for copy boundaries.

Assertions check values and lifecycle behavior. Timings, IDs, diagnostic text,
and optional-provider output vary between runs. A program failure stops the
runner and preserves its exit code.

When adding an example, use the next `NN_description.py` name, keep it
standalone with a `main()` entry point, assert its expected behavior, and avoid
mutating caller-owned inputs. Clean up any job and temporary resource. Document
its dependencies and expected result here, and link it from the relevant
function guide. The runner discovers numbered Python files automatically;
verify discovery with `python -m unittest scripts.test_run_examples`.

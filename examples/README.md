# Calc Flow examples

Start with the [documentation overview](../docs/introduction.md) and
[installation guide](../docs/getting-started.md). Start with Python expressions
in 01, SQL composition in `19_sql_expression_pipeline.py`, and streams in
`20_streaming_pipeline.py` and `21_streaming_outputs.py`. Continue to 05 for
async batch execution, 09 for financial features, and 14 for project export.
These programs use the internal Rust runtime and check observable results;
explicit graph, provider, and native-extension examples cover advanced integrations. Each Python program
is standalone; the [learning paths](../docs/examples.md) group them by task.

## Prepare and run

From a checkout with `calc-flow-python` built and installed, run one program:

```bash
uv run --no-sync python examples/01_datafusion_pipeline.py
```

Run the examples that need no external service, or choose a language:

```bash
uv run --no-sync python scripts/run_examples.py
uv run --no-sync python scripts/run_examples.py --surface python
uv run --no-sync python scripts/run_examples.py --surface rust
```

These commands work in Bash and PowerShell. The runner defaults JAX to CPU
when `JAX_PLATFORMS` is unset. NumPy is required by examples 06, 07, and 11;
JAX is optional in 07, which prints an explicit skip when it is unavailable.
The `dev` extra installs both providers. For individual application installs,
use `calc-flow-python[numpy]` or `calc-flow-python[jax]` as needed.

The [source installation](../docs/getting-started.md#build-and-install-from-source)
builds wheels. An editable developer environment can instead use
`uv sync --extra dev` and `uv run maturin develop`; avoid leaving a generated
`_native*.so` in the source tree when preparing release artifacts.
`--no-sync` runs against the environment you prepared without replacing its
native installation.

Examples 04 and 08–12 use application-owned connectors, finite synthetic data,
and temporary checkpoint roots. Example 15 uses the native file connector;
these programs require no external service. The SQL/streaming pipeline examples
listed below also need no service. Connector `*_source.py` files numbered 16–21
require prepared services and opt-in native connector features; the runner skips them unless
passed `--include-services`. See [connector setup](../docs/connectors/README.md) before running
them directly or enabling that flag. Example 14 also uses a temporary directory. For a
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
   gross amounts with `cf.compute`, project columns, and filter small orders.
   Checks orders `A-100` / `A-102` with gross values `30` / `40`, then collects
   reusable `totals` and `quantities` logical outputs.
   Guide: [batch calculations](../docs/batch-guide.md).
2. [02_sql_join.py](02_sql_join.py) — bind SQL aliases `o` and `f` to named
   table inputs `orders` and `fees`, then compose a column transform with `pipe`.
   Checks `doubled=[140, 216, 72]` in order-ID order. Guide: [SQL joins](../docs/batch-guide.md#named-inputs-and-sql-joins).
3. [03_registered_udf.py](03_registered_udf.py) — register and explicitly
   select a typed vectorized `double_amount` UDF. Checks totals
   `[200, 500, 800]`; prints registration metadata.
   Guide: [scalar functions](../docs/batch-guide.md#registered-scalar-functions).
4. [04_continuous_runtime.py](04_continuous_runtime.py) — run an async source
   and sink with replay cursors, managed checkpoints, status, and terminal
   wait. Checks lifecycle completion.
   Guide: [continuous streaming](../docs/streaming-guide.md).
5. [05_async_execution.py](05_async_execution.py) — await `cf.compute_async` alongside
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
   features composed by a reusable Python mapping function. Analyzes with a
   default runtime, prints explanation, and collects checked Arrow output.
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
    `Program.to_project()` document through JSON, YAML, and an async file store,
    then execute its physical `input`/`output` bindings after reload. Checks
    totals `[3, 7]` and unchanged program declarations.
    Guide: [project persistence](../docs/projects-guide.md).
15. [15_file_source.py](15_file_source.py) — read CSV, JSON Lines, and Parquet
    through the native file source, calculate order totals, and write them
    with an exactly-once Parquet sink. Checks `[20.0, 60.0]` for each format;
    creates and removes its own sample files and checkpoints.
16. [16_kafka_source.py](16_kafka_source.py) — consume JSON orders from an
    explicitly assigned Kafka partition, wait for sink delivery, and drain.
17. [17_postgresql_source.py](17_postgresql_source.py) — read a PostgreSQL
    repeatable-read snapshot in bounded pages with best-effort delivery.
18. [18_mysql_source.py](18_mysql_source.py) — read a MySQL InnoDB snapshot
    in primary-key order, with TLS enabled by default.
19. [19_clickhouse_source.py](19_clickhouse_source.py) — read a bounded
    ClickHouse snapshot using a frozen Arrow schema and composite cursor.
20. [20_http_source.py](20_http_source.py) — poll a JSON Lines endpoint with
    conditional requests, wait for sink delivery, and drain.
21. [21_websocket_source.py](21_websocket_source.py) — read JSON frames with
    blocking backpressure, wait for sink delivery, and drain.

The connector `*_source.py` examples numbered 16–21 each check Parquet totals
`[20.0, 60.0]` from two prepared
orders and print the effective delivery guarantee. They have a 60-second
deadline and clean up jobs, outputs, and checkpoints on exit. Each script
runs independently. The [connector overview](../docs/connectors/README.md)
links to one page per transport, each with wheel features, environment
variables, sample SQL/messages, local service commands, and delivery limits.

### SQL and streaming pipelines

These scripts are distinct from the connector files with the same numeric
prefix. Use the full filename when choosing a program; all three run without
external services and are included in the default example runner.

- [19_sql_expression_pipeline.py](19_sql_expression_pipeline.py) — compose
  reusable functions with `pipe`, SQL filtering, and column arithmetic. Checks
  `order_id=[1, 3]` and `net=[18.0, 27.0]`.
  Guide: [SQL pipelines](../docs/batch-guide.md#compose-sql-and-python-pipelines).
- [20_streaming_pipeline.py](20_streaming_pipeline.py) — retain native delta
  and nested rolling-mean state across two input batches, followed by SQL.
  The source stays open while `async for` receives `delta=[None, 2.0, 3.0]` and
  `mean_delta=[None, 2.0, 2.5]`. Exiting `async with` cancels the waiting source
  and cleans up temporary checkpoints; the latest timestamp remains buffered.
  Guide: [first stream](../docs/streaming-guide.md#first-python-continuous-job).
- [21_streaming_outputs.py](21_streaming_outputs.py) — branch a Program and
  consume independent `StreamOutput` events by name. Checks
  `double=[2, 4, 6]` and `large=[2, 3]` without assuming cross-output order.
  Guide: [named outputs](../docs/streaming-guide.md#named-streaming-outputs).

The explicitly registered [symbolic_event_window.py](symbolic_event_window.py)
example computes grouped one-minute trade count, volume, low, high, and
arithmetic average price. It checks native final output with explicit source
watermarks and a `price_range` branch sharing the same window declaration.
Guide: [symbolic event windows](../docs/symbolic-workflows.md#aggregate-event-time-windows).

## Rust counterparts

These programs are runtime implementation and extension references. Python is
the application API; the native examples retain explicit graph and trait usage.

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

Checks validate values and lifecycle behavior. Timings, IDs, diagnostic text,
and optional-provider output vary between runs. A program failure stops the
runner and preserves its exit code.

When adding an example, use the next `NN_description.py` name, keep it
standalone with a `main()` entry point, verify its expected behavior even under
Python optimization (`-O`), and avoid
mutating caller-owned inputs. Clean up any job and temporary resource. Document
its dependencies and expected result here, and link it from the relevant
function guide. The runner discovers numbered Python files automatically;
`symbolic_event_window.py` is also explicitly registered. Verify discovery
with `python -m unittest scripts.test_run_examples`.
List examples requiring external services in `SERVICE_PYTHON_EXAMPLES` in
the runner so they require `--include-services`, and add their setup to
[the connector instructions](../docs/connectors/README.md).

# Calc Flow

[![Linux CI](https://github.com/wegamekinglc/calc-flow/actions/workflows/ci-linux.yml/badge.svg?branch=main)](https://github.com/wegamekinglc/calc-flow/actions/workflows/ci-linux.yml)
[![Windows CI](https://github.com/wegamekinglc/calc-flow/actions/workflows/ci-windows.yml/badge.svg?branch=main)](https://github.com/wegamekinglc/calc-flow/actions/workflows/ci-windows.yml)
[![Coverage Status](https://coveralls.io/repos/github/wegamekinglc/calc-flow/badge.svg?branch=main)](https://coveralls.io/github/wegamekinglc/calc-flow?branch=main)

Calc Flow is a Python calculation library for Arrow tables and stateful streams.
Compose typed expressions, SQL, and reusable Python functions in one pipeline.
Collect a dataset or iterate its results as data arrives. Rust provides the internal runtime:
DataFusion table execution, graph compilation, state, checkpoints, and recovery.
Calc Flow Studio is a separate local FastAPI and React application.

## Install

Python 3.13 or newer:

```bash
uv add calc-flow-python
```

Optional array providers:

```bash
uv add "calc-flow-python[numpy]"
uv add "calc-flow-python[jax]"
```

## Python quickstart

```python
import pyarrow as pa
import calc_flow as cf

data = pa.table({"a": [1, 3], "b": [2, 4]})
result = cf.compute(data, lambda t: t.select(total=t["a"] + t["b"]))
assert result.to_pydict() == {"total": [3, 7]}
```

`cf.compute` infers the supported Arrow schema and returns a `pyarrow.Table`.
The builder receives a `TableExpr`; indexing selects columns, operators compose
calculations, and `select`, `with_columns`, and `filter` return new declarations.
Use `await cf.compute_async(...)` in an event loop.

Add a read-only SQL stage with `t.sql(query)`, using the local alias `input`.
Use `cf.sql(query, orders=orders, fees=fees)` for explicit named table declarations.
SQL returns another `TableExpr`, so `pipe`, column operators, and table methods
compose before and after it. `expression.pipe(function, *args, **kwargs)` calls
an ordinary synchronous function once while building the calculation.

For streams, use `async with output.stream(batches()) as results`, then
`async for table in results`. One native job retains state across batches and
owns cleanup. Declared event-time inputs default to validated nondecreasing
timestamps and watermarks that produce finalized results before EOF. Use
`watermarks` for explicit disorder or source-provided progress; see
[watermark policies](docs/streaming-guide.md#watermark-policies).
A `Program` yields named `StreamOutput` events for multiple
outputs. The convenience stream uses temporary checkpoints and best-effort
iterable delivery; durable recovery uses explicit source/sink bindings and a
managed checkpoint root.

Continue with [SQL and reusable pipelines](docs/batch-guide.md#compose-sql-and-python-pipelines)
and the [first streaming pipeline](docs/streaming-guide.md#first-python-continuous-job).
The runnable examples cover [SQL composition](examples/19_sql_expression_pipeline.py),
[stateful streaming](examples/20_streaming_pipeline.py), and
[named streaming outputs](examples/21_streaming_outputs.py).
[Expression workflows](docs/symbolic-workflows.md) cover financial features,
recovery, static matrices, and joins. Use explicit `Runtime`, plans, `Batch`,
UDFs, and `PipelineBuilder` for runtime integrations and diagnostics.

The [Rust runtime reference](docs/rust-api.md) and
[native examples](crates/calc-flow/examples/README.md) cover implementation and
extension work. The `calc-flow` crate's exports remain available to those users.

## Architecture

```text
crates/calc-flow  (Rust core: Batch, graph compiler, DataFusion, runners, stores)
  ├─ crates/calc-flow-connectors  (trusted transport implementations)
  └─ crates/calc-flow-python  (PyO3 _native binding + registered connectors)
       └─ python/calc_flow  (Python expressions, Arrow execution + integrations)
            └─ web-ui/backend  (calc-flow-studio FastAPI, /api/v3, loopback only)
                  └─ web-ui/src  (React + TypeScript + Vite + React Flow studio, via REST)
```

The native dependency edges are `crates/calc-flow ← calc-flow-connectors` and
`crates/calc-flow ← crates/calc-flow-python ← python/calc_flow ← web-ui/backend`.
The frontend talks to the backend over the `/api/v3` REST contract only; the
Python package is not a second engine.

| Path                           | Purpose                                                                                                                                  |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `crates/calc-flow/`            | Native core: batches, ports/operators, graph compiler, DataFusion runtime, UDF/provider registries, runners, checkpoints, project stores |
| `crates/calc-flow-connectors/` | Trusted file, Kafka, PostgreSQL, MySQL, ClickHouse, HTTP, and WebSocket connectors behind feature gates                                  |
| `crates/calc-flow-python/`     | PyO3 binding exposing the core as `calc_flow._native`                                                                                    |
| `python/calc_flow/`            | Python expressions and SQL, `pipe`, Arrow collection, owned stream results, lowering, and runtime integrations                           |
| `web-ui/backend/`              | `calc-flow-studio` FastAPI service under `/api/v3`, loopback-bound, spawned bounded continuous-job workers                               |
| `web-ui/src/`                  | React + TypeScript + Vite + React Flow studio; API types generated from `web-ui/openapi.json`                                            |
| `schemas/`                     | `project-v3.schema.json`, the canonical generated project contract                                                                       |
| `examples/`                    | Executable Python expression and integration examples                                                                                    |
| `benchmarks/`                  | pytest-benchmark harness (informational)                                                                                                 |

## Data and execution model

- Table data is Arrow-backed. The Rust runtime executes row expressions and SQL
  with DataFusion and owns native rolling, window, and cross-section operators.
- NumPy and JAX are optional Python array providers. They are registered
  explicitly and evaluate a bounded, allowlisted expression language.
- Raw tables or arrays never cross a graph or runner boundary; they are wrapped
  in immutable `Batch` envelopes.
- Project documents are strict, data-only JSON/YAML with
  `format_version: 3`. They select batch or stream runtime mode explicitly;
  stream documents reference registered connectors and named secrets without
  embedding credentials, callables, import paths, or table backend selectors.
- Table and mixed graph runs own one run-scoped DataFusion session. External-only
  NumPy/JAX runs own no DataFusion configuration, UDF state, or runtime and
  return an empty DataFusion metrics list.
- Convenience batch calls return Arrow tables. Explicit plan execution returns
  named `Batch` outputs, per-node timings, metadata, and DataFusion diagnostics.
- Python executions accept reusable frozen `ExecutionOptions` with
  deep-copied strict-JSON settings and a cooperative, timezone-aware deadline
  normalized to UTC.
- `TableExpr.stream` and `Program.stream` own a single native job with bounded
  backpressure. SQL accepts one alias in a stream and runs per native batch;
  SQL aggregation, sorting, and limits do not span batches.
- The source-driven `StreamingRunner` consumes a `StreamExecutionPlan`, owns
  async source/sink bindings, and returns a one-owner `StreamingJob`.
- Managed epoch checkpoints use `LocalStateBackend` segments and strict v3
  `CheckpointManifest` documents. Exactly-once compatibility is proved per
  requested output; ordinary sinks can provide at-least-once delivery on a
  lossless replayable route. Async iterable convenience inputs provide best effort.

The capabilities and execution model are introduced in
[docs/introduction.md](docs/introduction.md). The complete component and
lifecycle design is in [docs/design.md](docs/design.md), and the practical
continuous tutorial is [docs/streaming-guide.md](docs/streaming-guide.md).

## Trusted extensions

Python applications may register trusted vectorized DataFusion scalar UDFs on
a `Runtime`. Every registration declares provider, name, version, exact Arrow
input types, return type, and volatility. Graph nodes select registrations
explicitly with `(provider, name, version)` references. Serialized projects
contain references only.

Runtime extension authors use `UdfRegistry` for native DataFusion UDFs and
`ProviderRegistry` for explicitly registered external operators.

## Studio

`web-ui/backend/` is the independently packaged `calc-flow-studio` FastAPI
service. `web-ui/` is the React, TypeScript, Vite, and React Flow client.
The local service:

- exposes the v3 REST API under `/api/v3`;
- binds only to loopback and is intentionally single-user;
- validates and stores v3 project documents;
- runs bounded continuous jobs in spawned workers;
- serves generated frontend assets from the Studio wheel.

Start both development processes on macOS, Linux, or WSL:

```bash
./web-ui/scripts/start_web_ui.sh
```

On native Windows PowerShell:

```powershell
.\web-ui\scripts\start_web_ui.ps1
```

Open `http://127.0.0.1:5173`, then stop the managed processes with the
matching command for your platform:

```bash
./web-ui/scripts/stop_web_ui.sh
```

```powershell
.\web-ui\scripts\stop_web_ui.ps1
```

Both launchers keep logs and process state under `.calc-flow-web/`.

The checked OpenAPI contract is
[web-ui/openapi.json](web-ui/openapi.json); generated TypeScript request and
response types are in `web-ui/src/api/schema.d.ts`.

## Project contracts

Calc Flow 4.0 accepts strict project-v3 documents and exposes the Studio
`/api/v3` surface. Read [projects and persistence](docs/projects-guide.md)
for validation, serialization, and reloading a graph. Historical changes are
recorded in [CHANGELOG.md](CHANGELOG.md).

## Development

Large Cargo and Maturin outputs should use the repository `target/` tree.
The complete CI/full-verification command reference is below. Local changes use
the smallest affected checks under [AGENTS.md Verification](AGENTS.md#verification);
full regression, Rust 90% coverage, Studio backend 85% coverage, and routine
performance gates belong to CI:

```bash
uv sync --extra dev
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
uv run python scripts/run_rust_tests.py
CALC_FLOW_CONNECTOR_CONTAINERS=1 \
  CALC_FLOW_KAFKA_BOOTSTRAP=localhost:9092 \
  CALC_FLOW_PG_TEST_URL=postgresql://postgres:postgres@localhost:5432/postgres \
  CALC_FLOW_MYSQL_TEST_URL=mysql://root:calcflow-test@localhost:3306/calcflow \
  CH_TEST_URL=http://localhost:8123 \
  uv run python scripts/run_rust_coverage.py
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps

uv run maturin develop
JAX_PLATFORMS=cpu uv run pytest python/tests -q
JAX_PLATFORMS=cpu uv run python scripts/run_examples.py
uv run ruff check .
uv run ruff format --check .

cd web-ui/backend
uv run --project . --extra dev pytest --cov=calc_flow_studio

cd ..
npm ci
npm run sync:api
npm run build
npm test
npm run test:e2e
npm audit --omit=dev
```

Release gates also run `cargo audit`, `cargo deny --locked check`, package
inspectors, isolated wheel smoke tests, `cargo package`, and
`cargo publish --dry-run`. See [AGENTS.md](AGENTS.md) for the maintained
repository commands and constraints.

## Documentation

- **[Documentation index](docs/README.md)** — reading order for all published docs
- **[Introduction](docs/introduction.md)** — capabilities, vocabulary, and execution modes
- **[getting started](docs/getting-started.md)** — installation and smoke test
- **[Executable examples](docs/examples.md)** — verified example matrix and runner
- **[Batch calculations](docs/batch-guide.md)** — expressions, SQL, UDFs, and async execution
- **[Arrays and matrices](docs/array-guide.md)** — NumPy/JAX and static weights
- **[Continuous streaming](docs/streaming-guide.md)** — source-to-recovery tutorial
- **[Connectors](docs/connectors/README.md)** — transport configuration and guarantees
- **[Projects and persistence](docs/projects-guide.md)** — JSON/YAML and file stores
- **[Studio](docs/studio-guide.md)** — local editing, inspection, and job controls
- **[Python API](docs/python-api.md)** — Python surface and examples
- **[Expression workflows](docs/symbolic-workflows.md)** — declaration-to-Studio
  batch, stream, recovery, static matrix, and performance workflows
- **[Rust runtime reference](docs/rust-api.md)** — native surface and examples
- **[API reference](docs/api-reference.md)** — supported surfaces at a glance
- **[Expression API](docs/symbolic-api.md)** — declarations, analysis, and compile requirements
- **[Design and architecture](docs/design.md)** — component ownership and execution design
- **[Verification](docs/verification.md)** — documentation, examples, and implementation checks
- **[Python release guide](docs/python-release.md)** — packaging, verification,
  Trusted Publishers, and the PyPI procedure
- **[Benchmark suite](docs/benchmark-suite.md)** — complete CI tables, scale matrices,
  external-engine comparisons and historical regression evidence
- **[Changelog](CHANGELOG.md)** — the single history of changes

## License

Apache-2.0 — see [LICENSE](LICENSE).

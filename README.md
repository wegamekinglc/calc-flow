# Calc Flow

[![CI](https://github.com/wegamekinglc/calc-flow/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/wegamekinglc/calc-flow/actions/workflows/ci.yml)

Calc Flow 2.0 is a Rust-native calculation engine for immutable Arrow
micro-batches and stateful streams. The core crate compiles typed calculation
graphs, runs every table expression and query with Apache DataFusion, and owns
checkpoint/recovery semantics. The Python package is a PyO3 binding to that
engine; it is not a second implementation. Calc Flow Studio remains a separate
local FastAPI and React application.

## Install

Python 3.13 or newer:

```bash
uv add calc-flow
```

Optional array providers:

```bash
uv add "calc-flow[numpy]"
uv add "calc-flow[jax]"
```

Rust:

```toml
[dependencies]
calc-flow = "2.0.0"
```

## Python quickstart

```python
import pyarrow as pa

from calc_flow import Batch, PipelineBuilder

plan = (
    PipelineBuilder("totals")
    .expression("calculate", "total = quantity * unit_price")
    .compile()
)
result = plan.execute(
    {
        "input": Batch.from_pyarrow(
            pa.table({"quantity": [2, 3], "unit_price": [10, 4]})
        )
    }
)

assert result.outputs["output"].to_pyarrow()["total"].to_pylist() == [20, 12]
```

`PipelineBuilder` is functional: every method returns a new builder and leaves
its input unchanged. Unconnected input ports become graph inputs; unconnected
output ports become graph outputs. Use `execute_async()` inside an event loop.

See [the Python API guide](docs/python-api.md) and the executable
[examples](examples/README.md) for SQL, Python scalar UDFs, micro-batch
recovery, asyncio, and NumPy.

## Rust quickstart

The Rust crate exposes the native data, operator, graph, runtime, project, and
checkpoint types directly. A table `Batch` contains one or more Arrow
`RecordBatch` values plus immutable metadata. Build a graph with
`PipelineBuilder`, compile it against a `UdfRegistrySnapshot`, then await
`ExecutionPlan::execute`.

Run the checked examples:

```bash
cargo run -p calc-flow --example expression_pipeline
cargo run -p calc-flow --example micro_batch_recovery
```

See [the Rust API guide](docs/rust-api.md) for paired source examples and links
to the public types.

## Architecture

```text
crates/calc-flow  (Rust core: Batch, graph compiler, DataFusion, runners, stores)
  └─ crates/calc-flow-python  (PyO3 _native binding)
       └─ python/calc_flow  (pure-Python public API + functional adapters)
            └─ web-ui/backend  (calc-flow-studio FastAPI, /api/v2, loopback only)
                  └─ web-ui/src  (React + TypeScript + Vite + React Flow studio, via REST)
```

The native dependency edge is
`crates/calc-flow ← crates/calc-flow-python ← python/calc_flow ← web-ui/backend`.
The frontend talks to the backend over the `/api/v2` REST contract only; the
Python package is not a second engine.

| Path                       | Purpose                                                                                                                                  |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `crates/calc-flow/`        | Native core: batches, ports/operators, graph compiler, DataFusion runtime, UDF/provider registries, runners, checkpoints, project stores |
| `crates/calc-flow-python/` | PyO3 binding exposing the core as `calc_flow._native`                                                                                    |
| `python/calc_flow/`        | Pure-Python public API, functional `PipelineBuilder`, runner/store adapters, NumPy/JAX provider registration, exception hierarchy        |
| `web-ui/backend/`          | `calc-flow-studio` FastAPI service under `/api/v2`, loopback-bound, spawned bounded preview workers                                      |
| `web-ui/src/`              | React + TypeScript + Vite + React Flow studio; API types generated from `web-ui/openapi.json`                                            |
| `schemas/`                 | `project-v2.schema.json`, the canonical generated project contract                                                                       |
| `examples/`                | Executable v2 Python examples                                                                                                            |
| `benchmarks/`              | pytest-benchmark harness (informational)                                                                                                 |

## Data and execution model

- Table data is Arrow-backed and calculated only by DataFusion.
- NumPy and JAX are optional Python array providers. They are registered
  explicitly and evaluate a bounded, allowlisted expression language.
- Raw tables or arrays never cross a graph or runner boundary; they are wrapped
  in immutable `Batch` envelopes.
- Project documents are strict, data-only JSON/YAML with
  `format_version: 2`. They contain no callable source, import path, or table
  backend selector.
- Table and mixed graph runs own one run-scoped DataFusion session. External-only
  NumPy/JAX runs own no DataFusion configuration, UDF state, or runtime and
  return an empty DataFusion metrics list.
- Every graph run returns named outputs, per-node row counts/timings, and run
  metadata; table work additionally reports DataFusion plans and timings.
- Micro-batch and streaming runners deliver sinks before committing
  checkpoints. Delivery is at least once, and failed delivery restores owned
  in-memory state.

The canonical architecture is described in
[docs/introduction.md](docs/introduction.md).

## Trusted extensions

Python applications may register trusted vectorized DataFusion scalar UDFs on
a `Runtime`. Every registration declares provider, name, version, exact Arrow
input types, return type, and volatility. Graph nodes select registrations
explicitly with `(provider, name, version)` references. Serialized projects
contain references only.

Rust applications use `UdfRegistry` for native DataFusion UDFs and
`ProviderRegistry` for explicitly registered external operators.

## Studio

`web-ui/backend/` is the independently packaged `calc-flow-studio` FastAPI
service. `web-ui/` is the React, TypeScript, Vite, and React Flow client.
The local service:

- exposes the v2 REST API under `/api/v2`;
- binds only to loopback and is intentionally single-user;
- validates and stores v2 project documents;
- runs bounded previews in spawned workers;
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

## Compatibility

Calc Flow 2.0 is a clean break from the frozen Python v1 implementation. It
does not load v1 projects or checkpoints. See
[the v2 release guide](docs/v2-release.md) before upgrading. Historical v1
behavior remains available at the `v1-python-final` tag and as immutable
semantic fixtures under `tests/fixtures/v1/`.

## Development

Large Cargo and Maturin outputs should use the repository `target/` tree.
A typical local verification sequence is:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo llvm-cov --workspace --all-features --fail-under-lines 90
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps

uv sync --extra dev
uv run maturin develop
JAX_PLATFORMS=cpu uv run pytest python/tests -q
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
- **[Introduction](docs/introduction.md)** — architecture and data flow
- **[getting started](docs/getting-started.md)** — installation and smoke test
- **[Python API](docs/python-api.md)** — Python surface and examples
- **[Rust API](docs/rust-api.md)** — native surface and examples
- **[API reference](docs/api-reference.md)** — supported surfaces at a glance
- **[Benchmark harness](benchmarks/README.md)** — informational benchmarks
- **[v2 release and migration](docs/v2-release.md)** — v1-to-v2 boundary (history)

## License

Apache-2.0 — see [LICENSE](LICENSE).

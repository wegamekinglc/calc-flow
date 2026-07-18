# Calc Flow

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

Start both development processes:

```bash
./web-ui/scripts/start_web_ui.sh
```

Open `http://127.0.0.1:5173`, then stop them with:

```bash
./web-ui/scripts/stop_web_ui.sh
```

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

## Reference

- [API map](docs/api-reference.md)
- [Python API](docs/python-api.md)
- [Rust API](docs/rust-api.md)
- [v2 release and migration](docs/v2-release.md)
- [benchmark harness](benchmarks/README.md)
- [historical v1 final API](docs/v1-final-api.md)
- [historical v0.2 migration notes](docs/migration-v0.2.md)

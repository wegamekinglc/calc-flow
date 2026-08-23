# CLAUDE.md

This file is operational guidance for Claude Code (claude.ai/code) working in
this repository. Keep public project orientation in [README.md](README.md) and
published architecture docs under [docs/](docs/).

[AGENTS.md](AGENTS.md) is the authoritative agent guide for this repo: the
maintained command list, coding style, architecture summary, test layout, and
release invariants live there. When this file and AGENTS.md disagree, AGENTS.md
wins. Calc Flow is its own repository; follow the git conventions in AGENTS.md
(branch naming, imperative summaries under 72 chars, prefixed PR titles under
70 chars, `## Summary` / `## Test plan` PR bodies). Do not add tool-attribution
trailers to commits unless explicitly requested.

## What this is

Calc Flow 4.0 is a Rust-native micro-batch and streaming calculation engine.
The `calc-flow` crate owns immutable `Batch` values, graph compilation,
DataFusion execution, project validation, checkpointing, and runner semantics.
The Python package under `python/calc_flow/` is a PyO3 binding plus functional
adapters; it is not a second engine. `calc-flow-studio` is a separate local
FastAPI and React application. There is no `src/calc_flow/` pure-Python
implementation in 2.0; the frozen v1 implementation is preserved at the
`v1-python-final` tag. See [docs/introduction.md](docs/introduction.md) for the
data contract and execution model.

## Commands

The canonical command groups live in [AGENTS.md](AGENTS.md#commands) and are
reproduced here for convenience; repeat them exactly.

```bash
# Rust core and PyO3 Rust unit tests
uv sync --extra dev
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
uv run python scripts/run_rust_tests.py
CALC_FLOW_CONNECTOR_CONTAINERS=1 \
  CALC_FLOW_KAFKA_BOOTSTRAP=localhost:9092 \
  CALC_FLOW_PG_TEST_URL=postgresql://postgres:postgres@localhost:5432/postgres \
  CH_TEST_URL=http://localhost:8123 \
  uv run python scripts/run_rust_coverage.py
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps

# PyO3 package and Python adapters
uv run maturin develop
JAX_PLATFORMS=cpu uv run pytest python/tests -q
uv run ruff check .
uv run ruff format --check .
uv run ruff format .

# Studio backend
cd web-ui/backend
uv run --project . --extra dev pytest --cov=calc_flow_studio

# Studio frontend and generated API
cd web-ui
npm ci
npm run sync:api
npm run build
npm test
npm run test:e2e
npm audit --omit=dev

# Supply chain and release helpers
cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177 --ignore RUSTSEC-2026-0235
cargo deny --locked check
python -m unittest scripts.test_run_examples scripts.test_run_rust_tests scripts.test_run_rust_coverage scripts.test_inspect_wheel scripts.test_release_config
```

Run informational benchmarks with:

```bash
CALC_FLOW_BENCHMARK_SCALE=overhead \
  JAX_PLATFORMS=cpu \
  uv run --extra benchmark pytest benchmarks --benchmark-only
```

Use `./web-ui/scripts/start_web_ui.sh` and
`./web-ui/scripts/stop_web_ui.sh` for the managed local Studio.

Keep Cargo, Maturin, uv, coverage, and release outputs beneath the repository
`target/` tree when working from a constrained mirror, and never leave a
generated `python/calc_flow/_native*.so` in source.

## Running tests

There is no single top-level `pytest` invocation that covers the project; each
surface has its own runner.

- **Rust core** — unit tests live beside source under
  `crates/calc-flow/src/`; integration tests live under
  `crates/calc-flow/tests/`. The `calc-flow-python` PyO3 binding has no
  separate Rust integration-test directory — its behavior is covered by inline
  unit tests and the Python suite under `python/tests/`. Run `uv sync --extra
  dev`, then use `uv run python scripts/run_rust_tests.py` to test the core
  targets normally and the compiled PyO3 test executable serially with its
  runtime-only timeout. Enforce the 90% line floor with
  `scripts/run_rust_coverage.py` while the documented connector containers are
  available; it combines the ordinary workspace and real connector paths in
  one llvm-cov profile set.
- **Python binding and adapters** — `python/tests/`, run with
  `JAX_PLATFORMS=cpu uv run pytest python/tests -q` after
  `uv run maturin develop`. Define fixtures in focused test modules; there is no
  shared `conftest.py`.
- **Studio backend** — `web-ui/backend/tests/`, an independent suite with its
  own 85% coverage floor (`pytest --cov=calc_flow_studio`).
- **Studio frontend** — Vitest unit tests (`npm test`) and Playwright browser
  workflows (`npm run test:e2e`).
- **`tests/fixtures/v1/`** is immutable historical v1 parity evidence, not a v2
  runtime or package path. Leave it unchanged.

Every behavior change starts with a focused failing test that records the
expected failure before implementation.

## Code style

The full conventions are in [AGENTS.md](AGENTS.md#coding-style) and
[`.claude/rules/code-style.md`](.claude/rules/code-style.md). Highlights:

- **Rust** — Rust 2024, workspace MSRV 1.88.0, `unsafe_code = "forbid"`. Prefer
  immutable values, pure transformations, explicit error variants, and
  deterministic `BTreeMap` ordering where output is serialized or observed.
  Return `calc_flow::Result<T>` from fallible public core operations and
  preserve the failing field/path and source error. Own I/O, cancellation,
  stores, sources, sinks, and runner lifecycles in async code; never block
  Tokio executor threads. Add rustdoc and examples for new public APIs and keep
  `RUSTDOCFLAGS="-D warnings"` green.
- **Python** — Python 3.13+, four spaces, double quotes, retain
  `from __future__ import annotations`. Use built-in type syntax
  (`list[str]`, `dict[str, object]`, `A | B`). Prefer functions and frozen/slot
  containers; add a class only for identity, protocol, lifecycle, resource
  ownership, or explicitly stateful behavior. Never mutate caller-owned
  mappings, sequences, Arrow tables, arrays, or JSON values. Keep pure CPU-local
  transforms synchronous; blocking convenience methods must reject a running
  event loop. Do not add compatibility shims for the removed Python v1
  implementation.
- **Web** — apply the same functional-first and input-immutability rules to
  FastAPI and TypeScript. Use asynchronous FastAPI I/O and thread/process
  boundaries for blocking or CPU work. Update React state immutably with
  functional updates, clean up streams/timers/listeners/requests/child
  processes, and keep React Flow node-type maps outside render functions.
- **Markdown tables** — align columns with pipes and pad separator rows so
  dashes span the full column width, including cell spaces.

## Architecture

```text
crates/calc-flow  (Rust core: Batch, graph compiler, DataFusion, runners, stores)
  └─ crates/calc-flow-python  (PyO3 _native binding)
       └─ python/calc_flow  (pure-Python public API + functional adapters)
            └─ web-ui/backend  (calc-flow-studio FastAPI, /api/v3, loopback only)
                  └─ web-ui/src  (React + TypeScript + Vite + React Flow studio, via REST)
```

The native dependency edge is
`crates/calc-flow ← crates/calc-flow-python ← python/calc_flow ← web-ui/backend`.
The frontend talks to the backend over the `/api/v3` REST contract only.

| Path                       | Purpose                                                                                                                                  |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `crates/calc-flow/`        | Native core: batches, ports/operators, graph compiler, DataFusion runtime, UDF/provider registries, runners, checkpoints, project stores |
| `crates/calc-flow-python/` | PyO3 binding exposing the core as `calc_flow._native`                                                                                    |
| `python/calc_flow/`        | Pure-Python public API, functional `PipelineBuilder`, runner/store adapters, NumPy/JAX provider registration, exception hierarchy        |
| `web-ui/backend/`          | `calc-flow-studio` FastAPI service under `/api/v3`, loopback-bound, spawned bounded continuous-job workers                               |
| `web-ui/src/`              | React + TypeScript + Vite + React Flow studio; API types generated from `web-ui/openapi.json`                                            |
| `schemas/`                 | `project-v3.schema.json`, the canonical generated project contract                                                                       |
| `examples/`                | Executable v3 Python examples                                                                                                            |
| `benchmarks/`              | pytest-benchmark harness (informational)                                                                                                 |
| `docs/`                    | Published documentation                                                                                                                  |

### Core invariants

- `Batch` is the public graph/runner data envelope. Table batches contain Arrow
  record batches; external batches contain an explicitly registered provider
  payload. Raw tables and arrays never cross a graph, plan, or runner boundary.
- The stream message and context types (`StreamMessage`, `EventTime`, `Epoch`,
  `StreamJobContext`) expose no runner control-injection API; control messages
  are constructed only through crate-private constructors. The crate-root
  public `StreamingRunner` consumes a `StreamExecutionPlan` and returns an
  owning `StreamingJob` with whole-job preflight, bounded source/operator/sink
  tasks, job-scoped event-time progress, aligned epoch checkpoints, managed v3
  manifest recovery for durable restart, per-output delivery proof, and
  deterministic metrics. Control construction, task/coordinator/supervisor
  internals, and reaper ownership remain crate-private. The full contract is
  documented in the [stream message envelope](docs/runtime-envelope.md).
- Apache DataFusion 54 is the sole table engine. Table operations accept one
  expression/projection/filter node or one read-only `SELECT`/CTE SQL node.
  DDL, DML, utility statements, multiple statements, and table backend
  selectors are rejected. Each table or mixed run owns one run-scoped
  DataFusion session; external-only runs own no DataFusion configuration, UDF
  state, or runtime.
- `Port` declares name, `BatchKind`, required flag, and optional exact Arrow
  schema. `OperatorMetadata` owns shared graph metadata; custom finite and
  continuous operators implement `BatchOperator` and `StreamOperator`
  respectively. `ExpressionOperator` and `SqlOperator` implement both;
  `UnionOperator` and `WindowAggregateOperator` are stream-only. External
  operators resolve through lifecycle-specific factories in
  `ProviderRegistry`.
- `PipelineBuilder` consumes immutable graph-building steps. `compile_batch()`
  and `compile_stream()` validate endpoints, kinds, schemas, one-writer inputs,
  UDFs, cycles, deterministic topology, inputs/outputs, and fingerprint.
- `UdfRegistry` owns trusted native implementations;
  `UdfRegistrySnapshot` is captured at compile time; configurations carry only
  `UdfReference` values — never source, callables, or import paths.
- Continuous ownership and compatibility follow the canonical [Rust core
  guidance](AGENTS.md#rust-core): the crate-root facade is public and uses
  managed v3 recovery, while control, coordination, and reaping stay internal.
  The old v2 runners and public checkpoint store are removed with no
  compatibility layer.
- The canonical project format is a strict data-only `ProjectDocument` with
  `format_version: 3`. The Rust `ProjectSpec`, generated JSON Schema, Python
  `ProjectDocument`, FastAPI request models, and generated TypeScript contract
  all describe the same structure.

### Local Studio

`web-ui/backend/` is the independently packaged `calc-flow-studio` FastAPI
service. Its routes live under `/api/v3`; by default `serve()` rejects
non-loopback hosts, and changing that security boundary requires a separately
reviewed exception. `RunManager` owns spawned continuous-job workers with
concurrency, resident-memory, checkpoint-disk, cancellation, shutdown, and
lifecycle controls. `web-ui/` is React, TypeScript, Vite, and
React Flow; API types are generated from `web-ui/openapi.json` via
`npm run sync:api`, which also writes `web-ui/src/api/schema.d.ts`. Regenerate
both after any route, model, or version change.

## Documentation

Published docs live under [docs/](docs/) with a reading-order index in
[docs/README.md](docs/README.md). Start with
[docs/introduction.md](docs/introduction.md) for the architecture and data
flow. Keep normative docs describing the current/latest state only; record
fundamental changes in `CHANGELOG.md` and leave the historical release records
(`docs/v1-final-api.md`, `docs/v2-release.md`, `docs/migration-v0.2.md`)
untouched.

## Specialist agents

The canonical calc-flow agent team (spec → design → critique → implement →
test → review → document) is defined in `.codex/agents/`.
`.claude/agents/` mirrors those definitions for Claude compatibility;
synchronize team changes from `.codex/agents/` to `.claude/agents/`, never in
the reverse direction. `cf-doc-writer` owns the freshness of `docs/` and
curates `CHANGELOG.md`; invoke it when docs need reconciling against current
code or a change may warrant a changelog entry.

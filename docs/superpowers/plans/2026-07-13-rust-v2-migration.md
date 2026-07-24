# Calc Flow Rust V2 Migration Implementation Plan

> **Historical status:** The design, plan, and handoff were merged in PR #12
> before implementation began. The Rust v2 implementation was completed and
> merged in PR #13. Unchecked boxes preserve the original execution plan; they
> are not current pending work.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Calc Flow's Python core with a Rust core that has native Rust and Rust-backed Python APIs while retaining the Python/FastAPI Studio backend.

**Architecture:** Build all engine behavior in one `calc-flow` Rust crate with focused internal modules. After the Rust parity gate passes, add a thin PyO3 extension and Python facade for PyArrow, NumPy, and JAX, then switch the existing FastAPI workers and `/api/v2` Studio contract to that facade.

**Tech Stack:** Rust 1.88.0, Rust 2024 edition, Apache DataFusion 54.0.0, Arrow Rust 58.3.0, Tokio 1.52, Serde, Schemars, PyO3 0.28.3, pyo3-async-runtimes 0.28.0, pyo3-arrow 0.17.0, Maturin 1.14.1, Python 3.13+, PyArrow 24+, FastAPI, React, Vitest, and Playwright.

## Global Constraints

- Treat `docs/superpowers/specs/2026-07-13-rust-v2-migration-design.md` as the approved source of truth.
- Freeze Python v1 immediately; do not add features or compatibility shims to it.
- V2 is breaking: do not migrate v1 imports, project documents, fingerprints, checkpoints, or serialized state.
- Keep all canonical batches, validation, graph compilation, execution, state, DataFusion behavior, stores, and runners in Rust.
- Keep Apache DataFusion as the only table query and calculation engine.
- Keep NumPy and JAX as Python-hosted providers; a Rust-only host must reject graphs that require them.
- Keep the Studio backend in Python with FastAPI and spawned worker processes.
- Target Python 3.13 or newer.
- Target Linux x86_64/aarch64, macOS x86_64/arm64, and Windows x86_64 wheels.
- Use Rust 1.88.0 as the MSRV because DataFusion 54.0.0 requires it.
- Pin DataFusion to 54.0.0 and Arrow crates to 58.3.0.
- Pin PyO3 to 0.28.3, pyo3-async-runtimes to 0.28.0, and pyo3-arrow to 0.17.0 so the Python binding uses Arrow 58 throughout.
- Require 90% line coverage for `calc-flow` Rust core code and 85% for `calc_flow_studio` Python code.
- Treat all inputs as read-only. Return new values and confine mutation to owned stateful operators, runners, stores, and lifecycle managers.
- Reject executable objects, Python source, and import paths in project configuration.
- End every task with the focused tests, formatting/lint checks, and an intentional commit.

## Verified Version References

- DataFusion 54.0.0 uses Rust 1.88.0 and Arrow 58.3.0: <https://docs.rs/crate/datafusion/54.0.0/source/Cargo.toml>
- `pyo3-arrow` 0.17.x is the compatible bridge for PyO3 0.28 and Arrow 58: <https://docs.rs/pyo3-arrow/0.17.0/pyo3_arrow/>
- Maturin's mixed layout supports `python-source` and a private native submodule: <https://www.maturin.rs/project_layout>
- Rust 2024 edition is stable and virtual workspaces should select resolver 3: <https://doc.rust-lang.org/stable/edition-guide/rust-2024/cargo-resolver.html>

---

## Target File Map

### Rust core

- `Cargo.toml` — workspace, shared versions, profiles, and lint policy.
- `rust-toolchain.toml` — pinned MSRV toolchain and required components.
- `crates/calc-flow/Cargo.toml` — core crate features and dependencies.
- `crates/calc-flow/src/lib.rs` — curated Rust public API.
- `crates/calc-flow/src/error.rs` — typed errors and `Result` alias.
- `crates/calc-flow/src/json.rs` — JSON validation and canonical serialization.
- `crates/calc-flow/src/batch.rs` — immutable metadata, table batches, and opaque extension payloads.
- `crates/calc-flow/src/context.rs` — cancellation, deadlines, settings, and run identity.
- `crates/calc-flow/src/expression.rs` — assignment splitting and SQL projection.
- `crates/calc-flow/src/datafusion.rs` — query validation, per-run session, expressions, SQL, and metrics.
- `crates/calc-flow/src/udf.rs` — versioned native and external UDF references and registry snapshots.
- `crates/calc-flow/src/operator.rs` — ports, operator traits, built-in operators, and external provider contract.
- `crates/calc-flow/src/pipeline.rs` — graph builder, compilation, fingerprints, execution, and rollback.
- `crates/calc-flow/src/config.rs` — strict v2 project specification, schema generation, and compilation.
- `crates/calc-flow/src/checkpoint.rs` — v2 checkpoint value, validation, and store trait.
- `crates/calc-flow/src/project_store.rs` — canonical JSON/YAML and atomic project storage.
- `crates/calc-flow/src/io.rs` — async source, sink, source item, and batching adapter.
- `crates/calc-flow/src/runtime/mod.rs` — runtime exports and sink routing helpers.
- `crates/calc-flow/src/runtime/micro_batch.rs` — recovery, delivery, and periodic checkpoints.
- `crates/calc-flow/src/runtime/streaming.rs` — one-batch streaming steps and recovery.
- `crates/calc-flow/tests/*.rs` — public integration tests mirroring each source module.
- `crates/calc-flow/tests/support/mod.rs` — integration-test-only fixtures imported with `mod support;`.
- `crates/calc-flow/benches/*.rs` — Criterion benchmarks for compile and run overhead.

### Python binding and facade

- `crates/calc-flow-python/Cargo.toml` — PyO3 `cdylib` crate.
- `crates/calc-flow-python/src/lib.rs` — `_native` module registration.
- `crates/calc-flow-python/src/error.rs` — Rust-to-Python exception mapping.
- `crates/calc-flow-python/src/batch.rs` — PyArrow and opaque Python payload conversion.
- `crates/calc-flow-python/src/pipeline.rs` — builder, execution plan, result, sync, and async bindings.
- `crates/calc-flow-python/src/provider.rs` — Python callback provider implementation.
- `crates/calc-flow-python/src/config.rs` — project validation and JSON-schema binding functions.
- `python/calc_flow/__init__.py` — supported Python public API.
- `python/calc_flow/_native.pyi` — native-module type declarations.
- `python/calc_flow/array.py` — NumPy/JAX provider registration and safe expression evaluation.
- `python/calc_flow/config.py` — Pydantic-compatible Rust project wrapper.
- `python/calc_flow/errors.py` — documented Python exception exports.
- `python/calc_flow/py.typed` — PEP 561 marker.
- `python/tests/*.py` — binding, facade, array, async, and packaging tests.

### Studio, release, and documentation

- `web-ui/backend/src/calc_flow_studio/models.py` — `/api/v2` request and response models.
- `web-ui/backend/src/calc_flow_studio/run_manager.py` — bounded input conversion and Rust-backed worker execution.
- `web-ui/backend/src/calc_flow_studio/app.py` — `/api/v2` routes and Rust-owned schema exposure.
- `web-ui/openapi.json` and `web-ui/src/api/schema.d.ts` — generated v2 contracts.
- `web-ui/src/types.ts`, `web-ui/src/App.tsx`, and affected components — v2 field names and payloads.
- `.github/workflows/ci.yml` — Rust, Python binding, Studio, frontend, e2e, and package gates.
- `.github/workflows/release.yml` — multi-platform Rust and Python artifacts.
- `.github/workflows/benchmarks.yml` — Rust and Python boundary benchmarks.
- `README.md`, `docs/introduction.md`, `docs/api-reference.md`, and `examples/` — v2 user documentation.

---

### Task 1: Freeze Python V1 and Capture the Semantic Corpus

**Files:**
- Create: `docs/v1-final-api.md`
- Create: `scripts/export_v1_contract_fixtures.py`
- Create: `tests/fixtures/v1/manifest.json`
- Create: `tests/fixtures/v1/expression.arrow`
- Create: `tests/fixtures/v1/sql_left.arrow`
- Create: `tests/fixtures/v1/sql_right.arrow`
- Create: `tests/test_v1_contract_fixtures.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: the frozen Python v1 package under `src/calc_flow`.
- Produces: a committed `manifest.json` whose fixture entries have `input`, `operation`, `expected`, and `invariants` fields; Arrow IPC files used by Rust acceptance tests.

- [ ] **Step 1: Write the failing fixture-integrity test**

```python
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.ipc as ipc


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "v1"


def test_v1_contract_manifest_references_readable_arrow_files() -> None:
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text())
    assert manifest["format_version"] == 1
    assert {case["name"] for case in manifest["cases"]} == {
        "expression_assignment",
        "sql_join",
        "empty_table",
        "metadata_round_trip",
        "state_rollback",
    }
    for relative_path in manifest["arrow_files"]:
        with ipc.open_file(FIXTURE_DIR / relative_path) as reader:
            assert reader.schema is not None
```

- [ ] **Step 2: Run the integrity test and confirm it fails**

Run: `uv run pytest tests/test_v1_contract_fixtures.py -q`

Expected: FAIL because `tests/fixtures/v1/manifest.json` does not exist.

- [ ] **Step 3: Add the deterministic fixture exporter and manifest**

```python
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

from calc_flow import Batch, ExpressionOperator, Pipeline, SqlOperator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "fixtures" / "v1"


def write_table(name: str, table: pa.Table) -> str:
    path = OUTPUT / name
    with path.open("wb") as stream:
        with ipc.new_file(stream, table.schema) as writer:
            writer.write_table(table)
    return name


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    expression_input = pa.table({"a": [1, 3], "b": [2, 4]})
    expression_output = (
        Pipeline("fixture-expression")
        .then(ExpressionOperator("calculate", "total = a + b"))
        .execute({"input": Batch.table(expression_input)})
        .output.table_payload
    )
    left = pa.table({"id": [1, 2], "amount": [10, 20]})
    right = pa.table({"id": [2, 1], "rate": [3, 4]})
    sql_output = (
        Pipeline("fixture-sql")
        .then(
            SqlOperator(
                "join",
                "SELECT l.id, l.amount * r.rate AS total "
                "FROM l JOIN r ON l.id = r.id",
                aliases=("l", "r"),
            )
        )
        .execute({"l": Batch.table(left), "r": Batch.table(right)})
        .output.table_payload
    )
    arrow_files = [
        write_table("expression.arrow", expression_input),
        write_table("expression_expected.arrow", expression_output),
        write_table("sql_left.arrow", left),
        write_table("sql_right.arrow", right),
        write_table("sql_expected.arrow", sql_output),
        write_table("empty.arrow", pa.table({"value": pa.array([], type=pa.int64())})),
    ]
    manifest = {
        "format_version": 1,
        "arrow_files": arrow_files,
        "cases": [
            {"name": "expression_assignment", "input": "expression.arrow", "operation": "total = a + b", "expected": "expression_expected.arrow", "invariants": ["table_only", "metadata_preserved"]},
            {"name": "sql_join", "input": ["sql_left.arrow", "sql_right.arrow"], "operation": "join", "expected": "sql_expected.arrow", "invariants": ["single_select"]},
            {"name": "empty_table", "input": "empty.arrow", "operation": "identity", "expected": "empty.arrow", "invariants": ["schema_preserved"]},
            {"name": "metadata_round_trip", "input": "expression.arrow", "operation": "identity", "expected": "expression.arrow", "invariants": ["deeply_immutable_json"]},
            {"name": "state_rollback", "input": "expression.arrow", "operation": "fail_after_state", "expected": {"state": {}}, "invariants": ["rollback"]},
        ],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
```

Run: `uv run python scripts/export_v1_contract_fixtures.py`

Document every symbol currently exported by `src/calc_flow/__init__.py` in `docs/v1-final-api.md`, grouped by batch, graph, operator, runtime, configuration, storage, and UDF responsibilities. Mark the document as a behavioral reference, not a v2 compatibility promise. Add the same freeze statement to `README.md`.

- [ ] **Step 4: Verify the corpus and frozen v1 suite**

Run: `uv run pytest tests/test_v1_contract_fixtures.py tests/calc_flow -q`

Expected: PASS with all existing v1 tests unchanged.

- [ ] **Step 5: Commit and tag the frozen reference**

```bash
git add README.md docs/v1-final-api.md scripts/export_v1_contract_fixtures.py tests/fixtures/v1 tests/test_v1_contract_fixtures.py
git commit -m "test: freeze Python v1 behavior"
git tag v1-python-final
```

### Task 2: Establish the Rust Workspace and Quality Gates

**Files:**
- Create: `Cargo.toml`
- Create: `Cargo.lock`
- Create: `rust-toolchain.toml`
- Create: `crates/calc-flow/Cargo.toml`
- Create: `crates/calc-flow/src/lib.rs`
- Create: `crates/calc-flow/tests/workspace.rs`
- Modify: `.gitignore`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: no Rust code.
- Produces: workspace package `calc-flow` at version `2.0.0-alpha.1`, Rust 1.88.0, resolver 3, and CI commands used by every later task.

- [ ] **Step 1: Write the failing workspace smoke test**

```rust
#[test]
fn crate_reports_v2_version() {
    assert_eq!(calc_flow::VERSION, "2.0.0-alpha.1");
}
```

- [ ] **Step 2: Confirm the workspace does not exist**

Run: `cargo test -p calc-flow --test workspace`

Expected: FAIL because the root `Cargo.toml` does not exist.

- [ ] **Step 3: Add the pinned workspace manifests**

```toml
# Cargo.toml
[workspace]
members = ["crates/calc-flow"]
resolver = "3"

[workspace.package]
version = "2.0.0-alpha.1"
edition = "2024"
rust-version = "1.88.0"
license = "Apache-2.0"
repository = "https://github.com/wegamekinglc/calc-flow"

[workspace.dependencies]
async-trait = "0.1.89"
chrono = { version = "0.4.44", features = ["serde"] }
datafusion = { version = "54.0.0", default-features = false, features = ["datetime_expressions", "regex_expressions", "sql", "unicode_expressions"] }
futures = "0.3.31"
hex = "0.4.3"
parking_lot = "0.12.5"
proptest = "1.7.0"
regex = "1.12.2"
schemars = { version = "1.0.4", features = ["chrono04"] }
serde = { version = "1.0.228", features = ["derive"] }
serde-saphyr = { version = "0.0.29", default-features = false, features = ["deserialize", "serialize"] }
serde_json = "1.0.145"
sha2 = "0.10.9"
tempfile = "3.23.0"
thiserror = "2.0.17"
tokio = { version = "1.52.0", features = ["fs", "macros", "rt-multi-thread", "sync", "time"] }
tokio-util = "0.7.17"
tracing = "0.1.41"
uuid = { version = "1.18.1", features = ["serde", "v4"] }

[workspace.lints.rust]
unsafe_code = "forbid"
unused_qualifications = "deny"

[workspace.lints.clippy]
all = { level = "deny", priority = -1 }
pedantic = { level = "deny", priority = -1 }
module_name_repetitions = "allow"
must_use_candidate = "allow"
```

```toml
# rust-toolchain.toml
[toolchain]
channel = "1.88.0"
profile = "minimal"
components = ["clippy", "llvm-tools-preview", "rustfmt"]
```

```toml
# crates/calc-flow/Cargo.toml
[package]
name = "calc-flow"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
description = "Rust core for stateful DataFusion calculation pipelines"

[dependencies]
async-trait.workspace = true
chrono.workspace = true
datafusion.workspace = true
futures.workspace = true
hex.workspace = true
parking_lot.workspace = true
regex.workspace = true
schemars.workspace = true
serde.workspace = true
serde-saphyr.workspace = true
serde_json.workspace = true
sha2.workspace = true
tempfile.workspace = true
thiserror.workspace = true
tokio.workspace = true
tokio-util.workspace = true
tracing.workspace = true
uuid.workspace = true

[dev-dependencies]
proptest.workspace = true

[lints]
workspace = true
```

```rust
//! Calc Flow's Rust-native v2 calculation engine.

/// The crate version used by project and package diagnostics.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
```

```gitignore
/target/
/.superpowers/
```

```yaml
# Append to .github/workflows/ci.yml
  rust-core:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: dtolnay/rust-toolchain@1.88.0
        with:
          components: clippy,rustfmt
      - run: cargo fmt --all --check
      - run: cargo clippy --workspace --all-targets --all-features -- -D warnings
      - run: cargo test --workspace
      - run: cargo doc --workspace --no-deps
```

- [ ] **Step 4: Generate the lockfile and verify all quality commands**

Run: `cargo generate-lockfile`

Run: `cargo fmt --all --check`

Run: `cargo clippy --workspace --all-targets --all-features -- -D warnings`

Run: `cargo test --workspace`

Expected: every command exits 0 and `crate_reports_v2_version` passes.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock rust-toolchain.toml crates/calc-flow .gitignore .github/workflows/ci.yml
git commit -m "chore: establish Rust workspace"
```

### Task 3: Add Typed Errors, Canonical JSON, and Run Context

**Files:**
- Create: `crates/calc-flow/src/error.rs`
- Create: `crates/calc-flow/src/json.rs`
- Create: `crates/calc-flow/src/context.rs`
- Create: `crates/calc-flow/tests/context.rs`
- Modify: `crates/calc-flow/src/lib.rs`

**Interfaces:**
- Consumes: Tokio, Serde JSON, Chrono, UUID.
- Produces: `CalcFlowError`, `Result<T>`, `JsonMap`, `canonical_json`, `CancellationToken`, and `RunContext`.

- [ ] **Step 1: Write failing context and canonicalization tests**

```rust
use std::collections::BTreeMap;

use calc_flow::{CalcFlowError, CancellationToken, RunContext, canonical_json};
use serde_json::json;

#[test]
fn canonical_json_sorts_mapping_keys() {
    assert_eq!(canonical_json(&json!({"z": 1, "a": 2})).unwrap(), "{\"a\":2,\"z\":1}");
}

#[tokio::test]
async fn node_context_shares_cancellation() {
    let token = CancellationToken::new();
    let context = RunContext::new(BTreeMap::new(), None, token.clone()).unwrap();
    let node = context.for_node("calculate").unwrap();
    token.cancel();
    assert!(matches!(node.check_cancelled(), Err(CalcFlowError::Cancelled { .. })));
}
```

- [ ] **Step 2: Run and confirm missing exports**

Run: `cargo test -p calc-flow --test context`

Expected: FAIL with unresolved imports from `calc_flow`.

- [ ] **Step 3: Implement the shared primitives**

```rust
// src/error.rs
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CalcFlowError {
    #[error("invalid {field}: {message}")]
    InvalidArgument { field: String, message: String },
    #[error("project format version {found} is unsupported; expected {expected}")]
    UnsupportedVersion { expected: u32, found: u32 },
    #[error("graph compilation failed: {message}")]
    Compile { message: String },
    #[error("node {node_id} failed: {message}")]
    Operator { node_id: String, message: String },
    #[error("DataFusion failed for node {node_id:?}: {message}")]
    DataFusion { node_id: Option<String>, message: String },
    #[error("external provider {provider}:{name}@{version} failed: {message}")]
    ExternalProvider { provider: String, name: String, version: String, message: String },
    #[error("run {run_id} was cancelled")]
    Cancelled { run_id: String },
    #[error("checkpoint mismatch: {message}")]
    CheckpointMismatch { message: String },
    #[error("stored document is invalid: {message}")]
    Format { message: String },
    #[error("I/O failed for {path}: {source}")]
    Io { path: String, #[source] source: std::io::Error },
    #[error("internal invariant failed: {message}")]
    Internal { message: String },
}

pub type Result<T> = std::result::Result<T, CalcFlowError>;
```

```rust
// src/json.rs
use std::collections::BTreeMap;

use serde_json::Value;

use crate::{CalcFlowError, Result};

pub type JsonMap = BTreeMap<String, Value>;

pub fn canonical_json(value: &Value) -> Result<String> {
    serde_json::to_string(&sort_value(value)).map_err(|error| CalcFlowError::Format {
        message: error.to_string(),
    })
}

fn sort_value(value: &Value) -> Value {
    match value {
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), sort_value(value)))
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(sort_value).collect()),
        scalar => scalar.clone(),
    }
}
```

```rust
// src/context.rs
use std::{collections::BTreeMap, sync::Arc};

use chrono::{DateTime, Utc};
use serde_json::Value;
use tokio_util::sync::CancellationToken as TokioCancellationToken;
use uuid::Uuid;

use crate::{CalcFlowError, JsonMap, Result};

#[derive(Clone, Debug, Default)]
pub struct CancellationToken(TokioCancellationToken);

impl CancellationToken {
    pub fn new() -> Self { Self(TokioCancellationToken::new()) }
    pub fn cancel(&self) { self.0.cancel(); }
    pub fn is_cancelled(&self) -> bool { self.0.is_cancelled() }
}

#[derive(Clone, Debug)]
pub struct RunContext {
    run_id: Arc<str>,
    node_id: Option<Arc<str>>,
    settings: Arc<JsonMap>,
    deadline: Option<DateTime<Utc>>,
    cancellation: CancellationToken,
}

impl RunContext {
    pub fn new(settings: BTreeMap<String, Value>, deadline: Option<DateTime<Utc>>, cancellation: CancellationToken) -> Result<Self> {
        if deadline.is_some_and(|value| value.timezone() != Utc) {
            return Err(CalcFlowError::InvalidArgument { field: "deadline".into(), message: "must use UTC".into() });
        }
        Ok(Self { run_id: Uuid::new_v4().to_string().into(), node_id: None, settings: Arc::new(settings), deadline, cancellation })
    }

    pub fn for_node(&self, node_id: &str) -> Result<Self> {
        if node_id.trim().is_empty() {
            return Err(CalcFlowError::InvalidArgument { field: "node_id".into(), message: "must not be empty".into() });
        }
        let mut context = self.clone();
        context.node_id = Some(node_id.into());
        Ok(context)
    }

    pub fn check_cancelled(&self) -> Result<()> {
        if self.cancellation.is_cancelled() || self.deadline.is_some_and(|deadline| Utc::now() >= deadline) {
            return Err(CalcFlowError::Cancelled { run_id: self.run_id.to_string() });
        }
        Ok(())
    }

    pub fn run_id(&self) -> &str { &self.run_id }
    pub fn node_id(&self) -> Option<&str> { self.node_id.as_deref() }
    pub fn settings(&self) -> &JsonMap { &self.settings }
}
```

Re-export the six public names from `lib.rs` and keep helper functions private.

- [ ] **Step 4: Run focused and workspace checks**

Run: `cargo test -p calc-flow --test context && cargo clippy -p calc-flow --all-targets -- -D warnings`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/calc-flow/src crates/calc-flow/tests/context.rs
git commit -m "feat: add Rust run primitives"
```

### Task 4: Implement Immutable Batch Envelopes

**Files:**
- Create: `crates/calc-flow/src/batch.rs`
- Create: `crates/calc-flow/tests/batch.rs`
- Modify: `crates/calc-flow/src/lib.rs`

**Interfaces:**
- Consumes: `JsonMap`, Arrow 58.3 types re-exported by DataFusion.
- Produces: `BatchKind`, `BatchMetadata`, `TableBatch`, `ExternalPayload`, and `Batch`.

- [ ] **Step 1: Write failing table, metadata, and extension tests**

```rust
use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{Batch, BatchKind, BatchMetadata, ExternalPayload};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
use serde_json::json;

#[derive(Debug)]
struct TestArray;

impl ExternalPayload for TestArray {
    fn backend(&self) -> &str { "test" }
    fn len(&self) -> usize { 2 }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

#[test]
fn table_batch_preserves_metadata_and_rows() {
    let record = RecordBatch::try_from_iter(vec![("value", Arc::new(Int64Array::from(vec![1, 2])) as _)]).unwrap();
    let metadata = BatchMetadata::new("source", 7, BTreeMap::from([("nested".into(), json!({"ok": true}))])).unwrap();
    let batch = Batch::table(vec![record], metadata.clone()).unwrap();
    assert_eq!(batch.kind(), BatchKind::Table);
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.metadata(), &metadata);
}

#[test]
fn external_batch_is_owned_by_arc() {
    let batch = Batch::external(Arc::new(TestArray), BatchMetadata::default()).unwrap();
    assert_eq!(batch.kind(), BatchKind::Array);
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.external_payload().unwrap().backend(), "test");
}
```

- [ ] **Step 2: Confirm the batch API is missing**

Run: `cargo test -p calc-flow --test batch`

Expected: FAIL with unresolved batch imports.

- [ ] **Step 3: Implement the immutable batch types**

```rust
use std::{any::Any, collections::BTreeMap, fmt::Debug, sync::Arc};

use datafusion::arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{CalcFlowError, JsonMap, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BatchKind { Table, Array }

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BatchMetadata {
    source: String,
    sequence: u64,
    attributes: JsonMap,
}

impl Default for BatchMetadata {
    fn default() -> Self { Self { source: String::new(), sequence: 0, attributes: BTreeMap::new() } }
}

impl BatchMetadata {
    pub fn new(source: impl Into<String>, sequence: u64, attributes: BTreeMap<String, Value>) -> Result<Self> {
        let source = source.into();
        if source.contains('\0') { return Err(CalcFlowError::InvalidArgument { field: "metadata.source".into(), message: "must not contain NUL".into() }); }
        Ok(Self { source, sequence, attributes })
    }
    pub fn source(&self) -> &str { &self.source }
    pub fn sequence(&self) -> u64 { self.sequence }
    pub fn attributes(&self) -> &JsonMap { &self.attributes }
}

#[derive(Clone, Debug)]
pub struct TableBatch { schema: SchemaRef, batches: Arc<[RecordBatch]>, rows: usize }

impl TableBatch {
    fn new(batches: Vec<RecordBatch>) -> Result<Self> {
        let schema = batches.first().map(RecordBatch::schema).ok_or_else(|| CalcFlowError::InvalidArgument { field: "batches".into(), message: "must contain at least one RecordBatch; represent an empty table with one zero-row batch".into() })?;
        if batches.iter().any(|batch| batch.schema() != schema) { return Err(CalcFlowError::InvalidArgument { field: "batches".into(), message: "schemas must match".into() }); }
        let rows = batches.iter().map(RecordBatch::num_rows).sum();
        Ok(Self { schema, batches: batches.into(), rows })
    }
    pub fn schema(&self) -> &SchemaRef { &self.schema }
    pub fn batches(&self) -> &[RecordBatch] { &self.batches }
}

pub trait ExternalPayload: Any + Debug + Send + Sync {
    fn backend(&self) -> &str;
    fn len(&self) -> usize;
    fn as_any(&self) -> &dyn Any;
}

#[derive(Clone, Debug)]
enum BatchPayload { Table(TableBatch), External(Arc<dyn ExternalPayload>) }

#[derive(Clone, Debug)]
pub struct Batch { payload: BatchPayload, metadata: BatchMetadata }

impl Batch {
    pub fn table(batches: Vec<RecordBatch>, metadata: BatchMetadata) -> Result<Self> { Ok(Self { payload: BatchPayload::Table(TableBatch::new(batches)?), metadata }) }
    pub fn external(payload: Arc<dyn ExternalPayload>, metadata: BatchMetadata) -> Result<Self> { if payload.backend().is_empty() { return Err(CalcFlowError::InvalidArgument { field: "backend".into(), message: "must not be empty".into() }); } Ok(Self { payload: BatchPayload::External(payload), metadata }) }
    pub fn kind(&self) -> BatchKind { match self.payload { BatchPayload::Table(_) => BatchKind::Table, BatchPayload::External(_) => BatchKind::Array } }
    pub fn num_rows(&self) -> usize { match &self.payload { BatchPayload::Table(table) => table.rows, BatchPayload::External(payload) => payload.len() } }
    pub fn metadata(&self) -> &BatchMetadata { &self.metadata }
    pub fn table_payload(&self) -> Result<&TableBatch> { match &self.payload { BatchPayload::Table(table) => Ok(table), BatchPayload::External(_) => Err(CalcFlowError::InvalidArgument { field: "batch".into(), message: "expected table batch".into() }) } }
    pub fn external_payload(&self) -> Result<&Arc<dyn ExternalPayload>> { match &self.payload { BatchPayload::External(payload) => Ok(payload), BatchPayload::Table(_) => Err(CalcFlowError::InvalidArgument { field: "batch".into(), message: "expected array batch".into() }) } }
    pub fn with_metadata(&self, metadata: BatchMetadata) -> Self { Self { payload: self.payload.clone(), metadata } }
}
```

- [ ] **Step 4: Run batch tests and formatting**

Run: `cargo test -p calc-flow --test batch && cargo fmt --all --check`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/calc-flow/src/batch.rs crates/calc-flow/src/lib.rs crates/calc-flow/tests/batch.rs
git commit -m "feat: add immutable Rust batches"
```

### Task 5: Port Assignment Parsing and SQL Safety Validation

**Files:**
- Create: `crates/calc-flow/src/expression.rs`
- Create: `crates/calc-flow/tests/expression.rs`
- Modify: `crates/calc-flow/src/lib.rs`

**Interfaces:**
- Consumes: `CalcFlowError`, regex, DataFusion's SQL parser.
- Produces: `split_assignment(&str) -> Option<(&str, &str)>`, `sql_projection(&str, &str) -> Result<String>`, and `validate_select_query(&str) -> Result<String>`.

- [ ] **Step 1: Write failing expression and SQL tests**

```rust
use calc_flow::{split_assignment, sql_projection, validate_select_query};

#[test]
fn assignment_ignores_comparisons() {
    assert_eq!(split_assignment("total = a + b"), Some(("total", "a + b")));
    assert_eq!(split_assignment("a == b"), None);
    assert_eq!(split_assignment("a >= b"), None);
}

#[test]
fn projection_and_query_validation_are_restricted() {
    assert_eq!(sql_projection("total = a + b", "input").unwrap(), "SELECT *, (a + b) AS total FROM input");
    assert!(validate_select_query("WITH x AS (SELECT 1) SELECT * FROM x").is_ok());
    assert!(validate_select_query("DROP TABLE input").is_err());
    assert!(validate_select_query("SELECT 1; SELECT 2").is_err());
}
```

- [ ] **Step 2: Confirm the functions are absent**

Run: `cargo test -p calc-flow --test expression`

Expected: FAIL with unresolved imports.

- [ ] **Step 3: Implement parsing with a single-query AST check**

```rust
use datafusion::sql::{parser::DFParser, sqlparser::dialect::GenericDialect};
use regex::Regex;
use std::sync::OnceLock;

use crate::{CalcFlowError, Result};

pub fn split_assignment(expression: &str) -> Option<(&str, &str)> {
    static ASSIGNMENT: OnceLock<Regex> = OnceLock::new();
    let regex = ASSIGNMENT.get_or_init(|| Regex::new(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^=].*)$").expect("constant regex is valid"));
    let captures = regex.captures(expression)?;
    if expression.contains("==") || expression.contains("!=") || expression.contains("<=") || expression.contains(">=") { return None; }
    Some((captures.get(1)?.as_str(), captures.get(2)?.as_str().trim()))
}

pub fn sql_projection(expression: &str, table_name: &str) -> Result<String> {
    if !is_identifier(table_name) { return Err(CalcFlowError::InvalidArgument { field: "table_name".into(), message: "must be a SQL identifier".into() }); }
    Ok(match split_assignment(expression) {
        Some((name, value)) => format!("SELECT *, ({value}) AS {name} FROM {table_name}"),
        None => format!("SELECT ({}) AS result FROM {table_name}", expression.trim()),
    })
}

pub fn validate_select_query(query: &str) -> Result<String> {
    let statements = DFParser::parse_sql_with_dialect(query, &GenericDialect {}).map_err(|error| CalcFlowError::InvalidArgument { field: "query".into(), message: error.to_string() })?;
    if statements.len() != 1 || !matches!(statements.first(), Some(datafusion::sql::parser::Statement::Statement(statement)) if matches!(statement.as_ref(), datafusion::sql::sqlparser::ast::Statement::Query(_))) {
        return Err(CalcFlowError::InvalidArgument { field: "query".into(), message: "exactly one SELECT or CTE query is required".into() });
    }
    Ok(query.trim().trim_end_matches(';').trim().to_owned())
}

fn is_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    chars.next().is_some_and(|ch| ch == '_' || ch.is_ascii_alphabetic()) && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}
```

- [ ] **Step 4: Run focused tests and compare the v1 fixture expressions**

Run: `cargo test -p calc-flow --test expression`

Expected: PASS, including comparison-operator regressions.

- [ ] **Step 5: Commit**

```bash
git add crates/calc-flow/src/expression.rs crates/calc-flow/src/lib.rs crates/calc-flow/tests/expression.rs
git commit -m "feat: port expression validation"
```

### Task 6: Implement the Per-Run DataFusion Runtime

**Files:**
- Create: `crates/calc-flow/src/datafusion.rs`
- Create: `crates/calc-flow/tests/datafusion.rs`
- Modify: `crates/calc-flow/src/lib.rs`

**Interfaces:**
- Consumes: `Batch`, `BatchMetadata`, `sql_projection`, `validate_select_query`, DataFusion 54.0.0.
- Produces: `DataFusionConfig`, `DataFusionQueryMetric`, and async `DataFusionRuntime::{evaluate, sql, close}`.

- [ ] **Step 1: Write failing DataFusion acceptance tests**

```rust
use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{Batch, BatchMetadata, DataFusionConfig, DataFusionRuntime};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

fn input(values: Vec<i64>) -> Batch {
    let record = RecordBatch::try_from_iter(vec![("a", Arc::new(Int64Array::from(values)) as _)]).unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

#[tokio::test]
async fn runtime_evaluates_assignment_and_collects_metrics() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let output = runtime.evaluate("double = a * 2", &input(vec![1, 3]), None).await.unwrap();
    assert_eq!(output.num_rows(), 2);
    let metrics = runtime.metrics();
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].output_rows, 2);
    assert!(metrics[0].logical_plan.contains("double"));
}

#[tokio::test]
async fn runtime_joins_inputs_and_rejects_mutation_sql() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let tables = BTreeMap::from([("left".into(), input(vec![1, 2])), ("right".into(), input(vec![1, 2]))]);
    assert!(runtime.sql("SELECT l.a FROM left l JOIN right r ON l.a = r.a", &tables, None).await.is_ok());
    assert!(runtime.sql("DELETE FROM left", &tables, None).await.is_err());
}
```

- [ ] **Step 2: Verify the runtime is missing**

Run: `cargo test -p calc-flow --test datafusion`

Expected: FAIL with unresolved DataFusion runtime imports.

- [ ] **Step 3: Implement run-scoped registration, cleanup, and metrics**

```rust
use std::{collections::BTreeMap, sync::atomic::{AtomicBool, AtomicU64, Ordering}, time::Instant};

use datafusion::{dataframe::DataFrame, datasource::MemTable, execution::context::{SessionConfig, SessionContext}};
use parking_lot::Mutex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{sql_projection, validate_select_query, Batch, BatchMetadata, CalcFlowError, Result};

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct DataFusionConfig {
    pub batch_size: usize,
    pub target_partitions: usize,
}

impl Default for DataFusionConfig {
    fn default() -> Self { Self { batch_size: 8192, target_partitions: 1 } }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataFusionQueryMetric {
    pub query_id: u64,
    pub node_id: Option<String>,
    pub planning_ns: u64,
    pub execution_ns: u64,
    pub output_rows: usize,
    pub logical_plan: String,
    pub physical_plan: String,
}

pub struct DataFusionRuntime {
    context: SessionContext,
    metrics: Mutex<Vec<DataFusionQueryMetric>>,
    next_query: AtomicU64,
    closed: AtomicBool,
}

impl DataFusionRuntime {
    pub fn new(config: DataFusionConfig) -> Result<Self> {
        if config.batch_size == 0 || config.target_partitions == 0 { return Err(CalcFlowError::InvalidArgument { field: "datafusion".into(), message: "batch_size and target_partitions must be positive".into() }); }
        let session = SessionConfig::new().with_batch_size(config.batch_size).with_target_partitions(config.target_partitions);
        Ok(Self { context: SessionContext::new_with_config(session), metrics: Mutex::new(Vec::new()), next_query: AtomicU64::new(1), closed: AtomicBool::new(false) })
    }

    pub async fn evaluate(&self, expression: &str, input: &Batch, node_id: Option<&str>) -> Result<Batch> {
        let query = sql_projection(expression, "input")?;
        let tables = BTreeMap::from([("input".to_owned(), input.clone())]);
        self.sql(&query, &tables, node_id).await
    }

    pub async fn sql(&self, query: &str, tables: &BTreeMap<String, Batch>, node_id: Option<&str>) -> Result<Batch> {
        self.ensure_open()?;
        if tables.is_empty() { return Err(CalcFlowError::InvalidArgument { field: "tables".into(), message: "must not be empty".into() }); }
        let query = validate_select_query(query)?;
        let aliases = self.register_tables(tables)?;
        let planning_start = Instant::now();
        let dataframe = self.context.sql(&query).await.map_err(|error| self.error(node_id, error))?;
        let logical_plan = dataframe.logical_plan().display_indent_schema().to_string();
        let physical_plan = dataframe.clone().create_physical_plan().await.map_err(|error| self.error(node_id, error))?.display_indent().to_string();
        let planning_ns = nanos(planning_start.elapsed());
        let execution_start = Instant::now();
        let batches = dataframe.collect().await.map_err(|error| self.error(node_id, error));
        for alias in aliases { let _ = self.context.deregister_table(&alias); }
        let batches = batches?;
        let output_rows = batches.iter().map(datafusion::arrow::record_batch::RecordBatch::num_rows).sum();
        self.metrics.lock().push(DataFusionQueryMetric { query_id: self.next_query.fetch_add(1, Ordering::Relaxed), node_id: node_id.map(str::to_owned), planning_ns, execution_ns: nanos(execution_start.elapsed()), output_rows, logical_plan, physical_plan });
        Batch::table(batches, merged_metadata(tables))
    }

    pub fn metrics(&self) -> Vec<DataFusionQueryMetric> { self.metrics.lock().clone() }
    pub fn close(&self) { self.closed.store(true, Ordering::Release); }

    fn register_tables(&self, tables: &BTreeMap<String, Batch>) -> Result<Vec<String>> {
        let mut aliases = Vec::with_capacity(tables.len());
        for (alias, batch) in tables {
            let table = batch.table_payload()?;
            let provider = MemTable::try_new(table.schema().clone(), vec![table.batches().to_vec()]).map_err(|error| self.error(None, error))?;
            self.context.register_table(alias, std::sync::Arc::new(provider)).map_err(|error| self.error(None, error))?;
            aliases.push(alias.clone());
        }
        Ok(aliases)
    }

    fn ensure_open(&self) -> Result<()> { if self.closed.load(Ordering::Acquire) { Err(CalcFlowError::InvalidArgument { field: "runtime".into(), message: "is closed".into() }) } else { Ok(()) } }
    fn error(&self, node_id: Option<&str>, error: impl std::fmt::Display) -> CalcFlowError { CalcFlowError::DataFusion { node_id: node_id.map(str::to_owned), message: error.to_string() } }
}

fn merged_metadata(tables: &BTreeMap<String, Batch>) -> BatchMetadata { if tables.len() == 1 { tables.values().next().expect("length checked").metadata().clone() } else { BatchMetadata::default() } }
fn nanos(duration: std::time::Duration) -> u64 { u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX) }
```

Add a guard in `sql` so every error path after registration deregisters all aliases. Use a small private cleanup guard whose `Drop` implementation schedules only synchronous `deregister_table` calls; test a failed query followed by successful reuse of the same aliases.

- [ ] **Step 4: Run DataFusion tests and the v1 Arrow fixture comparison**

Run: `cargo test -p calc-flow --test datafusion -- --nocapture`

Expected: PASS; the expression and SQL fixture outputs equal the committed IPC expectations.

- [ ] **Step 5: Commit**

```bash
git add crates/calc-flow/src/datafusion.rs crates/calc-flow/src/lib.rs crates/calc-flow/tests/datafusion.rs
git commit -m "feat: add Rust DataFusion runtime"
```

### Task 7: Implement Versioned Rust UDF Registration

**Files:**
- Create: `crates/calc-flow/src/udf.rs`
- Create: `crates/calc-flow/tests/udf.rs`
- Modify: `crates/calc-flow/src/datafusion.rs`
- Modify: `crates/calc-flow/src/lib.rs`

**Interfaces:**
- Consumes: DataFusion `ScalarUDF` values and canonical JSON metadata.
- Produces: `UdfKind`, `UdfReference`, `UdfCatalogEntry`, `UdfRegistry`, and immutable `UdfRegistrySnapshot`.

- [ ] **Step 1: Write failing registry snapshot and conflict tests**

```rust
use calc_flow::{UdfKind, UdfReference, UdfRegistry};

#[test]
fn snapshot_is_immutable_and_catalog_is_data_only() {
    let mut registry = UdfRegistry::new();
    registry.register_external(UdfReference::new("python", "normalize", "1", UdfKind::ExternalScalar).unwrap(), 1).unwrap();
    let snapshot = registry.snapshot();
    registry.register_external(UdfReference::new("numpy", "clip", "1", UdfKind::ExternalArray).unwrap(), 3).unwrap();
    assert_eq!(snapshot.catalog().len(), 1);
    assert_eq!(snapshot.catalog()[0].name, "normalize");
}

#[test]
fn conflicting_versions_of_a_datafusion_name_are_rejected() {
    let references = [
        UdfReference::new("rust", "score", "1", UdfKind::DataFusionScalar).unwrap(),
        UdfReference::new("rust", "score", "2", UdfKind::DataFusionScalar).unwrap(),
    ];
    assert!(calc_flow::validate_selected_udfs(&references).is_err());
}
```

- [ ] **Step 2: Confirm UDF types are absent**

Run: `cargo test -p calc-flow --test udf`

Expected: FAIL with unresolved UDF imports.

- [ ] **Step 3: Add reference-based native and external registrations**

```rust
use std::{collections::BTreeMap, sync::Arc};

use datafusion::logical_expr::ScalarUDF;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{CalcFlowError, Result};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum UdfKind { DataFusionScalar, ExternalScalar, ExternalArray }

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize, JsonSchema)]
pub struct UdfReference { pub provider: String, pub name: String, pub version: String, pub kind: UdfKind }

impl UdfReference {
    pub fn new(provider: &str, name: &str, version: &str, kind: UdfKind) -> Result<Self> {
        for (field, value) in [("provider", provider), ("name", name), ("version", version)] {
            if value.is_empty() || !value.chars().all(|ch| ch == '-' || ch == '_' || ch == '.' || ch.is_ascii_alphanumeric()) { return Err(CalcFlowError::InvalidArgument { field: field.into(), message: "must be a non-empty portable identifier".into() }); }
        }
        Ok(Self { provider: provider.into(), name: name.into(), version: version.into(), kind })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UdfCatalogEntry { pub provider: String, pub name: String, pub version: String, pub kind: UdfKind, pub argument_count: usize }

#[derive(Default)]
pub struct UdfRegistry { native: BTreeMap<UdfReference, Arc<ScalarUDF>>, catalog: BTreeMap<UdfReference, UdfCatalogEntry> }

#[derive(Clone, Default)]
pub struct UdfRegistrySnapshot { native: Arc<BTreeMap<UdfReference, Arc<ScalarUDF>>>, catalog: Arc<Vec<UdfCatalogEntry>> }

impl UdfRegistry {
    pub fn new() -> Self { Self::default() }
    pub fn register_datafusion(&mut self, reference: UdfReference, udf: Arc<ScalarUDF>, argument_count: usize) -> Result<()> { if reference.kind != UdfKind::DataFusionScalar || udf.name() != reference.name { return Err(CalcFlowError::InvalidArgument { field: "udf".into(), message: "reference kind and DataFusion name must match".into() }); } self.insert_catalog(reference.clone(), argument_count)?; self.native.insert(reference, udf); Ok(()) }
    pub fn register_external(&mut self, reference: UdfReference, argument_count: usize) -> Result<()> { if reference.kind == UdfKind::DataFusionScalar { return Err(CalcFlowError::InvalidArgument { field: "udf.kind".into(), message: "external registration requires an external kind".into() }); } self.insert_catalog(reference, argument_count) }
    pub fn snapshot(&self) -> UdfRegistrySnapshot { UdfRegistrySnapshot { native: Arc::new(self.native.clone()), catalog: Arc::new(self.catalog.values().cloned().collect()) } }
    fn insert_catalog(&mut self, reference: UdfReference, argument_count: usize) -> Result<()> { if self.catalog.contains_key(&reference) { return Err(CalcFlowError::InvalidArgument { field: "udf".into(), message: "duplicate provider/name/version/kind".into() }); } self.catalog.insert(reference.clone(), UdfCatalogEntry { provider: reference.provider.clone(), name: reference.name.clone(), version: reference.version.clone(), kind: reference.kind, argument_count }); Ok(()) }
}

impl UdfRegistrySnapshot {
    pub fn resolve_native(&self, reference: &UdfReference) -> Result<Arc<ScalarUDF>> { self.native.get(reference).cloned().ok_or_else(|| CalcFlowError::Compile { message: format!("unknown UDF {}:{}@{}", reference.provider, reference.name, reference.version) }) }
    pub fn catalog(&self) -> &[UdfCatalogEntry] { &self.catalog }
}

pub fn validate_selected_udfs(references: &[UdfReference]) -> Result<()> {
    let mut versions = BTreeMap::new();
    for reference in references.iter().filter(|reference| reference.kind == UdfKind::DataFusionScalar) {
        if versions.insert(reference.name.clone(), reference.version.clone()).is_some_and(|version| version != reference.version) { return Err(CalcFlowError::Compile { message: format!("conflicting versions selected for {}", reference.name) }); }
    }
    Ok(())
}
```

Add `DataFusionRuntime::register_udfs(&UdfRegistrySnapshot, &[UdfReference])` and call `SessionContext::register_udf` only for selected references.

- [ ] **Step 4: Run registry and DataFusion UDF tests**

Run: `cargo test -p calc-flow --test udf --test datafusion`

Expected: PASS; catalog serialization contains no executable representation.

- [ ] **Step 5: Commit**

```bash
git add crates/calc-flow/src/udf.rs crates/calc-flow/src/datafusion.rs crates/calc-flow/src/lib.rs crates/calc-flow/tests/udf.rs
git commit -m "feat: add versioned Rust UDF registry"
```

### Task 8: Define Ports, Operators, and External Providers

**Files:**
- Create: `crates/calc-flow/src/operator.rs`
- Create: `crates/calc-flow/tests/operator.rs`
- Modify: `crates/calc-flow/src/lib.rs`

**Interfaces:**
- Consumes: batches, run context, DataFusion runtime, UDF references.
- Produces: `Port`, object-safe async `Operator`, `ExpressionOperator`, `SqlOperator`, `ExternalOperatorSpec`, `ExternalOperatorFactory`, and `ProviderRegistry`.

- [ ] **Step 1: Write failing port and built-in operator tests**

```rust
use std::collections::BTreeMap;

use calc_flow::{BatchKind, ExpressionOperator, Port, ProviderRegistry, SqlOperator};

#[test]
fn ports_reject_array_schemas_and_duplicate_names() {
    assert!(Port::new("input", BatchKind::Array, true, Some(vec![])).is_err());
    assert!(ExpressionOperator::new("calc", "b = a + 1", vec![], None, vec![]).is_ok());
}

#[test]
fn sql_aliases_become_required_table_ports() {
    let operator = SqlOperator::new("join", "SELECT * FROM left JOIN right USING(id)", vec!["left".into(), "right".into()], vec![]).unwrap();
    assert_eq!(operator.input_ports().iter().map(|port| port.name()).collect::<Vec<_>>(), ["left", "right"]);
    assert!(ProviderRegistry::default().resolve("numpy", "expression", "1").is_err());
}
```

- [ ] **Step 2: Confirm operator interfaces are absent**

Run: `cargo test -p calc-flow --test operator`

Expected: FAIL with unresolved operator imports.

- [ ] **Step 3: Implement the operator and provider contracts**

```rust
use std::{collections::BTreeMap, sync::Arc};

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use parking_lot::RwLock;
use serde_json::Value;

use crate::{Batch, BatchKind, CalcFlowError, DataFusionRuntime, JsonMap, Result, RunContext, UdfReference};

#[derive(Clone, Debug)]
pub struct Port { name: String, kind: BatchKind, required: bool, schema: Option<SchemaRef> }

impl Port {
    pub fn new(name: &str, kind: BatchKind, required: bool, fields: Option<Vec<datafusion::arrow::datatypes::Field>>) -> Result<Self> { if name.is_empty() { return Err(CalcFlowError::InvalidArgument { field: "port.name".into(), message: "must not be empty".into() }); } if kind == BatchKind::Array && fields.is_some() { return Err(CalcFlowError::InvalidArgument { field: "port.schema".into(), message: "array ports cannot declare Arrow schemas".into() }); } Ok(Self { name: name.into(), kind, required, schema: fields.map(|value| Arc::new(datafusion::arrow::datatypes::Schema::new(value))) }) }
    pub fn name(&self) -> &str { &self.name }
    pub fn kind(&self) -> BatchKind { self.kind }
    pub fn required(&self) -> bool { self.required }
    pub fn schema(&self) -> Option<&SchemaRef> { self.schema.as_ref() }
    pub fn validate(&self, batch: &Batch, endpoint: &str) -> Result<()> { if batch.kind() != self.kind { return Err(CalcFlowError::Compile { message: format!("{endpoint} expects {:?}, received {:?}", self.kind, batch.kind()) }); } if let Some(schema) = &self.schema { if batch.table_payload()?.schema() != schema { return Err(CalcFlowError::Compile { message: format!("{endpoint} schema mismatch") }); } } Ok(()) }
}

pub struct OperatorContext<'a> { pub run: &'a RunContext, pub datafusion: &'a DataFusionRuntime }

#[async_trait]
pub trait Operator: Send + Sync {
    fn name(&self) -> &str;
    fn input_ports(&self) -> &[Port];
    fn output_ports(&self) -> &[Port];
    fn configuration(&self) -> JsonMap;
    fn udf_references(&self) -> Vec<UdfReference> { Vec::new() }
    async fn process(&mut self, inputs: &BTreeMap<String, Batch>, context: &OperatorContext<'_>) -> Result<BTreeMap<String, Batch>>;
    fn snapshot(&self) -> Result<Value> { Ok(Value::Null) }
    fn restore(&mut self, state: &Value) -> Result<()> { if state.is_null() { Ok(()) } else { Err(CalcFlowError::Format { message: "stateless operator state must be null".into() }) } }
    fn reset(&mut self) -> Result<()> { self.restore(&Value::Null) }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExternalOperatorSpec { pub provider: String, pub name: String, pub version: String, pub options: JsonMap }

pub trait ExternalOperatorFactory: Send + Sync { fn create(&self, spec: &ExternalOperatorSpec, inputs: Vec<Port>, outputs: Vec<Port>) -> Result<Box<dyn Operator>>; }

#[derive(Default)]
pub struct ProviderRegistry { factories: RwLock<BTreeMap<(String, String, String), Arc<dyn ExternalOperatorFactory>>> }

impl ProviderRegistry {
    pub fn register(&self, provider: &str, name: &str, version: &str, factory: Arc<dyn ExternalOperatorFactory>) -> Result<()> { let key = (provider.into(), name.into(), version.into()); if self.factories.write().insert(key, factory).is_some() { return Err(CalcFlowError::InvalidArgument { field: "provider".into(), message: "duplicate provider/name/version".into() }); } Ok(()) }
    pub fn resolve(&self, provider: &str, name: &str, version: &str) -> Result<Arc<dyn ExternalOperatorFactory>> { self.factories.read().get(&(provider.into(), name.into(), version.into())).cloned().ok_or_else(|| CalcFlowError::Compile { message: format!("provider {provider}:{name}@{version} is unavailable") }) }
}
```

Implement `ExpressionOperator` and `SqlOperator` as immutable configuration plus `Operator::process` calls to `DataFusionRuntime`. Expression operators have one required `input` table port and one `output` table port; SQL operators have one required table port per alias and one `output` table port. Validate duplicate ports and exactly one expression calculation mode at construction.

- [ ] **Step 4: Run operator and DataFusion tests**

Run: `cargo test -p calc-flow --test operator --test datafusion`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/calc-flow/src/operator.rs crates/calc-flow/src/lib.rs crates/calc-flow/tests/operator.rs
git commit -m "feat: add Rust operator contracts"
```

### Task 9: Build and Compile Pipeline Graphs

**Files:**
- Create: `crates/calc-flow/src/pipeline.rs`
- Create: `crates/calc-flow/tests/pipeline_compile.rs`
- Modify: `crates/calc-flow/src/lib.rs`

**Interfaces:**
- Consumes: `Operator`, `Port`, `ProviderRegistry`, `UdfRegistrySnapshot`, canonical JSON.
- Produces: `PortEndpoint`, `Edge`, `PipelineBuilder`, immutable `ExecutionPlan`, external input/output maps, topological order, and deterministic fingerprint.

- [ ] **Step 1: Write failing graph compilation tests**

```rust
use calc_flow::{Edge, ExpressionOperator, PipelineBuilder, PortEndpoint, UdfRegistry};

#[test]
fn identical_graphs_have_identical_fingerprints() {
    let build = || PipelineBuilder::new("totals").unwrap().add_node("calc", Box::new(ExpressionOperator::new("calc", "b = a + 1", vec![], None, vec![]).unwrap())).unwrap();
    let registry = UdfRegistry::new().snapshot();
    assert_eq!(build().compile(&registry).unwrap().fingerprint(), build().compile(&registry).unwrap().fingerprint());
}

#[test]
fn cycles_and_multiple_input_writers_are_rejected() {
    let edge = Edge::new(PortEndpoint::new("a", "output").unwrap(), PortEndpoint::new("b", "input").unwrap());
    assert!(PipelineBuilder::new("cycle").unwrap().connect(edge.clone()).is_err());
}
```

- [ ] **Step 2: Confirm graph types are missing**

Run: `cargo test -p calc-flow --test pipeline_compile`

Expected: FAIL with unresolved pipeline imports.

- [ ] **Step 3: Implement owned functional graph construction and compilation**

```rust
use std::{collections::{BTreeMap, BTreeSet, VecDeque}, sync::Arc};

use parking_lot::Mutex;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{canonical_json, CalcFlowError, Operator, Port, Result, UdfRegistrySnapshot};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct PortEndpoint { pub node_id: String, pub port: String }
impl PortEndpoint { pub fn new(node_id: &str, port: &str) -> Result<Self> { if node_id.is_empty() || port.is_empty() { return Err(CalcFlowError::InvalidArgument { field: "endpoint".into(), message: "node and port must not be empty".into() }); } Ok(Self { node_id: node_id.into(), port: port.into() }) } }

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Edge { pub source: PortEndpoint, pub target: PortEndpoint }
impl Edge { pub fn new(source: PortEndpoint, target: PortEndpoint) -> Self { Self { source, target } } }

pub struct NodeDefinition { pub node_id: String, pub operator: Box<dyn Operator> }

pub struct PipelineBuilder { name: String, nodes: BTreeMap<String, NodeDefinition>, edges: Vec<Edge> }

pub(crate) struct CompiledNode { pub node_id: String, pub operator: Arc<tokio::sync::Mutex<Box<dyn Operator>>>, pub inbound: BTreeMap<String, PortEndpoint> }

pub struct ExecutionPlan {
    pub(crate) name: String,
    pub(crate) nodes: Vec<CompiledNode>,
    pub(crate) external_inputs: BTreeMap<String, PortEndpoint>,
    pub(crate) external_outputs: BTreeMap<String, PortEndpoint>,
    pub(crate) fingerprint: String,
    pub(crate) run_lock: tokio::sync::Mutex<()>,
    pub(crate) udfs: UdfRegistrySnapshot,
}

impl PipelineBuilder {
    pub fn new(name: &str) -> Result<Self> { if name.is_empty() { return Err(CalcFlowError::InvalidArgument { field: "pipeline.name".into(), message: "must not be empty".into() }); } Ok(Self { name: name.into(), nodes: BTreeMap::new(), edges: Vec::new() }) }
    pub fn add_node(mut self, node_id: &str, operator: Box<dyn Operator>) -> Result<Self> { if self.nodes.contains_key(node_id) { return Err(CalcFlowError::Compile { message: format!("duplicate node {node_id}") }); } self.nodes.insert(node_id.into(), NodeDefinition { node_id: node_id.into(), operator }); Ok(self) }
    pub fn connect(mut self, edge: Edge) -> Result<Self> { if !self.nodes.contains_key(&edge.source.node_id) || !self.nodes.contains_key(&edge.target.node_id) { return Err(CalcFlowError::Compile { message: "edge references an unknown node".into() }); } self.edges.push(edge); Ok(self) }
    pub fn compile(self, udfs: &UdfRegistrySnapshot) -> Result<ExecutionPlan> { validate_edges(&self.nodes, &self.edges)?; let order = topological_order(&self.nodes, &self.edges)?; let fingerprint = graph_fingerprint(&self.name, &self.nodes, &self.edges, udfs)?; build_plan(self, order, fingerprint, udfs.clone()) }
}
```

Implement the named private helpers with these exact rules: validate source and target ports; require compatible kinds and exact schemas; reject more than one writer per input; reject cycles with Kahn's algorithm; assign stable external names from unconnected ports; require at least one external output; serialize node IDs, operator configuration, ports, sorted edges, and selected UDF catalog through `canonical_json`; hash the bytes with SHA-256 and lowercase hex.

- [ ] **Step 4: Run graph validation and property tests**

Run: `cargo test -p calc-flow --test pipeline_compile`

Expected: PASS for deterministic topology, cycle rejection, schema checks, and writer checks.

- [ ] **Step 5: Commit**

```bash
git add crates/calc-flow/src/pipeline.rs crates/calc-flow/src/lib.rs crates/calc-flow/tests/pipeline_compile.rs
git commit -m "feat: compile Rust pipeline graphs"
```

### Task 10: Execute Plans with Metrics and Transactional State

**Files:**
- Modify: `crates/calc-flow/src/pipeline.rs`
- Create: `crates/calc-flow/tests/pipeline_execute.rs`
- Create: `crates/calc-flow/tests/support/mod.rs`

**Interfaces:**
- Consumes: compiled nodes, `RunContext`, `DataFusionRuntime`, immutable inputs.
- Produces: `ExecutionOptions`, `NodeTiming`, `RunMetadata`, `RunResult`, `ExecutionPlan::{execute, snapshot, restore, reset}`.

- [ ] **Step 1: Write failing execution and rollback tests**

```rust
mod support;

use std::collections::BTreeMap;

use calc_flow::{Batch, ExecutionOptions};

#[tokio::test]
async fn failed_run_restores_every_stateful_node() {
    let plan = support::state_then_fail_plan();
    let before = plan.snapshot().await.unwrap();
    assert!(plan.execute(support::single_table_input(), ExecutionOptions::default()).await.is_err());
    assert_eq!(plan.snapshot().await.unwrap(), before);
}

#[tokio::test]
async fn run_result_contains_named_outputs_and_timings() {
    let plan = support::expression_plan();
    let result = plan.execute(support::single_table_input(), ExecutionOptions::default()).await.unwrap();
    assert!(result.outputs.contains_key("output"));
    assert!(result.node_timings.contains_key("calculate"));
    assert_eq!(result.metadata.pipeline_fingerprint, plan.fingerprint());
}
```

- [ ] **Step 2: Confirm execution methods are missing**

Run: `cargo test -p calc-flow --test pipeline_execute`

Expected: FAIL because `ExecutionPlan::execute` and result types do not exist.

- [ ] **Step 3: Implement serialized run ownership and rollback**

```rust
#[derive(Clone, Debug, Default)]
pub struct ExecutionOptions { pub settings: crate::JsonMap, pub deadline: Option<chrono::DateTime<chrono::Utc>>, pub cancellation: crate::CancellationToken }

#[derive(Clone, Debug, serde::Serialize)]
pub struct NodeTiming { pub duration_ns: u64, pub input_rows: BTreeMap<String, usize>, pub output_rows: BTreeMap<String, usize> }

#[derive(Clone, Debug, serde::Serialize)]
pub struct RunMetadata { pub run_id: String, pub pipeline_name: String, pub pipeline_fingerprint: String }

#[derive(Clone, Debug)]
pub struct RunResult { pub outputs: BTreeMap<String, crate::Batch>, pub node_timings: BTreeMap<String, NodeTiming>, pub datafusion_metrics: Vec<crate::DataFusionQueryMetric>, pub metadata: RunMetadata }

impl ExecutionPlan {
    pub fn fingerprint(&self) -> &str { &self.fingerprint }

    pub async fn execute(&self, inputs: BTreeMap<String, crate::Batch>, options: ExecutionOptions) -> crate::Result<RunResult> {
        let _run_guard = self.run_lock.lock().await;
        validate_external_inputs(&self.external_inputs, &self.nodes, &inputs)?;
        let before = self.snapshot().await?;
        let context = crate::RunContext::new(options.settings, options.deadline, options.cancellation)?;
        let runtime = crate::DataFusionRuntime::new(crate::DataFusionConfig::default())?;
        let result = self.execute_nodes(inputs, &context, &runtime).await;
        match result {
            Ok((outputs, node_timings)) => Ok(RunResult { outputs, node_timings, datafusion_metrics: runtime.metrics(), metadata: RunMetadata { run_id: context.run_id().into(), pipeline_name: self.name.clone(), pipeline_fingerprint: self.fingerprint.clone() } }),
            Err(error) => { self.restore(&before).await?; Err(error) }
        }
    }

    pub async fn snapshot(&self) -> crate::Result<BTreeMap<String, serde_json::Value>> { let mut state = BTreeMap::new(); for node in &self.nodes { state.insert(node.node_id.clone(), node.operator.lock().await.snapshot()?); } Ok(state) }
    pub async fn restore(&self, state: &BTreeMap<String, serde_json::Value>) -> crate::Result<()> {
        let expected = self.nodes.iter().map(|node| node.node_id.as_str()).collect::<std::collections::BTreeSet<_>>();
        let actual = state.keys().map(String::as_str).collect::<std::collections::BTreeSet<_>>();
        if actual != expected { return Err(crate::CalcFlowError::CheckpointMismatch { message: "state node IDs do not match the plan".into() }); }
        for node in &self.nodes { node.operator.lock().await.restore(&state[&node.node_id])?; }
        Ok(())
    }
    pub async fn reset(&self) -> crate::Result<()> { for node in &self.nodes { node.operator.lock().await.reset()?; } Ok(()) }
}
```

Implement `execute_nodes` as deterministic topological execution: gather each node's inputs from external inputs or prior outputs, validate every port, check cancellation immediately before and after each operator call, reject missing/unknown operator outputs, record row counts and duration, and resolve terminal outputs by the compiled external-output map. Do not mutate caller-owned input mappings or batches.

- [ ] **Step 4: Run failure injection and cancellation tests**

Run: `cargo test -p calc-flow --test pipeline_execute -- --nocapture`

Expected: PASS; state equals the pre-run snapshot after every injected failure.

- [ ] **Step 5: Commit**

```bash
git add crates/calc-flow/src/pipeline.rs crates/calc-flow/tests/pipeline_execute.rs
git commit -m "feat: execute Rust plans transactionally"
```

### Task 11: Define the Strict V2 Project Schema

**Files:**
- Create: `crates/calc-flow/src/config.rs`
- Create: `crates/calc-flow/tests/config.rs`
- Create: `crates/calc-flow/examples/export_schema.rs`
- Create: `schemas/project-v2.schema.json`
- Modify: `crates/calc-flow/src/lib.rs`

**Interfaces:**
- Consumes: pipeline/operator/provider/UDF APIs.
- Produces: `PROJECT_FORMAT_VERSION`, `ProjectSpec`, `PipelineSpec`, `NodeSpec`, `OperatorSpec`, `PortSpec`, `EdgeSpec`, `RunOptions`, `ValidationIssue`, `ValidationReport`, `validate_project`, `compile_project`, and generated JSON Schema.

- [ ] **Step 1: Write failing strict-schema tests**

```rust
use calc_flow::{compile_project, project_json_schema, ProjectSpec, PROJECT_FORMAT_VERSION};

#[test]
fn project_rejects_v1_and_unknown_fields() {
    let v1 = r#"{"format_version":1,"id":"x","name":"x","pipeline":{"name":"p","nodes":[],"edges":[]},"data_sources":[],"run_options":{}}"#;
    assert!(serde_json::from_str::<ProjectSpec>(v1).is_err());
    let unknown = r#"{"format_version":2,"id":"x","name":"x","pipeline":{"name":"p","nodes":[],"edges":[]},"data_sources":[],"run_options":{},"callable":"os.system"}"#;
    assert!(serde_json::from_str::<ProjectSpec>(unknown).is_err());
}

#[test]
fn generated_schema_is_v2_and_stable() {
    assert_eq!(PROJECT_FORMAT_VERSION, 2);
    let schema = project_json_schema().unwrap();
    assert_eq!(schema["title"], "Calc Flow Project V2");
}
```

- [ ] **Step 2: Confirm config types do not exist**

Run: `cargo test -p calc-flow --test config`

Expected: FAIL with unresolved configuration imports.

- [ ] **Step 3: Add strict Serde/Schemars models and one compilation path**

```rust
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{BatchKind, DataFusionConfig, Edge, ExecutionPlan, ExternalOperatorSpec, JsonMap, PipelineBuilder, PortEndpoint, ProviderRegistry, Result, UdfReference, UdfRegistrySnapshot};

pub const PROJECT_FORMAT_VERSION: u32 = 2;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectSpec { pub format_version: u32, pub id: String, pub name: String, #[serde(default)] pub description: String, pub pipeline: PipelineSpec, #[serde(default)] pub data_sources: Vec<DataSourceSpec>, #[serde(default)] pub run_options: RunOptions }

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PipelineSpec { pub name: String, pub nodes: Vec<NodeSpec>, #[serde(default)] pub edges: Vec<EdgeSpec>, #[serde(default)] pub datafusion: DataFusionConfig }

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NodeSpec { pub id: String, pub operator: OperatorSpec, #[serde(default)] pub input_ports: Vec<PortSpec>, #[serde(default)] pub output_ports: Vec<PortSpec>, pub position: Option<PositionSpec> }

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OperatorSpec {
    Expression { expression: String, #[serde(default)] select: Vec<String>, filter: Option<String>, #[serde(default)] udfs: Vec<UdfReference> },
    Sql { query: String, aliases: Vec<String>, #[serde(default)] udfs: Vec<UdfReference> },
    External { provider: String, name: String, version: String, #[serde(default)] options: JsonMap },
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PortSpec { pub name: String, pub kind: BatchKind, pub required: bool, #[serde(default)] pub schema: Vec<ArrowFieldSpec> }

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArrowFieldSpec { pub name: String, pub data_type: String, pub nullable: bool }

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EdgeSpec { pub source_node: String, pub source_port: String, pub target_node: String, pub target_port: String }

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PositionSpec { pub x: f64, pub y: f64 }

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DataSourceSpec { pub id: String, pub input: String, pub format: String, pub data: Value }

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunOptions { pub max_input_bytes: usize, pub max_rows: usize, pub timeout_seconds: u64, pub memory_limit_mb: usize, pub output_rows: usize }
impl Default for RunOptions { fn default() -> Self { Self { max_input_bytes: 10 * 1024 * 1024, max_rows: 100_000, timeout_seconds: 30, memory_limit_mb: 512, output_rows: 1000 } } }

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ValidationIssue { pub path: String, pub code: String, pub message: String }
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ValidationReport { pub valid: bool, pub issues: Vec<ValidationIssue>, pub fingerprint: Option<String> }

pub fn project_json_schema() -> Result<Value> { let mut value = serde_json::to_value(schema_for!(ProjectSpec)).map_err(|error| crate::CalcFlowError::Format { message: error.to_string() })?; value["title"] = Value::String("Calc Flow Project V2".into()); Ok(value) }
```

Implement `validate_project` and `compile_project` as the only model-to-builder conversion path. Require `format_version == 2`, unique project/node/source IDs, one source per external input, supported Arrow type strings, bounded positive run options, and external-provider availability. Generate `schemas/project-v2.schema.json` by serializing `project_json_schema()` with sorted keys and a trailing newline.

- [ ] **Step 4: Run config tests and verify generated schema has no diff**

Run: `cargo test -p calc-flow --test config`

Run: `cargo run -p calc-flow --example export_schema > schemas/project-v2.schema.json && git diff --exit-code -- schemas/project-v2.schema.json`

Expected: PASS and no schema diff after regeneration.

- [ ] **Step 5: Commit**

```bash
git add crates/calc-flow/src/config.rs crates/calc-flow/src/lib.rs crates/calc-flow/tests/config.rs crates/calc-flow/examples/export_schema.rs schemas/project-v2.schema.json
git commit -m "feat: define v2 project schema"
```

### Task 12: Port Checkpoints and Project Storage

**Files:**
- Create: `crates/calc-flow/src/checkpoint.rs`
- Create: `crates/calc-flow/src/project_store.rs`
- Create: `crates/calc-flow/tests/checkpoint.rs`
- Create: `crates/calc-flow/tests/project_store.rs`
- Modify: `crates/calc-flow/src/lib.rs`

**Interfaces:**
- Consumes: `ProjectSpec`, canonical JSON, async filesystem APIs.
- Produces: `Checkpoint`, async `CheckpointStore`, `FileCheckpointStore`, async `ProjectStore`, `FileProjectStore`, JSON/YAML import/export, and SHA-256-safe filenames.

- [ ] **Step 1: Write failing round-trip, corruption, and containment tests**

```rust
use std::collections::BTreeMap;

use calc_flow::{Checkpoint, CheckpointStore, FileCheckpointStore};
use chrono::Utc;
use serde_json::json;

#[tokio::test]
async fn checkpoint_round_trip_is_v2_and_path_safe() {
    let directory = tempfile::tempdir().unwrap();
    let store = FileCheckpointStore::new(directory.path()).await.unwrap();
    let checkpoint = Checkpoint::new("../../orders", "abc", Some(json!(7)), 8, BTreeMap::from([("sum".into(), json!({"total": 10}))]), Utc::now()).unwrap();
    store.save(&checkpoint).await.unwrap();
    assert_eq!(store.load("../../orders").await.unwrap().unwrap(), checkpoint);
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[tokio::test]
async fn project_store_rejects_v1_yaml_and_duplicate_create() {
    let directory = tempfile::tempdir().unwrap();
    let store = calc_flow::FileProjectStore::new(directory.path()).await.unwrap();
    assert!(store.import_yaml("format_version: 1\nid: old\n").is_err());
}
```

- [ ] **Step 2: Confirm store types are missing**

Run: `cargo test -p calc-flow --test checkpoint --test project_store`

Expected: FAIL with unresolved storage imports.

- [ ] **Step 3: Implement versioned values and atomic same-directory writes**

```rust
use std::{collections::BTreeMap, path::{Path, PathBuf}};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{CalcFlowError, Result};

pub const CHECKPOINT_FORMAT_VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint { pub format_version: u32, pub pipeline_name: String, pub pipeline_fingerprint: String, pub source_cursor: Option<Value>, pub sequence: u64, pub state: BTreeMap<String, Value>, pub created_at: DateTime<Utc> }

impl Checkpoint {
    pub fn new(pipeline_name: &str, pipeline_fingerprint: &str, source_cursor: Option<Value>, sequence: u64, state: BTreeMap<String, Value>, created_at: DateTime<Utc>) -> Result<Self> { if pipeline_name.is_empty() || pipeline_fingerprint.is_empty() { return Err(CalcFlowError::InvalidArgument { field: "checkpoint".into(), message: "pipeline name and fingerprint must not be empty".into() }); } Ok(Self { format_version: CHECKPOINT_FORMAT_VERSION, pipeline_name: pipeline_name.into(), pipeline_fingerprint: pipeline_fingerprint.into(), source_cursor, sequence, state, created_at }) }
}

#[async_trait]
pub trait CheckpointStore: Send + Sync { async fn load(&self, pipeline_name: &str) -> Result<Option<Checkpoint>>; async fn save(&self, checkpoint: &Checkpoint) -> Result<()>; async fn delete(&self, pipeline_name: &str) -> Result<()>; }

pub struct FileCheckpointStore { directory: PathBuf }

impl FileCheckpointStore {
    pub async fn new(path: impl AsRef<Path>) -> Result<Self> { tokio::fs::create_dir_all(path.as_ref()).await.map_err(|source| CalcFlowError::Io { path: path.as_ref().display().to_string(), source })?; Ok(Self { directory: path.as_ref().to_owned() }) }
    fn path_for(&self, key: &str) -> PathBuf { let digest = Sha256::digest(key.as_bytes()); self.directory.join(format!("{}.json", hex::encode(digest))) }
}
```

Implement `CheckpointStore` and `ProjectStore` with these rules: cap document bytes before parsing; deserialize with unknown-field rejection; reject format versions other than 2; write sorted, pretty JSON to a named temporary file in the destination directory; flush and `sync_all`; atomically persist/replace; verify the stored object's ID or pipeline name matches the requested key; sort project listings by ID; distinguish conflict from not found. Use `serde-saphyr` typed deserialization with a finite parse budget and disable includes for YAML import. YAML export is data-only and passes through `ProjectSpec`.

- [ ] **Step 4: Run storage tests including injected corrupt files**

Run: `cargo test -p calc-flow --test checkpoint --test project_store`

Expected: PASS on Windows and Unix; path traversal names create only hashed files inside the configured directory.

- [ ] **Step 5: Commit**

```bash
git add crates/calc-flow/src/checkpoint.rs crates/calc-flow/src/project_store.rs crates/calc-flow/src/lib.rs crates/calc-flow/tests/checkpoint.rs crates/calc-flow/tests/project_store.rs
git commit -m "feat: port Rust persistence stores"
```

### Task 13: Port Async Sources, Sinks, and Runners

**Files:**
- Create: `crates/calc-flow/src/io.rs`
- Create: `crates/calc-flow/src/runtime/mod.rs`
- Create: `crates/calc-flow/src/runtime/micro_batch.rs`
- Create: `crates/calc-flow/src/runtime/streaming.rs`
- Create: `crates/calc-flow/tests/io.rs`
- Create: `crates/calc-flow/tests/micro_batch.rs`
- Create: `crates/calc-flow/tests/streaming.rs`
- Modify: `crates/calc-flow/tests/support/mod.rs`
- Modify: `crates/calc-flow/src/lib.rs`

**Interfaces:**
- Consumes: `ExecutionPlan`, checkpoint store, immutable batches.
- Produces: async `Source`, `Sink`, `SourceItem`, `BatchingSource`, `SinkRouter`, `MicroBatchRunner::next`, and `StreamingRunner::step`.

- [ ] **Step 1: Write failing at-least-once and rollback tests**

```rust
mod support;

#[tokio::test]
async fn micro_batch_checkpoints_only_after_all_sinks_succeed() {
    let fixture = support::runner_fixture_with_failing_second_sink();
    let mut runner = fixture.micro_batch_runner().await;
    assert!(runner.next().await.is_err());
    assert!(fixture.checkpoint_store.load("runner").await.unwrap().is_none());
    assert_eq!(fixture.plan.snapshot().await.unwrap(), fixture.initial_state);
}

#[tokio::test]
async fn streaming_recovers_once_and_advances_after_delivery() {
    let fixture = support::recovered_streaming_fixture().await;
    let mut runner = fixture.streaming_runner().await;
    let result = runner.step(fixture.batch(), &mut fixture.sinks()).await.unwrap();
    assert_eq!(result.metadata.pipeline_fingerprint, fixture.plan.fingerprint());
    assert_eq!(fixture.checkpoint_store.load("stream").await.unwrap().unwrap().sequence, 8);
}
```

- [ ] **Step 2: Confirm runner APIs are absent**

Run: `cargo test -p calc-flow --test io --test micro_batch --test streaming`

Expected: FAIL with unresolved runtime imports.

- [ ] **Step 3: Add async owned I/O contracts and delivery ordering**

```rust
use async_trait::async_trait;
use serde_json::Value;

use crate::{Batch, Result, RunContext};

#[derive(Clone, Debug)]
pub struct SourceItem { pub batch: Batch, pub cursor: Option<Value>, pub sequence: u64 }

#[async_trait]
pub trait Source: Send {
    async fn open(&mut self, cursor: Option<Value>) -> Result<()>;
    async fn next(&mut self) -> Result<Option<SourceItem>>;
}

#[async_trait]
pub trait Sink: Send {
    async fn write(&mut self, batch: &Batch, context: &RunContext) -> Result<()>;
}
```

```rust
pub struct MicroBatchRunner {
    plan: std::sync::Arc<crate::ExecutionPlan>,
    source: Box<dyn crate::Source>,
    sinks: crate::runtime::SinkRouter,
    checkpoints: std::sync::Arc<dyn crate::CheckpointStore>,
    checkpoint_every: u64,
    recovered: bool,
    delivered: u64,
}

impl MicroBatchRunner {
    pub async fn next(&mut self) -> crate::Result<Option<crate::RunResult>> {
        self.recover_once().await?;
        let Some(item) = self.source.next().await? else { return Ok(None); };
        let before = self.plan.snapshot().await?;
        let inputs = std::collections::BTreeMap::from([(self.plan.single_external_input()?.to_owned(), item.batch)]);
        let result = self.plan.execute(inputs, crate::ExecutionOptions::default()).await?;
        if let Err(error) = self.sinks.write_all(&result).await {
            self.plan.restore(&before).await?;
            return Err(error);
        }
        self.delivered += 1;
        if self.delivered % self.checkpoint_every == 0 {
            let checkpoint = crate::Checkpoint::new(&self.plan.name, self.plan.fingerprint(), item.cursor, item.sequence, self.plan.snapshot().await?, chrono::Utc::now())?;
            self.checkpoints.save(&checkpoint).await?;
        }
        Ok(Some(result))
    }
}
```

`BatchingSource` must accept only table batches and coalesce adjacent record batches until adding another source item would exceed `max_rows` or `max_bytes`; use Arrow buffer sizes and `concat_batches` without mutating source batches. `SinkRouter` maps named plan outputs to ordered sink lists and stops on the first failure. `StreamingRunner::step` uses the same pre-run snapshot, sink ordering, and checkpoint construction on every successful step. Recovery validates pipeline name and fingerprint before restoring state and opening the source cursor, and occurs once per runner lifecycle.

- [ ] **Step 4: Run I/O and runner fault-injection tests**

Run: `cargo test -p calc-flow --test io --test micro_batch --test streaming`

Expected: PASS for source recovery, multi-sink failure, checkpoint ordering, reset, and stale-fingerprint rejection.

- [ ] **Step 5: Commit**

```bash
git add crates/calc-flow/src/io.rs crates/calc-flow/src/runtime crates/calc-flow/src/lib.rs crates/calc-flow/tests/io.rs crates/calc-flow/tests/micro_batch.rs crates/calc-flow/tests/streaming.rs
git commit -m "feat: port Rust runtime runners"
```

### Task 14: Pass the Rust Core Parity Gate

**Files:**
- Create: `crates/calc-flow/tests/v1_fixtures.rs`
- Create: `crates/calc-flow/tests/properties.rs`
- Create: `crates/calc-flow/benches/core.rs`
- Create: `crates/calc-flow/examples/expression_pipeline.rs`
- Create: `crates/calc-flow/examples/micro_batch_recovery.rs`
- Modify: `crates/calc-flow/tests/support/mod.rs`
- Modify: `crates/calc-flow/Cargo.toml`
- Modify: `crates/calc-flow/src/lib.rs`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/benchmarks.yml`

**Interfaces:**
- Consumes: every Rust core module and frozen v1 fixture.
- Produces: a documented Rust public API and a hard gate before any Python binding task starts.

- [ ] **Step 1: Write the failing fixture parity test**

```rust
mod support;

use std::fs::File;

use datafusion::arrow::ipc::reader::FileReader;

#[tokio::test]
async fn expression_fixture_matches_frozen_semantics() {
    let input = FileReader::try_new(File::open("../../tests/fixtures/v1/expression.arrow").unwrap(), None).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
    let expected = FileReader::try_new(File::open("../../tests/fixtures/v1/expression_expected.arrow").unwrap(), None).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
    let plan = support::expression_plan();
    let result = plan.execute(support::input_from_batches(input), Default::default()).await.unwrap();
    assert_eq!(result.outputs["output"].table_payload().unwrap().batches(), expected);
}
```

- [ ] **Step 2: Run the full Rust suite and record current gaps**

Run: `cargo test -p calc-flow --all-targets`

Expected: FAIL until all fixture and property cases are represented.

- [ ] **Step 3: Add public exports, properties, examples, and benchmarks**

```rust
// benches/core.rs
use criterion::{criterion_group, criterion_main, Criterion};
use calc_flow::{ExpressionOperator, PipelineBuilder, UdfRegistry};

fn compile_expression(c: &mut Criterion) {
    c.bench_function("compile/expression", |b| {
        b.iter(|| {
            let mut builder = PipelineBuilder::new("bench").unwrap();
            builder.add_node("calculate", ExpressionOperator::new("total = a + b").unwrap()).unwrap();
            builder.compile(&UdfRegistry::new().snapshot()).unwrap()
        })
    });
}

criterion_group!(benches, compile_expression);
criterion_main!(benches);
```

Add `criterion = "0.8.0"` under dev dependencies and a `[[bench]]` entry with `harness = false`. Add proptests for canonical JSON round trips, generated acyclic graphs, generated cycles, metadata JSON values, and checkpoint state. Re-export only supported public types from `lib.rs`; leave compiled nodes, cleanup guards, and parsing helpers private. Add runnable Rust examples for an expression pipeline and checkpoint recovery.

Add `cargo-llvm-cov` 0.8.7 to CI and run:

```bash
cargo llvm-cov --workspace --all-features --fail-under-lines 90 --lcov --output-path rust-lcov.info
```

Update scheduled benchmarks to run `cargo bench -p calc-flow` before Python benchmarks.

- [ ] **Step 4: Execute the Rust parity gate**

Run: `cargo fmt --all --check`

Run: `cargo clippy --workspace --all-targets --all-features -- -D warnings`

Run: `cargo test --workspace --all-targets`

Run: `cargo llvm-cov --workspace --all-features --fail-under-lines 90`

Run: `cargo doc --workspace --no-deps`

Expected: all commands exit 0; do not begin Task 15 otherwise.

- [ ] **Step 5: Commit the core parity milestone**

```bash
git add crates/calc-flow .github/workflows/ci.yml .github/workflows/benchmarks.yml
git commit -m "test: complete Rust core parity gate"
```

### Task 15: Scaffold the Maturin and PyO3 Package

**Files:**
- Create: `crates/calc-flow-python/Cargo.toml`
- Create: `crates/calc-flow-python/src/lib.rs`
- Create: `crates/calc-flow-python/src/error.rs`
- Create: `python/calc_flow/__init__.py`
- Create: `python/calc_flow/errors.py`
- Create: `python/calc_flow/_native.pyi`
- Create: `python/calc_flow/py.typed`
- Create: `python/tests/test_import.py`
- Modify: `Cargo.toml`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: the complete Rust core public API.
- Produces: mixed Maturin package `calc_flow` with private native module `calc_flow._native` and Python exception hierarchy.

- [ ] **Step 1: Write the failing native import test**

```python
from __future__ import annotations

import calc_flow
from calc_flow import _native


def test_native_package_reports_v2() -> None:
    assert calc_flow.__version__ == "2.0.0a1"
    assert _native.version() == "2.0.0-alpha.1"
    assert issubclass(calc_flow.CompileError, calc_flow.CalcFlowError)
```

- [ ] **Step 2: Confirm the mixed package is not configured**

Run: `uv run pytest python/tests/test_import.py -q`

Expected: FAIL because `python/calc_flow` and the native extension do not exist.

- [ ] **Step 3: Add the binding crate and switch the root build backend**

```toml
# crates/calc-flow-python/Cargo.toml
[package]
name = "calc-flow-python"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
publish = false

[lib]
name = "calc_flow_python"
crate-type = ["cdylib"]

[dependencies]
async-trait.workspace = true
calc-flow = { path = "../calc-flow" }
datafusion.workspace = true
parking_lot.workspace = true
pyo3 = { version = "0.28.3", features = ["abi3-py313"] }
pyo3-async-runtimes = { version = "0.28.0", features = ["tokio-runtime"] }
pyo3-arrow = "0.17.0"
serde_json.workspace = true
tokio.workspace = true

[lints]
workspace = true
```

```toml
# pyproject.toml
[project]
name = "calc-flow"
version = "2.0.0a1"
description = "Rust-native DataFusion batch and streaming calculation pipelines"
readme = "README.md"
requires-python = ">=3.13"
dependencies = ["pydantic>=2.12.0,<3", "pyarrow>=24.0.0"]

[project.optional-dependencies]
numpy = ["numpy>=2.0.0"]
jax = ["jax>=0.4"]
array = ["jax>=0.4", "numpy>=2.0.0"]
benchmark = ["jax>=0.4", "numpy>=2.0.0", "psutil>=6.1", "pytest>=8", "pytest-benchmark>=5.2"]
dev = ["hypothesis>=6.100", "jax>=0.4", "numpy>=2.0.0", "psutil>=6.1", "pytest>=8", "pytest-cov>=7.1", "pytest-xdist>=3.6", "ruff>=0.14"]

[build-system]
requires = ["maturin>=1.14.1,<2.0"]
build-backend = "maturin"

[tool.maturin]
manifest-path = "crates/calc-flow-python/Cargo.toml"
python-source = "python"
module-name = "calc_flow._native"
features = ["pyo3/abi3-py313"]
strip = true

[tool.uv.workspace]
members = ["web-ui/backend"]

[tool.pytest.ini_options]
testpaths = ["python/tests"]

[tool.ruff]
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

```rust
use pyo3::prelude::*;

#[pyfunction]
fn version() -> &'static str { calc_flow::VERSION }

#[pymodule(gil_used = true)]
#[pyo3(name = "_native")]
fn calc_flow_python(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    error::register(module)?;
    Ok(())
}
```

Add `crates/calc-flow-python` to workspace members. In `error.rs`, use `create_exception!` to define `CalcFlowError`, `ConfigError`, `CompileError`, `ExecutionError`, `ProviderError`, `CheckpointError`, and `CancelledError`, plus a total `From<calc_flow::CalcFlowError> for PyErr` mapping. Export them from `python/calc_flow/errors.py` and set `__version__ = "2.0.0a1"` in `__init__.py`.

- [ ] **Step 4: Build and test the native package**

Run: `uv sync --extra dev`

Run: `uv run maturin develop`

Run: `uv run pytest python/tests/test_import.py -q`

Expected: PASS; `src/calc_flow` is not included in the built wheel.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock pyproject.toml crates/calc-flow-python python
git commit -m "feat: scaffold Rust Python package"
```

### Task 16: Bind Immutable Batches Through Arrow PyCapsules

**Files:**
- Create: `crates/calc-flow-python/src/batch.rs`
- Create: `python/tests/test_batch.py`
- Modify: `crates/calc-flow-python/src/lib.rs`
- Modify: `python/calc_flow/__init__.py`
- Modify: `python/calc_flow/_native.pyi`

**Interfaces:**
- Consumes: core `Batch`, `pyo3_arrow::PyTable`, Python JSON objects.
- Produces: frozen native Python `Batch` with `from_pyarrow`, `to_pyarrow`, `metadata`, `kind`, and `num_rows`.

- [ ] **Step 1: Write failing zero-copy and immutability tests**

```python
from __future__ import annotations

import pyarrow as pa
import pytest

from calc_flow import Batch


def test_table_batch_reuses_arrow_buffers_and_copies_metadata() -> None:
    table = pa.table({"value": [1, 2, 3]})
    metadata = {"nested": {"enabled": True}}
    batch = Batch.from_pyarrow(table, metadata=metadata)
    metadata["nested"]["enabled"] = False
    result = batch.to_pyarrow()
    assert result.column(0).chunk(0).buffers()[1].address == table.column(0).chunk(0).buffers()[1].address
    assert batch.metadata == {"nested": {"enabled": True}}
    assert batch.num_rows == 3


def test_table_accessor_rejects_array_payload() -> None:
    with pytest.raises(TypeError):
        Batch._from_external(object(), "test", 1, {}).to_pyarrow()
```

- [ ] **Step 2: Confirm native Batch is absent**

Run: `uv run pytest python/tests/test_batch.py -q`

Expected: FAIL because `calc_flow.Batch` is not exported.

- [ ] **Step 3: Add the frozen PyO3 batch and centralized JSON conversion**

```rust
use std::{fmt, sync::Arc};

use pyo3::{exceptions::PyTypeError, prelude::*, types::PyAny};
use pyo3_arrow::PyTable;

#[derive(Clone)]
pub struct PythonPayload { pub(crate) object: Py<PyAny>, backend: String, len: usize }

impl PythonPayload {
    pub(crate) fn object(&self, py: Python<'_>) -> Py<PyAny> { self.object.clone_ref(py) }
}

impl fmt::Debug for PythonPayload { fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result { formatter.debug_struct("PythonPayload").field("backend", &self.backend).field("len", &self.len).finish_non_exhaustive() } }
impl calc_flow::ExternalPayload for PythonPayload { fn backend(&self) -> &str { &self.backend } fn len(&self) -> usize { self.len } fn as_any(&self) -> &dyn std::any::Any { self } }

#[pyclass(name = "Batch", frozen, module = "calc_flow._native")]
pub struct PyBatch { pub(crate) inner: calc_flow::Batch }

#[pymethods]
impl PyBatch {
    #[staticmethod]
    #[pyo3(signature = (table, metadata=None))]
    fn from_pyarrow(py: Python<'_>, table: PyTable, metadata: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let (batches, _) = table.into_inner();
        let metadata = metadata_from_python(py, metadata)?;
        Ok(Self { inner: calc_flow::Batch::table(batches, metadata)? })
    }

    #[staticmethod]
    fn _from_external(object: Py<PyAny>, backend: String, len: usize, metadata: &Bound<'_, PyAny>) -> PyResult<Self> {
        let metadata = metadata_from_python(metadata.py(), Some(metadata))?;
        Ok(Self { inner: calc_flow::Batch::external(Arc::new(PythonPayload { object, backend, len }), metadata)? })
    }

    fn to_pyarrow<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let table = self.inner.table_payload().map_err(|_| PyTypeError::new_err("array batches do not contain a PyArrow table"))?;
        PyTable::try_new(table.batches().to_vec(), table.schema().clone())?.into_pyarrow(py)
    }

    #[getter]
    fn kind(&self) -> &'static str { match self.inner.kind() { calc_flow::BatchKind::Table => "table", calc_flow::BatchKind::Array => "array" } }
    #[getter]
    fn num_rows(&self) -> usize { self.inner.num_rows() }
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> { metadata_to_python(py, self.inner.metadata()) }
}
```

```rust
fn metadata_from_python(py: Python<'_>, value: Option<&Bound<'_, PyAny>>) -> PyResult<calc_flow::BatchMetadata> {
    let Some(value) = value else { return Ok(calc_flow::BatchMetadata::default()); };
    let json = py.import("json")?;
    let encoded: String = json.call_method1("dumps", (value,))?.extract()?;
    let attributes = serde_json::from_str::<std::collections::BTreeMap<String, serde_json::Value>>(&encoded)
        .map_err(|error| PyTypeError::new_err(format!("metadata must be a JSON-compatible mapping: {error}")))?;
    calc_flow::BatchMetadata::new("", 0, attributes).map_err(Into::into)
}

fn metadata_to_python<'py>(py: Python<'py>, metadata: &calc_flow::BatchMetadata) -> PyResult<Bound<'py, PyAny>> {
    let encoded = serde_json::to_string(metadata.attributes())
        .map_err(|error| PyTypeError::new_err(error.to_string()))?;
    py.import("json")?.call_method1("loads", (encoded,))
}
```

Treat the Python metadata mapping as `BatchMetadata.attributes`; source and sequence remain runner-owned fields. The JSON round trip makes a defensive copy and rejects executable or non-JSON objects. Register `PyBatch` in `_native` and export it as `calc_flow.Batch`.

- [ ] **Step 4: Run ownership and repeated-destruction tests**

Run: `uv run maturin develop && uv run pytest python/tests/test_batch.py -q`

Expected: PASS, including 10,000 create/drop iterations without reference growth.

- [ ] **Step 5: Commit**

```bash
git add crates/calc-flow-python/src python/calc_flow python/tests/test_batch.py
git commit -m "feat: bind Arrow batches to Python"
```

### Task 17: Bind Project Compilation and Sync/Async Execution

**Files:**
- Create: `crates/calc-flow-python/src/config.rs`
- Create: `crates/calc-flow-python/src/pipeline.rs`
- Create: `python/calc_flow/pipeline.py`
- Create: `python/tests/test_pipeline.py`
- Create: `python/tests/test_async.py`
- Modify: `crates/calc-flow-python/src/lib.rs`
- Modify: `python/calc_flow/__init__.py`
- Modify: `python/calc_flow/_native.pyi`

**Interfaces:**
- Consumes: JSON `ProjectSpec`, native batches, core `ExecutionPlan`.
- Produces: native `Runtime`, `ExecutionPlan`, `RunResult`, `validate_project_json`, `project_json_schema`; immutable Python `PipelineBuilder`; blocking and awaitable execution.

- [ ] **Step 1: Write failing Python builder and async tests**

```python
from __future__ import annotations

import asyncio

import pyarrow as pa
import pytest

from calc_flow import Batch, PipelineBuilder


def test_python_builder_compiles_through_rust() -> None:
    plan = PipelineBuilder("totals").expression("calc", "total = a + b").compile()
    result = plan.execute({"input": Batch.from_pyarrow(pa.table({"a": [1], "b": [2]}))})
    assert result.outputs["output"].to_pyarrow()["total"].to_pylist() == [3]


@pytest.mark.asyncio
async def test_execute_async_does_not_block_the_python_loop() -> None:
    plan = PipelineBuilder("totals").expression("calc", "total = a + b").compile()
    heartbeat = asyncio.create_task(asyncio.sleep(0))
    result = await plan.execute_async({"input": Batch.from_pyarrow(pa.table({"a": [1], "b": [2]}))})
    await heartbeat
    assert result.metadata["pipeline_name"] == "totals"


@pytest.mark.asyncio
async def test_blocking_execute_rejects_running_loop() -> None:
    plan = PipelineBuilder("totals").expression("calc", "total = a + b").compile()
    with pytest.raises(RuntimeError, match="execute_async"):
        plan.execute({"input": Batch.from_pyarrow(pa.table({"a": [1], "b": [2]}))})
```

- [ ] **Step 2: Confirm execution bindings are missing**

Run: `uv run pytest python/tests/test_pipeline.py python/tests/test_async.py -q`

Expected: FAIL because `PipelineBuilder` and execution bindings do not exist.

- [ ] **Step 3: Add a runtime-owned registry and plan wrapper**

```rust
#[pyclass(name = "Runtime", frozen, module = "calc_flow._native")]
pub struct PyRuntime {
    pub(crate) providers: std::sync::Arc<calc_flow::ProviderRegistry>,
    pub(crate) udfs: std::sync::Arc<parking_lot::RwLock<calc_flow::UdfRegistry>>,
    pub(crate) tokio: std::sync::Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyRuntime {
    #[new]
    fn new() -> PyResult<Self> { Ok(Self { providers: Default::default(), udfs: Default::default(), tokio: std::sync::Arc::new(tokio::runtime::Runtime::new().map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?) }) }

    fn compile_project(&self, project_json: &str) -> PyResult<PyExecutionPlan> {
        let project: calc_flow::ProjectSpec = serde_json::from_str(project_json).map_err(crate::error::config)?;
        let plan = calc_flow::compile_project(&project, &self.providers, &self.udfs.read().snapshot())?;
        Ok(PyExecutionPlan { inner: std::sync::Arc::new(plan), tokio: self.tokio.clone() })
    }
}

#[pyclass(name = "ExecutionPlan", frozen, module = "calc_flow._native")]
pub struct PyExecutionPlan { pub(crate) inner: std::sync::Arc<calc_flow::ExecutionPlan>, pub(crate) tokio: std::sync::Arc<tokio::runtime::Runtime> }

#[pymethods]
impl PyExecutionPlan {
    fn execute(&self, py: Python<'_>, inputs: &Bound<'_, pyo3::types::PyDict>) -> PyResult<PyRunResult> {
        let inputs = crate::batch::extract_inputs(inputs)?;
        let plan = self.inner.clone();
        let runtime = self.tokio.clone();
        py.detach(move || runtime.block_on(plan.execute(inputs, Default::default())).map(PyRunResult::from).map_err(Into::into))
    }

    fn execute_async<'py>(&self, py: Python<'py>, inputs: &Bound<'py, pyo3::types::PyDict>) -> PyResult<Bound<'py, PyAny>> {
        let inputs = crate::batch::extract_inputs(inputs)?;
        let plan = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move { plan.execute(inputs, Default::default()).await.map(PyRunResult::from).map_err(Into::into) })
    }
}
```

In `python/calc_flow/pipeline.py`, implement `PipelineBuilder` as a frozen dataclass containing a v2 project dictionary. `expression`, `sql`, and `connect` return new builders using copied tuples and mappings. `compile(runtime=None)` serializes with sorted compact JSON and calls `Runtime.compile_project`. The public Python `ExecutionPlan.execute` wrapper calls `asyncio.get_running_loop()` and raises `RuntimeError("execute() cannot run inside an event loop; use execute_async()")` before entering native code.

- [ ] **Step 4: Run sync, async, cancellation, and error-mapping tests**

Run: `uv run maturin develop && uv run pytest python/tests/test_pipeline.py python/tests/test_async.py -q`

Expected: PASS; Rust/DataFusion work does not hold the GIL and Python exceptions use the declared hierarchy.

- [ ] **Step 5: Commit**

```bash
git add crates/calc-flow-python/src python/calc_flow python/tests/test_pipeline.py python/tests/test_async.py
git commit -m "feat: expose Rust plans to Python"
```

### Task 18: Implement Python-Hosted NumPy and JAX Providers

**Files:**
- Create: `crates/calc-flow-python/src/provider.rs`
- Create: `python/calc_flow/array.py`
- Create: `python/tests/test_array.py`
- Modify: `crates/calc-flow-python/src/batch.rs`
- Modify: `crates/calc-flow-python/src/lib.rs`
- Modify: `python/calc_flow/__init__.py`

**Interfaces:**
- Consumes: core `ExternalOperatorFactory`, native `Runtime`, opaque `PythonPayload`.
- Produces: `Runtime.register_provider`, Python `register_numpy`, `register_jax`, restricted array expressions, and Python-host-only compilation.

- [ ] **Step 1: Write failing provider, immutability, and host-boundary tests**

```python
from __future__ import annotations

import numpy as np
import pytest

from calc_flow import Batch, PipelineBuilder, Runtime, register_numpy


def test_numpy_provider_owns_read_only_arrays() -> None:
    runtime = Runtime()
    register_numpy(runtime)
    source = np.array([1.0, 2.0])
    plan = PipelineBuilder("arrays").external("calc", "numpy", "expression", "1", {"expression": "x * 2"}).compile(runtime)
    result = plan.execute({"input": Batch.from_array(source, backend="numpy")})
    source[0] = 99
    output = result.outputs["output"].array
    assert output.tolist() == [2.0, 4.0]
    assert not output.flags.writeable


def test_missing_python_provider_fails_during_compile() -> None:
    with pytest.raises(Exception, match="provider numpy:expression@1 is unavailable"):
        PipelineBuilder("arrays").external("calc", "numpy", "expression", "1", {"expression": "x + 1"}).compile(Runtime())


def test_array_expression_rejects_python_execution_syntax() -> None:
    runtime = Runtime()
    register_numpy(runtime)
    with pytest.raises(ValueError):
        PipelineBuilder("unsafe").external("calc", "numpy", "expression", "1", {"expression": "__import__('os').system('whoami')"}).compile(runtime)
```

- [ ] **Step 2: Confirm provider registration is missing**

Run: `uv run pytest python/tests/test_array.py -q`

Expected: FAIL because array providers and `Batch.from_array` do not exist.

- [ ] **Step 3: Add the explicit PyO3 callback boundary and safe evaluators**

```rust
pub struct PythonOperatorFactory { callback: Py<PyAny> }

impl calc_flow::ExternalOperatorFactory for PythonOperatorFactory {
    fn create(&self, spec: &calc_flow::ExternalOperatorSpec, inputs: Vec<calc_flow::Port>, outputs: Vec<calc_flow::Port>) -> calc_flow::Result<Box<dyn calc_flow::Operator>> {
        Ok(Box::new(PythonOperator { name: spec.name.clone(), callback: Python::attach(|py| self.callback.clone_ref(py)), options: spec.options.clone(), inputs, outputs }))
    }
}

#[async_trait::async_trait]
impl calc_flow::Operator for PythonOperator {
    fn name(&self) -> &str { &self.name }
    fn input_ports(&self) -> &[calc_flow::Port] { &self.inputs }
    fn output_ports(&self) -> &[calc_flow::Port] { &self.outputs }
    fn configuration(&self) -> calc_flow::JsonMap { self.options.clone() }
    async fn process(&mut self, inputs: &std::collections::BTreeMap<String, calc_flow::Batch>, _context: &calc_flow::OperatorContext<'_>) -> calc_flow::Result<std::collections::BTreeMap<String, calc_flow::Batch>> {
        Python::attach(|py| crate::provider::call_python_operator(py, &self.callback, inputs, &self.options)).map_err(|error| calc_flow::CalcFlowError::ExternalProvider { provider: "python".into(), name: self.name.clone(), version: "1".into(), message: error.to_string() })
    }
}
```

```python
_ALLOWED_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.MatMult: operator.matmul, ast.Pow: operator.pow}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCTIONS = {"sum", "mean", "max", "min", "transpose", "reshape"}


def _owned_numpy(value: object) -> object:
    import numpy as np

    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


def _evaluate(node: ast.AST, values: Mapping[str, object], namespace: object) -> object:
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY:
        return _ALLOWED_BINARY[type(node.op)](_evaluate(node.left, values, namespace), _evaluate(node.right, values, namespace))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_evaluate(node.operand, values, namespace))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCTIONS:
        function = getattr(namespace, node.func.id)
        return function(*(_evaluate(argument, values, namespace) for argument in node.args))
    raise ValueError(f"unsupported array expression node: {type(node).__name__}")
```

Complete `array.py` with AST depth/node limits, explicit UDF references, `reshape` shape validation, argument-count validation, backend retention, and no `eval`/`exec`. `register_numpy` defensively copies and marks outputs read-only. `register_jax` verifies that outputs remain JAX Array API objects. The binding converts only `PythonPayload` values to callbacks and rejects external payloads created by other hosts.

- [ ] **Step 4: Run NumPy/JAX and callback rollback tests**

Run: `JAX_PLATFORMS=cpu uv run maturin develop && uv run pytest python/tests/test_array.py -q`

Expected: PASS for both providers, unsafe syntax rejection, backend retention, callback exceptions, and state rollback.

- [ ] **Step 5: Commit**

```bash
git add crates/calc-flow-python/src python/calc_flow python/tests/test_array.py
git commit -m "feat: add Python array providers"
```

### Task 19: Bind Python Scalar UDFs and the Canonical Project Schema

**Files:**
- Modify: `crates/calc-flow-python/src/config.rs`
- Create: `crates/calc-flow-python/src/udf.rs`
- Create: `python/calc_flow/config.py`
- Create: `python/calc_flow/udf.py`
- Create: `python/tests/test_config.py`
- Create: `python/tests/test_udf.py`
- Modify: `crates/calc-flow-python/src/lib.rs`
- Modify: `python/calc_flow/__init__.py`
- Modify: `python/calc_flow/_native.pyi`

**Interfaces:**
- Consumes: Rust-generated JSON Schema, runtime-owned UDF registry, Arrow PyCapsules.
- Produces: Pydantic-compatible `ProjectDocument`, canonical project JSON, validation reports, catalog output, and trusted Python scalar UDF registration.

- [ ] **Step 1: Write failing schema delegation and UDF tests**

```python
from __future__ import annotations

import pyarrow as pa
from pydantic import BaseModel, ValidationError
import pytest

from calc_flow import Batch, PipelineBuilder, ProjectDocument, Runtime


def test_project_document_uses_rust_schema_and_rejects_v1() -> None:
    schema = ProjectDocument.model_json_schema()
    assert schema["title"] == "Calc Flow Project V2"
    with pytest.raises(ValidationError):
        ProjectDocument.model_validate({"format_version": 1})


def test_python_scalar_udf_executes_inside_datafusion() -> None:
    runtime = Runtime()
    runtime.register_scalar_udf(
        provider="python",
        name="double",
        version="1",
        input_types=["int64"],
        return_type="int64",
        volatility="immutable",
        function=lambda value: pa.compute.multiply(value, 2),
    )
    plan = PipelineBuilder("udf").expression("calc", "result = double(value)", udfs=[("python", "double", "1")]).compile(runtime)
    output = plan.execute({"input": Batch.from_pyarrow(pa.table({"value": [2]}))}).outputs["output"].to_pyarrow()
    assert output["result"].to_pylist() == [4]
    assert "function" not in str(runtime.catalog())
```

- [ ] **Step 2: Confirm schema and UDF APIs are absent**

Run: `uv run pytest python/tests/test_config.py python/tests/test_udf.py -q`

Expected: FAIL because `ProjectDocument` and `register_scalar_udf` do not exist.

- [ ] **Step 3: Add Rust-delegating validation and a PyArrow UDF closure**

```python
from __future__ import annotations

import json
from typing import Any, ClassVar

from pydantic import RootModel, model_validator

from calc_flow import _native


JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


class ProjectDocument(RootModel[dict[str, JSONValue]]):
    @model_validator(mode="before")
    @classmethod
    def validate_with_rust(cls, value: Any) -> dict[str, JSONValue]:
        canonical = _native.validate_project_json(json.dumps(value, separators=(",", ":"), sort_keys=True))
        return json.loads(canonical)

    @classmethod
    def model_json_schema(cls, *args: object, **kwargs: object) -> dict[str, Any]:
        return json.loads(_native.project_json_schema())

    def canonical_json(self) -> str:
        return _native.validate_project_json(json.dumps(self.root, separators=(",", ":"), sort_keys=True))
```

```rust
pub fn python_scalar_udf(
    reference: calc_flow::UdfReference,
    input_types: Vec<datafusion::arrow::datatypes::DataType>,
    return_type: datafusion::arrow::datatypes::DataType,
    volatility: datafusion::logical_expr::Volatility,
    function: Py<PyAny>,
) -> PyResult<std::sync::Arc<datafusion::logical_expr::ScalarUDF>> {
    let implementation = std::sync::Arc::new(move |arguments: &[datafusion::logical_expr::ColumnarValue]| {
        Python::attach(|py| {
            let length = arguments.iter().find_map(|value| match value { datafusion::logical_expr::ColumnarValue::Array(array) => Some(array.len()), datafusion::logical_expr::ColumnarValue::Scalar(_) => None }).unwrap_or(1);
            let py_arguments = arguments.iter().map(|value| crate::udf::columnar_to_python(py, value, length)).collect::<PyResult<Vec<_>>>()?;
            let output = function.call1(py, pyo3::types::PyTuple::new(py, py_arguments)?)?;
            crate::udf::python_to_columnar(output.bind(py), &return_type, length)
        }).map_err(|error| datafusion::error::DataFusionError::Execution(format!("python UDF {}:{}@{} failed: {error}", reference.provider, reference.name, reference.version)))
    });
    Ok(std::sync::Arc::new(datafusion::logical_expr::create_udf(&reference.name, input_types, return_type, volatility, implementation)))
}
```

Use `pyo3-arrow` array wrappers for every Arrow argument and output. Validate input nullability, output Arrow type, scalar broadcasting, array length, and provider/name/version before inserting the resulting `ScalarUDF` into the runtime's Rust `UdfRegistry`. `Runtime.catalog()` returns only JSON-compatible provider/name/version/kind/signature/volatility metadata.

- [ ] **Step 4: Run schema, UDF, and catalog tests**

Run: `uv run maturin develop && uv run pytest python/tests/test_config.py python/tests/test_udf.py -q`

Expected: PASS for validation, type errors, length errors, unknown versions, conflicting versions, and catalog redaction.

- [ ] **Step 5: Commit**

```bash
git add crates/calc-flow-python/src python/calc_flow python/tests/test_config.py python/tests/test_udf.py
git commit -m "feat: bind Python configuration and UDFs"
```

### Task 20: Complete Python Store and Runner Parity

**Files:**
- Create: `crates/calc-flow-python/src/store.rs`
- Create: `crates/calc-flow-python/src/runtime.rs`
- Create: `python/calc_flow/store.py`
- Create: `python/calc_flow/runtime.py`
- Create: `python/tests/test_store.py`
- Create: `python/tests/test_runtime.py`
- Modify: `crates/calc-flow-python/src/lib.rs`
- Modify: `python/calc_flow/__init__.py`
- Modify: `python/calc_flow/_native.pyi`

**Interfaces:**
- Consumes: Rust file stores and async runners; Python source/sink callables.
- Produces: Python `FileProjectStore`, `FileCheckpointStore`, `MicroBatchRunner`, and `StreamingRunner`, each with async-first and guarded blocking methods.

- [ ] **Step 1: Write failing Python store and delivery tests**

```python
from __future__ import annotations

import pyarrow as pa
import pytest

from calc_flow import Batch, FileCheckpointStore, FileProjectStore, PipelineBuilder, StreamingRunner


@pytest.mark.asyncio
async def test_project_store_round_trips_v2_document(tmp_path) -> None:
    store = FileProjectStore(tmp_path)
    project = PipelineBuilder("stored").expression("calc", "b = a + 1").project
    await store.create(project)
    assert (await store.get(project.root["id"])).root == project.root


@pytest.mark.asyncio
async def test_streaming_sink_failure_rolls_back(tmp_path) -> None:
    plan = PipelineBuilder("stream").stateful_test_counter().compile()
    checkpoints = FileCheckpointStore(tmp_path)
    runner = StreamingRunner(plan, checkpoints)

    async def fail(_: Batch) -> None:
        raise RuntimeError("sink failed")

    before = await plan.snapshot_async()
    with pytest.raises(RuntimeError, match="sink failed"):
        await runner.step_async(Batch.from_pyarrow(pa.table({"value": [1]})), sinks={"output": [fail]})
    assert await plan.snapshot_async() == before
```

- [ ] **Step 2: Confirm store and runner wrappers are missing**

Run: `uv run pytest python/tests/test_store.py python/tests/test_runtime.py -q`

Expected: FAIL with missing exports.

- [ ] **Step 3: Bind async store methods and Python I/O adapters**

```rust
pub struct PythonSink { callback: Py<PyAny> }

#[async_trait::async_trait]
impl calc_flow::Sink for PythonSink {
    async fn write(&mut self, batch: &calc_flow::Batch, _context: &calc_flow::RunContext) -> calc_flow::Result<()> {
        let awaitable = Python::attach(|py| {
            let py_batch = Py::new(py, crate::batch::PyBatch { inner: batch.clone() })?;
            self.callback.call1(py, (py_batch,))
        }).map_err(crate::provider::external_error)?;
        if Python::attach(|py| awaitable.bind(py).hasattr("__await__")).unwrap_or(false) {
            pyo3_async_runtimes::tokio::into_future(Python::attach(|py| awaitable.into_bound(py))).map_err(crate::provider::external_error)?.await.map_err(crate::provider::external_error)?;
        }
        Ok(())
    }
}
```

Expose every filesystem method as an awaitable using `future_into_py`. Implement guarded blocking wrappers in `python/calc_flow/store.py` and `runtime.py` using the same running-loop check as `ExecutionPlan.execute`. Adapt Python sources with `open(cursor)` and `next()` methods; accept synchronous return values or awaitables; require each item to be `(Batch, cursor, sequence)`. Adapt sinks as shown and preserve sink order. Bind plan snapshot/restore/reset methods so runner tests can assert rollback directly.

- [ ] **Step 4: Run Python parity, coverage, and lint gates**

Run: `uv run maturin develop`

Run: `JAX_PLATFORMS=cpu uv run pytest python/tests -q --cov=calc_flow --cov-report=term-missing`

Run: `uv run ruff check python python/tests`

Run: `uv run ruff format --check python python/tests`

Expected: PASS; Python facade code is covered and all native core behavior remains covered by Rust's 90% gate.

- [ ] **Step 5: Commit the Python parity milestone**

```bash
git add crates/calc-flow-python/src python/calc_flow python/tests
git commit -m "feat: complete Python runtime API"
```

### Task 21: Move the FastAPI Contract to V2

**Files:**
- Modify: `web-ui/backend/pyproject.toml`
- Modify: `web-ui/backend/src/calc_flow_studio/models.py`
- Modify: `web-ui/backend/src/calc_flow_studio/app.py`
- Modify: `web-ui/backend/tests/test_models.py`
- Modify: `web-ui/backend/tests/test_app.py`

**Interfaces:**
- Consumes: `ProjectDocument`, Rust-backed stores, runtime catalog, Rust-generated project schema.
- Produces: `/api/v2/catalog`, `/api/v2/schema/project`, `/api/v2/projects`, `/api/v2/checkpoints`, and `/api/v2/runs` contracts.

- [ ] **Step 1: Change tests to require only V2 routes and schema**

```python
def test_openapi_contains_v2_and_no_v1_paths(app) -> None:
    paths = app.openapi()["paths"]
    assert "/api/v2/catalog" in paths
    assert "/api/v2/schema/project" in paths
    assert "/api/v2/projects/{project_id}" in paths
    assert "/api/v2/runs/{run_id}/events" in paths
    assert not any(path.startswith("/api/v1/") for path in paths)


def test_project_request_delegates_validation_to_rust() -> None:
    with pytest.raises(ValidationError):
        ProjectCreateRequest.model_validate({"format_version": 1})
```

- [ ] **Step 2: Run backend tests and observe V1 expectations fail**

Run: `cd web-ui/backend && uv run --project . --extra dev pytest tests/test_models.py tests/test_app.py -q`

Expected: FAIL because the application still registers `/api/v1` and imports v1 Pydantic config classes.

- [ ] **Step 3: Replace core Pydantic duplication with V2 route DTOs**

```python
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from calc_flow import ProjectDocument
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InputPayload(StrictModel):
    format: Literal["records", "columns", "arrow_ipc"]
    data: object
    source_id: str | None = None


class RunOptions(StrictModel):
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    memory_limit_mb: int = Field(default=512, ge=64, le=4096)
    max_input_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_rows: int = Field(default=100_000, ge=1)
    output_rows: int = Field(default=1000, ge=1, le=10_000)


class RunRequest(StrictModel):
    inputs: dict[str, InputPayload] = Field(default_factory=dict)
    options: RunOptions | None = None


class ProjectCreateRequest(ProjectDocument):
    pass


class ProjectSummary(StrictModel):
    id: str
    name: str
    description: str
    node_count: int
```

```python
API_PREFIX = "/api/v2"


@app.get(f"{API_PREFIX}/schema/project")
async def project_schema() -> dict[str, object]:
    return json.loads(_native.project_json_schema())


@app.post(f"{API_PREFIX}/projects", response_model=ProjectDocument, status_code=201)
async def create_project(request: ProjectCreateRequest) -> ProjectDocument:
    project = ProjectDocument.model_validate(request.root)
    await asyncio.to_thread(project_store.create_json, project.canonical_json())
    return project


@app.get(f"{API_PREFIX}/catalog")
async def catalog(request: Request) -> list[dict[str, object]]:
    return request.app.state.runtime.catalog()
```

V2 project IDs are client-generated portable identifiers and are immutable after creation; storage rejects conflicts. Return `ProjectDocument` or its `.root` mapping from the remaining project routes and build every path from `API_PREFIX`. Keep import/export byte limits, threadpool delegation for filesystem calls, loopback-only serving, heartbeat behavior, static frontend serving, and existing typed HTTP error mapping.

- [ ] **Step 4: Run model and route tests with coverage**

Run: `cd web-ui/backend && uv run --project . --extra dev pytest tests/test_models.py tests/test_app.py -q --cov=calc_flow_studio --cov-report=term-missing`

Expected: PASS; OpenAPI contains no `/api/v1` routes.

- [ ] **Step 5: Commit**

```bash
git add web-ui/backend/pyproject.toml web-ui/backend/src/calc_flow_studio/models.py web-ui/backend/src/calc_flow_studio/app.py web-ui/backend/tests/test_models.py web-ui/backend/tests/test_app.py
git commit -m "feat: move Studio API to v2"
```

### Task 22: Execute Rust Plans in Bounded Studio Workers

**Files:**
- Modify: `web-ui/backend/src/calc_flow_studio/run_manager.py`
- Modify: `web-ui/backend/tests/test_run_manager.py`
- Modify: `web-ui/backend/tests/test_app.py`

**Interfaces:**
- Consumes: bounded parent-process PyArrow inputs, project canonical JSON, Rust-backed Python runtime.
- Produces: worker execution with existing timeout/CPU/RSS/output/cancellation controls and v2 `RunResult` previews.

- [ ] **Step 1: Update worker tests to assert Rust-backed execution and limits**

```python
def test_process_worker_executes_rust_plan(run_manager, v2_project, table_input) -> None:
    run_id = run_manager.submit(v2_project, {"input": table_input})
    response = run_manager.wait(run_id, timeout=10)
    assert response.status == RunStatus.COMPLETED
    assert response.result["outputs"]["output"]["rows"] == [{"total": 3}]
    assert response.result["datafusion_metrics"][0]["logical_plan"]


def test_parent_rejects_oversized_arrow_before_spawn(run_manager, v2_project) -> None:
    payload = InputPayload(format="records", data=[{"value": "x" * 1024}])
    with pytest.raises(RunManagerError, match="max_input_bytes"):
        prepare_run(v2_project, {"input": payload}, max_input_bytes=16, max_rows=100)
```

- [ ] **Step 2: Run the worker tests and observe v1 compilation failures**

Run: `cd web-ui/backend && uv run --project . --extra dev pytest tests/test_run_manager.py -q`

Expected: FAIL because workers still call v1 `compile_project` and v1 result conversion.

- [ ] **Step 3: Convert only inside the spawned worker after parent bounds checks**

```python
def _execute_worker(project_json: str, prepared_inputs: dict[str, pa.Table], options: RunOptions, output_queue: Any) -> None:
    _apply_resource_limits(options)
    runtime = Runtime()
    register_numpy(runtime)
    register_jax(runtime)
    plan = runtime.compile_project_json(project_json)
    batches = {name: Batch.from_pyarrow(table) for name, table in prepared_inputs.items()}
    result = plan.execute(batches)
    output_queue.put({"type": "completed", "result": _result_payload(result, output_rows=options.output_rows)})
```

Keep `prepare_run` pure and parent-owned: decode records, columns, and bounded Arrow IPC into new `pa.Table` values; check input names against the compiled-project contract using Rust validation metadata; enforce total rows and bytes; never construct or pickle a PyO3 object in the parent. In the worker, register only providers and UDFs referenced by the project. Convert `RunResult` outputs to bounded JSON previews, include node timings and DataFusion plans, and normalize non-JSON scalar values. Preserve spawned processes, concurrent submission caps, timeout polling, RSS enforcement, cancellation, lifecycle cleanup, and forced termination.

- [ ] **Step 4: Run backend coverage and process tests**

Run: `cd web-ui/backend && JAX_PLATFORMS=cpu uv run --project . --extra dev pytest -q --cov=calc_flow_studio --cov-report=term-missing`

Expected: PASS with at least 85% backend coverage.

- [ ] **Step 5: Commit**

```bash
git add web-ui/backend/src/calc_flow_studio/run_manager.py web-ui/backend/tests/test_run_manager.py web-ui/backend/tests/test_app.py
git commit -m "feat: run Rust plans in Studio workers"
```

### Task 23: Regenerate and Adapt the React Studio to V2

**Files:**
- Modify: `web-ui/openapi.json`
- Modify: `web-ui/src/api/schema.d.ts`
- Modify: `web-ui/src/api/client.ts`
- Modify: `web-ui/src/types.ts`
- Modify: `web-ui/src/App.tsx`
- Modify: `web-ui/src/components/CalculationNode.tsx`
- Modify: `web-ui/src/components/NodeInspector.tsx`
- Modify: `web-ui/src/components/ProjectActions.tsx`
- Modify: affected files under `web-ui/src/**/*.test.ts*`
- Modify: `web-ui/e2e/studio.spec.ts`

**Interfaces:**
- Consumes: generated `/api/v2` OpenAPI schema and nested v2 `operator` configuration.
- Produces: v2 project editor, run client, checkpoint controls, event stream, and browser workflow.

- [ ] **Step 1: Update frontend tests to assert V2 requests and project shape**

```typescript
it('creates a v2 expression project', () => {
  const project = blankProject();
  expect(project.format_version).toBe(2);
  expect(project.pipeline.nodes[0].operator).toEqual({
    kind: 'expression',
    expression: 'total = a + b',
    select: [],
    filter: null,
    udfs: [],
  });
});

it('uses only v2 API paths', async () => {
  mockFetch.mockResolvedValue(jsonResponse([]));
  await listProjects();
  expect(mockFetch).toHaveBeenCalledWith('/api/v2/projects', expect.anything());
});
```

- [ ] **Step 2: Regenerate types and observe compile failures**

Run: `cd web-ui && npm run sync:api && npm run build`

Expected: FAIL where components still use flattened v1 node fields and `/api/v1` URLs.

- [ ] **Step 3: Update the canonical blank project and immutable editor transforms**

```typescript
export const blankProject = (): ProjectCreateRequest => ({
  format_version: 2,
  id: crypto.randomUUID(),
  name: 'Untitled flow',
  description: '',
  pipeline: {
    name: 'Main pipeline',
    nodes: [{
      id: 'calculate',
      operator: { kind: 'expression', expression: 'total = a + b', select: [], filter: null, udfs: [] },
      input_ports: [],
      output_ports: [],
      position: { x: 80, y: 100 },
    }],
    edges: [],
    datafusion: { batch_size: 8192, target_partitions: 1 },
  },
  data_sources: [],
  run_options: { max_input_bytes: 10 * 1024 * 1024, max_rows: 100_000, timeout_seconds: 30, memory_limit_mb: 512, output_rows: 1000 },
});
```

```typescript
const API_PREFIX = '/api/v2';

export const updateNodeOperator = (
  project: ProjectDocument,
  nodeId: string,
  update: (operator: OperatorSpec) => OperatorSpec,
): ProjectDocument => ({
  ...project,
  pipeline: {
    ...project.pipeline,
    nodes: project.pipeline.nodes.map((node) =>
      node.id === nodeId ? { ...node, operator: update(node.operator) } : node,
    ),
  },
});

export const listProjects = (): Promise<ProjectSummary[]> =>
  requestJson(`${API_PREFIX}/projects`);
```

Narrow inspector controls on `node.operator.kind`, then call `setProject((current) => updateNodeOperator(current, nodeId, update))`. Preserve React Flow node type maps outside render functions. Update result and metric previews only where generated v2 field types differ. Update the Playwright workflow to create, save, run, observe, checkpoint, and delete a v2 project.

- [ ] **Step 4: Run the complete Studio frontend gate**

Run: `cd web-ui && npm run build && npm test && npm run test:e2e`

Run: `cd web-ui && npm audit --omit=dev`

Expected: every command exits 0 and the generated OpenAPI files have no diff after `npm run sync:api`.

- [ ] **Step 5: Commit**

```bash
git add web-ui/openapi.json web-ui/src web-ui/e2e/studio.spec.ts
git commit -m "feat: adapt Studio to v2 projects"
```

### Task 24: Add Cross-Platform Wheel, Crate, Audit, and Release Gates

**Files:**
- Create: `.github/workflows/release.yml`
- Create: `deny.toml`
- Create: `scripts/smoke_wheel.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/benchmarks.yml`
- Modify: `web-ui/backend/pyproject.toml`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: complete Rust, Python, Studio, and frontend implementation.
- Produces: crates.io package validation, sdist, abi3 Python wheels for every approved platform, Studio wheel, audit reports, and clean-environment smoke tests.

- [ ] **Step 1: Add the failing artifact smoke script**

```python
from __future__ import annotations

import pyarrow as pa

from calc_flow import Batch, PipelineBuilder


plan = PipelineBuilder("wheel-smoke").expression("calculate", "total = a + b").compile()
result = plan.execute({"input": Batch.from_pyarrow(pa.table({"a": [1], "b": [2]}))})
assert result.outputs["output"].to_pyarrow()["total"].to_pylist() == [3]
```

- [ ] **Step 2: Confirm release artifacts are not yet built**

Run: `uv build`

Run: `cargo package -p calc-flow --allow-dirty`

Expected: at least one command fails until Maturin packaging metadata and crate include paths are complete.

- [ ] **Step 3: Add the full release matrix**

```yaml
name: Release artifacts
on:
  workflow_dispatch:
  push:
    tags: ["v2.*"]
permissions:
  contents: read
jobs:
  wheels:
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-latest
            target: x86_64
          - os: ubuntu-latest
            target: aarch64
          - os: macos-13
            target: x86_64
          - os: macos-14
            target: aarch64
          - os: windows-latest
            target: x64
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.13"
      - uses: PyO3/maturin-action@v1
        with:
          command: build
          target: ${{ matrix.target }}
          args: --release --out dist --find-interpreter
          manylinux: "2_28"
      - uses: actions/upload-artifact@v7
        with:
          name: wheel-${{ matrix.os }}-${{ matrix.target }}
          path: dist/*.whl
```

Generate the production workflow once with `maturin generate-ci github`, retain the explicit selected platform matrix above, and pin third-party actions to reviewed commits before merging. Add a Linux sdist job, `cargo package`, `cargo publish --dry-run`, wheel install smoke tests, Studio wheel install tests, `cargo audit`, `cargo deny check`, and `npm audit --omit=dev`. Set the Studio dependency to `calc-flow>=2.0.0a1,<3`. Verify wheels do not contain `src/calc_flow`, `web-ui`, test fixtures, or executable project data.

- [ ] **Step 4: Run local release checks and CI-equivalent verification**

Run: `cargo package -p calc-flow --allow-dirty`

Run: `uv build`

Run: `uv venv .wheel-smoke && uv pip install --python .wheel-smoke dist/calc_flow-*.whl && .wheel-smoke/Scripts/python scripts/smoke_wheel.py` on Windows, or `.wheel-smoke/bin/python scripts/smoke_wheel.py` on Unix.

Run: `cargo deny check`

Expected: package contents are minimal, smoke execution returns total 3, and audits pass.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows deny.toml scripts/smoke_wheel.py pyproject.toml web-ui/backend/pyproject.toml
git commit -m "ci: build Rust v2 release artifacts"
```

### Task 25: Complete Documentation, Remove the Frozen Python Core, and Cut V2

**Files:**
- Delete: `src/calc_flow/`
- Delete: `tests/calc_flow/`
- Replace: `examples/*.py`
- Modify: `README.md`
- Modify: `docs/introduction.md`
- Modify: `docs/api-reference.md`
- Create: `docs/rust-api.md`
- Create: `docs/python-api.md`
- Create: `docs/v2-release.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: all passing implementation and release gates.
- Produces: final v2 documentation, no Python v1 execution path, and release-ready version metadata.

- [ ] **Step 1: Add a failing repository test that forbids the v1 core**

```python
from pathlib import Path


def test_v2_repository_has_no_python_core_implementation() -> None:
    assert not Path("src/calc_flow").exists()
    assert not Path("tests/calc_flow").exists()
    assert Path("python/calc_flow").is_dir()
    assert Path("crates/calc-flow/src/lib.rs").is_file()
```

- [ ] **Step 2: Run it before deletion**

Run: `uv run pytest python/tests/test_repository_layout.py -q`

Expected: FAIL because the frozen v1 source and tests still exist.

- [ ] **Step 3: Remove v1 and publish complete v2 documentation**

Delete the frozen implementation and its old test suite only after Tasks 1–24 pass. Keep `tests/fixtures/v1` as historical semantic evidence. Replace Python examples with v2 builder, SQL, Python UDF, micro-batch, NumPy, and async examples. Add paired Rust examples to `docs/rust-api.md`. State clearly in `docs/v2-release.md`:

```markdown
## Compatibility

Calc Flow 2.0 does not load Calc Flow 1.x project documents or checkpoints.
Recreate projects with the v2 schema and restart stateful processing from a
chosen source boundary. No automated converter is provided.
```

Update `AGENTS.md` commands to include Cargo formatting, Clippy, tests, coverage, docs, Maturin development installs, Python tests, backend tests, frontend tests, e2e, and audits. Bump crate, Python, Studio, and frontend versions from alpha to `2.0.0` in one commit after the release candidate passes.

- [ ] **Step 4: Run the final clean-room verification**

Run: `cargo fmt --all --check`

Run: `cargo clippy --workspace --all-targets --all-features -- -D warnings`

Run: `cargo test --workspace --all-targets`

Run: `cargo llvm-cov --workspace --all-features --fail-under-lines 90`

Run: `cargo doc --workspace --no-deps`

Run: `uv sync --extra dev && uv run maturin develop && JAX_PLATFORMS=cpu uv run pytest python/tests -q`

Run: `uv run ruff check . && uv run ruff format --check .`

Run: `cd web-ui/backend && uv run --project . --extra dev pytest --cov=calc_flow_studio`

Run: `cd web-ui && npm ci && npm run sync:api && npm run build && npm test && npm run test:e2e && npm audit --omit=dev`

Run: `git diff --exit-code -- schemas/project-v2.schema.json web-ui/openapi.json web-ui/src/api/schema.d.ts`

Expected: every command exits 0, Rust core coverage is at least 90%, Studio backend coverage is at least 85%, and generated contracts are clean.

- [ ] **Step 5: Commit and tag**

```bash
git add -A
git commit -m "feat: release Rust-native Calc Flow v2"
git tag v2.0.0
```

---

## Implementation Order and Release Gates

1. Tasks 1–14 are the Rust core program. Task 14 is a hard gate.
2. Tasks 15–20 are the Python API program. Task 20 is a hard gate.
3. Tasks 21–23 migrate Studio without changing its Python/FastAPI ownership.
4. Task 24 proves packaging and platform support.
5. Task 25 is the only task authorized to delete the frozen Python v1 core.

Do not publish an intermediate v2 package that mixes Python v1 execution with Rust v2 execution. Alpha artifacts may be used in CI and internal testing, but the first supported v2 release must pass all five gates above.

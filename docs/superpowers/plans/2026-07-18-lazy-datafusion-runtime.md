# Lazy Run-Scoped DataFusion Runtime Implementation Plan

> **Historical status:** Implemented, retained by the measured evidence in
> the [lazy-runtime handoff](../handoffs/2026-07-18-lazy-datafusion-runtime.md),
> and merged in PR #14. Unchecked boxes preserve the original execution plan.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove DataFusion session-construction cost from external-only NumPy/JAX plans while preserving one isolated session per run for table execution and every public execution contract.

**Architecture:** Keep `DataFusionRuntime` and `OperatorContext` public APIs unchanged. Store validated configuration and resolved native UDFs in a run-local runtime, create its `SessionContext` through a private `OnceLock` only on the first table query, and retain the existing query lock, temporary table cleanup, metrics, close, transaction, rollback, cancellation, and runner paths.

**Tech Stack:** Rust 1.88.0, Rust 2024, Apache DataFusion 54.0.0, Tokio, Criterion 0.8.2, Cargo LLVM coverage, Python 3.13+, PyO3, pytest, Vitest, Playwright, and GitHub Actions.

## Global Constraints

- Treat `docs/superpowers/specs/2026-07-18-lazy-datafusion-runtime-design.md` as the source of truth.
- Use merged-main commit `b333121b282861ea03e006db7a2a232f6a6566c2` as the exact performance baseline.
- Preserve one `DataFusionRuntime` per `ExecutionPlan::execute` and at most one `SessionContext` per runtime.
- Do not reuse or pool a session across runs, plans, or runners.
- Preserve the public `DataFusionRuntime`, `OperatorContext`, `Operator`, `ExecutionPlan`, and `RunResult` APIs.
- Keep configuration and selected-UDF validation eager; defer only infallible `SessionContext` construction and native UDF installation.
- Preserve transactions, snapshots, rollback markers, cancellation, validation, routing, node timings, DataFusion metrics, checkpoints, and runner lifecycle behavior.
- Keep the query lock around first initialization, temporary aliases, planning, execution, metric recording, and alias cleanup.
- Do not change project documents, fingerprints, checkpoints, Python bindings, array providers, Studio APIs, generated schemas, or package versions.
- Do not add timing assertions to tests or CI.
- Start the behavior change with focused failing tests and record the expected failure before implementation.
- Keep Cargo, Maturin, uv, coverage, and benchmark output under `target/` and remove `python/calc_flow/_native*.so` before committing.
- Preserve unrelated user files and stage only the files named by the active task.

---

## Target File Map

- `crates/calc-flow/src/datafusion.rs` — lazy session ownership, UDF preparation, query initialization, and in-file private-state tests.
- `crates/calc-flow/tests/udf.rs` — public-behavior regression test for registering a native UDF after a lazy session already exists.
- `crates/calc-flow/benches/core.rs` — unchanged stable Criterion identities used for before/after evidence.
- `docs/superpowers/handoffs/2026-07-18-lazy-datafusion-runtime.md` — exact environment, commands, estimates, confidence intervals, gate decision, and compatibility evidence.

---

### Task 1: Record the Exact Merged-Main Baseline

**Files:**
- Read: `crates/calc-flow/benches/core.rs`
- Output only: `target/cargo/criterion/`

**Interfaces:**
- Consumes: Criterion cases `execute/datafusion_runtime_new`, `execute/datafusion_runtime_new_register_udfs`, `execute/external_passthrough_1000_rows`, and `execute/expression_1024_rows`.
- Produces: the `lazy-datafusion-before` Criterion baseline used by Task 3.

- [ ] **Step 1: Prove the production tree is still the exact merged-main implementation**

Run:

```bash
git diff --exit-code b333121b282861ea03e006db7a2a232f6a6566c2 -- \
  crates/calc-flow/src \
  crates/calc-flow/benches/core.rs
```

Expected: exit 0 with no output. The branch may contain only the committed design and plan documents.

- [ ] **Step 2: Run the focused correctness baseline**

Run:

```bash
CARGO_TARGET_DIR="$PWD/target/cargo" CARGO_BUILD_JOBS=1 \
  cargo test -p calc-flow --test datafusion --test udf --test pipeline_execute
```

Expected: all existing DataFusion, UDF, and execution tests pass.

- [ ] **Step 3: Save the same-host Criterion baseline**

Run:

```bash
CARGO_TARGET_DIR="$PWD/target/cargo" CARGO_BUILD_JOBS=1 \
  cargo bench -p calc-flow --bench core -- \
  --save-baseline lazy-datafusion-before
```

Expected: all stable Criterion cases complete and write baseline data beneath `target/cargo/criterion/`.

- [ ] **Step 4: Record the baseline point estimates before editing production code**

Read the four `estimates.json` files beneath `target/cargo/criterion/` and copy their slope point estimates and 95% confidence intervals into working notes. Do not commit generated Criterion output.

---

### Task 2: Defer the Run-Local Session Until the First Query

**Files:**
- Modify: `crates/calc-flow/src/datafusion.rs`
- Modify: `crates/calc-flow/tests/udf.rs`

**Interfaces:**
- Consumes: `DataFusionConfig`, `UdfRegistrySnapshot`, `UdfReference`, `ScalarUDF`, and the existing public runtime methods.
- Produces: private production method `DataFusionRuntime::context(&self) -> &SessionContext`; all public signatures remain unchanged and no test-only method is added.

- [ ] **Step 1: Add failing runtime-laziness unit tests**

Append an in-file test module to `crates/calc-flow/src/datafusion.rs`. The tests live in the owning module and inspect its private field directly, so production code needs no test-only probe:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_preparation_and_close_do_not_initialize_a_session() {
        let mut runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        assert!(runtime.context.get().is_none());

        runtime
            .register_udfs(&UdfRegistrySnapshot::default(), &[])
            .unwrap();
        assert!(runtime.context.get().is_none());

        runtime.close();
        assert!(runtime.context.get().is_none());
    }

    #[test]
    fn context_initializes_once_and_reuses_the_same_session() {
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        assert!(runtime.context.get().is_none());

        let first = std::ptr::from_ref(runtime.context());
        assert!(runtime.context.get().is_some());
        let second = std::ptr::from_ref(runtime.context());

        assert_eq!(first, second);
    }
}
```

- [ ] **Step 2: Add the public late-UDF-registration preservation test**

Add this behavior test to `crates/calc-flow/tests/udf.rs` using the existing `constant_udf` and `input` helpers:

```rust
#[tokio::test]
async fn runtime_registers_a_native_udf_after_the_lazy_session_exists() {
    let selected =
        UdfReference::new("rust", "late_value", "1", UdfKind::DataFusionScalar).unwrap();
    let mut registry = UdfRegistry::new();
    registry
        .register_datafusion(selected.clone(), constant_udf("late_value", 17), 0)
        .unwrap();
    let mut runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();

    runtime.evaluate("a", &input(), None).await.unwrap();
    runtime
        .register_udfs(&registry.snapshot(), &[selected])
        .unwrap();

    let output = runtime
        .evaluate("late_value()", &input(), None)
        .await
        .unwrap();
    let values = output.table_payload().unwrap().batches()[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(values.values(), &[17, 17]);
}
```

- [ ] **Step 3: Run RED and record the expected failure**

Run:

```bash
CARGO_TARGET_DIR="$PWD/target/cargo" CARGO_BUILD_JOBS=1 \
  cargo test -p calc-flow --lib --test udf
```

Expected: compilation fails because the current eager `context` field has no `get` method and the private lazy `context()` method does not exist.

- [ ] **Step 4: Replace eager session ownership with private lazy state**

Update the imports and runtime fields in `crates/calc-flow/src/datafusion.rs`:

```rust
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

pub struct DataFusionRuntime {
    config: DataFusionConfig,
    context: OnceLock<SessionContext>,
    selected_udfs: Vec<Arc<ScalarUDF>>,
    query_lock: AsyncMutex<()>,
    metrics: Mutex<Vec<DataFusionQueryMetric>>,
    next_query: AtomicU64,
    closed: AtomicBool,
}
```

Change `new` so it validates configuration but performs no DataFusion work:

```rust
pub fn new(config: DataFusionConfig) -> Result<Self> {
    config.validate()?;
    Ok(Self {
        config,
        context: OnceLock::new(),
        selected_udfs: Vec::new(),
        query_lock: AsyncMutex::new(()),
        metrics: Mutex::new(Vec::new()),
        next_query: AtomicU64::new(1),
        closed: AtomicBool::new(false),
    })
}
```

Update its rustdoc to say the runtime owns a session lazily rather than claiming construction creates the session.

- [ ] **Step 5: Prepare UDFs eagerly without forcing a session**

Keep the current validation and resolution pipeline, then replace direct eager registration with:

```rust
validate_udf_sql_namespace(&selected)?;
if let Some(context) = self.context.get() {
    for (_, udf) in selected {
        context.register_udf(udf.as_ref().clone());
    }
} else {
    self.selected_udfs
        .extend(selected.into_iter().map(|(_, udf)| udf));
}
Ok(())
```

Do not move `ensure_open`, `validate_selected_udfs`, snapshot resolution, or SQL namespace validation into the lazy initializer.

- [ ] **Step 6: Add the single production lazy initializer**

Add these private methods:

```rust
fn context(&self) -> &SessionContext {
    self.context.get_or_init(|| {
        let session = SessionConfig::new()
            .with_batch_size(self.config.batch_size)
            .with_target_partitions(self.config.target_partitions);
        let context = SessionContext::new_with_config(session);
        for udf in &self.selected_udfs {
            context.register_udf(udf.as_ref().clone());
        }
        context
    })
}
```

The private method is called by `sql` in production. Do not add an initialization-state method to the production type.

- [ ] **Step 7: Initialize only inside the locked query path**

In `sql`, keep both `ensure_open` checks and query validation in their current order. Immediately after acquiring `query_lock` and rechecking `ensure_open`, bind the context once:

```rust
let context = self.context();
let mut registrations = TableRegistrations::new(context);
for (alias, batch) in tables {
    registrations.register(alias, batch, node_id)?;
}
```

Use `context.sql(&query)` for planning. Do not call `context()` before the query lock, do not hold a separate synchronous lock across `.await`, and do not change `TableRegistrations` cleanup.

- [ ] **Step 8: Run GREEN focused verification**

Run:

```bash
CARGO_TARGET_DIR="$PWD/target/cargo" CARGO_BUILD_JOBS=1 \
  cargo test -p calc-flow --lib --test datafusion --test udf --test pipeline_execute
CARGO_TARGET_DIR="$PWD/target/cargo" \
  cargo fmt --all --check
CARGO_TARGET_DIR="$PWD/target/cargo" CARGO_BUILD_JOBS=1 \
  cargo clippy -p calc-flow --all-targets --all-features -- -D warnings
git diff --check
```

Expected: the new laziness tests and all existing DataFusion/UDF/execution tests pass; format, Clippy, and whitespace checks are clean.

- [ ] **Step 9: Commit the behavior change**

Run:

```bash
git add \
  crates/calc-flow/src/datafusion.rs \
  crates/calc-flow/tests/udf.rs
git diff --cached --check
git commit -m "perf: defer DataFusion session setup"
```

---

### Task 3: Prove and Record the Performance Result

**Files:**
- Read: `crates/calc-flow/benches/core.rs`
- Create: `docs/superpowers/handoffs/2026-07-18-lazy-datafusion-runtime.md`
- Output only: `target/cargo/criterion/`
- Output only: `target/benchmark-results/`

**Interfaces:**
- Consumes: the saved `lazy-datafusion-before` baseline and contract-v2 array benchmark metadata.
- Produces: a committed evidence handoff with a pass/fail decision against the 10-microsecond and 20% usefulness gate.

- [ ] **Step 1: Compare the candidate against the saved baseline**

Run:

```bash
CARGO_TARGET_DIR="$PWD/target/cargo" CARGO_BUILD_JOBS=1 \
  cargo bench -p calc-flow --bench core -- \
  --baseline lazy-datafusion-before
```

Expected: all Criterion cases complete. The external passthrough should show the intended reduction while the expression control remains functional.

- [ ] **Step 2: Collect compatible Python provider/plan evidence**

Build the current binding and run the 16 relevant contract-v2 cases into one report:

```bash
env -u CONDA_PREFIX \
  VIRTUAL_ENV="$PWD/.venv" \
  CARGO_TARGET_DIR="$PWD/target/cargo" \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv sync --extra dev --extra benchmark
env -u CONDA_PREFIX \
  VIRTUAL_ENV="$PWD/.venv" \
  CARGO_TARGET_DIR="$PWD/target/cargo" \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv run maturin develop --release
env -u CONDA_PREFIX \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
  CALC_FLOW_BENCHMARK_SCALE=overhead \
  JAX_PLATFORMS=cpu \
  uv run --extra benchmark pytest \
  benchmarks/test_array_provider.py benchmarks/test_array_plan.py \
  -q --benchmark-only \
  --benchmark-json=target/benchmark-results/lazy-datafusion.json
```

Expected: 16 cases pass and every report entry contains complete contract-v2 identity documents and fingerprints. Treat high-CoV results as diagnostic rather than a performance claim.

- [ ] **Step 3: Write the evidence handoff with actual measured values**

Create `docs/superpowers/handoffs/2026-07-18-lazy-datafusion-runtime.md` with the title `Lazy DataFusion Runtime Evidence` and these eight second-level sections in order:

1. `Outcome` — state whether both the 10-microsecond and 20% external-passthrough gates passed and whether the implementation is retained.
2. `Revisions` — record baseline `b333121b282861ea03e006db7a2a232f6a6566c2` and the literal 40-character implementation commit from `git rev-parse HEAD`.
3. `Environment` — record OS, architecture, normalized CPU brand, logical CPU count, Rust, Cargo, Python, NumPy, JAX, JAXlib, and every observable power-control setting.
4. `Commands` — copy the exact Criterion, Maturin, and Python benchmark commands executed from this plan.
5. `Criterion comparison` — use a Markdown table with columns `Case`, `Baseline us`, `Candidate us`, `Change us`, `Change %`, and `Candidate 95% CI`. Include the four stable case names from Step 1. Convert Criterion nanoseconds to microseconds, calculate `change_us = candidate_us - baseline_us`, and calculate `change_percent = 100 * change_us / baseline_us`.
6. `Contract-v2 Python evidence` — record the report SHA-256, 16-case count, machine/dependency compatibility result, provider/plan timing ranges, CoV ranges, and whether the existing noise rule permits a claim.
7. `Correctness and API evidence` — record focused test counts, unchanged public method signatures, external empty metrics, first-query UDF behavior, late UDF registration, and concurrent first-query coverage.
8. `Decision` — state the retained or reverted decision and tie it directly to the measured usefulness gate and correctness evidence.

Use literal measured values throughout. Do not claim incompatible comparisons or sum overlapping components.

- [ ] **Step 4: Validate the handoff and commit it**

Run:

```bash
rg -n '^## ' docs/superpowers/handoffs/2026-07-18-lazy-datafusion-runtime.md
git diff --check
```

Expected: `rg` prints the eight required headings in order and `git diff --check` passes. Read the complete file once to verify every reported value is literal and supported by the saved output.

Then run:

```bash
git add docs/superpowers/handoffs/2026-07-18-lazy-datafusion-runtime.md
git diff --cached --check
git commit -m "perf: record lazy DataFusion evidence"
```

---

### Task 4: Run the Complete Repository Verification Matrix

**Files:**
- Verify: entire repository
- Remove if generated: `python/calc_flow/_native*.so`
- Verify unchanged: `schemas/project-v2.schema.json`
- Verify unchanged: `web-ui/openapi.json`
- Verify unchanged: `web-ui/src/api/schema.d.ts`

**Interfaces:**
- Consumes: the complete implementation and evidence commits.
- Produces: exact command evidence that the branch satisfies every repository gate.

- [ ] **Step 1: Run all Rust gates**

Run:

```bash
CARGO_TARGET_DIR="$PWD/target/cargo" cargo fmt --all --check
CARGO_TARGET_DIR="$PWD/target/cargo" CARGO_BUILD_JOBS=1 \
  cargo clippy --workspace --all-targets --all-features -- -D warnings
CARGO_TARGET_DIR="$PWD/target/cargo" CARGO_BUILD_JOBS=1 \
  cargo test --workspace --all-targets --all-features
CARGO_TARGET_DIR="$PWD/target/cargo" CARGO_BUILD_JOBS=1 \
  cargo llvm-cov --workspace --all-features --fail-under-lines 90
CARGO_TARGET_DIR="$PWD/target/cargo" RUSTDOCFLAGS="-D warnings" \
  cargo doc --workspace --all-features --no-deps
```

Expected: every command passes and line coverage is at least 90%.

- [ ] **Step 2: Run Python binding and adapter gates**

Run:

```bash
UV_CACHE_DIR="$PWD/target/uv-cache" uv sync --extra dev
CARGO_TARGET_DIR="$PWD/target/cargo" UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv run maturin develop
JAX_PLATFORMS=cpu UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv run pytest python/tests -q
UV_CACHE_DIR="$PWD/target/uv-cache" uv run ruff check .
UV_CACHE_DIR="$PWD/target/uv-cache" uv run ruff format --check .
```

Expected: all tests and Ruff checks pass.

- [ ] **Step 3: Run Studio backend gates**

Run:

```bash
cd web-ui/backend
UV_CACHE_DIR="$OLDPWD/target/uv-cache" \
  uv run --project . --extra dev pytest --cov=calc_flow_studio
cd ../..
```

Expected: all tests pass and Studio coverage remains at least 85%.

- [ ] **Step 4: Run frontend and generated API gates**

Run:

```bash
cd web-ui
npm ci
npm run sync:api
npm run build
npm test
npm run test:e2e
npm audit --omit=dev
cd ..
```

Expected: generation, build, unit tests, browser workflow, and production dependency audit all pass.

- [ ] **Step 5: Run supply-chain and release-helper gates**

Run:

```bash
CARGO_TARGET_DIR="$PWD/target/cargo" \
  cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177
CARGO_TARGET_DIR="$PWD/target/cargo" cargo deny --locked check
python -m unittest scripts.test_inspect_wheel scripts.test_release_config
```

Expected: audit passes with only the two documented ignores, deny passes every category, and release-helper tests pass.

- [ ] **Step 6: Remove generated native/static output and verify tracked artifacts**

Remove any generated native module and other ignored build output without touching source files:

```bash
find python/calc_flow -maxdepth 1 -name '_native*.so' -delete
git status --short
git diff --exit-code -- \
  schemas/project-v2.schema.json \
  web-ui/openapi.json \
  web-ui/src/api/schema.d.ts
git diff --check
```

Expected: only the intentional committed files differ from `main`; generated contract files have no diff; whitespace is clean; no native module remains.

---

### Task 5: Review, Publish, and Close the Pull Request

**Files:**
- Review: `git diff origin/main...HEAD`
- Publish: branch `feature/lazy-datafusion-runtime`
- Create: GitHub pull request with base `main`

**Interfaces:**
- Consumes: a clean, fully verified branch and complete evidence handoff.
- Produces: an open PR whose exact final head has green CI and no unresolved actionable review thread.

- [ ] **Step 1: Perform independent requirements and code-quality reviews**

Review the exact diff against the design and this plan. Require explicit findings for:

- session lifetime and no cross-run reuse;
- UDF validation timing and late registration;
- first-query concurrency and alias cleanup;
- closed-runtime behavior;
- external-only metrics and plan semantics;
- node-timing attribution;
- benchmark compatibility and claims;
- public API and serialized-contract stability;
- tests covering every changed branch.

Address every Critical, Important, and Minor finding with a focused test where behavior changes, then rerun the affected gates. Commit fixes with imperative summaries under 72 characters.

- [ ] **Step 2: Confirm the intended commit and file scope**

Run:

```bash
git status --short --branch
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
git diff --check origin/main...HEAD
find python/calc_flow -maxdepth 1 -name '_native*.so' -print
```

Expected: a clean branch, only the design, plan, runtime/tests, and evidence files, clean whitespace, and no native module.

- [ ] **Step 3: Push the branch without force**

Run:

```bash
GIT_SSH_COMMAND='ssh -F /dev/null' \
  git push -u origin feature/lazy-datafusion-runtime
```

Expected: the remote branch is created and points at the exact local head.

- [ ] **Step 4: Open the pull request**

Use this title:

```text
perf: defer DataFusion setup for external plans
```

Use a body with these exact sections and replace counts/measurements with final evidence:

```markdown
## Summary

- lazily construct the run-scoped DataFusion session on the first table query while preserving the public runtime and operator contracts
- keep configuration and selected-UDF validation eager, including late registration on an already initialized runtime
- remove measured session setup from external-only NumPy/JAX plans and record same-host Criterion plus contract-v2 evidence

## Test plan

- full Rust format, Clippy, workspace tests, 90% coverage, and rustdoc gates from `AGENTS.md`
- full Python adapter, Studio backend, frontend build/unit/E2E, and dependency audit gates from `AGENTS.md`
- same-host Criterion comparison for runtime construction, external passthrough, and the table-expression control
- compatible contract-v2 NumPy/JAX provider and plan benchmark report
- generated project schema, OpenAPI, TypeScript API, native-module cleanup, and whitespace checks
```

Create the PR through the GitHub connector when available; use `gh pr create` only if connector coverage is insufficient.

- [ ] **Step 5: Monitor exact-head CI and address every failure**

Capture the pushed SHA, inspect all GitHub Actions checks for that SHA, and wait for every required job to reach a terminal state. For a failure, inspect the failing step and logs, reproduce it locally when possible, fix the root cause test-first, push a new commit, and restart exact-head verification. Do not rely on stale green checks from an earlier SHA.

- [ ] **Step 6: Audit and resolve review feedback**

Fetch PR conversation comments, reviews, and GraphQL review threads. Verify every suggestion against source and tests. Implement every actionable issue, reply with evidence, and resolve the thread when fixed. Re-run affected local gates and exact-head CI after every pushed fix.

- [ ] **Step 7: Prove final completion**

The task is complete only when all of the following are simultaneously true:

- the PR is open against `main` and its exact head equals local and remote;
- every required check on that exact head succeeds;
- merge state is clean and GitHub reports the PR mergeable;
- no actionable or unresolved review thread remains;
- the local worktree is clean;
- the evidence handoff contains no placeholders or unsupported claims;
- no generated native module or tracked generated-artifact diff remains.

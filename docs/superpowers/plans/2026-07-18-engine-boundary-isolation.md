# Engine Boundary Isolation Implementation Plan

**Historical status:** Implemented and merged in PR #14. Task 3 Step 4 was
rejected by the GC ownership test; the original checklist and rejection record
are preserved below.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make external NumPy/JAX execution independent of DataFusion runtime resources and configuration while preserving mixed graphs and existing table semantics.

**Architecture:** Replace the single trait-object node representation with a classified `OperatorDefinition` that dispatches external operators through an engine-neutral `OperatorContext` and built-in table operators through a private DataFusion seam. Compile optional `TablePlanResources` only when at least one table operator exists, and include DataFusion configuration in validation and fingerprints only for those plans.

**Tech Stack:** Rust 1.88.0, Rust 2024, Tokio, Apache DataFusion 54.0.0, PyO3 0.28.3, Python 3.13+, Criterion 0.8.2, pytest-benchmark 5.2.3.

## Global Constraints

- Preserve immutable `Batch` values, deterministic `BTreeMap` ordering, run transactions, rollback, cancellation, node timings, runners, sinks, and checkpoints.
- External/array operator code must not receive `DataFusionRuntime` through its execution context.
- Table-only and mixed plans retain one lazy run-scoped DataFusion session and eager configuration/UDF validation.
- Format-v2 projects remain readable and writable; array-only semantic fingerprints omit inactive DataFusion settings.
- `RunResult::datafusion_metrics` remains public and is empty for external-only plans.
- NumPy/JAX remain optional Python providers; do not add a Rust array evaluator or unchecked execution path.
- Do not split Cargo crates or introduce project format v3 in this implementation.
- Every production behavior change begins with a focused failing test.
- Never leave `python/calc_flow/_native*.so` in source.

---

### Task 1: Classify Table and External Operators

**Files:**
- Modify: `crates/calc-flow/src/operator.rs`
- Modify: `crates/calc-flow/src/pipeline.rs`
- Modify: `crates/calc-flow/src/lib.rs`
- Modify: `crates/calc-flow/tests/operator.rs`
- Modify: `crates/calc-flow/tests/support/mod.rs`
- Modify: `crates/calc-flow/tests/pipeline_execute.rs`
- Modify: `crates/calc-flow/tests/config.rs`
- Modify: `crates/calc-flow/benches/core.rs`
- Modify: `crates/calc-flow-python/src/provider.rs`

**Interfaces:**
- Consumes: existing `Operator`, `ExpressionOperator`, `SqlOperator`, `DataFusionRuntime`, `RunContext`, and `PipelineBuilder::add_node` call sites.
- Produces: engine-neutral `OperatorContext { run }`, public `OperatorDefinition`, `From<Box<T>>` conversions, and classified node dispatch without runtime downcasts.

- [x] **Step 1: Write failing context and plan-classification tests**

Change external test operators to construct only `OperatorContext { run: &run }`, then add these assertions to `crates/calc-flow/tests/pipeline_execute.rs`:

```rust
#[tokio::test]
async fn external_only_plan_requires_no_table_engine() {
    let probe = Arc::new(Probe::default());
    let plan = PipelineBuilder::new("external only")
        .unwrap()
        .add_node(
            "external",
            Box::new(TestOperator::transform("external", Action::Pass, probe)),
        )
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();

    assert!(!plan.requires_datafusion());
    assert_eq!(plan.datafusion_config(), None);
}
```

Add a mixed classification test using one `ExpressionOperator` and one
external `TestOperator`:

```rust
#[test]
fn mixed_plan_requires_the_table_engine() {
    let probe = Arc::new(Probe::default());
    let plan = PipelineBuilder::new("mixed")
        .unwrap()
        .add_node(
            "table",
            Box::new(
                ExpressionOperator::new("table", "b = a + 1", vec![], None, vec![])
                    .unwrap(),
            ),
        )
        .unwrap()
        .add_node(
            "external",
            Box::new(TestOperator::transform("external", Action::Pass, probe)),
        )
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();

    assert!(plan.requires_datafusion());
    assert_eq!(plan.datafusion_config(), Some(DataFusionConfig::default()));
}
```

- [x] **Step 2: Run tests and confirm the RED state**

Run:

```bash
CARGO_TARGET_DIR=target/cargo cargo test -p calc-flow --test pipeline_execute external_only_plan_requires_no_table_engine
CARGO_TARGET_DIR=target/cargo cargo test -p calc-flow --test pipeline_execute mixed_plan_requires_the_table_engine
```

Expected: compilation fails because `requires_datafusion` is absent,
`datafusion_config` does not return `Option`, and `OperatorContext` still
requires a DataFusion field.

- [x] **Step 3: Introduce the classified operator definition**

In `operator.rs`, remove DataFusion from the external context:

```rust
pub struct OperatorContext<'a> {
    pub run: &'a RunContext,
}
```

Move built-in table processing into inherent methods accepting explicit table
resources:

```rust
impl ExpressionOperator {
    pub fn configuration(&self) -> JsonMap {
        BTreeMap::from([
            (
                "expression".into(),
                self.expression
                    .as_ref()
                    .map_or(Value::Null, |value| Value::String(value.clone())),
            ),
            (
                "filter_expression".into(),
                self.filter_expression
                    .as_ref()
                    .map_or(Value::Null, |value| Value::String(value.clone())),
            ),
            (
                "select".into(),
                Value::Array(self.select.iter().cloned().map(Value::String).collect()),
            ),
            (
                "udfs".into(),
                Value::Array(self.udfs.iter().map(udf_configuration).collect()),
            ),
        ])
    }

    pub fn udf_references(&self) -> Vec<UdfReference> { self.udfs.clone() }

    pub(crate) async fn process_table(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        run: &RunContext,
        datafusion: &DataFusionRuntime,
    ) -> Result<BTreeMap<String, Batch>> {
        run.check_cancelled()?;
        let input = required_input(inputs, "input", self.name(), run.node_id())?;
        self.input_ports[0].validate(input, &format!("{}.input", self.name))?;
        let tables = BTreeMap::from([("input".into(), input.clone())]);
        let output = datafusion.sql(&self.query, &tables, run.node_id()).await?;
        run.check_cancelled()?;
        Ok(BTreeMap::from([("output".into(), output)]))
    }
}
```

Add the equivalent methods to `SqlOperator`, remove their `Operator`
implementations, and add:

```rust
pub enum OperatorDefinition {
    External(Box<dyn Operator>),
    Expression(ExpressionOperator),
    Sql(SqlOperator),
}

impl<T> From<Box<T>> for OperatorDefinition
where
    T: Operator + 'static,
{
    fn from(value: Box<T>) -> Self {
        Self::External(value)
    }
}

impl From<Box<dyn Operator>> for OperatorDefinition {
    fn from(value: Box<dyn Operator>) -> Self {
        Self::External(value)
    }
}

impl From<Box<ExpressionOperator>> for OperatorDefinition {
    fn from(value: Box<ExpressionOperator>) -> Self {
        Self::Expression(*value)
    }
}

impl From<Box<SqlOperator>> for OperatorDefinition {
    fn from(value: Box<SqlOperator>) -> Self {
        Self::Sql(*value)
    }
}
```

Implement delegating `name`, `input_ports`, `output_ports`, `configuration`,
`udf_references`, `snapshot`, `restore`, `reset`, `requires_datafusion`, and
`process` methods. `process` matches the enum, constructs only
`OperatorContext { run }` for `External`, and requires the supplied
`DataFusionRuntime` for `Expression`/`Sql`.

- [x] **Step 4: Store classified operators in pipeline nodes**

Change `NodeDefinition` and `CompiledNode` to own `OperatorDefinition`, make
`PipelineBuilder::add_node` generic, and keep all call sites source-compatible:

```rust
pub fn add_node<O>(mut self, node_id: &str, operator: O) -> Result<Self>
where
    O: Into<OperatorDefinition>,
{
    let operator = operator.into();
    if node_id.is_empty() {
        return Err(CalcFlowError::Compile {
            message: "node ID must not be empty".into(),
        });
    }
    if self.nodes.contains_key(node_id) {
        return Err(CalcFlowError::Compile {
            message: format!("duplicate node {node_id}"),
        });
    }
    self.nodes.insert(
        node_id.into(),
        NodeDefinition {
            node_id: node_id.into(),
            operator,
        },
    );
    Ok(self)
}
```

Re-export `OperatorDefinition` from `lib.rs`. Update the execution loop to call
the enum dispatcher with the run context and optional table runtime.

In `config.rs`, make the operator match return `OperatorDefinition` directly:

```rust
let operator = match &node.operator {
    OperatorSpec::Expression {
        expression,
        select,
        filter,
        udfs: references,
    } => {
        let (inputs, outputs) =
            builtin_ports(inputs, outputs, &["input"], &["output"], BatchKind::Table)?;
        OperatorDefinition::Expression(
            ExpressionOperator::new(
                &node.id,
                expression,
                select.clone(),
                filter.clone(),
                references.clone(),
            )?
            .with_ports(
                inputs.into_iter().next().unwrap(),
                outputs.into_iter().next().unwrap(),
            )?,
        )
    }
    OperatorSpec::Sql {
        query,
        aliases,
        udfs: references,
    } => {
        let expected = aliases.iter().map(String::as_str).collect::<Vec<_>>();
        let (inputs, outputs) =
            builtin_ports(inputs, outputs, &expected, &["output"], BatchKind::Table)?;
        OperatorDefinition::Sql(
            SqlOperator::new(&node.id, query, aliases.clone(), references.clone())?
                .with_ports(inputs, outputs.into_iter().next().unwrap())?,
        )
    }
    OperatorSpec::External { provider, name, version, options } => {
        let spec = ExternalOperatorSpec::new(provider, name, version, options.clone())?;
        OperatorDefinition::External(
            providers.resolve(provider, name, version)?.create(&spec, inputs, outputs)?,
        )
    }
};
```

Use the existing validated field and port values in the expression and SQL
arms; the discriminant must be selected from `OperatorSpec`, not from runtime
downcasting.

- [x] **Step 5: Update direct built-in and external operator tests**

Change the three direct built-in execution tests in `tests/operator.rs` to
exercise compiled plans, because `process_table` is a private core seam.
Construct external contexts with only `run`. Remove the obsolete test casting
built-ins to `&dyn Operator`; assert their inherent metadata methods instead.

- [x] **Step 6: Run focused tests and confirm GREEN**

Run:

```bash
CARGO_TARGET_DIR=target/cargo cargo test -p calc-flow --test operator --test pipeline_compile --test pipeline_execute --test config
CARGO_TARGET_DIR=target/cargo cargo test -p calc-flow-python provider::tests
```

Expected: all selected tests pass with no warning.

- [x] **Step 7: Commit the classified execution seam**

```bash
git add crates/calc-flow/src/operator.rs crates/calc-flow/src/pipeline.rs \
  crates/calc-flow/src/lib.rs crates/calc-flow/tests/operator.rs \
  crates/calc-flow/tests/support/mod.rs crates/calc-flow/tests/pipeline_execute.rs \
  crates/calc-flow/tests/config.rs crates/calc-flow/benches/core.rs \
  crates/calc-flow-python/src/provider.rs
git commit -m "refactor: isolate external operator context"
```

### Task 2: Make Table Resources Conditional

**Files:**
- Modify: `crates/calc-flow/src/pipeline.rs`
- Modify: `crates/calc-flow/src/config.rs`
- Modify: `crates/calc-flow/tests/config.rs`
- Modify: `crates/calc-flow/tests/pipeline_compile.rs`
- Modify: `crates/calc-flow/tests/pipeline_execute.rs`

**Interfaces:**
- Consumes: `OperatorDefinition::requires_datafusion`, selected UDF catalog, and engine-neutral execution from Task 1.
- Produces: `TablePlanResources`, `ExecutionPlan::requires_datafusion()`, `ExecutionPlan::datafusion_config() -> Option<DataFusionConfig>`, conditional validation, and conditional fingerprints.

- [x] **Step 1: Write failing array-only contract tests**

Add to `crates/calc-flow/tests/config.rs`:

```rust
#[test]
fn unused_datafusion_config_does_not_affect_external_only_plans() {
    let providers = provider_registry();
    let udfs = UdfRegistry::new().snapshot();
    let original = external_project();
    let mut changed = original.clone();
    changed.pipeline.datafusion = DataFusionConfig {
        batch_size: 0,
        target_partitions: 0,
    };

    let original_plan = compile_project(&original, &providers, &udfs).unwrap();
    let changed_plan = compile_project(&changed, &providers, &udfs).unwrap();

    assert!(!original_plan.requires_datafusion());
    assert_eq!(original_plan.fingerprint(), changed_plan.fingerprint());
    assert_eq!(changed_plan.datafusion_config(), None);
}
```

Add a table control proving the same invalid configuration remains rejected.

- [x] **Step 2: Run tests and confirm the RED state**

Run:

```bash
CARGO_TARGET_DIR=target/cargo cargo test -p calc-flow --test config unused_datafusion_config_does_not_affect_external_only_plans
```

Expected: compilation returns the existing DataFusion out-of-range error.

- [x] **Step 3: Add optional table plan resources**

Replace unconditional plan fields with:

```rust
pub(crate) struct TablePlanResources {
    config: DataFusionConfig,
    udfs: UdfRegistrySnapshot,
    selected_udfs: Vec<UdfReference>,
}

pub struct ExecutionPlan {
    pub(crate) name: String,
    pub(crate) nodes: Vec<CompiledNode>,
    pub(crate) external_inputs: BTreeMap<String, PortEndpoint>,
    pub(crate) external_outputs: BTreeMap<String, PortEndpoint>,
    pub(crate) fingerprint: String,
    pub(crate) run_lock: tokio::sync::Mutex<()>,
    lease_state: StdMutex<LeaseState>,
    operation_state: StdMutex<OperationState>,
    table: Option<TablePlanResources>,
}
```

Add public accessors:

```rust
pub const fn requires_datafusion(&self) -> bool {
    self.table.is_some()
}

pub const fn datafusion_config(&self) -> Option<DataFusionConfig> {
    match &self.table {
        Some(table) => Some(table.config),
        None => None,
    }
}
```

In `compile`, calculate `requires_datafusion` from nodes before validating the
configuration. Build `TablePlanResources` only when true.

- [x] **Step 4: Make project validation conditional**

In `config.rs`, add one shared helper that detects built-in table operator
specifications:

```rust
fn project_requires_datafusion(project: &ProjectSpec) -> bool {
    project.pipeline.nodes.iter().any(|node| {
        matches!(node.operator, OperatorSpec::Expression { .. } | OperatorSpec::Sql { .. })
    })
}
```

Call `validate_datafusion_config` only when the helper is true. Continue
deserializing and preserving the field for format-v2 round trips.

- [x] **Step 5: Omit inactive table settings from fingerprints**

Change `graph_fingerprint` to accept `Option<DataFusionConfig>`. Start with the
common graph object, then insert `datafusion` only for `Some(config)`. Keep the
table-plan JSON projection byte-for-byte equivalent so existing table
fingerprints do not change.

- [x] **Step 6: Execute with optional table resources**

In `execute_unlocked`, prepare a runtime only for `Some(table)`:

```rust
let mut runtime = match &self.table {
    Some(table) => {
        let mut runtime = DataFusionRuntime::new(table.config)?;
        runtime.register_udfs(&table.udfs, &table.selected_udfs)?;
        Some(runtime)
    }
    None => None,
};
let execution = self.execute_nodes(&inputs, &context, runtime.as_ref()).await;
let datafusion_metrics = runtime.as_ref().map_or_else(Vec::new, DataFusionRuntime::metrics);
if let Some(runtime) = &runtime {
    runtime.close();
}
```

Pass `runtime.as_ref()` through node dispatch; a table node without a runtime
returns `CalcFlowError::Internal`.

- [x] **Step 7: Run focused correctness tests and confirm GREEN**

Run:

```bash
CARGO_TARGET_DIR=target/cargo cargo test -p calc-flow --test config --test pipeline_compile --test pipeline_execute --test datafusion --test udf
```

Expected: all selected tests pass; array-only invalid DataFusion settings are
inactive while table controls still reject them.

- [x] **Step 8: Commit conditional resources and fingerprints**

```bash
git add crates/calc-flow/src/pipeline.rs crates/calc-flow/src/config.rs \
  crates/calc-flow/tests/config.rs crates/calc-flow/tests/pipeline_compile.rs \
  crates/calc-flow/tests/pipeline_execute.rs
git commit -m "refactor: activate DataFusion only for table plans"
```

### Task 3: Remove Repeated Python Boundary Work

**Files:**
- Modify: `crates/calc-flow-python/src/provider.rs`
- Modify: `crates/calc-flow-python/src/pipeline.rs`

**Interfaces:**
- Consumes: engine-neutral external provider context and existing Python GC-safe roots.
- Produces: creation-time `options_json` encoding while retaining independently
  clearable output payload roots.

- [x] **Step 1: Write a failing provider encoding unit test**

Add inside `provider.rs` tests:

```rust
#[test]
fn operator_creation_preencodes_provider_options() {
    let options = BTreeMap::from([("nested".into(), json!({"value": [1, 2, 3]}))]);
    assert_eq!(
        encode_provider_options(&options).unwrap(),
        r#"{"nested":{"value":[1,2,3]}}"#
    );
}
```

- [x] **Step 2: Run the unit test and confirm RED**

Run:

```bash
CARGO_TARGET_DIR=target/cargo cargo test -p calc-flow-python operator_creation_preencodes_provider_options
```

Expected: compilation fails because `encode_provider_options` is absent.

- [x] **Step 3: Encode immutable options once**

Add:

```rust
fn encode_provider_options(options: &calc_flow::JsonMap) -> calc_flow::Result<String> {
    serde_json::to_string(options).map_err(|source| calc_flow::CalcFlowError::Format {
        message: source.to_string(),
    })
}
```

Store both the original `JsonMap` for deterministic configuration and the
encoded string for callback execution. `call_python_operator` calls
`json_to_python(py, options_json)` so each execution still gets an independent
nested value.

- [x] **Step 4: Test and reject shared output payload roots**

The proposed change replaced:

```rust
PyBatch::from_inner_python(py, batch)?
```

with:

```rust
PyBatch::from_inner(batch)
```

The GC ownership test then retained all 100 cycles because `RunResult` and the
returned `Batch` shared one Rust `Arc<PythonPayload>`. The change was reverted;
both GC containers retain independently clearable Python roots.

- [x] **Step 5: Run Python ownership and mutation tests**

Build the binding, then run:

```bash
env -u CONDA_PREFIX VIRTUAL_ENV="$PWD/.venv" \
  CARGO_TARGET_DIR="$PWD/target/cargo" UV_CACHE_DIR="$PWD/target/uv-cache" \
  .venv/bin/maturin develop --release
JAX_PLATFORMS=cpu VIRTUAL_ENV="$PWD/.venv" \
  .venv/bin/pytest python/tests/test_array.py python/tests/test_pipeline.py -q
```

Expected: all selected tests pass, including provider option mutation, output
ownership, blocking/async execution, and GC traversal cases.

- [x] **Step 6: Remove the generated native module and commit**

```bash
rm -f python/calc_flow/_native*.so
git add crates/calc-flow-python/src/provider.rs crates/calc-flow-python/src/pipeline.rs
git commit -m "perf: reduce external provider boundary work"
```

### Task 4: Record Isolation Benchmark Evidence

**Files:**
- Modify: `crates/calc-flow/benches/core.rs`
- Create: `docs/superpowers/handoffs/2026-07-18-engine-boundary-isolation.md`
- Modify: `docs/introduction.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: conditional table resources and existing Criterion external passthrough control.
- Produces: an external-only plan preparation benchmark, same-host evidence, and corrected architecture documentation.

- [x] **Step 1: Add a failing benchmark compilation reference**

Add `execute/external_passthrough_1000_rows` assertions to the handoff template
only after the benchmark command produces a point estimate. No timing value is
hard-coded into tests.

- [x] **Step 2: Extend the benchmark control**

Retain `execute/external_passthrough_1000_rows` and add
`execute/external_plan_table_requirement` as a pure accessor/control:

```rust
c.bench_function("execute/external_plan_table_requirement", |b| {
    b.iter(|| black_box(plan.requires_datafusion()));
});
```

The result must remain `false`; the timing is informational.

- [x] **Step 3: Run same-host Criterion cases**

Run:

```bash
CARGO_TARGET_DIR=target/cargo CARGO_BUILD_JOBS=1 \
  cargo bench -p calc-flow --bench core -- \
  'execute/(datafusion_runtime_new|datafusion_runtime_new_register_udfs|expression_1024_rows|external_passthrough_1000_rows|external_plan_table_requirement)'
```

Expected: every case completes and the external requirement accessor returns
the compiled false value under `black_box`.

- [x] **Step 4: Run contract-v2 NumPy/JAX diagnostics**

After rebuilding the release binding, run:

```bash
CALC_FLOW_BENCHMARK_SCALE=overhead JAX_PLATFORMS=cpu \
  VIRTUAL_ENV="$PWD/.venv" .venv/bin/pytest \
  benchmarks/test_array_provider.py benchmarks/test_array_plan.py \
  -q --benchmark-only \
  --benchmark-json=target/benchmark-results/engine-isolation.json
```

Record report SHA-256, case count, fingerprint compatibility, means, CoVs, and
the 5% noise-rule decision without claiming a noisy improvement.

- [x] **Step 5: Correct architecture documentation**

Update the docs to state that external-only runs own no DataFusion runtime,
table/mixed runs own one lazy run-scoped session, and engine-neutral graph
lifecycle remains shared.

- [x] **Step 6: Write the evidence handoff**

Record exact head, environment, commands, correctness counts, Criterion point
estimates/confidence intervals, Python report identity, noise decision,
fingerprint impact, and retained invariants. Do not sum overlapping benchmark
components.

- [x] **Step 7: Remove generated native modules and commit evidence**

```bash
rm -f python/calc_flow/_native*.so
git add crates/calc-flow/benches/core.rs docs/introduction.md README.md \
  docs/superpowers/handoffs/2026-07-18-engine-boundary-isolation.md
git commit -m "perf: record isolated engine evidence"
```

### Task 5: Full Verification and PR Update

**Files:**
- Modify only files required by verification failures attributable to this change.

**Interfaces:**
- Consumes: all implementation tasks and repository guidance commands.
- Produces: a clean exact head, narrow commits, pushed branch, and updated PR #14 evidence.

- [ ] **Step 1: Run Rust verification**

```bash
CARGO_TARGET_DIR=target/cargo cargo fmt --all --check
CARGO_TARGET_DIR=target/cargo cargo clippy --workspace --all-targets --all-features -- -D warnings
CARGO_TARGET_DIR=target/cargo cargo test --workspace --all-targets --all-features
CARGO_TARGET_DIR=target/cargo cargo llvm-cov --workspace --all-features --fail-under-lines 90
CARGO_TARGET_DIR=target/cargo RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
```

Expected: every command exits zero and coverage is at least 90%.

- [ ] **Step 2: Run Python verification**

```bash
UV_CACHE_DIR=target/uv-cache uv sync --extra dev
env -u CONDA_PREFIX VIRTUAL_ENV="$PWD/.venv" \
  CARGO_TARGET_DIR="$PWD/target/cargo" UV_CACHE_DIR="$PWD/target/uv-cache" \
  .venv/bin/maturin develop --release
JAX_PLATFORMS=cpu VIRTUAL_ENV="$PWD/.venv" .venv/bin/pytest python/tests -q
VIRTUAL_ENV="$PWD/.venv" .venv/bin/ruff check .
VIRTUAL_ENV="$PWD/.venv" .venv/bin/ruff format --check .
rm -f python/calc_flow/_native*.so
```

Expected: all tests and Ruff checks pass, and no generated native module
remains.

- [ ] **Step 3: Run Studio backend and frontend verification**

Run the backend coverage suite, `npm ci`, API sync, build, unit tests, e2e tests,
and production dependency audit exactly as listed in `AGENTS.md`.

- [ ] **Step 4: Run supply-chain and artifact helper verification**

Run `cargo audit`, `cargo deny --locked check`, and the release helper unit
tests exactly as listed in `AGENTS.md`.

- [ ] **Step 5: Verify generated contracts and diff hygiene**

```bash
git diff --exit-code -- schemas/project-v2.schema.json web-ui/openapi.json web-ui/src/api/schema.d.ts
git diff --check
find python/calc_flow -maxdepth 1 -type f -name '_native*.so' -print -quit | grep -q . && exit 1 || true
git status --short
```

Expected: generated contracts are unchanged, diff checks pass, no native module
exists, and only intended files are modified before the final commit.

- [ ] **Step 6: Commit any verified final adjustments**

Stage only attributable files, inspect `git diff --cached --name-status`, run
`git diff --cached --check`, and commit with an imperative summary under 72
characters.

- [ ] **Step 7: Push and update PR #14**

Push `feature/lazy-datafusion-runtime`, update the PR title/body to explain
runtime and contract isolation plus the checkpoint fingerprint impact, and
verify checks against the exact pushed head SHA.

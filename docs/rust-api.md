# Rust API

The `calc-flow` crate is the implementation of Calc Flow 2.0. Its public
surface is re-exported from `calc_flow`; no Python process is required.

## Build and document

```bash
cargo add calc-flow@2.0.0
cargo test -p calc-flow --all-targets
RUSTDOCFLAGS="-D warnings" cargo doc -p calc-flow --no-deps
```

## Expression pipeline

This is the canonical first example, exercised by
[`crates/calc-flow/examples/expression_pipeline.rs`](../crates/calc-flow/examples/expression_pipeline.rs)
and a true twin of the Python quickstart:

```rust
use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, ExecutionOptions, ExpressionOperator, PipelineBuilder, UdfRegistry,
};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plan = PipelineBuilder::new("totals")?
        .add_node(
            "calculate",
            Box::new(ExpressionOperator::new(
                "calculate",
                "total = a + b",
                Vec::new(),
                None,
                Vec::new(),
            )?),
        )?
        .compile_batch(&UdfRegistry::new().snapshot())?;
    let input = RecordBatch::try_from_iter(vec![
        (
            "a",
            Arc::new(Int64Array::from(vec![1, 3])) as Arc<dyn datafusion::arrow::array::Array>,
        ),
        ("b", Arc::new(Int64Array::from(vec![2, 4])) as _),
    ])?;
    let result = plan
        .execute(
            BTreeMap::from([(
                "input".into(),
                Batch::table(vec![input], BatchMetadata::default())?,
            )]),
            ExecutionOptions::default(),
        )
        .await?;
    let output = result.outputs["output"].table_payload()?;
    let totals = output.batches()[0]
        .column_by_name("total")
        .expect("expression output contains total")
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("total is an Int64 column");

    assert_eq!(totals.values(), &[3, 7]);
    println!("calculated totals: {totals:?}");
    Ok(())
}
```

`PipelineBuilder` consumes and returns itself, so partially built graphs cannot
be mutated through aliases. Compilation validates ports and topology and binds
an immutable `UdfRegistrySnapshot`. `BatchExecutionPlan::execute` is
asynchronous and accepts only `Batch` values.

### Engine boundary

`ExpressionOperator` and `SqlOperator` are built-in operators. Each implements
both `BatchOperator` and `StreamOperator`; `UnionOperator` and
`WindowAggregateOperator` are stream-only. Add them through
`PipelineBuilder::add_node`, which accepts a `NodeOperator` conversion from a
boxed built-in, custom batch operator, or custom stream operator.

Every operator implements `OperatorMetadata`. Custom finite operators implement
`BatchOperator`, whose `BatchOperatorContext` carries the run-scoped context;
custom continuous operators implement `StreamOperator`, whose
`StreamOperatorContext` carries the stream-job context, operator identity,
current input watermark, and late-row counters. External operators resolve
through `ProviderRegistry` and lifecycle-specific factories, covered below.

`BatchExecutionPlan::datafusion_config()` returns `Option<DataFusionConfig>`:
`None` means the plan is external-only and owns no table resources, so its runs
create no DataFusion session or UDF snapshot.
`BatchExecutionPlan::requires_datafusion()` returns the classification alone.

## SQL operators

`SqlOperator::new` declares the input aliases that become operator ports. It
accepts one read-only `SELECT` or CTE. Connect upstream node outputs to those
aliases with `Edge`/`PortEndpoint`, or leave them unconnected as graph inputs.
DDL, DML, utility commands, and multiple statements are rejected before
execution. The canonical SQL example is
[`crates/calc-flow/examples/sql_join.rs`](../crates/calc-flow/examples/sql_join.rs),
which joins `orders` to `fees` for `net = amount - fee`:

```rust
use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, ExecutionOptions, PipelineBuilder, SqlOperator, UdfRegistry,
};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plan = PipelineBuilder::new("orders-and-fees")?
        .add_node(
            "join",
            Box::new(SqlOperator::new(
                "join",
                "SELECT orders.order_id, orders.amount - fees.fee AS net \
                 FROM orders JOIN fees ON orders.order_id = fees.order_id \
                 ORDER BY orders.order_id",
                vec!["orders".to_string(), "fees".to_string()],
                Vec::new(),
            )?),
        )?
        .compile_batch(&UdfRegistry::new().snapshot())?;
    let orders = RecordBatch::try_from_iter(vec![
        (
            "order_id",
            Arc::new(Int64Array::from(vec![1, 2, 3])) as Arc<dyn datafusion::arrow::array::Array>,
        ),
        ("amount", Arc::new(Int64Array::from(vec![75, 120, 40])) as _),
    ])?;
    let fees = RecordBatch::try_from_iter(vec![
        (
            "order_id",
            Arc::new(Int64Array::from(vec![1, 2, 3])) as Arc<dyn datafusion::arrow::array::Array>,
        ),
        ("fee", Arc::new(Int64Array::from(vec![5, 12, 4])) as _),
    ])?;
    let result = plan
        .execute(
            BTreeMap::from([
                (
                    "orders".into(),
                    Batch::table(vec![orders], BatchMetadata::default())?,
                ),
                (
                    "fees".into(),
                    Batch::table(vec![fees], BatchMetadata::default())?,
                ),
            ]),
            ExecutionOptions::default(),
        )
        .await?;
    let output = result.outputs["output"].table_payload()?;
    let net = output.batches()[0]
        .column_by_name("net")
        .expect("sql output contains net")
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("net is an Int64 column");

    assert_eq!(net.values(), &[70, 108, 36]);
    println!("net amounts: {net:?}");
    Ok(())
}
```

## Stream compilation and the private continuous runtime

`PipelineBuilder::compile_stream` produces an immutable
`StreamExecutionPlan`. The plan records stable edges, source and sink binding
slots, stream requirements, and the semantic fingerprint, but it does not
execute directly through a public source-driven runner.

The crate-private source-driven runtime consumes that plan after a whole-job
preflight. It runs one task per source, compiled operator, and graph output;
preserves per-ingress FIFO; routes source control through a job-scoped progress
driver; and converges source, operator, sink, queue, reservation, driver, and
reaper state on every terminal path. The progress driver owns watermark
generation, idle/reactivation, deterministic timers, and multi-ingress
progress.

`WindowSpec` declares fixed UTC tumbling or hopping windows and ordered
aggregates. `WindowAggregateOperator` is stream-only: it updates incremental
state, classifies late row-window assignments against the current input
watermark, emits closed windows in deterministic order, and snapshots retained
Arrow IPC deltas. `StateBackend` opens an exclusive lineage session;
`LocalStateBackend` provides immutable checksum-verified segment publication.
`CheckpointManifest` is the strict bounded v3 state-manifest contract. Barrier
coordination cuts live sources, aligns operator ingresses, stages operator
state, and publishes the manifest last as the sole durable recovery truth.
Transactional sinks commit only after manifest durability. Recovery restores
mixed ended/live jobs, and terminal checkpoints capture post-`on_end` state
without forwarding a post-end barrier. Exactly-once compatibility is proved
for each requested graph output before lifecycle work; checkpointed
ordinary-sink outputs remain at least once.

Launch failure and cancellation settle source, operator, sink, queue,
progress, state, checkpoint transaction, and reaper ownership. Cleanup
failures and panics remain typed secondary diagnostics and do not replace the
primary outcome.

This runtime is not exported. The existing public v2 `StreamingRunner` and
`MicroBatchRunner` retain their signatures and formed-batch behavior. Public
source-driven runner integration is not available. The private checkpoint
work adds no public API. Private task panics surface through the non-exhaustive
`CalcFlowError::TaskPanicked { task_id, message }` variant.

`edge_channel` retains its public signature and `EdgeBudget` retains the fields
`max_rows` and `max_bytes`. For `EdgeBudget::new(R, B)`, envelope count and
charged rows are each independently capped at `R`, while charged bytes are
capped at `B`. Direct callers must choose
`R >= max(required_row_limit, required_simultaneous_messages)`.

## Batches

- `Batch::table(Vec<RecordBatch>, BatchMetadata)` creates an Arrow table batch.
- `Batch::external(Arc<dyn ExternalPayload>, BatchMetadata)` creates an
  explicitly registered external-provider batch.
- `Batch::kind`, `num_rows`, `metadata`, `table_payload`, and
  `external_payload` expose immutable observations.
- `Batch::with_metadata` returns a new envelope.

`BatchMetadata::new(source, sequence, attributes)` validates the source and stores
JSON-compatible attributes. Its sequence is descriptive batch metadata.
`MicroBatchRunner` checkpoint ordering uses `SourceItem.sequence`, while
`StreamingRunner` maintains its own sequence counter; both are distinct from
`BatchMetadata.sequence`.

## UDFs and external providers

Create an application-owned `UdfRegistry`, register native DataFusion scalar
UDFs, then compile against `registry.snapshot()`. Operators hold serializable
`UdfReference` values; they never own source code or import paths.

`ProviderRegistry` binds `ExternalOperatorSpec` values to trusted
`BatchOperatorFactory` and `StreamOperatorFactory` implementations through
separate `register_batch`/`resolve_batch` and `register_stream`/`resolve_stream`
paths. External provider state and payloads must satisfy Rust's `Send + Sync`
boundaries.

## Micro-batch recovery

The complete checked example is
[`micro_batch_recovery.rs`](../crates/calc-flow/examples/micro_batch_recovery.rs).
Its central lifecycle is:

```rust
let mut first = MicroBatchRunner::new(
    Arc::clone(&plan),
    Box::new(ReplaySource::new()?),
    sink_router(&delivered)?,
    Arc::clone(&store) as Arc<dyn CheckpointStore>,
    1,
)?;
first.next().await?.expect("first source item");
drop(first);

let mut recovered = MicroBatchRunner::new(
    Arc::clone(&plan),
    Box::new(ReplaySource::new()?),
    sink_router(&delivered)?,
    Arc::clone(&store) as Arc<dyn CheckpointStore>,
    1,
)?;
recovered.next().await?.expect("replayed second item");
```

A `Source` implements async `open(cursor)` and `next()`. Each `SourceItem`
contains a formed batch, next JSON cursor, and sequence. A `Sink` writes a batch
with its `RunContext`. `SinkRouter` maps graph output names to sinks.

The runner restores the committed cursor when opening a source. It executes the
plan, delivers all sinks, and only then commits state/cursor/sequence. Failures
roll back owned state and retain the current item for retry, giving at-least-once
delivery.

Run the checked examples:

```bash
cargo run -p calc-flow --example expression_pipeline
cargo run -p calc-flow --example sql_join
cargo run -p calc-flow --example micro_batch_recovery
```

## Projects and stores

`ProjectSpec` is the strict v2 data model. Use `validate_project` for a
`ValidationReport`, `compile_project` for a `BatchExecutionPlan`, and
`project_json_schema` for the generated schema.

`FileProjectStore` and `FileCheckpointStore` are async atomic local stores.
Project JSON is canonical; YAML is safe import/export only. Document-size,
JSON-depth, and path rules are enforced before persistence. Print the canonical
schema with
[`crates/calc-flow/examples/export_schema.rs`](../crates/calc-flow/examples/export_schema.rs).

## Errors and cancellation

Public operations return `calc_flow::Result<T>`. `CalcFlowError` preserves
invalid arguments/documents, compilation errors, execution/provider failures,
checkpoint errors, I/O paths, cancellation, closed stream edges, and supervised
task panics. Private runtime panic capture keeps `TaskPanicked.message` valid
UTF-8 and at most 1,024 bytes including its ellipsis; non-string panic payloads
use a fixed message.

`ExecutionOptions` carries cancellation/deadline controls. Operators receive a
`RunContext` and must check cancellation at safe work boundaries.
`RunContext::settings()` returns the immutable run settings and
`RunContext::deadline()` returns the optional absolute UTC deadline by
reference, allowing providers to observe the authoritative run context without
duplicating it.

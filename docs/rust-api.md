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
        .compile(&UdfRegistry::new().snapshot())?;
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
an immutable `UdfRegistrySnapshot`. `ExecutionPlan::execute` is asynchronous and
accepts only `Batch` values.

### Engine boundary

`ExpressionOperator` and `SqlOperator` are built-in operators. They are not
implementations of the external `Operator` trait, so do not cast them to
`dyn Operator` or call their processing seam directly. Add them through
`PipelineBuilder::add_node`, which accepts an `OperatorDefinition` produced
from any boxed built-in, custom, or external operator.

Custom operators implement `Operator`, whose `OperatorContext` carries only the
run-scoped context (`run`). External operators resolve through
`ProviderRegistry` and `ExternalOperatorFactory`, covered below.

`ExecutionPlan::datafusion_config()` returns `Option<DataFusionConfig>`:
`None` means the plan is external-only and owns no table resources, so its runs
create no DataFusion session or UDF snapshot.
`ExecutionPlan::requires_datafusion()` returns the classification alone.

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
        .compile(&UdfRegistry::new().snapshot())?;
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
`ExternalOperatorFactory` implementations. External provider state and payloads
must satisfy Rust's `Send + Sync` boundaries.

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
`ValidationReport`, `compile_project` for an `ExecutionPlan`, and
`project_json_schema` for the generated schema.

`FileProjectStore` and `FileCheckpointStore` are async atomic local stores.
Project JSON is canonical; YAML is safe import/export only. Document-size,
JSON-depth, and path rules are enforced before persistence. Print the canonical
schema with
[`crates/calc-flow/examples/export_schema.rs`](../crates/calc-flow/examples/export_schema.rs).

## Errors and cancellation

Public operations return `calc_flow::Result<T>`. `CalcFlowError` preserves
invalid arguments/documents, compilation errors, execution/provider failures,
checkpoint errors, I/O paths, and cancellation.

`ExecutionOptions` carries cancellation/deadline controls. Operators receive a
`RunContext` and must check cancellation at safe work boundaries.

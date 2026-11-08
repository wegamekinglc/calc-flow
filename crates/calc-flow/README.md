# Calc Flow for Rust

`calc-flow` executes immutable Arrow batches and stateful streaming graphs with
Apache DataFusion. It provides graph compilation, SQL and expression operators,
async runners, checkpoints, and recovery. Native applications need no Python
interpreter.

## Install

The first crates.io release is being prepared. Once published, use Rust 1.94.0
or newer and these dependencies (the current shared release version is
`2026.11.2`):

```toml
[dependencies]
calc-flow = "=2026.11.2"
datafusion = { version = "=55.0.0", default-features = false }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Use DataFusion's Arrow re-exports for compatible table types. All table execution
uses DataFusion 55 / Arrow 59. The optional `benchmark-*` features expose
diagnostic internals; ordinary applications need no crate features. Transport
implementations live in the separate `calc-flow-connectors` workspace crate.

## A first calculation

```rust
use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, ExecutionOptions, ExpressionOperator, PipelineBuilder,
    UdfRegistry,
};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plan = PipelineBuilder::new("totals")?
        .add_node("calculate", Box::new(ExpressionOperator::new(
            "calculate", "total = a + b", Vec::new(), None, Vec::new(),
        )?))?
        .compile_batch(&UdfRegistry::new().snapshot())?;
    let input = RecordBatch::try_from_iter(vec![
        ("a", Arc::new(Int64Array::from(vec![1, 3])) as _),
        ("b", Arc::new(Int64Array::from(vec![2, 4])) as _),
    ])?;
    let result = plan.execute(
        BTreeMap::from([("input".into(), Batch::table(vec![input], BatchMetadata::default())?)]),
        ExecutionOptions::default(),
    ).await?;
    let output = result.outputs["output"].table_payload()?;
    let totals = output.batches()[0].column_by_name("total").unwrap()
        .as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(totals.values(), &[3, 7]);
    Ok(())
}
```

Fallible core operations return `calc_flow::Result<T>`. Plans consume immutable
`Batch` values and retain owned state within their runtime boundaries.

See the [Rust reference](https://github.com/wegamekinglc/calc-flow/blob/main/docs/rust-api.md)
and [native examples](https://github.com/wegamekinglc/calc-flow/tree/main/crates/calc-flow/examples)
for SQL, stream lifecycle, trusted UDF registration, and checkpoint contracts.
The repository's
[Rust release guide](https://github.com/wegamekinglc/calc-flow/blob/main/docs/rust-release.md)
describes verification and the registry dependency audit. Workspace
dependency patches do not propagate to consumers of a published crate.

## License

Apache-2.0; see `LICENSE` in this package.

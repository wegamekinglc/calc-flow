//! Python 07: selected Arrow columns times immutable Rust matrix weights.
#[path = "support/matrix.rs"]
mod matrix;
mod support;
use calc_flow::{
    Batch, ExecutionOptions, ProviderRegistry, StaticArrayValues, UdfRegistry, compile_project,
};
use std::collections::BTreeMap;
use support::{ExampleResult, floats, table};

#[tokio::main]
async fn main() -> ExampleResult {
    let providers = ProviderRegistry::default();
    matrix::register(&providers)?;
    let plan = compile_project(
        &matrix::project(false, false)?,
        &providers,
        &UdfRegistry::new().snapshot(),
    )?;
    let source = table(vec![
        ("quantity", floats(&[3.0, 1.0, 4.0])),
        ("unit_price", floats(&[10.0, 12.0, 10.0])),
    ]);
    let original = source.table_payload()?.batches()[0].clone();
    let weights = Batch::static_array_float(
        "rust-array",
        "float64",
        vec![2, 2],
        None,
        vec![2.0, 0.0, 0.0, 1.0],
    )?;
    let result = plan
        .execute(
            BTreeMap::from([
                ("table".into(), source.clone()),
                ("weights".into(), weights.clone()),
            ]),
            ExecutionOptions::default(),
        )
        .await?;
    let snapshot = result.outputs["output"].static_array_snapshot().unwrap();
    assert_eq!(snapshot.shape(), [3, 2]);
    let StaticArrayValues::Float(values) = snapshot.values() else {
        unreachable!()
    };
    assert_eq!(values, &[6.0, 10.0, 2.0, 12.0, 8.0, 10.0]);
    assert_eq!(source.table_payload()?.batches()[0], original);
    let snapshot = weights.static_array_snapshot().unwrap();
    let StaticArrayValues::Float(unchanged) = snapshot.values() else {
        unreachable!()
    };
    assert_eq!(unchanged, &[2.0, 0.0, 0.0, 1.0]);
    assert!(result.datafusion_metrics.is_empty());
    println!("Rust matrix result: {values:?}");
    Ok(())
}

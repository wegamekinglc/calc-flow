//! Python 11: identical batch/stream scores with static weights copied once.
#[path = "support/matrix.rs"]
mod matrix;
mod support;
use calc_flow::{
    Batch, BatchMetadata, ExecutionOptions, ProviderRegistry, StreamRequirements, UdfRegistry,
    compile_project, compile_stream_project_graph,
};
use std::{collections::BTreeMap, sync::Arc};
use support::{
    Collected, ExampleResult, ReplaySource, completed, f64_values, floats, runner, table,
};

#[tokio::main]
async fn main() -> ExampleResult {
    let providers = ProviderRegistry::default();
    let udfs = UdfRegistry::new().snapshot();
    let stream_project = matrix::project(true, true)?;
    let missing = compile_stream_project_graph(
        &stream_project,
        &providers,
        &udfs,
        &StreamRequirements::default(),
    );
    assert!(missing.is_err(), "provider must be registered explicitly");
    println!("expected provider gate: {}", missing.err().unwrap());
    matrix::register(&providers)?;
    let prices = table(vec![
        ("return", floats(&[0.02, -0.01, 0.04])),
        ("volatility", floats(&[0.10, 0.20, 0.15])),
    ]);
    let weights = Batch::static_array_float(
        "rust-array",
        "float64",
        vec![2, 1],
        None,
        vec![100.0, -10.0],
    )?;
    let batch = compile_project(&matrix::project(true, false)?, &providers, &udfs)?
        .execute(
            BTreeMap::from([
                ("table".into(), prices.clone()),
                ("weights".into(), weights.clone()),
            ]),
            ExecutionOptions::default(),
        )
        .await?;
    let scores = f64_values(&[batch.outputs["output"].clone()], "risk_score");
    let plan = compile_stream_project_graph(
        &stream_project,
        &providers,
        &udfs,
        &StreamRequirements::default(),
    )?;
    let record = &prices.table_payload()?.batches()[0];
    let parts = [record.slice(0, 2), record.slice(2, 1)]
        .into_iter()
        .map(|b| Batch::table(vec![b], BatchMetadata::default()))
        .collect::<calc_flow::Result<_>>()?;
    let source = ReplaySource::batches(parts)?;
    let output: Collected = Arc::default();
    let root = tempfile::tempdir()?;
    let job = runner(
        plan,
        BTreeMap::from([("table".into(), source.binding())]),
        &BTreeMap::from([("output".into(), Arc::clone(&output))]),
        root.path(),
    )?
    .with_static_inputs(BTreeMap::from([("weights".into(), weights)]))?
    .start()
    .await?;
    completed(&job).await?;
    let batches = output.lock().await;
    assert_eq!(f64_values(&batches, "risk_score"), scores);
    support::check_floats(&scores, &[Some(1.0), Some(-3.0), Some(2.5)]);
    let placements: Vec<_> = batches
        .iter()
        .map(|b| {
            b.metadata().attributes()["copy_bytes"]["weights"]
                .as_u64()
                .unwrap()
        })
        .collect();
    assert_eq!(placements, [16, 0]);
    println!("batch/stream risk scores: {scores:?}\nstatic-weight placement bytes: {placements:?}");
    Ok(())
}

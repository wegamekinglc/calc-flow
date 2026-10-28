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
    let udfs = UdfRegistry::new().snapshot();
    let providers = registered_provider(&udfs)?;
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
    let scores = batch_scores(&providers, &udfs, &prices, &weights).await?;
    let (streamed, placements) = stream_scores(&providers, &udfs, &prices, weights).await?;
    assert_eq!(streamed, scores);
    support::check_floats(&scores, &[Some(1.0), Some(-3.0), Some(2.5)]);
    assert_eq!(placements, [16, 0]);
    println!("batch/stream risk scores: {scores:?}\nstatic-weight placement bytes: {placements:?}");
    Ok(())
}

fn registered_provider(
    udfs: &calc_flow::UdfRegistrySnapshot,
) -> calc_flow::Result<ProviderRegistry> {
    let providers = ProviderRegistry::default();
    let stream_project = matrix::project(true, true)?;
    let missing = compile_stream_project_graph(
        &stream_project,
        &providers,
        udfs,
        &StreamRequirements::default(),
    );
    assert!(missing.is_err(), "provider must be registered explicitly");
    println!("expected provider gate: {}", missing.err().unwrap());
    matrix::register(&providers)?;
    Ok(providers)
}

async fn batch_scores(
    providers: &ProviderRegistry,
    udfs: &calc_flow::UdfRegistrySnapshot,
    prices: &Batch,
    weights: &Batch,
) -> calc_flow::Result<Vec<Option<f64>>> {
    let batch = compile_project(&matrix::project(true, false)?, providers, udfs)?
        .execute(
            BTreeMap::from([
                ("table".into(), prices.clone()),
                ("weights".into(), weights.clone()),
            ]),
            ExecutionOptions::default(),
        )
        .await?;
    Ok(f64_values(&[batch.outputs["output"].clone()], "risk_score"))
}

fn segmented_source(prices: &Batch) -> calc_flow::Result<ReplaySource> {
    let record = &prices.table_payload()?.batches()[0];
    let parts = [record.slice(0, 2), record.slice(2, 1)]
        .into_iter()
        .map(|b| Batch::table(vec![b], BatchMetadata::default()))
        .collect::<calc_flow::Result<_>>()?;
    ReplaySource::batches(parts)
}

fn stream_runner(
    root: &std::path::Path,
    providers: &ProviderRegistry,
    udfs: &calc_flow::UdfRegistrySnapshot,
    prices: &Batch,
    weights: Batch,
    output: &Collected,
) -> calc_flow::Result<calc_flow::StreamingRunner> {
    let plan = compile_stream_project_graph(
        &matrix::project(true, true)?,
        providers,
        udfs,
        &StreamRequirements::default(),
    )?;
    let source = segmented_source(prices)?;
    runner(
        plan,
        BTreeMap::from([("table".into(), source.binding())]),
        &BTreeMap::from([("output".into(), Arc::clone(output))]),
        root,
    )?
    .with_static_inputs(BTreeMap::from([("weights".into(), weights)]))
}

async fn stream_scores(
    providers: &ProviderRegistry,
    udfs: &calc_flow::UdfRegistrySnapshot,
    prices: &Batch,
    weights: Batch,
) -> ExampleResult<(Vec<Option<f64>>, Vec<u64>)> {
    let output: Collected = Arc::default();
    let root = tempfile::tempdir()?;
    let job = stream_runner(root.path(), providers, udfs, prices, weights, &output)?
        .start()
        .await?;
    completed(&job).await?;
    let batches = output.lock().await;
    let placements = batches
        .iter()
        .map(|b| {
            b.metadata().attributes()["copy_bytes"]["weights"]
                .as_u64()
                .unwrap()
        })
        .collect();
    Ok((f64_values(&batches, "risk_score"), placements))
}

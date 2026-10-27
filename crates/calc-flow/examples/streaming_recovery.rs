//! Python 08: a terminal managed checkpoint does not reopen or duplicate input.
mod support;
use calc_flow::PipelineBuilder;
use std::{collections::BTreeMap, path::Path, sync::Arc};
use support::{
    Collected, ExampleResult, ReplaySource, completed, expression, i64_values, ints, runner,
    stream_plan, table,
};

async fn run(
    root: &Path,
) -> ExampleResult<(Vec<Option<i64>>, Vec<usize>, Option<calc_flow::Epoch>)> {
    let plan = stream_plan(
        PipelineBuilder::new("streaming-recovery-example")?.add_node(
            "calculate",
            expression("calculate", "result = value * 2", &[], None)?,
        )?,
    )?;
    let source = ReplaySource::batches(
        [10, 20, 30]
            .map(|value| table(vec![("value", ints(&[value]))]))
            .into(),
    )?;
    let opens = Arc::clone(&source.opened_at);
    let values: Collected = Arc::default();
    let job = runner(
        plan,
        BTreeMap::from([("input".into(), source.binding())]),
        &BTreeMap::from([("output".into(), Arc::clone(&values))]),
        root,
    )?
    .start()
    .await?;
    let outcome = completed(&job).await?;
    let actual = i64_values(&values.lock().await, "result");
    let opened = opens.lock().await.clone();
    Ok((actual, opened, outcome.completed_epoch))
}

#[tokio::main]
async fn main() -> ExampleResult {
    let directory = tempfile::tempdir()?;
    let first = run(directory.path()).await?;
    let second = run(directory.path()).await?;
    assert_eq!(first.0, [Some(20), Some(40), Some(60)]);
    assert_eq!(first.1, [0]);
    assert!(first.2.is_some());
    assert!(second.0.is_empty());
    assert!(second.1.is_empty());
    assert_eq!(second.2, first.2);
    println!("first run: {first:?}\nterminal restart: {second:?}");
    Ok(())
}

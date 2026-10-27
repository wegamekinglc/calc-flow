//! Python 10: checkpoint nested rolling state, resume, then restart at EOF.
#[path = "support/rolling.rs"]
mod rolling_stages;
mod support;
use calc_flow::{JobState, PipelineBuilder};
use std::{collections::BTreeMap, sync::Arc, time::Duration};
use support::{
    Collected, ExampleResult, ReplaySource, check_floats, completed, f64_values, floats, runner,
    schema, stream_plan, strings, table, times, uints,
};

fn batches() -> Vec<calc_flow::Batch> {
    [100.0, 110.0, 105.0, 115.0, 100.0, 105.0]
        .iter()
        .zip(1..=6)
        .map(|(price, i)| {
            table(vec![
                ("ts", times(&[i * 1_000_000])),
                ("symbol", strings(&["AAA"])),
                ("seq", uints(&[u64::try_from(i).unwrap()])),
                ("price", floats(&[*price])),
            ])
        })
        .collect()
}
fn graph() -> calc_flow::Result<PipelineBuilder> {
    rolling_stages::delta_mean(
        "symbolic-streaming-recovery",
        schema(&batches()[0]),
        3,
        true,
    )
}

#[tokio::main]
async fn main() -> ExampleResult {
    let root = tempfile::tempdir()?;
    let output: Collected = Arc::default();
    let outputs = BTreeMap::from([("output".into(), Arc::clone(&output))]);
    let source = ReplaySource::batches(batches())?.pause_at(3);
    let pause = Arc::clone(&source.paused);
    let opens = Arc::clone(&source.opened_at);
    let first = runner(
        stream_plan(graph()?)?,
        BTreeMap::from([("input".into(), source.binding())]),
        &outputs,
        root.path(),
    )?
    .start()
    .await?;
    let checkpoint = tokio::time::timeout(Duration::from_secs(30), async {
        pause.notified().await;
        first.trigger_checkpoint().await
    })
    .await;
    let cancelled = first.cancel().await;
    assert_eq!(cancelled.state, JobState::Cancelled);
    assert_eq!(checkpoint??.as_u64(), 1);
    assert_eq!(*opens.lock().await, [0]);

    let resumed = ReplaySource::batches(batches())?;
    let reopened = Arc::clone(&resumed.opened_at);
    let second = runner(
        stream_plan(graph()?)?,
        BTreeMap::from([("input".into(), resumed.binding())]),
        &outputs,
        root.path(),
    )?
    .start()
    .await?;
    let terminal = completed(&second).await?;
    assert_eq!(*reopened.lock().await, [3]);
    let actual = f64_values(&output.lock().await, "average_gain");
    check_floats(
        &actual,
        &[
            None,
            Some(10.0),
            Some(5.0),
            Some(20.0 / 3.0),
            Some(10.0 / 3.0),
            Some(5.0),
        ],
    );
    let ended = ReplaySource::batches(batches())?;
    let terminal_opens = Arc::clone(&ended.opened_at);
    let third = runner(
        stream_plan(graph()?)?,
        BTreeMap::from([("input".into(), ended.binding())]),
        &outputs,
        root.path(),
    )?
    .start()
    .await?;
    assert_eq!(
        completed(&third).await?.completed_epoch,
        terminal.completed_epoch
    );
    assert!(terminal_opens.lock().await.is_empty());
    assert_eq!(f64_values(&output.lock().await, "average_gain"), actual);
    println!("recovered rolling values: {actual:?}");
    Ok(())
}

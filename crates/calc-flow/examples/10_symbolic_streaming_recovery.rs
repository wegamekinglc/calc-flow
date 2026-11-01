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
    let plan = stream_plan(graph()?)?;
    println!("{}: {:?}", plan.name(), plan.topological_order());
    let root = tempfile::tempdir()?;
    let output: Collected = Arc::default();
    let first_open = checkpoint_prefix(root.path(), &output).await?;
    let (terminal, resumed_open) = resume(root.path(), &output).await?;
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
    terminal_restart(root.path(), &output, terminal.completed_epoch).await?;
    println!(
        "recovered rolling values: {}",
        support::format_values(&actual)
    );
    println!(
        "source cursors: {:?} terminal epoch: {}",
        [first_open, resumed_open].concat(),
        terminal
            .completed_epoch
            .expect("terminal checkpoint")
            .as_u64()
    );
    Ok(())
}

fn recovery_runner(
    root: &std::path::Path,
    source: ReplaySource,
    output: &Collected,
) -> calc_flow::Result<calc_flow::StreamingRunner> {
    runner(
        stream_plan(graph()?)?,
        BTreeMap::from([("input".into(), source.binding())]),
        &BTreeMap::from([("output".into(), Arc::clone(output))]),
        root,
    )
}

async fn checkpoint_prefix(
    root: &std::path::Path,
    output: &Collected,
) -> ExampleResult<Vec<usize>> {
    let source = ReplaySource::batches(batches())?.pause_at(3);
    let pause = Arc::clone(&source.paused);
    let opens = Arc::clone(&source.opened_at);
    let first = recovery_runner(root, source, output)?.start().await?;
    let checkpoint = tokio::time::timeout(Duration::from_secs(30), async {
        pause.notified().await;
        first.trigger_checkpoint().await
    })
    .await;
    let cancelled = first.cancel().await;
    assert_eq!(cancelled.state, JobState::Cancelled);
    assert_eq!(checkpoint??.as_u64(), 1);
    let opened_at = opens.lock().await.clone();
    assert_eq!(opened_at, [0]);
    Ok(opened_at)
}

async fn resume(
    root: &std::path::Path,
    output: &Collected,
) -> ExampleResult<(calc_flow::JobOutcome, Vec<usize>)> {
    let source = ReplaySource::batches(batches())?;
    let reopened = Arc::clone(&source.opened_at);
    let second = recovery_runner(root, source, output)?.start().await?;
    let terminal = completed(&second).await?;
    let opened_at = reopened.lock().await.clone();
    assert_eq!(opened_at, [3]);
    Ok((terminal, opened_at))
}

async fn terminal_restart(
    root: &std::path::Path,
    output: &Collected,
    epoch: Option<calc_flow::Epoch>,
) -> ExampleResult {
    let before = f64_values(&output.lock().await, "average_gain");
    let source = ReplaySource::batches(batches())?;
    let opens = Arc::clone(&source.opened_at);
    let third = recovery_runner(root, source, output)?.start().await?;
    assert_eq!(completed(&third).await?.completed_epoch, epoch);
    assert!(opens.lock().await.is_empty());
    assert_eq!(f64_values(&output.lock().await, "average_gain"), before);
    Ok(())
}

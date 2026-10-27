//! Python 20: nested rolling stages emit before EOF and are explicitly cancelled.
#[path = "support/rolling.rs"]
mod rolling_stages;
mod support;
use std::{collections::BTreeMap, sync::Arc, time::Duration};
use support::{
    Collected, ExampleResult, ReplaySource, data, f64_values, floats, runner, schema, stream_plan,
    strings, table, times, uints, watermark,
};

#[tokio::main]
async fn main() -> ExampleResult {
    let batch = |ts: &[i64], prices: &[f64], seq: &[u64]| {
        table(vec![
            ("ts", times(ts)),
            ("symbol", strings(&["a", "a"])),
            ("seq", uints(seq)),
            ("price", floats(prices)),
        ])
    };
    let first = batch(&[1, 2], &[10.0, 12.0], &[1, 2]);
    let plan = stream_plan(rolling_stages::delta_mean(
        "streaming-pipeline",
        schema(&first),
        2,
        false,
    )?)?;
    let source = ReplaySource::new(vec![
        data(first, 1)?,
        watermark(1),
        data(batch(&[3, 4], &[15.0, 14.0], &[3, 4]), 3)?,
        watermark(3),
    ])
    .pause_at(4);
    let output: Collected = Arc::default();
    let root = tempfile::tempdir()?;
    let job = runner(
        plan,
        BTreeMap::from([("input".into(), source.binding())]),
        &BTreeMap::from([("output".into(), Arc::clone(&output))]),
        root.path(),
    )?
    .start()
    .await?;
    let ready = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if f64_values(&output.lock().await, "delta").len() >= 3 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await;
    job.cancel().await;
    ready?;
    let batches = output.lock().await;
    assert_eq!(f64_values(&batches, "delta"), [None, Some(2.0), Some(3.0)]);
    assert_eq!(
        f64_values(&batches, "mean_delta"),
        [None, Some(2.0), Some(2.5)]
    );
    support::print(&batches)
}

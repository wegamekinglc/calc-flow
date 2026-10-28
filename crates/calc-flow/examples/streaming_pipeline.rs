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
    let (input_schema, source) = source()?;
    let plan = stream_plan(rolling_stages::delta_mean(
        "streaming-pipeline",
        input_schema,
        2,
        false,
    )?)?;
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
    receive_before_eof(&job, &output).await?;
    let batches = output.lock().await;
    assert_eq!(f64_values(&batches, "delta"), [None, Some(2.0), Some(3.0)]);
    assert_eq!(
        f64_values(&batches, "mean_delta"),
        [None, Some(2.0), Some(2.5)]
    );
    support::print(&batches)
}

async fn receive_before_eof(job: &calc_flow::StreamingJob, output: &Collected) -> ExampleResult {
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
    Ok(())
}

fn source() -> calc_flow::Result<(datafusion::arrow::datatypes::SchemaRef, ReplaySource)> {
    let batch = |ts: &[i64], prices: &[f64], seq: &[u64]| {
        table(vec![
            ("ts", times(ts)),
            ("symbol", strings(&["a", "a"])),
            ("seq", uints(seq)),
            ("price", floats(prices)),
        ])
    };
    let first = batch(&[1, 2], &[10.0, 12.0], &[1, 2]);
    let input_schema = schema(&first);
    let source = ReplaySource::new(vec![
        data(first, 1)?,
        watermark(1),
        data(batch(&[3, 4], &[15.0, 14.0], &[3, 4]), 3)?,
        watermark(3),
    ])
    .pause_at(4);
    Ok((input_schema, source))
}

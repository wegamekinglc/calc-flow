//! Measure one ready input through its sink on a warmed, live grouped SUM graph.

use super::{
    Batch, ExampleResult, InputBatches, Instant, SinkBinding, configured_runner, load_input,
    validate_snapshots,
};
use async_trait::async_trait;
use calc_flow::StreamSink;
use serde_json::{Value, json};
use tokio::sync::mpsc::{self, Receiver, Sender};

struct OutputSender(Sender<Batch>);

#[async_trait]
impl StreamSink for OutputSender {
    async fn open(&mut self) -> calc_flow::Result<()> {
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> calc_flow::Result<()> {
        self.0
            .send(batch.clone())
            .await
            .map_err(|_| super::support::invalid("sparse_update", "output receiver closed"))
    }

    async fn close(&mut self) -> calc_flow::Result<()> {
        Ok(())
    }
}

async fn round_trips(
    batches: &[Batch],
    sender: Sender<Batch>,
    mut receiver: Receiver<Batch>,
) -> ExampleResult<(Vec<Batch>, Vec<u64>)> {
    let mut output = Vec::with_capacity(batches.len());
    let mut update_ns = Vec::with_capacity(batches.len().saturating_sub(1));
    for (index, batch) in batches.iter().cloned().enumerate() {
        let start = Instant::now();
        sender.send(batch).await?;
        let received = receiver.recv().await.ok_or("sink closed before output")?;
        let elapsed = start.elapsed();
        if index != 0 {
            update_ns.push(u64::try_from(elapsed.as_nanos())?);
        }
        output.push(received);
    }
    drop(sender);
    receiver.close();
    if receiver.recv().await.is_some() {
        return Err("unexpected output after final update".into());
    }
    Ok((output, update_ns))
}

async fn measure(batches: &[Batch], changelog: bool) -> ExampleResult<Value> {
    let (sender, input) = mpsc::channel(1);
    let (output, receiver) = mpsc::channel(1);
    let job = configured_runner(
        InputBatches::Live(input),
        SinkBinding::ordinary("snapshots", OutputSender(output))?,
        1,
        changelog,
    )?
    .start()
    .await?;
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        round_trips(batches, sender, receiver),
    )
    .await;
    if !matches!(result, Ok(Ok(_))) {
        job.cancel().await;
    }
    let (output, update_ns) = result??;
    super::support::completed(&job).await?;
    validated_observation(batches, &output, &update_ns, changelog, &job.status())
}

fn validated_observation(
    batches: &[Batch],
    output: &[Batch],
    update_ns: &[u64],
    changelog: bool,
    status: &calc_flow::JobStatus,
) -> ExampleResult<Value> {
    let validated = validate_snapshots(batches, output, "stream", changelog)?;
    Ok(json!({
        "update_ns": update_ns,
        "status": status,
        "output_rows": output.iter().map(Batch::num_rows).collect::<Vec<_>>(),
        "snapshot_sha256": validated.sha256,
        "schema_sha256": validated.schema_sha256,
        "process_observations": calc_flow::benchmark_internals::process_observations()?,
    }))
}

pub(super) async fn command(args: &[String]) -> ExampleResult {
    let [path, mode] = args else {
        return Err("--sparse-update SPARSE.arrow snapshot|changelog".into());
    };
    let changelog = match mode.as_str() {
        "snapshot" => false,
        "changelog" => true,
        _ => return Err("expected snapshot or changelog".into()),
    };
    let input = load_input(path).await?;
    let input_rows = input
        .batches
        .iter()
        .map(Batch::num_rows)
        .collect::<Vec<_>>();
    if input_rows.first() != Some(&10_000) || input_rows.get(1..) != Some(&[100; 20]) {
        return Err("expected 10,000 initial rows followed by twenty 100-row updates".into());
    }
    let result = measure(&input.batches, changelog).await?;
    println!(
        "{}",
        json!({"mode": mode, "input_sha256": input.sha256, "input_rows": input_rows, "result": result})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rejects_a_tail_output_reserved_before_receiver_closure() {
        let data = crate::support::table(vec![("x", crate::support::ints(&[0]))]);
        let (input, _source) = tokio::sync::mpsc::channel(1);
        let (sink, outputs) = tokio::sync::mpsc::channel(1);
        let permit = sink.reserve_owned().await.unwrap();
        let (result, _) = tokio::join!(
            biased;
            round_trips(&[], input, outputs),
            async move { permit.send(data) },
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("unexpected output")
        );
    }

    #[tokio::test]
    async fn warmed_updates_exclude_the_seed_and_reconstruct_each_result() {
        let batches = crate::generated(crate::options(10, 1, 100)).await.unwrap();
        let changes = measure(&batches, true).await.unwrap();
        let snapshots = measure(&batches, false).await.unwrap();
        assert_eq!(changes["update_ns"].as_array().unwrap().len(), 1);
        assert!(changes["update_ns"][0].as_u64().unwrap() > 0);
        assert_eq!(changes["snapshot_sha256"], snapshots["snapshot_sha256"]);
        assert_eq!(snapshots["output_rows"], json!([100, 100]));
        assert_eq!(changes["output_rows"][0], 100);
        assert!(changes["output_rows"][1].as_u64().unwrap() < 100);
    }
}

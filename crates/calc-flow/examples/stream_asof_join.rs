//! Python 22: equal watermarks do not finalize ASOF; tolerance limits quote age.
mod support;
use async_trait::async_trait;
use calc_flow::{
    AsofJoinSide, AsofStateLimits, Batch, Cursor, NativeWatermarkCapability, NodeOperator,
    PipelineBuilder, ReplayPositioning, Result, SourceBinding, SourceCapabilities,
    SourceDeliveryCapability, SourceEvent, SourceSchema, StreamAsofJoinOperator,
    StreamAsofJoinSpec, StreamSource, WatermarkPolicy,
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use support::{
    Collected, ExampleResult, completed, data, f64_values, floats, runner, schema, stream_plan,
    strings, table, times, uints, watermark,
};

struct QuoteSource {
    batch: Batch,
    stage: usize,
    advance: Arc<AtomicBool>,
}
#[async_trait]
impl StreamSource for QuoteSource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::Unsupported,
            delivery: SourceDeliveryCapability::Lossy,
            max_batch_rows: 3,
            max_batch_bytes: 65536,
            schema: SourceSchema::Exact(schema(&self.batch)),
            native_watermarks: NativeWatermarkCapability::EmitsNative,
        }
    }
    async fn open(&mut self, _: Option<Cursor>) -> Result<()> {
        self.stage = 0;
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let event = match self.stage {
            0 => data(self.batch.clone(), 1)?,
            1 => watermark(105),
            2 if !self.advance.load(Ordering::Acquire) => {
                tokio::time::sleep(Duration::from_millis(1)).await;
                return Ok(Some(SourceEvent::Idle));
            }
            2 => watermark(122),
            _ => return Ok(None),
        };
        self.stage += 1;
        Ok(Some(event))
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
fn rows(ts: &[i64], prices: &[f64]) -> Batch {
    table(vec![
        ("symbol", strings(&vec!["AAA"; ts.len()])),
        ("time", times(ts)),
        (
            "sequence",
            uints(&(0..u64::try_from(ts.len()).unwrap()).collect::<Vec<_>>()),
        ),
        ("price", floats(prices)),
    ])
}
fn side(prefix: &str) -> Result<AsofJoinSide> {
    AsofJoinSide::new(
        vec!["symbol".into()],
        "time".into(),
        vec!["sequence".into()],
        prefix.into(),
    )
}
#[tokio::main]
async fn main() -> ExampleResult {
    let trades = rows(&[105, 121], &[10.4, 10.6]);
    let quotes = rows(&[90, 100, 110], &[10.0, 10.2, 10.5]);
    let plan = plan(&trades, &quotes)?;
    let advance = Arc::new(AtomicBool::new(false));
    let sources = [("left", trades), ("right", quotes)]
        .into_iter()
        .map(|(id, batch)| {
            (
                id.into(),
                SourceBinding::new(QuoteSource {
                    batch,
                    stage: 0,
                    advance: Arc::clone(&advance),
                })
                .with_watermark_policy(WatermarkPolicy::SourceProvided),
            )
        })
        .collect();
    let output: Collected = Arc::default();
    let root = tempfile::tempdir()?;
    let job = runner(
        plan,
        sources,
        &BTreeMap::from([("output".into(), Arc::clone(&output))]),
        root.path(),
    )?
    .start()
    .await?;
    wait_for_equal_watermarks(&job).await?;
    assert!(
        output.lock().await.iter().all(|b| b.num_rows() == 0),
        "equal watermarks finalized too early"
    );
    println!("Both watermarks equal 105: no final output yet.");
    advance.store(true, Ordering::Release);
    completed(&job).await?;
    let batches = output.lock().await;
    assert_eq!(f64_values(&batches, "quote__price"), [Some(10.2), None]);
    support::print(&batches)
}

fn join_spec() -> Result<StreamAsofJoinSpec> {
    StreamAsofJoinSpec::new(
        side("trade")?,
        side("quote")?,
        Duration::from_micros(10),
        AsofStateLimits::new(100_000, 64 << 20)?,
    )
}

fn plan(trades: &Batch, quotes: &Batch) -> Result<calc_flow::StreamExecutionPlan> {
    let operator =
        StreamAsofJoinOperator::new("match", schema(trades), schema(quotes), join_spec()?)?;
    stream_plan(
        PipelineBuilder::new("stream-asof-join")?
            .add_node("match", NodeOperator::StreamAsofJoin(Box::new(operator)))?,
    )
}

async fn wait_for_equal_watermarks(job: &calc_flow::StreamingJob) -> ExampleResult {
    let ready = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if job.stream_asof_join_status().values().any(|s| {
                s.left.watermark_micros.map(calc_flow::EventTime::as_micros) == Some(105)
                    && s.right
                        .watermark_micros
                        .map(calc_flow::EventTime::as_micros)
                        == Some(105)
            }) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await;
    if let Err(error) = ready {
        job.cancel().await;
        return Err(error.into());
    }
    Ok(())
}

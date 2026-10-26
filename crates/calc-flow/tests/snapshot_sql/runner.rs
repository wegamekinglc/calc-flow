use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use calc_flow::{
    Batch, Cursor, EdgeBudget, JobState, JsonMap, ManagedCheckpointRuntime,
    NativeWatermarkCapability, PipelineBuilder, ReplayPositioning, Result, SinkBinding,
    SourceBinding, SourceCapabilities, SourceDeliveryCapability, SourceEvent, SourceSchema,
    StreamRequirements, StreamRuntimeConfig, StreamSink, StreamSource, StreamingRunner,
    UdfRegistrySnapshot,
};
use tokio::sync::{Notify, mpsc};

use super::support::*;

struct Source {
    events: mpsc::UnboundedReceiver<SourceEvent>,
    ended: Arc<Notify>,
    opened: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl StreamSource for Source {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 10,
            max_batch_bytes: 1 << 20,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }
    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        self.opened.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let event = self.events.recv().await;
        if event.is_none() {
            self.ended.notify_one();
        }
        Ok(event)
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct Sink(mpsc::UnboundedSender<Batch>);
#[async_trait::async_trait]
impl StreamSink for Sink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.0.send(batch.clone()).unwrap();
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

fn source() -> (mpsc::UnboundedSender<SourceEvent>, Source) {
    let (tx, events) = mpsc::unbounded_channel();
    (
        tx,
        Source {
            events,
            ended: Arc::new(Notify::new()),
            opened: Arc::new(AtomicUsize::new(0)),
        },
    )
}

fn data(values: &[i64], sequence: u64) -> SourceEvent {
    SourceEvent::Data {
        batch: batch(&[values]),
        cursor: Cursor::unbound(sequence.to_be_bytes().to_vec(), JsonMap::new()).unwrap(),
    }
}

async fn runner(
    root: &std::path::Path,
    p: Source,
    q: Source,
    sink: mpsc::UnboundedSender<Batch>,
    edge: EdgeBudget,
) -> StreamingRunner {
    let plan = PipelineBuilder::new("snapshot-recovery")
        .unwrap()
        .add_node("valuation", operator(QUERY).await)
        .unwrap()
        .compile_stream(
            &UdfRegistrySnapshot::default(),
            &StreamRequirements::default(),
        )
        .unwrap();
    StreamingRunner::new(
        plan,
        BTreeMap::from([
            (
                "p".into(),
                SourceBinding::new(p).with_watermark_policy(calc_flow::WatermarkPolicy::Disabled {
                    idle_timeout: None,
                }),
            ),
            (
                "q".into(),
                SourceBinding::new(q).with_watermark_policy(calc_flow::WatermarkPolicy::Disabled {
                    idle_timeout: None,
                }),
            ),
        ]),
        BTreeMap::from([(
            "output".into(),
            vec![SinkBinding::ordinary("sink", Sink(sink)).unwrap()],
        )]),
        ManagedCheckpointRuntime::new(root).unwrap(),
    )
    .unwrap()
    .with_runtime_config(StreamRuntimeConfig {
        edge_budget: edge,
        ..StreamRuntimeConfig::default()
    })
    .unwrap()
}

fn assert_batch(actual: &Batch, expected: &Batch) {
    use datafusion::arrow::compute::concat_batches;
    let a = actual.table_payload().unwrap();
    let e = expected.table_payload().unwrap();
    assert_eq!(
        concat_batches(a.schema(), a.batches()).unwrap(),
        concat_batches(e.schema(), e.batches()).unwrap()
    );
}

#[tokio::test]
async fn test_managed_eof_updates_repeated_checkpoints_live_and_terminal_restore() {
    tokio::time::timeout(Duration::from_secs(20), async {
        let directory = tempfile::tempdir().unwrap();
        let (p_tx, p) = source();
        let ended = p.ended.clone();
        let (q_tx, q) = source();
        let (sink, mut output) = mpsc::unbounded_channel();
        p_tx.send(data(&[1, 2], 1)).unwrap();
        drop(p_tx);
        q_tx.send(data(&[1], 1)).unwrap();
        let first = runner(directory.path(), p, q, sink, EdgeBudget::default())
            .await
            .start()
            .await
            .unwrap();
        assert_batch(
            &output.recv().await.unwrap(),
            &oracle(QUERY, batch(&[&[1, 2]]), batch(&[&[1]])).await,
        );
        ended.notified().await;
        q_tx.send(SourceEvent::Idle).unwrap();
        q_tx.send(data(&[2], 2)).unwrap();
        assert_batch(
            &output.recv().await.unwrap(),
            &oracle(QUERY, batch(&[&[1, 2]]), batch(&[&[2]])).await,
        );
        let epoch = first.trigger_checkpoint().await.unwrap();
        assert_eq!(
            first.trigger_checkpoint().await.unwrap(),
            epoch.next().unwrap()
        );
        first.cancel().await;
        drop(q_tx);

        let (p_tx, p) = source();
        drop(p_tx);
        let p_opened = p.opened.clone();
        let (q_tx, q) = source();
        q_tx.send(data(&[1, 1], 3)).unwrap();
        drop(q_tx);
        let (sink, mut output) = mpsc::unbounded_channel();
        let second = runner(directory.path(), p, q, sink, EdgeBudget::default())
            .await
            .start()
            .await
            .unwrap();
        assert_batch(
            &output.recv().await.unwrap(),
            &oracle(QUERY, batch(&[&[1, 2]]), batch(&[&[1, 1]])).await,
        );
        assert_eq!(second.wait().await.state, JobState::Completed);
        assert_eq!(
            p_opened.load(Ordering::SeqCst),
            0,
            "ended input must not reopen"
        );

        let (p_tx, p) = source();
        drop(p_tx);
        let (q_tx, q) = source();
        drop(q_tx);
        let p_opened = p.opened.clone();
        let q_opened = q.opened.clone();
        let (sink, mut output) = mpsc::unbounded_channel();
        let terminal = runner(directory.path(), p, q, sink, EdgeBudget::default())
            .await
            .start()
            .await
            .unwrap();
        assert_eq!(terminal.wait().await.state, JobState::Completed);
        assert_eq!(
            p_opened.load(Ordering::SeqCst) + q_opened.load(Ordering::SeqCst),
            0
        );
        assert!(
            output.try_recv().is_err(),
            "terminal restore does not replay output"
        );
    })
    .await
    .expect("snapshot managed recovery timed out");
}

#[tokio::test]
async fn test_native_oversized_input_fails_without_partial_snapshot_output() {
    let directory = tempfile::tempdir().unwrap();
    let (p_tx, p) = source();
    let (q_tx, q) = source();
    p_tx.send(data(&[1, 2], 1)).unwrap();
    drop(p_tx);
    q_tx.send(data(&[1], 1)).unwrap();
    drop(q_tx);
    let (sink, mut output) = mpsc::unbounded_channel();
    let result = runner(
        directory.path(),
        p,
        q,
        sink,
        EdgeBudget::new(1, 1 << 20).unwrap(),
    )
    .await
    .start()
    .await;
    assert!(
        result.is_err(),
        "source capability exceeds the entire-message edge budget"
    );
    assert!(output.try_recv().is_err());
}

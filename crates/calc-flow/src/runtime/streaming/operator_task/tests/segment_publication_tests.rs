use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::{
    Epoch, LocalStateBackend, StateBackend, StateHandle, StateLineageBackend, StateLineageKey,
    state::ManifestTransaction,
};

struct GatedPublication {
    inner: Box<dyn StateLineageBackend>,
    entered: Mutex<Option<tokio::sync::oneshot::Sender<usize>>>,
    release: Arc<tokio::sync::Notify>,
    fail: bool,
    gate_staging: bool,
    staged: Arc<AtomicUsize>,
}

impl GatedPublication {
    async fn wait_for_release(&self, segment_count: usize) -> Result<()> {
        self.entered
            .lock()
            .take()
            .unwrap()
            .send(segment_count)
            .unwrap();
        self.release.notified().await;
        if self.fail {
            return Err(CalcFlowError::Internal {
                message: "injected batch segment failure".into(),
            });
        }
        Ok(())
    }
}

#[async_trait]
impl StateLineageBackend for GatedPublication {
    fn identity_hash(&self) -> &str {
        self.inner.identity_hash()
    }
    async fn stage_segment(&self, handle: &StateHandle, bytes: &[u8]) -> Result<()> {
        self.inner.stage_segment(handle, bytes).await
    }
    async fn stage_segments(&self, segments: &[(&StateHandle, &[u8])]) -> Result<()> {
        if self.gate_staging {
            self.wait_for_release(segments.len()).await?;
        }
        self.inner.stage_segments(segments).await?;
        self.staged.fetch_add(segments.len(), Ordering::SeqCst);
        Ok(())
    }
    async fn validate_segment(&self, handle: &StateHandle) -> Result<()> {
        self.inner.validate_segment(handle).await
    }
    async fn publish_segment(&self, handle: &StateHandle) -> Result<()> {
        self.inner.publish_segment(handle).await
    }
    async fn publish_segments(&self, handles: &[StateHandle]) -> Result<()> {
        if !self.gate_staging {
            self.wait_for_release(handles.len()).await?;
        }
        self.inner.publish_segments(handles).await
    }
    async fn load_segment(&self, handle: &StateHandle) -> Result<Vec<u8>> {
        self.inner.load_segment(handle).await
    }
    async fn collect_orphans(&self, retained: &[StateHandle]) -> Result<usize> {
        self.inner.collect_orphans(retained).await
    }
}

#[tokio::test]
async fn opaque_multi_segment_capture_forwards_without_legacy_staging() {
    let directory = tempfile::tempdir().unwrap();
    let backend = LocalStateBackend::new(directory.path().join("state"))
        .await
        .unwrap();
    let key = StateLineageKey::new("orders", &"a".repeat(64)).unwrap();
    let (entered, mut admitted) = tokio::sync::oneshot::channel();
    let release = Arc::new(tokio::sync::Notify::new());
    let staged = Arc::new(AtomicUsize::new(0));
    let lineage: Arc<dyn StateLineageBackend> = Arc::new(GatedPublication {
        inner: backend.open_lineage(&key).await.unwrap(),
        entered: Mutex::new(Some(entered)),
        release,
        fail: false,
        gate_staging: true,
        staged: staged.clone(),
    });
    let transaction = Arc::new(
        ManifestTransaction::open(lineage, &key, directory.path().join("manifests"), 2)
            .await
            .unwrap(),
    );
    let aligned = Arc::new(tokio::sync::Notify::new());
    let (mut harness, mut checkpoint_rx) = checkpoint_harness(
        transaction,
        aligned.clone(),
        Behavior::StatefulMany,
        Arc::default(),
    );
    start(&mut harness).await;
    align_checkpoint(&mut harness, &aligned, false).await;

    let ack = tokio::time::timeout(Duration::from_secs(5), checkpoint_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(ack.snapshot.segments.len(), 2);
    assert!(ack.state.segments.is_empty());
    assert_eq!(
        harness.outputs[0]
            .recv()
            .await
            .unwrap()
            .unwrap()
            .as_barrier(),
        Some(Epoch::INITIAL)
    );
    assert!(matches!(
        admitted.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Empty)
    ));
    assert_eq!(staged.load(Ordering::SeqCst), 0);

    harness.cancellation.cancel();
    assert!(harness.supervisor.join_all().await.errors.is_empty());
}

fn checkpoint_harness(
    transaction: Arc<ManifestTransaction>,
    aligned: Arc<tokio::sync::Notify>,
    behavior: Behavior,
    observed: Arc<Mutex<Vec<(String, String, u64)>>>,
) -> (Harness, mpsc::Receiver<OperatorCheckpointAck>) {
    let output_port = Port::new("output", BatchKind::Table, false, None).unwrap();
    let operator = ProbeOperator {
        input_ports: ["input", "other"]
            .map(|name| Port::new(name, BatchKind::Table, true, None).unwrap())
            .into(),
        output_ports: vec![output_port.clone()],
        behavior,
        watermarks: Arc::default(),
        ends: Arc::default(),
        observed,
    };
    let (checkpoint_tx, checkpoint_rx) = mpsc::channel(1);
    let harness = harness_with_operator_capability(
        &["input", "other"],
        1,
        CompiledStreamOperator::External(Box::new(operator)),
        OperatorCheckpointCapability::CheckpointedStateful { state_version: 1 },
        output_port,
        Some(OperatorCheckpointPort {
            acks: checkpoint_tx,
            transaction: Some(transaction),
            terminal: None,
            alignment_fault: Some(Arc::new(move || {
                aligned.notify_one();
                Ok(())
            })),
        }),
        None,
    );
    (harness, checkpoint_rx)
}

async fn align_checkpoint(
    harness: &mut Harness,
    aligned: &tokio::sync::Notify,
    complete_with_eof: bool,
) {
    harness
        .inputs
        .get_mut("input")
        .unwrap()
        .send(StreamMessage::barrier(Epoch::INITIAL))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), aligned.notified())
        .await
        .unwrap();
    harness
        .inputs
        .get_mut("other")
        .unwrap()
        .send(if complete_with_eof {
            StreamMessage::end_of_input()
        } else {
            StreamMessage::barrier(Epoch::INITIAL)
        })
        .await
        .unwrap();
}

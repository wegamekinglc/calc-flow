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
async fn multi_segment_publication_must_finish_before_operator_ack_or_barrier() {
    assert_pending_preparation_cancels_before_staging().await;
    for (gate_staging, fail, cancel, complete_with_eof) in [
        (true, false, true, false),
        (true, false, true, true),
        (false, false, true, false),
        (false, false, true, true),
        (true, false, false, false),
        (true, true, false, false),
        (false, false, false, false),
        (false, true, false, false),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", &"a".repeat(64)).unwrap();
        let (entered, admitted) = tokio::sync::oneshot::channel();
        let release = Arc::new(tokio::sync::Notify::new());
        let staged = Arc::new(AtomicUsize::new(0));
        let lineage: Arc<dyn StateLineageBackend> = Arc::new(GatedPublication {
            inner: backend.open_lineage(&key).await.unwrap(),
            entered: Mutex::new(Some(entered)),
            release: release.clone(),
            fail,
            gate_staging,
            staged: staged.clone(),
        });
        let transaction = Arc::new(
            ManifestTransaction::open(lineage.clone(), &key, directory.path().join("manifests"), 2)
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
        align_checkpoint(&mut harness, &aligned, complete_with_eof).await;
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), admitted)
                .await
                .unwrap()
                .unwrap(),
            2
        );
        assert!(matches!(
            checkpoint_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        assert!(
            tokio::time::timeout(Duration::from_millis(20), harness.outputs[0].recv())
                .await
                .is_err()
        );
        if cancel {
            assert_cancelled_staging_settles(&mut harness, &mut checkpoint_rx, &release).await;
            assert_eq!(staged.load(Ordering::SeqCst), 2);
            continue;
        }
        release.notify_one();
        if fail {
            let report = harness.supervisor.join_all().await;
            assert_eq!(report.errors.len(), 1);
            assert!(checkpoint_rx.recv().await.is_none());
            assert!(harness.outputs[0].recv().await.unwrap().is_none());
        } else {
            let ack = tokio::time::timeout(Duration::from_secs(5), checkpoint_rx.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(ack.state.segments.len(), 2);
            for handle in &ack.state.segments {
                assert!(!lineage.load_segment(handle).await.unwrap().is_empty());
            }
            assert_eq!(
                harness.outputs[0]
                    .recv()
                    .await
                    .unwrap()
                    .unwrap()
                    .as_barrier(),
                Some(Epoch::INITIAL)
            );
            harness.cancellation.cancel();
            assert!(harness.supervisor.join_all().await.errors.is_empty());
        }
    }
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

async fn assert_pending_preparation_cancels_before_staging() {
    let directory = tempfile::tempdir().unwrap();
    let backend = LocalStateBackend::new(directory.path().join("state"))
        .await
        .unwrap();
    let key = StateLineageKey::new("orders", &"a".repeat(64)).unwrap();
    let (entered, mut admitted) = tokio::sync::oneshot::channel();
    let lineage: Arc<dyn StateLineageBackend> = Arc::new(GatedPublication {
        inner: backend.open_lineage(&key).await.unwrap(),
        entered: Mutex::new(Some(entered)),
        release: Arc::default(),
        fail: false,
        gate_staging: true,
        staged: Arc::default(),
    });
    let transaction = Arc::new(
        ManifestTransaction::open(lineage.clone(), &key, directory.path().join("manifests"), 2)
            .await
            .unwrap(),
    );
    let aligned = Arc::new(tokio::sync::Notify::new());
    let observed = Arc::new(Mutex::new(Vec::new()));
    let (mut harness, mut checkpoint) = checkpoint_harness(
        transaction,
        aligned.clone(),
        Behavior::PreparePending,
        observed.clone(),
    );
    start(&mut harness).await;
    align_checkpoint(&mut harness, &aligned, false).await;
    tokio::time::timeout(Duration::from_secs(5), async {
        while observed.lock().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    harness.cancellation.cancel();
    let report = tokio::time::timeout(Duration::from_secs(5), harness.supervisor.join_all())
        .await
        .unwrap();
    assert!(report.errors.is_empty());
    assert!(matches!(
        admitted.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Empty)
    ));
    assert!(checkpoint.recv().await.is_none());
    assert!(harness.outputs[0].recv().await.unwrap().is_none());
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

async fn assert_cancelled_staging_settles(
    harness: &mut Harness,
    checkpoint: &mut mpsc::Receiver<OperatorCheckpointAck>,
    release: &tokio::sync::Notify,
) {
    harness.cancellation.cancel();
    {
        let completion = harness.supervisor.join_all();
        tokio::pin!(completion);
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut completion)
                .await
                .is_err()
        );
        assert!(matches!(
            checkpoint.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        release.notify_one();
        assert!(completion.await.errors.is_empty());
    }
    assert!(checkpoint.recv().await.is_none());
    assert!(harness.outputs[0].recv().await.unwrap().is_none());
}

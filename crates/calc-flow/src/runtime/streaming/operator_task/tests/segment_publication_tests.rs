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
}

#[async_trait]
impl StateLineageBackend for GatedPublication {
    fn identity_hash(&self) -> &str {
        self.inner.identity_hash()
    }
    async fn stage_segment(&self, handle: &StateHandle, bytes: &[u8]) -> Result<()> {
        self.inner.stage_segment(handle, bytes).await
    }
    async fn validate_segment(&self, handle: &StateHandle) -> Result<()> {
        self.inner.validate_segment(handle).await
    }
    async fn publish_segment(&self, handle: &StateHandle) -> Result<()> {
        self.inner.publish_segment(handle).await
    }
    async fn publish_segments(&self, handles: &[StateHandle]) -> Result<()> {
        self.entered
            .lock()
            .take()
            .unwrap()
            .send(handles.len())
            .unwrap();
        self.release.notified().await;
        if self.fail {
            return Err(CalcFlowError::Internal {
                message: "injected batch publication failure".into(),
            });
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
    for fail in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("orders", &"a".repeat(64)).unwrap();
        let (entered, admitted) = tokio::sync::oneshot::channel();
        let release = Arc::new(tokio::sync::Notify::new());
        let lineage: Arc<dyn StateLineageBackend> = Arc::new(GatedPublication {
            inner: backend.open_lineage(&key).await.unwrap(),
            entered: Mutex::new(Some(entered)),
            release: release.clone(),
            fail,
        });
        let transaction = Arc::new(
            ManifestTransaction::open(lineage.clone(), &key, directory.path().join("manifests"), 2)
                .await
                .unwrap(),
        );
        let output_port = Port::new("output", BatchKind::Table, false, None).unwrap();
        let operator = ProbeOperator {
            input_ports: vec![Port::new("input", BatchKind::Table, true, None).unwrap()],
            output_ports: vec![output_port.clone()],
            behavior: Behavior::StatefulMany,
            watermarks: Arc::default(),
            ends: Arc::default(),
            observed: Arc::default(),
        };
        let (checkpoint_tx, mut checkpoint_rx) = mpsc::channel(1);
        let mut harness = harness_with_operator_capability(
            &["input"],
            1,
            CompiledStreamOperator::External(Box::new(operator)),
            OperatorCheckpointCapability::CheckpointedStateful { state_version: 1 },
            output_port,
            Some(OperatorCheckpointPort {
                acks: checkpoint_tx,
                transaction: Some(transaction),
                terminal: None,
                alignment_fault: None,
            }),
            None,
        );
        start(&mut harness).await;
        harness
            .inputs
            .get_mut("input")
            .unwrap()
            .send(StreamMessage::barrier(Epoch::INITIAL))
            .await
            .unwrap();
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

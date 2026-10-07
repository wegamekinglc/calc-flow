use super::*;
use crate::{
    Epoch, JoinStateLimits, JoinTimeBounds, LocalStateBackend, StateHandle, StateLineageBackend,
    StateLineageKey, StateSegment, StreamJoinOperator, StreamJoinSpec,
};
use datafusion::{
    arrow::datatypes::{DataType, Field, Schema, TimeUnit},
    execution::memory_pool::MemoryPool,
};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::oneshot;

const WIRE_BYTES: usize = 4_096;

struct ReadGate {
    entered: oneshot::Sender<(usize, usize, usize)>,
    release: std::sync::mpsc::Receiver<()>,
    fail: bool,
}

struct ObservedLineage {
    inner: Arc<dyn StateLineageBackend>,
    reads: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl StateLineageBackend for ObservedLineage {
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

    async fn load_segment(&self, handle: &StateHandle) -> Result<Vec<u8>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.inner.load_segment(handle).await
    }

    async fn collect_orphans(&self, retained: &[StateHandle]) -> Result<usize> {
        self.inner.collect_orphans(retained).await
    }
}

fn join_operator() -> CompiledStreamOperator {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
    ]));
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "time",
        "time",
        JoinTimeBounds::new(std::time::Duration::ZERO, std::time::Duration::ZERO).unwrap(),
        JoinStateLimits::new(32, 1 << 20, 32).unwrap(),
    )
    .unwrap();
    CompiledStreamOperator::StreamJoin(Box::new(
        StreamJoinOperator::new("join", schema.clone(), schema, spec).unwrap(),
    ))
}

fn join_pool(operator: &mut CompiledStreamOperator) -> Arc<dyn MemoryPool> {
    let CompiledStreamOperator::StreamJoin(join) = operator else {
        unreachable!();
    };
    join.checkpoint_preload_test_pool().unwrap()
}

async fn load_fixture(
    directory: &std::path::Path,
    pool: Arc<dyn MemoryPool>,
    gate: ReadGate,
) -> (
    Arc<ManifestTransaction>,
    OperatorManifestEntry,
    Arc<ObservedLineage>,
    Arc<crate::state::LocalStateLineageBackend>,
) {
    let backend = LocalStateBackend::new(directory.join("state"))
        .await
        .unwrap();
    let key = StateLineageKey::new("join-preload", &"1".repeat(64)).unwrap();
    let local = Arc::new(backend.open_local_lineage(&key).await.unwrap());
    let reads = Arc::new(AtomicUsize::new(0));
    let read_count = reads.clone();
    let gate = Mutex::new(Some(gate));
    local.set_prepaid_read_hook(Arc::new(move |bytes, capacity, credit, peak| {
        read_count.fetch_add(1, Ordering::SeqCst);
        assert_eq!(bytes.len(), WIRE_BYTES);
        assert_eq!(
            credit.consumer().name(),
            "sql-incremental:stream-join-preload"
        );
        assert_eq!(pool.reserved(), credit.size());
        let gate = gate.lock().take().expect("one actual paid wire read");
        gate.entered
            .send((capacity, pool.reserved(), peak))
            .unwrap();
        gate.release.recv().unwrap();
        if gate.fail {
            return Err(CalcFlowError::Internal {
                message: "late Join wire load failure".into(),
            });
        }
        Ok(())
    }));
    let lineage = Arc::new(ObservedLineage {
        inner: local.clone(),
        reads,
    });
    let transaction = Arc::new(
        ManifestTransaction::open(lineage.clone(), &key, directory.join("manifests"), 2)
            .await
            .unwrap(),
    );
    let staged = transaction
        .stage_operator_state(
            "join",
            Epoch::INITIAL,
            OperatorStateSnapshot {
                inline_metadata: BTreeMap::from([(
                    "fixture".into(),
                    serde_json::json!({"wire_only": true, "labels": ["left", "right"]}),
                )]),
                segments: BTreeMap::from([(
                    "left-base".into(),
                    StateSegment::new(vec![7; WIRE_BYTES]),
                )]),
            },
        )
        .await
        .unwrap();
    for handle in &staged.segments {
        lineage.publish_segment(handle).await.unwrap();
    }
    lineage.reads.store(0, Ordering::SeqCst);
    (
        transaction,
        OperatorManifestEntry {
            progress: BTreeMap::new(),
            inline_metadata: staged.inline_metadata,
            segments: staged.segments,
        },
        lineage,
        local,
    )
}

#[tokio::test(flavor = "current_thread")]
async fn test_join_wire_read_is_prepaid_and_last_snapshot_owner_refunds_same_pool() {
    let directory = tempfile::tempdir().unwrap();
    let mut operator = join_operator();
    let pool = join_pool(&mut operator);
    assert_eq!(pool.reserved(), 0);
    let (entered, observed) = oneshot::channel();
    let (release, blocked) = std::sync::mpsc::channel();
    let (transaction, entry, _, reader) = load_fixture(
        directory.path(),
        pool.clone(),
        ReadGate {
            entered,
            release: blocked,
            fail: false,
        },
    )
    .await;
    let loads = LoadOwner::default();
    let cancellation = CancellationToken::new();
    let mut loading = Box::pin(load_snapshot_with_join(
        &transaction,
        &loads,
        &operator,
        "join",
        &entry,
        &cancellation,
        Some(&reader),
    ));
    let (capacity, paid, _) = tokio::select! {
        observed = observed => observed.unwrap(),
        result = &mut loading => panic!("wire read did not enter its gate: {result:?}"),
    };
    release.send(()).unwrap();
    let snapshot = loading.await.unwrap();
    assert_eq!(snapshot.inline_metadata, entry.inline_metadata);
    assert_eq!(snapshot.segments["left-base"].bytes(), &[7; WIRE_BYTES]);
    assert!(
        paid >= capacity,
        "actual wire allocation {capacity} exceeds paid preload {paid}"
    );
    let last = snapshot.clone();
    let weak = Arc::downgrade(&snapshot.segments["left-base"].bytes_arc());
    operator.reset().unwrap();
    assert!(Arc::ptr_eq(&pool, &join_pool(&mut operator)));
    drop(snapshot);
    assert!(weak.upgrade().is_some());
    assert!(pool.reserved() >= capacity);
    drop(last);
    assert!(weak.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
    assert!(loads.close_and_drain().await.is_none());
}

fn initialized(operator: &CompiledStreamOperator) -> bool {
    let CompiledStreamOperator::StreamJoin(join) = operator else {
        unreachable!()
    };
    join.stream_runtime_initialized()
}

#[test]
fn test_join_preload_initializes_only_selected_nonempty_runtime() {
    let mut operator = join_operator();
    assert!(!initialized(&operator));
    prepare_join_preload(&mut operator, None).unwrap();
    let mut entry = OperatorManifestEntry {
        progress: BTreeMap::new(),
        inline_metadata: BTreeMap::new(),
        segments: Vec::new(),
    };
    prepare_join_preload(&mut operator, Some(&entry)).unwrap();
    assert!(!initialized(&operator));
    entry.segments.push(
        StateHandle::new(
            "join",
            Epoch::INITIAL,
            "left-base",
            "committed/a.segment",
            1,
            &"0".repeat(64),
        )
        .unwrap(),
    );
    prepare_join_preload(&mut operator, Some(&entry)).unwrap();
    assert!(initialized(&operator));
    let pool = join_pool(&mut operator);
    assert_eq!(pool.reserved(), 0);
    operator.reset().unwrap();
    assert!(Arc::ptr_eq(&pool, &join_pool(&mut operator)));
}

async fn abandoned_join_wire(fail: bool, cancelled: bool) {
    let directory = tempfile::tempdir().unwrap();
    let mut operator = join_operator();
    let pool = join_pool(&mut operator);
    let (entered, observed) = oneshot::channel();
    let (release, blocked) = std::sync::mpsc::channel();
    let (transaction, entry, lineage, reader) = load_fixture(
        directory.path(),
        pool.clone(),
        ReadGate {
            entered,
            release: blocked,
            fail,
        },
    )
    .await;
    let loads = LoadOwner::default();
    let cancellation = CancellationToken::new();
    let mut loading = Box::pin(load_snapshot_with_join(
        &transaction,
        &loads,
        &operator,
        "join",
        &entry,
        &cancellation,
        Some(&reader),
    ));
    let (capacity, paid, _) = tokio::select! {
        observation = observed => observation.unwrap(),
        result = &mut loading => panic!("load completed before real wire gate: {result:?}"),
    };
    assert!(paid >= capacity);
    drop(loading);
    if cancelled {
        cancellation.cancel();
    }
    drop(operator);
    assert_eq!(lineage.reads.load(Ordering::SeqCst), 1);
    assert_eq!(pool.reserved(), paid);
    let mut drain = Box::pin(loads.close_and_drain());
    assert!(futures::poll!(drain.as_mut()).is_pending());
    assert_eq!(pool.reserved(), paid);
    release.send(()).unwrap();
    let result = drain.await;
    if fail {
        assert!(
            matches!(result, Some(CalcFlowError::Internal { message }) if message == "late Join wire load failure")
        );
    } else {
        assert!(result.is_none());
    }
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn test_join_dropped_load_observer_keeps_real_wire_paid_until_drain() {
    abandoned_join_wire(false, false).await;
}

#[tokio::test(flavor = "current_thread")]
async fn test_join_late_load_error_refunds_only_after_actual_read_exit() {
    abandoned_join_wire(true, false).await;
}

#[tokio::test(flavor = "current_thread")]
async fn test_join_preload_refusal_and_identity_errors_do_not_read_or_change_checkpoint() {
    let directory = tempfile::tempdir().unwrap();
    let mut operator = join_operator();
    let pool = join_pool(&mut operator);
    let (entered, _observed) = oneshot::channel();
    let (_release, blocked) = std::sync::mpsc::channel();
    let (transaction, entry, lineage, reader) = load_fixture(
        directory.path(),
        pool.clone(),
        ReadGate {
            entered,
            release: blocked,
            fail: false,
        },
    )
    .await;
    let mut invalid = entry.clone();
    invalid.segments.push(
        StateHandle::new(
            "other",
            Epoch::INITIAL,
            "invalid",
            "committed/a.segment",
            1,
            &"0".repeat(64),
        )
        .unwrap(),
    );
    let cancel = CancellationToken::new();
    let loads = LoadOwner::default();
    assert!(matches!(
        load_snapshot_with_join(
            &transaction,
            &loads,
            &operator,
            "join",
            &invalid,
            &cancel,
            Some(&reader)
        )
        .await,
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    assert_eq!(pool.reserved(), 0);
    let mut duplicate = entry.clone();
    duplicate.segments.push(duplicate.segments[0].clone());
    assert!(matches!(
        load_snapshot_with_join(
            &transaction,
            &loads,
            &operator,
            "join",
            &duplicate,
            &cancel,
            Some(&reader)
        )
        .await,
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    let occupied =
        datafusion::execution::memory_pool::MemoryConsumer::new("preload-pressure").register(&pool);
    occupied.try_grow(1 << 30).unwrap();
    let error = load_snapshot_with_join(
        &transaction,
        &loads,
        &operator,
        "join",
        &entry,
        &cancel,
        Some(&reader),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::DataFusion { .. }));
    assert_eq!(pool.reserved(), occupied.size());
    assert_eq!(lineage.reads.load(Ordering::SeqCst), 0);
    assert_eq!(
        entry.inline_metadata["fixture"],
        serde_json::json!({"wire_only": true, "labels": ["left", "right"]})
    );
    drop(occupied);
    assert_eq!(pool.reserved(), 0);
    cancel.cancel();
    assert!(matches!(
        load_snapshot_with_join(
            &transaction,
            &loads,
            &operator,
            "join",
            &entry,
            &cancel,
            Some(&reader)
        )
        .await,
        Err(CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(lineage.reads.load(Ordering::SeqCst), 0);
    assert!(loads.close_and_drain().await.is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn test_join_cancelled_load_waits_for_real_wire_exit_before_refund() {
    abandoned_join_wire(false, true).await;
}

#[tokio::test(flavor = "current_thread")]
async fn test_join_long_local_root_wire_and_actual_backend_controls_are_prepaid() {
    let directory = tempfile::tempdir().unwrap();
    let mut root = directory.path().to_path_buf();
    for index in 0..38 {
        root.push(format!("{index:02}{}", "a".repeat(62)));
    }
    let mut operator = join_operator();
    let pool = join_pool(&mut operator);
    let (entered, observed) = oneshot::channel();
    let (release, blocked) = std::sync::mpsc::channel();
    let (transaction, entry, _, reader) = load_fixture(
        &root,
        pool.clone(),
        ReadGate {
            entered,
            release: blocked,
            fail: false,
        },
    )
    .await;
    let loads = LoadOwner::default();
    let cancellation = CancellationToken::new();
    let mut loading = Box::pin(load_snapshot_with_join(
        &transaction,
        &loads,
        &operator,
        "join",
        &entry,
        &cancellation,
        Some(&reader),
    ));
    let (wire, paid, actual_backend_peak) = tokio::select! {
        observed = observed => observed.unwrap(),
        result = &mut loading => panic!("load did not reach actual backend allocation gate: {result:?}"),
    };
    release.send(()).unwrap();
    let snapshot = loading.await.unwrap();
    assert_eq!(snapshot.segments["left-base"].bytes(), &[7; WIRE_BYTES]);
    drop(snapshot);
    assert_eq!(pool.reserved(), 0);
    assert!(
        paid >= wire + actual_backend_peak,
        "wire {wire} + actual local backend peak {actual_backend_peak} exceeds paid inventory {paid}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_uncertified_join_loader_preserves_legacy_bytes_without_runtime_init() {
    let directory = tempfile::tempdir().unwrap();
    let operator = join_operator();
    let pool: Arc<dyn MemoryPool> = Arc::new(
        datafusion::execution::memory_pool::GreedyMemoryPool::new(1 << 20),
    );
    let (entered, _) = oneshot::channel();
    let (_, blocked) = std::sync::mpsc::channel();
    let (transaction, entry, lineage, _) = load_fixture(
        directory.path(),
        pool.clone(),
        ReadGate {
            entered,
            release: blocked,
            fail: false,
        },
    )
    .await;
    let loads = LoadOwner::default();
    let snapshot = load_snapshot_with_join(
        &transaction,
        &loads,
        &operator,
        "join",
        &entry,
        &CancellationToken::new(),
        None,
    )
    .await
    .unwrap();
    assert_eq!(lineage.reads.load(Ordering::SeqCst), 1);
    assert_eq!(snapshot.inline_metadata, entry.inline_metadata);
    assert_eq!(snapshot.segments["left-base"].bytes(), &[7; WIRE_BYTES]);
    assert!(!initialized(&operator));
    assert_eq!(pool.reserved(), 0);
    drop(snapshot);
    assert!(loads.close_and_drain().await.is_none());
}

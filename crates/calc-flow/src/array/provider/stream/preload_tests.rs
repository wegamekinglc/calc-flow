use std::{collections::BTreeSet, path::Path, sync::Mutex, time::Duration};

use super::*;
use crate::{
    BatchKind, CancellationToken, CheckpointManifest, CheckpointManifestFields, EdgeBudget, Epoch,
    ExternalOperatorSpec, JsonMap, PipelineBuilder, Port, RecoveryStatus, StateLineageKey,
    StreamOperatorContext, StreamOperatorFactory, StreamRequirements, UdfRegistry,
    array::NativeArraySequenceFactory,
    pipeline::StreamRuntimePlanParts,
    state::{LocalStateBackend, NativeIoPoint, PreparedEpochManifest, PreparedManifestIdentity},
};

struct Fixture {
    directory: tempfile::TempDir,
    plan: Arc<StreamRuntimePlanParts>,
    inventory: Arc<NativeSequenceInventory>,
    job: StreamJobContext,
    local: Arc<LocalStateLineageBackend>,
    transaction: Arc<ManifestTransaction>,
    snapshot: OperatorStateSnapshot,
    entry: OperatorManifestEntry,
}
fn plan() -> StreamRuntimePlanParts {
    let options: JsonMap = serde_json::from_value(serde_json::json!({
        "ir_version":1, "input_dtype":"float64", "kernel":{"function":"ewma", "span":3},
    }))
    .unwrap();
    let operator = NativeArraySequenceFactory::default()
        .create(
            &ExternalOperatorSpec::new("native", "array_sequence", "1", options).unwrap(),
            vec![Port::new("input", BatchKind::Array, true, None).unwrap()],
            vec![Port::new("output", BatchKind::Array, true, None).unwrap()],
        )
        .unwrap();
    PipelineBuilder::new("native-io")
        .unwrap()
        .add_node("sequence", operator)
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
        .into_runtime_parts(EdgeBudget::default())
        .unwrap()
}
async fn transaction(
    local: &Arc<LocalStateLineageBackend>,
    key: &StateLineageKey,
    root: &Path,
    inventory: &Arc<NativeSequenceInventory>,
) -> Arc<ManifestTransaction> {
    Arc::new(
        ManifestTransaction::open(local.clone(), key, root, 2)
            .await
            .unwrap()
            .with_native_sequences(Arc::clone(inventory), Arc::clone(local))
            .unwrap(),
    )
}
async fn fixture() -> Fixture {
    let directory = tempfile::tempdir().unwrap();
    let mut plan = plan();
    let job = StreamJobContext::new(
        7,
        &plan.fingerprint,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let inventory = NativeSequenceInventory::for_plan(&plan, &job).unwrap();
    let backend = LocalStateBackend::new(directory.path().join("state"))
        .await
        .unwrap();
    let key = StateLineageKey::new(&plan.name, &plan.fingerprint).unwrap();
    let local = Arc::new(backend.open_local_lineage(&key).await.unwrap());
    let root = directory.path().join("manifests");
    let previous = transaction(&local, &key, &root, &inventory).await;
    let snapshot = captured(&mut plan, &job).await;
    let staged = previous
        .stage_operator_state_cancellable(
            "sequence",
            Epoch::INITIAL,
            snapshot.clone(),
            job.cancellation(),
        )
        .await
        .unwrap();
    let entry = OperatorManifestEntry {
        progress: BTreeMap::new(),
        inline_metadata: staged.inline_metadata.clone(),
        segments: staged.segments.clone(),
    };
    drop(staged);
    drop(previous);
    let transaction = transaction(&local, &key, &root, &inventory).await;
    Fixture {
        directory,
        plan: Arc::new(plan),
        inventory,
        job,
        local,
        transaction,
        snapshot,
        entry,
    }
}
async fn captured(
    plan: &mut StreamRuntimePlanParts,
    job: &StreamJobContext,
) -> OperatorStateSnapshot {
    let node = &mut plan.nodes[0];
    node.operator
        .prepare_checkpoint_async(&StreamOperatorContext::new(job, "sequence", None))
        .await
        .unwrap();
    let snapshot = node.operator.checkpoint(Epoch::INITIAL).unwrap();
    node.checkpoint_capability
        .encode_snapshot("sequence", snapshot)
        .unwrap()
}
impl Fixture {
    fn captures(&self) -> usize {
        inventory::compiled(&self.plan.nodes[0].operator)
            .unwrap()
            .captures
            .bytes()
    }
    fn spawn_read(
        &self,
    ) -> tokio::task::JoinHandle<std::result::Result<PreloadedSnapshot, NativeSequenceFailure>>
    {
        let plan = Arc::clone(&self.plan);
        let inventory = Arc::clone(&self.inventory);
        let transaction = Arc::clone(&self.transaction);
        let entry = self.entry.clone();
        let job = self.job.clone();
        tokio::spawn(async move {
            inventory
                .load_snapshot(
                    &plan.nodes[0].operator,
                    "sequence",
                    &entry,
                    &transaction,
                    &job,
                )
                .await
        })
    }
    async fn assert_locks_available(&self) {
        let guard = tokio::time::timeout(
            Duration::from_secs(1),
            self.transaction.native_load_guard(|| Ok(())),
        )
        .await
        .unwrap()
        .unwrap();
        let publication = tokio::time::timeout(
            Duration::from_secs(1),
            self.local.native_publication_guard(|| Ok(())),
        )
        .await
        .unwrap()
        .unwrap();
        drop(publication);
        drop(guard);
    }
}
fn blocking_hook(
    local: &LocalStateLineageBackend,
    point: NativeIoPoint,
) -> (
    tokio::sync::oneshot::Receiver<()>,
    std::sync::mpsc::Sender<()>,
) {
    let (entered, receive) = tokio::sync::oneshot::channel();
    let entered = Mutex::new(Some(entered));
    let (release, released) = std::sync::mpsc::channel();
    let released = Mutex::new(released);
    local.set_native_io_hook(Arc::new(move |actual| {
        if actual == point {
            if let Some(entered) = entered.lock().unwrap().take() {
                let _ = entered.send(());
            }
            released.lock().unwrap().recv().unwrap();
        }
        Ok(())
    }));
    (receive, release)
}

#[tokio::test]
async fn test_native_preload_lost_waiter_retains_credit_and_both_io_locks_until_settled() {
    let fixture = fixture().await;
    let before = fixture.captures();
    let (entered, release) = blocking_hook(&fixture.local, NativeIoPoint::Read);
    let read = fixture.spawn_read();
    tokio::time::timeout(Duration::from_secs(1), entered)
        .await
        .unwrap()
        .unwrap();
    read.abort();
    assert!(read.await.unwrap_err().is_cancelled());
    assert!(fixture.captures() > before);
    assert!(
        !fixture
            .transaction
            .native_is_verified(&fixture.entry.segments[0])
    );
    assert!(
        tokio::time::timeout(
            Duration::from_millis(10),
            fixture.transaction.native_load_guard(|| Ok(()))
        )
        .await
        .is_err()
    );
    assert!(
        tokio::time::timeout(
            Duration::from_millis(10),
            fixture.local.native_publication_guard(|| Ok(()))
        )
        .await
        .is_err()
    );
    release.send(()).unwrap();
    let operator = inventory::compiled(&fixture.plan.nodes[0].operator).unwrap();
    let pending =
        tokio::time::timeout(Duration::from_secs(1), operator.pending.enter(&fixture.job))
            .await
            .unwrap()
            .unwrap();
    assert_eq!(fixture.captures(), before);
    assert!(
        !fixture
            .transaction
            .native_is_verified(&fixture.entry.segments[0])
    );
    fixture.assert_locks_available().await;
    drop(pending);
}

#[tokio::test]
async fn test_native_preload_success_releases_io_locks_before_segment_lifetime_and_next_checkpoint()
{
    let fixture = fixture().await;
    let before = fixture.captures();
    let snapshot = fixture.spawn_read().await.unwrap().unwrap();
    assert_eq!(
        snapshot.segments["state"].bytes(),
        fixture.snapshot.segments["state"].bytes()
    );
    assert!(
        fixture
            .transaction
            .native_is_verified(&fixture.entry.segments[0])
    );
    assert!(fixture.captures() > before);
    fixture.assert_locks_available().await;
    let staged = fixture.transaction.stage_operator_state_cancellable(
        "sequence",
        Epoch::INITIAL.next().unwrap(),
        fixture.snapshot.clone(),
        fixture.job.cancellation(),
    );
    tokio::time::timeout(Duration::from_secs(1), staged)
        .await
        .unwrap()
        .unwrap();
    drop(snapshot);
    assert_eq!(fixture.captures(), before);
}

fn manifest(fixture: &Fixture) -> (PreparedManifestIdentity, CheckpointManifest) {
    let identity = PreparedManifestIdentity {
        pipeline_name: fixture.plan.name.clone(),
        pipeline_fingerprint: fixture.plan.fingerprint.clone(),
        runtime_config_hash: "a".repeat(64),
        source_ids: BTreeSet::new(),
        operator_ids: BTreeSet::from(["sequence".into()]),
        sink_ids: BTreeSet::new(),
        static_inputs: BTreeMap::new(),
    };
    let manifest = CheckpointManifest::new(CheckpointManifestFields {
        pipeline_name: identity.pipeline_name.clone(),
        pipeline_fingerprint: identity.pipeline_fingerprint.clone(),
        runtime_config_hash: identity.runtime_config_hash.clone(),
        epoch: Epoch::INITIAL,
        created_at: chrono::Utc::now(),
        recovery_status: RecoveryStatus::Final,
        sources: BTreeMap::new(),
        operators: BTreeMap::from([("sequence".into(), fixture.entry.clone())]),
        sinks: BTreeMap::new(),
        static_inputs: BTreeMap::new(),
    })
    .unwrap();
    (identity, manifest)
}
#[tokio::test]
async fn test_native_selection_defers_opaque_load_and_verified_evidence_until_controlled_read() {
    let mut fixture = fixture().await;
    fixture.local.set_native_io_hook(Arc::new(|point| {
        assert!(
            point != NativeIoPoint::OpaqueLoad,
            "native state cannot use opaque whole-buffer loader"
        );
        Ok(())
    }));
    let (identity, manifest) = manifest(&fixture);
    fixture
        .transaction
        .publish_cancellable(
            PreparedEpochManifest {
                manifest,
                staged_segments: BTreeMap::new(),
            },
            fixture.job.cancellation(),
        )
        .await
        .unwrap();
    let key = StateLineageKey::new(&fixture.plan.name, &fixture.plan.fingerprint).unwrap();
    fixture.transaction = transaction(
        &fixture.local,
        &key,
        &fixture.directory.path().join("manifests"),
        &fixture.inventory,
    )
    .await;
    let selected = fixture
        .transaction
        .select_latest_native_cancellable(&identity, fixture.job.cancellation())
        .await
        .unwrap()
        .unwrap();
    assert!(
        !fixture
            .transaction
            .native_is_verified(&fixture.entry.segments[0])
    );
    let loaded = fixture.spawn_read().await.unwrap().unwrap();
    assert!(
        fixture
            .transaction
            .native_is_verified(&fixture.entry.segments[0])
    );
    fixture
        .transaction
        .retain_native_cancellable(
            &identity,
            Some(&selected.manifest),
            fixture.job.cancellation(),
        )
        .await
        .unwrap();
    drop(loaded);
}

#[tokio::test]
async fn test_native_stage_lost_waiter_retains_captured_owner_and_both_io_locks_until_settled() {
    let fixture = fixture().await;
    let before = fixture.captures();
    let operator = inventory::compiled(&fixture.plan.nodes[0].operator).unwrap();
    let segment = StateSegment::new(fixture.snapshot.segments["state"].bytes().to_vec());
    let segment = operator
        .captures
        .retain(
            segment,
            &operator.configured.execution,
            operator.state.state_bytes(),
        )
        .unwrap();
    let (segment, dropped) = signal_drop(segment);
    let mut snapshot = fixture.snapshot.clone();
    snapshot.segments.insert("state".into(), segment);
    let (entered, release) = blocking_hook(&fixture.local, NativeIoPoint::Stage);
    let transaction = Arc::clone(&fixture.transaction);
    let cancellation = fixture.job.cancellation().clone();
    let stage = tokio::spawn(async move {
        transaction
            .stage_operator_state_cancellable(
                "sequence",
                Epoch::INITIAL.next().unwrap(),
                snapshot,
                &cancellation,
            )
            .await
    });
    tokio::time::timeout(Duration::from_secs(1), entered)
        .await
        .unwrap()
        .unwrap();
    stage.abort();
    assert!(matches!(stage.await, Err(error) if error.is_cancelled()));
    assert!(fixture.captures() > before);
    assert!(
        tokio::time::timeout(
            Duration::from_millis(10),
            fixture.transaction.native_load_guard(|| Ok(()))
        )
        .await
        .is_err()
    );
    assert!(
        tokio::time::timeout(
            Duration::from_millis(10),
            fixture.local.native_publication_guard(|| Ok(()))
        )
        .await
        .is_err()
    );
    release.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(1), dropped)
        .await
        .unwrap()
        .unwrap();
    fixture.assert_locks_available().await;
    assert_eq!(fixture.captures(), before);
}

struct DropSignal {
    previous: Option<Arc<dyn std::any::Any + Send + Sync>>,
    done: Option<tokio::sync::oneshot::Sender<()>>,
}
impl Drop for DropSignal {
    fn drop(&mut self) {
        drop(self.previous.take());
        if let Some(done) = self.done.take() {
            let _ = done.send(());
        }
    }
}
fn signal_drop(segment: StateSegment) -> (StateSegment, tokio::sync::oneshot::Receiver<()>) {
    let (done, receiver) = tokio::sync::oneshot::channel();
    let segment = segment.map_owner(move |previous| {
        Arc::new(DropSignal {
            previous,
            done: Some(done),
        })
    });
    (segment, receiver)
}

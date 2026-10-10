use std::collections::BTreeMap;

use async_trait::async_trait;
use calc_flow::{CalcFlowError, Epoch, Result, StateHandle, StateLineageBackend};
use parking_lot::Mutex;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Eq, PartialEq)]
enum Call {
    Load(String),
    Publish(String),
}

struct StagedSegment {
    bytes: Vec<u8>,
    validated: bool,
}

#[derive(Default)]
struct Store {
    staged: BTreeMap<StateHandle, StagedSegment>,
    committed: BTreeMap<String, Vec<u8>>,
    calls: Vec<Call>,
    published: Vec<StateHandle>,
    failure: Option<(Call, CalcFlowError)>,
}

impl Store {
    fn record(&mut self, call: Call) -> Result<()> {
        let should_fail = self.failure.as_ref().is_some_and(|(at, _)| *at == call);
        self.calls.push(call);
        if should_fail {
            return Err(self.failure.take().unwrap().1);
        }
        Ok(())
    }
}

#[derive(Default)]
struct StrictLegacyBackend {
    store: Mutex<Store>,
}

impl StrictLegacyBackend {
    fn reopen(&self) -> Self {
        Self {
            store: Mutex::new(Store {
                committed: self.store.lock().committed.clone(),
                ..Store::default()
            }),
        }
    }

    async fn stage_validated(&self, handle: &StateHandle, bytes: &[u8]) {
        self.stage_segment(handle, bytes).await.unwrap();
        self.validate_segment(handle).await.unwrap();
    }
}

#[async_trait]
impl StateLineageBackend for StrictLegacyBackend {
    fn identity_hash(&self) -> &'static str {
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }

    async fn stage_segment(&self, handle: &StateHandle, bytes: &[u8]) -> Result<()> {
        self.store.lock().staged.insert(
            handle.clone(),
            StagedSegment {
                bytes: bytes.to_vec(),
                validated: false,
            },
        );
        Ok(())
    }

    async fn validate_segment(&self, handle: &StateHandle) -> Result<()> {
        let mut store = self.store.lock();
        let staged = store
            .staged
            .get_mut(handle)
            .ok_or_else(|| missing(handle))?;
        verify_bytes(handle, &staged.bytes)?;
        staged.validated = true;
        Ok(())
    }

    async fn publish_segment(&self, handle: &StateHandle) -> Result<()> {
        let mut store = self.store.lock();
        store.record(Call::Publish(handle.segment_id().into()))?;
        if store.committed.contains_key(handle.relative_path()) {
            return Err(CalcFlowError::Conflict {
                resource: "committed segment".into(),
                key: handle.relative_path().into(),
            });
        }
        let staged = store.staged.get(handle).ok_or_else(|| missing(handle))?;
        if !staged.validated {
            return Err(CalcFlowError::InvalidArgument {
                field: "staged segment".into(),
                message: "publication requires validation".into(),
            });
        }
        let staged = store.staged.remove(handle).unwrap();
        store
            .committed
            .insert(handle.relative_path().into(), staged.bytes);
        store.published.push(handle.clone());
        Ok(())
    }

    async fn load_segment(&self, handle: &StateHandle) -> Result<Vec<u8>> {
        let mut store = self.store.lock();
        store.record(Call::Load(handle.segment_id().into()))?;
        let bytes = store
            .committed
            .get(handle.relative_path())
            .ok_or_else(|| missing(handle))?;
        verify_bytes(handle, bytes)?;
        Ok(bytes.clone())
    }

    async fn collect_orphans(&self, retained: &[StateHandle]) -> Result<usize> {
        let mut store = self.store.lock();
        let before = store.committed.len();
        store
            .committed
            .retain(|path, _| retained.iter().any(|handle| handle.relative_path() == path));
        Ok(before - store.committed.len())
    }
}

fn missing(handle: &StateHandle) -> CalcFlowError {
    CalcFlowError::NotFound {
        resource: "segment".into(),
        key: handle.relative_path().into(),
    }
}

fn verify_bytes(handle: &StateHandle, bytes: &[u8]) -> Result<()> {
    if handle.byte_len() != u64::try_from(bytes.len()).unwrap()
        || handle.sha256() != hex::encode(Sha256::digest(bytes))
    {
        return Err(CalcFlowError::CheckpointMismatch {
            message: format!("invalid bytes for {}", handle.relative_path()),
        });
    }
    Ok(())
}

fn handle(id: &str, bytes: &[u8]) -> StateHandle {
    StateHandle::new(
        "window",
        Epoch::INITIAL,
        id,
        &format!("committed/window/{id}.arrow"),
        u64::try_from(bytes.len()).unwrap(),
        &hex::encode(Sha256::digest(bytes)),
    )
    .unwrap()
}

#[tokio::test]
async fn test_empty_publication_calls_no_backend_operations() {
    let backend = StrictLegacyBackend::default();
    let lineage: &dyn StateLineageBackend = &backend;

    lineage.publish_segments(&[]).await.unwrap();

    let store = backend.store.lock();
    assert!(store.calls.is_empty());
    assert!(store.published.is_empty());
}

#[tokio::test]
async fn test_single_publication_preserves_validation_and_strict_repeat_contract() {
    let backend = StrictLegacyBackend::default();
    let segment = handle("one", b"state");
    backend.stage_segment(&segment, b"state").await.unwrap();

    assert!(matches!(
        backend.publish_segments(std::slice::from_ref(&segment)).await,
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "staged segment"
    ));
    assert!(backend.store.lock().published.is_empty());
    backend.validate_segment(&segment).await.unwrap();
    backend
        .publish_segments(std::slice::from_ref(&segment))
        .await
        .unwrap();
    assert!(matches!(
        backend.publish_segment(&segment).await,
        Err(CalcFlowError::Conflict { .. })
    ));

    backend.store.lock().calls.clear();
    backend
        .publish_segments(std::slice::from_ref(&segment))
        .await
        .unwrap();

    let store = backend.store.lock();
    assert_eq!(store.calls, [Call::Load("one".into())]);
    assert_eq!(store.published, [segment.clone()]);
    assert_eq!(store.committed[segment.relative_path()], b"state");
    assert!(store.staged.is_empty());
}

#[tokio::test]
async fn test_duplicates_publish_once_in_first_appearance_order_without_mutating_inputs() {
    let backend = StrictLegacyBackend::default();
    let last = handle("z-last", b"");
    let first = handle("a-first", b"a");
    backend.stage_validated(&last, b"").await;
    backend.stage_validated(&first, b"a").await;
    let handles = [last.clone(), first.clone(), last.clone(), first.clone()];
    let original = handles.clone();

    backend.publish_segments(&handles).await.unwrap();

    let store = backend.store.lock();
    assert_eq!(handles, original);
    assert_eq!(store.published, [last.clone(), first.clone()]);
    assert_eq!(
        store.calls,
        [
            Call::Load("z-last".into()),
            Call::Publish("z-last".into()),
            Call::Load("a-first".into()),
            Call::Publish("a-first".into()),
        ]
    );
    assert_eq!(store.committed[last.relative_path()], b"");
    assert_eq!(store.committed[first.relative_path()], b"a");
    assert!(store.staged.is_empty());
}

#[tokio::test]
async fn test_previous_session_committed_handles_need_no_staging_in_mixed_publication() {
    let previous = StrictLegacyBackend::default();
    let carried = handle("old", b"old-state");
    previous.stage_validated(&carried, b"old-state").await;
    previous.publish_segment(&carried).await.unwrap();
    let backend = previous.reopen();

    backend
        .publish_segments(std::slice::from_ref(&carried))
        .await
        .unwrap();
    {
        let mut store = backend.store.lock();
        assert_eq!(store.calls, [Call::Load("old".into())]);
        assert!(store.published.is_empty());
        assert!(store.staged.is_empty());
        store.calls.clear();
    }
    let new = handle("new", b"new-state");
    backend.stage_validated(&new, b"new-state").await;
    let handles = [carried.clone(), new.clone(), carried.clone()];
    let original = handles.clone();

    backend.publish_segments(&handles).await.unwrap();

    let store = backend.store.lock();
    assert_eq!(handles, original);
    assert_eq!(store.published, [new.clone()]);
    assert_eq!(
        store.calls,
        [
            Call::Load("old".into()),
            Call::Load("new".into()),
            Call::Publish("new".into()),
        ]
    );
    assert_eq!(store.committed[carried.relative_path()], b"old-state");
    assert_eq!(store.committed[new.relative_path()], b"new-state");
}

#[tokio::test]
async fn test_conflicting_final_handle_rejects_the_whole_batch_before_publication() {
    let first = handle("first", b"first-state");
    let second = handle("second", b"second-state");
    let different_owner = StateHandle::new(
        "other-owner",
        second.epoch(),
        second.segment_id(),
        second.relative_path(),
        second.byte_len(),
        second.sha256(),
    )
    .unwrap();
    for conflicting in [handle("second", b"different-state"), different_owner] {
        let backend = StrictLegacyBackend::default();
        backend.stage_validated(&first, b"first-state").await;
        backend.stage_validated(&second, b"second-state").await;
        let handles = [first.clone(), second.clone(), conflicting];
        let original = handles.clone();

        assert!(matches!(
            backend.publish_segments(&handles).await,
            Err(CalcFlowError::InvalidArgument { .. })
        ));

        let store = backend.store.lock();
        assert_eq!(handles, original);
        assert!(store.calls.is_empty());
        assert!(store.published.is_empty());
        assert!(store.committed.is_empty());
        assert_eq!(store.staged.len(), 2);
    }
}

#[tokio::test]
async fn test_committed_verification_error_is_preserved_and_stops_later_handles() {
    let backend = StrictLegacyBackend::default();
    let first = handle("first", b"first");
    let broken = handle("broken", b"broken");
    let later = handle("later", b"later");
    backend.stage_validated(&first, b"first").await;
    backend.stage_validated(&broken, b"broken").await;
    backend.stage_validated(&later, b"later").await;
    backend.store.lock().failure = Some((
        Call::Load("broken".into()),
        CalcFlowError::Io {
            path: broken.relative_path().into(),
            source: std::io::Error::from_raw_os_error(13),
        },
    ));

    let error = backend
        .publish_segments(&[first.clone(), broken.clone(), later])
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        CalcFlowError::Io { path, source }
            if path == broken.relative_path() && source.raw_os_error() == Some(13)
    ));
    let store = backend.store.lock();
    assert_eq!(store.published, [first]);
    assert_eq!(
        store.calls,
        [
            Call::Load("first".into()),
            Call::Publish("first".into()),
            Call::Load("broken".into()),
        ]
    );
    assert_eq!(store.staged.len(), 2);
}

#[tokio::test]
async fn test_legacy_publication_error_is_preserved_and_stops_later_handles() {
    let backend = StrictLegacyBackend::default();
    let first = handle("first", b"first");
    let broken = handle("broken", b"broken");
    let later = handle("later", b"later");
    backend.stage_validated(&first, b"first").await;
    backend.stage_validated(&broken, b"broken").await;
    backend.stage_validated(&later, b"later").await;
    backend.store.lock().failure = Some((
        Call::Publish("broken".into()),
        CalcFlowError::Io {
            path: broken.relative_path().into(),
            source: std::io::Error::from_raw_os_error(28),
        },
    ));

    let error = backend
        .publish_segments(&[first.clone(), broken.clone(), later])
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        CalcFlowError::Io { path, source }
            if path == broken.relative_path() && source.raw_os_error() == Some(28)
    ));
    let store = backend.store.lock();
    assert_eq!(store.published, [first]);
    assert_eq!(
        store.calls,
        [
            Call::Load("first".into()),
            Call::Publish("first".into()),
            Call::Load("broken".into()),
            Call::Publish("broken".into()),
        ]
    );
    assert_eq!(store.staged.len(), 2);
}

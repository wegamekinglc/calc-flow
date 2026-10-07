use super::*;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    task::Poll,
};

struct RequestProbe {
    payload: Option<Arc<Vec<u8>>>,
    pool: Arc<dyn MemoryPool>,
    entered: mpsc::Sender<usize>,
    release: Option<mpsc::Receiver<()>>,
    polls: Arc<AtomicUsize>,
}

impl Future for RequestProbe {
    type Output = Result<OperatorStateSnapshot>;
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        Poll::Pending
    }
}

impl Drop for RequestProbe {
    fn drop(&mut self) {
        self.entered.send(self.pool.reserved()).unwrap();
        if let Some(release) = self.release.take() {
            release.recv().unwrap();
        }
        drop(self.payload.take());
    }
}

fn credit(pool: &Arc<dyn MemoryPool>) -> Arc<MemoryReservation> {
    let credit = MemoryConsumer::new("funded-request-test").register(pool);
    credit.try_grow(4096).unwrap();
    Arc::new(credit)
}

#[test]
fn test_unpolled_funded_request_keeps_credit_through_actual_payload_drop() {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let owner = credit(&pool);
    let payload = Arc::new(vec![7; 4096]);
    let weak = Arc::downgrade(&payload);
    let (entered, observed) = mpsc::channel();
    let (release, blocked) = mpsc::channel();
    let polls = Arc::new(AtomicUsize::new(0));
    let request = FundedLoad::new(
        RequestProbe {
            payload: Some(payload),
            pool: pool.clone(),
            entered,
            release: Some(blocked),
            polls: polls.clone(),
        },
        owner,
    )
    .unwrap();
    let paid = pool.reserved();
    let dropping = std::thread::spawn(move || drop(request));
    assert_eq!(observed.recv().unwrap(), paid);
    assert!(weak.upgrade().is_some());
    assert_eq!(pool.reserved(), paid);
    release.send(()).unwrap();
    dropping.join().unwrap();
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    assert!(weak.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn test_closed_and_busy_load_reject_unpolled_request_before_refund() {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    for closed in [true, false] {
        let loads = LoadOwner::default();
        let (release, blocked) = tokio::sync::oneshot::channel();
        let mut first = Box::pin(loads.load(async move {
            blocked.await.unwrap();
            Ok(OperatorStateSnapshot::default())
        }));
        if closed {
            assert!(loads.close_and_drain().await.is_none());
        } else {
            assert!(futures::poll!(first.as_mut()).is_pending());
        }
        let (entered, observed) = mpsc::channel();
        let polls = Arc::new(AtomicUsize::new(0));
        let owner = credit(&pool);
        let payload = Arc::new(vec![7; 4096]);
        let weak = Arc::downgrade(&payload);
        let request = FundedLoad::new(
            RequestProbe {
                payload: Some(payload),
                pool: pool.clone(),
                entered,
                release: None,
                polls: polls.clone(),
            },
            owner,
        )
        .unwrap();
        let paid = pool.reserved();
        let error = loads.load(request).await.unwrap_err();
        assert!(matches!(
            error,
            CalcFlowError::Cancelled { .. } | CalcFlowError::Internal { .. }
        ));
        assert_eq!(observed.recv().unwrap(), paid);
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        assert!(weak.upgrade().is_none());
        assert_eq!(pool.reserved(), 0);
        if !closed {
            release.send(()).unwrap();
            first.await.unwrap();
            assert!(loads.close_and_drain().await.is_none());
        }
    }
}

#[test]
fn test_actual_request_and_segment_controls_fit_prepaid_constructor_inventory() {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let entry = OperatorManifestEntry {
        progress: BTreeMap::new(),
        inline_metadata: BTreeMap::from([(
            "metadata".into(),
            serde_json::json!({"labels": ["left", "right"], "nested": {"a": 1, "b": null}}),
        )]),
        segments: vec![
            crate::StateHandle::new(
                "join",
                crate::Epoch::INITIAL,
                "left-base",
                "committed/a.segment",
                4096,
                &"0".repeat(64),
            )
            .unwrap(),
        ],
    };
    let budget = inventory::request_bytes("join", &entry).unwrap();
    let owner = MemoryConsumer::new("constructor-inventory-test").register(&pool);
    owner.try_grow(budget).unwrap();
    let owner = Arc::new(owner);
    let mut paid = 0;
    let measured = allocation_counter::measure(|| {
        let request_id = String::from("join");
        let request = OperatorManifestEntry {
            progress: BTreeMap::new(),
            inline_metadata: entry.inline_metadata.clone(),
            segments: entry.segments.clone(),
        };
        let segment =
            crate::StateSegment::from_validated(vec![7; 4096], request.segments[0].sha256().into());
        let mut snapshot = OperatorStateSnapshot {
            inline_metadata: request.inline_metadata.clone(),
            segments: BTreeMap::from([("left-base".into(), segment)]),
        };
        let loaded = std::mem::take(&mut snapshot.segments);
        for (id, segment) in loaded {
            snapshot
                .segments
                .insert(id, segment.with_owner(owner.clone()));
        }
        let request = async move {
            drop(request_id);
            drop(request);
            Ok(snapshot)
        };
        let request = FundedLoad::new(request, owner.clone()).unwrap();
        paid = pool.reserved();
        drop(request);
    });
    assert!(
        measured.bytes_max <= paid as u64,
        "actual max {} exceeds prepaid {paid}",
        measured.bytes_max
    );
    assert_eq!(measured.bytes_current, 0);
    assert_eq!(pool.reserved(), paid);
    drop(owner);
    assert_eq!(pool.reserved(), 0);
}

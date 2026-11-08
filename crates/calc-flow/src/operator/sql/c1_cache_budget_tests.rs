use super::*;
use datafusion::{
    error::DataFusionError,
    execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryLimit, MemoryPool},
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const LIMIT: usize = 1 << 30;

#[derive(Clone, Copy, Debug)]
enum LimitKind {
    Finite,
    Unknown,
    Infinite,
}

#[derive(Debug)]
struct EnginePool {
    inner: GreedyMemoryPool,
    kind: LimitKind,
    refuse_next: AtomicBool,
    incremental_registrations: AtomicUsize,
}

impl EnginePool {
    fn new(kind: LimitKind) -> Arc<Self> {
        Arc::new(Self {
            inner: GreedyMemoryPool::new(LIMIT),
            kind,
            refuse_next: AtomicBool::new(false),
            incremental_registrations: AtomicUsize::new(0),
        })
    }
}

impl std::fmt::Display for EnginePool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(formatter)
    }
}

impl MemoryPool for EnginePool {
    fn register(&self, consumer: &MemoryConsumer) {
        if consumer.name().strip_prefix("sql-incremental:") == Some(NAME) {
            self.incremental_registrations
                .fetch_add(1, Ordering::SeqCst);
        }
    }

    fn name(&self) -> &'static str {
        "C1 control pool"
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.inner.grow(reservation, additional);
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.inner.shrink(reservation, shrink);
    }

    fn try_grow(
        &self,
        reservation: &MemoryReservation,
        additional: usize,
    ) -> datafusion::error::Result<()> {
        if self.refuse_next.swap(false, Ordering::SeqCst) {
            return Err(DataFusionError::ResourcesExhausted(
                "C1 metadata refusal".into(),
            ));
        }
        self.inner.try_grow(reservation, additional)
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }

    fn memory_limit(&self) -> MemoryLimit {
        match self.kind {
            LimitKind::Finite => MemoryLimit::Finite(LIMIT),
            LimitKind::Unknown => MemoryLimit::Unknown,
            LimitKind::Infinite => MemoryLimit::Infinite,
        }
    }
}

async fn warm_plan(
    runtime: &DataFusionRuntime,
    context: &StreamOperatorContext<'_>,
) -> (IncrementalSql, Vec<RecordBatch>) {
    let batch = input(&warm_keys(), 1);
    let mut plan = query_plan(runtime, &batch, SQL).await;
    let transaction = plan.update(&batch, context, NAME).await.unwrap();
    let records = transaction.records.clone();
    plan.commit(transaction);
    (plan, records)
}

#[tokio::test]
async fn test_c1_finite_and_unknown_pools_keep_original_ample_and_tight_paths() {
    for kind in [LimitKind::Finite, LimitKind::Unknown] {
        let pool = EnginePool::new(kind);
        let runtime =
            DataFusionRuntime::with_benchmark_pool(DataFusionConfig::default(), pool.clone())
                .unwrap();
        let job = job();
        let context = StreamOperatorContext::new(&job, NAME, None);
        let (mut plan, original) = warm_plan(&runtime, &context).await;
        assert_eq!(pool.incremental_registrations.load(Ordering::SeqCst), 2);
        assert!(plan.snapshot_aggregates.get(0).is_none());
        assert_eq!(plan.snapshot_aggregates.shared_bytes(), 0);
        let pressure = runtime.incremental_reservation("pressure");
        pressure.try_grow(LIMIT - pool.reserved()).unwrap();
        let expected_error = plan.input_workspace(1, NAME).err().unwrap().to_string();
        let keys = warm_keys();
        let error = plan
            .update(&input(&keys[9000..9001], 2), &context, NAME)
            .await
            .err()
            .unwrap();
        assert_eq!(error.to_string(), expected_error);
        assert_eq!(pool.reserved(), LIMIT);
        assert_eq!(state_totals(&plan), totals(&original));
        drop(pressure);
        let transaction = plan
            .update(&input(&keys[9000..9001], 2), &context, NAME)
            .await
            .unwrap();
        assert!(transaction.snapshot_aggregates.is_none());
        assert!(!Arc::ptr_eq(
            original[0].column(2),
            transaction.records[0].column(2)
        ));
        let mut expected = totals(&original);
        add_values(&mut expected, &keys[9000..9001], 2);
        assert_snapshot(&transaction.records, &keys, &expected);
        plan.commit(transaction);
        assert!(plan.snapshot_aggregates.get(0).is_none());
        assert_eq!(state_totals(&plan), expected);
    }
}

#[tokio::test]
async fn test_c1_metadata_refusal_falls_back_without_replacing_installed_cache() {
    let pool = EnginePool::new(LimitKind::Infinite);
    let runtime =
        DataFusionRuntime::with_benchmark_pool(DataFusionConfig::default(), pool.clone()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, NAME, None);
    let (plan, original) = warm_plan(&runtime, &context).await;
    let before = pool.reserved();
    let candidates = CandidateMap::with_hasher(RandomState::new());
    let charge = plan
        .output_charge(GROUPS, &candidates, &context, NAME)
        .await
        .unwrap();
    let reservation = runtime.incremental_reservation(NAME);
    pool.refuse_next.store(true, Ordering::SeqCst);
    let (records, cache) = plan
        .output_records(GROUPS, &candidates, &reservation, &context, NAME)
        .await
        .unwrap();
    assert!(cache.is_none());
    assert_eq!(reservation.size(), charge);
    assert!(!Arc::ptr_eq(original[0].column(2), records[0].column(2)));
    assert!(Arc::ptr_eq(
        original[0].column(2),
        &plan.snapshot_aggregates.get(0).unwrap()[0]
    ));
    assert_eq!(totals(&records), totals(&original));
    drop((records, cache, reservation));
    assert_eq!(pool.reserved(), before);
}

#[tokio::test]
async fn test_c1_shared_raw_arrays_keep_one_engine_charge() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, NAME, None);
    let (plan, original) = warm_plan(&runtime, &context).await;
    let pool = runtime.incremental_memory_pool();
    let before = pool.reserved();
    let candidates = CandidateMap::with_hasher(RandomState::new());
    let charge = plan
        .output_charge(GROUPS, &candidates, &context, NAME)
        .await
        .unwrap();
    let shared = plan.snapshot_aggregates.shared_bytes();
    let reservation = runtime.incremental_reservation(NAME);
    let (records, cache) = plan
        .output_records(GROUPS, &candidates, &reservation, &context, NAME)
        .await
        .unwrap();
    assert_eq!(reservation.size(), charge - shared);
    assert_eq!(cache.as_ref().unwrap().shared_bytes(), shared);
    for (old, current) in original.iter().zip(&records) {
        assert!(Arc::ptr_eq(old.column(2), current.column(2)));
    }
    drop((records, cache, reservation));
    assert_eq!(pool.reserved(), before);
}

#[test]
fn test_c1_cache_cap_includes_arrays_and_metadata_and_skips_unfunded_entries() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let pool = runtime.incremental_memory_pool();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "result",
        DataType::Utf8,
        false,
    )]));
    let key: Arc<[u8]> = Arc::from(&b"key"[..]);
    let empty =
        SnapshotAggregates::new(schema.clone(), runtime.incremental_reservation(NAME), true);
    let mut cache = empty
        .prepare((&schema, CHUNK_ROWS * 2, 1), std::iter::empty(), |_| {
            key.clone()
        })
        .unwrap();
    let reservation = runtime.incremental_reservation(NAME);
    reservation.try_grow(16 * 1024 * 1024).unwrap();
    cache.set_output_credit(reservation.size());
    let value = "x".repeat(512);
    let first: ArrayRef = Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
        value.as_str(),
        CHUNK_ROWS,
    )));
    let second: ArrayRef = Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
        value.as_str(),
        CHUNK_ROWS,
    )));
    cache.stage(
        (0, CHUNK_ROWS, &key),
        std::slice::from_ref(&first),
        &reservation,
    );
    cache.stage(
        (1, CHUNK_ROWS, &key),
        std::slice::from_ref(&second),
        &reservation,
    );
    assert!(cache.get(0).is_some());
    assert!(cache.get(1).is_none());
    drop(reservation);
    assert!(pool.reserved() <= 8 * 1024 * 1024);
    cache.clear();
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_c1_insufficient_transferable_credit_keeps_chunk_uncached() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "result",
        DataType::Int64,
        false,
    )]));
    let key: Arc<[u8]> = Arc::from(&b"key"[..]);
    let empty =
        SnapshotAggregates::new(schema.clone(), runtime.incremental_reservation(NAME), true);
    let mut cache = empty
        .prepare((&schema, 1, 1), std::iter::empty(), |_| key.clone())
        .unwrap();
    let reservation = MemoryConsumer::new("output").register(&runtime.incremental_memory_pool());
    reservation.try_grow(4096).unwrap();
    let column: ArrayRef = Arc::new(Int64Array::from(vec![Some(1)]));
    cache.set_output_credit(1);
    cache.stage((0, 1, &key), &[column], &reservation);
    assert!(cache.get(0).is_none());
    assert_eq!(cache.shared_bytes(), 0);
    assert_eq!(reservation.size(), 4096);
}

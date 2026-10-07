use super::*;
use crate::runtime::streaming::gather_work::TestService;
use datafusion::arrow::{
    array::{ArrayRef, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryLimit, MemoryPool};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
struct ObservedPool {
    inner: GreedyMemoryPool,
    peak: AtomicUsize,
}

impl std::fmt::Display for ObservedPool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "columnar admission test pool")
    }
}

impl ObservedPool {
    fn new(limit: usize) -> Self {
        Self {
            inner: GreedyMemoryPool::new(limit),
            peak: AtomicUsize::new(0),
        }
    }
    fn observe(&self) {
        self.peak.fetch_max(self.inner.reserved(), Ordering::SeqCst);
    }
}

impl MemoryPool for ObservedPool {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.inner.grow(reservation, additional);
        self.observe();
    }
    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.inner.shrink(reservation, shrink);
    }
    fn try_grow(
        &self,
        reservation: &MemoryReservation,
        additional: usize,
    ) -> datafusion::common::Result<()> {
        self.inner.try_grow(reservation, additional)?;
        self.observe();
        Ok(())
    }
    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
}

thread_local! {
    static FORCE_LEGACY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(in super::super) fn legacy_requested() -> bool {
    FORCE_LEGACY.get()
}

struct LegacyGuard(bool);

impl LegacyGuard {
    fn new(value: bool) -> Self {
        Self(FORCE_LEGACY.replace(value))
    }
}

impl Drop for LegacyGuard {
    fn drop(&mut self) {
        FORCE_LEGACY.set(self.0);
    }
}

fn schema(key: DataType, sequence: DataType) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", key, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", sequence, false),
    ]))
}

fn fixture(schema: &SchemaRef) -> StreamAsofJoinOperator {
    shaped_fixture(
        schema,
        &["key".into()],
        &["seq".into()],
        AsofLatePolicy::Error,
    )
}

fn shaped_fixture(
    schema: &SchemaRef,
    keys: &[String],
    sequence: &[String],
    late: AsofLatePolicy,
) -> StreamAsofJoinOperator {
    let side = |prefix: &str| {
        AsofJoinSide::new(
            keys.to_vec(),
            "time".into(),
            sequence.to_vec(),
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        std::time::Duration::ZERO,
        super::super::super::AsofStateLimits::new(100_000, 128 << 20).unwrap(),
    )
    .unwrap()
    .with_late_policy(late);
    StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap()
}

fn record(schema: &SchemaRef, keys: ArrayRef, sequences: ArrayRef, times: Vec<i64>) -> RecordBatch {
    RecordBatch::try_new(
        schema.clone(),
        vec![
            keys,
            Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")),
            sequences,
        ],
    )
    .unwrap()
}

fn batch(records: Vec<RecordBatch>) -> Batch {
    Batch::table(records, crate::BatchMetadata::default()).unwrap()
}

fn key_batch(rows: usize, distinct: usize, wide: bool) -> Batch {
    let schema = schema(DataType::Utf8, DataType::Int64);
    let keys = (0..distinct)
        .map(|id| {
            if wide {
                format!("非ASCII键{id:02}{}", "long".repeat(8))
            } else {
                format!("k{id:02}")
            }
        })
        .collect::<Vec<_>>();
    let values = (0..rows)
        .map(|row| keys[row % distinct].as_str())
        .collect::<Vec<_>>();
    let record = record(
        &schema,
        Arc::new(StringArray::from(values)),
        Arc::new(Int64Array::from_iter_values(
            0..i64::try_from(rows).unwrap(),
        )),
        (0..i64::try_from(rows).unwrap()).collect(),
    );
    batch(vec![record])
}

fn assert_chunks_equal(fast: state::PreparedLeftChunk, legacy: state::PreparedLeftChunk) {
    assert_eq!(fast.journal_version(), legacy.journal_version());
    assert_eq!(
        fast.capacity_bytes("asof").unwrap(),
        legacy.capacity_bytes("asof").unwrap()
    );
    let (fast_owner, fast) = fast.into_parts();
    let (legacy_owner, legacy) = legacy.into_parts();
    assert_eq!(fast_owner.key, legacy_owner.key);
    assert_eq!(fast_owner.record, legacy_owner.record);
    assert_eq!(
        Arc::strong_count(&fast_owner),
        Arc::strong_count(&legacy_owner)
    );
    assert_eq!(fast.checkpoint_capacities(), legacy.checkpoint_capacities());
    assert_eq!(fast.times, legacy.times);
    assert_eq!(fast.positions, legacy.positions);
    assert_eq!(fast.start, legacy.start);
    assert_eq!(fast.keys, legacy.keys);
    assert_eq!(fast.key_counts, legacy.key_counts);
    assert_eq!(fast.key_ids, legacy.key_ids);
    assert_eq!(
        fast.owners.allocation_bytes(),
        legacy.owners.allocation_bytes()
    );
    assert_eq!(fast.owners.encoded_length(), legacy.owners.encoded_length());
    assert_eq!(
        fast.owners.addresses().count(),
        legacy.owners.addresses().count()
    );
    assert_eq!(fast.sequences.capacity(), legacy.sequences.capacity());
    assert_eq!(fast.sequences.kind(), legacy.sequences.kind());
    assert_eq!(
        fast.sequences.integer_slice(0..fast.sequences.len()),
        legacy.sequences.integer_slice(0..legacy.sequences.len())
    );
    for row in 0..fast.sequences.len() {
        assert_eq!(fast.sequences.get(row), legacy.sequences.get(row));
    }
}

async fn prepared(
    operator: &mut StreamAsofJoinOperator,
    batch: &Batch,
    input: ValidatedInput,
    context: &StreamOperatorContext<'_>,
    legacy: bool,
) -> (Admission, allocation_counter::AllocationInfo) {
    let guard = LegacyGuard::new(legacy);
    let mut future = Box::pin(operator.prepare_admission(input, batch, context));
    let mut allocation = allocation_counter::AllocationInfo::default();
    let result = futures::future::poll_fn(|context| {
        let mut polled = None;
        allocation += allocation_counter::measure(|| {
            polled = Some(future.as_mut().poll(context));
        });
        polled.unwrap()
    })
    .await;
    drop(guard);
    (result.unwrap(), allocation)
}

async fn differential(batch: &Batch, service: &TestService, eligible: bool) {
    let schema = batch.table_payload().unwrap().batches()[0].schema();
    let mut operator = fixture(&schema);
    differential_input(&mut operator, batch, service, eligible).await;
}

async fn differential_input(
    operator: &mut StreamAsofJoinOperator,
    batch: &Batch,
    service: &TestService,
    eligible: bool,
) {
    let pool = operator.runtime.pool.clone();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(service.owner("columnar-differential".into()));
    let context = StreamOperatorContext::new(&job, "asof", None);
    let input = operator.validate_admission("left", batch).unwrap();
    let reserved = pool.reserved();
    let (mut legacy, _) = prepared(operator, batch, input, &context, true).await;
    let (mut fast, _) = prepared(operator, batch, input, &context, false).await;
    assert_eq!(fast.rows.len(), legacy.rows.len());
    for ((fast_order, fast_ref), (legacy_order, legacy_ref)) in fast.rows.iter().zip(&legacy.rows) {
        assert_eq!(fast_order, legacy_order);
        assert_eq!(
            (fast_ref.batch_index, fast_ref.row, fast_ref.key_index),
            (legacy_ref.batch_index, legacy_ref.row, legacy_ref.key_index)
        );
    }
    assert_eq!(fast.accepted, legacy.accepted);
    let borrowed: usize = fast
        .left_chunks
        .as_ref()
        .unwrap()
        .iter()
        .map(state::PreparedLeftChunk::borrowed_identity_rows)
        .sum();
    assert_eq!(borrowed, if eligible { 0 } else { fast.rows.len() });
    let fast_chunks = fast.left_chunks.take().unwrap();
    let legacy_chunks = legacy.left_chunks.take().unwrap();
    assert_eq!(fast_chunks.len(), legacy_chunks.len());
    for (fast, legacy) in fast_chunks.into_iter().zip(legacy_chunks) {
        assert_chunks_equal(fast, legacy);
    }
    drop((fast, legacy));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), reserved);
}

fn check_batches(cases: Vec<(Batch, bool)>) {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    for (batch, eligible) in cases {
        runtime.block_on(differential(&batch, &service, eligible));
    }
    drop(runtime);
    service.shutdown();
}

#[test]
fn dictionary_growth_and_owned_columns_match_legacy() {
    let cases = [0, 1, 1024]
        .into_iter()
        .flat_map(|rows| {
            [1, 3, 5, 64].into_iter().flat_map(move |keys| {
                [false, true]
                    .into_iter()
                    .map(move |wide| (key_batch(rows, keys, wide), rows != 0))
            })
        })
        .collect();
    check_batches(cases);
}

fn integer_arrays() -> Vec<ArrayRef> {
    use datafusion::arrow::array::{
        Int8Array, Int16Array, Int32Array, UInt8Array, UInt16Array, UInt32Array,
    };
    vec![
        Arc::new(Int8Array::from(vec![i8::MIN, -1, 0, i8::MAX])),
        Arc::new(Int16Array::from(vec![i16::MIN, -1, 0, i16::MAX])),
        Arc::new(Int32Array::from(vec![i32::MIN, -1, 0, i32::MAX])),
        Arc::new(Int64Array::from(vec![i64::MIN, -1, 0, i64::MAX])),
        Arc::new(UInt8Array::from(vec![0, 1, u8::MAX - 1, u8::MAX])),
        Arc::new(UInt16Array::from(vec![0, 1, u16::MAX - 1, u16::MAX])),
        Arc::new(UInt32Array::from(vec![0, 1, u32::MAX - 1, u32::MAX])),
        Arc::new(UInt64Array::from(vec![0, 1, u64::MAX - 1, u64::MAX])),
    ]
}

#[test]
fn native_integer_extremes_and_equal_time_order_match_canonical_legacy() {
    let sequences = integer_arrays().into_iter().map(|values| {
        let schema = schema(DataType::Utf8, values.data_type().clone());
        (
            batch(vec![record(
                &schema,
                Arc::new(StringArray::from(vec![""; 4])),
                values,
                vec![0; 4],
            )]),
            true,
        )
    });
    let keys = integer_arrays().into_iter().map(|values| {
        let schema = schema(values.data_type().clone(), DataType::Int64);
        (
            batch(vec![record(
                &schema,
                values,
                Arc::new(Int64Array::from(vec![0; 4])),
                vec![0; 4],
            )]),
            true,
        )
    });
    check_batches(sequences.chain(keys).collect());
}

#[test]
fn utf8_empty_non_ascii_and_record_cuts_match_legacy() {
    use datafusion::arrow::array::LargeStringArray;
    let values = vec!["", "\0", "a", "é", "中"];
    let mut cases = Vec::new();
    for keys in [
        Arc::new(StringArray::from(values.clone())) as ArrayRef,
        Arc::new(LargeStringArray::from(values)) as ArrayRef,
    ] {
        let schema = schema(keys.data_type().clone(), DataType::Int64);
        cases.push((
            batch(vec![record(
                &schema,
                keys,
                Arc::new(Int64Array::from(vec![0; 5])),
                vec![0; 5],
            )]),
            true,
        ));
    }
    let full = key_batch(1024, 64, false);
    let record = &full.table_payload().unwrap().batches()[0];
    for cut in [0, 1, 127, 1023, 1024] {
        cases.push((
            batch(vec![record.slice(0, cut), record.slice(cut, 1024 - cut)]),
            true,
        ));
    }
    check_batches(cases);
}

#[test]
fn reversed_and_string_sequence_shapes_keep_legacy_preparation() {
    let ordered = key_batch(1024, 64, false);
    let indices = datafusion::arrow::array::UInt32Array::from_iter_values((0..1024).rev());
    let reversed = datafusion::arrow::compute::take_record_batch(
        &ordered.table_payload().unwrap().batches()[0],
        &indices,
    )
    .unwrap();
    let schema = schema(DataType::Utf8, DataType::Utf8);
    let strings = record(
        &schema,
        Arc::new(StringArray::from(vec!["key"; 4])),
        Arc::new(StringArray::from(vec!["", "a", "é", "中"])),
        vec![0; 4],
    );
    check_batches(vec![
        (batch(vec![reversed]), false),
        (batch(vec![strings]), false),
    ]);
}

async fn allocation_pair(rows: usize, service: &TestService) -> (u64, u64) {
    let batch = key_batch(rows, 64, false);
    let schema = batch.table_payload().unwrap().batches()[0].schema();
    let mut operator = fixture(&schema);
    let pool = operator.runtime.pool.clone();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(service.owner("columnar-allocation".into()));
    let context = StreamOperatorContext::new(&job, "asof", None);
    let input = operator.validate_admission("left", &batch).unwrap();
    // Prime the existing worker registry before comparing caller allocations.
    drop(
        operator
            .prepare_admission(input, &batch, &context)
            .await
            .unwrap(),
    );
    let (legacy, legacy_caller) = prepared(&mut operator, &batch, input, &context, true).await;
    let (fast, fast_caller) = prepared(&mut operator, &batch, input, &context, false).await;
    let (legacy_thread, legacy_chunk) = legacy.left_chunks.as_ref().unwrap()[0]
        .preparation_allocation()
        .unwrap();
    let (fast_thread, fast_chunk) = fast.left_chunks.as_ref().unwrap()[0]
        .preparation_allocation()
        .unwrap();
    assert_eq!(legacy_thread, fast_thread);
    assert_eq!(fast_thread == std::thread::current().id(), rows <= 4096);
    assert_eq!(legacy_chunk.count_total, fast_chunk.count_total + 1);
    assert_eq!(
        legacy_chunk.bytes_total - fast_chunk.bytes_total,
        (rows * size_of::<(&LeftOrder, u32)>()) as u64
    );
    assert!(
        fast_chunk.count_total <= 128,
        "actual chunk allocation: {fast_chunk:?}"
    );
    assert!(
        fast_caller.count_total <= 512,
        "caller allocation: {fast_caller:?}"
    );
    eprintln!(
        "rows={rows}; caller legacy={legacy_caller:?}; caller fast={fast_caller:?}; actual chunk legacy={legacy_chunk:?}; actual chunk fast={fast_chunk:?}"
    );
    let counts = (fast_caller.count_total, fast_chunk.count_total);
    drop((legacy, fast));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
    counts
}

#[test]
fn caller_and_actual_worker_allocations_scale_with_distinct_keys() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let small = runtime.block_on(allocation_pair(1024, &service));
    let large = runtime.block_on(allocation_pair(64_000, &service));
    assert!(
        large.0 <= small.0 + 64,
        "caller allocations grew per row: {small:?} -> {large:?}"
    );
    assert!(
        large.1 <= small.1 + 2,
        "worker allocations grew per row: {small:?} -> {large:?}"
    );
    drop(runtime);
    service.shutdown();
}

async fn admit(
    operator: &mut StreamAsofJoinOperator,
    batch: &Batch,
    context: &StreamOperatorContext<'_>,
    legacy: bool,
) {
    let input = operator.validate_admission("left", batch).unwrap();
    let (admission, _) = prepared(operator, batch, input, context, legacy).await;
    operator
        .install_admission("left", admission, input, context)
        .await
        .unwrap();
}

async fn assert_capture_equal(
    fast: &mut StreamAsofJoinOperator,
    legacy: &mut StreamAsofJoinOperator,
    context: &StreamOperatorContext<'_>,
) -> crate::OperatorStateSnapshot {
    use crate::StreamOperator;
    fast.prepare_checkpoint_async(context).await.unwrap();
    legacy.prepare_checkpoint_async(context).await.unwrap();
    let fast = fast.checkpoint(crate::Epoch::INITIAL).unwrap();
    let legacy = legacy.checkpoint(crate::Epoch::INITIAL).unwrap();
    assert_eq!(fast.inline_metadata, legacy.inline_metadata);
    assert_eq!(fast.segments, legacy.segments);
    fast
}

fn output_records(output: &mut crate::EdgeCollector) -> Vec<RecordBatch> {
    output
        .drain("output")
        .iter()
        .flat_map(|message| {
            message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
                .iter()
                .cloned()
        })
        .collect()
}

async fn prefix_capture(
    fast: &mut StreamAsofJoinOperator,
    legacy: &mut StreamAsofJoinOperator,
    job: &StreamJobContext,
) -> crate::OperatorStateSnapshot {
    use crate::{
        IngressProgress, IngressProgressSnapshot, IngressState, OperatorMetadata, StreamOperator,
    };
    use std::collections::BTreeMap;
    let frontier = crate::EventTime::from_micros(512);
    let progress = IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, Some(frontier)),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, Some(frontier)),
        ),
    ]));
    let context = StreamOperatorContext::with_ingress_progress(job, "asof", None, progress);
    let mut fast_output = crate::EdgeCollector::new(fast.output_ports().to_vec());
    let mut legacy_output = crate::EdgeCollector::new(legacy.output_ports().to_vec());
    fast.on_watermark(frontier, &context, &mut fast_output)
        .await
        .unwrap();
    legacy
        .on_watermark(frontier, &context, &mut legacy_output)
        .await
        .unwrap();
    let records = output_records(&mut fast_output);
    assert_eq!(records, output_records(&mut legacy_output));
    let sequences = records
        .iter()
        .flat_map(|record| {
            record
                .column_by_name("left__seq")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    assert_eq!(sequences, (0..512).collect::<Vec<_>>());
    assert_eq!(fast.status.emitted_left_rows, 512);
    assert_eq!(fast.status, legacy.status);
    assert_capture_equal(fast, legacy, &context).await
}

async fn checkpoint_continuation(service: &TestService) {
    use crate::StreamOperator;
    let first = key_batch(1024, 64, true);
    let schema = first.table_payload().unwrap().batches()[0].schema();
    let mut fast = fixture(&schema);
    let mut legacy = fixture(&schema);
    let mut restored = fixture(&schema);
    let mut legacy_restored = fixture(&schema);
    let pools = [
        fast.runtime.pool.clone(),
        legacy.runtime.pool.clone(),
        restored.runtime.pool.clone(),
        legacy_restored.runtime.pool.clone(),
    ];
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(service.owner("columnar-wire".into()));
    let context = StreamOperatorContext::new(&job, "asof", None);
    admit(&mut fast, &first, &context, false).await;
    admit(&mut legacy, &first, &context, true).await;
    assert_eq!(fast.status, legacy.status);
    let snapshot = assert_capture_equal(&mut fast, &mut legacy, &context).await;
    let repeated = assert_capture_equal(&mut fast, &mut legacy, &context).await;
    assert_eq!(snapshot.inline_metadata, repeated.inline_metadata);
    assert_eq!(snapshot.segments, repeated.segments);
    restored.restore(&snapshot).unwrap();
    legacy_restored.restore(&snapshot).unwrap();
    let next_record = repeated_key_batch(&schema, 1024, 1024);
    let mut columns = next_record.columns().to_vec();
    columns[1] = Arc::new(TimestampMicrosecondArray::from(vec![2048; 1024]).with_timezone("UTC"));
    let next = batch(vec![RecordBatch::try_new(schema, columns).unwrap()]);
    admit(&mut fast, &next, &context, false).await;
    admit(&mut legacy, &next, &context, true).await;
    admit(&mut restored, &next, &context, false).await;
    admit(&mut legacy_restored, &next, &context, true).await;
    let live_continued = assert_capture_equal(&mut fast, &mut legacy, &context).await;
    let continued = assert_capture_equal(&mut restored, &mut legacy_restored, &context).await;
    let live_prefix = prefix_capture(&mut fast, &mut legacy, &job).await;
    let restored_prefix = prefix_capture(&mut restored, &mut legacy_restored, &job).await;
    drop((
        snapshot,
        repeated,
        live_continued,
        continued,
        live_prefix,
        restored_prefix,
        fast,
        legacy,
        restored,
        legacy_restored,
    ));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    for pool in pools {
        assert_eq!(pool.reserved(), 0);
    }
}

#[test]
fn journal_segments_and_restored_continuation_match_legacy() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(checkpoint_continuation(&service));
    drop(runtime);
    service.shutdown();
}

async fn boundary_fallback(service: &TestService) {
    use crate::StreamOperator;
    let schema = schema(DataType::Utf8, DataType::Int64);
    let mut operator = fixture(&schema);
    let mut restored = fixture(&schema);
    let pools = [operator.runtime.pool.clone(), restored.runtime.pool.clone()];
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(service.owner("columnar-boundary".into()));
    let context = StreamOperatorContext::new(&job, "asof", None);
    let seed = batch(vec![record(
        &schema,
        Arc::new(StringArray::from(vec!["key"; 2])),
        Arc::new(Int64Array::from(vec![0, 1])),
        vec![100, 0],
    )]);
    admit(&mut operator, &seed, &context, false).await;
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(crate::Epoch::INITIAL).unwrap();
    restored.restore(&snapshot).unwrap();
    for candidate in [&mut operator, &mut restored] {
        for (time, sequence, eligible) in [(50, 2, false), (100, 1, true)] {
            let input = batch(vec![record(
                &schema,
                Arc::new(StringArray::from(vec!["key"])),
                Arc::new(Int64Array::from(vec![sequence])),
                vec![time],
            )]);
            differential_input(candidate, &input, service, eligible).await;
        }
        let duplicate = batch(vec![record(
            &schema,
            Arc::new(StringArray::from(vec!["key"])),
            Arc::new(Int64Array::from(vec![0])),
            vec![100],
        )]);
        let input = candidate.validate_admission("left", &duplicate).unwrap();
        let bytes = candidate.status.state_bytes;
        let error = candidate
            .prepare_admission(input, &duplicate, &context)
            .await
            .err()
            .unwrap();
        assert!(matches!(
            error,
            crate::CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofDuplicateIdentity,
                ..
            }
        ));
        assert_eq!(candidate.status.left.duplicate_rows, 1);
        assert_eq!(candidate.status.state_bytes, bytes);
    }
    drop((operator, restored, snapshot));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    for pool in pools {
        assert_eq!(pool.reserved(), 0);
    }
}

#[test]
fn restored_and_disordered_live_maximum_guards_overlap_and_duplicates() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(boundary_fallback(&service));
    drop(runtime);
    service.shutdown();
}

#[test]
fn mixed_late_input_uses_legacy_after_payload_compaction() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let schema = schema(DataType::Utf8, DataType::Int64);
        let mut operator = shaped_fixture(
            &schema,
            &["key".into()],
            &["seq".into()],
            AsofLatePolicy::Drop,
        );
        operator.status.left.watermark_micros = Some(crate::EventTime::from_micros(50));
        let input = batch(vec![record(
            &schema,
            Arc::new(StringArray::from(vec!["key"; 3])),
            Arc::new(Int64Array::from(vec![0, 1, 2])),
            vec![0, 50, 100],
        )]);
        differential_input(&mut operator, &input, &service, false).await;
        assert_eq!(operator.status.left.late_rows, 1);
        assert_eq!(operator.status.left.accepted_rows, 0);
    });
    drop(runtime);
    service.shutdown();
}

fn payload_batch() -> Batch {
    let source = key_batch(1024, 64, false);
    let record = &source.table_payload().unwrap().batches()[0];
    let mut fields = record.schema().fields().iter().cloned().collect::<Vec<_>>();
    fields.push(Arc::new(Field::new("payload", DataType::Utf8, false)));
    let mut columns = record.columns().to_vec();
    let payload = "变量宽度".repeat(128);
    columns.push(Arc::new(StringArray::from(vec![payload.as_str(); 1024])));
    batch(vec![
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap(),
    ])
}

#[test]
fn full_projected_zero_column_and_oversized_slices_keep_payload_owners() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let input = payload_batch();
        let schema = input.table_payload().unwrap().batches()[0].schema();
        for projection in [None, Some(vec![2]), Some(Vec::new())] {
            let mut operator = fixture(&schema);
            if let Some(columns) = projection {
                operator.set_output_projection(columns).unwrap();
            }
            differential_input(&mut operator, &input, &service, true).await;
            let record = &input.table_payload().unwrap().batches()[0];
            let sliced = batch(vec![record.slice(17, 3)]);
            differential_input(&mut operator, &sliced, &service, true).await;
        }
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn composite_keys_and_sequences_keep_validated_legacy_ranges() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("seq", DataType::Int64, false),
            Field::new("extra", DataType::Int64, false),
        ]));
        let input = batch(vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(vec![""; 4])),
                    Arc::new(TimestampMicrosecondArray::from(vec![0; 4]).with_timezone("UTC")),
                    Arc::new(Int64Array::from(vec![i64::MIN, -1, 0, i64::MAX])),
                    Arc::new(Int64Array::from(vec![i64::MIN, -1, 0, i64::MAX])),
                ],
            )
            .unwrap(),
        ]);
        for (keys, sequence) in [
            (vec!["key".into(), "extra".into()], vec!["seq".into()]),
            (vec!["key".into()], vec!["seq".into(), "extra".into()]),
        ] {
            let mut operator = shaped_fixture(&schema, &keys, &sequence, AsofLatePolicy::Error);
            differential_input(&mut operator, &input, &service, false).await;
        }
    });
    drop(runtime);
    service.shutdown();
}

async fn budget_case(
    input: &Batch,
    limit: usize,
    legacy: bool,
    service: &TestService,
) -> (Option<StreamingFailureReason>, StreamAsofJoinStatus, usize) {
    use crate::{OperatorMetadata, StreamOperator};
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut operator = fixture(&schema);
    let pool = Arc::new(ObservedPool::new(limit));
    operator.runtime.pool = pool.clone();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(service.owner("columnar-budget".into()));
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let guard = LegacyGuard::new(legacy);
    let result = operator
        .process_data("left", input.clone(), &context, &mut output)
        .await;
    drop(guard);
    let reason = match result {
        Ok(()) => None,
        Err(crate::CalcFlowError::OperatorReason { reason_code, .. }) => {
            assert_eq!(operator.status.state_rows, 0);
            assert!(operator.state.batches.is_empty());
            Some(reason_code)
        }
        Err(error) => panic!("unexpected budget failure: {error}"),
    };
    let outcome = (
        reason,
        operator.status.clone(),
        pool.peak.load(Ordering::SeqCst),
    );
    drop((operator, output));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
    outcome
}

fn preparation_thresholds(input: &Batch) -> Vec<usize> {
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let operator = fixture(&schema);
    let validated = ValidatedInput {
        index: 0,
        watermark: None,
    };
    let identity = operator.identity_workspace(input, validated).unwrap();
    let payload = operator.input_workspace(input, validated).unwrap();
    let descriptor = operator.reserve_left_work(1).unwrap();
    vec![
        identity.reservation.size(),
        identity.reservation.size() + payload.size(),
        identity.reservation.size() + payload.size() + descriptor.size(),
    ]
}

#[test]
fn exact_reservation_thresholds_and_whole_refund_match_legacy() {
    let input = key_batch(4097, 64, false);
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let baseline = runtime.block_on(budget_case(&input, 128 << 20, true, &service));
    assert_eq!(baseline.0, None);
    let mut thresholds = preparation_thresholds(&input);
    thresholds.push(baseline.2);
    let mut limits = thresholds
        .into_iter()
        .flat_map(|limit| [limit - 1, limit])
        .collect::<Vec<_>>();
    limits.push(32 << 20);
    for limit in limits {
        let legacy = runtime.block_on(budget_case(&input, limit, true, &service));
        let fast = runtime.block_on(budget_case(&input, limit, false, &service));
        assert_eq!(fast, legacy, "limit={limit}");
    }
    drop(runtime);
    service.shutdown();
}

fn held_retirement(
    operator: &StreamAsofJoinOperator,
    context: &StreamOperatorContext<'_>,
) -> (
    std::sync::mpsc::Sender<()>,
    tokio::task::JoinHandle<()>,
    std::sync::Weak<Vec<u8>>,
) {
    let reservation = operator.reserve_workspace(4096).unwrap();
    let ticket = operator.retirement.register(context).unwrap();
    let owner = Arc::new(vec![0_u8; 4096]);
    let weak = Arc::downgrade(&owner);
    let (release, gate) = std::sync::mpsc::channel();
    let worker = tokio::task::spawn_blocking(move || {
        gate.recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        drop((owner, reservation));
        drop(ticket);
    });
    (release, worker, weak)
}

#[tokio::test(flavor = "current_thread")]
async fn eligible_entry_waits_survive_reset_restore_and_cancelled_wait() {
    use crate::{OperatorMetadata, StreamOperator};
    let input = key_batch(1, 1, false);
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut operator = fixture(&schema);
    let saved = operator.capture(crate::Epoch::INITIAL).unwrap();
    let pool = operator.runtime.pool.clone();
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, cancellation.clone());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let (release, worker, weak) = held_retirement(&operator, &context);
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    for restore in [false, true] {
        let mut pending =
            Box::pin(operator.process_data("left", input.clone(), &context, &mut output));
        assert!(futures::poll!(pending.as_mut()).is_pending());
        drop(pending);
        assert_eq!(operator.status.left.accepted_rows, 0);
        if restore {
            operator.restore(&saved).unwrap();
        } else {
            operator.reset().unwrap();
        }
    }
    let mut pending = Box::pin(operator.process_data("left", input.clone(), &context, &mut output));
    assert!(futures::poll!(pending.as_mut()).is_pending());
    cancellation.cancel();
    assert!(matches!(
        pending.await,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(pool.reserved(), 4096);
    assert!(weak.upgrade().is_some());
    let mut drain = Box::pin(job.gather_owner().close_and_drain());
    assert!(futures::poll!(drain.as_mut()).is_pending());
    release.send(()).unwrap();
    worker.await.unwrap();
    assert!(drain.await.is_empty());
    assert!(weak.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
    let resumed = StreamJobContext::new(2, "asof", JsonMap::new(), None, CancellationToken::new());
    let resumed_context = StreamOperatorContext::new(&resumed, "asof", None);
    operator
        .process_data("left", input, &resumed_context, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.status.left.accepted_rows, 1);
    drop((operator, saved, output));
    assert!(resumed.gather_owner().close_and_drain().await.is_empty());
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn eligible_commit_wait_cancellation_keeps_committed_state_and_credit() {
    use crate::StreamOperator;
    let input = key_batch(1, 1, false);
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut operator = fixture(&schema);
    let pool = operator.runtime.pool.clone();
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, cancellation.clone());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let validated = operator.validate_admission("left", &input).unwrap();
    let admission = operator
        .prepare_admission(validated, &input, &context)
        .await
        .unwrap();
    assert_eq!(
        admission.left_chunks.as_ref().unwrap()[0].borrowed_identity_rows(),
        0
    );
    let (release, worker, weak) = held_retirement(&operator, &context);
    let mut pending = Box::pin(operator.install_admission("left", admission, validated, &context));
    assert!(futures::poll!(pending.as_mut()).is_pending());
    cancellation.cancel();
    assert!(matches!(
        pending.await,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(operator.status.left.accepted_rows, 1);
    assert_eq!(operator.status.state_rows, 1);
    assert!(pool.reserved() >= 4096);
    assert!(weak.upgrade().is_some());
    let snapshot = operator.capture(crate::Epoch::INITIAL).unwrap();
    operator.reset().unwrap();
    operator.restore(&snapshot).unwrap();
    assert_eq!(operator.status.left.accepted_rows, 1);
    let mut drain = Box::pin(job.gather_owner().close_and_drain());
    assert!(futures::poll!(drain.as_mut()).is_pending());
    drop((operator, snapshot));
    assert!(weak.upgrade().is_some());
    release.send(()).unwrap();
    worker.await.unwrap();
    assert!(drain.await.is_empty());
    assert!(weak.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
}

fn large_fixture() -> (StreamAsofJoinOperator, SchemaRef) {
    let (operator, schema) = identity_fixture();
    let spec = StreamAsofJoinSpec::new(
        operator.spec.left().clone(),
        operator.spec.right().clone(),
        std::time::Duration::ZERO,
        super::super::super::AsofStateLimits::new(100_000, 128 << 20).unwrap(),
    )
    .unwrap();
    (
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap(),
        schema,
    )
}

fn eligible_prepare(rows: usize) -> usize {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (mut operator, schema) = large_fixture();
    let pool = operator.runtime.pool.clone();
    let borrowed = runtime.block_on(async {
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
            .with_gather_owner(service.owner("columnar-red".into()));
        let context = StreamOperatorContext::new(&job, "asof", None);
        let batch = Batch::table(
            vec![repeated_key_batch(&schema, 0, rows)],
            crate::BatchMetadata::default(),
        )
        .unwrap();
        let admitted = operator
            .prepare_admission(
                ValidatedInput {
                    index: 0,
                    watermark: None,
                },
                &batch,
                &context,
            )
            .await
            .unwrap();
        assert_eq!(admitted.rows.len(), rows);
        assert!(
            admitted
                .rows
                .iter()
                .all(|(_, reference)| reference.key_index == 0)
        );
        let borrowed = admitted
            .left_chunks
            .as_ref()
            .unwrap()
            .iter()
            .map(state::PreparedLeftChunk::borrowed_identity_rows)
            .sum();
        drop(admitted);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        borrowed
    });
    drop(runtime);
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
    borrowed
}

#[test]
fn eligible_owned_worker_avoids_borrowed_identity_vector() {
    assert_eq!(eligible_prepare(64_000), 0);
}

#[test]
fn eligible_inline_avoids_borrowed_identity_vector() {
    assert_eq!(eligible_prepare(1024), 0);
}

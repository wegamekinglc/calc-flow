use super::*;

fn keyed_left_batch(keys: Vec<i64>, times: Vec<i64>) -> Batch {
    let original = left_batch(times);
    let record = &original.table_payload().unwrap().batches()[0];
    let mut columns = record.columns().to_vec();
    columns[0] = Arc::new(Int64Array::from(keys));
    Batch::table(
        vec![RecordBatch::try_new(record.schema(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

async fn admitted_left(
    operator: &mut StreamJoinOperator,
    context: &StreamOperatorContext<'_>,
    source: &Batch,
) -> AdmissionBundle {
    let plan = operator.begin_batch("left", source).unwrap();
    let mut bundle = operator.admission_bundle(&plan, source).unwrap();
    for (index, record) in source.table_payload().unwrap().batches().iter().enumerate() {
        let base = (0..index)
            .map(|previous| source.table_payload().unwrap().batches()[previous].num_rows())
            .sum::<usize>();
        operator
            .admit_record(record, &plan, "left", context, &mut bundle, base)
            .await
            .unwrap();
    }
    bundle
}

#[tokio::test]
async fn test_retained_rows_skips_unretained_blocks_without_row_visits() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    // The opposite side ended, so the retain threshold expires every row before
    // any state work: the admission path must not visit them row by row.
    let context = progress_context(
        &job,
        (IngressState::Active, Some(0)),
        (IngressState::Ended, None),
    );
    let source = keyed_left_batch(vec![7; 130], (0..130).collect());
    let bundle = Box::pin(admitted_left(&mut operator, &context, &source)).await;
    assert_eq!(bundle.admitted.len(), 130);
    reset_join_work();
    let retained = retained_rows(
        &bundle.admitted,
        &[0],
        "match",
        None,
        &operator.key_dictionary,
        &context,
        &mut operator.key_work,
    )
    .await
    .unwrap();
    assert!(retained.is_empty());
    assert_eq!(
        join_work().admission_segment_lookups,
        1,
        "fully expired batches may only resolve group boundaries, not visit rows"
    );
    assert_eq!(
        join_work().charge_column_resolutions,
        0,
        "all-zero blocks must skip the payload charge column walk"
    );
}

#[tokio::test]
async fn test_retained_rows_visits_only_retained_bits() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    // Rows with event time >= 100 retain; blocks 0..64 stay empty, 64..128 are
    // mixed, and the tail block is fully retained.
    let context = progress_context(
        &job,
        (IngressState::Active, Some(0)),
        (IngressState::Active, Some(60_000_100)),
    );
    let source = keyed_left_batch(vec![7; 130], (0..130).collect());
    let record = source.table_payload().unwrap().batches()[0].clone();
    let bundle = Box::pin(admitted_left(&mut operator, &context, &source)).await;
    assert_eq!(bundle.admitted.len(), 130);
    reset_join_work();
    let retained = retained_rows(
        &bundle.admitted,
        &[0],
        "match",
        None,
        &operator.key_dictionary,
        &context,
        &mut operator.key_work,
    )
    .await
    .unwrap();
    assert_eq!(retained.len(), 30);
    assert_eq!(
        retained
            .iter()
            .map(|row| (row.event_time, row.row_id, row.charge))
            .collect::<Vec<_>>(),
        (100..130)
            .map(|row| {
                (
                    EventTime::from_micros(row),
                    u64::try_from(row).unwrap(),
                    state_row_charge(&record, usize::try_from(row).unwrap(), &[0], "match")
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>()
    );
    assert_eq!(join_work().charge_column_resolutions, 6);
    assert!(
        join_work().admission_segment_lookups <= 40,
        "mixed blocks must visit only set bits, observed {}",
        join_work().admission_segment_lookups
    );
}

fn keyed_batch(left: bool, keys: Vec<i64>, times: Vec<i64>) -> Batch {
    let original = if left {
        left_batch(times)
    } else {
        right_batch(times)
    };
    let record = &original.table_payload().unwrap().batches()[0];
    let mut columns = record.columns().to_vec();
    columns[0] = Arc::new(Int64Array::from(keys));
    Batch::table(
        vec![RecordBatch::try_new(record.schema(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn retain_progress(job: &StreamJobContext) -> StreamOperatorContext<'_> {
    progress_context(
        job,
        (IngressState::Active, Some(0)),
        (IngressState::Active, Some(60_000_100)),
    )
}

/// The dimension side needs its own low-watermark context: its rows arrive
/// before the right watermark advances past the retention threshold.
fn dimension_progress(job: &StreamJobContext) -> StreamOperatorContext<'_> {
    progress_context(
        job,
        (IngressState::Active, Some(0)),
        (IngressState::Active, None),
    )
}

#[tokio::test]
async fn test_project_admitted_uses_zero_copy_views_for_unretained_groups() {
    let records = || {
        vec![
            keyed_left_batch(vec![7; 30], (100..130).collect())
                .table_payload()
                .unwrap()
                .batches()[0]
                .clone(),
            keyed_left_batch(vec![7; 40], (0..40).collect())
                .table_payload()
                .unwrap()
                .batches()[0]
                .clone(),
        ]
    };
    let job = job();
    let context = retain_progress(&job);
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    operator.set_output_projection(vec![0, 1, 3, 4]).unwrap();
    let input = Batch::table(records(), BatchMetadata::default()).unwrap();
    let prepared = operator
        .prepare_batch("left", &input, &context)
        .await
        .unwrap();
    assert_eq!(prepared.chunks.len(), 30);
    assert!(
        !prepared.admitted.projection_is_view(0),
        "retained groups keep the copied payload path"
    );
    assert!(
        prepared.admitted.projection_is_view(1),
        "groups without retained rows must not copy their payloads"
    );
    drop(prepared);
    // The zero-copy group still feeds the output gather with its rows.
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    operator.set_output_projection(vec![0, 1, 3, 4]).unwrap();
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "right",
            right_batch(vec![0]),
            &dimension_progress(&job),
            &mut output,
        )
        .await
        .unwrap();
    operator
        .process_data(
            "left",
            Batch::table(records(), BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    assert_eq!(operator.status().emitted_match_rows, 70);
    let mut times: Vec<i64> = Vec::new();
    let mut keys: Vec<i64> = Vec::new();
    for message in output.drain("output") {
        let record = &message
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()[0];
        times.extend(
            record
                .column(1)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap()
                .values()
                .iter(),
        );
        keys.extend(
            record
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .iter(),
        );
    }
    assert_eq!(times, (100..130).chain(0..40).collect::<Vec<_>>());
    assert_eq!(keys, vec![7_i64; 70]);
}

#[tokio::test]
async fn test_key_identity_resolves_each_distinct_key_once() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "right",
            keyed_batch(false, (0..64).collect(), vec![0; 64]),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    let keys = (0..65_536)
        .map(|row| i64::from(row % 64))
        .collect::<Vec<_>>();
    let probe = keyed_batch(true, keys, vec![0; 65_536]);
    let plan = operator.begin_batch("left", &probe).unwrap();
    let mut bundle = operator.admission_bundle(&plan, &probe).unwrap();
    operator
        .admit_record(
            &probe.table_payload().unwrap().batches()[0],
            &plan,
            "left",
            &context,
            &mut bundle,
            0,
        )
        .await
        .unwrap();
    let hashes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed = Arc::clone(&hashes);
    let hook = typed_key::observe(Arc::new(move |phase, _| {
        if phase == typed_key::KeyPhase::Hash {
            observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }));
    reset_join_work();
    let prepared = operator
        .native_probe_keys(&plan, &bundle.admitted)
        .unwrap()
        .unwrap();
    drop(hook);
    assert_eq!(
        hashes.load(std::sync::atomic::Ordering::SeqCst),
        64,
        "canonical hashing must run once per distinct key"
    );
    assert_eq!(
        join_work().dictionary_probes,
        64,
        "persistent-dictionary probes must follow distinct keys, not rows"
    );
    assert_eq!(prepared.keys.len(), 64);
    assert_eq!(prepared.id(0), 0);
    assert_eq!(prepared.id(65), 1);
    assert_eq!(prepared.id(65_535), 63);
}

use datafusion::arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, Int8Array, LargeStringArray, UInt64Array,
};
use datafusion::execution::memory_pool::{
    MemoryConsumer, MemoryPool, MemoryReservation, UnboundedMemoryPool,
};
use std::fmt;

struct IdentityOutcome {
    ids: Vec<u32>,
    keys: Vec<Option<Vec<u8>>>,
    inventory: (usize, u64, usize, bool),
}

async fn resolve_identity(
    parents: &[Arc<RecordBatch>],
    indices: &[usize],
    rows: &[(usize, usize, bool)],
    per_row: bool,
) -> IdentityOutcome {
    let admitted = rows
        .iter()
        .map(|&(parent, row, retain)| AdmittedRow {
            record: columnar::RowPayload::Rowed {
                parent: Arc::clone(&parents[parent]),
                row,
            },
            event_time: EventTime::from_micros(0),
            row_id: u64::try_from(row).unwrap(),
            retain,
        })
        .collect::<Vec<_>>();
    let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
    let master = MemoryConsumer::new("identity-case").register(&pool);
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut dictionary = key_dictionary::Dictionary::default();
    let forced = per_row.then(key_dictionary::force_per_row_keys);
    let prepared = dictionary
        .prepare(&admitted, indices, master, Some(&context), "match")
        .await
        .unwrap()
        .unwrap();
    drop(forced);
    let keys = prepared
        .keys
        .iter()
        .map(|key| key.as_ref().map(|key| key.as_slice().to_vec()))
        .collect();
    let ids = prepared.ids.clone();
    dictionary.commit(prepared.dictionary);
    for &(parent, row, retain) in rows {
        if retain {
            let frame =
                encode_join_key_columns_v1(parents[parent].columns(), row, indices).unwrap();
            assert!(
                dictionary.canonical(&frame).is_some(),
                "retained key frames must resolve canonically"
            );
        }
    }
    IdentityOutcome {
        ids,
        keys,
        inventory: dictionary.inventory(),
    }
}

fn record_of(columns: Vec<ArrayRef>) -> Arc<RecordBatch> {
    let fields = (0..columns.len())
        .map(|index| {
            Field::new(
                format!("column{index}"),
                columns[index].data_type().clone(),
                true,
            )
        })
        .collect::<Vec<_>>();
    Arc::new(RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap())
}

async fn assert_identity_paths_agree(
    tag: &str,
    parents: &[Arc<RecordBatch>],
    indices: &[usize],
    rows: &[(usize, usize, bool)],
    hash_notes: usize,
) {
    let hashes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed = Arc::clone(&hashes);
    let hook = typed_key::observe(Arc::new(move |phase, _| {
        if phase == typed_key::KeyPhase::Hash {
            observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }));
    let deduplicated = Box::pin(resolve_identity(parents, indices, rows, false)).await;
    drop(hook);
    let per_row = Box::pin(resolve_identity(parents, indices, rows, true)).await;
    assert_eq!(deduplicated.ids, per_row.ids, "{tag}: per-row ids");
    assert_eq!(deduplicated.keys, per_row.keys, "{tag}: canonical owners");
    assert_eq!(
        deduplicated.inventory, per_row.inventory,
        "{tag}: committed dictionary inventory"
    );
    assert_eq!(
        hashes.load(std::sync::atomic::Ordering::SeqCst),
        hash_notes,
        "{tag}: default path must hash distinct keys, not rows"
    );
}

async fn assert_typed_identity_paths_agree() {
    let multi = record_of(vec![
        Arc::new(StringArray::from(vec![
            Some("a"),
            None,
            Some("b"),
            Some("a"),
        ])),
        Arc::new(Int64Array::from(vec![Some(1), Some(1), None, Some(1)])),
        Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(true),
        ])),
        Arc::new(Date32Array::from(vec![
            Some(0),
            None,
            Some(-719_162),
            Some(0),
        ])),
        Arc::new(UInt64Array::from(vec![
            Some(0),
            Some(u64::MAX),
            Some(1),
            Some(0),
        ])),
        Arc::new(Int8Array::from(vec![
            Some(-1),
            None,
            Some(i8::MAX),
            Some(-1),
        ])),
        Arc::new(LargeStringArray::from(vec!["l0", "l1", "l0", "l0"])),
    ]);
    Box::pin(assert_identity_paths_agree(
        "multi-column typed keys with nulls",
        &[multi],
        &[0, 1, 2, 3, 4, 5, 6],
        &[
            (0, 0, true),
            (0, 1, true),
            (0, 2, false),
            (0, 3, true),
            (0, 1, true),
        ],
        3,
    ))
    .await;
    let wide_indices = record_of(vec![Arc::new(Int64Array::from(vec![3, 4]))]);
    for (columns, notes) in [(63_usize, 2_usize), (64, 384), (65, 390)] {
        Box::pin(assert_identity_paths_agree(
            "small-key gate boundary",
            &[Arc::clone(&wide_indices)],
            &vec![0; columns],
            &[(0, 0, true), (0, 1, false), (0, 0, true), (0, 1, true)],
            notes,
        ))
        .await;
    }
    let left = record_of(vec![Arc::new(Int64Array::from(vec![7, 8]))]);
    let right = record_of(vec![Arc::new(Int64Array::from(vec![8, 7]))]);
    // Deduplication is per group, so a key repeated across records pays one
    // pending-hit resolution per record, never per row.
    Box::pin(assert_identity_paths_agree(
        "cross-group duplicates",
        &[left, right],
        &[0],
        &[(0, 0, true), (0, 1, false), (1, 0, true), (1, 1, true)],
        4,
    ))
    .await;
}

#[tokio::test]
async fn test_deduplicated_key_identity_matches_per_row_resolution() {
    let signed = record_of(vec![Arc::new(Int64Array::from(vec![7, 8, 9]))]);
    Box::pin(assert_identity_paths_agree(
        "mixed retain and birth order",
        &[signed],
        &[0],
        &[
            (0, 0, false),
            (0, 1, true),
            (0, 0, true),
            (0, 1, false),
            (0, 2, true),
        ],
        3,
    ))
    .await;
    Box::pin(assert_identity_paths_agree(
        "unresolved rows precede the first retained row of their key",
        &[record_of(vec![Arc::new(Int64Array::from(vec![1, 2]))])],
        &[0],
        &[(0, 0, false), (0, 1, true), (0, 0, true)],
        2,
    ))
    .await;
    let strings = record_of(vec![Arc::new(StringArray::from(vec![
        "", "é\0", "AAPL", "AAPL",
    ]))]);
    Box::pin(assert_identity_paths_agree(
        "empty and repeated strings",
        &[strings],
        &[0],
        &[(0, 0, true), (0, 1, false), (0, 2, true), (0, 3, true)],
        3,
    ))
    .await;
    let nulls = record_of(vec![Arc::new(StringArray::from(vec![
        None,
        Some(""),
        Some("AAPL"),
        None,
        Some(""),
        Some("AAPL"),
    ]))]);
    Box::pin(assert_identity_paths_agree(
        "null and empty string share one key column",
        &[nulls],
        &[0],
        &[
            (0, 0, false),
            (0, 1, true),
            (0, 2, true),
            (0, 3, true),
            (0, 4, false),
            (0, 5, true),
        ],
        2,
    ))
    .await;
    let wide = record_of(vec![Arc::new(StringArray::from(vec![
        "x".repeat(5000).as_str(),
        "y".repeat(5000).as_str(),
        "x".repeat(5000).as_str(),
    ]))]);
    Box::pin(assert_identity_paths_agree(
        "chunked keys larger than one quantum",
        &[wide],
        &[0],
        &[(0, 0, true), (0, 1, false), (0, 2, true)],
        8,
    ))
    .await;
    Box::pin(assert_typed_identity_paths_agree()).await;
}

/// Refuses the very first funded grow so `Dictionary::prepare` cannot fund
/// its batch-local interning tier and must use the per-row path.
#[derive(Debug)]
struct FirstGrantRefusalPool {
    inner: UnboundedMemoryPool,
    refused: std::sync::atomic::AtomicBool,
}

impl fmt::Display for FirstGrantRefusalPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("identity-first-refusal")
    }
}

impl MemoryPool for FirstGrantRefusalPool {
    fn name(&self) -> &'static str {
        "identity-first-refusal"
    }
    fn grow(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.grow(credit, bytes);
    }
    fn shrink(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.shrink(credit, bytes);
    }
    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
    fn memory_limit(&self) -> datafusion::execution::memory_pool::MemoryLimit {
        self.inner.memory_limit()
    }
    fn try_grow(&self, credit: &MemoryReservation, bytes: usize) -> datafusion::common::Result<()> {
        use std::sync::atomic::Ordering;
        if bytes > 0 && !self.refused.swap(true, Ordering::SeqCst) {
            return Err(datafusion::common::DataFusionError::ResourcesExhausted(
                "local interning budget refused".into(),
            ));
        }
        self.inner.try_grow(credit, bytes)
    }
}

#[tokio::test]
async fn test_local_table_refusal_falls_back_to_per_row_identity() {
    let parent = record_of(vec![Arc::new(Int64Array::from(vec![1, 2, 3]))]);
    let rows = [
        (0_usize, 0_usize, true),
        (0, 1, true),
        (0, 0, false),
        (0, 2, true),
    ];
    let admitted = rows
        .iter()
        .map(|&(_, row, retain)| AdmittedRow {
            record: columnar::RowPayload::Rowed {
                parent: Arc::clone(&parent),
                row,
            },
            event_time: EventTime::from_micros(0),
            row_id: u64::try_from(row).unwrap(),
            retain,
        })
        .collect::<Vec<_>>();
    let pool: Arc<dyn MemoryPool> = Arc::new(FirstGrantRefusalPool {
        inner: UnboundedMemoryPool::default(),
        refused: std::sync::atomic::AtomicBool::new(false),
    });
    let master = MemoryConsumer::new("identity-fallback").register(&pool);
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut dictionary = key_dictionary::Dictionary::default();
    let hashes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed = Arc::clone(&hashes);
    let hook = typed_key::observe(Arc::new(move |phase, _| {
        if phase == typed_key::KeyPhase::Hash {
            observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }));
    reset_join_work();
    let refused = dictionary
        .prepare(&admitted, &[0], master, Some(&context), "match")
        .await
        .unwrap()
        .unwrap();
    drop(hook);
    assert_eq!(
        hashes.load(std::sync::atomic::Ordering::SeqCst),
        rows.len(),
        "a refused local table must fall back to per-row resolution"
    );
    assert_eq!(join_work().dictionary_probes, 3);
    let mut expected = key_dictionary::Dictionary::default();
    let oracle_pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
    let plain = MemoryConsumer::new("identity-fallback-oracle").register(&oracle_pool);
    let forced = key_dictionary::force_per_row_keys();
    let oracle = expected
        .prepare(&admitted, &[0], plain, Some(&context), "match")
        .await
        .unwrap()
        .unwrap();
    drop(forced);
    assert_eq!(refused.ids, oracle.ids);
    assert_eq!(refused.keys.len(), oracle.keys.len());
    for (refused_key, oracle_key) in refused.keys.iter().zip(&oracle.keys) {
        assert_eq!(
            refused_key.as_ref().map(|k| k.as_slice()),
            oracle_key.as_ref().map(|k| k.as_slice())
        );
    }
}

/// Builds one admitted group over non-contiguous physical rows, so every
/// offset the key paths read comes from promoted sparse indices rather than
/// the dense `first + ordinal` prefix.
async fn sparse_admitted(
    operator: &mut StreamJoinOperator,
    source: &Batch,
    context: &StreamOperatorContext<'_>,
) -> AdmittedBatch {
    let mut admitted = AdmittedBatch::reserve(operator, source).unwrap();
    let index = admitted.begin_record(0, 0);
    let mut quantum = columnar::Quantum::default();
    for (physical, retain) in [(0_usize, true), (2, false), (3, true), (5, true)] {
        admitted
            .ensure_selection(index, physical, context, &mut quantum)
            .await
            .unwrap();
        admitted.push_ready(
            index,
            physical,
            EventTime::from_micros(i64::try_from(physical).expect("test offsets fit i64")),
            retain,
        );
    }
    admitted.complete_record();
    admitted
}

/// Resolves one sparse group's identity through either path and reports the
/// ids, canonical owners, committed inventory, and retained-frame checks.
async fn sparse_identity(
    admitted: &AdmittedBatch,
    parent: &Arc<RecordBatch>,
    retained: &[usize],
    per_row: bool,
) -> (Vec<u32>, Vec<Option<Vec<u8>>>, (usize, u64, usize, bool)) {
    let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
    let master = MemoryConsumer::new("sparse-identity").register(&pool);
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut dictionary = key_dictionary::Dictionary::default();
    let forced = per_row.then(key_dictionary::force_per_row_keys);
    let prepared = dictionary
        .prepare(admitted, &[0], master, Some(&context), "match")
        .await
        .unwrap()
        .unwrap();
    drop(forced);
    let keys = prepared
        .keys
        .iter()
        .map(|key| key.as_ref().map(|key| key.as_slice().to_vec()))
        .collect();
    let ids = prepared.ids.clone();
    dictionary.commit(prepared.dictionary);
    for &physical in retained {
        let frame = encode_join_key_columns_v1(parent.columns(), physical, &[0]).unwrap();
        assert!(
            dictionary.canonical(&frame).is_some(),
            "retained sparse row {physical} must resolve its canonical frame"
        );
    }
    (ids, keys, dictionary.inventory())
}

#[tokio::test]
async fn test_sparse_promoted_group_interns_through_both_identity_paths() {
    let parent = record_of(vec![Arc::new(Int64Array::from(vec![7, 8, 7, 9, 7, 8]))]);
    let source = Batch::table(vec![parent.as_ref().clone()], BatchMetadata::default()).unwrap();
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let admitted = Box::pin(sparse_admitted(&mut operator, &source, &context)).await;
    assert_eq!(admitted.len(), 4);
    assert_eq!(
        admitted.sparse_index_len(),
        4,
        "offsets must flow through promoted sparse indices, not a dense prefix"
    );
    let deduplicated = Box::pin(sparse_identity(&admitted, &parent, &[0, 3, 5], false)).await;
    let forced = key_dictionary::force_per_row_keys();
    let per_row = Box::pin(sparse_identity(&admitted, &parent, &[0, 3, 5], true)).await;
    drop(forced);
    assert_eq!(deduplicated.0, per_row.0, "sparse group ids");
    assert_eq!(deduplicated.1, per_row.1, "sparse canonical owners");
    assert_eq!(deduplicated.2, per_row.2, "sparse dictionary inventory");
    assert_eq!(
        deduplicated.0,
        vec![0, 0, 1, 2],
        "the repeated sparse key shares one identity and births follow row order"
    );
}

#[tokio::test]
async fn test_stream_join_identity_paths_produce_identical_state() {
    let default = Box::pin(run_join_identity_scenario(false)).await;
    let forced = key_dictionary::force_per_row_keys();
    let per_row = Box::pin(run_join_identity_scenario(true)).await;
    drop(forced);
    assert_eq!(default, per_row);
}

async fn run_join_identity_scenario(
    _per_row: bool,
) -> (
    Vec<Vec<Vec<datafusion::arrow::array::ArrayData>>>,
    Vec<Vec<u8>>,
    Vec<i64>,
    Vec<u64>,
    Vec<u64>,
    (usize, u64, usize, bool),
) {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let plain = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "right",
            keyed_batch(false, (0..4).collect(), vec![0; 4]),
            &plain,
            &mut output,
        )
        .await
        .unwrap();
    let progress = progress_context(
        &job,
        (IngressState::Active, Some(0)),
        (IngressState::Active, Some(60_000_100)),
    );
    let first = vec![
        keyed_batch(true, vec![0, 1, 0, 2], vec![200, 50, 300, 150])
            .table_payload()
            .unwrap()
            .batches()[0]
            .clone(),
        keyed_batch(true, vec![1, 3], vec![120, 30])
            .table_payload()
            .unwrap()
            .batches()[0]
            .clone(),
    ];
    operator
        .process_data(
            "left",
            Batch::table(first, BatchMetadata::default()).unwrap(),
            &progress,
            &mut output,
        )
        .await
        .unwrap();
    operator
        .process_data(
            "left",
            keyed_batch(true, vec![0, 1], vec![500, 400]),
            &progress,
            &mut output,
        )
        .await
        .unwrap();
    let emitted = output
        .drain("output")
        .iter()
        .map(|message| {
            message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
                .iter()
                .map(|record| {
                    record
                        .columns()
                        .iter()
                        .map(Array::to_data)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let state = operator
        .state
        .left
        .iter()
        .map(|row| (row.encoded_key.as_slice().to_vec(), row.charge))
        .unzip::<_, _, Vec<_>, Vec<_>>();
    let inventory = operator.key_dictionary.inventory();
    (
        emitted,
        state.0,
        operator
            .state
            .left
            .iter()
            .map(|row| row.event_time.as_micros())
            .collect(),
        operator.state.left.iter().map(|row| row.row_id).collect(),
        state.1,
        inventory,
    )
}

use std::sync::{Arc, Weak};

use datafusion::arrow::{
    array::{
        Array, ArrayRef, Int64Array, LargeBinaryArray, StringArray, TimestampMicrosecondArray,
    },
    buffer::{Buffer, OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::{
    GreedyMemoryPool, MemoryConsumer, MemoryPool, MemoryReservation,
};
use tokio_util::bytes::Bytes;

use super::{CompactBatch, compact};
use crate::EventTime;
use crate::operator::join::{StoredRow, state_row_charge};

const PADDING: usize = 1 << 20;
const CONSTRUCTOR_BUDGET: usize = 128 << 10;
const KEY: [u8; 17] = [5, 0, 0, 0, 0, 8, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0];

struct SourceBacking {
    bytes: Vec<u8>,
    _credit: MemoryReservation,
}

struct SourceLease(Arc<SourceBacking>);

impl AsRef<[u8]> for SourceLease {
    fn as_ref(&self) -> &[u8] {
        &self.0.bytes
    }
}

type SourceRecord = (RecordBatch, Weak<SourceBacking>);

fn source_record(pool: &Arc<dyn MemoryPool>) -> SourceRecord {
    let credit = MemoryConsumer::new("compact-credit-source-backing").register(pool);
    credit.try_grow(PADDING + 3).unwrap();
    let mut bytes = Vec::with_capacity(PADDING + 3);
    bytes.resize(PADDING, 0x78);
    bytes.extend_from_slice(b"abc");
    assert_eq!(bytes.capacity(), PADDING + 3);
    let owner = Arc::new(SourceBacking {
        bytes,
        _credit: credit,
    });
    let weak = Arc::downgrade(&owner);
    let values = Buffer::from(Bytes::from_owner(SourceLease(owner)));
    let padding = i32::try_from(PADDING).unwrap();
    let strings = StringArray::try_new(
        OffsetBuffer::new(ScalarBuffer::from(vec![
            0,
            padding,
            padding + 1,
            padding + 3,
        ])),
        values,
        None,
    )
    .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("value", DataType::Utf8, false),
    ]));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![99, 7, 7])),
            Arc::new(TimestampMicrosecondArray::from(vec![0, 95, 96])),
            Arc::new(strings),
        ],
    )
    .unwrap();
    (record, weak)
}

fn selected_rows(source: &RecordBatch) -> [StoredRow; 2] {
    [(1, 95, 121), (2, 96, 122)].map(|(index, time, charge)| {
        assert_eq!(
            state_row_charge(source, index, &[0], "compact-credit").unwrap(),
            charge
        );
        StoredRow {
            record: source.slice(index, 1).into(),
            event_time: EventTime::from_micros(time),
            row_id: u64::try_from(index - 1).unwrap(),
            charge,
            encoded_key: Arc::new(KEY.to_vec().into()),
        }
    })
}

fn assert_compacted(compacted: &CompactBatch) {
    let record = &compacted.record;
    assert_eq!(record.num_rows(), 2);
    let keys = record
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let times = record
        .column(1)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    let strings = record
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(keys.values().as_ref(), &[7, 7]);
    assert_eq!(times.values().as_ref(), &[95, 96]);
    assert_eq!(strings.value_offsets(), &[0, 1, 3]);
    assert_eq!(strings.value_data(), b"abc");
    assert_eq!((strings.value(0), strings.value(1)), ("a", "bc"));
    assert_eq!(
        (
            state_row_charge(record, 0, &[0], "compact-credit").unwrap(),
            state_row_charge(record, 1, &[0], "compact-credit").unwrap()
        ),
        (121, 122)
    );
}

#[test]
fn test_compaction_borrows_large_backing_with_a_small_constructor_budget() {
    assert_flat_concat_peak_is_prepaid();
    let source_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(PADDING + 3));
    let (source, owner) = source_record(&source_pool);
    let rows = selected_rows(&source);
    let strings = source
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let original = strings.value_data().as_ptr();
    for row in &rows {
        let view = row.record.column_view(2);
        let value = view.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(value.value_data().as_ptr(), original);
        assert_eq!(value.value_data().len(), PADDING + 3);
    }
    assert_eq!(source_pool.reserved(), PADDING + 3);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(CONSTRUCTOR_BUDGET));
    let workspace = MemoryConsumer::new("compact-credit-workspace").register(&pool);
    let references = rows.iter().collect::<Vec<_>>();
    let mut result = None;
    let allocations = allocation_counter::measure(|| {
        result = Some(compact(
            &references,
            source.schema(),
            &workspace,
            &|| Ok(()),
        ));
    });
    let compacted = result
        .unwrap()
        .expect("borrowed input backing is already owned; only new constructors need credit");
    assert!(allocations.bytes_max <= u64::try_from(CONSTRUCTOR_BUDGET).unwrap());
    assert!(pool.reserved() > 0);
    assert_eq!(source_pool.reserved(), PADDING + 3);
    assert_compacted(&compacted);
    let values = compacted
        .record
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_ne!(values.value_data().as_ptr(), original);
    assert_eq!(strings.value(0).len(), PADDING);
    assert!(strings.value(0).bytes().all(|byte| byte == 0x78));
    assert_eq!((strings.value(1), strings.value(2)), ("a", "bc"));
    assert!(owner.upgrade().is_some());
    drop(references);
    drop(rows);
    drop(source);
    assert!(owner.upgrade().is_none());
    assert_eq!(source_pool.reserved(), 0);
    assert_compacted(&compacted);
    drop(compacted);
    drop(workspace);
    assert_eq!(pool.reserved(), 0);
}

fn assert_flat_concat_peak_is_prepaid() {
    let arrays: [ArrayRef; 3] = [
        Arc::new(Int64Array::from(vec![Some(7), None, Some(9)])),
        Arc::new(StringArray::from(vec![Some("a"), None, Some("bc")])),
        Arc::new(LargeBinaryArray::from(vec![
            Some(b"a".as_slice()),
            None,
            Some(b"bc".as_slice()),
        ])),
    ];
    for source in arrays {
        assert_flat_concat_allocations(&source);
    }
}

fn assert_flat_concat_allocations(source: &ArrayRef) {
    let selected = (0..129)
        .map(|row| source.slice(row % source.len(), 1))
        .collect::<Vec<_>>();
    let arrays = selected.iter().map(Arc::as_ref).collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        source.data_type().clone(),
        true,
    )]));
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(CONSTRUCTOR_BUDGET));
    let workspace = MemoryConsumer::new("flat-concat-peak").register(&pool);
    let mut paid = 0;
    let allocations = allocation_counter::measure(|| {
        let credit = workspace.new_empty();
        super::budget::payload_seed()
            .and_then(|bytes| super::accounting::reserve(&credit, bytes))
            .unwrap();
        let funding = super::Funding::new([Arc::clone(&schema), Arc::clone(&schema)], credit, None);
        super::admit(&arrays, &workspace, &funding, &|| Ok(())).unwrap();
        let output = datafusion::arrow::compute::concat(&arrays).unwrap();
        paid = pool.reserved();
        assert_eq!(output.len(), 129);
        assert_eq!(output.null_count(), 43);
        for (index, expected) in selected.iter().enumerate() {
            assert_eq!(output.slice(index, 1).to_data(), expected.to_data());
        }
        drop((output, funding));
    });
    eprintln!(
        "flat concat {:?}: heap peak={}B, prepaid={}B, live after drop={}B",
        source.data_type(),
        allocations.bytes_max,
        paid,
        allocations.bytes_current
    );
    assert!(
        allocations.bytes_max <= u64::try_from(paid).unwrap(),
        "{allocations:?}, paid={paid}"
    );
    assert_eq!(allocations.bytes_current, 0);
    assert_eq!(source.len(), 3);
    drop(workspace);
    assert_eq!(pool.reserved(), 0);
}

#[derive(Debug)]
struct CompactionPool {
    inner: GreedyMemoryPool,
    peak: std::sync::atomic::AtomicUsize,
    releases: std::sync::atomic::AtomicUsize,
}

impl CompactionPool {
    fn new(limit: usize) -> Self {
        Self {
            inner: GreedyMemoryPool::new(limit),
            peak: std::sync::atomic::AtomicUsize::new(0),
            releases: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn note_peak(&self) {
        self.peak
            .fetch_max(self.inner.reserved(), std::sync::atomic::Ordering::Relaxed);
    }

    fn peak(&self) -> usize {
        self.peak.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn releases(&self) -> usize {
        self.releases.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl std::fmt::Display for CompactionPool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, formatter)
    }
}

impl MemoryPool for CompactionPool {
    fn name(&self) -> &'static str {
        "compaction-test"
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.inner.grow(reservation, additional);
        self.note_peak();
    }

    fn shrink(&self, reservation: &MemoryReservation, bytes: usize) {
        self.inner.shrink(reservation, bytes);
        if bytes != 0 {
            self.releases
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn try_grow(
        &self,
        reservation: &MemoryReservation,
        additional: usize,
    ) -> datafusion::common::Result<()> {
        self.inner.try_grow(reservation, additional)?;
        self.note_peak();
        Ok(())
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }

    fn memory_limit(&self) -> datafusion::execution::memory_pool::MemoryLimit {
        self.inner.memory_limit()
    }
}

fn flat_compaction_source(rows: usize) -> Arc<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("text", DataType::Utf8, true),
    ]));
    Arc::new(
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![7; rows])),
                Arc::new(TimestampMicrosecondArray::from_iter_values(
                    (0..rows).map(|row| i64::try_from(row).unwrap()),
                )),
                Arc::new(
                    (0..rows)
                        .map(|row| (row % 3 != 0).then_some(if row % 2 == 0 { "a" } else { "bc" }))
                        .collect::<StringArray>(),
                ),
            ],
        )
        .unwrap(),
    )
}

fn compaction_rows(source: &Arc<RecordBatch>, indices: &[usize]) -> Vec<StoredRow> {
    parent_compaction_rows(indices.iter().map(|&row| (source, row)))
}

fn parent_compaction_rows<'a>(
    selected: impl Iterator<Item = (&'a Arc<RecordBatch>, usize)>,
) -> Vec<StoredRow> {
    let key = Arc::new(KEY.to_vec().into());
    selected
        .enumerate()
        .map(|(id, (source, row))| StoredRow {
            record: crate::operator::join::columnar::RowPayload::Rowed {
                parent: Arc::clone(source),
                row,
            },
            event_time: EventTime::from_micros(i64::try_from(row).unwrap()),
            row_id: u64::try_from(id).unwrap(),
            charge: state_row_charge(source, row, &[0], "compact-many").unwrap(),
            encoded_key: Arc::clone(&key),
        })
        .collect()
}

fn selected_parent_records(
    sources: &[Arc<RecordBatch>],
    selected: &[(usize, usize)],
) -> RecordBatch {
    let records = selected
        .iter()
        .map(|&(parent, row)| sources[parent].slice(row, 1))
        .collect::<Vec<_>>();
    datafusion::arrow::compute::concat_batches(&sources[0].schema(), &records).unwrap()
}

fn assert_single_contiguous_run_releases_padded_backing() {
    let source_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(PADDING + 3));
    let (source, owner) = source_record(&source_pool);
    let source = Arc::new(source);
    let rows = compaction_rows(&source, &[1, 2]);
    let references = rows.iter().collect::<Vec<_>>();
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(CONSTRUCTOR_BUDGET));
    let workspace = MemoryConsumer::new("compact-one-run").register(&pool);
    let compacted = compact(&references, source.schema(), &workspace, &|| Ok(())).unwrap();
    assert_compacted(&compacted);
    drop(references);
    drop(rows);
    drop(source);
    assert!(
        owner.upgrade().is_none(),
        "the two-row run must not retain the padding"
    );
    assert_eq!(source_pool.reserved(), 0);
    drop(workspace);
    assert!(pool.reserved() > 0);
    assert_compacted(&compacted);
    drop(compacted);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_compaction_dense_rows_fit_bounded_constructor_credit() {
    const ROWS: usize = 8_193;
    assert_single_contiguous_run_releases_padded_backing();
    let source = flat_compaction_source(ROWS);
    let indices = (0..ROWS).collect::<Vec<_>>();
    let rows = compaction_rows(&source, &indices);
    let references = rows.iter().collect::<Vec<_>>();
    let tracking = Arc::new(CompactionPool::new(16 << 20));
    let pool: Arc<dyn MemoryPool> = tracking.clone();
    let workspace = MemoryConsumer::new("compact-many").register(&pool);
    let mut result = None;
    let allocations = allocation_counter::measure(|| {
        result = Some(compact(
            &references,
            source.schema(),
            &workspace,
            &|| Ok(()),
        ));
    });
    let compacted = result
        .unwrap()
        .expect("contiguous slices must fit the constructor budget");
    assert_eq!(compacted.record, *source);
    assert!(allocations.bytes_max <= u64::try_from(tracking.peak()).unwrap());
    assert!(
        allocations.count_total < 512,
        "contiguous selection must not allocate one Arrow slice per row: {allocations:?}"
    );
    drop(workspace);
    assert!(pool.reserved() > 0, "output backing must remain funded");
    assert!(
        usize::try_from(allocations.bytes_current).unwrap() <= pool.reserved(),
        "live compacted allocations must remain funded after scratch retirement: {allocations:?}"
    );
    assert_eq!(compacted.record, *source);
    drop(compacted);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(source.column(2).null_count(), ROWS.div_ceil(3));
}

const MIXED_VIEW_VALUES: [[Option<&str>; 6]; 2] = [
    [
        Some("unselected first parent prefix"),
        Some("first parent long selected value"),
        None,
        Some("another first parent long value"),
        Some("short"),
        Some("first tail"),
    ],
    [
        Some("unselected second parent prefix"),
        Some("second parent long selected value"),
        Some("second parent middle"),
        None,
        Some("other"),
        Some("second tail"),
    ],
];

type MixedSource = (
    Arc<RecordBatch>,
    Vec<Weak<SourceBacking>>,
    Arc<dyn MemoryPool>,
);

fn mixed_compaction_source(parent: usize) -> MixedSource {
    use datafusion::arrow::array::{BooleanArray, DictionaryArray, ListArray, StringViewArray};
    use datafusion::arrow::datatypes::Int32Type;

    let source_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let views = StringViewArray::from_iter(MIXED_VIEW_VALUES[parent]);
    let (views, buffers, nulls) = views.into_parts();
    let mut owners = Vec::new();
    let buffers = buffers
        .iter()
        .map(|buffer| {
            let bytes = buffer.as_slice().to_vec();
            let credit = MemoryConsumer::new("mixed-view-source").register(&source_pool);
            credit.try_grow(bytes.capacity()).unwrap();
            let owner = Arc::new(SourceBacking {
                bytes,
                _credit: credit,
            });
            owners.push(Arc::downgrade(&owner));
            Buffer::from(Bytes::from_owner(SourceLease(owner)))
        })
        .collect::<Vec<_>>();
    let views = StringViewArray::try_new(views, buffers, nulls).unwrap();
    let dictionary: DictionaryArray<Int32Type> = if parent == 0 {
        ["red", "blue", "green", "red", "blue", "green"]
    } else {
        ["orange", "green", "blue", "orange", "green", "blue"]
    }
    .into_iter()
    .collect();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![7; 6])),
        Arc::new(BooleanArray::from(vec![
            Some(true),
            None,
            Some(false),
            Some(true),
            None,
            Some(false),
        ])),
        Arc::new(dictionary),
        Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1), None]),
            None,
            Some(vec![]),
            Some(vec![Some(4)]),
            Some(vec![None, Some(5)]),
            Some(vec![Some(6)]),
        ])),
        Arc::new(views),
    ];
    let names = ["key", "flag", "kind", "nested", "view"];
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .zip(names)
            .map(|(column, name)| {
                Field::new(name, column.data_type().clone(), column.null_count() > 0)
            })
            .collect::<Vec<_>>(),
    ));
    (
        Arc::new(RecordBatch::try_new(schema, columns).unwrap()),
        owners,
        source_pool,
    )
}

fn mixed_parent_selection() -> Vec<(usize, usize)> {
    let mut selected = (0..2_049)
        .map(|row| (row % 2, 1 + (3 * row) % 4))
        .collect::<Vec<_>>();
    selected[1_022..1_027].copy_from_slice(&[(0, 1), (0, 2), (0, 3), (1, 3), (1, 4)]);
    selected
}

fn assert_selected_view_values(record: &RecordBatch, selected: &[(usize, usize)]) {
    let view = record
        .column(4)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::StringViewArray>()
        .unwrap();
    for (index, &(parent, row)) in selected.iter().enumerate() {
        let actual = (!view.is_null(index)).then(|| view.value(index));
        assert_eq!(actual, MIXED_VIEW_VALUES[parent][row]);
    }
}

#[test]
fn test_compaction_sparse_mixed_rows_preserve_values_and_output_funding() {
    let (left, left_owners, left_pool) = mixed_compaction_source(0);
    let (right, right_owners, right_pool) = mixed_compaction_source(1);
    let sources = [left, right];
    let owners = [left_owners, right_owners].concat();
    let originals = sources.each_ref().map(|source| source.as_ref().clone());
    let selected = mixed_parent_selection();
    let expected = selected_parent_records(&sources, &selected);
    let rows = parent_compaction_rows(
        selected
            .iter()
            .map(|&(parent, row)| (&sources[parent], row)),
    );
    let references = rows.iter().collect::<Vec<_>>();
    let tracking = Arc::new(CompactionPool::new(128 << 20));
    let pool: Arc<dyn MemoryPool> = tracking.clone();
    let workspace = MemoryConsumer::new("compact-mixed").register(&pool);
    let mut result = None;
    let allocations = allocation_counter::measure(|| {
        result = Some(compact(
            &references,
            sources[0].schema(),
            &workspace,
            &|| Ok(()),
        ));
    });
    let compacted = result.unwrap().unwrap();
    assert_eq!(compacted.record, expected);
    for (source, original) in sources.iter().zip(&originals) {
        assert_eq!(source.as_ref(), original);
    }
    assert!(allocations.bytes_max <= u64::try_from(tracking.peak()).unwrap());
    assert!(tracking.releases() > 1);
    drop(expected);
    drop(originals);
    drop(references);
    drop(rows);
    drop(sources);
    drop(workspace);
    assert!(
        pool.reserved() > 0,
        "the private CompactBatch still owns output credit"
    );
    assert!(
        usize::try_from(allocations.bytes_current).unwrap() <= pool.reserved(),
        "live compacted allocations must remain funded after scratch retirement: {allocations:?}"
    );
    assert!(owners.iter().all(|owner| owner.upgrade().is_some()));
    assert!(left_pool.reserved() > 0 && right_pool.reserved() > 0);
    assert_selected_view_values(&compacted.record, &selected);
    drop(compacted);
    assert_eq!(pool.reserved(), 0);
    assert!(owners.iter().all(|owner| owner.upgrade().is_none()));
    assert_eq!((left_pool.reserved(), right_pool.reserved()), (0, 0));
}

fn assert_dense_run_scan_is_cancellable() {
    let source = flat_compaction_source(8_193);
    let original = source.as_ref().clone();
    let indices = (0..8_193).collect::<Vec<_>>();
    let rows = compaction_rows(&source, &indices);
    let references = rows.iter().collect::<Vec<_>>();
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(16 << 20));
    let workspace = MemoryConsumer::new("compact-dense-cancel").register(&pool);
    let checks = std::cell::Cell::new(0);
    let result = compact(&references, source.schema(), &workspace, &|| {
        checks.set(checks.get() + 1);
        if checks.get() == 257 {
            return Err(crate::CalcFlowError::Cancelled {
                run_id: "compact-dense-cancel".into(),
            });
        }
        Ok(())
    });
    assert!(matches!(
        result,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(checks.get(), 257);
    assert_eq!(*source, original);
    drop(workspace);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_compaction_cancellation_after_partial_progress_releases_credit() {
    assert_dense_run_scan_is_cancellable();
    let source = flat_compaction_source(4);
    let original = source.as_ref().clone();
    let indices = (0..8_193).map(|row| (3 * row) % 4).collect::<Vec<_>>();
    let rows = compaction_rows(&source, &indices);
    let references = rows.iter().collect::<Vec<_>>();
    let tracking = Arc::new(CompactionPool::new(16 << 20));
    let pool: Arc<dyn MemoryPool> = tracking.clone();
    let workspace = MemoryConsumer::new("compact-cancel").register(&pool);
    let cancelled = std::cell::Cell::new(false);
    let result = compact(&references, source.schema(), &workspace, &|| {
        if tracking.releases() > 0 {
            cancelled.set(true);
            return Err(crate::CalcFlowError::Cancelled {
                run_id: "compact-cancel".into(),
            });
        }
        Ok(())
    });
    assert!(
        cancelled.get(),
        "the first bounded scratch owner must have retired"
    );
    assert!(matches!(
        result,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(*source, original);
    drop(workspace);
    assert_eq!(pool.reserved(), 0);
}

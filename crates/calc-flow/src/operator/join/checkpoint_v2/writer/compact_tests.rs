use std::sync::{Arc, Weak};

use datafusion::arrow::{
    array::{Int64Array, StringArray, TimestampMicrosecondArray},
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

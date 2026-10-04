use super::*;
use crate::CalcFlowError;
use datafusion::arrow::{
    array::StringArray,
    buffer::{Buffer, OffsetBuffer, ScalarBuffer},
};

fn allocation_payload(record: RecordBatch) -> RowPayload {
    RowPayload {
        batch: Arc::new(PayloadBatch {
            key: (0, 0),
            record: Arc::new(record),
            body_bytes: 0,
            encoded_charge_bytes: 0,
            encoded: OnceLock::new(),
        }),
        row: 0,
    }
}

fn assert_allocation_workspace_error(error: &CalcFlowError) {
    assert!(
        matches!(
            error,
            CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
                ..
            }
        ),
        "unexpected allocation error: {error:?}"
    );
}

fn assert_registration_rejects_without_retention(row: &RowPayload, limit: usize) {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
    let mut credit = MemoryConsumer::new("allocation-refusal").register(&pool);
    let selected = [vec![0], Vec::new()];
    let mut builder = OutputPlanBuilder::new(1, Some(&selected), &mut credit, "asof").unwrap();
    let initial = credit.size();
    let owners = row
        .batch
        .record
        .columns()
        .iter()
        .map(Arc::strong_count)
        .collect::<Vec<_>>();
    let error = builder
        .push(row.view(), None, &mut credit, "asof")
        .unwrap_err();
    assert_allocation_workspace_error(&error);
    assert_eq!(credit.size(), initial);
    assert_eq!(pool.reserved(), initial);
    assert_eq!(
        row.batch
            .record
            .columns()
            .iter()
            .map(Arc::strong_count)
            .collect::<Vec<_>>(),
        owners,
    );
    drop((builder, credit));
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn source_backing_deduplicates_aliases_and_pays_full_slice_owner() {
    let mut storage = Vec::<i64>::with_capacity(8_192);
    storage.extend([7, 11, 13]);
    let owner = Buffer::from_vec(storage);
    let full = owner.capacity();
    let first = ScalarBuffer::<i64>::new(owner.clone(), 1, 1);
    let second = ScalarBuffer::<i64>::new(owner.clone(), 2, 1);
    assert_eq!(first.inner().data_ptr(), second.inner().data_ptr());
    assert_ne!(first.inner().as_ptr(), second.inner().as_ptr());
    assert!(first.inner().ptr_offset() > 0);
    assert_eq!(first.inner().capacity(), full);
    assert_eq!(second.inner().capacity(), full);
    assert!(full > first.inner().len() + second.inner().len() + (32 << 10));
    let schema = Arc::new(Schema::new(vec![
        Field::new("selected", DataType::Int64, false),
        Field::new("unselected", DataType::Int64, false),
    ]));
    let row = allocation_payload(
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::new(first, None)),
                Arc::new(Int64Array::new(second, None)),
            ],
        )
        .unwrap(),
    );
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(2 * full - 1));
    let mut credit = MemoryConsumer::new("allocation-alias").register(&pool);
    credit.try_grow(17).unwrap();
    let before = credit.size();
    assert_eq!(
        backing::source_bytes(&row.batch.record, &credit, "asof").unwrap(),
        full as u64
    );
    assert_eq!(credit.size(), before);
    assert_eq!(pool.reserved(), before);
    let selected = [vec![0], Vec::new()];
    let mut builder = OutputPlanBuilder::new(2, Some(&selected), &mut credit, "asof").unwrap();
    let initial = credit.size();
    builder.push(row.view(), None, &mut credit, "asof").unwrap();
    let registered = credit.size();
    assert!(registered - initial >= full);
    assert!(registered - initial < 2 * full);
    builder.push(row.view(), None, &mut credit, "asof").unwrap();
    assert_eq!(credit.size(), registered);
    drop((builder, credit));
    assert_eq!(pool.reserved(), 0);
    assert_registration_rejects_without_retention(&row, full - 1);
}

#[test]
fn source_backing_pays_empty_unselected_owner_and_refunds_scratch() {
    let owner = Buffer::from_vec(vec![0_u8; 32 << 10]);
    let empty = owner.slice_with_length(4_096, 0);
    assert!(empty.is_empty());
    assert!(empty.capacity() >= 32 << 10);
    assert_eq!(empty.data_ptr(), owner.data_ptr());
    let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 0]));
    let offset_capacity = offsets.inner().inner().capacity();
    let numbers = Int64Array::from(vec![7]);
    let number_capacity = numbers.values().inner().capacity();
    let expected = owner.capacity() + offset_capacity + number_capacity;
    let strings = StringArray::new(offsets, empty, None);
    assert_eq!(strings.value(0), "");
    let schema = Arc::new(Schema::new(vec![
        Field::new("selected", DataType::Int64, false),
        Field::new("unselected", DataType::Utf8, false),
    ]));
    let row = allocation_payload(
        RecordBatch::try_new(schema, vec![Arc::new(numbers), Arc::new(strings)]).unwrap(),
    );
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(2 * expected));
    let mut credit = MemoryConsumer::new("allocation-empty").register(&pool);
    credit.try_grow(19).unwrap();
    assert_eq!(
        backing::source_bytes(&row.batch.record, &credit, "asof").unwrap(),
        expected as u64
    );
    assert_eq!(credit.size(), 19);
    assert_eq!(pool.reserved(), 19);
    let selected = [vec![0], Vec::new()];
    let mut builder = OutputPlanBuilder::new(1, Some(&selected), &mut credit, "asof").unwrap();
    let initial = credit.size();
    builder.push(row.view(), None, &mut credit, "asof").unwrap();
    assert!(credit.size() - initial >= expected);
    drop((builder, credit));
    assert_eq!(pool.reserved(), 0);
    assert_registration_rejects_without_retention(&row, owner.capacity() - 1);
    let scratch_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(17 + 3 * 1_024 - 1));
    let scratch_credit = MemoryConsumer::new("allocation-scratch-refusal").register(&scratch_pool);
    scratch_credit.try_grow(17).unwrap();
    let error = backing::source_bytes(&row.batch.record, &scratch_credit, "asof").unwrap_err();
    assert_allocation_workspace_error(&error);
    assert_eq!(scratch_credit.size(), 17);
    assert_eq!(scratch_pool.reserved(), 17);
    drop(scratch_credit);
    assert_eq!(scratch_pool.reserved(), 0);
}

use super::*;
use crate::{CancellationToken, JsonMap, StreamJobContext};
use datafusion::{
    arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    },
    execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool},
};

fn rows(count: usize) -> Vec<StoredRow> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int64,
        false,
    )]));
    let record = Arc::new(
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from_iter_values(
                (0..count).map(|index| i64::try_from(index).unwrap()),
            ))],
        )
        .unwrap(),
    );
    let key = Arc::new(Vec::new().into());
    (0..count)
        .map(|index| StoredRow {
            record: RowPayload::Rowed {
                parent: Arc::clone(&record),
                row: index,
            },
            event_time: EventTime::from_micros(i64::try_from(index).unwrap()),
            row_id: index as u64,
            charge: 100,
            encoded_key: Arc::clone(&key),
        })
        .collect()
}

#[test]
fn test_chunk_removal_and_append_preserve_other_references_without_dense_repairs() {
    let mut state: StateRows = rows(8 * CHUNK_ROWS).into();
    let survivor = 7 * CHUNK_ROWS + 3;
    let before = state.payload(survivor).columns().as_ptr();
    super::super::reset_join_work();
    for reference in 0..CHUNK_ROWS {
        state.remove(reference);
    }
    let work = super::super::join_work();
    assert_eq!(work.retained_chunk_visits, CHUNK_ROWS);
    assert_eq!(work.retained_descriptor_views, CHUNK_ROWS);
    assert_eq!(work.native_dense_repairs, 0);
    assert_eq!(state.chunks.len(), 7);
    assert_eq!(state.row_id(survivor), survivor as u64);
    assert_eq!(state.payload(survivor).columns().as_ptr(), before);
    let added = state.append(rows(17));
    assert_eq!(
        added,
        (8 * CHUNK_ROWS..8 * CHUNK_ROWS + 17).collect::<Vec<_>>()
    );
    assert_eq!(state.row_id(survivor), survivor as u64);
    assert_eq!(state.payload(survivor).columns().as_ptr(), before);
}

#[tokio::test]
async fn test_chunk_admission_refusal_and_abandonment_preserve_state_and_refund_credit() {
    let job = StreamJobContext::new(
        1,
        "chunk-state",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "join", None);
    let state: StateRows = rows(2).into();
    let incoming = rows(CHUNK_ROWS);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(8_192));
    let master = MemoryConsumer::new("chunk-control").register(&pool);
    assert!(
        state
            .prepare_append(&incoming, &master, &context, false)
            .await
            .is_err()
    );
    assert_eq!(state.len(), 2);
    assert_eq!(state.identity(1), (EventTime::from_micros(1), 1));
    assert_eq!(pool.reserved(), 0);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_000_000));
    let master = MemoryConsumer::new("chunk-control").register(&pool);
    let append = state
        .prepare_append(&incoming, &master, &context, false)
        .await
        .unwrap();
    assert!(pool.reserved() > 0);
    drop(append);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(state.len(), 2);
}

#[test]
fn test_repeated_owner_replacement_reuses_one_spare_source_slot() {
    for owners in [1, 4] {
        let group = CHUNK_ROWS / owners;
        let initial = (0..owners).flat_map(|_| rows(group)).collect::<Vec<_>>();
        let mut state: StateRows = initial.into();
        assert_eq!(state.chunks[&0].sources.capacity(), owners + 1);
        for _ in 0..8 {
            let replacements = (0..owners).flat_map(|_| rows(group)).collect::<Vec<_>>();
            let work = allocation_counter::measure(|| {
                for (index, row) in replacements.into_iter().enumerate() {
                    state.replace_payload(index, row.record);
                }
            });
            assert_eq!(work.count_total, 0, "source overlap uses prepaid slots");
            assert_eq!(state.chunks[&0].sources.capacity(), owners + 1);
            assert_eq!(state.chunks[&0].sources.iter().flatten().count(), owners);
            assert_eq!(state.len(), CHUNK_ROWS);
        }
    }
}

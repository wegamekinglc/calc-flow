use super::*;
use datafusion::arrow::{datatypes::Schema, record_batch::RecordBatch};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
use std::sync::Arc;

fn row(key: u64, time: i64, row_id: u64) -> StoredRow {
    let record = RecordBatch::new_empty(Arc::new(Schema::empty()));
    StoredRow {
        record: record.clone().into(),
        event_time: EventTime::from_micros(time),
        row_id,
        charge: 0,
        encoded_key: Arc::new(key.to_be_bytes().to_vec().into()),
    }
}

fn funded_index(rows: &[StoredRow]) -> NativeIndex {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let credit = MemoryConsumer::new("stream-join-native").register(&pool);
    credit
        .try_grow(BASE_BYTES + ENTRY_BYTES * rows.len())
        .unwrap();
    NativeIndex::new(rows, credit)
}

fn probe_row_indexes(index: &NativeIndex, key: u64, range: (i64, i64)) -> Vec<(u64, u32)> {
    let bytes = key.to_be_bytes();
    index
        .key_id(&bytes)
        .map(|id| {
            index
                .window_by_id(
                    id,
                    (
                        EventTime::from_micros(range.0),
                        EventTime::from_micros(range.1),
                    ),
                )
                .iter()
                .map(|entry| (entry.row_id, entry.row_index))
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn test_dictionary_eviction_recycling_and_swap_relocation_keep_lookups_exact() {
    let key_a = 0xA_u64;
    let key_b = 0xB_u64;
    let key_c = 0xC_u64;
    let rows = vec![
        row(key_a, 100, 0),
        row(key_b, 200, 1),
        row(key_b, 250, 2),
        row(key_c, 300, 3),
        row(key_a, 150, 4),
    ];
    let mut index = funded_index(&rows);

    // Full-range windows see every entry of each key in (time, row_id) order.
    assert_eq!(
        probe_row_indexes(&index, key_a, (0, 1_000)),
        vec![(0, 0), (4, 4)]
    );
    assert_eq!(
        probe_row_indexes(&index, key_b, (0, 1_000)),
        vec![(1, 1), (2, 2)]
    );
    assert_eq!(probe_row_indexes(&index, key_c, (0, 1_000)), vec![(3, 3)]);
    // Inclusive bounds keep boundary times.
    assert_eq!(probe_row_indexes(&index, key_b, (200, 200)), vec![(1, 1)]);
    assert_eq!(probe_row_indexes(&index, key_b, (201, 249)), vec![]);

    // Evicting key B entirely (its slot leaves the dictionary, id 1 is
    // relocated onto key C) must not corrupt lookups of the survivors.
    index.remove(&rows[1], None);
    index.remove(&rows[2], None);
    assert_eq!(probe_row_indexes(&index, key_b, (0, 1_000)), vec![]);
    assert_eq!(
        probe_row_indexes(&index, key_a, (0, 1_000)),
        vec![(0, 0), (4, 4)]
    );
    assert_eq!(probe_row_indexes(&index, key_c, (0, 1_000)), vec![(3, 3)]);

    // Re-interning the recycled key after later appends keeps every window
    // exact, including entries appended out of (time, row_id) order.
    index.append(
        5,
        &[row(key_b, 400, 5), row(key_a, 50, 6), row(key_b, 350, 7)],
    );
    assert_eq!(
        probe_row_indexes(&index, key_b, (0, 1_000)),
        vec![(7, 7), (5, 5)]
    );
    assert_eq!(probe_row_indexes(&index, key_b, (340, 360)), vec![(7, 7)]);
    assert_eq!(
        probe_row_indexes(&index, key_a, (0, 1_000)),
        vec![(6, 6), (0, 0), (4, 4)]
    );
    assert_eq!(probe_row_indexes(&index, key_c, (0, 1_000)), vec![(3, 3)]);

    // Swap-remove relocation with a moved row keeps the moved row's index.
    let last = row(key_c, 500, 8);
    index.append(8, &[last.clone()]);
    let removed = row(key_a, 100, 0);
    let moved = Some((&last, 0));
    index.remove(&removed, moved);
    assert_eq!(
        probe_row_indexes(&index, key_a, (0, 1_000)),
        vec![(6, 6), (4, 4)]
    );
    assert_eq!(
        probe_row_indexes(&index, key_c, (0, 1_000)),
        vec![(3, 3), (8, 0)]
    );
    assert_eq!(index.entries_len(), 6);
}

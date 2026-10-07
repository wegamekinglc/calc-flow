use super::*;
use datafusion::arrow::{datatypes::Schema, record_batch::RecordBatch};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};

fn rows(count: usize) -> Vec<StoredRow> {
    let record = RecordBatch::new_empty(Arc::new(Schema::empty()));
    (0..count)
        .map(|index| StoredRow {
            record: record.clone().into(),
            event_time: EventTime::from_micros(i64::try_from(index).unwrap()),
            row_id: index as u64,
            charge: 0,
            encoded_key: Arc::new((index as u64).to_be_bytes().to_vec().into()),
        })
        .collect()
}

fn order(count: usize, disordered: bool) -> Vec<usize> {
    let mut values = (0..count).collect::<Vec<_>>();
    if disordered {
        let mut seed = 0x726f_7769_645f_6b65_u64;
        for last in (1..count).rev() {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            values.swap(last, seed_index(seed, last + 1));
        }
    }
    values
}

fn seed_index(seed: u64, length: usize) -> usize {
    usize::try_from(seed % u64::try_from(length).unwrap()).unwrap()
}

fn check_allocation_cut(
    allocation: allocation_counter::AllocationInfo,
    live: &mut i64,
    funded_before: usize,
    index: &NativeIndex,
) {
    assert!(
        *live + i64::try_from(allocation.bytes_max).unwrap()
            <= i64::try_from(funded_before).unwrap()
    );
    *live += allocation.bytes_current;
    assert!(*live >= 0);
    assert!(
        usize::try_from(*live).unwrap() <= index.funded_bytes(),
        "live nodes={live}, funding={}, entries={}",
        index.funded_bytes(),
        index.entries.len()
    );
    assert_eq!(
        index.funded_bytes(),
        BASE_BYTES + ENTRY_BYTES * index.entries.len()
    );
}

fn insert_rows(index: &mut NativeIndex, rows: &[StoredRow], order: &[usize], mut live: i64) -> i64 {
    for (dense, &identity) in order.iter().enumerate() {
        let append = index.reserve(1).unwrap();
        let funded = index.funded_bytes();
        let allocation = allocation_counter::measure(|| {
            index.append(dense, &rows[identity..=identity]);
        });
        append.commit();
        check_allocation_cut(allocation, &mut live, funded, index);
    }
    live
}

fn remove_rows(
    index: &mut NativeIndex,
    rows: &[StoredRow],
    order: &[usize],
    random: bool,
    mut live: i64,
) -> i64 {
    let mut dense = order.to_vec();
    let mut seed = 0x6465_6e73_655f_726d_u64;
    while !dense.is_empty() {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let position = if random {
            seed_index(seed, dense.len())
        } else {
            0
        };
        let identity = dense.swap_remove(position);
        let moved = dense.get(position).map(|&id| (&rows[id], position));
        let funded = index.funded_bytes();
        let allocation = allocation_counter::measure(|| index.remove(&rows[identity], moved));
        check_allocation_cut(allocation, &mut live, funded, index);
    }
    live
}

fn allocated_index(pool: &Arc<dyn MemoryPool>) -> (NativeIndex, i64) {
    let mut index = None;
    let allocation = allocation_counter::measure(|| {
        let credit = MemoryConsumer::new("stream-join-native").register(pool);
        credit.try_grow(BASE_BYTES).unwrap();
        index = Some(NativeIndex::new(&[], credit));
    });
    assert!(allocation.bytes_max <= BASE_BYTES as u64);
    (index.unwrap(), allocation.bytes_current)
}

#[test]
fn test_native_index_live_allocations_remain_funded_through_insert_and_eviction() {
    let rows = rows(4_096);
    for disordered in [false, true] {
        for random in [false, true] {
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
            let (mut index, controls) = allocated_index(&pool);
            let order = order(rows.len(), disordered);
            let live = insert_rows(&mut index, &rows, &order, controls);
            let live = remove_rows(&mut index, &rows, &order, random, live);
            assert_eq!(pool.reserved(), BASE_BYTES);
            let released = allocation_counter::measure(|| drop(index));
            assert_eq!(live + released.bytes_current, 0);
            assert_eq!(pool.reserved(), 0);
        }
    }
}

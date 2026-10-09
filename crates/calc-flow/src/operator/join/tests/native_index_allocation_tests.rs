use super::*;
use datafusion::arrow::{datatypes::Schema, record_batch::RecordBatch};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};

fn rows(count: usize, distinct_keys: usize) -> Vec<StoredRow> {
    let record = RecordBatch::new_empty(Arc::new(Schema::empty()));
    (0..count)
        .map(|index| StoredRow {
            record: record.clone().into(),
            event_time: EventTime::from_micros(i64::try_from(index).unwrap()),
            row_id: index as u64,
            charge: 0,
            encoded_key: Arc::new(
                ((index % distinct_keys) as u64)
                    .to_be_bytes()
                    .to_vec()
                    .into(),
            ),
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
        "live bytes={live}, funding={}, rows={}",
        index.funded_bytes(),
        index.row_count()
    );
    assert_eq!(index.funded_bytes(), index.resident_bytes());
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
        let allocation = allocation_counter::measure(|| {
            index.remove(&rows[identity], moved, |position| {
                &rows[dense[position]].encoded_key
            });
        });
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
fn test_native_index_bulk_construction_peak_and_resident_allocations_remain_funded() {
    for (count, distinct_keys) in [(1, 1), (129, 1), (129, 17), (4_096, 4_096)] {
        let rows = rows(count, distinct_keys);
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let prepaid = NativeIndex::build_charge(count).unwrap();
        let mut index = None;
        let allocated = allocation_counter::measure(|| {
            let credit = MemoryConsumer::new("stream-join-native").register(&pool);
            credit.try_grow(prepaid).unwrap();
            index = Some(NativeIndex::new(&rows, credit));
        });
        let index = index.unwrap();
        assert!(
            allocated.bytes_max <= u64::try_from(prepaid).unwrap(),
            "rows={count}, distinct_keys={distinct_keys}, peak={}, prepaid={prepaid}",
            allocated.bytes_max
        );
        assert!(allocated.bytes_current > 0);
        assert!(
            usize::try_from(allocated.bytes_current).unwrap() <= index.funded_bytes(),
            "rows={count}, distinct_keys={distinct_keys}, live={}, funding={}",
            allocated.bytes_current,
            index.funded_bytes()
        );
        assert_eq!(index.row_count(), count);
        assert_eq!(index.funded_bytes(), index.resident_bytes());
        assert_eq!(pool.reserved(), index.resident_bytes());
        let released = allocation_counter::measure(|| drop(index));
        assert_eq!(allocated.bytes_current + released.bytes_current, 0);
        assert_eq!(pool.reserved(), 0);
    }
}

#[test]
fn test_native_index_live_allocations_remain_funded_through_insert_and_eviction() {
    for distinct_keys in [1, 17, 4_096] {
        let rows = rows(4_096, distinct_keys);
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
}

#[test]
fn test_native_index_retained_capacities_remain_funded_during_empty_run_reuse() {
    let rows = rows(64, 17);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let (mut index, controls) = allocated_index(&pool);
    let mut dense = order(rows.len(), true);
    let mut live = insert_rows(&mut index, &rows, &dense, controls);
    let removed = (0..rows.len()).step_by(17).collect::<Vec<_>>();
    for &identity in &removed {
        let position = dense.iter().position(|&id| id == identity).unwrap();
        dense.swap_remove(position);
        let moved = dense.get(position).map(|&id| (&rows[id], position));
        let funded = index.funded_bytes();
        let allocation = allocation_counter::measure(|| {
            index.remove(&rows[identity], moved, |position| {
                &rows[dense[position]].encoded_key
            });
        });
        check_allocation_cut(allocation, &mut live, funded, &index);
        assert_eq!(index.row_count(), dense.len());
    }
    assert!(index.funded_bytes() > BASE_BYTES);
    for identity in removed {
        let append = index.reserve(1).unwrap();
        let funded = index.funded_bytes();
        let allocation = allocation_counter::measure(|| {
            index.append(dense.len(), &rows[identity..=identity]);
        });
        append.commit();
        dense.push(identity);
        check_allocation_cut(allocation, &mut live, funded, &index);
        assert_eq!(index.row_count(), dense.len());
    }
    let released = allocation_counter::measure(|| drop(index));
    assert_eq!(live + released.bytes_current, 0);
    assert_eq!(pool.reserved(), 0);
}

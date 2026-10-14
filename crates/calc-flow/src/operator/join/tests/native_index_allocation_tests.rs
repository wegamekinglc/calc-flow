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
    let prepaid = NativeIndex::build_charge(0).unwrap();
    let mut index = None;
    let allocation = allocation_counter::measure(|| {
        let credit = MemoryConsumer::new("stream-join-native").register(pool);
        credit.try_grow(prepaid).unwrap();
        index = Some(NativeIndex::new(&[], credit));
    });
    assert!(allocation.bytes_max <= BASE_BYTES as u64);
    (index.unwrap(), allocation.bytes_current)
}

fn assert_native_index_bulk_construction_peak_and_resident_allocations_remain_funded() {
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
    assert_native_index_bulk_construction_peak_and_resident_allocations_remain_funded();
    assert_native_index_live_allocations_remain_funded_through_insert_and_eviction();
    assert_native_index_retained_capacities_remain_funded_during_empty_run_reuse();
}

fn assert_native_index_live_allocations_remain_funded_through_insert_and_eviction() {
    for distinct_keys in [1, 17, 4_096] {
        let rows = rows(4_096, distinct_keys);
        for disordered in [false, true] {
            for random in [false, true] {
                let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
                let (mut index, controls) = allocated_index(&pool);
                let order = order(rows.len(), disordered);
                let live = insert_rows(&mut index, &rows, &order, controls);
                let live = remove_rows(&mut index, &rows, &order, random, live);
                assert_eq!(pool.reserved(), NativeIndex::build_charge(0).unwrap());
                let released = allocation_counter::measure(|| drop(index));
                assert_eq!(live + released.bytes_current, 0);
                assert_eq!(pool.reserved(), 0);
            }
        }
    }
}

fn assert_native_index_retained_capacities_remain_funded_during_empty_run_reuse() {
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
    assert!(index.funded_bytes() > NativeIndex::build_charge(0).unwrap());
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

fn string_probe_rows(large: bool) -> (Vec<AdmittedRow>, Vec<std::sync::Weak<RecordBatch>>) {
    use datafusion::arrow::{
        array::{ArrayRef, LargeStringArray, StringArray},
        datatypes::Field,
    };

    let values = [
        String::new(),
        "é\0値".into(),
        "prefix".into(),
        "prefix-more".into(),
        "x".repeat(64),
        "x".repeat(65),
    ];
    let parents = [false, true].map(|reverse| {
        let mut padded = vec!["unused-before".to_owned()];
        padded.extend(if reverse {
            values.iter().rev().cloned().collect::<Vec<_>>()
        } else {
            values.to_vec()
        });
        padded.push("unused-after".into());
        let array: ArrayRef = if large {
            Arc::new(LargeStringArray::from(padded))
        } else {
            Arc::new(StringArray::from(padded))
        };
        let schema = Arc::new(Schema::new(vec![Field::new(
            "key",
            array.data_type().clone(),
            false,
        )]));
        Arc::new(
            RecordBatch::try_new(schema, vec![array])
                .unwrap()
                .slice(1, 6),
        )
    });
    let weak = parents.iter().map(Arc::downgrade).collect();
    let positions = [
        (0, 4),
        (1, 1),
        (0, 0),
        (0, 3),
        (1, 5),
        (1, 2),
        (0, 1),
        (1, 4),
        (0, 5),
        (1, 0),
        (0, 2),
        (1, 3),
        (0, 4),
        (1, 5),
    ];
    let rows = positions
        .into_iter()
        .enumerate()
        .map(|(position, (parent, row))| AdmittedRow {
            record: super::super::columnar::RowPayload::Rowed {
                parent: Arc::clone(&parents[parent]),
                row,
            },
            event_time: EventTime::from_micros(0),
            row_id: position as u64,
            retain: true,
        })
        .collect();
    (rows, weak)
}

fn canonical_probe_frames(rows: &[AdmittedRow]) -> Vec<Vec<u8>> {
    rows.iter()
        .map(|row| {
            super::super::encode_join_key_columns_v1(
                row.record.columns(),
                row.record.offset(),
                &[0],
            )
            .unwrap()
        })
        .collect()
}

fn assert_probe_frame_identity(keys: &NativeKeys, expected: &[Vec<u8>]) {
    assert_eq!(
        keys.row_keys().map(|key| key.to_vec()).collect::<Vec<_>>(),
        expected,
    );
    for (left, first) in expected.iter().enumerate() {
        for (right, second) in expected.iter().enumerate() {
            assert_eq!(keys.id(left) == keys.id(right), first == second);
            assert_eq!(
                Arc::ptr_eq(keys.key(left), keys.key(right)),
                first == second
            );
        }
    }
}

#[test]
fn test_string_probe_interns_canonical_frames_once_per_distinct_value() {
    for large in [false, true] {
        let (rows, parents) = string_probe_rows(large);
        let expected = canonical_probe_frames(&rows);
        let distinct = expected
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        assert_eq!(distinct, 6);
        assert!(distinct < rows.len());
        let layout = ArenaLayout::measure(&rows, &[0]).unwrap().unwrap();
        assert!(layout.fits(rows.len(), 1));
        assert_eq!(layout.charge, probe_key_charge(&rows, &[0]).unwrap());
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(layout.charge));
        let credit = MemoryConsumer::new("string-interner").register(&pool);
        credit.try_grow(layout.charge).unwrap();
        super::super::reset_join_work();
        let keys = intern_probe_keys(
            &rows,
            &[0],
            Arc::new(credit),
            "string-interner",
            Some(layout),
        )
        .unwrap();
        let work = super::super::join_work();
        assert_probe_frame_identity(&keys, &expected);
        drop(rows);
        assert!(parents.iter().all(|parent| parent.upgrade().is_none()));
        assert_eq!(work.key_encodings, distinct);
        assert_eq!(
            work.arena_frames, distinct,
            "duplicate values must not be framed"
        );
        drop(keys);
        assert_eq!(pool.reserved(), 0);
    }
}

#[test]
fn test_string_probe_forced_hash_collisions_keep_tagged_canonical_identity() {
    let (mut rows, mut parents) = string_probe_rows(false);
    let (mut large_rows, large_parents) = string_probe_rows(true);
    rows.append(&mut large_rows);
    parents.extend(large_parents);
    let expected = canonical_probe_frames(&rows);
    let distinct = expected
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    assert_eq!(distinct, 12);
    let layout = ArenaLayout::measure(&rows, &[0]).unwrap().unwrap();
    assert!(layout.fits(rows.len(), 1));
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(layout.charge));
    let credit = MemoryConsumer::new("tagged-string-collisions").register(&pool);
    credit.try_grow(layout.charge).unwrap();
    super::super::reset_join_work();
    let keys = colliding_probe_keys(&rows, &[0], credit).unwrap();
    let work = super::super::join_work();
    assert_probe_frame_identity(&keys, &expected);
    assert_eq!(&expected[0][1..], &expected[14][1..]);
    assert_ne!(expected[0][0], expected[14][0]);
    assert!(!Arc::ptr_eq(keys.key(0), keys.key(14)));
    drop(rows);
    assert!(parents.iter().all(|parent| parent.upgrade().is_none()));
    assert_eq!(work.key_encodings, distinct);
    assert_eq!(
        work.arena_frames, distinct,
        "collisions must use the string miss path"
    );
    drop(keys);
    assert_eq!(pool.reserved(), 0);
}

fn probe_array_rows(
    values: datafusion::arrow::array::ArrayRef,
    offsets: &[usize],
) -> Vec<AdmittedRow> {
    let schema = Arc::new(Schema::new(vec![datafusion::arrow::datatypes::Field::new(
        "key",
        values.data_type().clone(),
        true,
    )]));
    let parent = Arc::new(RecordBatch::try_new(schema, vec![values]).unwrap());
    offsets
        .iter()
        .enumerate()
        .map(|(position, &row)| AdmittedRow {
            record: super::super::columnar::RowPayload::Rowed {
                parent: Arc::clone(&parent),
                row,
            },
            event_time: EventTime::from_micros(0),
            row_id: position as u64,
            retain: true,
        })
        .collect()
}

fn assert_probe_fallback(rows: &Vec<AdmittedRow>, layout: Option<ArenaLayout>) {
    let expected = canonical_probe_frames(rows);
    let charge = probe_key_charge(rows, &[0]).unwrap();
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(charge));
    let credit = MemoryConsumer::new("string-fallback").register(&pool);
    credit.try_grow(charge).unwrap();
    super::super::reset_join_work();
    let keys = intern_probe_keys(rows, &[0], Arc::new(credit), "string-fallback", layout).unwrap();
    let work = super::super::join_work();
    assert_probe_frame_identity(&keys, &expected);
    assert_eq!(work.arena_frames, 0);
    assert_eq!(work.borrowed_key_hashes, rows.len());
    drop(keys);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_string_probe_keeps_empty_null_parent_and_nonfitting_fallbacks() {
    use datafusion::arrow::array::{StringArray, StringViewArray};

    let empty = ArenaLayout::measure(&[], &[0]).unwrap().unwrap();
    assert!(!empty.fits(0, 1));
    assert_probe_fallback(&Vec::new(), Some(empty));

    let null_parent = probe_array_rows(
        Arc::new(StringArray::from(vec![Some("kept"), None])),
        &[0, 0],
    );
    assert!(ArenaLayout::measure(&null_parent, &[0]).unwrap().is_none());
    assert_probe_fallback(&null_parent, None);

    let tiny = probe_array_rows(Arc::new(StringArray::from(vec![""])), &[0]);
    let layout = ArenaLayout::measure(&tiny, &[0]).unwrap().unwrap();
    assert_eq!(layout.charge, probe_key_charge(&tiny, &[0]).unwrap());
    assert!(!layout.fits(1, 1));
    assert_probe_fallback(&tiny, Some(layout));

    let unsupported = probe_array_rows(Arc::new(StringViewArray::from(vec!["kept"])), &[0]);
    assert!(ArenaLayout::measure(&unsupported, &[0]).unwrap().is_none());
    assert!(!native_type(&DataType::Utf8View));
}

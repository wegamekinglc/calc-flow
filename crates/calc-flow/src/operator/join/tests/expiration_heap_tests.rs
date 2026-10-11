use super::*;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};

#[path = "expiration_admission_tests.rs"]
mod admission;

fn rows(count: usize) -> Vec<StoredRow> {
    let record = RecordBatch::new_empty(Arc::new(Schema::empty()));
    (0..count)
        .map(|index| StoredRow {
            record: record.clone().into(),
            event_time: EventTime::from_micros(i64::try_from((index * 73) % count).unwrap()),
            row_id: index as u64,
            charge: 0,
            encoded_key: Arc::new(Vec::new().into()),
        })
        .collect()
}

#[tokio::test]
async fn test_expiration_heap_matches_tree_through_dense_removal_and_append() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    for restored in [false, true] {
        let mut retained = if restored { rows(65) } else { Vec::new() };
        let mut index = ExpirationIndex::restored(&retained).unwrap();
        let mut oracle = retained
            .iter()
            .enumerate()
            .map(|(ordinal, row)| ((row.event_time, row.row_id), ordinal as u128))
            .collect::<BTreeMap<_, _>>();
        let mut ordinal = retained.len() as u128;
        let mut next_id = retained.len() as u64;
        for round in 0..8 {
            let mut incoming = rows(17 + round * 3);
            for row in &mut incoming {
                row.row_id = next_id;
                next_id += 1;
                row.event_time = EventTime::from_micros(row.event_time.as_micros() % 7 - 3);
                oracle.insert((row.event_time, row.row_id), ordinal);
                ordinal += 1;
            }
            let plan = index
                .prepare_append(incoming.len(), &context)
                .await
                .unwrap();
            let allocations =
                allocation_counter::measure(|| index.append(&retained, &incoming, plan));
            assert_eq!(allocations.count_total, 0, "heap commit cannot allocate");
            retained.extend(incoming);
            let evict = if round == 7 { retained.len() } else { 11 };
            for _ in 0..evict {
                let ((time, id), expected_ordinal) = oracle.pop_first().unwrap();
                assert_eq!(index.first_time(&retained), Some(time));
                let (dense, actual_ordinal) = index.pop(&retained).unwrap();
                let row = retained.swap_remove(dense);
                assert_eq!(
                    (row.event_time, row.row_id, actual_ordinal),
                    (time, id, expected_ordinal)
                );
            }
            let mut expected = oracle
                .iter()
                .map(|((_, id), ordinal)| (*ordinal, *id))
                .collect::<Vec<_>>();
            expected.sort_unstable();
            assert_eq!(
                index
                    .identities(&retained)
                    .into_iter()
                    .map(|(id, _, _)| id)
                    .collect::<Vec<_>>(),
                expected.into_iter().map(|(_, id)| id).collect::<Vec<_>>()
            );
        }
        assert!(index.pop(&retained).is_none());
    }
}

#[tokio::test]
async fn test_restored_heap_keeps_funding_through_empty_reuse_and_growth() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    for count in [0, 17] {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_000_000));
        let credit = MemoryConsumer::new("expiration-test").register(&pool);
        let mut retained = rows(count);
        let mut index = ExpirationIndex::restored_funded(&retained, &credit, &|| Ok(())).unwrap();
        let initial = pool.reserved();
        assert!(initial >= count * 32);
        assert!(initial > 0, "empty V2 side retains funded origin");
        while let Some((dense, _)) = index.pop(&retained) {
            retained.swap_remove(dense);
        }
        assert_eq!(
            pool.reserved(),
            initial,
            "emptying does not destroy capacities"
        );
        let blocker = credit.new_empty();
        blocker.try_grow(1_000_000 - pool.reserved()).unwrap();
        assert!(index.prepare_append(count + 1, &context).await.is_err());
        assert_eq!(pool.reserved(), 1_000_000);
        assert!(index.identities(&retained).is_empty());
        drop(blocker);
        assert_eq!(pool.reserved(), initial);
        let incoming = rows(count + 1);
        let dropped = index
            .prepare_append(incoming.len(), &context)
            .await
            .unwrap();
        assert!(pool.reserved() > initial);
        drop(dropped);
        assert_eq!(
            pool.reserved(),
            initial,
            "uninstalled growth releases only new storage"
        );
        let plan = index
            .prepare_append(incoming.len(), &context)
            .await
            .unwrap();
        let during = pool.reserved();
        index.append(&retained, &incoming, plan);
        retained.extend(incoming);
        assert_eq!(
            pool.reserved(),
            during - initial,
            "replacement retires the old owner"
        );
        let resident = pool.reserved();
        let no_growth = index.prepare_append(0, &context).await.unwrap();
        index.append(&retained, &[], no_growth);
        assert_eq!(pool.reserved(), resident);
        drop(index);
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test]
async fn test_cancelled_heap_growth_refunds_replacement_and_preserves_original() {
    use std::{future::Future, task::Poll};
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4_000_000));
    let credit = MemoryConsumer::new("expiration-cancel").register(&pool);
    let retained = rows(16_384);
    let index = ExpirationIndex::restored_funded(&retained, &credit, &|| Ok(())).unwrap();
    let resident = pool.reserved();
    let identities = index.identities(&retained);
    let mut future = Box::pin(index.prepare_append(1, &context));
    let mut interrupted = false;
    let result = std::future::poll_fn(|cx| match future.as_mut().poll(cx) {
        Poll::Ready(result) => Poll::Ready(result),
        Poll::Pending => {
            assert!(pool.reserved() > resident);
            interrupted = true;
            job.cancellation().cancel();
            Poll::Pending
        }
    })
    .await;
    assert!(interrupted, "large replacement copy must cooperate");
    assert!(result.is_err());
    drop(future);
    assert_eq!(pool.reserved(), resident);
    assert!(index.identities(&retained) == identities);
    drop(index);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_restored_expiration_index_has_bounded_construction_and_resident_storage() {
    let rows = rows(1_024);
    let mut index = None;
    let allocated = allocation_counter::measure(|| {
        index = Some(ExpirationIndex::restored(&rows).unwrap());
    });
    let bound = rows.len() * (2 * size_of::<usize>() + size_of::<u128>());
    assert!(
        allocated.bytes_max <= bound as u64,
        "expiration construction peak={} exceeds exact array bound={bound}",
        allocated.bytes_max
    );
    assert_eq!(allocated.bytes_current, i64::try_from(bound).unwrap());
    let released = allocation_counter::measure(|| drop(index));
    assert_eq!(released.bytes_current, -allocated.bytes_current);
}

#[tokio::test]
async fn test_expiration_bulk_clear_keeps_funded_capacity_and_retention_ordinals() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_000_000));
    let credit = MemoryConsumer::new("expiration-clear").register(&pool);
    let mut retained = rows(4);
    let mut index = ExpirationIndex::restored_funded(&retained, &credit, &|| Ok(())).unwrap();
    assert_eq!(index.last_time(), Some(EventTime::from_micros(3)));
    let original = pool.reserved();
    index.clear();
    retained.clear();
    assert_eq!(index.last_time(), None);
    assert_eq!(
        pool.reserved(),
        original,
        "empty storage keeps its funding origin and capacity"
    );
    let blocker = credit.new_empty();
    blocker.try_grow(1_000_000 - original).unwrap();
    assert!(index.prepare_append(5, &context).await.is_err());
    assert_eq!(pool.reserved(), 1_000_000);
    drop(blocker);
    let mut incoming = rows(2);
    incoming[0].event_time = EventTime::from_micros(-5);
    incoming[1].event_time = EventTime::from_micros(-7);
    let plan = index.prepare_append(2, &context).await.unwrap();
    index.append(&retained, &incoming, plan);
    retained.extend(incoming);
    assert_eq!(pool.reserved(), original);
    assert_eq!(index.last_time(), Some(EventTime::from_micros(-5)));
    let (dense, ordinal) = index.pop(&retained).unwrap();
    assert_eq!(ordinal, 5, "clear preserves the next retention ordinal");
    retained.swap_remove(dense);
    assert_eq!(index.last_time(), Some(EventTime::from_micros(-5)));
    let (dense, ordinal) = index.pop(&retained).unwrap();
    assert_eq!(ordinal, 4);
    retained.swap_remove(dense);
    assert_eq!(index.last_time(), None);
    assert_eq!(pool.reserved(), original);
    drop(index);
    assert_eq!(pool.reserved(), 0);
}

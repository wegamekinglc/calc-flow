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
async fn test_expiration_runs_match_tree_through_stable_removal_and_append() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    for restored in [false, true] {
        let mut retained: StateRows = if restored { rows(65) } else { Vec::new() }.into();
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
            let references = retained.append_indices(incoming.len());
            let plan = index
                .prepare_append(&incoming, &references, &context, false)
                .await
                .unwrap();
            let allocations = allocation_counter::measure(|| index.append(plan));
            assert!(
                allocations.count_total <= 2,
                "commit allocates only the run-map node"
            );
            assert_eq!(retained.append(incoming), references);
            let evict = if round == 7 { retained.len() } else { 11 };
            for _ in 0..evict {
                let ((time, id), expected_ordinal) = oracle.pop_first().unwrap();
                assert_eq!(index.first_time(&retained), Some(time));
                let (reference, actual_ordinal) = index.pop(&retained).unwrap();
                let row = retained.remove(reference);
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
async fn test_restored_runs_refund_entries_and_keep_funding_origin_after_emptying() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    for count in [0, 17] {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_000_000));
        let credit = MemoryConsumer::new("expiration-test").register(&pool);
        let mut retained: StateRows = rows(count).into();
        let mut index = ExpirationIndex::restored_funded(&retained, &credit, &|| Ok(())).unwrap();
        while let Some((reference, _)) = index.pop(&retained) {
            retained.remove(reference);
        }
        let resident = pool.reserved();
        assert_eq!(resident, index.funded_bytes());
        assert!(resident > 0, "the head container keeps its funded origin");
        let incoming = rows(count + 1);
        let references = retained.append_indices(incoming.len());
        let blocker = credit.new_empty();
        blocker.try_grow(1_000_000 - resident).unwrap();
        assert!(
            index
                .prepare_append(&incoming, &references, &context, false)
                .await
                .is_err()
        );
        assert_eq!(pool.reserved(), 1_000_000);
        assert!(index.identities(&retained).is_empty());
        drop(blocker);
        let dropped = index
            .prepare_append(&incoming, &references, &context, false)
            .await
            .unwrap();
        assert!(pool.reserved() > resident);
        drop(dropped);
        assert_eq!(
            pool.reserved(),
            resident,
            "uninstalled runs refund all new storage"
        );
        let plan = index
            .prepare_append(&incoming, &references, &context, false)
            .await
            .unwrap();
        index.append(plan);
        retained.append(incoming);
        assert_eq!(pool.reserved(), index.funded_bytes());
        let before = pool.reserved();
        let empty = index
            .prepare_append(&[], &[], &context, false)
            .await
            .unwrap();
        index.append(empty);
        assert_eq!(pool.reserved(), before);
        drop(index);
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test]
async fn test_cancelled_run_preparation_refunds_new_storage_and_preserves_original() {
    use std::{future::Future, task::Poll};
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4_000_000));
    let credit = MemoryConsumer::new("expiration-cancel").register(&pool);
    let retained: StateRows = rows(16_384).into();
    let mut index = ExpirationIndex::restored_funded(&retained, &credit, &|| Ok(())).unwrap();
    let resident = pool.reserved();
    let identities = index.identities(&retained);
    let incoming = rows(16_384);
    let references = retained.append_indices(incoming.len());
    let mut future = Box::pin(index.prepare_append(&incoming, &references, &context, false));
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
    assert!(interrupted, "large run preparation must cooperate");
    assert!(result.is_err());
    drop(future);
    assert_eq!(pool.reserved(), resident);
    assert!(index.identities(&retained) == identities);
    drop(index);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_restored_run_construction_has_bounded_allocation_and_releases_storage() {
    let retained = rows(1_024);
    let mut index = None;
    let allocated = allocation_counter::measure(|| {
        index = Some(ExpirationIndex::restored(&retained).unwrap());
    });
    let bound = retained.len() * 32 + 4_096;
    assert!(
        allocated.bytes_max <= bound as u64,
        "run construction peak={} exceeds {bound}",
        allocated.bytes_max
    );
    assert!(allocated.bytes_current > 0);
    let released = allocation_counter::measure(|| drop(index));
    assert_eq!(released.bytes_current, -allocated.bytes_current);
}

#[tokio::test]
async fn test_expiration_clear_preserves_ordinals_and_funded_reuse() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_000_000));
    let credit = MemoryConsumer::new("expiration-clear").register(&pool);
    let mut retained: StateRows = rows(4).into();
    let mut index = ExpirationIndex::restored_funded(&retained, &credit, &|| Ok(())).unwrap();
    index.clear();
    retained.clear();
    assert_eq!(index.last_time(), None);
    assert_eq!(pool.reserved(), index.funded_bytes());
    let incoming = rows(2);
    let references = retained.append_indices(incoming.len());
    let plan = index
        .prepare_append(&incoming, &references, &context, false)
        .await
        .unwrap();
    index.append(plan);
    retained.append(incoming);
    let mut ordinals = Vec::new();
    while let Some((reference, ordinal)) = index.pop(&retained) {
        retained.remove(reference);
        ordinals.push(ordinal);
    }
    ordinals.sort_unstable();
    assert_eq!(
        ordinals,
        [4, 5],
        "clear preserves the next retention ordinal"
    );
    assert_eq!(index.last_time(), None);
    assert_eq!(pool.reserved(), index.funded_bytes());
    drop(index);
    assert_eq!(pool.reserved(), 0);
}

/// The bulk cut must equal repeated per-row pops on random out-of-order runs
/// (spec FR10/AC5): the retained set, its admission order, and the run heads
/// stay identical to a per-row pop oracle.
#[tokio::test]
async fn test_bulk_cuts_match_per_row_pops_on_random_out_of_order_runs() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    for seed in 0_u64..32 {
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005) | 1;
        let mut draw = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        let mut retained: StateRows = Vec::new().into();
        let mut index = ExpirationIndex::restored(&retained).unwrap();
        // The per-row oracle: repeated global-head pops are exactly repeated
        // removals of the `(event_time, row_id)` minimum.
        let mut oracle: BTreeMap<(EventTime, u64), u128> = BTreeMap::new();
        let mut ordinal = 0_u128;
        let mut next_id = 0_u64;
        let expired = move |time: EventTime| time.as_micros() < 20;
        for round in 0..8 {
            let count = 1 + draw() as usize % 40;
            let mut incoming = rows(count);
            for row in &mut incoming {
                row.row_id = next_id;
                next_id += 1;
                // Out-of-order times inside the run (spec FR10).
                row.event_time = EventTime::from_micros(i64::from(draw() % 61) - 30);
                oracle.insert((row.event_time, row.row_id), ordinal);
                ordinal += 1;
            }
            let references = retained.append_indices(incoming.len());
            let plan = index
                .prepare_append(&incoming, &references, &context, false)
                .await
                .unwrap();
            index.append(plan);
            assert_eq!(retained.append(incoming), references);
            // Bulk: plan read-only, then replay the cuts.
            let cuts = index.expired_cuts(&retained, &expired);
            let mut removed = Vec::new();
            for &(id, cut) in &cuts {
                for reference in index.run_entries(id, cut) {
                    removed.push((reference, retained.identity(reference)));
                }
            }
            index.apply_cuts(&cuts, &retained, &expired);
            for (reference, identity) in removed {
                retained.remove(reference);
                assert!(
                    oracle.remove(&identity).is_some(),
                    "cut identity {identity:?}"
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
                expected.into_iter().map(|(_, id)| id).collect::<Vec<_>>(),
                "seed {seed} round {round} retained set differs"
            );
        }
    }
}

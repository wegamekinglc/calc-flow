use super::*;
use crate::operator::asof::output_plan::take_position_promotions;

fn owner(side: u8, id: u64, record: Arc<RecordBatch>) -> Arc<state::PayloadBatch> {
    Arc::new(state::PayloadBatch {
        key: (side, id),
        record,
        encoded: OnceLock::new(),
        encoded_charge_bytes: 0,
        body_bytes: 0,
    })
}

fn matching_right_owners(operator: &mut StreamAsofJoinOperator) {
    let rows = operator
        .state
        .left
        .iter()
        .map(|((time, key, sequence), row)| ((*time, key.clone(), sequence.into_owned()), row))
        .collect::<Vec<_>>();
    let record = operator
        .state
        .batches
        .values()
        .next()
        .unwrap()
        .0
        .record
        .clone();
    let owners = [owner(1, 0, record.clone()), owner(1, 1, record)];
    let key = rows[0].0.1.clone();
    let split = rows.len() / 2;
    let mut bucket = state::RightBucket::new();
    for (index, ((time, _, sequence), left)) in rows.into_iter().enumerate() {
        let right = operator.state.attach(&state::RowPayload {
            batch: owners[usize::from(index >= split)].clone(),
            row: left.row as usize,
        });
        bucket.insert((time, sequence), Some(right));
    }
    operator.state.right.insert(key, bucket);
    operator.state.rebuild_encoding_owners();
}

#[tokio::test]
async fn owner_matching_multiple_right_payloads_never_promotes_a_compact_prefix() {
    let count = 128;
    let mut operator = operator(count);
    matching_right_owners(&mut operator);
    assert_eq!(operator.state.batches.len(), 3);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(16 << 20));
    let mut credit = MemoryConsumer::new("owner-position-mode").register(&pool);
    let job = StreamJobContext::new(
        925,
        "owner-position-mode",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "asof", None);
    take_position_promotions();
    let result = match_output_prefix(&operator, count, &context, None, &mut credit)
        .await
        .unwrap();
    assert_eq!(result.plan.len, count);
    assert_eq!(result.plan.matched, count as u64);
    assert_eq!(
        take_position_promotions(),
        0,
        "Tokio owner matching must select legacy tuples before any prefix conversion"
    );
    let expected = (0..count)
        .map(|row| (usize::from(row >= count / 2) + 1, row))
        .collect::<Vec<_>>();
    assert_eq!(result.plan.right.positions, expected);
    drop((result, credit));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn owner_compact_eligibility_checks_only_the_bounded_payload_metadata() {
    let template = operator(1);
    let record = template
        .state
        .batches
        .values()
        .next()
        .unwrap()
        .0
        .record
        .clone();
    let cases: &[(&[u8], bool)] = &[
        (&[], true),
        (&[0], true),
        (&[1], true),
        (&[0, 0], true),
        (&[0, 1], true),
        (&[1, 1], false),
        (&[0, 0, 0], false),
        (&[0, 1, 1], false),
    ];
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(16 << 20));
    let job = StreamJobContext::new(
        926,
        "owner-position-proof",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "asof", None);
    for &(sides, compact) in cases {
        let mut operator = operator(0);
        for (index, &side) in sides.iter().enumerate() {
            operator.state.attach(&state::RowPayload {
                batch: owner(side, index as u64, record.clone()),
                row: 0,
            });
        }
        assert_eq!(operator.state.batches.len(), sides.len());
        let mut credit = MemoryConsumer::new("owner-position-proof").register(&pool);
        let result = match_output_prefix(&operator, 0, &context, None, &mut credit)
            .await
            .unwrap();
        assert_eq!(
            result.plan.right.positions.view().prefix.is_some(),
            compact,
            "live owner sides {sides:?}"
        );
        drop((result, credit));
    }
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn owner_compact_eligibility_declines_sparse_historical_payload_backing() {
    let template = operator(1);
    let record = template
        .state
        .batches
        .values()
        .next()
        .unwrap()
        .0
        .record
        .clone();
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(16 << 20));
    let job = StreamJobContext::new(
        927,
        "sparse-owner-position-proof",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "asof", None);
    for count in 0..=2 {
        let mut operator = operator(0);
        operator.state.batches = state::PayloadPool::with_backing_buckets(64, 64);
        for index in 0..count {
            operator.state.attach(&state::RowPayload {
                batch: owner(u8::try_from(index).unwrap(), index as u64, record.clone()),
                row: 0,
            });
        }
        assert_eq!(operator.state.batches.len(), count);
        assert!(operator.state.batches.backing_buckets().1 >= 64);
        let mut credit = MemoryConsumer::new("sparse-owner-position-proof").register(&pool);
        let result = match_output_prefix(&operator, 0, &context, None, &mut credit)
            .await
            .unwrap();
        assert!(
            result.plan.right.positions.view().prefix.is_none(),
            "sparse historical backing must fall back before scanning metadata"
        );
        drop((result, credit));
    }
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    assert_eq!(pool.reserved(), 0);
}

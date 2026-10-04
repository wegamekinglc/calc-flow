use super::{fixture, indexed_input};
use crate::{
    AsofStateLimits, Batch, BatchMetadata, CancellationToken, EdgeCollector, IngressProgress,
    IngressProgressSnapshot, IngressState, JsonMap, OperatorMetadata, StreamAsofJoinOperator,
    StreamAsofJoinSpec, StreamJobContext, StreamOperator, StreamOperatorContext,
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

fn with_cursor(batch: &Batch, side: &str) -> Batch {
    batch.with_source_cursor(Some(Arc::new(
        crate::Cursor::new(side, vec![1], JsonMap::new()).unwrap(),
    )))
}

#[tokio::test]
async fn replay_recording_pressure_preserves_admission_success() {
    let (template, _) = fixture();
    let schema = template.schemas[0].clone();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let progress = IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, None),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, None),
        ),
    ]));
    let context = StreamOperatorContext::with_ingress_progress(&job, "asof", None, progress);
    let mut pools = Vec::new();
    for limit in [32 * 1024, 64 * 1024, 128 * 1024] {
        for count in [1, 4, 16, 64, 256] {
            let spec = StreamAsofJoinSpec::new(
                template.spec.left().clone(),
                template.spec.right().clone(),
                Duration::from_micros(10),
                AsofStateLimits::new(10_000, limit).unwrap(),
            )
            .unwrap();
            let mut plain =
                StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec.clone())
                    .unwrap();
            let mut recorded =
                StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
            recorded.replay = Some(Box::new(recorded.new_replay_log().unwrap()));
            recorded.status.state_bytes = recorded.current_inventory(None).unwrap().bytes;
            let rows = (0..count)
                .map(|sequence| ("A", 100, i64::from(sequence)))
                .collect::<Vec<_>>();
            let batch = Batch::table(
                vec![indexed_input(&schema, &rows)],
                BatchMetadata::default(),
            )
            .unwrap();
            for side in ["left", "right"] {
                let baseline = plain
                    .process_data(
                        side,
                        batch.clone(),
                        &context,
                        &mut EdgeCollector::new(plain.output_ports().to_vec()),
                    )
                    .await;
                let candidate = recorded
                    .process_data(
                        side,
                        with_cursor(&batch, side),
                        &context,
                        &mut EdgeCollector::new(recorded.output_ports().to_vec()),
                    )
                    .await;
                assert_eq!(
                    candidate.is_ok(),
                    baseline.is_ok(),
                    "limit={limit}, rows={count}, side={side}: baseline={baseline:?}, candidate={candidate:?}"
                );
                if baseline.is_err() {
                    break;
                }
                if limit == 128 * 1024 && count == 1 && side == "left" {
                    assert!(recorded.replay.is_some());
                }
            }
            assert_status(&plain, &recorded);
            let baseline = plain
                .on_end(
                    &context,
                    &mut EdgeCollector::new(plain.output_ports().to_vec()),
                )
                .await;
            let candidate = recorded
                .on_end(
                    &context,
                    &mut EdgeCollector::new(recorded.output_ports().to_vec()),
                )
                .await;
            if baseline.is_ok() {
                assert!(
                    candidate.is_ok(),
                    "limit={limit}, rows={count}, finalization: {candidate:?}"
                );
                assert_status(&plain, &recorded);
            }
            plain.reset().unwrap();
            recorded.reset().unwrap();
            pools.extend([plain.runtime.pool.clone(), recorded.runtime.pool.clone()]);
        }
    }
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    for pool in pools {
        assert_eq!(pool.reserved(), 0);
    }
}

fn assert_status(plain: &StreamAsofJoinOperator, recorded: &StreamAsofJoinOperator) {
    let mut actual = recorded.status();
    actual.state_bytes = plain.status.state_bytes;
    assert_eq!(actual, plain.status);
}

#[tokio::test]
async fn replay_anchor_preparation_cancellation_keeps_log_owned() {
    let (mut operator, _) = fixture();
    operator.replay = Some(Box::new(operator.new_replay_log().unwrap()));
    operator.status.state_bytes = operator.current_inventory(None).unwrap().bytes;
    let status = operator.status();
    let pool = operator.runtime.pool.clone();
    let reserved = pool.reserved();
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, cancellation);
    let context = StreamOperatorContext::new(&job, "asof", None);
    assert!(operator.prepare_replay_anchor(&context).await.is_err());
    assert_eq!(operator.status(), status);
    assert!(operator.replay.is_some());
    assert_eq!(pool.reserved(), reserved);
    drop(operator);
    assert_eq!(pool.reserved(), 0);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}

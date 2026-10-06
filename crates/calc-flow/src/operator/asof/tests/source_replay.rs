use super::{fixture, indexed_input};
use crate::{
    AsofStateLimits, Batch, BatchMetadata, CancellationToken, EdgeCollector, IngressProgress,
    IngressProgressSnapshot, IngressState, JsonMap, OperatorMetadata, StreamAsofJoinOperator,
    StreamAsofJoinSpec, StreamAsofJoinStatus, StreamJobContext, StreamOperator,
    StreamOperatorContext, runtime::streaming::gather_work::TestService,
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

fn with_cursor(batch: &Batch, side: &str) -> Batch {
    batch.with_source_cursor(Some(Arc::new(
        crate::Cursor::new(side, vec![1], JsonMap::new()).unwrap(),
    )))
}

#[test]
fn replay_recording_pressure_preserves_admission_success() {
    let (template, _) = fixture();
    let schema = template.schemas[0].clone();
    let service = TestService::new(8, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        for limit in [32 * 1024, 64 * 1024, 128 * 1024] {
            for count in [1, 4, 16, 64, 256] {
                run_pressure_pair(&template, &schema, limit, count, &service).await;
            }
        }
    });
    drop(runtime);
    service.shutdown();
}

fn pressure_progress() -> IngressProgressSnapshot {
    IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, None),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, None),
        ),
    ]))
}

async fn run_pressure_pair(
    template: &StreamAsofJoinOperator,
    schema: &datafusion::arrow::datatypes::SchemaRef,
    limit: u64,
    count: i32,
    service: &TestService,
) {
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::from_micros(10),
        AsofStateLimits::new(10_000, limit).unwrap(),
    )
    .unwrap();
    let plain =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec.clone()).unwrap();
    let mut recorded =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    recorded.replay = Some(Box::new(recorded.new_replay_log().unwrap()));
    recorded.status.state_bytes = recorded.current_inventory(None).unwrap().bytes;
    let rows = (0..count)
        .map(|sequence| ("A", 100, i64::from(sequence)))
        .collect::<Vec<_>>();
    let batch = Batch::table(vec![indexed_input(schema, &rows)], BatchMetadata::default()).unwrap();
    let baseline = run_pressure_operator(plain, &batch, false, service).await;
    let candidate =
        run_pressure_operator(recorded, &batch, limit == 128 * 1024 && count == 1, service).await;
    assert_eq!(
        candidate.accepted, baseline.accepted,
        "limit={limit}, rows={count}"
    );
    assert_status(&baseline.admitted, candidate.admitted);
    if baseline.finalized.is_ok() {
        assert!(
            candidate.finalized.is_ok(),
            "limit={limit}, rows={count}, finalization: {:?}",
            candidate.finalized
        );
        assert_status(&baseline.finished, candidate.finished);
    }
}

struct PressureOutcome {
    accepted: Vec<bool>,
    admitted: StreamAsofJoinStatus,
    finalized: crate::Result<()>,
    finished: StreamAsofJoinStatus,
}

async fn run_pressure_operator(
    mut operator: StreamAsofJoinOperator,
    batch: &Batch,
    keep_replay: bool,
    service: &TestService,
) -> PressureOutcome {
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(service.owner("asof-pressure".into()));
    let context =
        StreamOperatorContext::with_ingress_progress(&job, "asof", None, pressure_progress());
    let recorded = operator.replay.is_some();
    let mut accepted = Vec::new();
    for side in ["left", "right"] {
        let input = if recorded {
            with_cursor(batch, side)
        } else {
            batch.clone()
        };
        let result = operator
            .process_data(
                side,
                input,
                &context,
                &mut EdgeCollector::new(operator.output_ports().to_vec()),
            )
            .await;
        accepted.push(result.is_ok());
        if result.is_err() {
            break;
        }
        if keep_replay && side == "left" {
            assert!(operator.replay.is_some());
        }
    }
    let admitted = operator.status();
    let finalized = operator
        .on_end(
            &context,
            &mut EdgeCollector::new(operator.output_ports().to_vec()),
        )
        .await;
    let finished = operator.status();
    let pool = operator.runtime.pool.clone();
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    operator.reset().unwrap();
    drop(operator);
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
    PressureOutcome {
        accepted,
        admitted,
        finalized,
        finished,
    }
}

fn assert_status(plain: &StreamAsofJoinStatus, mut recorded: StreamAsofJoinStatus) {
    recorded.state_bytes = plain.state_bytes;
    assert_eq!(&recorded, plain);
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

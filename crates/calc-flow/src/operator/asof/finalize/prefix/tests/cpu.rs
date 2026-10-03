use super::*;
use crate::{
    AsofJoinSide, AsofStateLimits, Batch, BatchMetadata, CancellationToken, EdgeCollector, JsonMap,
    OperatorMetadata, StreamAsofJoinSpec, StreamJobContext, StreamOperator,
    runtime::streaming::gather_work::TestService,
};
use datafusion::arrow::{
    array::{Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use std::time::Duration;

pub(super) fn fixture() -> (StreamAsofJoinOperator, Batch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::Int64, false),
    ]));
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["seq".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::ZERO,
        AsofStateLimits::new(1000, 2 << 20).unwrap(),
    )
    .unwrap();
    let operator =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![0; 64])),
            Arc::new(TimestampMicrosecondArray::from_iter_values(0..64).with_timezone("UTC")),
            Arc::new(Int64Array::from_iter_values(0..64)),
        ],
    )
    .unwrap();
    (
        operator,
        Batch::table(vec![record], BatchMetadata::default()).unwrap(),
    )
}

async fn prepare_with_blocked_tokio(service: &TestService) -> bool {
    let (mut operator, batch) = fixture();
    let pool = operator.runtime.pool.clone();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(service.owner("prefix-cpu".into()));
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", batch, &context, &mut output)
        .await
        .unwrap();
    let mut prefix = LeftPrefix::default();
    for (order, row) in operator.state.left.iter().take(32) {
        prefix
            .visit(&order, row, &operator.state.batches, "asof")
            .unwrap();
    }
    assert!(
        operator
            .state
            .left
            .drain_workspace_bytes(&prefix.batches, &operator.state.batches, "asof")
            .unwrap()
            > 0
    );
    let expected = operator
        .state
        .left
        .iter()
        .skip(32)
        .map(|(order, _)| (*order.0, order.1.clone(), order.2.into_owned()))
        .collect::<Vec<_>>();
    let (entered, started) = tokio::sync::oneshot::channel();
    let (release, gate) = std::sync::mpsc::channel();
    let blocker = tokio::task::spawn_blocking(move || {
        entered.send(()).unwrap();
        gate.recv_timeout(Duration::from_secs(10)).unwrap();
    });
    started.await.unwrap();
    let mut completed = true;
    for _ in 0..2 {
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            operator.prepare_left_drain(&prefix, &context),
        )
        .await;
        let Ok(Ok((prepared, workspace))) = result else {
            completed = false;
            break;
        };
        let mut candidate = operator.state.left.clone();
        candidate.drain_prefix(
            prefix.count,
            &prefix.batches,
            &operator.state.batches,
            prepared,
        );
        let actual = candidate
            .iter()
            .map(|(order, _)| (*order.0, order.1.clone(), order.2.into_owned()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert_eq!(operator.state.left.len(), 64);
        drop(workspace);
        assert_eq!(service.joined_workers(), 0);
    }
    release.send(()).unwrap();
    blocker.await.unwrap();
    if completed {
        assert_eq!(service.available_capacity().0, 0);
    }
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    drop((operator, output));
    if completed {
        assert_eq!(pool.reserved(), 0);
    }
    completed
}

#[test]
fn prefix_compaction_reuses_owned_workers_with_tokio_blocked() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    let completed = runtime.block_on(prepare_with_blocked_tokio(&service));
    drop(runtime);
    service.shutdown();
    assert!(
        completed,
        "left prefix compaction waits for Tokio blocking capacity"
    );
}

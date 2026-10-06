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

struct RetirementCollector {
    release: Option<std::sync::mpsc::Receiver<()>>,
    entered: Option<tokio::sync::oneshot::Sender<()>>,
}

#[async_trait::async_trait]
impl StreamCollector for RetirementCollector {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        let release = self.release.take().unwrap();
        let (started, ready) = tokio::sync::oneshot::channel();
        tokio::task::spawn_blocking(move || {
            started.send(()).unwrap();
            release.recv_timeout(Duration::from_secs(10)).unwrap();
        });
        ready.await.unwrap();
        self.entered.take().unwrap().send(()).unwrap();
        tokio::task::yield_now().await;
        Ok(())
    }
}

async fn prefix_retirement_is_awaited(service: &TestService) -> bool {
    let (mut operator, batch) = fixture();
    let record = &batch.table_payload().unwrap().batches()[0];
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(service.owner("prefix-retirement".into()));
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut preload = EdgeCollector::new(operator.output_ports().to_vec());
    for row in 0..64 {
        operator
            .process_data(
                "left",
                Batch::table(vec![record.slice(row, 1)], BatchMetadata::default()).unwrap(),
                &context,
                &mut preload,
            )
            .await
            .unwrap();
    }
    let headroom = operator.checkpoint_workspace().unwrap();
    let (_, prefix_workspace) = operator.finalizable_rows(None, true, &context).unwrap();
    let mut count = 50;
    let output = operator
        .prepare_output(&mut count, &prefix_workspace, &context)
        .await
        .unwrap();
    assert!(
        operator
            .state
            .batches
            .project_remove(&output.prefix.batches, "asof")
            .unwrap()
            .replace
    );
    let (release, blocked) = std::sync::mpsc::channel();
    let (entered, ready) = tokio::sync::oneshot::channel();
    let mut collector = RetirementCollector {
        release: Some(blocked),
        entered: Some(entered),
    };
    let mut commit =
        Box::pin(operator.commit_prefix_output(output, headroom, &context, &mut collector));
    tokio::select! {
        result = &mut commit => panic!("prefix returned before collector accepted: {result:?}"),
        result = ready => result.unwrap(),
    }
    let waits_for_retirement = futures::poll!(commit.as_mut()).is_pending();
    release.send(()).unwrap();
    if waits_for_retirement {
        commit.await.unwrap();
    }
    drop(prefix_workspace);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    waits_for_retirement
}

#[test]
fn committed_prefix_waits_for_payload_retirement_before_reusing_workspace() {
    let service = TestService::new(1, 8).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    let waits = runtime.block_on(prefix_retirement_is_awaited(&service));
    drop(runtime);
    service.shutdown();
    assert!(
        waits,
        "accepted prefix returned while retired payloads still held workspace"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn dropped_retirement_waits_stay_funded_across_reset_restore_and_cancellation() {
    let (mut operator, batch) = fixture();
    let empty = Batch::table(
        vec![batch.table_payload().unwrap().batches()[0].slice(0, 0)],
        BatchMetadata::default(),
    )
    .unwrap();
    let saved = operator.capture(crate::Epoch::INITIAL).unwrap();
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, cancellation.clone());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let pool = operator.runtime.pool.clone();
    let reservation = operator.reserve_workspace(4_096).unwrap();
    let ticket = operator.retirement.register(&context).unwrap();
    let owner = Arc::new(vec![0_u8; 4_096]);
    let weak = Arc::downgrade(&owner);
    let (entered, ready) = tokio::sync::oneshot::channel();
    let (release, blocked) = std::sync::mpsc::channel();
    let worker = tokio::task::spawn_blocking(move || {
        entered.send(()).unwrap();
        blocked.recv_timeout(Duration::from_secs(10)).unwrap();
        drop((owner, reservation));
        drop(ticket);
    });
    ready.await.unwrap();
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    for restore in [false, true] {
        let mut mutation =
            Box::pin(operator.process_data("left", empty.clone(), &context, &mut output));
        assert!(futures::poll!(mutation.as_mut()).is_pending());
        drop(mutation);
        if restore {
            operator.restore(&saved).unwrap();
        } else {
            operator.reset().unwrap();
        }
    }
    let mut checkpoint = Box::pin(operator.prepare_checkpoint_async(&context));
    assert!(futures::poll!(checkpoint.as_mut()).is_pending());
    cancellation.cancel();
    assert!(matches!(
        checkpoint.await,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(pool.reserved(), 4_096);
    assert!(weak.upgrade().is_some());
    let unchanged = operator.capture(crate::Epoch::INITIAL).unwrap();
    assert_eq!(unchanged.inline_metadata, saved.inline_metadata);
    assert_eq!(unchanged.segments, saved.segments);
    let mut drain = Box::pin(job.gather_owner().close_and_drain());
    assert!(futures::poll!(drain.as_mut()).is_pending());
    release.send(()).unwrap();
    worker.await.unwrap();
    assert!(drain.await.is_empty());
    let resumed = StreamJobContext::new(2, "asof", JsonMap::new(), None, CancellationToken::new());
    let resumed_context = StreamOperatorContext::new(&resumed, "asof", None);
    operator
        .process_data("left", empty, &resumed_context, &mut output)
        .await
        .unwrap();
    assert_eq!(pool.reserved(), 0);
    assert!(weak.upgrade().is_none());
}

async fn managed_retirement_is_drained() -> bool {
    use crate::operator::asof::state::{PayloadBatch, PayloadPool, PreparedPayloadRemoval};
    let (operator, batch) = fixture();
    let pool = operator.runtime.pool.clone();
    let reservation = operator.reserve_workspace(4_096).unwrap();
    let payload = Arc::new(PayloadBatch {
        key: (0, 1),
        record: Arc::new(batch.table_payload().unwrap().batches()[0].clone()),
        encoded: std::sync::OnceLock::new(),
        encoded_charge_bytes: 0,
        body_bytes: 1_536,
    });
    let weak = Arc::downgrade(&payload);
    drop(batch);
    let job = StreamJobContext::new(
        1,
        "managed-retirement",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "asof", None);
    let ticket = operator.retirement.register(&context).unwrap();
    let (entered, ready) = tokio::sync::oneshot::channel();
    let (release, blocked) = std::sync::mpsc::channel();
    tokio::task::spawn_blocking(move || {
        entered.send(()).unwrap();
        blocked.recv_timeout(Duration::from_secs(10)).unwrap();
    });
    ready.await.unwrap();
    let payloads = PayloadPool::default();
    let layout = payloads
        .project_remove(&std::collections::BTreeMap::new(), "asof")
        .unwrap();
    let mut prepared =
        PreparedPayloadRemoval::capture(&payloads, &layout, reservation, Some(ticket));
    prepared.retain(1, &payload, 1);
    drop(payload);
    drop(prepared);
    drop(operator);
    let mut drain = Box::pin(job.gather_owner().close_and_drain());
    let first_waits = futures::poll!(drain.as_mut()).is_pending();
    drop(drain);
    let mut drain = Box::pin(job.gather_owner().close_and_drain());
    let waits = futures::poll!(drain.as_mut()).is_pending() && first_waits;
    let rejected = crate::operator::asof::retirement::Owner::default();
    assert!(matches!(
        rejected.register(&context),
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    rejected.wait(&context).await.unwrap();
    assert_eq!(pool.reserved(), 4_096);
    assert!(weak.upgrade().is_some());
    release.send(()).unwrap();
    if waits {
        assert!(drain.await.is_empty());
        assert_eq!(pool.reserved(), 0);
        assert!(weak.upgrade().is_none());
    }
    waits
}

#[test]
fn managed_job_drain_owns_retirement_after_operator_drop() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    let waits = runtime.block_on(managed_retirement_is_drained());
    drop(runtime);
    assert!(
        waits,
        "managed cleanup returned before funded retired payloads were destroyed"
    );
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

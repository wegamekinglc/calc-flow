use super::*;
use crate::{
    AsofJoinSide, AsofStateLimits, CancellationToken, EdgeBudget, EdgeCollector, Epoch,
    OperatorMetadata, StreamAsofJoinSpec, StreamJobContext, StreamOperator,
};
use datafusion::arrow::{
    array::{Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use std::time::Duration;

const ROWS: usize = 65_537;
const WORKSPACE: usize = 256 << 20;

async fn admitted() -> (StreamAsofJoinOperator, StreamJobContext, EdgeCollector) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::Int64, false),
    ]));
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["sequence".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::ZERO,
        AsofStateLimits::new(ROWS as u64, WORKSPACE as u64).unwrap(),
    )
    .unwrap();
    let mut operator =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let batch = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["A"; ROWS])),
                    Arc::new(TimestampMicrosecondArray::from(vec![100; ROWS]).with_timezone("UTC")),
                    Arc::new(Int64Array::from_iter_values(
                        0..i64::try_from(ROWS).unwrap(),
                    )),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    let job = StreamJobContext::new(911, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", batch, &context, &mut output)
        .await
        .unwrap();
    (operator, job, output)
}

#[tokio::test]
async fn test_large_prefix_uses_configured_edge_rows() {
    let (mut operator, job, mut output) = admitted().await;
    let pool = operator.runtime.pool.clone();
    let context = StreamOperatorContext::new(&job, "asof", None)
        .with_output_budget(EdgeBudget::new(ROWS, 64 << 20).unwrap());
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    let mut restored = StreamAsofJoinOperator::new(
        "asof",
        operator.schemas[0].clone(),
        operator.schemas[1].clone(),
        operator.spec.clone(),
    )
    .unwrap();
    let restored_pool = restored.runtime.pool.clone();
    restored.restore(&snapshot).unwrap();
    operator.on_end(&context, &mut output).await.unwrap();
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    let batch = messages[0].as_data().unwrap();
    assert_eq!(batch.num_rows(), ROWS);
    let records = batch.table_payload().unwrap().batches();
    let sequences = records
        .iter()
        .flat_map(|record| {
            record
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        sequences,
        (0..i64::try_from(ROWS).unwrap()).collect::<Vec<_>>()
    );
    assert!(
        records
            .iter()
            .all(|record| record.column(5).null_count() == record.num_rows())
    );
    assert_eq!(operator.status.pending_left_rows, 0);
    restored.on_end(&context, &mut output).await.unwrap();
    let resumed = output.drain("output");
    assert_eq!(resumed.len(), 1);
    let resumed_batch = resumed[0].as_data().unwrap();
    assert_eq!(resumed_batch.metadata(), batch.metadata());
    assert_eq!(resumed_batch.table_payload().unwrap().batches(), records);
    assert_eq!(restored.status, operator.status);
    drop((
        operator, restored, snapshot, output, context, messages, resumed,
    ));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(restored_pool.reserved(), 0);
}

#[tokio::test]
async fn test_large_prefix_respects_explicit_smaller_edge() {
    let (mut operator, job, mut output) = admitted().await;
    let pool = operator.runtime.pool.clone();
    let context = StreamOperatorContext::new(&job, "asof", None)
        .with_output_budget(EdgeBudget::new(ROWS / 2, 64 << 20).unwrap());
    operator.on_end(&context, &mut output).await.unwrap();
    let messages = output.drain("output");
    let batches = messages
        .iter()
        .map(|message| message.as_data().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        batches
            .iter()
            .map(|batch| batch.num_rows())
            .collect::<Vec<_>>(),
        [32_768, 32_768, 1]
    );
    assert_eq!(
        batches
            .iter()
            .map(|batch| batch.metadata().sequence())
            .collect::<Vec<_>>(),
        [0, 32_768, 65_536]
    );
    assert_eq!(operator.status.pending_left_rows, 0);
    assert_eq!(operator.next_output_sequence, ROWS as u64);
    drop((operator, output, context, messages));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_cancelled_large_prefix_preserves_checkpoint_and_credit() {
    let (mut operator, job, output) = admitted().await;
    let pool = operator.runtime.pool.clone();
    let before = operator.capture(Epoch::INITIAL).unwrap();
    let reserved = pool.reserved();
    let context = StreamOperatorContext::new(&job, "asof", None)
        .with_output_budget(EdgeBudget::new(ROWS, 64 << 20).unwrap());
    job.cancellation().cancel();
    assert!(matches!(
        operator.finalizable_rows(None, true, &context),
        Err(CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(pool.reserved(), reserved);
    let after = operator.capture(Epoch::INITIAL).unwrap();
    assert_eq!(before.inline_metadata, after.inline_metadata);
    assert_eq!(before.segments, after.segments);
    drop((operator, output, context, before, after));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_large_prefix_halves_to_available_workspace_without_consuming_state() {
    let (operator, job, output) = admitted().await;
    let pool = operator.runtime.pool.clone();
    let before = pool.reserved();
    let status = operator.status.clone();
    let context = StreamOperatorContext::new(&job, "asof", None)
        .with_output_budget(EdgeBudget::new(ROWS, 64 << 20).unwrap());
    let allowance = prefix_workspace_bytes(
        ROWS / 2,
        operator.state.left.iter_workspace_bytes(),
        &operator.name,
    )
    .unwrap();
    let occupied = operator
        .reserve_workspace((WORKSPACE - before) as u64 - allowance)
        .unwrap();
    let (count, credit) = operator.finalizable_rows(None, true, &context).unwrap();
    assert_eq!(count, ROWS / 2);
    assert_eq!(credit.size() as u64, allowance);
    assert_eq!(operator.status, status);
    drop((credit, occupied));
    assert_eq!(pool.reserved(), before);
    drop((operator, output, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_oversized_prefix_reservation_fails_without_overflow_or_charge() {
    let (operator, job, output) = admitted().await;
    let pool = operator.runtime.pool.clone();
    let before = pool.reserved();
    let result = prefix_workspace_bytes(usize::MAX, 0, &operator.name)
        .and_then(|bytes| operator.reserve_workspace(bytes));
    assert!(matches!(
        result,
        Err(CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        })
    ));
    assert_eq!(pool.reserved(), before);
    drop((operator, output));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

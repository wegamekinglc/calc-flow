use std::{sync::Arc, time::Duration};

use datafusion::{
    arrow::{
        array::{Int64Array, TimestampMicrosecondArray},
        datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
        record_batch::RecordBatch,
    },
    execution::memory_pool::MemoryPool,
};
use tokio::sync::mpsc;

use super::{
    Harness, OperatorCheckpointAck, OperatorCheckpointPort, harness_with_operator_capability, start,
};
use crate::{
    Batch, BatchMetadata, Epoch, JoinStateLimits, JoinTimeBounds, LocalStateBackend,
    OperatorMetadata, StateBackend, StateLineageBackend, StateLineageKey, StreamJoinOperator,
    StreamJoinSpec, StreamMessage,
    pipeline::{CompiledStreamOperator, OperatorCheckpointCapability},
    state::ManifestTransaction,
};

struct Fixture {
    harness: Harness,
    acks: mpsc::Receiver<OperatorCheckpointAck>,
    pool: Arc<dyn MemoryPool>,
    _directory: tempfile::TempDir,
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
    ]))
}

fn operator() -> StreamJoinOperator {
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "ts",
        "ts",
        JoinTimeBounds::new(Duration::from_micros(10), Duration::from_micros(10)).unwrap(),
        JoinStateLimits::new(100, 1_000_000, 100).unwrap(),
    )
    .unwrap();
    StreamJoinOperator::new("node", schema(), schema(), spec).unwrap()
}

fn row(time: i64) -> Batch {
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(TimestampMicrosecondArray::from(vec![time])),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

async fn fixture() -> Fixture {
    let directory = tempfile::tempdir().unwrap();
    let backend = LocalStateBackend::new(directory.path().join("state"))
        .await
        .unwrap();
    let key = StateLineageKey::new("join-capture-owner", &"b".repeat(64)).unwrap();
    let lineage: Arc<dyn StateLineageBackend> =
        Arc::from(backend.open_lineage(&key).await.unwrap());
    let transaction =
        ManifestTransaction::open(lineage, &key, directory.path().join("manifests"), 2)
            .await
            .unwrap();
    let mut operator = operator();
    let pool = operator.checkpoint_preload_test_pool().unwrap();
    let port = operator.output_ports()[0].clone();
    let (sender, acks) = mpsc::channel(1);
    let harness = harness_with_operator_capability(
        &["left", "right"],
        1,
        CompiledStreamOperator::StreamJoin(Box::new(operator)),
        OperatorCheckpointCapability::CheckpointedStateful { state_version: 1 },
        port,
        Some(OperatorCheckpointPort {
            acks: sender,
            transaction: Some(Arc::new(transaction)),
            terminal: None,
            alignment_fault: None,
        }),
        None,
    );
    Fixture {
        harness,
        acks,
        pool,
        _directory: directory,
    }
}

async fn send(harness: &mut Harness, side: &str, message: StreamMessage) {
    harness
        .inputs
        .get_mut(side)
        .unwrap()
        .send(message)
        .await
        .unwrap();
}

async fn collect_barrier(harness: &mut Harness) {
    let mut output_rows = 0;
    loop {
        let message = tokio::time::timeout(Duration::from_secs(5), harness.outputs[0].recv())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        if message.as_barrier() == Some(Epoch::INITIAL) {
            break;
        }
        if let Some(batch) = message.as_data() {
            output_rows += batch.num_rows();
        }
    }
    assert_eq!(output_rows, 1);
}

fn assert_ack(ack: &OperatorCheckpointAck) {
    let metadata = &ack.state.inline_metadata;
    assert_eq!(ack.node_id, "node");
    assert_eq!(ack.epoch, Epoch::INITIAL);
    assert_eq!(metadata["layout_version"], 2);
    assert_eq!(metadata["next_left_row_id"], 1);
    assert_eq!(metadata["next_right_row_id"], 1);
    assert_eq!(metadata["next_output_sequence"], 1);
    assert_eq!(metadata["metrics"]["left"]["retained_rows"], 1);
    assert_eq!(metadata["metrics"]["right"]["retained_rows"], 1);
    assert_eq!(metadata["metrics"]["emitted_match_rows"], 1);
    assert_eq!(metadata["v2_inventory"]["base_epoch"], 1);
    assert_eq!(metadata["v2_inventory"]["deltas"], serde_json::json!([]));
    assert_eq!(ack.state.segments.len(), 4);
}

#[tokio::test]
async fn test_join_checkpoint_ack_retains_metadata_credit_after_operator_exit() {
    let mut fixture = fixture().await;
    start(&mut fixture.harness).await;
    send(&mut fixture.harness, "left", StreamMessage::data(row(95))).await;
    send(
        &mut fixture.harness,
        "left",
        StreamMessage::barrier(Epoch::INITIAL),
    )
    .await;
    send(&mut fixture.harness, "right", StreamMessage::data(row(100))).await;
    send(
        &mut fixture.harness,
        "right",
        StreamMessage::barrier(Epoch::INITIAL),
    )
    .await;
    let ack = tokio::time::timeout(Duration::from_secs(5), fixture.acks.recv())
        .await
        .unwrap()
        .unwrap();
    assert_ack(&ack);
    collect_barrier(&mut fixture.harness).await;
    assert!(fixture.pool.reserved() > 0);
    let gather = fixture.harness.gather.clone();
    fixture.harness.cancellation.cancel();
    let report = tokio::time::timeout(
        Duration::from_secs(5),
        fixture.harness.supervisor.join_all(),
    )
    .await
    .unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    drop(fixture.harness);
    drop(fixture.acks);
    let failures = tokio::time::timeout(Duration::from_secs(5), gather.close_and_drain())
        .await
        .unwrap();
    assert!(failures.is_empty(), "{failures:?}");
    drop(gather);
    assert_ack(&ack);
    assert!(
        fixture.pool.reserved() > 0,
        "live checkpoint acknowledgement must retain its metadata credit"
    );
    drop(ack);
    assert_eq!(fixture.pool.reserved(), 0);
}

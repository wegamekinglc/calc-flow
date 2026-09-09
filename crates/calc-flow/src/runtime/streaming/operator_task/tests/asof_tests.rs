use super::{
    Harness, OperatorCheckpointAck, OperatorCheckpointPort, harness_with_operator_capability, start,
};
use crate::{
    AsofJoinSide, AsofStateLimits, Batch, BatchMetadata, EdgeReceiver, Epoch, EventTime,
    LocalStateBackend, OperatorMetadata, StateBackend, StateLineageBackend, StateLineageKey,
    StreamAsofJoinOperator, StreamAsofJoinSpec, StreamMessage,
    pipeline::{CompiledStreamOperator, OperatorCheckpointCapability},
    state::ManifestTransaction,
};
use datafusion::arrow::{
    array::{Array, Int64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use std::{sync::Arc, time::Duration};
use tokio::sync::mpsc;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("value", DataType::Int64, false),
    ]))
}

fn row(time: i64, sequence: u64) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(StringArray::from(vec!["a"])),
                    Arc::new(TimestampMicrosecondArray::from(vec![time]).with_timezone("UTC")),
                    Arc::new(UInt64Array::from(vec![sequence])),
                    Arc::new(Int64Array::from(vec![time])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn operator() -> StreamAsofJoinOperator {
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "ts".into(),
            vec!["sequence".into()],
            prefix.into(),
        )
        .unwrap()
    };
    StreamAsofJoinOperator::new(
        "node",
        schema(),
        schema(),
        StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            Duration::from_micros(10),
            AsofStateLimits::new(1_000, 16 * 1024 * 1024).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}

struct TaskFixture {
    harness: Harness,
    acks: mpsc::Receiver<OperatorCheckpointAck>,
    epoch: Epoch,
    _directory: tempfile::TempDir,
}

impl TaskFixture {
    async fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let backend = LocalStateBackend::new(directory.path().join("state"))
            .await
            .unwrap();
        let key = StateLineageKey::new("asof-task", &"a".repeat(64)).unwrap();
        let lineage: Arc<dyn StateLineageBackend> =
            Arc::from(backend.open_lineage(&key).await.unwrap());
        let transaction =
            ManifestTransaction::open(lineage, &key, directory.path().join("manifests"), 2)
                .await
                .unwrap();
        let operator = operator();
        let port = operator.output_ports()[0].clone();
        let (sender, acks) = mpsc::channel(1);
        let mut harness = harness_with_operator_capability(
            &["left", "right"],
            1,
            CompiledStreamOperator::StreamAsofJoin(Box::new(operator)),
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
        start(&mut harness).await;
        Self {
            harness,
            acks,
            epoch: Epoch::INITIAL,
            _directory: directory,
        }
    }

    async fn send(&mut self, side: &str, message: StreamMessage) {
        self.harness
            .inputs
            .get_mut(side)
            .unwrap()
            .send(message)
            .await
            .unwrap();
    }

    async fn watermark(&mut self, side: &str, time: i64) {
        self.send(side, StreamMessage::watermark(EventTime::from_micros(time)))
            .await;
    }

    async fn fence(&mut self, active: &[&str]) -> Vec<StreamMessage> {
        for side in active {
            self.send(side, StreamMessage::barrier(self.epoch)).await;
        }
        let ack = tokio::time::timeout(Duration::from_secs(5), self.acks.recv())
            .await
            .expect("ASOF barrier must finish")
            .expect("ASOF checkpoint acknowledgement");
        assert_eq!(ack.epoch, self.epoch);
        let mut messages = Vec::new();
        loop {
            let message = receive(&mut self.harness.outputs[0]).await;
            if let Some(epoch) = message.as_barrier() {
                assert_eq!(epoch, self.epoch);
                break;
            }
            messages.push(message);
        }
        self.epoch = self.epoch.next().unwrap();
        messages
    }

    async fn finish(mut self) -> Vec<StreamMessage> {
        for side in ["left", "right"] {
            self.send(side, StreamMessage::end_of_input()).await;
        }
        let report =
            tokio::time::timeout(Duration::from_secs(5), self.harness.supervisor.join_all())
                .await
                .expect("ASOF task must finish");
        assert!(report.errors.is_empty(), "{report:?}");
        let mut messages = Vec::new();
        loop {
            let message = receive(&mut self.harness.outputs[0]).await;
            let ended = message.is_end_of_input();
            messages.push(message);
            if ended {
                break;
            }
        }
        messages
    }
}

async fn receive(receiver: &mut EdgeReceiver) -> StreamMessage {
    tokio::time::timeout(Duration::from_secs(5), receiver.recv())
        .await
        .expect("ASOF output must arrive")
        .unwrap()
        .expect("ASOF output stays open")
}

fn selected(messages: &[StreamMessage]) -> Vec<(i64, Option<i64>)> {
    messages
        .iter()
        .filter_map(StreamMessage::as_data)
        .flat_map(|batch| batch.table_payload().unwrap().batches())
        .flat_map(|batch| {
            let left = batch
                .column_by_name("left__ts")
                .unwrap()
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            let right = batch
                .column_by_name("right__ts")
                .unwrap()
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            (0..batch.num_rows())
                .map(|index| {
                    (
                        left.value(index),
                        (!right.is_null(index)).then(|| right.value(index)),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn watermarks(messages: &[StreamMessage]) -> Vec<i64> {
    messages
        .iter()
        .filter_map(StreamMessage::as_watermark)
        .map(EventTime::as_micros)
        .collect()
}

#[tokio::test]
async fn test_asof_strict_dual_watermark_and_data_before_safe_frontier() {
    let mut task = TaskFixture::new().await;
    task.send("right", StreamMessage::data(row(100, 1))).await;
    task.send("left", StreamMessage::data(row(105, 1))).await;
    task.watermark("left", 106).await;
    task.watermark("right", 105).await;
    let waiting = task.fence(&["left", "right"]).await;
    assert!(selected(&waiting).is_empty());
    assert_eq!(watermarks(&waiting), [104]);
    task.watermark("right", 106).await;
    let final_output = task.fence(&["left", "right"]).await;
    assert_eq!(selected(&final_output), [(105, Some(100))]);
    assert!(final_output[0].as_data().is_some());
    assert_eq!(watermarks(&final_output), [105]);
    task.watermark("right", 106).await;
    task.watermark("left", 106).await;
    assert!(task.fence(&["left", "right"]).await.is_empty());
    assert!(selected(&task.finish().await).is_empty());
}

#[tokio::test]
async fn test_asof_reactivation_drains_without_new_aggregate_watermark() {
    let mut task = TaskFixture::new().await;
    task.send("right", StreamMessage::data(row(100, 1))).await;
    task.send("left", StreamMessage::data(row(105, 1))).await;
    task.watermark("right", 90).await;
    task.watermark("left", 1_000).await;
    assert!(selected(&task.fence(&["left", "right"]).await).is_empty());
    task.send("right", StreamMessage::idle()).await;
    assert!(task.fence(&["left", "right"]).await.is_empty());
    task.watermark("right", 106).await;
    let resumed = task.fence(&["left", "right"]).await;
    assert_eq!(selected(&resumed), [(105, Some(100))]);
    assert_eq!(watermarks(&resumed), [105]);
    assert!(resumed[0].as_data().is_some());
    task.finish().await;
}

#[tokio::test]
async fn test_asof_idle_never_forwards_with_or_without_pending_left() {
    for pending in [false, true] {
        let mut task = TaskFixture::new().await;
        task.watermark("left", 90).await;
        task.watermark("right", 90).await;
        task.fence(&["left", "right"]).await;
        if pending {
            task.send("left", StreamMessage::data(row(105, 1))).await;
        }
        task.send("left", StreamMessage::idle()).await;
        task.send("right", StreamMessage::idle()).await;
        let idle = task.fence(&["left", "right"]).await;
        assert!(idle.is_empty(), "pending={pending}: {idle:?}");
        task.send("right", StreamMessage::data(row(100, 1))).await;
        if !pending {
            task.send("left", StreamMessage::data(row(105, 1))).await;
        }
        task.watermark("left", 106).await;
        task.watermark("right", 106).await;
        let resumed = task.fence(&["left", "right"]).await;
        assert_eq!(selected(&resumed), [(105, Some(100))]);
        assert!(!resumed.iter().any(StreamMessage::is_idle));
        task.finish().await;
    }
}

#[tokio::test]
async fn test_asof_single_eof_still_waits_for_other_side_to_close_left_time() {
    let mut task = TaskFixture::new().await;
    task.send("left", StreamMessage::data(row(105, 1))).await;
    task.watermark("right", 105).await;
    task.send("left", StreamMessage::end_of_input()).await;
    let waiting = task.fence(&["right"]).await;
    assert!(selected(&waiting).is_empty());
    task.send("right", StreamMessage::data(row(105, 2))).await;
    task.watermark("right", 106).await;
    assert_eq!(selected(&task.fence(&["right"]).await), [(105, Some(105))]);
    task.send("right", StreamMessage::end_of_input()).await;
    let report = task.harness.supervisor.join_all().await;
    assert!(report.errors.is_empty(), "{report:?}");
    assert!(
        receive(&mut task.harness.outputs[0])
            .await
            .is_end_of_input()
    );
}

#[tokio::test]
async fn test_asof_minimum_watermark_has_no_underflow_and_maximum_requires_eof() {
    for time in [i64::MIN, i64::MAX] {
        let mut task = TaskFixture::new().await;
        task.send("left", StreamMessage::data(row(time, 1))).await;
        task.send("right", StreamMessage::data(row(time, 1))).await;
        task.watermark("left", time).await;
        task.watermark("right", time).await;
        let waiting = task.fence(&["left", "right"]).await;
        assert!(selected(&waiting).is_empty());
        assert_eq!(
            watermarks(&waiting),
            if time == i64::MIN {
                vec![]
            } else {
                vec![i64::MAX - 1]
            }
        );
        let final_output = task.finish().await;
        assert_eq!(selected(&final_output), [(time, Some(time))]);
        assert!(final_output.last().unwrap().is_end_of_input());
        assert!(!watermarks(&final_output).contains(&i64::MAX));
    }
}

fn downstream_operator(kind: &str, input: SchemaRef) -> CompiledStreamOperator {
    let common = serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "event_time": "left__ts",
        "sequence_by": ["left__sequence"],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"}
    });
    let mut value = common;
    if kind == "rolling" {
        value["partition_by"] = serde_json::json!(["left__key"]);
        value["value_policy"] = serde_json::json!("stateful_numeric_v1");
        value["outputs"] = serde_json::json!([{
            "kind": "mean", "primitive_version": 1, "input": "left__value",
            "output": "result", "frame": {"kind": "rows", "size": 2}, "min_periods": 1
        }]);
        CompiledStreamOperator::Rolling(
            crate::RollingOperator::new("node", input, serde_json::from_value(value).unwrap())
                .unwrap(),
        )
    } else {
        value["entity_by"] = serde_json::json!(["left__key"]);
        value["partition_by"] = serde_json::json!([]);
        value["grouping"] = serde_json::json!({"kind": "exact_time"});
        value["value_policy"] = serde_json::json!("nan_exclude_preserve_v1");
        value["outputs"] = serde_json::json!([{
            "kind": "demean", "primitive_version": 1, "input": "left__value",
            "output": "result", "min_samples": 1
        }]);
        CompiledStreamOperator::CrossSection(
            crate::CrossSectionOperator::new("node", input, serde_json::from_value(value).unwrap())
                .unwrap(),
        )
    }
}

async fn downstream_result(kind: &str, messages: Vec<StreamMessage>) -> Vec<StreamMessage> {
    let input = operator().output_ports()[0].schema().unwrap().clone();
    let operator = downstream_operator(kind, input);
    let port = match &operator {
        CompiledStreamOperator::Rolling(operator) => operator.output_ports()[0].clone(),
        CompiledStreamOperator::CrossSection(operator) => operator.output_ports()[0].clone(),
        _ => unreachable!(),
    };
    let mut harness = super::harness_with_operator(&["input"], 1, operator, port, None, None);
    start(&mut harness).await;
    for message in messages {
        harness
            .inputs
            .get_mut("input")
            .unwrap()
            .send(message)
            .await
            .unwrap();
    }
    let report = harness.supervisor.join_all().await;
    assert!(report.errors.is_empty(), "{kind}: {report:?}");
    assert_eq!(harness.progress.snapshot().late_rows, 0, "{kind}");
    let mut output = Vec::new();
    loop {
        let message = receive(&mut harness.outputs[0]).await;
        let ended = message.is_end_of_input();
        output.push(message);
        if ended {
            break;
        }
    }
    output
}

#[tokio::test]
async fn test_asof_safe_frontier_preserves_later_equal_time_in_rolling_and_cross_section() {
    for kind in ["rolling", "cross_section"] {
        let mut task = TaskFixture::new().await;
        task.watermark("left", 105).await;
        task.watermark("right", 105).await;
        let mut messages = task.fence(&["left", "right"]).await;
        assert_eq!(watermarks(&messages), [104]);
        task.send("left", StreamMessage::data(row(105, 1))).await;
        task.send("right", StreamMessage::data(row(105, 1))).await;
        task.watermark("left", 106).await;
        task.watermark("right", 106).await;
        messages.extend(task.fence(&["left", "right"]).await);
        messages.extend(task.finish().await);
        let output = downstream_result(kind, messages).await;
        assert_eq!(selected(&output), [(105, Some(105))], "{kind}");
        let values = output
            .iter()
            .filter_map(StreamMessage::as_data)
            .flat_map(|batch| batch.table_payload().unwrap().batches())
            .flat_map(|batch| {
                batch
                    .column_by_name("result")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<datafusion::arrow::array::Float64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            values,
            if kind == "rolling" {
                vec![105.0]
            } else {
                vec![0.0]
            }
        );
    }
}

async fn union_fixture() -> TaskFixture {
    let schema = operator().output_ports()[0].schema().unwrap().clone();
    let ports = ["left", "right"]
        .into_iter()
        .map(|name| {
            crate::Port::with_schema_ref(name, crate::BatchKind::Table, true, Some(schema.clone()))
                .unwrap()
        })
        .collect();
    let operator = crate::UnionOperator::new("node", ports).unwrap();
    let port = operator.output_ports()[0].clone();
    let (sender, acks) = mpsc::channel(1);
    let mut harness = super::harness_with_operator(
        &["left", "right"],
        1,
        CompiledStreamOperator::Union(operator),
        port,
        Some(OperatorCheckpointPort {
            acks: sender,
            transaction: None,
            terminal: None,
            alignment_fault: None,
        }),
        None,
    );
    start(&mut harness).await;
    TaskFixture {
        harness,
        acks,
        epoch: Epoch::INITIAL,
        _directory: tempfile::tempdir().unwrap(),
    }
}

#[tokio::test]
async fn test_asof_idle_cannot_advance_a_multi_input_downstream_past_reactivated_rows() {
    for pending in [false, true] {
        let mut task = TaskFixture::new().await;
        let mut downstream = union_fixture().await;
        task.watermark("left", 90).await;
        task.watermark("right", 90).await;
        for message in task.fence(&["left", "right"]).await {
            downstream.send("left", message).await;
        }
        downstream.watermark("right", 1_000).await;
        assert_eq!(
            watermarks(&downstream.fence(&["left", "right"]).await),
            [89]
        );
        if pending {
            task.send("left", StreamMessage::data(row(105, 1))).await;
        }
        task.send("left", StreamMessage::idle()).await;
        task.send("right", StreamMessage::idle()).await;
        for message in task.fence(&["left", "right"]).await {
            downstream.send("left", message).await;
        }
        assert!(
            downstream.fence(&["left", "right"]).await.is_empty(),
            "pending={pending}"
        );
        if !pending {
            task.send("left", StreamMessage::data(row(105, 1))).await;
        }
        task.send("right", StreamMessage::data(row(100, 1))).await;
        task.watermark("left", 106).await;
        task.watermark("right", 106).await;
        for message in task.fence(&["left", "right"]).await {
            downstream.send("left", message).await;
        }
        let resumed = downstream.fence(&["left", "right"]).await;
        assert_eq!(selected(&resumed), [(105, Some(100))]);
        assert_eq!(watermarks(&resumed), [105]);
        task.finish().await;
        downstream.finish().await;
    }
}

#[tokio::test]
async fn test_asof_restore_rejects_pending_left_at_or_behind_output_frontier_before_ready() {
    use crate::StreamOperator as _;
    let mut original = operator();
    let job = crate::StreamJobContext::new(
        7,
        "fingerprint",
        crate::JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = crate::StreamOperatorContext::new(&job, "node", None);
    let mut output = crate::EdgeCollector::new(original.output_ports().to_vec());
    original
        .process_data("left", row(105, 1), &context, &mut output)
        .await
        .unwrap();
    let snapshot = original.checkpoint(Epoch::INITIAL).unwrap();
    let restored = operator();
    let port = restored.output_ports()[0].clone();
    let progress = ["left", "right"]
        .into_iter()
        .map(|name| {
            (
                name.into(),
                crate::OperatorIngressManifestEntry {
                    state: crate::ManifestIngressState::Active,
                    watermark: Some(EventTime::from_micros(106)),
                },
            )
        })
        .collect();
    let mut harness = harness_with_operator_capability(
        &["left", "right"],
        1,
        CompiledStreamOperator::StreamAsofJoin(Box::new(restored)),
        OperatorCheckpointCapability::CheckpointedStateful { state_version: 1 },
        port,
        None,
        Some(super::OperatorRestoreState {
            snapshot,
            progress,
            output_frontier: Some(EventTime::from_micros(105)),
            next_epoch: Epoch::new(2).unwrap(),
        }),
    );
    harness.entry.send(true).unwrap();
    let ack = tokio::time::timeout(Duration::from_secs(5), harness.ack.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        ack.result,
        Err(crate::CalcFlowError::CheckpointMismatch { .. })
    ));
    assert!(harness.supervisor.join_all().await.errors.is_empty());
    assert!(harness.outputs[0].recv().await.unwrap().is_none());
}

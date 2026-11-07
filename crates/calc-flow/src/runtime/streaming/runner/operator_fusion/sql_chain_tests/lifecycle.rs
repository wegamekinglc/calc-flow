use super::*;
use crate::runtime::streaming::operator_task::{OperatorCheckpointCommand, OperatorRestoreState};
use crate::{Epoch, EventTime, LocalStateBackend, StateBackend, StateLineageKey};

pub(super) fn floating_key_input(sequence: u64, values: &[i64]) -> Batch {
    use datafusion::arrow::array::Float64Array;
    let batch = input(sequence, values, true);
    let record = &batch.table_payload().unwrap().batches()[0];
    let fields = record
        .schema()
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| {
            if index == 1 {
                Field::new("stock", DataType::Float64, false)
            } else {
                field.as_ref().clone()
            }
        })
        .collect::<Vec<_>>();
    let mut columns = record.columns().to_vec();
    columns[1] = Arc::new(Float64Array::from(vec![1.0; values.len()]));
    Batch::table(
        vec![RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()],
        batch.metadata().clone(),
    )
    .unwrap()
}

fn test_input(fallback: bool, sequence: u64, values: &[i64]) -> Batch {
    if fallback {
        floating_key_input(sequence, values)
    } else {
        input(sequence, values, true)
    }
}

pub(super) fn tagged(batch: &Batch, case: &str) -> Batch {
    Batch::table(
        batch.table_payload().unwrap().batches().to_vec(),
        BatchMetadata::new(
            batch.metadata().source(),
            batch.metadata().sequence(),
            [("case".into(), serde_json::json!(case))].into(),
        )
        .unwrap(),
    )
    .unwrap()
}

pub(super) async fn transaction(
    plan: &StreamExecutionPlan,
    root: &std::path::Path,
) -> Arc<crate::state::ManifestTransaction> {
    let key = StateLineageKey::new(plan.name(), plan.fingerprint()).unwrap();
    let backend = LocalStateBackend::new(root.join("state")).await.unwrap();
    Arc::new(
        crate::state::ManifestTransaction::open(
            Arc::from(backend.open_lineage(&key).await.unwrap()),
            &key,
            root.join("manifests"),
            2,
        )
        .await
        .unwrap(),
    )
}

pub(super) async fn capture(
    transaction: &Arc<crate::state::ManifestTransaction>,
    fallback: bool,
    case: &str,
) -> (
    BTreeMap<String, crate::OperatorManifestEntry>,
    BatchMetadata,
) {
    let parts = plan("trades", !fallback)
        .into_runtime_parts(EdgeBudget::default())
        .unwrap();
    let mut before =
        harness_with_state(parts, false, BTreeMap::new(), Some(transaction.clone())).await;
    before.runtime.data_gate.send(true).unwrap();
    before
        .send(StreamMessage::data(tagged(
            &test_input(fallback, 0, &[2, 5]),
            case,
        )))
        .await;
    let first = before.receive().await;
    assert_eq!(first.as_data().unwrap().num_rows(), 1);
    let metadata = first.as_data().unwrap().metadata().clone();
    before
        .send(StreamMessage::watermark(EventTime::from_micros(2)))
        .await;
    before.send(StreamMessage::barrier(Epoch::INITIAL)).await;
    assert_eq!(
        before.receive().await.as_watermark(),
        Some(EventTime::from_micros(2))
    );
    assert_eq!(before.receive().await.as_barrier(), Some(Epoch::INITIAL));
    let mut entries = BTreeMap::new();
    for _ in 0..2 {
        let ack = before.acks.recv().await.unwrap();
        assert_eq!(ack.epoch, Epoch::INITIAL);
        entries.insert(ack.node_id, ack.state);
    }
    assert_eq!(
        entries.keys().map(String::as_str).collect::<Vec<_>>(),
        [HEAD, TAIL]
    );
    assert_eq!(
        entries[TAIL].inline_metadata["state_layout"],
        serde_json::json!(if fallback { 4 } else { 3 })
    );
    before.cancellation.cancel();
    before.finish().await;
    (entries, metadata)
}

pub(super) async fn restores(
    transaction: &crate::state::ManifestTransaction,
    entries: &BTreeMap<String, crate::OperatorManifestEntry>,
    parts: &StreamRuntimePlanParts,
) -> BTreeMap<String, OperatorRestoreState> {
    let mut restored = BTreeMap::new();
    for node in &parts.nodes {
        let entry = &entries[&node.node_id];
        let snapshot = transaction
            .load_operator_state(&node.node_id, entry)
            .await
            .unwrap();
        restored.insert(
            node.node_id.clone(),
            OperatorRestoreState {
                snapshot: node
                    .checkpoint_capability
                    .decode_snapshot(&node.node_id, snapshot)
                    .unwrap(),
                progress: entry.progress.clone(),
                output_frontier: None,
                next_epoch: Epoch::INITIAL.next().unwrap(),
            },
        );
    }
    restored
}

async fn terminal_checkpoint(harness: &mut Harness) {
    let epoch = Epoch::INITIAL.next().unwrap();
    for _ in 0..2 {
        let node = harness.terminal_ready.recv().await.unwrap();
        harness.terminal_commands[&node]
            .send(OperatorCheckpointCommand::Terminal(epoch))
            .await
            .unwrap();
    }
    for _ in 0..2 {
        let ack = harness.acks.recv().await.unwrap();
        assert_eq!(ack.epoch, epoch);
        assert_eq!(
            ack.state.progress.values().next().unwrap().state,
            crate::ManifestIngressState::Ended
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn p1b_unfused_native_and_retained_checkpoints_restore_into_the_sql_group() {
    for fallback in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let plan = plan("trades", !fallback);
        let transaction = transaction(&plan, directory.path()).await;
        let (entries, _) = capture(&transaction, fallback, &format!("p1b-resume-{fallback}")).await;
        let parts = plan.into_runtime_parts(EdgeBudget::default()).unwrap();
        let restored = restores(&transaction, &entries, &parts).await;
        let mut after = harness_with_state(parts, true, restored, Some(transaction)).await;
        assert_eq!(after.core.sql_recovery.live_counts(), (0, 0));
        after.runtime.data_gate.send(true).unwrap();
        after
            .send(StreamMessage::data(test_input(fallback, 1, &[3])))
            .await;
        let message = after.receive().await;
        let table = message.as_data().unwrap().table_payload().unwrap();
        assert_eq!(
            table.batches()[0]
                .column_by_name("position")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            0
        );
        after
            .send(StreamMessage::watermark(EventTime::from_micros(3)))
            .await;
        after.send(StreamMessage::idle()).await;
        after.send(StreamMessage::end_of_input()).await;
        assert_eq!(
            after.receive().await.as_watermark(),
            Some(EventTime::from_micros(3))
        );
        assert_eq!(after.receive().await.kind(), crate::StreamMessageKind::Idle);
        assert!(after.receive().await.is_end_of_input());
        terminal_checkpoint(&mut after).await;
        after.finish().await;
        assert!(
            after
                .core
                .metrics
                .snapshot()
                .edges
                .values()
                .all(|edge| edge.channel.charged_rows == 0 && edge.channel.charged_bytes == 0)
        );
    }
}

use super::*;

#[path = "rebase_trust_tests.rs"]
mod trust_tests;

#[path = "rebase_trust_capture_tests.rs"]
mod trust_capture_tests;

#[path = "rebase_edge_tests.rs"]
mod edge_tests;

#[path = "rebase_owner_tests.rs"]
mod owner_tests;

async fn admit_left(join: &mut StreamJoinOperator, context: &StreamOperatorContext<'_>) {
    let mut output = EdgeCollector::new(join.output_ports().to_vec());
    join.process_data(
        "left",
        Batch::table(vec![record("left")], BatchMetadata::default()).unwrap(),
        context,
        &mut output,
    )
    .await
    .unwrap();
}

async fn checkpointed(
    context: &StreamOperatorContext<'_>,
) -> (StreamJoinOperator, OperatorStateSnapshot) {
    let mut join = seeded(context).await;
    let mut snapshot = join.checkpoint(Epoch::INITIAL).unwrap();
    for epoch in 2..=4 {
        admit_left(&mut join, context).await;
        snapshot = join.checkpoint(Epoch::new(epoch).unwrap()).unwrap();
    }
    assert!(join.state.deltas.pending.is_empty());
    assert_eq!(join.v2_writer.dirty_epochs, 4);
    (join, snapshot)
}

fn count_base(join: &mut StreamJoinOperator) -> Arc<AtomicUsize> {
    let count = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&count);
    join.set_checkpoint_writer_base_test_hook(Arc::new(move |credit| {
        assert!(credit.size() > 0);
        observed.fetch_add(1, Ordering::SeqCst);
    }));
    count
}

fn assert_same_live(expected: &StreamJoinOperator, actual: &StreamJoinOperator) {
    assert_eq!(actual.status(), expected.status());
    assert_eq!(
        (
            actual.state.next_left_row_id,
            actual.state.next_right_row_id,
            actual.state.next_output_sequence,
            actual.state.ended,
        ),
        (
            expected.state.next_left_row_id,
            expected.state.next_right_row_id,
            expected.state.next_output_sequence,
            expected.state.ended,
        )
    );
    for (actual, expected) in actual
        .state
        .left
        .iter()
        .chain(actual.state.right.iter())
        .zip(
            expected
                .state
                .left
                .iter()
                .chain(expected.state.right.iter()),
        )
    {
        assert_eq!(actual.row_id, expected.row_id);
        assert_eq!(actual.event_time, expected.event_time);
        assert_eq!(actual.charge, expected.charge);
        assert_eq!(
            actual.encoded_key.as_slice(),
            expected.encoded_key.as_slice()
        );
        assert_eq!(*actual.record.view(), *expected.record.view());
    }
}

#[tokio::test]
async fn test_automatic_rebase_preserves_whole_payload_owners_and_current_state() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "prepared-reuse", None);
    let (mut join, old) = checkpointed(&context).await;
    let old_bytes = old
        .segments
        .iter()
        .map(|(name, segment)| (name.clone(), segment.bytes().to_vec()))
        .collect::<std::collections::BTreeMap<_, _>>();
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    let bases = count_base(&mut join);
    let previous = join.status();
    let rows = Arc::as_ptr(&join.state.left.0);
    join.prepare_checkpoint_async(&context).await.unwrap();
    assert_eq!(
        bases.load(Ordering::SeqCst),
        0,
        "eligible captured history must not run the full payload encoder"
    );
    assert_eq!(join.status(), previous);
    assert_eq!(Arc::as_ptr(&join.state.left.0), rows);
    assert_eq!(join.state.last_checkpoint_epoch, Epoch::new(4));
    let captured = join.checkpoint(Epoch::new(5).unwrap()).unwrap();
    assert_eq!(captured.inline_metadata["v2_inventory"]["base_epoch"], 4);
    assert_eq!(
        captured.inline_metadata["v2_inventory"]["deltas"],
        serde_json::json!([])
    );
    assert_eq!(
        captured.inline_metadata["v2_inventory"]["payloads"],
        old.inline_metadata["v2_inventory"]["payloads"]
    );
    let mut payloads = 0;
    for (name, segment) in &old.segments {
        assert_eq!(segment.bytes(), old_bytes[name]);
        if name.contains("-payload-") {
            payloads += 1;
            assert_eq!(segment, &captured.segments[name]);
            assert!(Arc::ptr_eq(
                &segment.bytes_arc(),
                &captured.segments[name].bytes_arc()
            ));
        }
    }
    assert_eq!(captured.segments.len(), payloads + 2);
    let mut restored = operator();
    restored.restore(&captured).unwrap();
    assert_same_live(&join, &restored);
    assert_eq!(restored.state.last_checkpoint_epoch, Epoch::new(5));
    drop((join, restored, old, captured));
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_rebase_continuation_matches_strictly_restored_outputs_and_state() {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let (mut join, old) = checkpointed(&context).await;
    join.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = join.checkpoint(Epoch::new(5).unwrap()).unwrap();
    let mut restored = operator();
    restored.restore(&snapshot).unwrap();
    let mut outputs = Vec::new();
    for operator in [&mut join, &mut restored] {
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "right",
                Batch::table(vec![record("right")], BatchMetadata::default()).unwrap(),
                &context,
                &mut output,
            )
            .await
            .unwrap();
        let messages = output.drain("output");
        assert!(!messages.is_empty());
        outputs.push(
            messages
                .iter()
                .map(|message| {
                    let data = message.as_data().unwrap();
                    (
                        data.metadata().sequence(),
                        data.table_payload().unwrap().batches().to_vec(),
                    )
                })
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(outputs[0], outputs[1]);
    assert_same_live(&join, &restored);
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    drop((join, restored, old, snapshot, outputs));
    drop(context);
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    drop(enabled);
    assert_eq!(pool.reserved(), 0);
}

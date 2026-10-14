use super::*;
use crate::{EventTime, IngressProgress, IngressProgressSnapshot, IngressState};
use std::collections::BTreeMap;

fn progressed(job: &StreamJobContext) -> StreamOperatorContext<'_> {
    StreamOperatorContext::with_ingress_progress(
        job,
        "prepared-reuse",
        None,
        IngressProgressSnapshot::new(BTreeMap::from([
            (
                "left".into(),
                IngressProgress::new(IngressState::Active, None),
            ),
            (
                "right".into(),
                IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1_000))),
            ),
        ])),
    )
}

async fn replace_left(join: &mut StreamJoinOperator, context: &StreamOperatorContext<'_>) {
    let original = record("left");
    let replacement = RecordBatch::try_new(
        original.schema(),
        vec![
            Arc::clone(original.column(0)),
            Arc::new(TimestampMicrosecondArray::from(vec![1_095, 1_096])),
            Arc::clone(original.column(2)),
        ],
    )
    .unwrap();
    let mut output = EdgeCollector::new(join.output_ports().to_vec());
    for _ in 0..4 {
        join.process_data(
            "left",
            Batch::table(vec![replacement.clone()], BatchMetadata::default()).unwrap(),
            context,
            &mut output,
        )
        .await
        .unwrap();
    }
    assert!(output.drain("output").is_empty());
}

#[tokio::test]
async fn test_off_same_charge_replacement_invalidates_history_until_full_rebuild() {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let (mut join, old) = checkpointed(&context).await;
    assert!(join.v2_writer.history_trusted);
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    let cut = join
        .v2_writer
        .tracker
        .snapshot(join.state.last_checkpoint_epoch);
    let before = join.status();
    let disabled = job().with_checkpointing(false);
    let off = StreamOperatorContext::new(&disabled, "prepared-reuse", None);
    join.on_ingress_progress("right", &progressed(&disabled))
        .await
        .unwrap();
    assert!(join.state.left.is_empty());
    assert!(!join.v2_writer.history_trusted);
    replace_left(&mut join, &off).await;
    assert_eq!(join.status().left.retained_rows, before.left.retained_rows);
    assert_eq!(
        join.status().left.retained_bytes,
        before.left.retained_bytes
    );
    assert_ne!(
        join.v2_writer
            .tracker
            .snapshot(join.state.last_checkpoint_epoch),
        cut
    );
    assert!(join.state.deltas.pending.is_empty());
    assert_eq!(join.state.left[0].event_time.as_micros(), 1_095);
    assert_eq!(join.state.left[0].row_id, 8);
    assert!(join.checkpoint(Epoch::new(5).unwrap()).is_err());
    let bases = count_base(&mut join);
    join.prepare_checkpoint_async(&context).await.unwrap();
    assert_eq!(bases.load(Ordering::SeqCst), 1);
    assert!(join.v2_writer.history_trusted);
    let captured = join.checkpoint(Epoch::new(5).unwrap()).unwrap();
    let mut restored = operator();
    restored.restore(&captured).unwrap();
    assert_same_live(&join, &restored);
    admit_left(&mut join, &context).await;
    join.prepare_checkpoint_async(&context).await.unwrap();
    assert!(join.v2_writer.prepared_delta.is_some());
    assert_eq!(bases.load(Ordering::SeqCst), 1);
    let delta = join.checkpoint(Epoch::new(6).unwrap()).unwrap();
    restored.restore(&delta).unwrap();
    assert_same_live(&join, &restored);
    drop((old, captured, delta, join, restored));
    drop((context, off));
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    assert!(disabled.gather_owner().close_and_drain().await.is_empty());
    drop((enabled, disabled));
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_emitted_counter_overflow_is_refused_before_emission() {
    for (checkpointing, projected) in [(false, false), (true, true)] {
        Box::pin(emitted_overflow(checkpointing, projected)).await;
    }
}

async fn emitted_overflow(checkpointing: bool, projected: bool) {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let mut candidate = operator();
    if projected {
        candidate.set_output_projection(vec![2]).unwrap();
    }
    let (mut join, old) = checkpointed_from(&context, candidate).await;
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    let tracker = join.v2_writer.tracker;
    let sequence = join.state.next_output_sequence;
    join.state.metrics.emitted_match_rows = u64::MAX;
    let before = join.status();
    let disabled = job().with_checkpointing(false);
    let off = StreamOperatorContext::new(&disabled, "prepared-reuse", None);
    let mut output = EdgeCollector::new(join.output_ports().to_vec());
    let failure = join
        .process_data(
            "right",
            Batch::table(vec![record("right")], BatchMetadata::default()).unwrap(),
            if checkpointing { &context } else { &off },
            &mut output,
        )
        .await
        .unwrap_err();
    assert!(failure.to_string().contains("emitted_match_rows"));
    let actual = (
        join.status(),
        join.state.next_output_sequence,
        output.drain("output").is_empty(),
    );
    assert!(join.state.deltas.pending.is_empty());
    assert_ne!(join.v2_writer.tracker, tracker);
    assert_eq!(join.v2_writer.history_trusted, checkpointing);
    drop((old, join, output));
    drop((context, off));
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    assert!(disabled.gather_owner().close_and_drain().await.is_empty());
    drop((enabled, disabled));
    assert_eq!(pool.reserved(), 0);
    assert_eq!(
        actual,
        (before, sequence, true),
        "counter preflight must precede all output"
    );
}

#[tokio::test]
async fn test_off_end_invalidates_history_until_reset_or_validated_restore() {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let (mut join, old) = checkpointed(&context).await;
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    let tracker = join.v2_writer.tracker;
    let disabled = job().with_checkpointing(false);
    let off = StreamOperatorContext::new(&disabled, "prepared-reuse", None);
    let mut output = EdgeCollector::new(join.output_ports().to_vec());
    join.on_end(&off, &mut output).await.unwrap();
    assert!(join.state.ended);
    assert!(join.state.left.is_empty());
    assert!(join.state.right.is_empty());
    assert!(join.state.deltas.pending.is_empty());
    assert_ne!(join.v2_writer.tracker, tracker);
    assert!(!join.v2_writer.history_trusted);
    join.restore(&old).unwrap();
    assert!(join.v2_writer.history_trusted);
    join.on_end(&off, &mut output).await.unwrap();
    assert!(!join.v2_writer.history_trusted);
    join.reset().unwrap();
    assert!(join.v2_writer.history_trusted);
    assert!(join.state.left.is_empty());
    drop((old, join, output));
    drop((context, off));
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    assert!(disabled.gather_owner().close_and_drain().await.is_empty());
    drop((enabled, disabled));
    assert_eq!(pool.reserved(), 0);
}

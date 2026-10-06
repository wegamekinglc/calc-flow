use super::*;
use crate::runtime::streaming::gather_work::admission_probe::{AdmissionProbe, AdmissionStage};

const LIMIT: usize = 64 << 20;

#[tokio::test(flavor = "current_thread")]
async fn gather_admission_attempt_budget_retries_real_output_prefix() {
    prefix_retry(AdmissionStage::Attempt).await;
}

#[tokio::test(flavor = "current_thread")]
async fn gather_admission_generation_budget_retries_real_output_prefix() {
    prefix_retry(AdmissionStage::Generation).await;
}

#[tokio::test(flavor = "current_thread")]
async fn gather_admission_single_row_budget_preserves_state_and_domain() {
    let (mut op, job, mut output) = admitted(1).await;
    let before = op.capture(Epoch::INITIAL).unwrap();
    let rows = op.status.pending_left_rows;
    let sequence = op.next_output_sequence;
    let probe = AdmissionProbe::install(
        job.gather_owner(),
        AdmissionStage::Attempt,
        op.runtime.pool.clone(),
        LIMIT,
    );
    let context = progressed(&job);
    let result = op
        .on_watermark(EventTime::from_micros(105), &context, &mut output)
        .await;
    let event = probe.take_event();
    job.gather_owner().close_and_drain().await;
    let no_output = output
        .drain("output")
        .iter()
        .all(|message| message.as_data().is_none());
    let state_unchanged =
        op.status.pending_left_rows == rows && op.next_output_sequence == sequence;
    let after = op.capture(Epoch::INITIAL).unwrap();
    assert!(
        matches!(
            result,
            Err(CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded
                    | StreamingFailureReason::AsofOutputLimitExceeded,
                ..
            })
        ),
        "real one-row native fee denial lost ASOF error domain: {result:?}"
    );
    assert!(no_output && state_unchanged);
    assert_eq!(before.segments, after.segments);
    check_event(event, AdmissionStage::Attempt);
}

async fn prefix_retry(stage: AdmissionStage) {
    let (mut op, job, mut output) = admitted(4).await;
    let probe = AdmissionProbe::install(job.gather_owner(), stage, op.runtime.pool.clone(), LIMIT);
    let context = progressed(&job);
    let result = op
        .on_watermark(EventTime::from_micros(105), &context, &mut output)
        .await;
    let event = probe.take_event();
    job.gather_owner().close_and_drain().await;
    let batches = output
        .drain("output")
        .into_iter()
        .filter_map(|message| message.as_data().cloned())
        .collect::<Vec<_>>();
    assert!(
        result.is_ok(),
        "real native fee denial stopped adaptive ASOF prefix retry: {result:?}"
    );
    check_event(event, stage);
    assert_eq!(
        batches.iter().map(Batch::num_rows).collect::<Vec<_>>(),
        vec![2, 2]
    );
    let left = sequences(&batches, "left__seq");
    let right = sequences(&batches, "right__seq");
    assert_eq!(left, vec![Some(1), Some(2), Some(3), Some(4)]);
    assert_eq!(right, vec![Some(10); 4]);
    assert_eq!(op.status.pending_left_rows, 0);
    assert_eq!(op.next_output_sequence, 4);
}

fn check_event(
    event: Option<crate::runtime::streaming::gather_work::admission_probe::AdmissionEvent>,
    stage: AdmissionStage,
) {
    let event = event.expect("actual MemoryReservation fee boundary was not exercised");
    assert_eq!(event.stage, stage);
    assert_eq!(event.operator, "operator:budget-asof");
    assert!(
        event.task.is_none(),
        "direct callback has no managed TaskId"
    );
    assert_eq!(event.available + 1, event.fee);
}

async fn admitted(rows: i64) -> (StreamAsofJoinOperator, StreamJobContext, EdgeCollector) {
    let (template, _) = fixture();
    let schema = template.schemas[0].clone();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        Duration::from_micros(10),
        AsofStateLimits::new(1000, LIMIT as u64).unwrap(),
    )
    .unwrap();
    let mut op =
        StreamAsofJoinOperator::new("budget-asof", schema.clone(), schema.clone(), spec).unwrap();
    let job = StreamJobContext::new(
        42,
        "budget-asof",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    let context = StreamOperatorContext::new(&job, "budget-asof", None);
    let right = Batch::table(
        vec![indexed_input(&schema, &[("A", 100, 10)])],
        BatchMetadata::default(),
    )
    .unwrap();
    let left = Batch::table(
        vec![indexed_input(
            &schema,
            &(1..=rows)
                .map(|row| ("A", 100 + row, row))
                .collect::<Vec<_>>(),
        )],
        BatchMetadata::default(),
    )
    .unwrap();
    op.process_data("right", right, &context, &mut output)
        .await
        .unwrap();
    op.process_data("left", left, &context, &mut output)
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
    (op, job, output)
}

fn progressed(job: &StreamJobContext) -> StreamOperatorContext<'_> {
    let watermark = EventTime::from_micros(105);
    let progress = IngressProgressSnapshot::new(
        ["left", "right"]
            .map(|side| {
                (
                    side.into(),
                    IngressProgress::new(IngressState::Active, Some(watermark)),
                )
            })
            .into_iter()
            .collect(),
    );
    StreamOperatorContext::with_ingress_progress(job, "budget-asof", Some(watermark), progress)
}

fn sequences(batches: &[Batch], name: &str) -> Vec<Option<i64>> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .table_payload()
                .unwrap()
                .batches()
                .iter()
                .flat_map(|record| {
                    record
                        .column_by_name(name)
                        .unwrap()
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .iter()
                })
        })
        .collect()
}

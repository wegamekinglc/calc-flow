use super::{Harness, harness_with_operator_capability, start};
use crate::{
    Batch, BatchMetadata, EventTime, JoinStateLimits, JoinTimeBounds, OperatorMetadata,
    StreamJoinOperator, StreamJoinSpec, StreamMessage,
    pipeline::{CompiledStreamOperator, OperatorCheckpointCapability},
};
use datafusion::arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use std::{sync::Arc, time::Duration};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, true),
        Field::new("ts", DataType::Timestamp(TimeUnit::Millisecond, None), true),
    ]))
}

fn batch(key: i64, rows: usize) -> Batch {
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(vec![key; rows])),
            Arc::new(datafusion::arrow::array::TimestampMillisecondArray::from(
                vec![1; rows],
            )),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn operator() -> StreamJoinOperator {
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "ts",
        "ts",
        JoinTimeBounds::new(Duration::from_micros(10), Duration::from_micros(10)).unwrap(),
        JoinStateLimits::new(1000, 1_000_000, 1000).unwrap(),
    )
    .unwrap();
    StreamJoinOperator::new("node", schema(), schema(), spec).unwrap()
}

async fn fixture() -> Harness {
    let operator = operator();
    let port = operator.output_ports()[0].clone();
    let mut harness = harness_with_operator_capability(
        &["left", "right"],
        1,
        CompiledStreamOperator::StreamJoin(Box::new(operator)),
        OperatorCheckpointCapability::CheckpointedStateful { state_version: 1 },
        port,
        None,
        None,
    );
    start(&mut harness).await;
    harness
}

async fn wait_for(
    harness: &Harness,
    predicate: impl Fn(&super::super::OperatorProgressSnapshot) -> bool,
) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !predicate(&harness.progress.snapshot()) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("operator must reach the observed boundary");
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

async fn stop(mut harness: Harness) {
    harness.cancellation.cancel();
    let report = tokio::time::timeout(Duration::from_secs(5), harness.supervisor.join_all())
        .await
        .unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    let progress = harness.progress.snapshot();
    assert_eq!(progress.preparation_tasks_started, 2);
    assert_eq!(progress.preparation_tasks_exited, 2);
    let gather = harness.gather.clone();
    drop(harness);
    assert!(
        tokio::time::timeout(Duration::from_secs(5), gather.close_and_drain())
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn test_ingress_admission_joins_workers_waiting_for_input() {
    let harness = fixture().await;
    wait_for(&harness, |p| p.preparation_tasks_started == 2).await;
    stop(harness).await;
}

#[tokio::test]
async fn test_ingress_admission_overlaps_blocked_emission_without_publication() {
    let mut harness = fixture().await;
    send(&mut harness, "right", StreamMessage::data(batch(1, 1))).await;
    wait_for(&harness, |p| {
        p.stream_join.as_ref().unwrap().right.retained_rows == 1
    })
    .await;
    send(&mut harness, "left", StreamMessage::data(batch(1, 1))).await;
    wait_for(&harness, |p| p.fully_fanned_out_batches == 1).await;
    send(&mut harness, "left", StreamMessage::data(batch(1, 64))).await;
    wait_for(&harness, |p| p.input_batches == 3).await;
    send(&mut harness, "right", StreamMessage::data(batch(2, 1))).await;
    wait_for(&harness, |p| p.prepared_ingresses.get("right") == Some(&2)).await;
    let progress = harness.progress.snapshot();
    let status = progress.stream_join.unwrap();
    assert_eq!(status.left.retained_rows, 1);
    assert_eq!(status.right.retained_rows, 1);
    assert_eq!(
        progress.input_batches, 3,
        "prepared opposite input is not installed"
    );
    assert_eq!(progress.fully_fanned_out_batches, 1);
    stop(harness).await;
}

#[tokio::test]
async fn test_ingress_admission_sliced_records_match_serial_output_status_and_fees() {
    use crate::{
        CancellationToken, EdgeCollector, JsonMap, StreamJobContext, StreamOperator,
        StreamOperatorContext,
    };
    use datafusion::arrow::array::TimestampMillisecondArray;

    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(
                (0..65)
                    .map(|row| (row != 2).then_some(1))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(TimestampMillisecondArray::from(
                (0..65)
                    .map(|row| (row != 3).then_some(1))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    let input = Batch::table(
        vec![record.slice(1, 32), record.slice(33, 32)],
        BatchMetadata::default(),
    )
    .unwrap();
    let original = input.table_payload().unwrap().batches().to_vec();
    let mut serial = operator();
    let job = StreamJobContext::new(1, "serial", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "node", None);
    let mut output = EdgeCollector::new(serial.output_ports().to_vec());
    serial
        .process_data("right", batch(1, 1), &context, &mut output)
        .await
        .unwrap();
    serial
        .process_data("left", input.clone(), &context, &mut output)
        .await
        .unwrap();

    let mut harness = fixture().await;
    send(&mut harness, "right", StreamMessage::data(batch(1, 1))).await;
    wait_for(&harness, |p| {
        p.stream_join.as_ref().unwrap().right.retained_rows == 1
    })
    .await;
    send(&mut harness, "left", StreamMessage::data(input.clone())).await;
    wait_for(&harness, |p| {
        p.stream_join.as_ref().unwrap().left.retained_rows == 62
    })
    .await;
    assert_eq!(
        harness.progress.snapshot().stream_join.unwrap(),
        serial.status()
    );
    let actual = harness.outputs[0].recv().await.unwrap().unwrap();
    let expected = output.drain("output");
    assert_eq!(
        actual.as_data().unwrap().table_payload().unwrap().batches(),
        expected[0]
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()
    );
    assert_eq!(input.table_payload().unwrap().batches(), original);
    drop((actual, expected, output, serial));
    stop(harness).await;
    assert!(
        tokio::time::timeout(Duration::from_secs(5), job.gather_owner().close_and_drain())
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn test_ingress_admission_reads_one_envelope_ahead_and_joins_on_cancel() {
    let mut harness = fixture().await;
    send(&mut harness, "right", StreamMessage::data(batch(1, 1))).await;
    wait_for(&harness, |p| p.input_batches == 1).await;
    send(&mut harness, "left", StreamMessage::data(batch(1, 1))).await;
    wait_for(&harness, |p| p.fully_fanned_out_batches == 1).await;
    send(&mut harness, "left", StreamMessage::data(batch(1, 64))).await;
    wait_for(&harness, |p| p.input_batches == 3).await;
    send(&mut harness, "right", StreamMessage::data(batch(2, 1))).await;
    wait_for(&harness, |p| p.prepared_ingresses.get("right") == Some(&2)).await;
    send(&mut harness, "right", StreamMessage::data(batch(3, 64))).await;
    let mut third = Box::pin(
        harness
            .inputs
            .get_mut("right")
            .unwrap()
            .send(StreamMessage::data(batch(4, 1))),
    );
    assert!(
        futures::poll!(third.as_mut()).is_pending(),
        "next envelope must retain edge backpressure"
    );
    drop(third);
    assert_eq!(harness.progress.snapshot().prepared_ingresses["right"], 2);
    stop(harness).await;
}

#[tokio::test]
async fn test_ingress_admission_defers_opposite_error_until_install_and_joins_after_failed_emission()
 {
    let mut harness = fixture().await;
    send(&mut harness, "right", StreamMessage::data(batch(1, 1))).await;
    wait_for(&harness, |p| p.input_batches == 1).await;
    send(&mut harness, "left", StreamMessage::data(batch(1, 1))).await;
    wait_for(&harness, |p| p.fully_fanned_out_batches == 1).await;
    send(&mut harness, "left", StreamMessage::data(batch(1, 64))).await;
    wait_for(&harness, |p| p.input_batches == 3).await;
    let invalid = Batch::table(
        vec![RecordBatch::new_empty(Arc::new(Schema::empty()))],
        BatchMetadata::default(),
    )
    .unwrap();
    send(&mut harness, "right", StreamMessage::data(invalid)).await;
    wait_for(&harness, |p| p.prepared_ingresses.get("right") == Some(&2)).await;
    assert_eq!(harness.progress.snapshot().input_batches, 3);
    harness.outputs.clear();
    let report = tokio::time::timeout(Duration::from_secs(5), harness.supervisor.join_all())
        .await
        .unwrap();
    assert_eq!(report.errors.len(), 1);
    assert!(
        matches!(&report.errors[0].error, crate::CalcFlowError::EdgeClosed { edge } if edge == "output->0"),
        "{report:?}"
    );
    let progress = harness.progress.snapshot();
    assert_eq!(progress.stream_join.unwrap().left.retained_rows, 1);
    assert_eq!(progress.preparation_tasks_exited, 2);
    stop(harness).await;
}

#[tokio::test]
async fn test_ingress_admission_uses_current_opposite_watermark_for_retention() {
    let mut harness = fixture().await;
    send(
        &mut harness,
        "right",
        StreamMessage::watermark(EventTime::from_micros(2000)),
    )
    .await;
    // A watermark on only one side produces no output, so observe its callback.
    wait_for(&harness, |p| {
        p.stream_join.as_ref().unwrap().right.watermark_micros == Some(EventTime::from_micros(2000))
    })
    .await;
    send(&mut harness, "left", StreamMessage::data(batch(1, 1))).await;
    wait_for(&harness, |p| p.input_batches == 1).await;
    send(&mut harness, "left", StreamMessage::end_of_input()).await;
    send(&mut harness, "right", StreamMessage::end_of_input()).await;
    let report = tokio::time::timeout(Duration::from_secs(5), harness.supervisor.join_all())
        .await
        .unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    let progress = harness.progress.snapshot();
    assert_eq!(progress.stream_join.unwrap().left.retained_rows, 0);
    assert_eq!(progress.preparation_tasks_exited, 2);
    loop {
        let message = harness.outputs[0].recv().await.unwrap().unwrap();
        assert!(message.as_data().is_none());
        if message.is_end_of_input() {
            break;
        }
    }
    stop(harness).await;
}

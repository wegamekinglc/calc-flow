use super::*;
use crate::{
    EdgeCollector, Epoch, OperatorMetadata, StreamOperator,
    operator::ingress::{Preparation, Sender},
};

async fn prepared_input(
    operator: &mut StreamAsofJoinOperator,
    side: &str,
    input: &Batch,
    watermark: Option<crate::EventTime>,
    context: &StreamOperatorContext<'_>,
    output: &mut EdgeCollector,
) -> Result<()> {
    let facts = operator.ingress_plan(side)?.prepare(input, watermark).await;
    let (sender, preparation) = Preparation::asof_channel();
    let Sender::Asof(sender) = sender else {
        unreachable!()
    };
    sender
        .send(facts)
        .await
        .unwrap_or_else(|_| panic!("live fact receiver"));
    drop(sender);
    let context = StreamOperatorContext::new(context.job(), "asof", None)
        .with_ingress_preparation(Some(&preparation));
    operator
        .process_data(side, input.clone(), &context, output)
        .await
}

async fn assert_parity(input: Batch, watermark: Option<i64>) {
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut serial = shaped_fixture(
        &schema,
        &["key".into()],
        &["seq".into()],
        AsofLatePolicy::Drop,
    );
    let mut pipeline = shaped_fixture(
        &schema,
        &["key".into()],
        &["seq".into()],
        AsofLatePolicy::Drop,
    );
    for operator in [&mut serial, &mut pipeline] {
        operator.status.left.watermark_micros = watermark.map(crate::EventTime::from_micros);
        operator.status.right.watermark_micros = watermark.map(crate::EventTime::from_micros);
    }
    let original = input.table_payload().unwrap().batches().to_vec();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut expected = EdgeCollector::new(serial.output_ports().to_vec());
    let mut actual = EdgeCollector::new(pipeline.output_ports().to_vec());
    for side in ["right", "left"] {
        serial
            .process_data(side, input.clone(), &context, &mut expected)
            .await
            .unwrap();
        prepared_input(
            &mut pipeline,
            side,
            &input,
            watermark.map(crate::EventTime::from_micros),
            &context,
            &mut actual,
        )
        .await
        .unwrap();
        assert_eq!(serial.status, pipeline.status);
    }
    let snapshot = serial.capture(Epoch::INITIAL).unwrap();
    let prepared = pipeline.capture(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata, prepared.inline_metadata);
    assert_eq!(snapshot.segments, prepared.segments);
    pipeline.reset().unwrap();
    pipeline.restore(&prepared).unwrap();
    serial.reset().unwrap();
    serial.restore(&snapshot).unwrap();
    serial.on_end(&context, &mut expected).await.unwrap();
    pipeline.on_end(&context, &mut actual).await.unwrap();
    let records = |output: &mut EdgeCollector| {
        output
            .drain("output")
            .iter()
            .filter_map(crate::StreamMessage::as_data)
            .flat_map(|batch| batch.table_payload().unwrap().batches().iter().cloned())
            .collect::<Vec<_>>()
    };
    assert_eq!(records(&mut expected), records(&mut actual));
    assert_eq!(input.table_payload().unwrap().batches(), original);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}

#[tokio::test]
async fn test_ingress_admission_asof_matches_serial_checkpoint_and_continuation() {
    let input = direct::string_unsigned_input(4097);
    let records = input.table_payload().unwrap().batches();
    assert_parity(
        batch(vec![
            records[0].slice(0, 0),
            records[0].slice(1, 2047),
            records[1].clone(),
            records[2].clone(),
        ]),
        None,
    )
    .await;
    assert_parity(key_batch(129, 64, true), None).await;
    assert_parity(key_batch(129, 64, false), Some(10)).await;
}

#[tokio::test]
async fn test_ingress_admission_asof_rechecks_stale_watermark_and_resident_identity() {
    let input = direct::string_unsigned_input(64);
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut operator = shaped_fixture(
        &schema,
        &["key".into()],
        &["seq".into()],
        AsofLatePolicy::Drop,
    );
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    // A proof of internal order cannot authorize appending a duplicate to state.
    prepared_input(&mut operator, "left", &input, None, &context, &mut output)
        .await
        .unwrap();
    let before = operator.status.state_rows;
    let error = prepared_input(&mut operator, "left", &input, None, &context, &mut output)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        crate::CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofDuplicateIdentity,
            ..
        }
    ));
    assert_eq!(operator.status.state_rows, before);
    // Prepared classification is only usable at the watermark that produced it.
    operator.status.right.watermark_micros = Some(crate::EventTime::from_micros(1));
    prepared_input(&mut operator, "right", &input, None, &context, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.status.right.accepted_rows, 0);
    assert_eq!(operator.status.right.late_rows, 64);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}

#[tokio::test]
async fn test_ingress_admission_asof_keeps_schema_and_protocol_error_priority() {
    let declared = schema(DataType::Utf8, DataType::UInt64);
    let schema = Arc::new(Schema::new(
        declared
            .fields()
            .iter()
            .map(|field| field.as_ref().clone().with_nullable(true))
            .collect::<Vec<_>>(),
    ));
    let input = batch(vec![record(
        &schema,
        Arc::new(StringArray::from(vec![None, Some("a")])),
        Arc::new(UInt64Array::from(vec![0, 1])),
        vec![0, 0],
    )]);
    let mut operator = fixture(&declared);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let error = prepared_input(&mut operator, "left", &input, None, &context, &mut output)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        crate::CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofInvalidInput,
            ..
        }
    ));
    assert_eq!(operator.status.state_rows, 0);
    operator.terminal = true;
    let valid = direct::string_unsigned_input(2);
    let error = prepared_input(&mut operator, "left", &valid, None, &context, &mut output)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        crate::CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofProtocolError,
            ..
        }
    ));
    assert_eq!(operator.runtime.pool.reserved(), 0);
}

#[tokio::test]
async fn test_ingress_admission_cancelled_fact_wait_keeps_cancellation_and_state_atomic() {
    let input = direct::string_unsigned_input(2);
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut operator = fixture(&schema);
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, cancellation.clone());
    let (sender, preparation) = Preparation::asof_channel();
    let context =
        StreamOperatorContext::new(&job, "asof", None).with_ingress_preparation(Some(&preparation));
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let mut pending = Box::pin(operator.process_data("left", input, &context, &mut output));
    assert!(futures::poll!(pending.as_mut()).is_pending());
    cancellation.cancel();
    drop(sender);
    assert!(matches!(
        pending.await,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(operator.status.state_rows, 0);
    assert_eq!(operator.status.left.accepted_rows, 0);
    assert_eq!(operator.runtime.pool.reserved(), 0);
    assert!(output.drain("output").is_empty());
}

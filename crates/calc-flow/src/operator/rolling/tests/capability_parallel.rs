use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use datafusion::arrow::array::TimestampMicrosecondArray;

use crate::runtime::streaming::entity_work::JobEntityWorkOwner;

use super::*;

#[tokio::test]
async fn test_capability_parallel_cold_rows37_matches_serial_output_and_state() {
    let mut spec = kernel_spec(json!([aggregate_output("mean", "price", "mean", 37)]));
    spec.numerical_profile = RollingNumericalProfile::StableV1;
    let input = parallel_rollback_record(0, 64_000);
    assert_cold_parallel(spec, input).await;
}

#[tokio::test]
async fn test_capability_parallel_two_entities_execute_nonempty_lanes() {
    let spec = kernel_spec(json!([aggregate_output("mean", "price", "mean", 37)]));
    let input = float64_fast_record(
        &(0..64_000_u32)
            .map(|row| {
                (
                    i64::from(row),
                    if row.is_multiple_of(2) { "a" } else { "b" },
                    u64::from(row),
                    Some(f64::from(row % 101)),
                )
            })
            .collect::<Vec<_>>(),
    );
    assert_cold_parallel(spec, input).await;
}

async fn assert_cold_parallel(spec: RollingSpec, input: RecordBatch) {
    let mut serial = RollingOperator::new("rolling", input.schema(), spec.clone()).unwrap();
    let mut parallel = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    let job = crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let serial_context = StreamOperatorContext::new(&job, "rolling", None);
    let expected = close(&mut serial, &serial_context, input.clone()).await;
    let launches = Arc::new(AtomicU64::new(0));
    let owner = JobEntityWorkOwner::new(7, Arc::clone(&launches));
    owner.set_worker_limit_for_test(2);
    let task = owner.test_client(3, "operator:rolling".into());
    let scope = task.callback_scope();
    let context = StreamOperatorContext::new(&job, "rolling", None)
        .with_entity_work(task.context_client().unwrap());
    let actual = close(&mut parallel, &context, input).await;
    scope.settle_abandoned().await;
    drop(context);
    drop(scope);
    owner.close_admission();
    assert!(owner.drain().await.is_empty());
    assert_eq!(actual, expected);
    let bits = |record: &RecordBatch| {
        record
            .column(6)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .iter()
            .map(|value| value.map(f64::to_bits))
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&actual), bits(&expected));
    assert_eq!(
        parallel
            .state
            .typed_kernel_state
            .as_ref()
            .unwrap()
            .canonical_state_for_test(),
        serial
            .state
            .typed_kernel_state
            .as_ref()
            .unwrap()
            .canonical_state_for_test()
    );
    assert_cached_rolling_charge(&parallel);
    assert_cached_rolling_charge(&serial);
    let epoch = Epoch::new(1).unwrap();
    parallel.checkpoint(epoch).unwrap();
    serial.checkpoint(epoch).unwrap();
    assert_eq!(
        parallel.encode_state(epoch).unwrap(),
        serial.encode_state(epoch).unwrap()
    );
    assert_eq!(parallel.state.charge, serial.state.charge);
    assert_eq!(
        launches.load(AtomicOrdering::SeqCst),
        2,
        "a cold, independently executable Rows37 plan must execute two owned lanes"
    );
}

pub(super) async fn close(
    operator: &mut RollingOperator,
    context: &StreamOperatorContext<'_>,
    input: RecordBatch,
) -> RecordBatch {
    let watermark = input
        .column(0)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap()
        .value(input.num_rows() - 1);
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            Batch::table(vec![input], BatchMetadata::default()).unwrap(),
            context,
            &mut output,
        )
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(watermark), context, &mut output)
        .await
        .unwrap();
    let records = output
        .drain("output")
        .into_iter()
        .flat_map(|message| {
            message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
                .to_vec()
        })
        .collect::<Vec<_>>();
    concat_batches(operator.output_ports()[0].schema().unwrap(), &records).unwrap()
}

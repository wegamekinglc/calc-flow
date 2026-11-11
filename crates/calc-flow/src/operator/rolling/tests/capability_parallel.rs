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

#[tokio::test]
async fn test_capability_parallel_two_input_four_readouts_match_exact_ap_oracle() {
    let names = (0..1_000)
        .map(|key| format!("e{key:04}"))
        .collect::<Vec<_>>();
    let input = float64_pair_record(
        &(0..131_074_u32)
            .map(|seq| {
                let key = seq % 1_000;
                let ordinal = f64::from(seq / 1_000);
                (
                    i64::from(seq) * 1_000,
                    names[key as usize].as_str(),
                    u64::from(seq),
                    Some(2.0 * ordinal + f64::from(key) / 1_024.0),
                    Some(6.0 * ordinal + f64::from(key) / 2_048.0),
                )
            })
            .collect::<Vec<_>>(),
    );
    let input_columns = input.num_columns();
    let spec = kernel_spec(json!([
        aggregate_output("sum", "price", "a_sum", 37),
        aggregate_output("mean", "price", "a_mean", 37),
        aggregate_output("sum", "volume", "b_sum", 61),
        aggregate_output("mean", "volume", "b_mean", 61),
    ]));
    let planned = RollingOperator::new("rolling", input.schema(), spec.clone()).unwrap();
    assert_eq!(planned.compiled.kernel_plan.typed_group_count(), 2);
    assert_eq!(planned.compiled.outputs.len(), 4);
    assert!(
        planned
            .compiled
            .outputs
            .iter()
            .all(|output| output.output_type == DataType::Float64)
    );
    let actual = assert_cold_parallel(spec, input).await;
    for row in 0..131_074_u32 {
        let key = f64::from(row % 1_000);
        let ordinal = row / 1_000;
        let a_count = (ordinal + 1).min(37);
        let b_count = (ordinal + 1).min(61);
        let a_mean = f64::from(2 * (ordinal + 1 - a_count) + a_count - 1) + key / 1_024.0;
        let b_mean = f64::from(6 * (ordinal + 1 - b_count) + 3 * (b_count - 1)) + key / 2_048.0;
        for (column, expected) in [
            f64::from(a_count) * a_mean,
            a_mean,
            f64::from(b_count) * b_mean,
            b_mean,
        ]
        .into_iter()
        .enumerate()
        {
            let column = actual
                .column(input_columns + column)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            assert!(column.is_valid(row as usize));
            assert_eq!(column.value(row as usize).to_bits(), expected.to_bits());
        }
    }
}

async fn assert_cold_parallel(spec: RollingSpec, input: RecordBatch) -> RecordBatch {
    let input_columns = input.num_columns();
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
    for (actual, expected) in actual.columns()[input_columns..]
        .iter()
        .zip(&expected.columns()[input_columns..])
    {
        let bits = |column: &ArrayRef| {
            column
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .iter()
                .map(|value| value.map(f64::to_bits))
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(actual), bits(expected));
    }
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
        "a cold, independently executable plan must execute two owned lanes"
    );
    actual
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

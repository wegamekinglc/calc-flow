use std::sync::atomic::{AtomicU64, Ordering};

use crate::runtime::streaming::entity_work::JobEntityWorkOwner;

use super::*;

#[tokio::test]
async fn test_capability_parallel_stable_v2_mixed_rebases_across_restore() {
    let mut deviation = duration_output("stddev", "price", "time_stddev", 8_192);
    deviation["ddof"] = json!(1);
    let mut spec = kernel_spec(json!([
        aggregate_output("mean", "price", "row_mean", 37),
        duration_output("sum", "price", "time_sum", 4_096),
        ddof_output("variance", "price", "row_variance", 73, 1),
        deviation,
    ]));
    spec.numerical_profile = RollingNumericalProfile::StableV2Preview;
    let names = entity_names();
    let inputs = (0..3)
        .map(|batch| {
            float64_fast_record(
                &(batch * 64_000..(batch + 1) * 64_000)
                    .map(|row| {
                        let value = (row % 97 != 0).then(|| {
                            1.0e12 + f64::from(u32::try_from((row / 64) % 9).unwrap()) / 10.0
                        });
                        (
                            timestamp(row),
                            names[row % 64].as_str(),
                            sequence(row),
                            value,
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    assert_parallel_lifecycle(spec, inputs).await;
}

#[tokio::test]
async fn test_capability_parallel_extrema_pair_special_values_across_restore() {
    let spec = kernel_spec(json!([
        duration_output("min", "price", "time_min", 4_096),
        aggregate_output("max", "price", "row_max", 37),
        pair_output(
            "covariance",
            "price",
            "volume",
            "covariance",
            json!({"kind": "rows", "size": 73}),
            1,
        ),
        pair_output(
            "correlation",
            "price",
            "volume",
            "correlation",
            json!({"kind": "duration", "micros": 8_192}),
            1,
        ),
    ]));
    let names = entity_names();
    let inputs = (0..3)
        .map(|batch| {
            float64_pair_record(
                &(batch * 64_000..(batch + 1) * 64_000)
                    .map(|row| {
                        let price = match (row / 64) % 31 {
                            0 => None,
                            1 => Some(f64::NAN),
                            2 => Some(f64::INFINITY),
                            3 => Some(f64::NEG_INFINITY),
                            4 => Some(-0.0),
                            _ => Some(f64::from(u32::try_from(row % 101).unwrap())),
                        };
                        let volume = (row % 97 != 0)
                            .then(|| f64::from(u32::try_from(row % 67).unwrap()) - 33.0);
                        (
                            timestamp(row),
                            names[row % 64].as_str(),
                            sequence(row),
                            price,
                            volume,
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    assert_parallel_lifecycle(spec, inputs).await;
}

fn entity_names() -> Vec<String> {
    (0..64).map(|entity| format!("e{entity:04}")).collect()
}

fn timestamp(row: usize) -> i64 {
    i64::try_from(row).unwrap()
}

fn sequence(row: usize) -> u64 {
    u64::try_from(row).unwrap()
}

async fn assert_parallel_lifecycle(spec: RollingSpec, inputs: Vec<RecordBatch>) {
    let schema = inputs[0].schema();
    let mut serial = RollingOperator::new("rolling", schema.clone(), spec.clone()).unwrap();
    let mut parallel = RollingOperator::new("rolling", schema.clone(), spec.clone()).unwrap();
    let job = crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    for (step, input) in inputs.into_iter().enumerate() {
        let before = input_ipc(&input);
        let context =
            StreamOperatorContext::new(&job, "rolling", serial.state.last_input_watermark);
        let expected = capability_parallel::close(&mut serial, &context, input.clone()).await;
        let actual = parallel_batch(&mut parallel, &job, input.clone()).await;
        assert_complete_bits(&actual, &expected);
        assert_eq!(input_ipc(&input), before);
        assert_cached_rolling_charge(&parallel);
        assert_cached_rolling_charge(&serial);
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
                .canonical_state_for_test(),
        );
        if step == 1 || step == 2 {
            let epoch = Epoch::new(u64::try_from(step).unwrap()).unwrap();
            let parallel_snapshot = parallel.checkpoint(epoch).unwrap();
            let serial_snapshot = serial.checkpoint(epoch).unwrap();
            assert_eq!(
                parallel.encode_state(epoch).unwrap(),
                serial.encode_state(epoch).unwrap()
            );
            assert_eq!(parallel.state.charge, serial.state.charge);
            if step == 1 {
                serial = RollingOperator::new("rolling", schema.clone(), spec.clone()).unwrap();
                parallel = RollingOperator::new("rolling", schema.clone(), spec.clone()).unwrap();
                StreamOperator::restore(&mut serial, &serial_snapshot).unwrap();
                StreamOperator::restore(&mut parallel, &parallel_snapshot).unwrap();
            }
        }
    }
}

async fn parallel_batch(
    operator: &mut RollingOperator,
    job: &crate::StreamJobContext,
    input: RecordBatch,
) -> RecordBatch {
    let launches = Arc::new(AtomicU64::new(0));
    let owner = JobEntityWorkOwner::new(7, Arc::clone(&launches));
    owner.set_worker_limit_for_test(2);
    let task = owner.test_client(3, "operator:rolling".into());
    let scope = task.callback_scope();
    let context = StreamOperatorContext::new(job, "rolling", operator.state.last_input_watermark)
        .with_entity_work(task.context_client().unwrap());
    let output = capability_parallel::close(operator, &context, input).await;
    scope.settle_abandoned().await;
    drop(context);
    drop(scope);
    owner.close_admission();
    assert!(owner.drain().await.is_empty());
    assert_eq!(launches.load(Ordering::SeqCst), 2);
    output
}

fn assert_complete_bits(actual: &RecordBatch, expected: &RecordBatch) {
    assert_eq!(actual.schema(), expected.schema());
    assert_eq!(actual.num_rows(), expected.num_rows());
    for (actual, expected) in actual.columns().iter().zip(expected.columns()) {
        assert_eq!(actual.to_data(), expected.to_data());
        if let Some(actual) = actual.as_any().downcast_ref::<Float64Array>() {
            let expected = expected.as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.map(f64::to_bits))
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.map(f64::to_bits))
                    .collect::<Vec<_>>(),
            );
        }
    }
}

fn input_ipc(input: &RecordBatch) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut writer =
        datafusion::arrow::ipc::writer::StreamWriter::try_new(&mut bytes, &input.schema()).unwrap();
    writer.write(input).unwrap();
    writer.finish().unwrap();
    drop(writer);
    bytes
}

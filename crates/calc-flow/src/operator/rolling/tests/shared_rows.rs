use super::*;
use crate::{OperatorStateSnapshot, StateSegment, StreamMessage};

fn shared_rows_fixture_spec(profile: RollingNumericalProfile) -> RollingSpec {
    let mut spec = kernel_spec(json!([
        aggregate_output("mean", "price", "fast", 5),
        aggregate_output("mean", "price", "slow", 80)
    ]));
    spec.numerical_profile = profile;
    spec
}

fn shared_rows_fixture_record(start: u32, count: u32) -> RecordBatch {
    let rows = (start..start + count)
        .map(|row| {
            let value = match row % 23 {
                0 => None,
                1 => Some(-0.0),
                _ => Some(1e12 + f64::from(row) * 0.3),
            };
            (
                i64::from(row),
                if row % 2 == 0 { "a" } else { "b" },
                u64::from(row),
                value,
            )
        })
        .collect::<Vec<_>>();
    float64_fast_record(&rows)
}

fn shared_rows_snapshot_json(snapshot: &OperatorStateSnapshot) -> Value {
    json!({
        "inline_metadata": snapshot.inline_metadata,
        "segments": snapshot.segments.iter().map(|(id, segment)| {
            (id.clone(), json!({"bytes": hex::encode(segment.bytes()), "sha256": segment.sha256()}))
        }).collect::<BTreeMap<_, _>>()
    })
}

fn shared_rows_snapshot_from_json(value: &Value) -> OperatorStateSnapshot {
    let segments = value["segments"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(id, segment)| {
            let bytes = hex::decode(segment["bytes"].as_str().unwrap()).unwrap();
            let segment_value = StateSegment::new(bytes);
            assert_eq!(segment_value.sha256(), segment["sha256"].as_str().unwrap());
            (id.clone(), segment_value)
        })
        .collect();
    OperatorStateSnapshot {
        inline_metadata: serde_json::from_value(value["inline_metadata"].clone()).unwrap(),
        segments,
    }
}

fn shared_rows_output_bits(output: Vec<StreamMessage>) -> Vec<Vec<Option<u64>>> {
    let mut columns = vec![Vec::new(), Vec::new()];
    for message in output {
        let batch = message.as_data().unwrap();
        for record in batch.table_payload().unwrap().batches() {
            for (column, values) in columns.iter_mut().zip(record.columns().iter().skip(6)) {
                let values = values.as_any().downcast_ref::<Float64Array>().unwrap();
                column.extend(values.iter().map(|value| value.map(f64::to_bits)));
            }
        }
    }
    columns
}

async fn shared_rows_fixture_continue(
    operator: &mut RollingOperator,
    job: &crate::StreamJobContext,
) -> Vec<Vec<Option<u64>>> {
    let context = StreamOperatorContext::new(job, "rolling", Some(EventTime::from_micros(160)));
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let batch = Batch::table(
        vec![shared_rows_fixture_record(165, 68)],
        BatchMetadata::default(),
    )
    .unwrap();
    operator
        .process_data("input", batch, &context, &mut output)
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(232), &context, &mut output)
        .await
        .unwrap();
    shared_rows_output_bits(output.drain("output"))
}

async fn shared_rows_fixture_case(profile: RollingNumericalProfile) -> Value {
    let spec = shared_rows_fixture_spec(profile);
    let mut source =
        RollingOperator::new("rolling", Arc::new(kernel_schema()), spec.clone()).unwrap();
    let job = crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "rolling", None);
    let mut output = crate::EdgeCollector::new(source.output_ports().to_vec());
    let prefix = Batch::table(
        vec![
            shared_rows_fixture_record(1, 64),
            shared_rows_fixture_record(65, 100),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    source
        .process_data("input", prefix, &context, &mut output)
        .await
        .unwrap();
    source
        .on_watermark(EventTime::from_micros(160), &context, &mut output)
        .await
        .unwrap();
    output.drain("output");
    let snapshot = source.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let snapshot_json = shared_rows_snapshot_json(&snapshot);
    let charge = json!({"rows": source.state.charge.rows, "bytes": source.state.charge.bytes});
    let mut restored = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    StreamOperator::restore(
        &mut restored,
        &shared_rows_snapshot_from_json(&snapshot_json),
    )
    .unwrap();
    let uninterrupted = shared_rows_fixture_continue(&mut source, &job).await;
    let resumed = shared_rows_fixture_continue(&mut restored, &job).await;
    assert_eq!(uninterrupted[0].len(), 72);
    assert_eq!(resumed[0].len(), 72);
    json!({"snapshot": snapshot_json, "charge": charge, "uninterrupted": uninterrupted, "resumed": resumed})
}

#[tokio::test]
async fn shared_rows_legacy_checkpoint_and_writer_preserve_each_execution_route() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/operator/rolling/tests/fixtures/shared-rows-8d8be7d9.json");
    let fixture: Value =
        serde_json::from_str(&tokio::fs::read_to_string(path).await.unwrap()).unwrap();
    assert_eq!(
        fixture["runtime_source_commit"],
        "8d8be7d907a095dbc0dc2b73dafcfdd842979cdf"
    );
    for (name, profile) in [
        ("stable_v1", RollingNumericalProfile::StableV1),
        ("stable_v2", RollingNumericalProfile::StableV2Preview),
    ] {
        let actual = shared_rows_fixture_case(profile).await;
        assert_eq!(
            actual, fixture["cases"][name],
            "legacy writer/continuation mismatch for {name}"
        );
    }
}

#[test]
fn shared_rows_admission_keeps_float32_single_and_same_frame_on_owned_queues() {
    let input = shared_rows_fixture_record(1, 12);
    let spec = shared_rows_fixture_spec(RollingNumericalProfile::StableV1);
    let compiled = compile_spec(&spec, &input.schema()).unwrap();
    let execution = compiled
        .kernel_plan
        .open_and_fill(&input, "rolling")
        .unwrap()
        .unwrap();
    assert_eq!(execution.state.shared_rows_count(), 2);
    let replayed = compiled
        .kernel_plan
        .rebuild_history(&input, "rolling")
        .unwrap()
        .unwrap();
    assert_eq!(replayed.shared_rows_count(), 2);

    let mut fields = input
        .schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields[3] = fields[3].clone().with_data_type(DataType::Float32);
    let mut columns = input.columns().to_vec();
    columns[3] = datafusion::arrow::compute::cast(&columns[3], &DataType::Float32).unwrap();
    let float32 = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
    let compiled = compile_spec(&spec, &float32.schema()).unwrap();
    let execution = compiled
        .kernel_plan
        .open_and_fill(&float32, "rolling")
        .unwrap()
        .unwrap();
    assert_eq!(execution.state.shared_rows_count(), 0);

    for outputs in [
        json!([aggregate_output("mean", "price", "single", 5)]),
        json!([
            aggregate_output("mean", "price", "same", 5),
            aggregate_output("mean", "price", "again", 5)
        ]),
    ] {
        let compiled = compile_spec(&kernel_spec(outputs), &input.schema()).unwrap();
        let execution = compiled
            .kernel_plan
            .open_and_fill(&input, "rolling")
            .unwrap()
            .unwrap();
        assert_eq!(execution.state.shared_rows_count(), 0);
    }
}

#[tokio::test]
async fn shared_rows_legacy_restore_rejection_preserves_existing_cache_and_budget() {
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};

    let pool = Arc::new(GreedyMemoryPool::new(64 << 20));
    let _scope = kernel::shared_rows_test_pool(pool.clone());
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/operator/rolling/tests/fixtures/shared-rows-8d8be7d9.json");
    let fixture: Value =
        serde_json::from_str(&tokio::fs::read_to_string(path).await.unwrap()).unwrap();
    let snapshot = shared_rows_snapshot_from_json(&fixture["cases"]["stable_v2"]["snapshot"]);
    let spec = shared_rows_fixture_spec(RollingNumericalProfile::StableV2Preview);
    let mut target = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let job = crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "rolling", None);
    let mut output = crate::EdgeCollector::new(target.output_ports().to_vec());
    target
        .process_data(
            "input",
            Batch::table(
                vec![shared_rows_fixture_record(0, 1)],
                BatchMetadata::default(),
            )
            .unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    target
        .on_watermark(EventTime::from_micros(0), &context, &mut output)
        .await
        .unwrap();
    let installed_charge = (target.state.charge.rows, target.state.charge.bytes);
    assert!(
        fixture["cases"]["stable_v2"]["charge"]["rows"]
            .as_u64()
            .unwrap()
            > installed_charge.0
    );
    target
        .set_state_budget(
            StateBudget::new(installed_charge.0, installed_charge.1.max(1_048_576)).unwrap(),
        )
        .unwrap();
    assert_eq!(
        target
            .state
            .typed_kernel_state
            .as_ref()
            .unwrap()
            .shared_rows_count(),
        1
    );
    let before = (
        target.encode_state(Epoch::new(1).unwrap()).unwrap(),
        format!("{:?}", target.state.typed_kernel_state),
    );
    let installed_credit = pool.reserved();
    assert!(installed_credit > 0);
    let error = StreamOperator::restore(&mut target, &snapshot).unwrap_err();
    assert!(error.to_string().contains("state budget"), "{error}");
    assert_eq!(
        (target.state.charge.rows, target.state.charge.bytes),
        installed_charge
    );
    assert_eq!(
        (
            target.encode_state(Epoch::new(1).unwrap()).unwrap(),
            format!("{:?}", target.state.typed_kernel_state)
        ),
        before
    );
    let wrong_profile = shared_rows_fixture_spec(RollingNumericalProfile::StableV1);
    let mut wrong =
        RollingOperator::new("rolling", Arc::new(kernel_schema()), wrong_profile).unwrap();
    assert!(matches!(
        StreamOperator::restore(&mut wrong, &snapshot),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    assert!(wrong.state.typed_kernel_state.is_none());
    assert_eq!(wrong.state.charge.rows, 0);
    assert_eq!(pool.reserved(), installed_credit);
    drop(target);
    drop(wrong);
    assert_eq!(pool.reserved(), 0);
}

struct RejectSharedSerial(bool);

#[async_trait]
impl StreamCollector for RejectSharedSerial {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        if self.0 {
            Err(CalcFlowError::Cancelled { run_id: "7".into() })
        } else {
            Err(internal_error("expected P3 serial refusal"))
        }
    }
}

fn p3_serial_record(start: u32, count: u32) -> RecordBatch {
    float64_fast_record(
        &(start..start + count)
            .map(|row| (i64::from(row), "a", u64::from(row), Some(f64::from(row))))
            .collect::<Vec<_>>(),
    )
}

fn p3_admit_owned_record(operator: &mut RollingOperator, record: &RecordBatch) {
    let charge = budget::ordered_buffer_record_charge(record, &operator.compiled, "rolling")
        .unwrap()
        .unwrap();
    operator.state.charge = operator
        .state
        .charge
        .checked_add(charge, "rolling")
        .unwrap();
}

fn p3_serial_resident(operator: &RollingOperator) -> Value {
    let typed = operator.state.typed_kernel_state.as_ref().unwrap();
    json!({
        "snapshot": operator.encode_state(Epoch::new(1).unwrap()).unwrap(),
        "typed": format!("{typed:?}"),
        "owners": typed.shared_rows_owner_addresses(),
        "charge": [operator.state.charge.rows, operator.state.charge.bytes],
        "watermark": operator.state.last_input_watermark.map(EventTime::as_micros),
        "sequence": operator.state.next_output_sequence,
    })
}

#[tokio::test]
async fn p3_serial_dual_rows_refusal_and_cancel_release_candidate_credit_before_replay() {
    for cancelled in [false, true] {
        p3_check_serial_rollback(cancelled).await;
    }
}

async fn p3_check_serial_rollback(cancelled: bool) {
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};

    let pool = Arc::new(GreedyMemoryPool::new(64 << 20));
    let _scope = kernel::shared_rows_test_pool(pool.clone());
    let spec = kernel_spec(json!([
        aggregate_output("mean", "price", "fast", 5),
        aggregate_output("mean", "price", "slow", 20)
    ]));
    let mut operator = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let job = crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "rolling", None);
    let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            Batch::table(vec![p3_serial_record(1, 32)], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(32), &context, &mut collector)
        .await
        .unwrap();
    collector.drain("output");
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let alias = operator.state.typed_kernel_state.clone().unwrap();
    let before = p3_serial_resident(&operator);
    let resident_credit = pool.reserved();
    assert!(resident_credit > 0);
    assert_eq!(alias.shared_rows_count(), 1);
    let next = p3_serial_record(33, 1);
    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(32)));
    p3_admit_owned_record(&mut operator, &next);
    let result = operator
        .emit_ordered(
            vec![next.clone()],
            None,
            &context,
            &mut RejectSharedSerial(cancelled),
        )
        .await;
    let error = result.unwrap_err();
    if cancelled {
        assert!(matches!(error, CalcFlowError::Cancelled { run_id } if run_id == "7"));
    } else {
        assert!(error.to_string().contains("expected P3 serial refusal"));
    }
    assert_eq!(p3_serial_resident(&operator), before);
    assert_eq!(pool.reserved(), resident_credit);
    assert!(collector.drain("output").is_empty());
    p3_admit_owned_record(&mut operator, &next);
    operator
        .emit_ordered(vec![next], None, &context, &mut collector)
        .await
        .unwrap();
    let output = shared_rows_output_bits(collector.drain("output"));
    assert_eq!(
        output,
        vec![
            vec![Some(31.0_f64.to_bits())],
            vec![Some(23.5_f64.to_bits())]
        ]
    );
    assert_eq!(pool.reserved(), 2 * resident_credit);
    drop(snapshot);
    drop(alias);
    assert_eq!(pool.reserved(), resident_credit);
    drop(operator);
    assert_eq!(pool.reserved(), 0);
}

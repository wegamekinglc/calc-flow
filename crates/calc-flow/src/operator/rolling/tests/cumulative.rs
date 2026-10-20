use super::*;
use datafusion::arrow::array::Int64Array;

fn cumulative_spec() -> RollingSpec {
    serde_json::from_value(json!({
        "configuration_version": 1,
        "state_layout_version": 4,
        "partition_by": ["symbol"], "event_time": "ts", "sequence_by": ["sequence"],
        "outputs": [
            {"kind":"cumulative_sum", "primitive_version":1, "input":"price", "output":"total", "min_periods":1},
            {"kind":"cumulative_min", "primitive_version":1, "input":"price", "output":"low", "min_periods":1},
            {"kind":"cumulative_max", "primitive_version":1, "input":"price", "output":"high", "min_periods":1},
            {"kind":"cumulative_variance", "primitive_version":1, "input":"price", "output":"variance", "min_periods":1, "ddof":1},
            {"kind":"cumulative_variance", "primitive_version":1, "input":"price", "output":"population", "min_periods":2, "ddof":0}
        ],
        "allowed_lateness_micros":0,
        "late_policy":{"kind":"error", "scope":"envelope"},
        "value_policy":"stateful_numeric_v1"
    })).unwrap()
}

fn mixed_cumulative_spec() -> RollingSpec {
    let mut value = serde_json::to_value(cumulative_spec()).unwrap();
    let outputs = value["outputs"].as_array_mut().unwrap();
    outputs.push(json!({"kind":"cumulative_mean", "primitive_version":1,
        "input":"price", "output":"mean", "min_periods":1}));
    outputs.push(aggregate_output("mean", "price", "rolling_mean", 3));
    serde_json::from_value(value).unwrap()
}

#[test]
fn test_cumulative_single_and_mixed_stages_select_typed_transition() {
    let mut selections = Vec::new();
    for profile in [
        RollingNumericalProfile::StableV1,
        RollingNumericalProfile::StableV2Preview,
    ] {
        let mut mixed = mixed_cumulative_spec();
        mixed.numerical_profile = profile;
        let mut cumulative = cumulative_spec();
        cumulative.numerical_profile = profile;
        let compiled = compile_spec(&cumulative, &kernel_schema()).unwrap();
        let expected = match profile {
            RollingNumericalProfile::StableV1 => {
                "241fc56daeb51a3fea8f780c215658956b3b75437852305f4ac6c657bb4f131a"
            }
            RollingNumericalProfile::StableV2Preview => {
                "37d8c3f0e6b451348349c5fd69da6346be96cbb0e69d584830f35ae1cd473646"
            }
        };
        assert_eq!(
            compiled.kernel_plan.fingerprint(),
            expected,
            "layout 4 identity at e59f26cb"
        );
        for output in &cumulative.outputs {
            let single = RollingSpec {
                outputs: vec![output.clone()],
                ..cumulative.clone()
            };
            selections.push(
                compile_spec(&single, &kernel_schema())
                    .unwrap()
                    .kernel_plan
                    .selection(),
            );
        }
        selections.push(
            compile_spec(&mixed, &kernel_schema())
                .unwrap()
                .kernel_plan
                .selection(),
        );
    }
    assert!(
        selections
            .iter()
            .all(|selection| *selection == KernelSelection::OrderedPrimitive),
        "{selections:?}"
    );
}

fn cumulative_volume_spec(profile: RollingNumericalProfile) -> RollingSpec {
    let mut value = serde_json::to_value(cumulative_spec()).unwrap();
    for output in value["outputs"].as_array_mut().unwrap() {
        output["input"] = json!("volume");
    }
    let mut spec: RollingSpec = serde_json::from_value(value).unwrap();
    spec.numerical_profile = profile;
    spec
}

#[test]
fn test_cumulative_typed_state_rejects_a_different_transition() {
    let spec = cumulative_spec();
    let compile = |index: usize| {
        compile_spec(
            &RollingSpec {
                outputs: vec![spec.outputs[index].clone()],
                ..spec.clone()
            },
            &kernel_schema(),
        )
        .unwrap()
        .kernel_plan
    };
    let sum = compile(0);
    let min = compile(1);
    assert_eq!(sum.fingerprint(), min.fingerprint());
    let input = float64_fast_record(&[(1, "a", 1, Some(2.0))]);
    let state = sum.open_and_fill(&input, "rolling").unwrap().unwrap().state;
    assert!(
        min.update_and_fill(
            &state,
            &float64_fast_record(&[(2, "a", 2, Some(3.0))]),
            "rolling"
        )
        .is_err()
    );
}

#[test]
fn test_cumulative_typed_primitives_match_general() {
    use datafusion::arrow::array::Float32Array;
    let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(2), None, Some(0), Some(3)]));
    let mut cases = [
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Float32,
        DataType::Float64,
    ]
    .iter()
    .map(|dtype| datafusion::arrow::compute::cast(&values, dtype).unwrap())
    .collect::<Vec<_>>();
    cases.push(Arc::new(Float64Array::from(vec![
        None,
        Some(f64::NAN),
        Some(-0.0),
        Some(0.0),
        Some(1e12),
        Some(1e12 + 2.0),
        Some(f64::INFINITY),
        Some(f64::NEG_INFINITY),
    ])));
    cases.push(Arc::new(Float32Array::from(vec![
        Some(-0.0),
        None,
        Some(f32::NAN),
        Some(2.5),
        Some(f32::INFINITY),
    ])));
    cases.push(Arc::new(Float64Array::from(vec![
        Some(1e308),
        Some(-1e308),
        Some(1.0),
    ])));
    cases.push(Arc::new(Float64Array::new_null(0)));
    for profile in [
        RollingNumericalProfile::StableV1,
        RollingNumericalProfile::StableV2Preview,
    ] {
        for values in &cases {
            let (schema, input) = primitive_volume_record(Arc::clone(values));
            assert_typed_matches_general(&cumulative_volume_spec(profile), &schema, &input);
        }
    }
}

#[test]
fn test_cumulative_typed_integer_overflow_matches_general() {
    let cases: [ArrayRef; 2] = [
        Arc::new(Int64Array::from(vec![i64::MAX, 1])),
        Arc::new(UInt64Array::from(vec![u64::MAX, 1])),
    ];
    for values in cases {
        let (schema, input) = primitive_volume_record(values);
        let compiled = compile_spec(
            &cumulative_volume_spec(RollingNumericalProfile::StableV1),
            &schema,
        )
        .unwrap();
        let fast = compiled
            .kernel_plan
            .open_and_fill(&input, "rolling")
            .err()
            .unwrap();
        let rows = (0..input.num_rows())
            .map(|index| read_buffered_row(&input, index, &compiled, "rolling").unwrap())
            .collect::<Vec<_>>();
        let general =
            compute_output_columns(&rows, &RollingHistories::default(), &compiled, "rolling")
                .err()
                .unwrap();
        assert_eq!(fast.to_string(), general.to_string());
    }
    let mut spec = cumulative_volume_spec(RollingNumericalProfile::StableV1);
    spec.outputs.truncate(1);
    let mut value = serde_json::to_value(spec).unwrap();
    value["outputs"][0]["min_periods"] = json!(3);
    let spec = serde_json::from_value(value).unwrap();
    let (schema, input) = primitive_volume_record(Arc::new(Int64Array::from(vec![
        i64::MAX,
        i64::MAX,
        -i64::MAX,
    ])));
    assert_typed_matches_general(&spec, &schema, &input);
}

#[test]
fn test_cumulative_preserves_unordered_and_unsupported_stage_fallback() {
    let spec = mixed_cumulative_spec();
    let compiled = compile_spec(&spec, &kernel_schema()).unwrap();
    let input = float64_fast_record(&[(2, "a", 2, Some(2.0)), (1, "a", 1, Some(1.0))]);
    assert!(
        compiled
            .kernel_plan
            .open_and_fill(&input, "rolling")
            .unwrap()
            .is_none()
    );
    let mut value = serde_json::to_value(spec).unwrap();
    value["outputs"].as_array_mut().unwrap().push(lag_price(1));
    let spec = serde_json::from_value(value).unwrap();
    assert_eq!(
        compile_spec(&spec, &kernel_schema())
            .unwrap()
            .kernel_plan
            .selection(),
        KernelSelection::General
    );
}

async fn advance_cumulative_stream(
    operator: &mut RollingOperator,
    input: &RecordBatch,
    watermark: i64,
    job: &crate::StreamJobContext,
) -> RecordBatch {
    let context = StreamOperatorContext::new(job, "rolling", None);
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            Batch::table(vec![input.clone()], BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(watermark), &context, &mut output)
        .await
        .unwrap();
    drain(&mut output).remove(0)
}

async fn cumulative_stream_roundtrip(profile: RollingNumericalProfile, rolling: bool) {
    let mut spec = mixed_cumulative_spec();
    spec.numerical_profile = profile;
    if !rolling {
        spec.outputs.pop();
    }
    let mut value = serde_json::to_value(&spec).unwrap();
    value["outputs"].as_array_mut().unwrap().push(lag_price(1));
    let fallback_spec = serde_json::from_value(value).unwrap();
    let mut fast =
        RollingOperator::new("rolling", Arc::new(kernel_schema()), spec.clone()).unwrap();
    let mut general =
        RollingOperator::new("rolling", Arc::new(kernel_schema()), fallback_spec).unwrap();
    let job = crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let first = float64_fast_record(&[
        (1, "a", 1, Some(1e12)),
        (1, "b", 1, None),
        (2, "a", 2, Some(1e12 + 2.0)),
        (2, "b", 2, Some(10.0)),
    ]);
    let out = advance_cumulative_stream(&mut fast, &first, 1, &job).await;
    let fallback = advance_cumulative_stream(&mut general, &first, 1, &job).await;
    for (actual, expected) in out.columns().iter().zip(fallback.columns()) {
        assert_eq!(actual.to_data(), expected.to_data());
    }
    assert!(fast.state.buffer.is_empty());
    assert!(!fast.state.ordered.is_empty());
    assert!(fast.state.typed_kernel_state.is_some());
    assert_eq!(
        out.column(3).to_data().buffers()[0].as_ptr(),
        first.column(3).to_data().buffers()[0].as_ptr()
    );
    if !rolling {
        assert!(
            fast.state
                .histories
                .by_entity
                .values()
                .all(|state| state.rows.is_empty() && state.columnar.records.is_empty())
        );
    }
    let snapshot = fast.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let mut restored = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    let next = float64_fast_record(&[
        (3, "a", 3, Some(1e12 + 4.0)),
        (3, "b", 3, Some(f64::NAN)),
        (4, "a", 4, None),
        (4, "b", 4, Some(20.0)),
    ]);
    let expected = advance_cumulative_stream(&mut general, &next, 4, &job).await;
    for operator in [&mut fast, &mut restored] {
        let actual = advance_cumulative_stream(operator, &next, 4, &job).await;
        for (actual, expected) in actual.columns().iter().zip(expected.columns()) {
            assert_eq!(actual.to_data(), expected.to_data());
        }
        assert!(operator.state.typed_kernel_state.is_some());
    }
}

#[tokio::test]
async fn test_cumulative_ordered_and_fallback_streams_match_across_batches_and_restore() {
    for profile in [
        RollingNumericalProfile::StableV1,
        RollingNumericalProfile::StableV2Preview,
    ] {
        for rolling in [false, true] {
            cumulative_stream_roundtrip(profile, rolling).await;
        }
    }
}

async fn assert_cumulative_prefix_overflow_is_atomic(
    operator: &mut RollingOperator,
    input: &RecordBatch,
    job: &crate::StreamJobContext,
) {
    let context = StreamOperatorContext::new(job, "rolling", Some(EventTime::from_micros(4)));
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            Batch::table(vec![input.clone()], BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    let before = format!("{:?}", operator.state.typed_kernel_state);
    let error = operator
        .on_watermark(EventTime::from_micros(5), &context, &mut output)
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cumulative integer sum overflowed"),
        "{error}"
    );
    assert!(drain(&mut output).is_empty());
    assert_eq!(format!("{:?}", operator.state.typed_kernel_state), before);
}

#[tokio::test]
async fn test_cumulative_mixed_integer_restore_replays_only_window_history() {
    for profile in [
        RollingNumericalProfile::StableV1,
        RollingNumericalProfile::StableV2Preview,
    ] {
        let mut spec = cumulative_volume_spec(profile);
        spec.outputs.truncate(1);
        spec.outputs.push(
            serde_json::from_value(aggregate_output("mean", "volume", "rolling_mean", 2)).unwrap(),
        );
        let mut source =
            RollingOperator::new("rolling", Arc::new(kernel_schema()), spec.clone()).unwrap();
        let job = crate::StreamJobContext::new(
            7,
            TEST_FINGERPRINT,
            JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        );
        let input = int64_fast_record(&[
            Some(-i64::MAX),
            Some(i64::MAX),
            Some(i64::MAX),
            Some(0),
            Some(1),
        ]);
        let prefix = advance_cumulative_stream(&mut source, &input.slice(0, 3), 3, &job).await;
        let totals = prefix
            .column_by_name("total")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(totals.values().as_ref(), &[-i64::MAX, 0, i64::MAX]);
        let snapshot = source.checkpoint(Epoch::new(1).unwrap()).unwrap();
        let mut restored =
            RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
        StreamOperator::restore(&mut restored, &snapshot).unwrap();
        let expected = advance_cumulative_stream(&mut source, &input.slice(3, 1), 4, &job).await;
        let actual = advance_cumulative_stream(&mut restored, &input.slice(3, 1), 4, &job).await;
        assert_eq!(actual, expected, "{profile:?}");
        let total = actual
            .column_by_name("total")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(total.value(0), i64::MAX);
        for operator in [&mut source, &mut restored] {
            assert_cumulative_prefix_overflow_is_atomic(operator, &input.slice(4, 1), &job).await;
        }
    }
}

fn drain(collector: &mut crate::EdgeCollector) -> Vec<RecordBatch> {
    collector
        .drain("output")
        .iter()
        .flat_map(|message| {
            message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
                .to_vec()
        })
        .collect()
}

fn remove_cumulative_summaries(
    snapshot: &crate::OperatorStateSnapshot,
) -> crate::OperatorStateSnapshot {
    use datafusion::arrow::{array::BooleanArray, compute::filter_record_batch};

    let mut snapshot = snapshot.clone();
    let (segment_id, segment) = snapshot.segments.iter().next().unwrap();
    let segment_id = segment_id.clone();
    let mut reader = FileReader::try_new(Cursor::new(segment.bytes()), None).unwrap();
    let record = reader.next().unwrap().unwrap();
    let kinds = record
        .column(0)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .unwrap();
    let keep = kinds
        .values()
        .iter()
        .map(|kind| *kind != 4)
        .collect::<BooleanArray>();
    let stripped = filter_record_batch(&record, &keep).unwrap();
    assert!(stripped.num_rows() < record.num_rows());
    let mut bytes = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut bytes, stripped.schema().as_ref()).unwrap();
        writer.write(&stripped).unwrap();
        writer.finish().unwrap();
    }
    let replacement = crate::StateSegment::new(bytes);
    let handle = &mut snapshot
        .inline_metadata
        .get_mut("segment_inventory")
        .unwrap()[0]["handle"];
    handle["byte_len"] = json!(replacement.bytes().len());
    handle["sha256"] = json!(replacement.sha256());
    snapshot.segments.insert(segment_id, replacement);
    snapshot
}

async fn pending_checkpoint(
    profile: RollingNumericalProfile,
    prefix: bool,
) -> (
    RollingOperator,
    crate::OperatorStateSnapshot,
    crate::StreamJobContext,
) {
    let mut spec = cumulative_spec();
    spec.numerical_profile = profile;
    let mut source = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let job = crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "rolling", None);
    let rows = if prefix {
        vec![(1, "a", 1, Some(10.0)), (2, "a", 2, Some(20.0))]
    } else {
        source
            .set_state_budget(StateBudget::new(1, 1 << 20).unwrap())
            .unwrap();
        vec![(2, "a", 2, Some(20.0))]
    };
    let mut output = crate::EdgeCollector::new(source.output_ports().to_vec());
    source
        .process_data(
            "input",
            Batch::table(vec![float64_fast_record(&rows)], BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    source
        .on_watermark(EventTime::from_micros(1), &context, &mut output)
        .await
        .unwrap();
    let snapshot = source.checkpoint(Epoch::new(1).unwrap()).unwrap();
    (source, snapshot, job)
}

#[tokio::test]
async fn test_cumulative_snapshot_rejects_missing_prefix_inventory() {
    for profile in [
        RollingNumericalProfile::StableV1,
        RollingNumericalProfile::StableV2Preview,
    ] {
        let (mut source, snapshot, job) = pending_checkpoint(profile, true).await;
        let mut restored =
            RollingOperator::new("rolling", Arc::new(kernel_schema()), source.spec().clone())
                .unwrap();
        StreamOperator::restore(&mut restored, &snapshot).unwrap();
        let corrupted = remove_cumulative_summaries(&snapshot);
        assert!(
            StreamOperator::restore(&mut restored, &corrupted).is_err(),
            "{profile:?}: restore accepted a processed prefix without any cumulative summaries"
        );
        let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(1)));
        let mut output = crate::EdgeCollector::new(source.output_ports().to_vec());
        for operator in [&mut source, &mut restored] {
            operator
                .on_watermark(EventTime::from_micros(2), &context, &mut output)
                .await
                .unwrap();
        }
        let output = drain(&mut output);
        assert_eq!(output[0], output[1]);
        let total = output[1]
            .column_by_name("total")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(total.value(0).to_bits(), 30.0_f64.to_bits());
    }
}

#[tokio::test]
async fn test_cumulative_snapshot_restores_pending_only_entity() {
    for profile in [
        RollingNumericalProfile::StableV1,
        RollingNumericalProfile::StableV2Preview,
    ] {
        let (mut source, snapshot, job) = pending_checkpoint(profile, false).await;
        let mut restored =
            RollingOperator::new("rolling", Arc::new(kernel_schema()), source.spec().clone())
                .unwrap();
        restored
            .set_state_budget(StateBudget::new(1, 1 << 20).unwrap())
            .unwrap();
        StreamOperator::restore(&mut restored, &snapshot).unwrap();
        assert!(restored.state.histories.by_entity.is_empty());
        let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(1)));
        let mut output = crate::EdgeCollector::new(source.output_ports().to_vec());
        for operator in [&mut source, &mut restored] {
            operator
                .on_watermark(EventTime::from_micros(2), &context, &mut output)
                .await
                .unwrap();
        }
        let output = drain(&mut output);
        assert_eq!(output[0], output[1]);
        let total = output[1]
            .column_by_name("total")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(total.value(0).to_bits(), 20.0_f64.to_bits());
    }
}

#[tokio::test]
async fn test_cumulative_snapshot_restores_summaries_and_pending_rows() {
    use crate::{CancellationToken, StreamJobContext};
    let spec = cumulative_spec();
    let mut source =
        RollingOperator::new("rolling", Arc::new(kernel_schema()), spec.clone()).unwrap();
    assert_eq!(source.state_layout_version(), 4);
    let job = StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "rolling", None);
    let mut collector = crate::EdgeCollector::new(source.output_ports().to_vec());
    let first = float64_fast_record(&[
        (1, "a", 1, Some(1e12)),
        (1, "b", 1, None),
        (2, "a", 2, Some(1e12 + 2.0)),
        (2, "b", 2, Some(10.0)),
    ]);
    source
        .process_data(
            "input",
            Batch::table(vec![first], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    source
        .on_watermark(EventTime::from_micros(1), &context, &mut collector)
        .await
        .unwrap();
    drain(&mut collector);
    let snapshot = source.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let mut restored = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    let mut unknown = snapshot.clone();
    unknown
        .inline_metadata
        .insert("state_layout_version".into(), json!(99));
    assert!(StreamOperator::restore(&mut restored, &unknown).is_err());
    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(1)));
    for operator in [&mut source, &mut restored] {
        operator
            .on_watermark(EventTime::from_micros(2), &context, &mut collector)
            .await
            .unwrap();
    }
    let output = drain(&mut collector);
    assert_eq!(output.len(), 2);
    assert_eq!(output[0], output[1]);
    let variance = output[1]
        .column_by_name("variance")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(variance.value(0).to_bits(), 2.0_f64.to_bits());
    assert!(variance.is_null(1));
    let low = output[1]
        .column_by_name("low")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(low.value(0).to_bits(), 1e12_f64.to_bits());
    assert_eq!(low.value(1).to_bits(), 10.0_f64.to_bits());
    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(2)));
    for operator in [&mut source, &mut restored] {
        let next = float64_fast_record(&[(3, "a", 3, Some(1e12 + 4.0)), (3, "b", 3, Some(20.0))]);
        operator
            .process_data(
                "input",
                Batch::table(vec![next], BatchMetadata::default()).unwrap(),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        operator
            .on_watermark(EventTime::from_micros(3), &context, &mut collector)
            .await
            .unwrap();
    }
    let output = drain(&mut collector);
    assert_eq!(output[0], output[1]);
    let variance = output[1]
        .column_by_name("variance")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(variance.value(0).to_bits(), 4.0_f64.to_bits());
    assert_eq!(variance.value(1).to_bits(), 50.0_f64.to_bits());
}

#[test]
fn test_cumulative_versions_and_ddof_fail_closed() {
    let mut spec = cumulative_spec();
    spec.state_layout_version = 2;
    assert!(RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).is_err());
    let mut value = serde_json::to_value(cumulative_spec()).unwrap();
    value["outputs"][0]["primitive_version"] = json!(99);
    assert!(
        RollingOperator::new(
            "rolling",
            Arc::new(kernel_schema()),
            serde_json::from_value(value).unwrap()
        )
        .is_err()
    );
    let mut value = serde_json::to_value(cumulative_spec()).unwrap();
    value["outputs"][3]["ddof"] = json!(2);
    assert!(
        RollingOperator::new(
            "rolling",
            Arc::new(kernel_schema()),
            serde_json::from_value(value).unwrap()
        )
        .is_err()
    );
}

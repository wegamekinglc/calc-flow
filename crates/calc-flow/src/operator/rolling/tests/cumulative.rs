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
    let mixed = mixed_cumulative_spec();
    let cumulative = cumulative_spec();
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
    assert!(
        selections
            .iter()
            .all(|selection| *selection == KernelSelection::OrderedPrimitive),
        "{selections:?}"
    );
}

fn cumulative_volume_spec() -> RollingSpec {
    let mut value = serde_json::to_value(cumulative_spec()).unwrap();
    for output in value["outputs"].as_array_mut().unwrap() {
        output["input"] = json!("volume");
    }
    serde_json::from_value(value).unwrap()
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
    for values in &cases {
        let (schema, input) = primitive_volume_record(Arc::clone(values));
        assert_typed_matches_general(&cumulative_volume_spec(), &schema, &input);
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
        let compiled = compile_spec(&cumulative_volume_spec(), &schema).unwrap();
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
    let mut spec = cumulative_volume_spec();
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
fn test_cumulative_preserves_unordered_and_typed_lag_execution() {
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
    value["outputs"].as_array_mut().unwrap().push(lag_label(1));
    let spec = serde_json::from_value(value).unwrap();
    assert_eq!(
        compile_spec(&spec, &general_schema())
            .unwrap()
            .kernel_plan
            .selection(),
        KernelSelection::OrderedPrimitive
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

async fn cumulative_stream_roundtrip(rolling: bool) {
    let mut spec = mixed_cumulative_spec();
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
    for rolling in [false, true] {
        cumulative_stream_roundtrip(rolling).await;
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
async fn test_cumulative_primitive_slab_outputs_preserve_storage() {
    for data_type in [DataType::Int8, DataType::UInt64, DataType::Float32] {
        let values = datafusion::arrow::compute::cast(
            &Int64Array::from(vec![Some(2), None, Some(3)]),
            &data_type,
        )
        .unwrap();
        let (schema, input) = primitive_volume_record(values);
        let spec = cumulative_volume_spec();
        let mut source = RollingOperator::new("rolling", Arc::new(schema), spec.clone()).unwrap();
        let job = crate::StreamJobContext::new(
            7,
            TEST_FINGERPRINT,
            JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        );
        let prefix = advance_cumulative_stream(&mut source, &input.slice(0, 2), 2, &job).await;
        let sum_type = match data_type {
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => DataType::Int64,
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                DataType::UInt64
            }
            DataType::Float32 | DataType::Float64 => DataType::Float64,
            _ => unreachable!(),
        };
        assert_eq!(
            prefix.column_by_name("total").unwrap().data_type(),
            &sum_type
        );
        for name in ["low", "high"] {
            assert_eq!(prefix.column_by_name(name).unwrap().data_type(), &data_type);
        }
        for name in ["variance", "population"] {
            assert_eq!(
                prefix.column_by_name(name).unwrap().data_type(),
                &DataType::Float64
            );
        }
        let snapshot = source.checkpoint(Epoch::new(1).unwrap()).unwrap();
        assert_eq!(snapshot.inline_metadata["state_layout_version"], json!(7));
        let mut restored = RollingOperator::new("rolling", input.schema(), spec).unwrap();
        StreamOperator::restore(&mut restored, &snapshot).unwrap();
        let expected = advance_cumulative_stream(&mut source, &input.slice(2, 1), 3, &job).await;
        let actual = advance_cumulative_stream(&mut restored, &input.slice(2, 1), 3, &job).await;
        assert_eq!(actual, expected);
        assert_eq!(
            actual.column_by_name("total").unwrap().data_type(),
            &sum_type
        );
    }
}

#[tokio::test]
async fn test_count_slab_restore_accepts_nested_values() {
    use datafusion::arrow::array::{Int32Builder, ListBuilder};

    let spec = kernel_spec(json!([
        aggregate_output("count", "volume", "two", 2),
        {
            "kind":"count", "primitive_version":1, "input":"volume", "output":"three",
            "frame":{"kind":"rows", "size":3}, "min_periods":1
        }
    ]));
    let mut values = ListBuilder::new(Int32Builder::new());
    values.values().append_value(1);
    values.append(true);
    values.append(false);
    values.values().append_value(4);
    values.append(true);
    let (schema, input) = primitive_volume_record(Arc::new(values.finish()));
    let mut source = RollingOperator::new("rolling", Arc::new(schema), spec.clone()).unwrap();
    assert_eq!(source.state_layout_version(), 7);
    let job = crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let prefix = advance_cumulative_stream(&mut source, &input.slice(0, 2), 2, &job).await;
    assert_eq!(
        prefix
            .column_by_name("two")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[1, 1]
    );
    let snapshot = source.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert_eq!(snapshot.inline_metadata["state_layout_version"], json!(7));
    let mut restored = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    let expected = advance_cumulative_stream(&mut source, &input.slice(2, 1), 3, &job).await;
    let actual = advance_cumulative_stream(&mut restored, &input.slice(2, 1), 3, &job).await;
    assert_eq!(actual, expected);
}

#[tokio::test]
async fn test_float64_lag_delta_slab_restore_preserves_reference_rows() {
    let spec = kernel_spec(json!([
        lag_price(1),
        delta_price(1),
        {
            "kind":"lag", "primitive_version":1, "input":"price", "output":"lag_two",
            "periods":2
        }
    ]));
    let input = float64_fast_record(&[
        (1, "a", 1, None),
        (1, "b", 1, Some(10.0)),
        (2, "a", 2, Some(f64::NAN)),
        (2, "b", 2, Some(20.0)),
        (3, "a", 3, Some(1.0)),
        (3, "b", 3, Some(30.0)),
        (4, "a", 4, Some(3.0)),
        (4, "b", 4, Some(40.0)),
    ]);
    let mut source = RollingOperator::new("rolling", input.schema(), spec.clone()).unwrap();
    assert_eq!(source.state_layout_version(), 7);
    let job = crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let prefix = advance_cumulative_stream(&mut source, &input.slice(0, 6), 3, &job).await;
    let lag = prefix
        .column_by_name("price_lag")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert!(lag.is_null(0) && lag.is_null(1) && lag.is_null(2));
    assert!(lag.value(4).is_nan());
    assert_eq!(lag.value(5).to_bits(), 20.0_f64.to_bits());
    let snapshot = source.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert_eq!(snapshot.inline_metadata["state_layout_version"], json!(7));
    let mut restored = RollingOperator::new("rolling", input.schema(), spec).unwrap();
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    let expected = advance_cumulative_stream(&mut source, &input.slice(6, 2), 4, &job).await;
    let actual = advance_cumulative_stream(&mut restored, &input.slice(6, 2), 4, &job).await;
    assert_eq!(actual, expected);
}

#[tokio::test]
async fn test_cumulative_integer_slab_restore_and_overflow_are_atomic() {
    let spec = cumulative_volume_spec();
    let mut source =
        RollingOperator::new("rolling", Arc::new(kernel_schema()), spec.clone()).unwrap();
    assert_eq!(source.state_layout_version(), 7);
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
    assert_eq!(
        prefix
            .column_by_name("total")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[-i64::MAX, 0, i64::MAX]
    );
    let snapshot = source.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert_eq!(snapshot.inline_metadata["state_layout_version"], json!(7));
    let mut restored = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    let expected = advance_cumulative_stream(&mut source, &input.slice(3, 1), 4, &job).await;
    let actual = advance_cumulative_stream(&mut restored, &input.slice(3, 1), 4, &job).await;
    assert_eq!(actual, expected);
    for operator in [&mut source, &mut restored] {
        assert_cumulative_prefix_overflow_is_atomic(operator, &input.slice(4, 1), &job).await;
    }
}

#[tokio::test]
async fn test_cumulative_mixed_integer_restore_replays_only_window_history() {
    let mut spec = cumulative_volume_spec();
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
    let mut restored = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    let expected = advance_cumulative_stream(&mut source, &input.slice(3, 1), 4, &job).await;
    let actual = advance_cumulative_stream(&mut restored, &input.slice(3, 1), 4, &job).await;
    assert_eq!(actual, expected);
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

fn remove_cumulative_prefix(
    snapshot: &crate::OperatorStateSnapshot,
) -> crate::OperatorStateSnapshot {
    let mut snapshot = snapshot.clone();
    snapshot.segments.clear();
    snapshot
}

async fn pending_checkpoint(
    prefix: bool,
) -> (
    RollingOperator,
    crate::OperatorStateSnapshot,
    crate::StreamJobContext,
) {
    let spec = cumulative_spec();
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
    let (mut source, snapshot, job) = pending_checkpoint(true).await;
    let mut restored =
        RollingOperator::new("rolling", Arc::new(kernel_schema()), source.spec().clone()).unwrap();
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    let corrupted = remove_cumulative_prefix(&snapshot);
    assert!(
        StreamOperator::restore(&mut restored, &corrupted).is_err(),
        "restore accepted a processed prefix without any cumulative summaries"
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

#[tokio::test]
async fn test_cumulative_snapshot_restores_pending_only_entity() {
    let (mut source, snapshot, job) = pending_checkpoint(false).await;
    let mut restored =
        RollingOperator::new("rolling", Arc::new(kernel_schema()), source.spec().clone()).unwrap();
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

#[tokio::test]
async fn test_cumulative_snapshot_restores_summaries_and_pending_rows() {
    use crate::{CancellationToken, StreamJobContext};
    let spec = cumulative_spec();
    let mut source =
        RollingOperator::new("rolling", Arc::new(kernel_schema()), spec.clone()).unwrap();
    assert_eq!(source.state_layout_version(), 7);
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

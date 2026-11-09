use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, ExecutionOptions, PipelineBuilder, RollingOperator, RollingSpec,
    UdfRegistry,
};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, Float32Array, Float64Array, StringArray, TimestampMicrosecondArray,
        UInt64Array,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::{Value, json};

fn document(outputs: Value) -> Value {
    json!({"configuration_version":1, "state_layout_version":5,
        "partition_by":["entity"], "event_time":"ts", "sequence_by":["seq"],
        "outputs": outputs, "allowed_lateness_micros":0,
        "late_policy":{"kind":"error","scope":"envelope"},
        "value_policy":"stateful_numeric_v1"})
}

fn declarations() -> Value {
    json!([
        {"kind":"rsi","primitive_version":1,"input":"price","output":"rsi","period":2},
        {"kind":"atr","primitive_version":1,"high":"high","low":"low","close":"close","output":"atr","period":2}
    ])
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("entity", DataType::Utf8, false),
        Field::new("seq", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
        Field::new("high", DataType::Float32, true),
        Field::new("low", DataType::Float64, true),
        Field::new("close", DataType::Float32, true),
    ]))
}

fn input() -> Batch {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(TimestampMicrosecondArray::from(vec![1, 2, 3, 4]).with_timezone("UTC")),
        Arc::new(StringArray::from(vec!["a"; 4])),
        Arc::new(UInt64Array::from(vec![1, 2, 3, 4])),
        Arc::new(Float64Array::from(vec![1.0, 2.0, 1.0, 3.0])),
        Arc::new(Float32Array::from(vec![10.0, 12.0, 11.0, 14.0])),
        Arc::new(Float64Array::from(vec![8.0, 9.0, 8.0, 11.0])),
        Arc::new(Float32Array::from(vec![9.0, 10.0, 9.0, 13.0])),
    ];
    Batch::table(
        vec![RecordBatch::try_new(schema(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

#[test]
fn test_indicator_declarations_roundtrip_and_require_layout_five() {
    let value = document(declarations());
    let spec: RollingSpec = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(serde_json::to_value(&spec).unwrap(), value);
    assert_eq!(
        RollingOperator::new("features", schema(), spec)
            .unwrap()
            .state_layout_version(),
        5
    );
    let mut old = value;
    old["state_layout_version"] = json!(4);
    let spec = serde_json::from_value(old).unwrap();
    assert!(
        RollingOperator::new("features", schema(), spec)
            .unwrap_err()
            .to_string()
            .contains("layout")
    );
}

#[tokio::test]
async fn test_table_rsi_atr_use_fixed_ieee_seed_with_mixed_floating_operands() {
    let spec: RollingSpec = serde_json::from_value(document(declarations())).unwrap();
    let operator = RollingOperator::new("features", schema(), spec).unwrap();
    let plan = PipelineBuilder::new("indicators")
        .unwrap()
        .add_node("features", operator)
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap();
    let result = plan
        .execute(
            BTreeMap::from([("input".into(), input())]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    let batch = &result.outputs["output"].table_payload().unwrap().batches()[0];
    for (name, expected) in [
        ("rsi", vec![None, None, Some(50.0), Some(100.0 / 1.2)]),
        ("atr", vec![None, None, Some(3.0), Some(4.0)]),
    ] {
        let values = batch
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(values.iter().collect::<Vec<_>>(), expected);
        assert_eq!(values.null_count(), 2);
    }
}

fn with_values(input: Batch, name: &str, values: ArrayRef) -> Batch {
    let record = &input.table_payload().unwrap().batches()[0];
    let mut columns = record.columns().to_vec();
    columns[record.schema().index_of(name).unwrap()] = values;
    Batch::table(
        vec![RecordBatch::try_new(record.schema(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

#[tokio::test]
async fn test_atr_infinity_is_rejected_before_buffering_even_with_missing_close() {
    use calc_flow::{
        CancellationToken, EdgeCollector, JsonMap, OperatorMetadata, StreamJobContext,
        StreamOperator, StreamOperatorContext,
    };
    let spec = serde_json::from_value(document(declarations())).unwrap();
    let mut operator = RollingOperator::new("features", schema(), spec).unwrap();
    let job = StreamJobContext::new(
        1,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let batch = with_values(
        input(),
        "high",
        Arc::new(Float32Array::from(vec![f32::INFINITY; 4])),
    );
    let batch = with_values(batch, "close", Arc::new(Float32Array::from(vec![None; 4])));
    let error = operator
        .process_data("input", batch, &context, &mut output)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("high"), "{error}");
    assert!(output.drain("output").is_empty());
    operator
        .process_data("input", input(), &context, &mut output)
        .await
        .unwrap();
    operator.on_end(&context, &mut output).await.unwrap();
    assert_eq!(
        output
            .drain("output")
            .iter()
            .map(|message| message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>())
            .sum::<usize>(),
        4
    );
}

#[tokio::test]
async fn test_numerical_failure_preserves_pending_rows_for_whole_finalization_callback() {
    use calc_flow::{
        CancellationToken, EdgeCollector, EventTime, JsonMap, OperatorMetadata, StreamJobContext,
        StreamOperator, StreamOperatorContext,
    };
    for ending in [false, true] {
        let spec = serde_json::from_value(document(declarations())).unwrap();
        let mut operator = RollingOperator::new("features", schema(), spec).unwrap();
        let job = StreamJobContext::new(
            1,
            "a".repeat(64),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "features", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        let batch = with_values(
            input(),
            "price",
            Arc::new(Float64Array::from(vec![1.0, 2.0, -1e308, 1e308])),
        );
        operator
            .process_data("input", batch, &context, &mut output)
            .await
            .unwrap();
        for _ in 0..2 {
            let result = if ending {
                operator.on_end(&context, &mut output).await
            } else {
                operator
                    .on_watermark(EventTime::from_micros(4), &context, &mut output)
                    .await
            };
            let error = result.unwrap_err();
            assert!(error.to_string().contains("price"), "{error}");
            assert!(output.drain("output").is_empty());
        }
    }
}

#[tokio::test]
async fn test_finite_indicator_peak_budget_covers_candidate_and_output() {
    use calc_flow::StateBudget;
    let spec = serde_json::from_value(document(declarations())).unwrap();
    let mut operator = RollingOperator::new("features", schema(), spec).unwrap();
    operator
        .set_state_budget(StateBudget::new(128, 512).unwrap())
        .unwrap();
    let plan = PipelineBuilder::new("bounded indicators")
        .unwrap()
        .add_node("features", operator)
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap();
    let error = plan
        .execute(
            BTreeMap::from([("input".into(), input())]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("budget"), "{error}");
}

fn slice_input(offset: usize, length: usize) -> Batch {
    let batch = input();
    let record = batch.table_payload().unwrap().batches()[0].slice(offset, length);
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

#[tokio::test]
async fn test_layout_five_restores_full_indicator_prefix_in_new_operator() {
    use calc_flow::{
        CancellationToken, EdgeCollector, Epoch, EventTime, JsonMap, OperatorMetadata,
        StreamJobContext, StreamOperator, StreamOperatorContext,
    };
    let spec: RollingSpec = serde_json::from_value(document(declarations())).unwrap();
    let mut first = RollingOperator::new("features", schema(), spec.clone()).unwrap();
    let job = StreamJobContext::new(
        1,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(first.output_ports().to_vec());
    first
        .process_data("input", slice_input(0, 3), &context, &mut output)
        .await
        .unwrap();
    first
        .on_watermark(EventTime::from_micros(3), &context, &mut output)
        .await
        .unwrap();
    output.drain("output");
    first.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = first.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert_eq!(
        snapshot.segments.len(),
        1,
        "constant-state groups still require durable records"
    );
    let mut restored = RollingOperator::new("features", schema(), spec).unwrap();
    restored.restore_async(&snapshot, &context).await.unwrap();
    restored
        .process_data("input", slice_input(3, 1), &context, &mut output)
        .await
        .unwrap();
    restored.on_end(&context, &mut output).await.unwrap();
    let messages = output.drain("output");
    let record = &messages[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    let rsi = record
        .column_by_name("rsi")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let atr = record
        .column_by_name("atr")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(rsi.iter().collect::<Vec<_>>(), vec![Some(100.0 / 1.2)]);
    assert_eq!(atr.iter().collect::<Vec<_>>(), vec![Some(4.0)]);
}

#[tokio::test]
async fn test_finalization_peak_rejection_preserves_pending_rows_for_retry() {
    use calc_flow::{
        CancellationToken, EdgeCollector, JsonMap, OperatorMetadata, StateBudget, StreamJobContext,
        StreamOperator, StreamOperatorContext,
    };
    let spec = serde_json::from_value(document(declarations())).unwrap();
    let mut operator = RollingOperator::new("features", schema(), spec).unwrap();
    operator
        .set_state_budget(StateBudget::new(128, 4096).unwrap())
        .unwrap();
    let job = StreamJobContext::new(
        1,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", input(), &context, &mut output)
        .await
        .unwrap();
    let error = operator.on_end(&context, &mut output).await.unwrap_err();
    assert!(error.to_string().contains("budget"), "{error}");
    assert!(output.drain("output").is_empty());
    operator
        .set_state_budget(StateBudget::new(128, 65536).unwrap())
        .unwrap();
    operator.on_end(&context, &mut output).await.unwrap();
    assert_eq!(
        output
            .drain("output")
            .iter()
            .map(|message| message.as_data().unwrap().num_rows())
            .sum::<usize>(),
        4
    );
}

#[tokio::test]
async fn test_indicator_capture_credit_survives_reset_until_last_snapshot_owner_drops() {
    use calc_flow::{
        CancellationToken, EdgeCollector, Epoch, JsonMap, OperatorMetadata, StateBudget,
        StreamJobContext, StreamOperator, StreamOperatorContext,
    };
    let spec = serde_json::from_value(document(declarations())).unwrap();
    let mut operator = RollingOperator::new("features", schema(), spec).unwrap();
    let job = StreamJobContext::new(
        1,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", input(), &context, &mut output)
        .await
        .unwrap();
    operator.on_end(&context, &mut output).await.unwrap();
    output.drain("output");
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let shared_snapshot = snapshot.clone();
    operator.reset().unwrap();
    let budget = StateBudget::new(128, 1).unwrap();
    assert!(operator.set_state_budget(budget).is_err());
    drop(snapshot);
    assert!(operator.set_state_budget(budget).is_err());
    drop(shared_snapshot);
    operator.set_state_budget(budget).unwrap();
}

#[tokio::test]
async fn test_indicator_capture_requires_preparation_of_current_revision() {
    use calc_flow::{
        CancellationToken, EdgeCollector, Epoch, JsonMap, OperatorMetadata, StreamJobContext,
        StreamOperator, StreamOperatorContext,
    };
    let spec = serde_json::from_value(document(declarations())).unwrap();
    let mut operator = RollingOperator::new("features", schema(), spec).unwrap();
    let job = StreamJobContext::new(
        1,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", slice_input(0, 3), &context, &mut output)
        .await
        .unwrap();
    let Err(error) = operator.checkpoint(Epoch::new(1).unwrap()) else {
        panic!("unprepared revision must reject capture");
    };
    assert!(error.to_string().contains("prepar"), "{error}");
    operator.prepare_checkpoint_async(&context).await.unwrap();
    operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    operator
        .process_data("input", slice_input(3, 1), &context, &mut output)
        .await
        .unwrap();
    let error = operator.checkpoint(Epoch::new(2).unwrap()).unwrap_err();
    assert!(error.to_string().contains("prepar"), "{error}");
    operator.prepare_checkpoint_async(&context).await.unwrap();
    operator.checkpoint(Epoch::new(2).unwrap()).unwrap();
}

#[tokio::test]
async fn test_indicator_peak_counts_shared_nested_input_backing_once() {
    use calc_flow::{
        CancellationToken, EdgeCollector, JsonMap, OperatorMetadata, StateBudget, StreamJobContext,
        StreamOperator, StreamOperatorContext,
    };
    use datafusion::arrow::{array::ListArray, buffer::OffsetBuffer};
    let input = input();
    let record = &input.table_payload().unwrap().batches()[0];
    let context_values: ArrayRef = Arc::new(Float64Array::from(vec![42.0; 32768]));
    let context = ListArray::new(
        Arc::new(Field::new("item", DataType::Float64, true)),
        OffsetBuffer::new(vec![0, 8192, 16384, 24576, 32768].into()),
        context_values,
        None,
    );
    let mut fields = record.schema().fields().to_vec();
    fields.push(Arc::new(Field::new(
        "context",
        context.data_type().clone(),
        true,
    )));
    let schema = Arc::new(Schema::new(fields));
    let mut columns = record.columns().to_vec();
    columns.push(Arc::new(context));
    let batch = Batch::table(
        vec![RecordBatch::try_new(schema.clone(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap();
    let spec = serde_json::from_value(document(declarations())).unwrap();
    let mut operator = RollingOperator::new("features", schema, spec).unwrap();
    operator
        .set_state_budget(StateBudget::new(128, 768 << 10).unwrap())
        .unwrap();
    let job = StreamJobContext::new(
        1,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", batch, &context, &mut output)
        .await
        .unwrap();
    operator.on_end(&context, &mut output).await.unwrap();
    assert_eq!(
        output
            .drain("output")
            .iter()
            .map(|message| message.as_data().unwrap().num_rows())
            .sum::<usize>(),
        4
    );
}

#[tokio::test]
async fn test_cancelled_indicator_input_leaves_no_pending_rows_for_retry() {
    use calc_flow::{
        CalcFlowError, CancellationToken, EdgeCollector, JsonMap, OperatorMetadata,
        StreamJobContext, StreamOperator, StreamOperatorContext,
    };
    let spec = serde_json::from_value(document(declarations())).unwrap();
    let mut operator = RollingOperator::new("features", schema(), spec).unwrap();
    let token = CancellationToken::new();
    token.cancel();
    let cancelled = StreamJobContext::new(1, "a".repeat(64), JsonMap::new(), None, token);
    let context = StreamOperatorContext::new(&cancelled, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let Err(error) = operator
        .process_data("input", input(), &context, &mut output)
        .await
    else {
        panic!("cancelled input must not commit pending rows");
    };
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    let retry = StreamJobContext::new(
        2,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&retry, "features", None);
    operator
        .process_data("input", input(), &context, &mut output)
        .await
        .unwrap();
    operator.on_end(&context, &mut output).await.unwrap();
    assert_eq!(output.drain("output")[0].as_data().unwrap().num_rows(), 4);
}

#[test]
fn test_indicator_input_budget_rejects_before_copying_large_owned_strings() {
    let text = "x".repeat(1 << 20);
    let extra: ArrayRef = Arc::new(StringArray::from(vec![text.as_str(); 4]));
    assert_input_budget_before_copy(extra);
}

#[test]
fn test_indicator_input_budget_rejects_repeated_dictionary_value_expansion_before_copy() {
    use datafusion::arrow::{array::StringDictionaryBuilder, datatypes::Int8Type};
    let text = "x".repeat(1 << 20);
    let mut extra = StringDictionaryBuilder::<Int8Type>::new();
    for _ in 0..4 {
        extra.append(&text).unwrap();
    }
    assert_input_budget_before_copy(Arc::new(extra.finish()));
}

#[test]
fn test_indicator_input_budget_rejects_null_fixed_list_before_child_materialization() {
    use datafusion::arrow::{array::FixedSizeListArray, buffer::NullBuffer};
    let width = 131_072;
    let values: ArrayRef = Arc::new(Float64Array::from(vec![42.0; 4 * width]));
    let extra = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float64, true)),
        i32::try_from(width).unwrap(),
        values,
        Some(NullBuffer::from(vec![false; 4])),
    );
    assert_input_budget_before_copy(Arc::new(extra));
}

fn assert_input_budget_before_copy(extra: ArrayRef) {
    use calc_flow::{
        CancellationToken, EdgeCollector, JsonMap, OperatorMetadata, StateBudget, StreamJobContext,
        StreamOperator, StreamOperatorContext,
    };
    let source = input();
    let record = &source.table_payload().unwrap().batches()[0];
    let mut fields = record.schema().fields().to_vec();
    fields.push(Arc::new(Field::new(
        "notes",
        extra.data_type().clone(),
        extra.null_count() > 0,
    )));
    let schema = Arc::new(Schema::new(fields));
    let mut columns = record.columns().to_vec();
    columns.push(extra);
    let batch = Batch::table(
        vec![RecordBatch::try_new(schema.clone(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap();
    let spec = serde_json::from_value(document(declarations())).unwrap();
    let mut operator = RollingOperator::new("features", schema, spec).unwrap();
    operator
        .set_state_budget(StateBudget::new(128, 64 << 10).unwrap())
        .unwrap();
    let job = StreamJobContext::new(
        1,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut result = Ok(());
    let measured = allocation_counter::measure(|| {
        result = runtime.block_on(operator.process_data("input", batch, &context, &mut output));
    });
    assert!(result.unwrap_err().to_string().contains("budget"));
    assert!(
        measured.bytes_total < 65_536,
        "rejected input copied large string payload: {}",
        measured.bytes_total
    );
}

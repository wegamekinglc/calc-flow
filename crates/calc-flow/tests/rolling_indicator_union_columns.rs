//! Indicator execution preserves dense Union payloads alongside numeric outputs.

use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, ExecutionOptions, JsonMap,
    OperatorMetadata, PipelineBuilder, RollingOperator, StreamJobContext, StreamOperator,
    StreamOperatorContext, UdfRegistry,
};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, Float64Array, Int64Array, StringArray, StringViewArray,
        TimestampMicrosecondArray, UInt64Array, UnionArray,
    },
    datatypes::{DataType, Field, Schema, TimeUnit, UnionFields},
    record_batch::RecordBatch,
};
use serde_json::json;

const TEXT: &str = "0123456789abcdefghijklmnopqrstuv";

fn input() -> Batch {
    let fields = UnionFields::try_new(
        [3, 127],
        [
            Field::new("text", DataType::Utf8View, false),
            Field::new("number", DataType::Int64, false),
        ],
    )
    .unwrap();
    let auxiliary = UnionArray::try_new(
        fields,
        vec![3_i8, 127].into(),
        Some(vec![0_i32, 0].into()),
        vec![
            Arc::new(StringViewArray::from(vec![TEXT])),
            Arc::new(Int64Array::from(vec![7])),
        ],
    )
    .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("entity", DataType::Utf8, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::UInt64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("auxiliary", auxiliary.data_type().clone(), false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec!["a", "a"])),
        Arc::new(TimestampMicrosecondArray::from(vec![1, 2]).with_timezone("UTC")),
        Arc::new(UInt64Array::from(vec![1, 2])),
        Arc::new(Float64Array::from(vec![1.0, 2.0])),
        Arc::new(auxiliary),
    ];
    Batch::table(
        vec![RecordBatch::try_new(schema, columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn operator(batch: &Batch) -> RollingOperator {
    let spec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":5,
        "partition_by":["entity"],"event_time":"ts","sequence_by":["seq"],
        "outputs":[{"kind":"rsi","primitive_version":1,"input":"price","output":"rsi","period":1}],
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},
        "value_policy":"stateful_numeric_v1"
    }))
    .unwrap();
    RollingOperator::new(
        "features",
        batch.table_payload().unwrap().schema().clone(),
        spec,
    )
    .unwrap()
}

fn assert_auxiliary(batch: &Batch) {
    let records = batch.table_payload().unwrap().batches();
    assert_eq!(records.len(), 1);
    let values = records[0]
        .column_by_name("auxiliary")
        .unwrap()
        .as_any()
        .downcast_ref::<UnionArray>()
        .unwrap();
    assert_eq!(values.type_ids().as_ref(), &[3, 127]);
    let text = values
        .child(3)
        .as_any()
        .downcast_ref::<StringViewArray>()
        .unwrap();
    assert_eq!(text.value(values.value_offset(0)), TEXT);
    let number = values
        .child(127)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(number.value(values.value_offset(1)), 7);
}

fn assert_output(batch: &Batch) {
    assert_auxiliary(batch);
    let records = batch.table_payload().unwrap().batches();
    let values = records[0]
        .column_by_name("rsi")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(values.iter().collect::<Vec<_>>(), [None, Some(100.0)]);
}

#[tokio::test]
async fn test_finite_indicator_preserves_union_views_and_high_type_ids() {
    let input = input();
    let plan = PipelineBuilder::new("indicator-union-columns")
        .unwrap()
        .add_node("features", operator(&input))
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap();
    let result = plan
        .execute(
            BTreeMap::from([("input".into(), input.clone())]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    assert_output(&result.outputs["output"]);
    assert_auxiliary(&input);
}

#[tokio::test]
async fn test_owned_stream_finalization_preserves_union_columns() {
    let input = input();
    let mut operator = operator(&input);
    let job = StreamJobContext::new(
        17,
        "f".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut outputs = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", input.clone(), &context, &mut outputs)
        .await
        .unwrap();
    operator.on_end(&context, &mut outputs).await.unwrap();
    let outputs = outputs.drain("output");
    assert_eq!(outputs.len(), 1);
    assert_output(outputs[0].as_data().unwrap());
    assert_auxiliary(&input);
}

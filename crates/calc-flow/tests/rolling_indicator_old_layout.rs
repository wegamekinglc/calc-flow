//! Explicit old-only layout 5 retains legacy scalar ownership and budgets.

use std::sync::Arc;

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, EventTime, JsonMap,
    OperatorMetadata, RollingOperator, RollingSpec, StateBudget, StreamJobContext, StreamOperator,
    StreamOperatorContext,
};
use datafusion::arrow::{
    array::{
        ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array, new_null_array,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::json;

fn schema() -> Arc<Schema> {
    let mut fields = vec![
        Field::new("entity", DataType::Utf8, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::UInt64, false),
        Field::new("price", DataType::Float64, false),
    ];
    let list = DataType::List(Arc::new(Field::new("item", DataType::UInt8, true)));
    fields.extend((0..8).map(|index| Field::new(format!("aux_{index}"), list.clone(), true)));
    Arc::new(Schema::new(fields))
}
fn operator() -> RollingOperator {
    let spec:RollingSpec=serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":5,"partition_by":["entity"],
        "event_time":"ts","sequence_by":["seq"],
        "outputs":[{"kind":"lag","primitive_version":1,"input":"price","output":"previous","periods":1}],
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},
        "value_policy":"stateful_numeric_v1"
    })).unwrap();
    let mut operator = RollingOperator::new("features", schema(), spec).unwrap();
    operator
        .set_state_budget(StateBudget::new(32, 8 << 10).unwrap())
        .unwrap();
    operator
}
fn input() -> Batch {
    let schema = schema();
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec!["a"])),
        Arc::new(TimestampMicrosecondArray::from(vec![1]).with_timezone("UTC")),
        Arc::new(UInt64Array::from(vec![1])),
        Arc::new(Float64Array::from(vec![2.0])),
    ];
    columns.extend(
        schema
            .fields()
            .iter()
            .skip(4)
            .map(|field| new_null_array(field.data_type(), 1)),
    );
    Batch::table(
        vec![RecordBatch::try_new(schema, columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}
#[tokio::test]
async fn test_old_only_layout_five_nested_null_projection_keeps_retained_budget() {
    let mut source = operator();
    let job = StreamJobContext::new(
        41,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(source.output_ports().to_vec());
    source
        .process_data("input", input(), &context, &mut output)
        .await
        .unwrap();
    source
        .on_watermark(EventTime::from_micros(2), &context, &mut output)
        .await
        .unwrap();
    let snapshot = source.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert!(!snapshot.segments.is_empty());
    let mut restored = operator();
    restored
        .restore(&snapshot)
        .expect("old-only NULL projection must not retain the entire IPC block per scalar");
}

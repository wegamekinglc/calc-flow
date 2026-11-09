use crate::{OperatorStateSnapshot, RollingOperator, RollingSpec, StateBudget, StreamOperator};
use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use serde_json::json;
use std::sync::Arc;

fn operator() -> RollingOperator {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
    ]));
    let spec:RollingSpec=serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":5,"partition_by":["seq"],
        "event_time":"time","sequence_by":["seq"],
        "outputs":[{"kind":"rsi","primitive_version":1,"input":"price","output":"rsi","period":2}],
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},"value_policy":"stateful_numeric_v1"
    })).unwrap();
    let mut operator = RollingOperator::new("features", schema, spec).unwrap();
    operator
        .set_state_budget(StateBudget::new(32, 64 << 10).unwrap())
        .unwrap();
    operator
}
#[test]
fn test_indicator_restore_rejects_metadata_before_copying_unfunded_strings() {
    let mut operator = operator();
    let mut snapshot = OperatorStateSnapshot::default();
    snapshot
        .inline_metadata
        .insert("unknown".into(), json!("x".repeat(1 << 20)));
    let mut error = None;
    let actual = allocation_counter::measure(|| {
        error = StreamOperator::restore(&mut operator, &snapshot).err();
    });
    assert!(error.is_some());
    assert!(
        actual.bytes_max <= 64 << 10,
        "unfunded restore allocation: {actual:?}"
    );
}

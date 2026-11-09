use super::*;
use crate::operator::rolling::{
    BufferedRow, KeyValue, RollingHistories, RollingOperator, RollingSpec, compute_output_columns,
    state_v3,
};
use datafusion::{
    arrow::{
        datatypes::{DataType, Field, TimeUnit},
        ipc::reader::FileReader,
    },
    common::ScalarValue,
};
use serde_json::json;
use std::{io::Cursor, sync::Arc};

fn operator() -> RollingOperator {
    let schema = Arc::new(Schema::new(vec![
        Field::new("entity", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, false),
    ]));
    let spec: RollingSpec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":5,
        "partition_by":["entity"],"event_time":"time","sequence_by":["sequence"],
        "outputs":[
            {"kind":"rsi","primitive_version":1,"input":"price","output":"rsi","period":2},
            {"kind":"min","primitive_version":1,"input":"price","output":"minimum","frame":{"kind":"rows","size":2},"min_periods":1},
            {"kind":"ewma","primitive_version":1,"input":"price","output":"ewma","span":2,"min_periods":1},
            {"kind":"cumulative_sum","primitive_version":1,"input":"price","output":"total","min_periods":1}
        ],
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},
        "value_policy":"stateful_numeric_v1"
    })).unwrap();
    RollingOperator::new("decode", schema, spec).unwrap()
}
fn row(key: &str, time: i64, price: f64) -> BufferedRow {
    BufferedRow::new(
        vec![Some(KeyValue::String(key.into()))],
        vec![KeyValue::Unsigned(u64::try_from(time).unwrap())],
        time,
        vec![
            ScalarValue::Utf8(Some(key.into())),
            ScalarValue::TimestampMicrosecond(Some(time), Some("UTC".into())),
            ScalarValue::UInt64(Some(u64::try_from(time).unwrap())),
            ScalarValue::Float64(Some(price)),
        ],
    )
}
fn record(operator: &RollingOperator) -> RecordBatch {
    let rows = [
        complete(row("entity b", 1, 1.0), operator),
        complete(row("entity a", 2, 2.0), operator),
        complete(row("entity b", 3, 3.0), operator),
    ];
    let computed = compute_output_columns(
        &rows,
        &RollingHistories::default(),
        &operator.compiled,
        "decode",
    )
    .unwrap();
    let mut histories = RollingHistories::default();
    histories.apply(computed.touched);
    let mut buffer = std::collections::BTreeMap::new();
    let pending = complete(row("pending", 10, 7.0), operator);
    buffer.insert(pending.identity.clone(), pending);
    let bytes = state_v3::encode(
        &histories,
        &buffer,
        operator.input_ports[0].schema().unwrap(),
        &operator.compiled,
        "fingerprint",
        "decode",
    )
    .unwrap();
    FileReader::try_new(Cursor::new(bytes), None)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
}
fn complete(mut row: BufferedRow, operator: &RollingOperator) -> BufferedRow {
    for field in operator.input_ports[0]
        .schema()
        .unwrap()
        .fields()
        .iter()
        .skip(row.values.len())
    {
        row.values
            .push(ScalarValue::try_new_null(field.data_type()).unwrap());
    }
    row
}
fn verify(operator: &RollingOperator, record: &RecordBatch) {
    let schema = operator.input_ports[0].schema().unwrap();
    let mut quote = None;
    let planning = allocation_counter::measure(|| {
        quote = Some(plan(record, schema, &operator.compiled, &mut || Ok(())).unwrap());
    });
    assert_eq!(planning.count_total, 0);
    let quote = quote.unwrap();
    let mut decoded = None;
    let actual = allocation_counter::measure(|| {
        decoded = Some(
            state_v3::decode_record_checked(
                record,
                schema,
                &operator.compiled,
                "decode",
                &mut || Ok(()),
            )
            .unwrap(),
        );
    });
    assert!(quote.candidate_bytes >= u64::try_from(actual.bytes_current).unwrap());
    assert!(
        quote.candidate_bytes + quote.scratch_bytes >= actual.bytes_max,
        "candidate={}, scratch={}, actual={actual:?}",
        quote.candidate_bytes,
        quote.scratch_bytes
    );
    let pending = decoded.unwrap().buffer.len();
    assert_eq!(pending, usize::from(record.num_rows() > 0));
}
#[test]
fn test_restore_plan_checks_cancellation_without_allocating_state() {
    let operator = operator();
    let record = record(&operator);
    let error = plan(
        &record,
        operator.input_ports[0].schema().unwrap(),
        &operator.compiled,
        &mut || {
            Err(CalcFlowError::Cancelled {
                run_id: "restore".into(),
            })
        },
    )
    .err()
    .unwrap();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
}

#[test]
fn test_restore_plan_precedes_actual_scalar_and_window_rebuild_allocations() {
    let operator = operator();
    verify(&operator, &record(&operator));
}
#[test]
fn test_restore_plan_counts_nested_null_wrappers_on_every_control_row() {
    let original = operator();
    let mut fields = original.input_ports[0].schema().unwrap().fields().to_vec();
    let nested = DataType::Dictionary(
        Box::new(DataType::Int8),
        Box::new(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Utf8,
            true,
        )))),
    );
    fields.push(Arc::new(Field::new("unused", nested, true)));
    let operator = RollingOperator::new(
        "decode",
        Arc::new(Schema::new(fields)),
        original.spec.clone(),
    )
    .unwrap();
    verify(&operator, &record(&operator));
}
#[test]
fn test_restore_plan_empty_record_only_needs_decoder_control_workspace() {
    let operator = operator();
    let empty = record(&operator).slice(0, 0);
    verify(&operator, &empty);
    let quote = plan(
        &empty,
        operator.input_ports[0].schema().unwrap(),
        &operator.compiled,
        &mut || Ok(()),
    )
    .unwrap();
    assert_eq!(quote.candidate_bytes, 0);
}

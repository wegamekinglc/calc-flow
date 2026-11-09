use super::*;
use crate::operator::rolling::{
    BufferedRow, RollingHistories, RollingOperator, RollingSpec, compute_output_columns,
    indicators::footprint_state, state_v3,
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
    let input = Arc::new(Schema::new(vec![
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
            {"kind":"mean","primitive_version":1,"input":"price","output":"mean","frame":{"kind":"rows","size":2},"min_periods":1},
            {"kind":"ewma","primitive_version":1,"input":"price","output":"ewma","span":2,"min_periods":1},
            {"kind":"cumulative_sum","primitive_version":1,"input":"price","output":"total","min_periods":1}
        ],
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},
        "value_policy":"stateful_numeric_v1"
    })).unwrap();
    RollingOperator::new("record", input, spec).unwrap()
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
fn state(operator: &RollingOperator) -> RollingStreamState {
    let rows = [row("b", 1, 1.0), row("a", 2, 2.0), row("b", 3, 3.0)];
    let computed = compute_output_columns(
        &rows,
        &RollingHistories::default(),
        &operator.compiled,
        "record",
    )
    .unwrap();
    let mut state = RollingStreamState::default();
    state.histories.apply(computed.touched);
    let pending = row("c", 10, 7.0);
    state.buffer.insert(pending.identity.clone(), pending);
    state
}
fn inputs<'a>(
    state: &'a RollingStreamState,
    source_footprint: &'a Footprint,
    operator: &'a RollingOperator,
) -> Inputs<'a> {
    Inputs {
        state,
        source_footprint,
        input_schema: operator.input_ports[0].schema().unwrap(),
        compiled: &operator.compiled,
        fingerprint: "fingerprint",
        node: "record",
    }
}

#[test]
fn test_checkpoint_record_matches_actual_layout_five_reference_encoder() {
    let operator = operator();
    let state = state(&operator);
    let (_, source) = footprint_state::state(&state, "record").unwrap();
    let expected = state_v3::encode(
        &state.histories,
        &state.buffer,
        operator.input_ports[0].schema().unwrap(),
        &operator.compiled,
        "fingerprint",
        "record",
    )
    .unwrap();
    let expected = FileReader::try_new(Cursor::new(expected), None)
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let mut reservations = 0;
    let prepared = prepare(
        inputs(&state, &source, &operator),
        &mut |quote, scratch| {
            assert!(quote.bytes("record")? + scratch > 0);
            reservations += 1;
            Ok(())
        },
        &mut || Ok(()),
    )
    .unwrap();
    assert!(reservations > 2);
    assert_eq!(prepared.record, expected);
    assert!(Arc::ptr_eq(
        prepared.record.schema_ref(),
        prepared.schema.as_schema()
    ));
    assert!(
        prepared
            .footprint
            .incremental_bytes(&source, "record")
            .unwrap()
            > 0
    );
    assert_eq!(prepared.scratch_bytes, 0);
}

#[test]
fn test_checkpoint_record_admits_before_any_new_owned_allocation() {
    let operator = operator();
    let state = state(&operator);
    let (_, source) = footprint_state::state(&state, "record").unwrap();
    let mut error = None;
    let actual = allocation_counter::measure(|| {
        error = Some(
            prepare(
                inputs(&state, &source, &operator),
                &mut |_, _| {
                    Err(CalcFlowError::Cancelled {
                        run_id: String::new(),
                    })
                },
                &mut || Ok(()),
            )
            .err()
            .unwrap(),
        );
    });
    assert!(matches!(error.unwrap(), CalcFlowError::Cancelled { .. }));
    assert_eq!(actual.count_total, 0);
}

#[test]
fn test_checkpoint_record_empty_is_a_real_zero_row_record() {
    let operator = operator();
    let state = RollingStreamState::default();
    let source = Footprint::default();
    let prepared = prepare(
        inputs(&state, &source, &operator),
        &mut |_, _| Ok(()),
        &mut || Ok(()),
    )
    .unwrap();
    assert_eq!(prepared.record.num_rows(), 0);
    assert_eq!(prepared.record.num_columns(), 12);
    assert_eq!(prepared.record.schema().metadata().len(), 7);
    assert!(
        prepared
            .record
            .columns()
            .iter()
            .all(|column| column.is_empty())
    );
}

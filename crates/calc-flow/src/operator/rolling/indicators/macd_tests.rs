use std::sync::Arc;

use datafusion::arrow::datatypes::{Field, Schema};
use serde_json::json;

use super::{Group, MacdKernel, State, advance, readout_width};
use crate::operator::rolling::{CompiledEvaluation, RollingOperator, RollingSpec};

fn operator(outputs: &serde_json::Value) -> RollingOperator {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "time",
            datafusion::arrow::datatypes::DataType::Timestamp(
                datafusion::arrow::datatypes::TimeUnit::Microsecond,
                Some("UTC".into()),
            ),
            false,
        ),
        Field::new("seq", datafusion::arrow::datatypes::DataType::UInt64, false),
        Field::new(
            "price",
            datafusion::arrow::datatypes::DataType::Float64,
            true,
        ),
    ]));
    let spec: RollingSpec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":6,"partition_by":["seq"],
        "event_time":"time","sequence_by":["seq"],"outputs":outputs,
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},
        "value_policy":"stateful_numeric_v1"
    }))
    .unwrap();
    RollingOperator::new("features", schema, spec).unwrap()
}
fn output(name: &str, component: &str) -> serde_json::Value {
    json!({"kind":"macd_component","primitive_version":1,"input":"price","output":name,"component":component})
}
fn group() -> Group {
    operator(&json!([output("line", "line")]))
        .compiled
        .indicator_groups[0]
        .clone()
}

#[test]
fn test_macd_normalizes_defaults_and_aliases_into_one_complete_group() {
    let mut explicit = output("signal", "signal");
    for (name, value) in [
        ("fast_span", 12),
        ("slow_span", 26),
        ("signal_span", 9),
        ("min_periods", 1),
        ("signal_min_periods", 1),
    ] {
        explicit[name] = json!(value);
    }
    let operator = operator(&json!([
        {"kind":"rsi","primitive_version":1,"input":"price","output":"rsi","period":2},
        output("line", "line"), explicit, output("histogram", "histogram"), output("alias", "line")
    ]));
    assert_eq!(operator.compiled.indicator_groups.len(), 2);
    assert_eq!(
        readout_width(&operator.compiled.indicator_groups).unwrap(),
        4
    );
    let indices = operator
        .compiled
        .outputs
        .iter()
        .map(|output| {
            let CompiledEvaluation::Indicator(index) = output.evaluation else {
                panic!("indicator readout")
            };
            index
        })
        .collect::<Vec<_>>();
    assert_eq!(indices, [0, 1, 2, 3, 1]);
    assert_eq!(
        operator.compiled.indicator_groups[1]
            .fresh()
            .logical_bytes(),
        56
    );
}

#[test]
fn test_macd_group_key_includes_all_five_normalized_parameters() {
    let mut declarations = vec![output("base", "histogram")];
    for (index, (name, value)) in [
        ("fast_span", 11),
        ("slow_span", 27),
        ("signal_span", 10),
        ("min_periods", 30),
        ("signal_min_periods", 30),
    ]
    .into_iter()
    .enumerate()
    {
        let mut declaration = output(&format!("changed_{index}"), "histogram");
        declaration[name] = json!(value);
        declarations.push(declaration);
    }
    let operator = operator(&json!(declarations));
    assert_eq!(operator.compiled.indicator_groups.len(), 6);
    assert_eq!(
        readout_width(&operator.compiled.indicator_groups).unwrap(),
        18
    );
}

#[test]
fn test_macd_codec_preserves_nonfinite_raw_bits_and_rejects_identity_or_count_changes() {
    let group = group();
    let state = State::Macd(MacdKernel {
        positions: 7,
        fast_count: 2,
        fast_value: f64::from_bits(0x7ff8_0000_0000_0042),
        slow_count: 2,
        slow_value: f64::NEG_INFINITY,
        signal_count: 7,
        signal_value: -0.0,
    });
    let bytes = group.encode(&state).unwrap();
    assert_eq!(bytes.len(), 104);
    assert_eq!(group.decode(&bytes, 7).unwrap().positions(), 7);
    assert_eq!(
        group.encode(&group.decode(&bytes, 7).unwrap()).unwrap(),
        bytes
    );
    for offset in [0, 4, 5, 8, 16, 24, 32, 40, 48, 56, 72, 88] {
        let mut invalid = bytes.clone();
        invalid[offset] ^= 0x80;
        assert!(group.decode(&invalid, 7).is_err(), "offset={offset}");
    }
    assert!(group.decode(&bytes[..103], 7).is_err());
}

#[test]
fn test_macd_flat_readouts_advance_once_and_preserve_existing_rsi_encoding() {
    let group = group();
    let mut states = [group.fresh()];
    let values = advance(
        &[group],
        &mut states,
        &[
            datafusion::common::ScalarValue::Null,
            datafusion::common::ScalarValue::Null,
            datafusion::common::ScalarValue::Float64(Some(2.0)),
        ],
        "features",
        1,
    )
    .unwrap();
    assert_eq!(values, [Some(0.0); 3]);
    assert_eq!(states[0].positions(), 1);
    let rsi = Group::Rsi {
        input: 2,
        field: "price".into(),
        period: 2,
    };
    let bytes = rsi.encode(&rsi.fresh()).unwrap();
    let mut expected = vec![0; 64];
    expected[0] = 1;
    expected[4] = 3;
    expected[16] = 2;
    assert_eq!(bytes, expected);
}

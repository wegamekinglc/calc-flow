use serde_json::{Value, json};

use crate::JsonMap;

fn object(properties: Value, required: &[&str]) -> Value {
    Value::Object(
        [
            ("type".into(), "object".into()),
            ("properties".into(), properties),
            ("required".into(), json!(required)),
            ("additionalProperties".into(), false.into()),
        ]
        .into_iter()
        .collect(),
    )
}
fn positive() -> Value {
    json!({"type":"integer","minimum":1,"maximum":u64::MAX})
}
fn minimum() -> Value {
    let mut value = positive();
    value["default"] = json!(1);
    value
}
fn kernel(version: u64) -> Value {
    let mut values = vec![
        object(
            json!({"function":{"const":"rolling_mean"}, "window":positive(),
            "min_periods":minimum(), "numerical_profile":{"enum":["stable_v1","stable_v2"],"default":"stable_v1"}}),
            &["function", "window"],
        ),
        object(
            json!({"function":{"const":"ewma"}, "span":positive(), "min_periods":minimum()}),
            &["function", "span"],
        ),
    ];
    if version >= 2 {
        values.push(object(
            json!({"function":{"const":"rsi"},"period":positive()}),
            &["function", "period"],
        ));
        values.push(object(
            json!({"function":{"const":"atr"},"period":positive()}),
            &["function", "period"],
        ));
    }
    if version >= 3 {
        values.push(macd_kernel());
    }
    json!({"oneOf": values})
}
fn positive_default(default: u64) -> Value {
    let mut value = positive();
    value["default"] = default.into();
    value
}
fn macd_kernel() -> Value {
    object(
        json!({"function":{"const":"macd_components"},
        "fast_span":positive_default(12),"slow_span":positive_default(26),
        "signal_span":positive_default(9),"min_periods":positive_default(1),
        "signal_min_periods":positive_default(1)}),
        &["function"],
    )
}
fn count(default: usize) -> Value {
    json!({"type":"integer","minimum":0,"maximum":usize::MAX,"default":default})
}
fn versioned(version: &Value, kernels: u64) -> Value {
    object(
        json!({
            "ir_version":version, "input_dtype":{"enum":["float32","float64"]},
            "kernel":kernel(kernels),
            "execution":object(json!({"max_elements":count(10_000_000),"max_allocation_bytes":count(256*1024*1024)}), &[]),
            "state":object(json!({"max_state_bytes":count(256*1024*1024)}), &[])
        }),
        &["ir_version", "input_dtype", "kernel"],
    )
}
fn options() -> Value {
    let mut schema = versioned(&json!({"enum":[1,2,3]}), 3);
    schema["oneOf"] = json!([
        versioned(&json!({"const":1}), 1),
        versioned(&json!({"const":2}), 2),
        versioned(&json!({"const":3}), 3)
    ]);
    schema
}
pub(super) fn descriptor() -> JsonMap {
    let descriptor = json!({
        "provider":"native", "name":"array_sequence", "version":"1",
        "port_binding":"fixed",
        "input_ports":[{"name":"input","kind":"array","required":true}],
        "output_ports":[{"name":"output","kind":"array","required":true}],
        "options_schema":null,
        "options_json_schema":crate::json::canonical_json(&options()).expect("schema contains only JSON data"),
        "modes":["stream"], "stateful":true, "microbatch_invariant":true,
        "requires_watermark":false, "finality":"per_row_final",
        "checkpoint_support":"checkpointed_stateful", "state_version":1, "state_layouts":[1,2,3],
        "deterministic":true, "replay_safe":true, "supports_static_inputs":false,
        "partition_contract":"single_ordered_sequence",
        "array_rules":{"supported_dtypes":["float32","float64"],
            "safe_dtype_rule":{"name":"native_float_dtype","version":"1"},
            "shape_rules":[{"name":"native_sequence_vector","version":"1"},{"name":"native_sequence_indicators","version":"2"},{"name":"native_sequence_indicators","version":"3"}]}
    });
    serde_json::from_value(descriptor).expect("descriptor is a JSON object")
}

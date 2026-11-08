use serde_json::{Value, json};

use crate::{JsonMap, json::canonical_json};

fn object(properties: Vec<(&str, Value)>, required: &[&str]) -> Value {
    let properties: serde_json::Map<_, _> = properties
        .into_iter()
        .map(|(name, value)| (name.to_owned(), value))
        .collect();
    json!({"type":"object","properties":properties,"required":required,"additionalProperties":false})
}
fn identifier() -> Value {
    json!({"type":"string","minLength":1,"maxLength":64,"pattern":"^[A-Za-z0-9_.-]+$"})
}
fn dtype() -> Value {
    json!({"enum":["bool","float32","float64"]})
}
fn shape(rank: u8) -> Value {
    json!({"type":"array","maxItems":rank,"items":{"anyOf":[{"type":"null"},{"type":"integer","minimum":0,"maximum":usize::MAX}]}})
}
fn positive() -> Value {
    json!({"type":"integer","minimum":1,"maximum":u64::MAX})
}
fn optional_count() -> Value {
    let mut value = positive();
    value["default"] = json!(1);
    value
}
fn input(rank: u8) -> Value {
    object(
        vec![
            ("name", identifier()),
            ("dtype", dtype()),
            ("shape", shape(rank)),
        ],
        &["name", "dtype", "shape"],
    )
}
fn output(rank: u8) -> Value {
    object(
        vec![
            ("name", identifier()),
            ("node", identifier()),
            ("dtype", dtype()),
            ("shape", shape(rank)),
        ],
        &["name", "node", "dtype", "shape"],
    )
}
fn node(op: &str, properties: Vec<(&str, Value)>, required: &[&str]) -> Value {
    let mut fields = vec![("id", identifier()), ("op", json!({"const":op}))];
    fields.extend(properties);
    let mut names = vec!["id", "op"];
    names.extend_from_slice(required);
    object(fields, &names)
}
fn literals() -> Vec<Value> {
    let mut number = json!({
        "type": "number",
        "description": concat!(
            "Finite serde_json numeric literal. Retained i64/u64 representations must lie in ",
            "[-9007199254740992,9007199254740992]; finite f64 representations retain their full range. ",
            "Upstream parsing can round integer tokens outside i64/u64 to f64; their original spelling cannot be recovered. ",
            "The native factory enforces representation-dependent validation; this value-level JSON Schema cannot express it."
        )
    });
    let wide = number.clone();
    number["exclusiveMinimum"] = json!(-3.402_823_567_797_336_6e38_f64);
    number["exclusiveMaximum"] = json!(3.402_823_567_797_336_6e38_f64);
    [
        ("bool", json!({"type":"boolean"})),
        ("float32", number),
        ("float64", wide),
    ]
    .into_iter()
    .map(|(dtype, value)| {
        node(
            "literal",
            vec![("dtype", json!({"const":dtype})), ("value", value)],
            &["dtype", "value"],
        )
    })
    .collect()
}
fn pointwise() -> Vec<Value> {
    vec![
        node(
            "unary",
            vec![
                (
                    "function",
                    json!({"enum":["neg","abs","sqrt","log","exp","logical_not"]}),
                ),
                ("input", identifier()),
            ],
            &["function", "input"],
        ),
        node(
            "binary",
            vec![
                (
                    "function",
                    json!({"enum":["add","subtract","multiply","divide","pow","equal","not_equal","less","less_equal","greater","greater_equal","logical_and","logical_or","logical_xor"]}),
                ),
                ("left", identifier()),
                ("right", identifier()),
            ],
            &["function", "left", "right"],
        ),
        node(
            "where",
            vec![
                ("condition", identifier()),
                ("if_true", identifier()),
                ("if_false", identifier()),
            ],
            &["condition", "if_true", "if_false"],
        ),
    ]
}
fn aggregates(version: u8) -> Vec<Value> {
    vec![
        reduction(version),
        node(
            "rolling_mean",
            vec![
                ("input", identifier()),
                ("window", positive()),
                ("min_periods", optional_count()),
                (
                    "numerical_profile",
                    json!({"enum":["stable_v1","stable_v2"],"default":"stable_v1"}),
                ),
            ],
            &["input", "window"],
        ),
        node(
            "ewma",
            vec![
                ("input", identifier()),
                ("span", positive()),
                ("min_periods", optional_count()),
            ],
            &["input", "span"],
        ),
    ]
}
fn reduction(version: u8) -> Value {
    let mut fields = vec![
        ("function", json!({"enum":["sum","mean","min","max"]})),
        ("input", identifier()),
        ("skip_nulls", json!({"type":"boolean","default":false})),
    ];
    if version == 2 {
        fields.push((
            "axis",
            json!({"anyOf":[{"type":"null"},{"enum":[0,1]}],"default":null}),
        ));
        fields.push(("keepdims", json!({"type":"boolean","default":false})));
    }
    node("reduce", fields, &["function", "input"])
}
fn layouts() -> Vec<Value> {
    vec![
        node(
            "matmul",
            vec![("left", identifier()), ("right", identifier())],
            &["left", "right"],
        ),
        node(
            "reshape",
            vec![
                ("input", identifier()),
                (
                    "shape",
                    json!({"type":"array","maxItems":2,"items":{"type":"integer","minimum":0,"maximum":usize::MAX}}),
                ),
            ],
            &["input", "shape"],
        ),
        node("transpose", vec![("input", identifier())], &["input"]),
    ]
}
fn execution() -> Value {
    object(
        vec![
            (
                "max_elements",
                json!({"type":"integer","minimum":0,"maximum":usize::MAX,"default":10_000_000}),
            ),
            (
                "max_allocation_bytes",
                json!({"type":"integer","minimum":0,"maximum":usize::MAX,"default":268_435_456}),
            ),
        ],
        &[],
    )
}
fn program(version: u8) -> Value {
    let mut nodes = vec![node("input", vec![("name", identifier())], &["name"])];
    nodes.extend(literals());
    nodes.extend(pointwise());
    nodes.extend(aggregates(version));
    if version == 2 {
        nodes.extend(layouts());
    }
    object(
        vec![
            ("ir_version", json!({"const":version})),
            (
                "inputs",
                json!({"type":"array","minItems":1,"maxItems":32,"items":input(version)}),
            ),
            (
                "outputs",
                json!({"type":"array","minItems":1,"maxItems":16,"items":output(version)}),
            ),
            (
                "nodes",
                json!({"type":"array","minItems":1,"maxItems":128,"items":{"oneOf":nodes}}),
            ),
            ("execution", execution()),
        ],
        &["ir_version", "inputs", "nodes", "outputs"],
    )
}
fn schema() -> Value {
    let mut schema = object(
        vec![
            ("ir_version", json!({})),
            ("inputs", json!({})),
            ("nodes", json!({})),
            ("outputs", json!({})),
            ("execution", json!({})),
        ],
        &["ir_version", "inputs", "nodes", "outputs"],
    );
    schema["oneOf"] = json!([program(1), program(2)]);
    schema["$schema"] = json!("https://json-schema.org/draft/2020-12/schema");
    schema["x-calc-flow-max-compact-json-bytes"] = json!(65536);
    schema["x-calc-flow-max-node-depth"] = json!(24);
    schema["description"] = json!(
        "native:array_expr@1 batch-only IR1 rank-0/1 and IR2 packed rank-0/1/2 programs. IR2 permits at most one matmul; concrete shapes use checked element counts and scalar-or-identical-shape broadcasting. Node references must be acyclic and reachable. Output assertions must equal abstract inference. Pow requires a finite numeric literal exponent of magnitude <=64 before float32 conversion. Runtime concrete extents and shared owned allocation are revalidated."
    );
    schema
}

pub(super) fn descriptor() -> JsonMap {
    let schema = canonical_json(&schema()).expect("bounded fixed schema is canonical JSON");
    serde_json::from_value(json!({
        "provider":"native", "name":"array_expr", "version":"1",
        "port_binding":"options", "input_ports":[], "output_ports":[],
        "options_schema":null, "options_json_schema":schema,
        "modes":["batch"], "stateful":false, "microbatch_invariant":false,
        "requires_watermark":false, "checkpoint_support":"unproven", "state_version":null,
        "supports_static_inputs":false, "partition_contract":"none", "finality":"unproven",
        "deterministic":true, "replay_safe":true,
        "array_rules":{"supported_dtypes":["bool","float32","float64"],
            "safe_dtype_rule":{"name":"native_float_bool_dtype","version":"1"},
            "shape_rules":[{"name":"native_scalar_vector_broadcast","version":"1"},{"name":"native_packed_rank2","version":"1"},{"name":"native_matmul","version":"1"},{"name":"native_axis_reduction","version":"1"}]}
    }))
    .expect("fixed capability descriptor is an object")
}

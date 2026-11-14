use calc_flow::array::{NativeArrayFactory, NativeTableArrayFactory};
use serde_json::{Value, json};

#[test]
fn test_native_capabilities_expose_strict_executable_table_schema_and_static_lifecycles() {
    let array = NativeArrayFactory::capability_descriptor();
    assert_eq!(array["supports_static_inputs"], true);
    assert_eq!(array["modes"], json!(["batch"]));
    assert_eq!(array["checkpoint_support"], "unproven");
    let table = NativeTableArrayFactory::capability_descriptor();
    assert_eq!(table["provider"], "native");
    assert_eq!(table["name"], "table_array");
    assert_eq!(table["version"], "1");
    assert_eq!(table["modes"], json!(["batch", "stream"]));
    assert_eq!(table["stateful"], false);
    assert_eq!(table["microbatch_invariant"], true);
    assert_eq!(table["checkpoint_support"], "stateless");
    assert_eq!(table["state_version"], Value::Null);
    assert_eq!(table["supports_static_inputs"], true);
    assert_eq!(table["partition_contract"], "row_axis_independent");
    assert_eq!(table["finality"], "per_row_final");
    let rules = table["array_rules"]["shape_rules"].as_array().unwrap();
    assert!(rules.contains(&json!({"name":"native_table_static_rhs","version":"1"})));
    let schema: Value =
        serde_json::from_str(table["options_json_schema"].as_str().unwrap()).unwrap();
    assert_eq!(schema["additionalProperties"], false);
    assert_eq!(schema["properties"]["ir_version"]["const"], 2);
    assert_eq!(schema["properties"]["outputs"]["maxItems"], 1);
    let inputs = schema["properties"]["inputs"]["items"]["oneOf"]
        .as_array()
        .unwrap();
    let table_input = inputs
        .iter()
        .find(|input| input["properties"]["kind"]["const"] == "table")
        .unwrap();
    assert_eq!(table_input["additionalProperties"], false);
    assert_eq!(table_input["properties"]["columns"]["minItems"], 1);
    let static_input = inputs
        .iter()
        .find(|input| input["properties"]["kind"]["const"] == "array")
        .unwrap();
    assert_eq!(static_input["properties"]["binding"]["const"], "static");
    assert_eq!(static_input["additionalProperties"], false);
    let nodes = schema["properties"]["nodes"]["items"]["oneOf"]
        .as_array()
        .unwrap();
    for op in [
        "input",
        "literal",
        "unary",
        "binary",
        "where",
        "reduce",
        "rolling_mean",
        "ewma",
        "matmul",
        "reshape",
        "transpose",
    ] {
        assert!(
            nodes
                .iter()
                .any(|node| node["properties"]["op"]["const"] == op),
            "missing {op} schema"
        );
    }
    assert!(
        nodes
            .iter()
            .all(|node| node["additionalProperties"] == false)
    );
}

//! Public project-v3 cutover acceptance tests.

use calc_flow::{PROJECT_FORMAT_VERSION, ProjectSpec, project_json_schema};
use serde_json::json;

#[test]
fn canonical_project_surface_accepts_only_v3_graph_documents() {
    assert_eq!(PROJECT_FORMAT_VERSION, 3);
    let schema = project_json_schema().expect("schema generates");
    assert_eq!(schema["title"], "Calc Flow Project V3");
    assert_eq!(schema["properties"]["format_version"]["const"], 3);

    let v3 = json!({
        "format_version": 3,
        "id": "orders",
        "name": "Orders",
        "runtime": {"mode": "batch", "options": {}},
        "graph": {
            "name": "orders",
            "nodes": [{
                "id": "calc",
                "operator": {"kind": "expression", "expression": "b = a + 1"}
            }]
        },
        "data_sources": [{
            "id": "fixture",
            "input": "input",
            "format": "json",
            "data": [{"a": 1}]
        }]
    });
    serde_json::from_value::<ProjectSpec>(v3).expect("canonical v3 document parses");

    let v2 = json!({
        "format_version": 2,
        "id": "orders",
        "name": "Orders",
        "pipeline": {"name": "orders", "nodes": []}
    });
    let error = serde_json::from_value::<ProjectSpec>(v2).expect_err("v2 fails closed");
    assert!(error.to_string().contains("expected 3"), "{error}");
}

#[test]
fn canonical_v3_schema_matches_the_committed_artifact() {
    let generated = project_json_schema().expect("schema generates");
    let committed = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../schemas/project-v3.schema.json"
    ))
    .expect("committed schema is readable");
    let generated = serde_json::to_string_pretty(&generated).expect("schema encodes");
    assert_eq!(generated, committed);
}

#[test]
fn test_side_output_schema_requires_both_explicit_v1_fields() {
    let schema = project_json_schema().unwrap();
    let variants = schema["$defs"]["LatePolicySpec"]["oneOf"]
        .as_array()
        .unwrap();
    let side = variants
        .iter()
        .find(|variant| variant["properties"]["kind"]["const"] == "side_output")
        .unwrap();
    assert_eq!(side["additionalProperties"], false);
    for field in ["metrics_version", "schema_version"] {
        assert!(side["required"].as_array().unwrap().contains(&json!(field)));
        assert_eq!(side["properties"][field]["minimum"], 1);
        assert_eq!(side["properties"][field]["maximum"], 1);
    }
}

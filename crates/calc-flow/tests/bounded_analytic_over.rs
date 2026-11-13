use calc_flow::{
    AnalyticExpr, NullPlacement, OverOrder, OverSpec, RowsFrame, SortDirection, SortKey,
    StateBudget,
};

#[test]
fn test_over_strict_declarations() {
    let key = || SortKey::new("score", SortDirection::Ascending, NullPlacement::Last).unwrap();
    assert!(OverOrder::new(vec![], vec![key()], vec![]).is_err());
    assert!(OverOrder::new(vec!["score".into()], vec![key()], vec!["seq".into()]).is_err());
    let order = OverOrder::new(vec!["p".into()], vec![key()], vec!["seq".into()]).unwrap();
    let spec = OverSpec::bounded_source(order, StateBudget::new(1000, 8 << 20).unwrap()).unwrap();
    let wire = serde_json::to_value(&spec).unwrap();
    assert_eq!(wire["order"]["sequence_by"], serde_json::json!(["seq"]));
    assert_eq!(wire["history"]["kind"], "bounded_source");
    assert!(serde_json::from_value::<OverSpec>(wire).is_ok());
    assert!(RowsFrame::preceding(u64::from(u32::MAX) + 1).is_err());
    assert!(AnalyticExpr::lag("v", "").is_err());
    assert!(
        serde_json::from_value::<AnalyticExpr>(
            serde_json::json!({"kind":"lag","column":"v","name":"prev","offset":2})
        )
        .is_err()
    );
}

#[test]
fn test_over_schema_limits_match_strict_import_domain() {
    let schema = serde_json::to_value(schemars::schema_for!(OverSpec)).unwrap();
    let budget_ref = schema["properties"]["state_budget"]["$ref"]
        .as_str()
        .unwrap();
    let budget = &schema["$defs"][budget_ref.strip_prefix("#/$defs/").unwrap()];
    assert_eq!(
        budget["properties"]["max_rows"]["maximum"],
        u64::from(u32::MAX)
    );
    let error = OverSpec::bounded_source(
        OverOrder::new(
            vec![],
            vec![SortKey::new("v", SortDirection::Ascending, NullPlacement::Last).unwrap()],
            vec!["seq".into()],
        )
        .unwrap(),
        StateBudget::new(u64::from(u32::MAX) + 1, 8 << 20).unwrap(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("over.spec.state_budget"), "{error}");
}

#[tokio::test]
async fn test_over_schema_error_preserves_logical_relation_role() {
    use calc_flow::arrow::datatypes::{DataType, Field, Schema};
    use calc_flow::{CompileMode, CompileOptions, InputOptions, Program, UdfRegistry, table_input};
    use std::{collections::BTreeMap, sync::Arc};
    let t = table_input(
        "events",
        Arc::new(Schema::new(vec![
            Field::new("v", DataType::Int64, true),
            Field::new("seq", DataType::UInt64, false),
        ])),
        InputOptions::default(),
    )
    .unwrap();
    let order = OverOrder::new(
        vec![],
        vec![SortKey::new("v", SortDirection::Ascending, NullPlacement::Last).unwrap()],
        vec!["seq".into()],
    )
    .unwrap();
    let out = t
        .over(
            vec![AnalyticExpr::lag("missing", "prev").unwrap()],
            OverSpec::bounded_source(order, StateBudget::new(9, 8 << 20).unwrap()).unwrap(),
        )
        .unwrap();
    let p = Program::new(
        "bad",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), out)]),
    )
    .unwrap();
    let error = p
        .compile_batch(&UdfRegistry::new().snapshot(), &CompileOptions::default())
        .await
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("outputs.out.over.expressions[0].column"),
        "{error}"
    );
}

#[test]
fn test_over_schema_requires_nonempty_expression_names_and_operands() {
    let schema = serde_json::to_value(schemars::schema_for!(AnalyticExpr)).unwrap();
    for variant in schema["oneOf"].as_array().unwrap() {
        assert_eq!(variant["properties"]["name"]["minLength"], 1);
        if variant["properties"].get("column").is_some() {
            assert_eq!(variant["properties"]["column"]["minLength"], 1);
        }
    }
}

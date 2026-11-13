use calc_flow::arrow::datatypes::{DataType, Field, Schema};
use calc_flow::{CompileMode, InputOptions, Program, UdfRegistry, table_input};
use std::{collections::BTreeMap, sync::Arc};
#[tokio::test]
async fn test_over_sql_df_canonical_node_and_analysis() {
    use calc_flow::{
        AnalyticExpr, NullPlacement, OverOrder, OverSpec, RowsFrame, SortDirection, SortKey,
        StateBudget,
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Int64, true),
        Field::new("seq", DataType::UInt64, false),
    ]));
    let input = table_input("events", schema, InputOptions::default()).unwrap();
    let query = "SELECT *, ROW_NUMBER() OVER (ORDER BY v, seq) AS rn, LAG(v,1) OVER (ORDER BY v,seq) AS prev, SUM(v) OVER (ORDER BY v,seq ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total FROM CF_OVER_BOUNDED('a',9,8388608,ARRAY['seq'])";
    let sql = calc_flow::sql::sql(query, BTreeMap::from([("a".into(), input.clone())])).unwrap();
    let df = input
        .over(
            vec![
                AnalyticExpr::row_number("rn").unwrap(),
                AnalyticExpr::lag("v", "prev").unwrap(),
                AnalyticExpr::sum("v", RowsFrame::cumulative(), "total").unwrap(),
            ],
            OverSpec::bounded_source(
                OverOrder::new(
                    vec![],
                    vec![SortKey::new("v", SortDirection::Ascending, NullPlacement::Last).unwrap()],
                    vec!["seq".into()],
                )
                .unwrap(),
                StateBudget::new(9, 8 << 20).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let mut configs = vec![];
    let registry = UdfRegistry::new().snapshot();
    for out in [sql, df] {
        let program = Program::new(
            "over",
            CompileMode::Batch,
            BTreeMap::from([("out".into(), out)]),
        )
        .unwrap();
        let project = program.to_project(&registry).await.unwrap();
        let json = serde_json::to_value(project).unwrap();
        configs.push(
            json["graph"]["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .find(|n| n["operator"]["kind"] == "bounded_over")
                .unwrap()["operator"]
                .clone(),
        );
        let report = program
            .analyze(&registry, &calc_flow::AnalysisOptions::default())
            .await
            .unwrap();
        assert_eq!(report.analysis_revision, 11);
    }
    assert_eq!(configs[0], configs[1]);
}

#[test]
fn test_over_sql_rejects_unsupported_functions_frames_and_mixed_shapes() {
    let base = "SELECT *, {expr} FROM CF_OVER_BOUNDED('a',9,8388608,ARRAY['seq'])";
    for expression in [
        "SUM(v) OVER (ORDER BY v,seq) AS total",
        "SUM(v) OVER (ORDER BY v,seq RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total",
        "SUM(v) OVER (ORDER BY v,seq ROWS 1 PRECEDING) AS total",
        "SUM(v) OVER (ORDER BY v,seq ROWS BETWEEN 1 FOLLOWING AND CURRENT ROW) AS total",
        "ROW_NUMBER() OVER (ORDER BY v,seq ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS rn",
        "LAG(v,2) OVER (ORDER BY v,seq) AS prev",
        "LAG(v,1,0) OVER (ORDER BY v,seq) AS prev",
        "LAG(v,1) IGNORE NULLS OVER (ORDER BY v,seq) AS prev",
        "RANK() OVER (ORDER BY v,seq) AS rn",
        "SUM(DISTINCT v) OVER (ORDER BY v,seq ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total",
        "ROW_NUMBER() OVER (ORDER BY v,seq DESC) AS rn",
        "ROW_NUMBER() OVER (ORDER BY v,seq) AS rn, LAG(v,1) OVER (ORDER BY other,seq) AS prev",
    ] {
        let text = base.replace("{expr}", expression);
        let error = calc_flow::sql::over_query(&text, &["a".into()])
            .unwrap_err()
            .to_string();
        assert!(error.contains("sql.over"), "{text}: {error}");
    }
    for text in [
        "SELECT *, ROW_NUMBER() OVER (ORDER BY v,seq) AS rn FROM CF_OVER_BOUNDED('a',9,8388608)",
        "SELECT * FROM (SELECT *, ROW_NUMBER() OVER (ORDER BY v,seq) AS rn FROM CF_OVER_BOUNDED('a',9,8388608,ARRAY['seq'])) x",
        "SELECT DISTINCT *, ROW_NUMBER() OVER (ORDER BY v,seq) AS rn FROM CF_OVER_BOUNDED('a',9,8388608,ARRAY['seq'])",
        "SELECT *, ROW_NUMBER() OVER (ORDER BY v,seq) AS rn FROM CF_OVER_WINDOW('a','ts',0,10,9,8388608,ARRAY['seq'],'DROP')",
    ] {
        assert!(
            calc_flow::sql::over_query(text, &["a".into()]).is_err(),
            "{text}"
        );
    }
}

#[test]
fn test_over_sql_budget_diagnostic_and_legacy_parse_path() {
    let text = "SELECT *, ROW_NUMBER() OVER (ORDER BY v,seq) AS rn FROM CF_OVER_BOUNDED('a',0,8388608,ARRAY['seq'])";
    let error = calc_flow::sql::over_query(text, &["a".into()]).unwrap_err();
    let calc_flow::CalcFlowError::InvalidArgument { field, .. } = error else {
        panic!("literal error expected");
    };
    assert_eq!(field, "sql.over.spec.state_budget.max_rows");
    let error = calc_flow::sql::over_query("SELECT FROM", &["a".into()]).unwrap_err();
    let calc_flow::CalcFlowError::InvalidArgument { field, .. } = error else {
        panic!("parse error expected");
    };
    assert_eq!(field, "sql.query");
}

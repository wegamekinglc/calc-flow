use super::*;
use crate::{
    BatchKind, Edge, InputOptions, PipelineBuilder, Port, PortEndpoint, SqlOperator,
    StreamRequirements, UdfRegistry, count_rows, table_input,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

fn source() -> Result<TableExpr> {
    table_input(
        "events",
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)])),
        InputOptions::default(),
    )
}

#[test]
fn aggregate_stream_mode_error_uses_aggregate_input_path() -> Result<()> {
    let input = source()?;
    let joined = crate::sql::sql(
        "SELECT lhs.x FROM lhs JOIN rhs ON lhs.x = rhs.x",
        BTreeMap::from([("lhs".into(), input.clone()), ("rhs".into(), input)]),
    )?;
    let result = joined.aggregate(vec![count_rows().alias("rows")?], vec![])?;
    let error = Program::new(
        "mode_path",
        CompileMode::Stream,
        BTreeMap::from([("result".into(), result)]),
    )
    .unwrap_err();
    let CalcFlowError::InvalidArgument { field, .. } = error else {
        panic!("{error}");
    };
    assert_eq!(field, "outputs.result.aggregate.value");
    Ok(())
}

#[tokio::test]
async fn missing_aggregate_group_uses_group_role_path() -> Result<()> {
    let result = source()?.aggregate(vec![count_rows().alias("rows")?], vec!["missing".into()])?;
    let program = Program::new(
        "group_path",
        CompileMode::Batch,
        BTreeMap::from([("result".into(), result)]),
    )?;
    let error = program
        .collect_schema(&UdfRegistry::new().snapshot())
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("outputs.result.aggregate.group_by[0]: schema_mismatch:"),
        "{error}"
    );
    Ok(())
}

fn aggregate_operator(id: &str, alias: &str, query: &str) -> Result<SqlOperator> {
    let port = |name| {
        Port::new(
            name,
            BatchKind::Table,
            true,
            Some(vec![Field::new("x", DataType::Int64, true)]),
        )
    };
    SqlOperator::new(id, query, vec![alias.into()], vec![])?
        .with_ports(vec![port(alias)?], port("output")?)
}

fn replacement_graph(alias: &str, query: &str) -> Result<PipelineBuilder> {
    PipelineBuilder::new("replacement_relation")?
        .with_update_semantics(UpdateSemantics::LatestRelation)
        .add_node(
            "upstream",
            Box::new(aggregate_operator(
                "upstream",
                "input",
                "SELECT SUM(x) AS x FROM input",
            )?),
        )?
        .add_node(
            "replace",
            Box::new(aggregate_operator("replace", alias, query)?),
        )?
        .connect(Edge::new(
            PortEndpoint::new("upstream", "output")?,
            PortEndpoint::new("replace", alias)?,
        ))
}

#[test]
fn replacement_rejects_undeclared_relations_and_qualifiers_before_runtime() -> Result<()> {
    for query in [
        "SELECT SUM(x) AS x FROM undeclared",
        "SELECT SUM(x) AS x FROM public.input",
        "SELECT SUM(x) AS x FROM datafusion.public.input",
        "SELECT SUM(undeclared.x) AS x FROM input",
        "SELECT SUM(input.x) AS x FROM input AS renamed",
        "SELECT SUM(renamed.x) AS x FROM undeclared AS renamed",
        "SELECT SUM(x) AS x FROM input AS renamed(other)",
    ] {
        let error = replacement_graph("input", query)?
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .err()
            .unwrap_or_else(|| panic!("unexpected admission: {query}"));
        assert!(
            matches!(error, CalcFlowError::QueryUpdateSemantics { .. }),
            "{query}: {error}"
        );
        assert!(
            error.to_string().contains("declared bare input alias"),
            "{query}: {error}"
        );
    }
    assert!(
        replacement_graph("InputData", "SELECT SUM(x) AS x FROM \"InputData\"")?
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn append_sql_keeps_qualified_reference_admission() -> Result<()> {
    for query in [
        "SELECT SUM(x) AS x FROM public.input",
        "SELECT SUM(x) AS x FROM datafusion.public.input",
    ] {
        replacement_graph("input", query)?
            .with_update_semantics(UpdateSemantics::LegacyAppend)
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )?;
    }
    Ok(())
}

#[test]
fn replacement_accepts_bound_aliases_with_sql_identifier_rules() -> Result<()> {
    for (alias, query) in [
        ("input", "SELECT SUM(x) AS x FROM INPUT"),
        ("input", "SELECT SUM(input.x) AS x FROM input"),
        ("input", "SELECT SUM(renamed.x) AS x FROM input AS renamed"),
        (
            "input",
            "SELECT SUM(\"Renamed\".x) AS x FROM input AS \"Renamed\"",
        ),
        ("InputData", "SELECT SUM(inputdata.x) AS x FROM inputdata"),
    ] {
        replacement_graph(alias, query)?.compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )?;
    }
    Ok(())
}

#[test]
fn replacement_requires_alias_metadata_in_declaration_analysis() -> Result<()> {
    let fields = serde_json::json!([{"name":"x","data_type":"int64","nullable":true}]);
    let base = serde_json::json!({
        "nodes": [
            {"path":"source","kind":"table_input","query":null,"schema":fields},
            {"path":"upstream","kind":"sql","query":"SELECT SUM(x) AS x FROM input","schema":fields},
            {"path":"replace","kind":"sql","query":"SELECT SUM(x) AS x FROM input","schema":fields}
        ],
        "edges": [
            {"path":"upstream.input","source_path":"source","target_path":"upstream","input_alias":"input"},
            {"path":"replace.input","source_path":"upstream","target_path":"replace"}
        ],
        "options":{},
        "update_semantics":"latest_relation"
    });
    for has_alias in [false, true] {
        let mut declaration = base.clone();
        if has_alias {
            declaration["edges"][1]["input_alias"] = "input".into();
        }
        let graph = serde_json::from_value(declaration).unwrap();
        let report = crate::query_analysis::analyze_graph(graph)?;
        assert_eq!(report.check_updates(CompileMode::Stream).is_ok(), has_alias);
    }
    Ok(())
}

use calc_flow::arrow::datatypes::{DataType, Field, Schema};
use calc_flow::{
    CompileMode, CompileOptions, InputOptions, OperatorSpec, Program, Result, StreamRequirements,
    UdfRegistry, table_input,
};
use std::{collections::BTreeMap, sync::Arc};

fn sql_program() -> Result<Program> {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let a = table_input("a", schema.clone(), InputOptions::default())?;
    let z = table_input("z", schema.clone(), InputOptions::default())?;
    let relation = calc_flow::sql::sql(
        "SELECT x FROM z UNION ALL SELECT x FROM a",
        BTreeMap::from([("a".into(), a), ("z".into(), z)]),
    )?;
    Program::new(
        "union",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), relation)]),
    )
}

#[tokio::test]
async fn sql_union_uses_native_union_with_textual_arm_order() -> Result<()> {
    let program = sql_program()?;
    let udfs = UdfRegistry::new().snapshot();
    let project = program.to_project(&udfs).await?;
    let union = project
        .graph
        .nodes
        .iter()
        .find(|node| matches!(node.operator, OperatorSpec::Union))
        .expect("SQL must lower to native union");
    let first = project
        .graph
        .edges
        .iter()
        .find(|edge| edge.target_node == union.id && edge.target_port == "input_0")
        .unwrap();
    let arm = project
        .graph
        .nodes
        .iter()
        .find(|node| node.id == first.source_node)
        .unwrap();
    assert!(matches!(&arm.operator, OperatorSpec::Sql { aliases, .. } if aliases == &["z"]));
    program
        .compile_batch(&udfs, &CompileOptions::default())
        .await?;
    program
        .for_mode(CompileMode::Stream)?
        .compile_stream(
            &udfs,
            &StreamRequirements::default(),
            &CompileOptions::default(),
        )
        .await?;
    Ok(())
}

#[tokio::test]
async fn sql_union_arm_schema_errors_keep_the_arm_path() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let a = table_input("a", schema.clone(), InputOptions::default())?;
    let b = table_input("b", schema, InputOptions::default())?;
    let value = calc_flow::sql::sql(
        "SELECT x FROM a UNION ALL SELECT missing FROM b",
        BTreeMap::from([("a".into(), a), ("b".into(), b)]),
    )?;
    let batch = Program::new(
        "bad-arm",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), value)]),
    )?;
    let report = batch
        .analyze(
            &UdfRegistry::new().snapshot(),
            &calc_flow::AnalysisOptions::default(),
        )
        .await?;
    assert!(report.diagnostics.iter().any(|issue| issue.path == "outputs.out.sql.arms[1]" && issue.code == "schema_mismatch"), "{:?}", report.diagnostics);
    Ok(())
}

mod asof_support;
use calc_flow::{
    CompileMode, CompileOptions, InputOptions, Program, Result, UdfRegistry, table_input,
};
use std::collections::BTreeMap;

#[tokio::test]
async fn dataframe_and_sql_asof_share_native_ports_and_finite_results() -> Result<()> {
    let program = declaration()?;
    let udfs = UdfRegistry::new().snapshot();

    let plan = program
        .compile_batch(&udfs, &CompileOptions::default())
        .await?;
    let result = plan
        .execute(
            BTreeMap::from([
                ("a".into(), asof_support::batch(&[("A", 10, 1, 99)])),
                ("b".into(), asof_support::batch(&[("A", 8, 1, 7)])),
            ]),
            calc_flow::ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(result.outputs["df"].num_rows(), 1);
    assert_eq!(
        result.outputs["df"].table_payload()?.batches(),
        result.outputs["sql"].table_payload()?.batches()
    );
    let plan = program
        .for_mode(CompileMode::Stream)?
        .compile_stream(
            &udfs,
            &calc_flow::StreamRequirements::default(),
            &CompileOptions::default(),
        )
        .await?;
    stream::verify(plan, fixture()).await
}

fn declaration() -> Result<Program> {
    let options = InputOptions {
        entity_by: vec!["key".into()],
        event_time: Some("time".into()),
        sequence_by: vec!["sequence".into()],
    };
    let a = table_input("a", asof_support::schema(), options.clone())?;
    let b = table_input("b", asof_support::schema(), options)?;
    let spec = asof_support::spec(2);
    let df = a.asof_join(&b, spec.clone())?;
    let config = serde_json::json!({"spec": spec});
    let sql = calc_flow::sql::sql(
        &format!("SELECT * FROM CF_ASOF_JOIN('a', 'b', '{config}')"),
        BTreeMap::from([("a".into(), a), ("b".into(), b)]),
    )?;
    Program::new(
        "asof",
        CompileMode::Batch,
        BTreeMap::from([("df".into(), df), ("sql".into(), sql)]),
    )
}

fn fixture() -> BTreeMap<String, calc_flow::Batch> {
    BTreeMap::from([
        ("a".into(), asof_support::batch(&[("A", 10, 1, 99)])),
        ("b".into(), asof_support::batch(&[("A", 8, 1, 7)])),
    ])
}
fn values(batches: &[calc_flow::Batch]) -> Vec<Option<i64>> {
    batches
        .iter()
        .flat_map(|b| b.table_payload().unwrap().batches())
        .flat_map(|r| {
            r.column_by_name("right__value")
                .unwrap()
                .as_any()
                .downcast_ref::<calc_flow::arrow::array::Int64Array>()
                .unwrap()
                .iter()
        })
        .collect()
}
#[path = "finite_asof_frontend/rename.rs"]
mod rename;
#[path = "finite_asof_frontend/stream.rs"]
mod stream;

#[tokio::test]
async fn asof_project_roundtrip_uses_physical_bindings_and_shared_schema() -> Result<()> {
    let p = declaration()?;
    let udfs = UdfRegistry::new().snapshot();
    let old = p.compile_batch(&udfs, &CompileOptions::default()).await?;
    let project = p.to_project(&udfs).await?;
    let reloaded =
        calc_flow::compile_project(&project, &calc_flow::ProviderRegistry::default(), &udfs)?;
    let result = reloaded
        .execute(
            rebound(old.external_inputs(), reloaded.external_inputs()),
            calc_flow::ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(result.outputs.len(), 2);
    for batch in result.outputs.values() {
        assert_eq!(values(std::slice::from_ref(batch)), [Some(7)]);
    }
    verify_schema(&result)?;
    roundtrip_stream(p).await
}
fn rebound(
    logical: &BTreeMap<String, calc_flow::PortEndpoint>,
    physical: &BTreeMap<String, calc_flow::PortEndpoint>,
) -> BTreeMap<String, calc_flow::Batch> {
    logical
        .iter()
        .map(|(name, endpoint)| {
            let binding = physical
                .iter()
                .find(|(_, p)| p.node_id == endpoint.node_id && p.port == endpoint.port)
                .unwrap()
                .0;
            (binding.clone(), fixture()[name].clone())
        })
        .collect()
}
async fn roundtrip_stream(p: Program) -> Result<()> {
    let p = p.for_mode(CompileMode::Stream)?;
    let udfs = UdfRegistry::new().snapshot();
    let old = p
        .compile_stream(
            &udfs,
            &calc_flow::StreamRequirements::default(),
            &CompileOptions::default(),
        )
        .await?;
    let project = p.to_project(&udfs).await?;
    let plan = calc_flow::compile_stream_project_graph(
        &project,
        &calc_flow::ProviderRegistry::default(),
        &udfs,
        &calc_flow::StreamRequirements::default(),
    )?;
    let inputs = rebound(old.external_inputs(), plan.external_inputs());
    stream::verify(plan, inputs).await
}

fn verify_schema(result: &calc_flow::RunResult) -> Result<()> {
    let expected = calc_flow::OperatorMetadata::output_ports(&asof_support::operator(2))[0]
        .schema()
        .unwrap()
        .clone();
    for batch in result.outputs.values() {
        assert_eq!(batch.table_payload()?.schema(), &expected);
    }
    Ok(())
}

fn source(name: &str) -> Result<calc_flow::TableExpr> {
    table_input(
        name,
        asof_support::schema(),
        InputOptions {
            entity_by: vec!["key".into()],
            event_time: Some("time".into()),
            sequence_by: vec!["sequence".into()],
        },
    )
}
fn nested_declaration() -> Result<Program> {
    let a = source("a")?.sql("SELECT * FROM input WHERE key = 'A'")?;
    let first = a.asof_join(&source("b")?, asof_support::spec(2))?;
    let post = first.sql("SELECT left__key AS key, left__time AS time, left__sequence AS sequence, left__value AS value FROM input")?;
    let second = post.asof_join(&source("c")?, asof_support::spec(2))?;
    Program::new(
        "nested",
        CompileMode::Batch,
        BTreeMap::from([("first".into(), first), ("second".into(), second)]),
    )
}
#[tokio::test]
async fn rust_nested_asof_preserves_renamed_identity_and_one_shared_owner() -> Result<()> {
    let p = nested_declaration()?;
    let udfs = UdfRegistry::new().snapshot();
    let project = p.to_project(&udfs).await?;
    assert_eq!(
        project
            .graph
            .nodes
            .iter()
            .filter(|n| matches!(n.operator, calc_flow::OperatorSpec::StreamAsofJoin { .. }))
            .count(),
        2
    );
    let plan = p.compile_batch(&udfs, &CompileOptions::default()).await?;
    let mut inputs = fixture();
    inputs.insert("c".into(), asof_support::batch(&[("A", 8, 3, 3)]));
    let result = plan
        .execute(inputs, calc_flow::ExecutionOptions::default())
        .await?;
    assert_eq!(
        values(std::slice::from_ref(&result.outputs["first"])),
        [Some(7)]
    );
    assert_eq!(
        values(std::slice::from_ref(&result.outputs["second"])),
        [Some(3)]
    );
    Ok(())
}

fn invalid_nullable_program() -> Result<Program> {
    let fields = asof_support::schema()
        .fields()
        .iter()
        .map(|f| {
            if f.name() == "key" {
                f.as_ref().clone().with_nullable(true)
            } else {
                f.as_ref().clone()
            }
        })
        .collect::<Vec<_>>();
    let left = table_input(
        "a",
        std::sync::Arc::new(calc_flow::arrow::datatypes::Schema::new(fields)),
        InputOptions {
            entity_by: vec!["key".into()],
            event_time: Some("time".into()),
            sequence_by: vec!["sequence".into()],
        },
    )?;
    let joined = left.asof_join(&source("b")?, asof_support::spec(2))?;
    Program::new(
        "invalid",
        CompileMode::Batch,
        BTreeMap::from([("match".into(), joined)]),
    )
}
#[tokio::test]
async fn rust_asof_schema_errors_keep_public_spec_path_and_reason() -> Result<()> {
    let report = invalid_nullable_program()?
        .analyze(
            &UdfRegistry::new().snapshot(),
            &calc_flow::AnalysisOptions::default(),
        )
        .await?;
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.path == "outputs.match.asof_join.spec.left.keys[0]"
                && d.code == "nullable_identity_field"),
        "{:?}",
        report.diagnostics
    );
    Ok(())
}

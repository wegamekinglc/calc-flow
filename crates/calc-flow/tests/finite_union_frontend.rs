use calc_flow::arrow::datatypes::{DataType, Field, Schema};
use calc_flow::{
    CompileMode, CompileOptions, InputOptions, OperatorSpec, Program, Result, StreamRequirements,
    UdfRegistry, table_input,
};
use std::{collections::BTreeMap, sync::Arc};

fn union_program() -> Result<Program> {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let a = table_input("a", schema.clone(), InputOptions::default())?;
    let b = table_input("b", schema, InputOptions::default())?;
    Program::new(
        "union",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), a.union_all(&b)?)]),
    )
}

#[tokio::test]
async fn dataframe_union_exports_and_compiles_both_lifecycles() -> Result<()> {
    let program = union_program()?;
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let udfs = UdfRegistry::new().snapshot();
    assert_eq!(program.collect_schema(&udfs).await?["out"], schema);
    let project = program.to_project(&udfs).await?;
    assert!(
        project
            .graph
            .nodes
            .iter()
            .any(|node| matches!(node.operator, OperatorSpec::Union))
    );
    program
        .compile_batch(&udfs, &CompileOptions::default())
        .await?;
    let stream = program.for_mode(CompileMode::Stream)?;
    assert_eq!(stream.fingerprint(), program.fingerprint());
    stream
        .compile_stream(
            &udfs,
            &StreamRequirements::default(),
            &CompileOptions::default(),
        )
        .await?;
    Ok(())
}

#[tokio::test]
async fn append_union_proves_unordered_eof_history() -> Result<()> {
    use calc_flow::{
        AnalysisOptions, Comparison, ComparisonCutoff, HistoryScope, PortabilityRequest,
        SupportStatus,
    };
    let program = union_program()?;
    let request = PortabilityRequest::new(
        Comparison::FinalRelation,
        HistoryScope::AcceptedHistory,
        ComparisonCutoff::AllSourcesEof,
    )?;
    let report = program
        .analyze(
            &UdfRegistry::new().snapshot(),
            &AnalysisOptions {
                portability: Some(request),
            },
        )
        .await?;
    assert_eq!((report.schema_version, report.analysis_revision), (2, 7));
    assert_eq!(
        report.portability,
        Some(SupportStatus::Supported),
        "{:?}",
        report.diagnostics
    );
    let latest = PortabilityRequest::new(
        Comparison::FinalRelation,
        HistoryScope::LatestRelation,
        ComparisonCutoff::AllSourcesEof,
    )?;
    let report = program
        .analyze(
            &UdfRegistry::new().snapshot(),
            &AnalysisOptions {
                portability: Some(latest),
            },
        )
        .await?;
    assert!(
        report
            .diagnostics
            .iter()
            .any(|issue| issue.code == "history_scope_mismatch")
    );
    Ok(())
}

#[tokio::test]
async fn union_normalizes_declaration_metadata_without_mutating_it() -> Result<()> {
    use std::collections::HashMap;
    let original = Arc::new(
        Schema::new(vec![
            Field::new("x", DataType::Int64, true)
                .with_metadata(HashMap::from([("user".into(), "left".into())])),
        ])
        .with_metadata(HashMap::from([("schema".into(), "left".into())])),
    );
    let canonical = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let a = table_input("a", original.clone(), InputOptions::default())?;
    let b = table_input("b", canonical.clone(), InputOptions::default())?;
    let value = a.union_all(&b)?;
    let program = Program::new(
        "metadata",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), value)]),
    )?;
    let udfs = UdfRegistry::new().snapshot();
    assert_eq!(program.collect_schema(&udfs).await?["out"], canonical);
    let plan = program
        .compile_batch(&udfs, &CompileOptions::default())
        .await?;
    assert_metadata_execution(&plan, original.clone(), canonical).await?;
    assert_eq!(original.metadata()["schema"], "left");
    assert_eq!(original.field(0).metadata()["user"], "left");
    Ok(())
}

fn metadata_data(schema: Arc<Schema>, value: i64) -> calc_flow::Batch {
    use calc_flow::arrow::{array::Int64Array, record_batch::RecordBatch};
    calc_flow::Batch::table(
        vec![
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![Some(value)]))])
                .unwrap(),
        ],
        calc_flow::BatchMetadata::default(),
    )
    .unwrap()
}

async fn assert_metadata_execution(
    plan: &calc_flow::BatchExecutionPlan,
    raw: Arc<Schema>,
    canonical: Arc<Schema>,
) -> Result<()> {
    let expected = canonical.clone();
    let out = plan
        .execute(
            BTreeMap::from([
                ("a".into(), metadata_data(raw, 1)),
                ("b".into(), metadata_data(canonical, 2)),
            ]),
            calc_flow::ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(out.outputs["out"].num_rows(), 2);
    assert_eq!(out.outputs["out"].table_payload()?.schema(), &expected);
    Ok(())
}

#[tokio::test]
async fn union_does_not_invent_a_cross_ingress_batch_boundary() -> Result<()> {
    use calc_flow::{
        AnalysisOptions, Comparison, ComparisonCutoff, HistoryScope, PortabilityRequest,
        SupportStatus,
    };
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let a = table_input("a", schema.clone(), InputOptions::default())?;
    let b = table_input("b", schema, InputOptions::default())?;
    let program = Program::new(
        "boundaries",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), a.union_all(&b)?)]),
    )?;
    let report = program
        .analyze(
            &UdfRegistry::new().snapshot(),
            &AnalysisOptions {
                portability: Some(PortabilityRequest::new(
                    Comparison::PerBatch,
                    HistoryScope::CurrentBatch,
                    ComparisonCutoff::BatchBoundary,
                )?),
            },
        )
        .await?;
    assert_eq!(report.portability, Some(SupportStatus::Unknown));
    Ok(())
}

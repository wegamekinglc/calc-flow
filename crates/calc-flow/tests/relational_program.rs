use calc_flow::arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use calc_flow::{
    AnalysisOptions, Batch, BatchMetadata, Comparison, ComparisonCutoff, CompileMode,
    CompileOptions, ExecutionOptions, HistoryScope, InputOptions, PortabilityRequest, Program,
    SupportStatus, UdfRegistry, lit, table_input,
};
use std::{collections::BTreeMap, sync::Arc};

fn data(values: Vec<Option<i64>>) -> calc_flow::Result<Batch> {
    let array = Arc::new(Int64Array::from(values));
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    Batch::table(
        vec![RecordBatch::try_new(schema, vec![array]).unwrap()],
        BatchMetadata::default(),
    )
}
fn values(batch: &Batch, name: &str) -> Vec<Option<i64>> {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| {
            let array = record
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            array.iter().collect::<Vec<_>>()
        })
        .collect()
}

fn mixed_program() -> calc_flow::Result<Program> {
    let source = table_input(
        "events",
        data(vec![])?.table_payload()?.schema().clone(),
        InputOptions::default(),
    )?;
    let scaled = source.with_columns(vec![(source.column("x")? + lit(1_i64)?).alias("y")?])?;
    let first = calc_flow::sql::sql(
        "SELECT x, y FROM s WHERE y > 1",
        BTreeMap::from([("s".into(), scaled.clone())]),
    )?;
    let selected = first.select(vec![(first.column("y")? * lit(2_i64)?).alias("z")?])?;
    let result = selected.sql("SELECT z FROM input")?;
    Program::new(
        "mixed",
        CompileMode::Batch,
        BTreeMap::from([("scaled".into(), scaled), ("result".into(), result)]),
    )
}

#[tokio::test]
async fn named_mixed_dag_reuses_declarations_and_isolates_executions() -> calc_flow::Result<()> {
    let program = mixed_program()?;
    let stream = program.for_mode(CompileMode::Stream)?;
    assert_eq!(stream.fingerprint(), program.fingerprint());
    let udfs = UdfRegistry::new().snapshot();
    let schemas = program.collect_schema(&udfs).await?;
    assert_eq!(
        schemas["result"].field(0),
        &Field::new("z", DataType::Int64, true)
    );
    let request = PortabilityRequest::new(
        Comparison::PerBatch,
        HistoryScope::CurrentBatch,
        ComparisonCutoff::BatchBoundary,
    )?;
    let report = program
        .analyze(
            &udfs,
            &AnalysisOptions {
                portability: Some(request.clone()),
            },
        )
        .await?;
    assert_eq!(report.portability, Some(SupportStatus::Supported));
    assert!(
        report
            .nodes
            .iter()
            .all(|node| !node.path.contains("graph.nodes"))
    );
    let options = CompileOptions {
        portability: Some(request),
    };
    assert!(
        program
            .explain(&udfs, &AnalysisOptions::default())
            .await?
            .contains(program.name())
    );
    let stream_plan = stream
        .compile_stream(&udfs, &calc_flow::StreamRequirements::default(), &options)
        .await?;
    assert_ne!(
        program.compile_batch(&udfs, &options).await?.fingerprint(),
        stream_plan.fingerprint()
    );
    verify_execution(&program, &udfs, &options).await?;
    Ok(())
}

async fn verify_execution(
    program: &Program,
    udfs: &calc_flow::UdfRegistrySnapshot,
    options: &CompileOptions,
) -> calc_flow::Result<()> {
    let plan = program.compile_batch(udfs, options).await?;
    assert_eq!(
        plan.external_inputs().keys().collect::<Vec<_>>(),
        vec!["events"]
    );
    assert_eq!(
        plan.external_outputs().keys().collect::<Vec<_>>(),
        vec!["result", "scaled"]
    );
    for (input, expected) in [
        (
            vec![Some(1), None, Some(3), Some(1)],
            vec![Some(4), Some(8), Some(4)],
        ),
        (vec![Some(5)], vec![Some(12)]),
        (vec![], vec![]),
    ] {
        let result = plan
            .execute(
                BTreeMap::from([("events".into(), data(input)?)]),
                ExecutionOptions::default(),
            )
            .await?;
        assert_eq!(values(&result.outputs["result"], "z"), expected);
    }
    verify_export(program, udfs).await?;
    Ok(())
}

async fn verify_export(
    program: &Program,
    udfs: &calc_flow::UdfRegistrySnapshot,
) -> calc_flow::Result<()> {
    let project = program.to_project(udfs).await?;
    let serialized = calc_flow::export_project_json(&project)?;
    let reloaded = calc_flow::import_project_json(serialized.as_bytes())?;
    assert_eq!(project, reloaded);
    let reloaded =
        calc_flow::compile_project(&reloaded, &calc_flow::ProviderRegistry::default(), udfs)?;
    let inputs = reloaded
        .external_inputs()
        .keys()
        .map(|name| Ok((name.clone(), data(vec![Some(1)])?)))
        .collect::<calc_flow::Result<BTreeMap<_, _>>>()?;
    let outputs = reloaded
        .execute(inputs, ExecutionOptions::default())
        .await?
        .outputs;
    assert!(outputs.values().any(|batch| {
        batch
            .table_payload()
            .unwrap()
            .schema()
            .field_with_name("z")
            .is_ok()
            && values(batch, "z") == vec![Some(4)]
    }));
    Ok(())
}

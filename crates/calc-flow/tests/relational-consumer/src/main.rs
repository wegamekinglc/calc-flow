use async_trait::async_trait;
use calc_flow::arrow::{
    array::{Float64Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use calc_flow::{
    AnalysisOptions, Batch, BatchMetadata, Comparison, ComparisonCutoff, CompileMode,
    CompileOptions, Cursor, DeliveryGuarantee, ExecutionOptions, HistoryScope, InputOptions,
    JobState, JsonMap, NativeWatermarkCapability, PortabilityRequest, Program, ReplayPositioning,
    Result, SinkBinding, SourceBinding, SourceCapabilities, SourceDeliveryCapability, SourceEvent,
    SourceSchema, StreamRequirements, StreamSink, StreamSource, StreamingRunner, SupportStatus,
    TableExpr, UdfRegistry, UpdateSemantics, lit, table_input,
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

fn data(values: Vec<Option<i64>>) -> Result<Batch> {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    Batch::table(
        vec![RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values))]).unwrap()],
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
            record
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>()
        })
        .collect()
}
fn source_input() -> Result<TableExpr> {
    table_input(
        "events",
        data(vec![])?.table_payload()?.schema().clone(),
        InputOptions::default(),
    )
}
fn scaled(source: &TableExpr) -> Result<TableExpr> {
    source.with_columns(vec![(source.column("x")? + lit(1_i64)?).alias("y")?])
}
fn dataframe(shared: &TableExpr) -> Result<TableExpr> {
    let selected = shared.filter(shared.column("y")?.gt(lit(1_i64)?))?;
    selected.select(vec![(shared.column("y")? * lit(2_i64)?).alias("z")?])
}
fn df_sql_df_sql(shared: &TableExpr) -> Result<TableExpr> {
    let first = calc_flow::sql::sql(
        "SELECT y FROM s WHERE y > 1",
        BTreeMap::from([("s".into(), shared.clone())]),
    )?;
    let selected = first.select(vec![(first.column("y")? * lit(2_i64)?).alias("z")?])?;
    selected.sql("SELECT z FROM input")
}
fn sql_df_sql(source: &TableExpr) -> Result<(TableExpr, TableExpr)> {
    let first = source.sql("SELECT x FROM input")?;
    let selected = first.select(vec![(first.column("x")? + lit(1_i64)?).alias("z")?])?;
    Ok((first, selected.sql("SELECT z FROM input")?))
}
fn declaration(shape: &str) -> Result<(TableExpr, TableExpr, Vec<Option<i64>>)> {
    let source = source_input()?;
    match shape {
        "sql" => Ok((
            source.clone(),
            source.sql("SELECT x + 1 AS z FROM input")?,
            vec![Some(2), None, Some(4), Some(2)],
        )),
        "sql_df_sql" => sql_declaration(&source),
        _ => filtered_declaration(&source, shape),
    }
}

fn sql_declaration(source: &TableExpr) -> Result<(TableExpr, TableExpr, Vec<Option<i64>>)> {
    let (shared, result) = sql_df_sql(source)?;
    Ok((shared, result, vec![Some(2), None, Some(4), Some(2)]))
}
fn filtered_declaration(
    source: &TableExpr,
    shape: &str,
) -> Result<(TableExpr, TableExpr, Vec<Option<i64>>)> {
    let shared = scaled(source)?;
    let result = if shape == "dataframe" {
        dataframe(&shared)?
    } else {
        df_sql_df_sql(&shared)?
    };
    Ok((shared, result, vec![Some(4), Some(8), Some(4)]))
}

struct Source {
    batches: std::vec::IntoIter<Batch>,
    schema: calc_flow::arrow::datatypes::SchemaRef,
    opened: Arc<AtomicUsize>,
    closed: Arc<AtomicUsize>,
    sequence: u64,
}
#[async_trait]
impl StreamSource for Source {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::Unsupported,
            delivery: SourceDeliveryCapability::Lossy,
            max_batch_rows: 100,
            max_batch_bytes: 1 << 20,
            schema: SourceSchema::Exact(self.schema.clone()),
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        assert!(cursor.is_none());
        self.opened.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let Some(batch) = self.batches.next() else {
            return Ok(None);
        };
        self.sequence += 1;
        let batch =
            batch.with_metadata(BatchMetadata::new("events", self.sequence, JsonMap::new())?);
        Ok(Some(SourceEvent::Data {
            batch,
            cursor: Cursor::new(
                "events",
                self.sequence.to_be_bytes().to_vec(),
                JsonMap::new(),
            )?,
        }))
    }
    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
struct Sink(Arc<Mutex<Vec<Batch>>>);
#[async_trait]
impl StreamSink for Sink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.0.lock().unwrap().push(batch.clone());
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

async fn run_stream(
    program: &Program,
    batches: Vec<Batch>,
    options: &CompileOptions,
) -> Result<BTreeMap<String, Vec<Batch>>> {
    let udfs = UdfRegistry::new().snapshot();
    let requirements = StreamRequirements {
        delivery: program
            .outputs()
            .keys()
            .map(|name| (name.clone(), DeliveryGuarantee::BestEffort))
            .collect(),
    };
    let plan = program
        .compile_stream(&udfs, &requirements, options)
        .await?;
    let opened = Arc::new(AtomicUsize::new(0));
    let closed = Arc::new(AtomicUsize::new(0));
    let source = Source {
        schema: batches[0].table_payload()?.schema().clone(),
        batches: batches.into_iter(),
        opened: opened.clone(),
        closed: closed.clone(),
        sequence: 0,
    };
    let mut captured = BTreeMap::new();
    let mut sinks = BTreeMap::new();
    for name in program.outputs().keys() {
        let captured_batches = Arc::new(Mutex::new(vec![]));
        sinks.insert(
            name.clone(),
            vec![SinkBinding::ordinary(name, Sink(captured_batches.clone()))?],
        );
        captured.insert(name.clone(), captured_batches);
    }
    let outcome = StreamingRunner::without_checkpoints(
        plan,
        BTreeMap::from([(
            "events".into(),
            SourceBinding::new(source)
                .with_watermark_policy(calc_flow::WatermarkPolicy::Disabled { idle_timeout: None }),
        )]),
        sinks,
    )?
    .with_runtime_config(calc_flow::StreamRuntimeConfig {
        checkpointing: false,
        sql_replacement_state_budget: Some(calc_flow::StateBudget::new(100, 8 << 20)?),
        ..Default::default()
    })?
    .start()
    .await?
    .wait()
    .await;
    assert_eq!(outcome.state, JobState::Completed, "{outcome:?}");
    assert!(outcome.errors.is_empty());
    assert_eq!(opened.load(Ordering::SeqCst), 1);
    assert_eq!(closed.load(Ordering::SeqCst), 1);
    Ok(captured
        .into_iter()
        .map(|(name, batches)| (name, batches.lock().unwrap().clone()))
        .collect())
}

async fn verify_frontend(
    batch: &Program,
    options: &CompileOptions,
) -> Result<calc_flow::BatchExecutionPlan> {
    let udfs = UdfRegistry::new().snapshot();
    assert_eq!(
        batch.collect_schema(&udfs).await?["result"].field(0),
        &Field::new("z", DataType::Int64, true)
    );
    assert_eq!(
        batch
            .analyze(
                &udfs,
                &AnalysisOptions {
                    portability: options.portability.clone()
                }
            )
            .await?
            .portability,
        Some(SupportStatus::Supported)
    );
    assert!(
        batch
            .explain(&udfs, &AnalysisOptions::default())
            .await?
            .contains(batch.fingerprint())
    );
    batch.compile_batch(&udfs, options).await
}
async fn check_case(
    plan: &calc_flow::BatchExecutionPlan,
    stream: &Program,
    options: &CompileOptions,
    input: Batch,
    mut expected: Vec<Option<i64>>,
) -> Result<()> {
    expected.sort();
    let result = plan
        .execute(
            BTreeMap::from([("events".into(), input.clone())]),
            ExecutionOptions::default(),
        )
        .await?;
    let mut actual = values(&result.outputs["result"], "z");
    actual.sort();
    assert_eq!(actual, expected);
    let streamed = run_stream(stream, vec![input], options).await?;
    let mut actual: Vec<_> = streamed["result"]
        .iter()
        .flat_map(|b| values(b, "z"))
        .collect();
    actual.sort();
    assert_eq!(actual, expected);
    Ok(())
}
async fn check_export(batch: &Program, input: Batch, mut expected: Vec<Option<i64>>) -> Result<()> {
    let udfs = UdfRegistry::new().snapshot();
    let project = batch.to_project(&udfs).await?;
    let graph = calc_flow::PipelineSpec {
        name: project.graph.name.clone(),
        nodes: project.graph.nodes.clone(),
        edges: project.graph.edges.clone(),
        datafusion: project.graph.datafusion,
        update_semantics: project.graph.update_semantics,
    };
    let project = calc_flow::ProjectSpec { graph, ..project };
    assert!(
        project
            .data_sources
            .iter()
            .all(|source| source.data.as_array().is_some_and(Vec::is_empty))
    );
    let json = calc_flow::export_project_json(&project)?;
    let reloaded = calc_flow::import_project_json(json.as_bytes())?;
    let reloaded =
        calc_flow::compile_project(&reloaded, &calc_flow::ProviderRegistry::default(), &udfs)?;
    let bindings = reloaded
        .external_inputs()
        .keys()
        .map(|name| (name.clone(), input.clone()))
        .collect();
    let outputs = reloaded
        .execute(bindings, ExecutionOptions::default())
        .await?
        .outputs;
    expected.sort();
    let result = outputs
        .values()
        .find(|b| {
            b.table_payload()
                .unwrap()
                .schema()
                .field_with_name("z")
                .is_ok()
        })
        .unwrap();
    let mut actual = values(result, "z");
    actual.sort();
    assert_eq!(actual, expected);
    Ok(())
}
async fn check_shape(shape: &str) -> Result<()> {
    let (shared, result, expected) = declaration(shape)?;
    let batch = Program::new(
        shape,
        CompileMode::Batch,
        BTreeMap::from([("shared".into(), shared), ("result".into(), result)]),
    )?
    .with_update_semantics(UpdateSemantics::LatestRelation);
    let stream = batch.for_mode(CompileMode::Stream)?;
    assert_eq!(batch.fingerprint(), stream.fingerprint());
    assert_eq!(batch.inputs().len(), 1);
    let request = PortabilityRequest::new(
        Comparison::PerBatch,
        HistoryScope::CurrentBatch,
        ComparisonCutoff::BatchBoundary,
    )?;
    let options = CompileOptions {
        portability: Some(request),
    };
    let plan = verify_frontend(&batch, &options).await?;
    let stream_plan = stream
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
            &options,
        )
        .await?;
    assert_ne!(plan.fingerprint(), stream_plan.fingerprint());
    drop(stream_plan);
    check_shape_execution(shape, &batch, &stream, &plan, &options, expected).await
}
async fn check_shape_execution(
    shape: &str,
    batch: &Program,
    stream: &Program,
    plan: &calc_flow::BatchExecutionPlan,
    options: &CompileOptions,
    expected: Vec<Option<i64>>,
) -> Result<()> {
    let input = data(vec![Some(1), None, Some(3), Some(1)])?;
    check_case(plan, stream, options, input.clone(), expected.clone()).await?;
    for empty in [vec![], vec![None, None]] {
        let expected = if matches!(shape, "dataframe" | "df_sql_df_sql") {
            vec![]
        } else {
            empty.clone()
        };
        check_case(plan, stream, options, data(empty)?, expected).await?;
    }
    check_export(batch, input, expected).await?;
    println!("{shape}: schema, analysis, batch, owned stream, export/reload passed");
    Ok(())
}

async fn check_batch_state(batch: &Program) -> Result<()> {
    let plan = batch
        .compile_batch(&UdfRegistry::new().snapshot(), &CompileOptions::default())
        .await?;
    for (input, expected) in [(vec![Some(1)], 1), (vec![Some(2), Some(3)], 5)] {
        let result = plan
            .execute(
                BTreeMap::from([("events".into(), data(input)?)]),
                ExecutionOptions::default(),
            )
            .await?;
        assert_eq!(
            values(&result.outputs["result"], "total"),
            vec![Some(expected)]
        );
    }
    Ok(())
}
async fn check_stream_state(stream: &Program) -> Result<()> {
    let first = run_stream(
        stream,
        vec![data(vec![Some(1)])?, data(vec![Some(2), Some(3)])?],
        &CompileOptions::default(),
    )
    .await?;
    assert_eq!(
        first["result"]
            .iter()
            .flat_map(|b| values(b, "total"))
            .collect::<Vec<_>>(),
        vec![Some(1), Some(6)]
    );
    let fresh = run_stream(
        stream,
        vec![data(vec![Some(5)])?],
        &CompileOptions::default(),
    )
    .await?;
    assert_eq!(
        fresh["result"]
            .iter()
            .flat_map(|b| values(b, "total"))
            .collect::<Vec<_>>(),
        vec![Some(5)]
    );
    println!("state: same batch plan 1/5; same stream job 1/6; new stream job 5");
    Ok(())
}
#[tokio::main]
async fn main() -> Result<()> {
    check_aggregate().await?;
    check_aggregate_groups().await?;
    check_unsigned_negation().await?;
    check_metadata_stream().await?;
    for shape in ["dataframe", "sql", "df_sql_df_sql", "sql_df_sql"] {
        check_shape(shape).await?;
    }
    check_state().await?;
    Ok(())
}

async fn check_unsigned_negation() -> Result<()> {
    let source = table_input(
        "events",
        Arc::new(Schema::new(vec![Field::new("u", DataType::UInt64, true)])),
        InputOptions::default(),
    )?;
    let result = source.select(vec![(-source.column("u")?).alias("negative.value")?])?;
    let error = result
        .collect_schema(&UdfRegistry::new().snapshot(), CompileMode::Batch)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("outputs.result[\"negative.value\"]")
            && error.contains("unsupported_type")
            && error.contains("signed"),
        "{error}"
    );
    Ok(())
}

async fn check_state() -> Result<()> {
    let source = source_input()?;
    let total = source.sql("SELECT SUM(x) AS total FROM input")?;
    let batch = Program::new(
        "state",
        CompileMode::Batch,
        BTreeMap::from([("result".into(), total)]),
    )?;
    check_batch_state(&batch).await?;
    check_stream_state(&batch.for_mode(CompileMode::Stream)?).await?;
    Ok(())
}

fn metadata_batch() -> Result<Batch> {
    use std::collections::HashMap;
    let field = Field::new("x", DataType::Int64, true)
        .with_metadata(HashMap::from([("unit".into(), "USD".into())]));
    let schema = Arc::new(Schema::new_with_metadata(
        vec![field],
        HashMap::from([("origin".into(), "consumer".into())]),
    ));
    let record =
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![Some(1)]))]).unwrap();
    Batch::table(vec![record], BatchMetadata::default())
}
fn metadata_program(batch: &Batch) -> Result<Program> {
    let source = table_input(
        "events",
        batch.table_payload()?.schema().clone(),
        InputOptions::default(),
    )?;
    Program::new(
        "metadata",
        CompileMode::Stream,
        BTreeMap::from([("result".into(), source)]),
    )
}
async fn check_metadata_stream() -> Result<()> {
    let batch = metadata_batch()?;
    let schema = batch.table_payload()?.schema().clone();
    let program = metadata_program(&batch)?;
    let actual = run_stream(&program, vec![batch.clone()], &CompileOptions::default()).await?;
    assert_eq!(values(&actual["result"][0], "x"), vec![Some(1)]);
    assert_eq!(batch.table_payload()?.schema(), &schema);
    println!("metadata: owned stream accepts schema and field metadata without mutating input");
    Ok(())
}

fn aggregate_declaration() -> Result<Program> {
    let source = source_input()?;
    let aggregate = source.aggregate(aggregate_reducers(&source)?, vec![])?;
    Program::new(
        "public-aggregate",
        CompileMode::Batch,
        BTreeMap::from([("result".into(), aggregate)]),
    )
}

fn aggregate_reducers(source: &TableExpr) -> Result<Vec<calc_flow::NamedAggregate>> {
    let column = source.column("x")?;
    Ok(vec![
        column.sum().alias("total")?,
        column.count().alias("non_null")?,
        calc_flow::count_rows().alias("rows")?,
        column.min().alias("minimum")?,
        column.max().alias("maximum")?,
        column.mean().alias("mean")?,
    ])
}

async fn check_aggregate() -> Result<()> {
    let program = aggregate_declaration()?;
    check_aggregate_schema(&program).await?;
    check_aggregate_finite(&program).await?;
    check_aggregate_replacement(&program).await?;
    println!("aggregate: root exports, exact schema, empty/NULL oracles and latest 1/6 passed");
    Ok(())
}

async fn check_aggregate_schema(program: &Program) -> Result<()> {
    let udfs = UdfRegistry::new().snapshot();
    let schema = &program.collect_schema(&udfs).await?["result"];
    assert_eq!(schema.field(0), &Field::new("total", DataType::Int64, true));
    assert_eq!(
        schema.field(1),
        &Field::new("non_null", DataType::Int64, false)
    );
    assert_eq!(
        schema.field(3),
        &Field::new("minimum", DataType::Int64, true)
    );
    assert_eq!(
        schema.field(4),
        &Field::new("maximum", DataType::Int64, true)
    );
    assert_eq!(
        schema.field(5),
        &Field::new("mean", DataType::Float64, true)
    );
    Ok(())
}

async fn check_aggregate_finite(program: &Program) -> Result<()> {
    let udfs = UdfRegistry::new().snapshot();
    let plan = program
        .compile_batch(&udfs, &CompileOptions::default())
        .await?;
    for (input, sum, non_null, rows) in [
        (vec![], None, 0, 0),
        (vec![None, None], None, 0, 2),
        (vec![Some(1), None, Some(3)], Some(4), 2, 3),
    ] {
        let out = plan
            .execute(
                BTreeMap::from([("events".into(), data(input)?)]),
                ExecutionOptions::default(),
            )
            .await?;
        check_aggregate_output(&out.outputs["result"], sum, non_null, rows)?;
    }
    Ok(())
}

fn check_aggregate_output(
    output: &Batch,
    sum: Option<i64>,
    non_null: i64,
    rows: i64,
) -> Result<()> {
    assert_eq!(values(output, "total"), vec![sum]);
    assert_eq!(values(output, "non_null"), vec![Some(non_null)]);
    assert_eq!(values(output, "rows"), vec![Some(rows)]);
    assert_eq!(values(output, "minimum"), vec![sum.map(|_| 1)]);
    assert_eq!(values(output, "maximum"), vec![sum.map(|_| 3)]);
    let means = output
        .table_payload()?
        .batches()
        .iter()
        .flat_map(|record| {
            record
                .column_by_name("mean")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .iter()
        })
        .collect::<Vec<_>>();
    assert_eq!(means, vec![sum.map(|_| 2.0)]);
    Ok(())
}

async fn check_aggregate_replacement(program: &Program) -> Result<()> {
    let latest = Program::new(
        "external-replacement",
        CompileMode::Stream,
        BTreeMap::from([(
            "result".into(),
            program.outputs()["result"].sql("SELECT SUM(total) AS total FROM input")?,
        )]),
    )?
    .with_update_semantics(UpdateSemantics::LatestRelation);
    let actual = run_stream(
        &latest,
        vec![data(vec![Some(1)])?, data(vec![Some(2), Some(3)])?],
        &CompileOptions::default(),
    )
    .await?;
    assert_eq!(
        actual["result"]
            .iter()
            .flat_map(|b| values(b, "total"))
            .collect::<Vec<_>>(),
        vec![Some(1), Some(6)]
    );
    Ok(())
}

async fn check_aggregate_groups() -> Result<()> {
    let program = aggregate_groups_declaration()?;
    let plan = program
        .compile_batch(&UdfRegistry::new().snapshot(), &CompileOptions::default())
        .await?;
    let out = plan
        .execute(
            BTreeMap::from([("events".into(), data(vec![Some(1), None, Some(1)])?)]),
            ExecutionOptions::default(),
        )
        .await?;
    let mut actual = values(&out.outputs["result"], "x")
        .into_iter()
        .zip(values(&out.outputs["result"], "rows"))
        .collect::<Vec<_>>();
    actual.sort();
    assert_eq!(actual, vec![(None, Some(1)), (Some(1), Some(2))]);
    let empty = plan
        .execute(
            BTreeMap::from([("events".into(), data(vec![])?)]),
            ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(empty.outputs["result"].num_rows(), 0);
    Ok(())
}

fn aggregate_groups_declaration() -> Result<Program> {
    let source = source_input()?;
    let grouped = source.aggregate(
        vec![calc_flow::count_rows().alias("rows")?],
        vec!["x".into()],
    )?;
    Program::new(
        "groups",
        CompileMode::Batch,
        BTreeMap::from([("result".into(), grouped)]),
    )
}

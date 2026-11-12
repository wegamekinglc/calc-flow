use async_trait::async_trait;
use calc_flow::arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use calc_flow::{
    AnalysisOptions, Batch, BatchMetadata, CalcFlowError, CompileMode, CompileOptions, Cursor,
    ExecutionOptions, InputInterpretation, InputOptions, JobState, JsonMap,
    ManagedCheckpointRuntime, NativeWatermarkCapability, Program, ReplayPositioning, Result,
    SinkBinding, SourceBinding, SourceCapabilities, SourceDeliveryCapability, SourceEvent,
    SourceSchema, StreamExecutionPlan, StreamRequirements, StreamSink, StreamSource,
    StreamingErrorCategory, StreamingRunner, SupportStatus, UdfRegistry, UpdateKind,
    UpdateSemantics, lit, table_input,
};
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]))
}
fn data(parts: &[&[i64]]) -> Batch {
    Batch::table(
        parts
            .iter()
            .map(|values| {
                RecordBatch::try_new(schema(), vec![Arc::new(Int64Array::from(values.to_vec()))])
                    .unwrap()
            })
            .collect(),
        BatchMetadata::default(),
    )
    .unwrap()
}
fn program(chain: bool, transform: bool) -> Result<Program> {
    let source = table_input("events", schema(), InputOptions::default())?;
    let total = source.sql("SELECT SUM(x) AS total FROM input")?;
    let total = if transform {
        transformed_total(&total)?
    } else {
        total
    };
    let result = if chain {
        total.sql("SELECT SUM(total) AS total FROM input")?
    } else {
        total
    };
    Program::new(
        "latest",
        CompileMode::Stream,
        BTreeMap::from([("result".into(), result)]),
    )
}
fn transformed_total(total: &calc_flow::TableExpr) -> Result<calc_flow::TableExpr> {
    total
        .filter(total.column("total")?.ge(lit(0_i64)?))?
        .sql("WITH visible AS (SELECT total FROM input) SELECT total FROM visible")
}
async fn plan(program: &Program) -> Result<StreamExecutionPlan> {
    program
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
            &CompileOptions::default(),
        )
        .await
}

#[tokio::test]
async fn latest_admission_preserves_logical_consumer_and_exported_selection() -> Result<()> {
    let udfs = UdfRegistry::new().snapshot();
    for transform in [false, true] {
        let legacy = program(true, transform)?;
        let latest = legacy.with_update_semantics(UpdateSemantics::LatestRelation);
        assert_eq!(legacy.update_semantics(), UpdateSemantics::LegacyAppend);
        assert_ne!(latest.fingerprint(), legacy.fingerprint());
        assert_eq!(
            latest.for_mode(CompileMode::Batch)?.fingerprint(),
            latest.fingerprint()
        );
        assert_eq!(
            latest
                .with_update_semantics(UpdateSemantics::LegacyAppend)
                .fingerprint(),
            legacy.fingerprint()
        );
        let report = latest.analyze(&udfs, &AnalysisOptions::default()).await?;
        assert_eq!((report.schema_version, report.analysis_revision), (2, 5));
        assert!(report.request.is_none());
        let edge = report
            .edges
            .iter()
            .find(|edge| edge.path == "outputs.result.sql.tables.input")
            .unwrap();
        assert_eq!(edge.stream.producer_update, UpdateKind::CompleteSnapshot);
        assert_eq!(
            edge.stream.consumer_interpretation,
            InputInterpretation::CompleteReplacement
        );
        assert_eq!(edge.stream.support, SupportStatus::Supported);
        plan(&latest).await?;
        latest
            .for_mode(CompileMode::Batch)?
            .compile_batch(&udfs, &CompileOptions::default())
            .await?;
        assert_exported_selection(&latest, &udfs).await?;
    }
    Ok(())
}

async fn assert_exported_selection(
    latest: &Program,
    udfs: &calc_flow::UdfRegistrySnapshot,
) -> Result<()> {
    let exported = latest.to_project(udfs).await?;
    let encoded = calc_flow::export_project_json(&exported)?;
    let reloaded = calc_flow::import_project_json(encoded.as_bytes())?;
    assert_eq!(
        reloaded.graph.update_semantics,
        UpdateSemantics::LatestRelation
    );
    calc_flow::compile_stream_project_graph(
        &reloaded,
        &calc_flow::ProviderRegistry::default(),
        udfs,
        &StreamRequirements::default(),
    )?;
    Ok(())
}

struct Source {
    batches: std::vec::IntoIter<Batch>,
    calls: Arc<AtomicUsize>,
    sequence: u64,
}
#[async_trait]
impl StreamSource for Source {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 100,
            max_batch_bytes: 1 << 20,
            schema: SourceSchema::Exact(schema()),
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }
    async fn open(&mut self, _: Option<Cursor>) -> Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.sequence += 1;
        self.batches
            .next()
            .map(|batch| {
                Ok(SourceEvent::Data {
                    batch: batch.with_metadata(BatchMetadata::new(
                        "events",
                        self.sequence,
                        JsonMap::new(),
                    )?),
                    cursor: Cursor::new(
                        "events",
                        self.sequence.to_be_bytes().to_vec(),
                        JsonMap::new(),
                    )?,
                })
            })
            .transpose()
    }
    async fn close(&mut self) -> Result<()> {
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
async fn run(
    plan: StreamExecutionPlan,
    batches: Vec<Batch>,
    root: Option<&Path>,
    calls: &Arc<AtomicUsize>,
) -> Result<Vec<Batch>> {
    let output = Arc::new(Mutex::new(vec![]));
    let runner = make_runner(plan, batches, root, calls, &output)?;
    let (start, cleanup) = runner.start_with_cleanup();
    let started = start.await;
    if let Err(error) = started {
        cleanup.await?;
        return Err(error);
    }
    let job = started.unwrap();
    let outcome = job.wait().await;
    drop(job);
    cleanup.await?;
    assert_eq!(outcome.state, JobState::Completed, "{outcome:?}");
    assert!(outcome.errors.is_empty(), "{outcome:?}");
    Ok(output.lock().unwrap().clone())
}

fn make_runner(
    plan: StreamExecutionPlan,
    batches: Vec<Batch>,
    root: Option<&Path>,
    calls: &Arc<AtomicUsize>,
    output: &Arc<Mutex<Vec<Batch>>>,
) -> Result<StreamingRunner> {
    let sources = BTreeMap::from([(
        "events".into(),
        SourceBinding::new(Source {
            batches: batches.into_iter(),
            calls: calls.clone(),
            sequence: 0,
        })
        .with_watermark_policy(calc_flow::WatermarkPolicy::Disabled { idle_timeout: None }),
    )]);
    let sinks = BTreeMap::from([(
        "result".into(),
        vec![SinkBinding::ordinary("capture", Sink(output.clone()))?],
    )]);
    let runner = match root {
        Some(root) => {
            StreamingRunner::new(plan, sources, sinks, ManagedCheckpointRuntime::new(root)?)?
        }
        None => StreamingRunner::without_checkpoints(plan, sources, sinks)?,
    };
    Ok(runner)
}

async fn run_program(
    program: &Program,
    batches: Vec<Batch>,
    root: Option<&Path>,
    calls: &Arc<AtomicUsize>,
) -> Result<Vec<Batch>> {
    run(plan(program).await?, batches, root, calls).await
}

fn values(batch: &Batch) -> Vec<i64> {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect()
}

#[tokio::test]
async fn terminal_sum_keeps_cumulative_snapshots_in_both_choices() -> Result<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    for semantics in [
        UpdateSemantics::LegacyAppend,
        UpdateSemantics::LatestRelation,
    ] {
        let total = program(false, false)?.with_update_semantics(semantics);
        let output = run_program(
            &total,
            vec![data(&[&[1]]), data(&[&[2], &[3]])],
            None,
            &calls,
        )
        .await?;
        assert_eq!(
            output.iter().map(values).collect::<Vec<_>>(),
            vec![vec![1], vec![6]]
        );
    }
    Ok(())
}

#[tokio::test]
async fn legacy_chain_keeps_seven() -> Result<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let legacy = program(true, false)?;
    let output = run_program(&legacy, vec![data(&[&[1]]), data(&[&[2, 3]])], None, &calls).await?;
    assert_eq!(
        output.iter().map(values).collect::<Vec<_>>(),
        vec![vec![1], vec![7]]
    );
    Ok(())
}

#[tokio::test]
async fn latest_filter_delivers_empty_events() -> Result<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let latest = program(false, true)?.with_update_semantics(UpdateSemantics::LatestRelation);
    let output = run_program(&latest, vec![data(&[&[1]]), data(&[&[-2]])], None, &calls).await?;
    assert_eq!(output.len(), 2);
    assert_eq!(values(&output[0]), vec![1]);
    assert_eq!(output[1].num_rows(), 0);
    assert_eq!(
        output[0].table_payload()?.schema(),
        output[1].table_payload()?.schema()
    );
    assert!(run_program(&latest, vec![], None, &calls).await?.is_empty());
    Ok(())
}

#[tokio::test]
async fn latest_batch_plan_isolates_warm_executions() -> Result<()> {
    let batch = program(false, false)?
        .with_update_semantics(UpdateSemantics::LatestRelation)
        .for_mode(CompileMode::Batch)?;
    let plan = batch
        .compile_batch(&UdfRegistry::new().snapshot(), &CompileOptions::default())
        .await?;
    for (input, expected) in [(data(&[&[1]]), 1), (data(&[&[2, 3]]), 5)] {
        let result = plan
            .execute(
                BTreeMap::from([("events".into(), input)]),
                ExecutionOptions::default(),
            )
            .await?;
        assert_eq!(values(&result.outputs["result"]), vec![expected]);
    }
    Ok(())
}

#[tokio::test]
async fn managed_recovery_rejects_both_update_identity_mismatches_before_source() -> Result<()> {
    for original in [
        UpdateSemantics::LegacyAppend,
        UpdateSemantics::LatestRelation,
    ] {
        let directory = tempfile::tempdir().unwrap();
        let original_program = program(false, false)?.with_update_semantics(original);
        let calls = Arc::new(AtomicUsize::new(0));
        let saved = run_program(
            &original_program,
            vec![data(&[&[1]])],
            Some(directory.path()),
            &calls,
        )
        .await?;
        assert_eq!(values(&saved[0]), vec![1]);
        let changed = if original == UpdateSemantics::LegacyAppend {
            UpdateSemantics::LatestRelation
        } else {
            UpdateSemantics::LegacyAppend
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let error = run_program(
            &original_program.with_update_semantics(changed),
            vec![data(&[&[2, 3]])],
            Some(directory.path()),
            &calls,
        )
        .await
        .unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Streaming(ref error) if error.category() == StreamingErrorCategory::CheckpointMismatch),
            "{error}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(
            run_program(
                &original_program,
                vec![data(&[&[2, 3]])],
                Some(directory.path()),
                &calls
            )
            .await?
            .is_empty()
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
    Ok(())
}

#[tokio::test]
async fn one_multirecord_batch_is_one_event_and_empty_data_is_an_event() -> Result<()> {
    let source = table_input("events", schema(), InputOptions::default())?;
    let output = source.sql("SELECT x FROM input WHERE x >= 0")?;
    let latest = Program::new(
        "whole",
        CompileMode::Stream,
        BTreeMap::from([("result".into(), output)]),
    )?
    .with_update_semantics(UpdateSemantics::LatestRelation);
    let calls = Arc::new(AtomicUsize::new(0));
    let output = run_program(
        &latest,
        vec![data(&[&[1], &[2, 3]]), data(&[&[]])],
        None,
        &calls,
    )
    .await?;
    assert_eq!(output.len(), 2);
    assert_eq!(values(&output[0]), vec![1, 2, 3]);
    assert_eq!(output[1].num_rows(), 0);
    assert_eq!(
        output[0].table_payload()?.schema(),
        output[1].table_payload()?.schema()
    );
    Ok(())
}

#[tokio::test]
async fn unsupported_snapshot_consumers_remain_unknown_and_deterministic() -> Result<()> {
    let source = table_input("events", schema(), InputOptions::default())?;
    let total = source.sql("SELECT SUM(x) AS total FROM input")?;
    for query in [
        "SELECT DISTINCT total FROM input",
        "SELECT total FROM input ORDER BY total",
        "SELECT total FROM input LIMIT 1",
    ] {
        let output = total.sql(query)?;
        let latest = Program::new(
            "unproved",
            CompileMode::Stream,
            BTreeMap::from([("risk.total".into(), output)]),
        )?
        .with_update_semantics(UpdateSemantics::LatestRelation);
        let udfs = UdfRegistry::new().snapshot();
        let a = latest.analyze(&udfs, &AnalysisOptions::default()).await?;
        let b = latest.analyze(&udfs, &AnalysisOptions::default()).await?;
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            serde_json::to_value(&b).unwrap()
        );
        let edge = a
            .edges
            .iter()
            .find(|edge| edge.path == "outputs[\"risk.total\"].sql.tables.input")
            .unwrap();
        assert_eq!(edge.stream.support, SupportStatus::Unknown);
        let error = plan(&latest).await.err().unwrap();
        assert!(
            error.to_string().contains("update_semantics_unproved"),
            "{error}"
        );
    }
    Ok(())
}

#[test]
fn low_level_latest_rejects_missing_boundary_proof_synchronously() {
    let builder = calc_flow::PipelineBuilder::new("unproved")
        .unwrap()
        .with_update_semantics(UpdateSemantics::LatestRelation)
        .add_node(
            "select",
            Box::new(
                calc_flow::SqlOperator::new(
                    "select",
                    "SELECT x FROM input",
                    vec!["input".into()],
                    vec![],
                )
                .unwrap(),
            ),
        )
        .unwrap();
    let error = builder
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .err()
        .unwrap();
    assert!(matches!(error, CalcFlowError::QueryUpdateSemantics { .. }));
    assert!(
        error.to_string().contains("update_event_boundary_unproved"),
        "{error}"
    );
}

#[tokio::test]
async fn invalid_project_update_literal_has_stable_path_and_code() -> Result<()> {
    let project = program(false, false)?
        .to_project(&UdfRegistry::new().snapshot())
        .await?;
    let mut value = serde_json::to_value(&project).unwrap();
    for literal in [
        serde_json::json!("future"),
        serde_json::json!(false),
        serde_json::json!(null),
        serde_json::json!({}),
    ] {
        value["graph"]["update_semantics"] = literal;
        let error =
            calc_flow::import_project_json(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        let CalcFlowError::ProjectValidation { issues } = error else {
            panic!("{error}");
        };
        assert_eq!(issues[0].path, "graph.update_semantics");
        assert_eq!(issues[0].code, "invalid_literal");
    }
    Ok(())
}

#[tokio::test]
async fn dataframe_sql_replacement_chain_is_six_legacy_is_seven() -> Result<()> {
    for semantics in [
        UpdateSemantics::LatestRelation,
        UpdateSemantics::LegacyAppend,
    ] {
        let program = replacement_chain(semantics)?;
        check_replacement_chain_stream(&program, semantics).await?;
        check_replacement_chain_batch(&program).await?;
    }
    Ok(())
}

fn replacement_chain(semantics: UpdateSemantics) -> Result<Program> {
    let source = table_input("events", schema(), InputOptions::default())?;
    let source = source.sql("SELECT x FROM input")?;
    let first = source.aggregate(vec![source.column("x")?.sum().alias("total")?], vec![])?;
    Program::new(
        "aggregate",
        CompileMode::Stream,
        BTreeMap::from([("result".into(), visible_replacement_total(&first)?)]),
    )
    .map(|program| program.with_update_semantics(semantics))
}

fn visible_replacement_total(first: &calc_flow::TableExpr) -> Result<calc_flow::TableExpr> {
    let visible = first.select(vec![first.column("total")?.alias("total")?])?;
    let visible = visible.filter(visible.column("total")?.ge(lit(0_i64)?))?;
    visible.sql("SELECT SUM(total) AS total FROM input")
}

async fn check_replacement_chain_stream(
    program: &Program,
    semantics: UpdateSemantics,
) -> Result<()> {
    let output = Arc::new(Mutex::new(vec![]));
    let runner = make_runner(
        plan(program).await?,
        vec![data(&[&[1]]), data(&[&[2], &[3]])],
        None,
        &Arc::new(AtomicUsize::new(0)),
        &output,
    )?
    .with_runtime_config(calc_flow::StreamRuntimeConfig {
        checkpointing: false,
        sql_replacement_state_budget: Some(calc_flow::StateBudget::new(100, 1 << 24)?),
        ..Default::default()
    })?;
    let job = runner.start().await?;
    assert_eq!(job.wait().await.state, JobState::Completed);
    let actual = output
        .lock()
        .unwrap()
        .iter()
        .flat_map(values)
        .collect::<Vec<_>>();
    let last = if semantics == UpdateSemantics::LatestRelation {
        6
    } else {
        7
    };
    assert_eq!(actual, vec![1, last]);
    Ok(())
}

async fn check_replacement_chain_batch(program: &Program) -> Result<()> {
    let plan = program
        .for_mode(CompileMode::Batch)?
        .compile_batch(&UdfRegistry::new().snapshot(), &CompileOptions::default())
        .await?;
    for (parts, expected) in [(vec![1], 1), (vec![2, 3], 5)] {
        let result = plan
            .execute(
                BTreeMap::from([("events".into(), data(&[&parts]))]),
                ExecutionOptions::default(),
            )
            .await?;
        assert_eq!(values(&result.outputs["result"]), vec![expected]);
    }
    Ok(())
}

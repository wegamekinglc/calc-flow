use calc_flow::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use calc_flow::{
    AggregateFunction, CompileMode, InputOptions, Program, Result, WindowSpec, table_input,
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

#[tokio::test]
async fn dataframe_and_sql_windows_export_native_nodes() -> Result<()> {
    exercise_frontends(true).await?;
    exercise_frontends(false).await
}

#[tokio::test]
async fn mixed_window_paths_preserve_time_proof_and_independent_union() -> Result<()> {
    let udfs = calc_flow::UdfRegistry::new().snapshot();
    for sql in [false, true] {
        let input = table_input(
            "events",
            sample_batch().table_payload()?.schema().clone(),
            InputOptions::default(),
        )?;
        let bars = mixed_window(&input, sql)?;
        let post = bars
            .select(vec![bars.column("n")?.alias("n")?])?
            .sql("SELECT * FROM input")?;
        let program = Program::new(
            "mixed",
            CompileMode::Batch,
            BTreeMap::from([
                ("bars".into(), bars),
                ("post".into(), post),
                ("outside".into(), input.union_all(&input)?),
            ]),
        )?;
        let project = program.to_project(&udfs).await?;
        assert_eq!(
            project
                .graph
                .nodes
                .iter()
                .filter(|n| matches!(n.operator, calc_flow::OperatorSpec::Window { .. }))
                .count(),
            1
        );
        let plan = program
            .compile_batch(&udfs, &calc_flow::CompileOptions::default())
            .await?;
        let result = plan
            .execute(
                BTreeMap::from([("events".into(), sample_batch())]),
                calc_flow::ExecutionOptions::default(),
            )
            .await?;
        assert_eq!(result.outputs["bars"].num_rows(), 3);
        assert_eq!(result.outputs["post"].num_rows(), 3);
        assert_eq!(result.outputs["outside"].num_rows(), 6);
    }
    Ok(())
}

fn mixed_window(input: &calc_flow::TableExpr, sql: bool) -> Result<calc_flow::TableExpr> {
    let renamed = input.sql("SELECT input.ts AS time, q FROM input WHERE q > 0")?;
    if sql {
        calc_flow::sql::sql(
            "SELECT window_start, window_end, COUNT(q) AS n FROM CF_TUMBLE('a', 'time', 10) GROUP BY window_start, window_end",
            BTreeMap::from([("a".into(), renamed)]),
        )
    } else {
        renamed.window(
            WindowSpec::tumbling("time", Duration::from_micros(10))?.aggregate(
                AggregateFunction::Count,
                "q",
                "n",
            )?,
        )
    }
}
#[tokio::test]
async fn rewritten_sql_time_is_rejected_at_the_window_field() -> Result<()> {
    let input = table_input(
        "events",
        sample_batch().table_payload()?.schema().clone(),
        InputOptions::default(),
    )?;
    let rewritten = input.sql("SELECT ts + INTERVAL '1 microsecond' AS ts, q FROM input")?;
    let bars = rewritten.window(
        WindowSpec::tumbling("ts", Duration::from_micros(10))?.aggregate(
            AggregateFunction::Count,
            "q",
            "n",
        )?,
    )?;
    let program = Program::new(
        "time",
        CompileMode::Batch,
        BTreeMap::from([("bars".into(), bars)]),
    )?;
    let error = program
        .collect_schema(&calc_flow::UdfRegistry::new().snapshot())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("outputs.bars.window.event_time"), "{error}");
    assert!(
        error.contains("unchanged input column or pure rename"),
        "{error}"
    );
    Ok(())
}
async fn exercise_frontends(hopping: bool) -> Result<()> {
    let program = window_program(hopping)?;
    let udfs = calc_flow::UdfRegistry::new().snapshot();
    let options = calc_flow::CompileOptions::default();
    let project = program.to_project(&udfs).await?;
    assert_eq!(
        project
            .graph
            .nodes
            .iter()
            .filter(|n| matches!(n.operator, calc_flow::OperatorSpec::Window { .. }))
            .count(),
        2
    );
    let batch = program.compile_batch(&udfs, &options).await?;
    let result = batch
        .execute(
            BTreeMap::from([("events".into(), sample_batch())]),
            calc_flow::ExecutionOptions::default(),
        )
        .await?;
    for output in result.outputs.values() {
        assert_eq!(output.num_rows(), if hopping { 5 } else { 3 });
    }

    let stream = program
        .for_mode(CompileMode::Stream)?
        .compile_stream(&udfs, &calc_flow::StreamRequirements::default(), &options)
        .await?;
    let expected = if hopping {
        vec![(-10, 1), (-5, 2), (0, 1), (5, 1), (10, 1)]
    } else {
        vec![(-10, 1), (0, 1), (10, 1)]
    };
    assert_stream(stream, &expected).await?;
    Ok(())
}

fn window_program(hopping: bool) -> Result<Program> {
    let input = table_input(
        "events",
        Arc::new(Schema::new(vec![
            Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
            Field::new("q", DataType::Int64, true),
        ])),
        InputOptions::default(),
    )?;
    let spec = if hopping {
        WindowSpec::hopping("ts", Duration::from_micros(10), Duration::from_micros(5))?
    } else {
        WindowSpec::tumbling("ts", Duration::from_micros(10))?
    };
    let df = input.window(spec.aggregate(AggregateFunction::Count, "q", "n")?)?;
    let call = if hopping {
        "CF_HOP('a', 'ts', 10, 5)"
    } else {
        "CF_TUMBLE('a', 'ts', 10)"
    };
    let sql = calc_flow::sql::sql(
        &format!(
            "SELECT window_start, window_end, COUNT(q) AS n_sql FROM {call} GROUP BY window_start, window_end"
        ),
        BTreeMap::from([("a".into(), input)]),
    )?;
    Program::new(
        "windows",
        CompileMode::Batch,
        BTreeMap::from([("df".into(), df), ("sql".into(), sql)]),
    )
}

fn sample_batch() -> calc_flow::Batch {
    use calc_flow::arrow::{
        array::{Int64Array, TimestampMicrosecondArray},
        record_batch::RecordBatch,
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("q", DataType::Int64, true),
    ]));
    calc_flow::Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(TimestampMicrosecondArray::from(vec![-1, 0, 10])),
                    Arc::new(Int64Array::from(vec![1, 1, 1])),
                ],
            )
            .unwrap(),
        ],
        calc_flow::BatchMetadata::default(),
    )
    .unwrap()
}
struct FiniteSource {
    step: usize,
}
#[async_trait::async_trait]
impl calc_flow::StreamSource for FiniteSource {
    fn capabilities(&self) -> calc_flow::SourceCapabilities {
        calc_flow::SourceCapabilities {
            replay_positioning: calc_flow::ReplayPositioning::ExactPauseReportAndSeek,
            delivery: calc_flow::SourceDeliveryCapability::Lossless,
            max_batch_rows: 3,
            max_batch_bytes: 4096,
            schema: calc_flow::SourceSchema::Exact(
                sample_batch().table_payload().unwrap().schema().clone(),
            ),
            native_watermarks: calc_flow::NativeWatermarkCapability::NeverEmits,
        }
    }
    async fn open(&mut self, _cursor: Option<calc_flow::Cursor>) -> Result<()> {
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<calc_flow::SourceEvent>> {
        self.step += 1;
        if self.step > 1 {
            return Ok(None);
        }
        Ok(Some(calc_flow::SourceEvent::Data {
            batch: sample_batch(),
            cursor: calc_flow::Cursor::unbound(vec![1], BTreeMap::new())?,
        }))
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
struct RowsSink(Arc<tokio::sync::Mutex<Vec<(i64, u64)>>>);
#[async_trait::async_trait]
impl calc_flow::StreamSink for RowsSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &calc_flow::Batch) -> Result<()> {
        use calc_flow::arrow::array::{TimestampMicrosecondArray, UInt64Array};
        let mut rows = self.0.lock().await;
        for record in batch.table_payload()?.batches() {
            let starts = record
                .column(0)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            let counts = record
                .column(2)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            rows.extend((0..record.num_rows()).map(|i| (starts.value(i), counts.value(i))));
        }
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
async fn assert_stream(
    plan: calc_flow::StreamExecutionPlan,
    expected: &[(i64, u64)],
) -> Result<()> {
    let mut observed = Vec::new();
    let sinks = plan
        .sink_binding_ids()
        .iter()
        .enumerate()
        .map(|(i, id)| {
            let rows = Arc::new(tokio::sync::Mutex::new(Vec::new()));
            observed.push(rows.clone());
            Ok((
                (*id).to_owned(),
                vec![calc_flow::SinkBinding::ordinary(
                    &format!("sink_{i}"),
                    RowsSink(rows),
                )?],
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let source_id = plan.source_binding_ids()[0].to_owned();
    let root = tempfile::tempdir().unwrap();
    let runner = calc_flow::StreamingRunner::new(
        plan,
        BTreeMap::from([(
            source_id,
            calc_flow::SourceBinding::new(FiniteSource { step: 0 })
                .with_watermark_policy(calc_flow::WatermarkPolicy::Disabled { idle_timeout: None }),
        )]),
        sinks,
        calc_flow::ManagedCheckpointRuntime::new(root.path())?,
    )?;
    let outcome = runner.start().await?.wait().await;
    assert_eq!(format!("{:?}", outcome.state), "Completed", "{outcome:?}");
    for rows in observed {
        assert_eq!(rows.lock().await.as_slice(), expected);
    }
    Ok(())
}

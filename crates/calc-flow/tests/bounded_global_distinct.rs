use calc_flow::CompileOptions;
use std::{collections::BTreeMap, sync::Arc};

use calc_flow::arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use calc_flow::{
    Batch, BatchMetadata, CompileMode, DistinctSpec, ExecutionOptions, InputOptions, Program,
    Result, StateBudget, UdfRegistry, table_input,
};

#[tokio::test]
async fn test_distinct_boundedness_is_checked_before_source_or_sink_open() -> Result<()> {
    use calc_flow::{
        SinkBinding, SourceBinding, SourceBoundedness, StreamRequirements, StreamingRunner,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    let opened = Arc::new(AtomicUsize::new(0));
    for boundedness in [SourceBoundedness::Unknown, SourceBoundedness::Unbounded] {
        let input = table_input(
            "events",
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)])),
            InputOptions::default(),
        )?;
        let out = input.distinct(
            vec!["x".into()],
            DistinctSpec::bounded_source(StateBudget::new(2, 8 << 20)?),
        )?;
        let program = Program::new(
            "preflight",
            CompileMode::Stream,
            BTreeMap::from([("out".into(), out)]),
        )?;
        let plan = program
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
                &CompileOptions::default(),
            )
            .await?;
        let runner = StreamingRunner::without_checkpoints(
            plan,
            BTreeMap::from([(
                "events".into(),
                SourceBinding::new(ProofSource {
                    opened: opened.clone(),
                    boundedness,
                }),
            )]),
            BTreeMap::from([(
                "out".into(),
                vec![SinkBinding::ordinary("sink", ProofSink(opened.clone()))?],
            )]),
        )?;
        let Err(error) = runner.start().await else {
            panic!("unproved source was opened")
        };
        assert!(
            error
                .to_string()
                .contains(if boundedness == SourceBoundedness::Unknown {
                    "source_boundedness_unproved"
                } else {
                    "source_unbounded"
                }),
            "{error}"
        );
        assert_eq!(opened.load(Ordering::SeqCst), 0);
    }
    Ok(())
}

struct ProofSource {
    opened: Arc<std::sync::atomic::AtomicUsize>,
    boundedness: calc_flow::SourceBoundedness,
}
#[async_trait::async_trait]
impl calc_flow::StreamSource for ProofSource {
    fn boundedness(&self) -> calc_flow::SourceBoundedness {
        self.boundedness
    }
    fn capabilities(&self) -> calc_flow::SourceCapabilities {
        calc_flow::SourceCapabilities {
            replay_positioning: calc_flow::ReplayPositioning::Unsupported,
            delivery: calc_flow::SourceDeliveryCapability::Lossless,
            max_batch_rows: 2,
            max_batch_bytes: 1024,
            schema: calc_flow::SourceSchema::Exact(Arc::new(Schema::new(vec![Field::new(
                "x",
                DataType::Int64,
                true,
            )]))),
            native_watermarks: calc_flow::NativeWatermarkCapability::NeverEmits,
        }
    }
    async fn open(&mut self, _: Option<calc_flow::Cursor>) -> Result<()> {
        self.opened
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<calc_flow::SourceEvent>> {
        Ok(None)
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
struct ProofSink(Arc<std::sync::atomic::AtomicUsize>);
#[async_trait::async_trait]
impl calc_flow::StreamSink for ProofSink {
    async fn open(&mut self) -> Result<()> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
    async fn write(&mut self, _: &Batch) -> Result<()> {
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct FiniteSource(std::collections::VecDeque<calc_flow::SourceEvent>);
#[async_trait::async_trait]
impl calc_flow::StreamSource for FiniteSource {
    fn boundedness(&self) -> calc_flow::SourceBoundedness {
        calc_flow::SourceBoundedness::Bounded
    }
    fn capabilities(&self) -> calc_flow::SourceCapabilities {
        calc_flow::SourceCapabilities {
            replay_positioning: calc_flow::ReplayPositioning::Unsupported,
            delivery: calc_flow::SourceDeliveryCapability::Lossless,
            max_batch_rows: 2,
            max_batch_bytes: 1024,
            schema: calc_flow::SourceSchema::Exact(Arc::new(Schema::new(vec![Field::new(
                "x",
                DataType::Int64,
                true,
            )]))),
            native_watermarks: calc_flow::NativeWatermarkCapability::NeverEmits,
        }
    }
    async fn open(&mut self, _: Option<calc_flow::Cursor>) -> Result<()> {
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<calc_flow::SourceEvent>> {
        Ok(self.0.pop_front())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
type Written = Arc<std::sync::Mutex<BTreeMap<String, Vec<Batch>>>>;
struct StoreSink(String, Written);
#[async_trait::async_trait]
impl calc_flow::StreamSink for StoreSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.1
            .lock()
            .unwrap()
            .entry(self.0.clone())
            .or_default()
            .push(batch.clone());
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn test_rust_df_and_sql_stream_literal_final_relations_and_old_per_batch() -> Result<()> {
    use calc_flow::{
        Cursor, JobState, JsonMap, SinkBinding, SourceBinding, SourceEvent, StreamRequirements,
        StreamingRunner, WatermarkPolicy,
    };
    let input = table_input(
        "events",
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)])),
        InputOptions::default(),
    )?;
    let df = input.distinct(
        vec!["x".into()],
        DistinctSpec::bounded_source(StateBudget::new(4, 8 << 20)?),
    )?;
    let sq = calc_flow::sql::sql(
        "SELECT DISTINCT x FROM CF_DISTINCT_BOUNDED('a',4,8388608)",
        BTreeMap::from([("a".into(), input.clone())]),
    )?;
    let legacy = input.sql("SELECT DISTINCT x FROM input")?;
    let program = Program::new(
        "literal_stream",
        CompileMode::Stream,
        BTreeMap::from([
            ("df".into(), df),
            ("sql".into(), sq),
            ("legacy".into(), legacy),
        ]),
    )?;
    let plan = program
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
            &CompileOptions::default(),
        )
        .await?;
    let source = FiniteSource(
        (1..=2)
            .map(|sequence| SourceEvent::Data {
                batch: x_data(vec![Some(1), None]),
                cursor: Cursor::unbound(vec![sequence], JsonMap::new()).unwrap(),
            })
            .collect(),
    );
    let written = Written::default();
    let sinks = ["df", "sql", "legacy"]
        .into_iter()
        .map(|name| {
            Ok((
                name.into(),
                vec![SinkBinding::ordinary(
                    name,
                    StoreSink(name.into(), written.clone()),
                )?],
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let job = StreamingRunner::without_checkpoints(
        plan,
        BTreeMap::from([(
            "events".into(),
            SourceBinding::new(source)
                .with_watermark_policy(WatermarkPolicy::Disabled { idle_timeout: None }),
        )]),
        sinks,
    )?
    .start()
    .await?;
    assert_eq!(job.wait().await.state, JobState::Completed);
    let written = written.lock().unwrap();
    for name in ["df", "sql"] {
        assert_eq!(written[name].len(), 1);
        let mut values = written[name][0]
            .table_payload()?
            .batches()
            .iter()
            .flat_map(|r| {
                r.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
            })
            .collect::<Vec<_>>();
        values.sort_unstable();
        assert_eq!(values, vec![None, Some(1)]);
    }
    assert_eq!(written["legacy"].len(), 2);
    Ok(())
}

fn unique_operator(rows: u64) -> calc_flow::DistinctOperator {
    calc_flow::DistinctOperator::new(
        "unique",
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)])),
        vec!["x".into()],
        DistinctSpec::bounded_source(StateBudget::new(rows, 8 << 20).unwrap()),
    )
    .unwrap()
}

#[test]
fn test_distinct_budget_failure_names_node_policy_role_required_and_limit() {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let error = calc_flow::DistinctOperator::new(
        "budget_node",
        schema,
        vec!["x".into()],
        DistinctSpec::bounded_source(StateBudget::new(2, 1).unwrap()),
    )
    .err()
    .unwrap()
    .to_string();
    for text in [
        "budget_node",
        "copy_visible_v2",
        "schema",
        "required",
        "limit",
    ] {
        assert!(error.contains(text), "{error}");
    }
}
fn x_data(values: Vec<Option<i64>>) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)])),
                vec![Arc::new(Int64Array::from(values))],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

struct RefuseOutput;
#[async_trait::async_trait]
impl calc_flow::StreamCollector for RefuseOutput {
    async fn emit(&mut self, _: &str, _: Batch) -> Result<()> {
        Err(calc_flow::CalcFlowError::Internal {
            message: "refused final output".into(),
        })
    }
}

#[tokio::test]
async fn test_restore_rebuilds_capture_without_retaining_input_wire_owner() -> Result<()> {
    use calc_flow::{
        CancellationToken, EdgeCollector, Epoch, JsonMap, OperatorMetadata, StreamJobContext,
        StreamOperator, StreamOperatorContext,
    };
    let job = StreamJobContext::new(
        1,
        "restore_owned",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "unique", None);
    let mut op = unique_operator(2);
    let mut collector = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("input", x_data(vec![Some(1)]), &context, &mut collector)
        .await?;
    let capture = op.checkpoint(Epoch::INITIAL)?;
    let original_wire = capture.segments["data"].bytes_arc();
    let mut resumed = unique_operator(2);
    resumed.restore(&capture)?;
    drop(capture);
    drop(op);
    assert_eq!(
        Arc::strong_count(&original_wire),
        1,
        "restore must detach cached capture from input segment ownership"
    );
    resumed.on_end(&context, &mut collector).await?;
    assert_eq!(collector.drain("output").len(), 1);
    Ok(())
}

#[tokio::test]
async fn test_duplicate_row_budget_and_failed_emit_preserve_committed_capture() -> Result<()> {
    use calc_flow::{
        CancellationToken, EdgeCollector, Epoch, JsonMap, OperatorMetadata, StreamJobContext,
        StreamOperator, StreamOperatorContext,
    };
    let job = StreamJobContext::new(1, "atomic", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "unique", None);
    let mut op = unique_operator(2);
    let mut collector = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("input", x_data(vec![Some(1)]), &context, &mut collector)
        .await?;
    let before = op.checkpoint(Epoch::INITIAL)?;
    assert!(
        op.process_data(
            "input",
            x_data(vec![Some(1), Some(1)]),
            &context,
            &mut collector
        )
        .await
        .is_err()
    );
    assert_eq!(
        op.checkpoint(Epoch::INITIAL)?.segments["control"],
        before.segments["control"]
    );
    assert!(op.on_end(&context, &mut RefuseOutput).await.is_err());
    assert_eq!(
        op.checkpoint(Epoch::INITIAL)?.segments["control"],
        before.segments["control"]
    );
    op.process_data("input", x_data(vec![Some(1)]), &context, &mut collector)
        .await?;
    let accepted = op.checkpoint(Epoch::INITIAL)?;
    let control: serde_json::Value =
        serde_json::from_slice(accepted.segments["control"].bytes()).unwrap();
    assert_eq!(control["accepted_rows"], 2);
    op.on_end(&context, &mut collector).await?;
    assert_eq!(collector.drain("output").len(), 1);
    Ok(())
}
#[tokio::test]
async fn test_distinct_checkpoint_preserves_duplicate_history_and_closed_state() -> Result<()> {
    use calc_flow::{
        CancellationToken, EdgeCollector, Epoch, JsonMap, OperatorMetadata, StreamJobContext,
        StreamOperator, StreamOperatorContext,
    };
    let job = StreamJobContext::new(1, "test", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "unique", None);
    let mut op = unique_operator(2);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("input", x_data(vec![Some(1)]), &context, &mut output)
        .await?;
    let capture = op.checkpoint(Epoch::INITIAL)?;
    assert_eq!(
        capture.inline_metadata.get("state_layout"),
        Some(&serde_json::json!(2))
    );
    assert_eq!(
        capture.inline_metadata.get("accounting_version"),
        Some(&serde_json::json!(2))
    );
    let mut resumed = unique_operator(2);
    resumed.restore(&capture)?;
    resumed
        .process_data("input", x_data(vec![Some(1)]), &context, &mut output)
        .await?;
    let next = resumed.checkpoint(Epoch::INITIAL)?;
    let control: serde_json::Value =
        serde_json::from_slice(next.segments["control"].bytes()).unwrap();
    assert_eq!(control["accepted_rows"], 2);
    assert_eq!(control["retained_rows"], 2);
    resumed.on_end(&context, &mut output).await?;
    let events = output.drain("output");
    assert_eq!(events.len(), 1);
    let record = &events[0].as_data().unwrap().table_payload()?.batches()[0];
    assert_eq!(
        record
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(1)]
    );
    let terminal = resumed.checkpoint(Epoch::INITIAL)?;
    let mut closed = unique_operator(2);
    closed.restore(&terminal)?;
    closed.on_end(&context, &mut output).await?;
    assert!(output.drain("output").is_empty());
    let mut bad = capture.clone();
    bad.inline_metadata
        .insert("state_layout".into(), serde_json::json!(1));
    assert!(op.restore(&bad).is_err());
    let after = op.checkpoint(Epoch::INITIAL)?;
    assert_eq!(after.segments["control"], capture.segments["control"]);
    Ok(())
}

#[tokio::test]
async fn test_distinct_sql_macro_exports_native_policy() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let input = table_input("events", schema, InputOptions::default())?;
    let output = calc_flow::sql::sql(
        "SELECT DISTINCT x FROM CF_DISTINCT_BOUNDED('a', 2, 8388608)",
        BTreeMap::from([("a".into(), input)]),
    )?;
    let program = Program::new(
        "unique_sql",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), output)]),
    )?;
    let project = program.to_project(&UdfRegistry::new().snapshot()).await?;
    let value = serde_json::to_value(project).unwrap();
    let nodes = value["graph"]["nodes"].as_array().unwrap();
    assert_eq!(
        nodes
            .iter()
            .filter(|n| n["operator"]["kind"] == "distinct")
            .count(),
        1
    );
    let node = nodes
        .iter()
        .find(|n| n["operator"]["kind"] == "distinct")
        .unwrap();
    assert_eq!(node["operator"]["columns"], serde_json::json!(["x"]));
    assert_eq!(
        node["operator"]["spec"]["ingress_policy"],
        "copy_visible_v2"
    );
    Ok(())
}

#[tokio::test]
async fn test_distinct_stream_copies_raw_owner_and_waits_for_real_end() -> Result<()> {
    use calc_flow::{
        CancellationToken, DistinctOperator, EdgeCollector, JsonMap, OperatorMetadata,
        StreamJobContext, StreamOperator, StreamOperatorContext,
    };
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let mut op = DistinctOperator::new(
        "unique",
        schema.clone(),
        vec!["x".into()],
        DistinctSpec::bounded_source(StateBudget::new(2, 8 << 20)?),
    )?;
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    let job = StreamJobContext::new(1, "test", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "unique", None);
    let source = Arc::new(Int64Array::from(vec![Some(1); 1_048_576]));
    let observed_buffer = source.values().inner().clone();
    let sliced = calc_flow::arrow::array::Array::slice(source.as_ref(), 1, 1);
    let raw = Batch::table(
        vec![RecordBatch::try_new(schema.clone(), vec![sliced]).unwrap()],
        BatchMetadata::default(),
    )?;
    drop(source);
    op.process_data("input", raw, &context, &mut output).await?;
    assert_eq!(
        observed_buffer.strong_count(),
        1,
        "only the observer may keep the original buffer; H must be independent"
    );
    assert!(output.drain("output").is_empty());
    let second = Batch::table(
        vec![
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![Some(1)]))]).unwrap(),
        ],
        BatchMetadata::default(),
    )?;
    op.process_data("input", second, &context, &mut output)
        .await?;
    assert!(output.drain("output").is_empty());
    op.on_end(&context, &mut output).await?;
    assert_eq!(output.drain("output").len(), 1);
    op.on_end(&context, &mut output).await?;
    assert!(output.drain("output").is_empty());
    Ok(())
}

#[tokio::test]
async fn test_distinct_v2_finite_owned_history_and_policy() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let input = table_input("events", schema.clone(), InputOptions::default())?;
    let output = input.distinct(
        vec!["x".into()],
        DistinctSpec::bounded_source(StateBudget::new(5, 8 << 20)?),
    )?;
    let program = Program::new(
        "unique",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), output)]),
    )?;
    let project = program.to_project(&UdfRegistry::new().snapshot()).await?;
    let spec = serde_json::to_value(project).unwrap();
    let distinct = spec["graph"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["operator"]["kind"] == "distinct")
        .unwrap();
    assert_eq!(
        distinct["operator"]["spec"]["ingress_policy"],
        "copy_visible_v2"
    );
    let plan = program
        .compile_batch(&UdfRegistry::new().snapshot(), &CompileOptions::default())
        .await?;
    for (values, expected) in [
        (
            vec![Some(1), None, Some(1), None, Some(2)],
            vec![None, Some(1), Some(2)],
        ),
        (vec![Some(7)], vec![Some(7)]),
        (vec![], vec![]),
    ] {
        let record =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(values))]).unwrap();
        let batch = Batch::table(vec![record], BatchMetadata::default())?;
        let result = plan
            .execute(
                BTreeMap::from([("events".into(), batch)]),
                ExecutionOptions::default(),
            )
            .await?;
        let table = result.outputs["out"].table_payload()?;
        assert_eq!(table.schema(), &schema);
        let mut actual = table
            .batches()
            .iter()
            .flat_map(|r| {
                r.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
            })
            .collect::<Vec<_>>();
        actual.sort_unstable();
        assert_eq!(actual, expected);
    }
    Ok(())
}

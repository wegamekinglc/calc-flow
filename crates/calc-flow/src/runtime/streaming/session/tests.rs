use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex},
};

use datafusion::arrow::{
    array::{
        ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray, TimestampSecondArray, UInt64Array,
    },
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
    util::display::{ArrayFormatter, FormatOptions},
};
use serde_json::{Value, json};

use super::{SessionWatermark, StreamSession};
use crate::runtime::streaming::progress::{
    GeneratedWatermarkState,
    prepare::{ArrowTimestampUnit, ResolvedEventTimeColumn},
};
use crate::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, Cursor, Edge, EdgeBudget, Epoch, EventTime,
    ExpressionOperator, JobState, JsonMap, ManagedCheckpointRuntime, NativeWatermarkCapability,
    OperatorMetadata, PipelineBuilder, Port, PortEndpoint, ReplayPositioning, RollingOperator,
    RollingSpec, SinkBinding, SourceBinding, SourceCapabilities, SourceDeliveryCapability,
    SourceEvent, SourceSchema, StreamExecutionPlan, StreamRequirements, StreamRuntimeConfig,
    StreamSink, StreamSource, StreamingRunner, UdfRegistry, WatermarkPolicy,
};

type Row = (i64, &'static str, u64, f64);

fn input_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, false),
    ]))
}

fn input_batch(rows: &[Row]) -> Batch {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(rows.iter().map(|row| row.0))
                .with_timezone("UTC"),
        ),
        Arc::new(StringArray::from_iter_values(rows.iter().map(|row| row.1))),
        Arc::new(UInt64Array::from_iter_values(rows.iter().map(|row| row.2))),
        Arc::new(Float64Array::from_iter_values(rows.iter().map(|row| row.3))),
    ];
    Batch::table(
        vec![RecordBatch::try_new(input_schema(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn error_policy() -> Value {
    json!({"kind": "error", "scope": "envelope"})
}

fn rolling_spec(late_policy: &Value) -> RollingSpec {
    serde_json::from_value(json!({
        "configuration_version": 1, "state_layout_version": 1,
        "partition_by": ["symbol"], "event_time": "ts", "sequence_by": ["sequence"],
        "outputs": [{
            "kind": "mean", "primitive_version": 1, "input": "price", "output": "price_mean",
            "frame": {"kind": "rows", "size": 2}, "min_periods": 1,
        }],
        "allowed_lateness_micros": 0, "late_policy": late_policy,
        "value_policy": "stateful_numeric_v1",
    }))
    .unwrap()
}

fn rolling(node_id: &str, late_policy: &Value) -> RollingOperator {
    RollingOperator::new(node_id, input_schema(), rolling_spec(late_policy)).unwrap()
}

fn compile(builder: PipelineBuilder) -> StreamExecutionPlan {
    builder
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

fn connect(
    builder: PipelineBuilder,
    source: (&str, &str),
    target: (&str, &str),
) -> PipelineBuilder {
    builder
        .connect(Edge::new(
            PortEndpoint::new(source.0, source.1).unwrap(),
            PortEndpoint::new(target.0, target.1).unwrap(),
        ))
        .unwrap()
}

/// An exact projection of rolling output columns `ts, symbol, price_mean`.
fn projection(node_id: &str, rolled: &SchemaRef) -> ExpressionOperator {
    let projected = Arc::new(rolled.project(&[0, 1, 4]).unwrap());
    ExpressionOperator::new(
        node_id,
        "",
        vec!["ts".into(), "symbol".into(), "price_mean".into()],
        None,
        vec![],
    )
    .unwrap()
    .with_ports(
        Port::with_schema_ref("input", BatchKind::Table, true, Some(Arc::clone(rolled))).unwrap(),
        Port::with_schema_ref("output", BatchKind::Table, true, Some(projected)).unwrap(),
    )
    .unwrap()
}

fn rolling_projection_plan() -> StreamExecutionPlan {
    let rolling = rolling("rolling", &error_policy());
    let rolled = rolling.output_ports()[0].schema().unwrap().clone();
    let builder = PipelineBuilder::new("session")
        .unwrap()
        .add_node("rolling", rolling)
        .unwrap()
        .add_node("project", Box::new(projection("project", &rolled)))
        .unwrap();
    compile(connect(
        builder,
        ("rolling", "output"),
        ("project", "input"),
    ))
}

fn session_config() -> StreamRuntimeConfig {
    StreamRuntimeConfig {
        checkpointing: false,
        ..StreamRuntimeConfig::default()
    }
}

fn open_session(
    plan: StreamExecutionPlan,
    watermarks: BTreeMap<String, SessionWatermark>,
) -> StreamSession {
    StreamSession::new(plan, &session_config(), watermarks).unwrap()
}

fn micros(value: i64) -> EventTime {
    EventTime::from_micros(value)
}

/// Renders each row as `column|column|...` in emission order.
fn rendered_rows(batches: &[Batch]) -> Vec<String> {
    let options = FormatOptions::default();
    let mut rows = Vec::new();
    for batch in batches {
        for record in batch.table_payload().unwrap().batches() {
            let formatters = record
                .columns()
                .iter()
                .map(|column| ArrayFormatter::try_new(column.as_ref(), &options).unwrap())
                .collect::<Vec<_>>();
            for row in 0..record.num_rows() {
                rows.push(
                    formatters
                        .iter()
                        .map(|formatter| formatter.value(row).to_string())
                        .collect::<Vec<_>>()
                        .join("|"),
                );
            }
        }
    }
    rows
}

struct ScriptedSource(VecDeque<SourceEvent>);

#[async_trait::async_trait]
impl StreamSource for ScriptedSource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 1_024,
            max_batch_bytes: 1 << 20,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: NativeWatermarkCapability::EmitsNative,
        }
    }

    async fn open(&mut self, _cursor: Option<Cursor>) -> crate::Result<()> {
        Ok(())
    }

    async fn next(&mut self) -> crate::Result<Option<SourceEvent>> {
        Ok(self.0.pop_front())
    }

    async fn close(&mut self) -> crate::Result<()> {
        Ok(())
    }
}

struct CollectingSink(Arc<Mutex<Vec<Batch>>>);

#[async_trait::async_trait]
impl StreamSink for CollectingSink {
    async fn open(&mut self) -> crate::Result<()> {
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> crate::Result<()> {
        self.0.lock().unwrap().push(batch.clone());
        Ok(())
    }

    async fn close(&mut self) -> crate::Result<()> {
        Ok(())
    }
}

fn data_event(rows: &[Row], offset: usize) -> SourceEvent {
    SourceEvent::Data {
        batch: input_batch(rows),
        cursor: Cursor::unbound(offset.to_be_bytes().to_vec(), JsonMap::new()).unwrap(),
    }
}

/// Runs the same plan through a checkpoint-off `StreamingRunner` (the oracle).
fn runner_outputs(
    plan: StreamExecutionPlan,
    events: Vec<SourceEvent>,
    policy: WatermarkPolicy,
) -> BTreeMap<String, Vec<Batch>> {
    let source_id = plan.source_binding_ids()[0].to_owned();
    let collected = plan
        .sink_binding_ids()
        .into_iter()
        .map(|output| (output.to_owned(), Arc::<Mutex<Vec<Batch>>>::default()))
        .collect::<BTreeMap<_, _>>();
    let sinks = collected
        .iter()
        .enumerate()
        .map(|(index, (output, batches))| {
            let sink = CollectingSink(Arc::clone(batches));
            let binding = SinkBinding::ordinary(&format!("sink_{index}"), sink).unwrap();
            (output.clone(), vec![binding])
        })
        .collect();
    let source = SourceBinding::new(ScriptedSource(events.into())).with_watermark_policy(policy);
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let outcome = runtime.block_on(async {
        StreamingRunner::without_checkpoints(plan, BTreeMap::from([(source_id, source)]), sinks)
            .unwrap()
            .start()
            .await
            .unwrap()
            .wait()
            .await
    });
    assert_eq!(outcome.state, JobState::Completed, "{outcome:?}");
    collected
        .into_iter()
        .map(|(output, batches)| (output, std::mem::take(&mut *batches.lock().unwrap())))
        .collect()
}

/// Explicit-watermark pushes and the equivalent source-provided runner events.
fn explicit_pushes() -> Vec<(Vec<Row>, i64)> {
    vec![
        (vec![(10, "a", 0, 1.0), (10, "b", 1, 2.0)], 10),
        (vec![(11, "a", 2, 3.0)], 11),
        (vec![(12, "b", 3, 4.0), (13, "a", 4, 5.0)], 12),
    ]
}

fn explicit_events(pushes: &[(Vec<Row>, i64)]) -> Vec<SourceEvent> {
    pushes
        .iter()
        .enumerate()
        .flat_map(|(offset, (rows, watermark))| {
            [
                data_event(rows, offset),
                SourceEvent::Watermark(micros(*watermark)),
            ]
        })
        .collect()
}

#[test]
fn test_rolling_projection_with_explicit_watermarks_matches_runner() {
    let pushes = explicit_pushes();
    let mut session = open_session(rolling_projection_plan(), BTreeMap::new());
    let input = session.input_ids()[0].to_owned();
    let output = session.output_ids()[0].to_owned();
    let mut rows = Vec::new();
    let mut emitted_per_push = Vec::new();
    for (batch_rows, watermark) in &pushes {
        let emitted = session
            .push(&input, input_batch(batch_rows), Some(micros(*watermark)))
            .unwrap();
        let emitted = rendered_rows(emitted.batches(&output));
        emitted_per_push.push(emitted.len());
        rows.extend(emitted);
    }
    let finished = session.finish().unwrap();
    rows.extend(rendered_rows(finished.batches(&output)));

    let expected = runner_outputs(
        rolling_projection_plan(),
        explicit_events(&pushes),
        WatermarkPolicy::SourceProvided,
    );
    assert_eq!(emitted_per_push, [2, 1, 1]);
    assert_eq!(rows.len(), 5);
    assert_eq!(rows, rendered_rows(&expected[&output]));
}

#[test]
fn durable_session_checkpoint_restores_operator_and_input_progress() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let mut config = session_config();
    config.checkpointing = true;
    let mut session = StreamSession::new_with_checkpoints(
        rolling_projection_plan(),
        &config,
        BTreeMap::new(),
        ManagedCheckpointRuntime::new(&root).unwrap(),
    )
    .unwrap();
    assert!(
        session
            .nodes
            .iter()
            .all(crate::runtime::streaming::operator_task::caller_driven::CallerDrivenOperator::checkpointing)
    );
    let input = session.input_ids()[0].to_owned();
    let output = session.output_ids()[0].to_owned();
    session
        .push(
            &input,
            input_batch(&[(10, "a", 0, 1.0), (10, "b", 1, 2.0)]),
            Some(micros(10)),
        )
        .unwrap();
    assert_eq!(session.checkpoint().unwrap(), Epoch::new(1).unwrap());
    drop(session);

    let mut restored = StreamSession::restore(
        rolling_projection_plan(),
        &config,
        BTreeMap::new(),
        ManagedCheckpointRuntime::new(&root).unwrap(),
    )
    .unwrap();
    let emitted = restored
        .push(&input, input_batch(&[(11, "a", 2, 3.0)]), Some(micros(11)))
        .unwrap();
    assert_eq!(
        rendered_rows(emitted.batches(&output)),
        vec!["1970-01-01T00:00:00.000011Z|a|2.0".to_owned()]
    );
    assert_eq!(restored.checkpoint().unwrap(), Epoch::new(2).unwrap());
}

fn timestamp_schema(unit: TimeUnit) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Timestamp(unit, Some("UTC".into())), false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, false),
    ]))
}

fn timestamp_batch(unit: TimeUnit, times: &[i64]) -> Batch {
    let timestamps: ArrayRef = match unit {
        TimeUnit::Second => {
            Arc::new(TimestampSecondArray::from(times.to_vec()).with_timezone("UTC"))
        }
        TimeUnit::Millisecond => {
            Arc::new(TimestampMillisecondArray::from(times.to_vec()).with_timezone("UTC"))
        }
        TimeUnit::Microsecond => {
            Arc::new(TimestampMicrosecondArray::from(times.to_vec()).with_timezone("UTC"))
        }
        TimeUnit::Nanosecond => {
            Arc::new(TimestampNanosecondArray::from(times.to_vec()).with_timezone("UTC"))
        }
    };
    let columns: Vec<ArrayRef> = vec![
        timestamps,
        Arc::new(StringArray::from_iter_values(times.iter().map(|_| "a"))),
        Arc::new(UInt64Array::from_iter_values(
            times.iter().enumerate().map(|(index, _)| index as u64),
        )),
        Arc::new(Float64Array::from_iter_values(times.iter().map(|_| 1.0))),
    ];
    Batch::table(
        vec![RecordBatch::try_new(timestamp_schema(unit), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn timestamp_session_plan(unit: TimeUnit) -> StreamExecutionPlan {
    pass_through_plan(&timestamp_schema(unit))
}

fn timestamp_cases() -> [(TimeUnit, ArrowTimestampUnit); 4] {
    [
        (TimeUnit::Second, ArrowTimestampUnit::Second),
        (TimeUnit::Millisecond, ArrowTimestampUnit::Millisecond),
        (TimeUnit::Microsecond, ArrowTimestampUnit::Microsecond),
        (TimeUnit::Nanosecond, ArrowTimestampUnit::Nanosecond),
    ]
}

fn expected_nanos(value: i64, unit: ArrowTimestampUnit) -> i128 {
    let factor = match unit {
        ArrowTimestampUnit::Second => 1_000_000_000,
        ArrowTimestampUnit::Millisecond => 1_000_000,
        ArrowTimestampUnit::Microsecond => 1_000,
        ArrowTimestampUnit::Nanosecond => 1,
    };
    i128::from(value) * factor
}

fn assert_generated_max(session: &StreamSession, value: i64, unit: ArrowTimestampUnit) {
    let maximum = session
        .input
        .progress
        .generator
        .as_ref()
        .and_then(GeneratedWatermarkState::max_observed_nanos);
    assert_eq!(maximum, Some(expected_nanos(value, unit)));
}

fn assert_restored_call_matches(
    control: &mut StreamSession,
    restored: &mut StreamSession,
    input: &str,
    output: &str,
    batch: Batch,
) {
    let expected = control.push(input, batch.clone(), None).unwrap();
    let actual = restored.push(input, batch, None).unwrap();
    assert_same_batches(actual.batches(output), expected.batches(output));
    assert_eq!(restored.watermark_for_test(), control.watermark_for_test());
}

fn restore_session(
    plan: TimeUnit,
    config: &StreamRuntimeConfig,
    watermarks: BTreeMap<String, SessionWatermark>,
    root: &std::path::Path,
) -> StreamSession {
    StreamSession::restore(
        timestamp_session_plan(plan),
        config,
        watermarks,
        ManagedCheckpointRuntime::new(root).unwrap(),
    )
    .unwrap()
}

fn ordered_restore_case(unit: TimeUnit, arrow_unit: ArrowTimestampUnit, raw: i64) {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let mut config = session_config();
    config.checkpointing = true;
    let plan = timestamp_session_plan(unit);
    let input = source_id(&plan);
    let output = plan.sink_binding_ids()[0].to_owned();
    let watermarks = ordered(&input);
    let mut control = open_session(plan, watermarks.clone());
    let mut durable = StreamSession::new_with_checkpoints(
        timestamp_session_plan(unit),
        &config,
        watermarks.clone(),
        ManagedCheckpointRuntime::new(&root).unwrap(),
    )
    .unwrap();
    assert_restored_call_matches(
        &mut control,
        &mut durable,
        &input,
        &output,
        timestamp_batch(unit, &[raw]),
    );
    assert_eq!(durable.checkpoint().unwrap(), Epoch::new(1).unwrap());
    drop(durable);

    let mut restored = restore_session(unit, &config, watermarks.clone(), &root);
    let assert_restored_progress = |session: &StreamSession| {
        assert_eq!(session.input.progress.max_event_time, Some(raw));
        let generator = session.input.progress.generator.as_ref().unwrap();
        let durable_nanos = expected_nanos(raw, arrow_unit).div_euclid(1_000) * 1_000;
        assert_eq!(generator.max_observed_nanos(), Some(durable_nanos));
    };
    assert_restored_progress(&restored);
    assert_restored_call_matches(
        &mut control,
        &mut restored,
        &input,
        &output,
        timestamp_batch(unit, &[]),
    );
    assert_restored_progress(&restored);
    assert_eq!(restored.checkpoint().unwrap(), Epoch::new(2).unwrap());
    drop(restored);

    let mut restored = restore_session(unit, &config, watermarks, &root);
    assert_restored_call_matches(
        &mut control,
        &mut restored,
        &input,
        &output,
        timestamp_batch(unit, &[raw + 1]),
    );
    assert_generated_max(&restored, raw + 1, arrow_unit);
    assert_generated_max(&control, raw + 1, arrow_unit);
    assert_same_batches(
        restored.finish().unwrap().batches(&output),
        control.finish().unwrap().batches(&output),
    );
}

#[test]
fn durable_ordered_restore_preserves_timestamp_units() {
    for (unit, arrow_unit) in timestamp_cases() {
        for raw in [5_000, -5_000] {
            ordered_restore_case(unit, arrow_unit, raw);
        }
    }
    ordered_restore_case(TimeUnit::Nanosecond, ArrowTimestampUnit::Nanosecond, -101);
}

fn bounded_restore_case(unit: TimeUnit, arrow_unit: ArrowTimestampUnit, raw: i64) {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let mut config = session_config();
    config.checkpointing = true;
    let plan = timestamp_session_plan(unit);
    let input = source_id(&plan);
    let watermarks = BTreeMap::from([(
        input.clone(),
        SessionWatermark::BoundedOutOfOrderness {
            event_time_column: "ts".into(),
            max_out_of_orderness: std::time::Duration::from_nanos(1_500),
        },
    )]);
    let mut session = StreamSession::new_with_checkpoints(
        plan,
        &config,
        watermarks.clone(),
        ManagedCheckpointRuntime::new(&root).unwrap(),
    )
    .unwrap();
    session
        .push(&input, timestamp_batch(unit, &[raw]), None)
        .unwrap();
    assert_eq!(session.checkpoint().unwrap(), Epoch::new(1).unwrap());
    drop(session);

    let mut restored = restore_session(unit, &config, watermarks.clone(), &root);
    assert_generated_max(&restored, raw, arrow_unit);
    restored
        .push(&input, timestamp_batch(unit, &[]), None)
        .unwrap();
    assert_eq!(restored.checkpoint().unwrap(), Epoch::new(2).unwrap());
    drop(restored);
    let restored = restore_session(unit, &config, watermarks, &root);
    assert_generated_max(&restored, raw, arrow_unit);
}

#[test]
fn durable_bounded_restore_preserves_observed_maximum() {
    for (unit, arrow_unit) in timestamp_cases() {
        for raw in [5_000, -5_000] {
            bounded_restore_case(unit, arrow_unit, raw);
        }
    }
}

fn ordered(input: &str) -> BTreeMap<String, SessionWatermark> {
    BTreeMap::from([(
        input.to_owned(),
        SessionWatermark::Ordered {
            event_time_column: "ts".into(),
        },
    )])
}

fn source_id(plan: &StreamExecutionPlan) -> String {
    plan.source_binding_ids()[0].to_owned()
}

fn ordered_session() -> (StreamSession, String, String) {
    let plan = rolling_projection_plan();
    let input = source_id(&plan);
    let session = open_session(plan, ordered(&input));
    let output = session.output_ids()[0].to_owned();
    (session, input, output)
}

#[test]
fn test_ordered_watermark_keeps_latest_timestamp_open_until_later_push() {
    let (mut session, input, output) = ordered_session();
    let first = session
        .push(
            &input,
            input_batch(&[(10, "a", 0, 1.0), (10, "b", 1, 2.0)]),
            None,
        )
        .unwrap();
    assert!(first.is_empty());
    let second = session
        .push(&input, input_batch(&[(11, "a", 2, 3.0)]), None)
        .unwrap();
    assert_eq!(rendered_rows(second.batches(&output)).len(), 2);
    let finished = session.finish().unwrap();
    assert_eq!(
        rendered_rows(finished.batches(&output)),
        ["1970-01-01T00:00:00.000011Z|a|2.0"]
    );

    let (mut session, input, output) = ordered_session();
    assert!(
        session
            .push(&input, input_batch(&[(10, "a", 0, 1.0)]), None)
            .unwrap()
            .is_empty()
    );
    let advanced = session.advance(&input, micros(10)).unwrap();
    assert_eq!(
        rendered_rows(advanced.batches(&output)),
        ["1970-01-01T00:00:00.000010Z|a|1.0"]
    );
}

#[test]
fn test_ordered_rejects_row_at_explicit_watermark_and_stays_usable() {
    let (mut session, input, output) = ordered_session();
    let closed = session
        .push(&input, input_batch(&[(10, "a", 0, 1.0)]), Some(micros(10)))
        .unwrap();
    assert_eq!(rendered_rows(closed.batches(&output)).len(), 1);
    let error = session
        .push(&input, input_batch(&[(10, "a", 1, 2.0)]), None)
        .unwrap_err();
    assert!(
        matches!(&error, CalcFlowError::InvalidArgument { field, .. }
            if field == &format!("session.inputs.{input}.event_time")),
        "{error:?}"
    );
    assert!(
        session
            .push(&input, input_batch(&[(11, "a", 2, 3.0)]), None)
            .unwrap()
            .is_empty()
    );
    let finished = session.finish().unwrap();
    assert_eq!(
        rendered_rows(finished.batches(&output)),
        ["1970-01-01T00:00:00.000011Z|a|2.0"]
    );
}

fn nanosecond_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            true,
        ),
        Field::new("value", DataType::Float64, false),
    ]))
}

fn nanosecond_batch(times: &[Option<i64>]) -> Batch {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(TimestampNanosecondArray::from(times.to_vec()).with_timezone("UTC")),
        Arc::new(Float64Array::from_iter_values(times.iter().map(|_| 1.0))),
    ];
    Batch::table(
        vec![RecordBatch::try_new(nanosecond_schema(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

/// A pass-through exact projection over `schema`'s columns.
fn pass_through_plan(schema: &SchemaRef) -> StreamExecutionPlan {
    let columns = schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    let operator = ExpressionOperator::new("pass", "", columns, None, vec![])
        .unwrap()
        .with_ports(
            Port::with_schema_ref("input", BatchKind::Table, true, Some(Arc::clone(schema)))
                .unwrap(),
            Port::with_schema_ref("output", BatchKind::Table, true, Some(Arc::clone(schema)))
                .unwrap(),
        )
        .unwrap();
    compile(
        PipelineBuilder::new("pass")
            .unwrap()
            .add_node("pass", Box::new(operator))
            .unwrap(),
    )
}

fn expression_plan(
    schema: &SchemaRef,
    select: Vec<String>,
    filter: Option<&str>,
) -> StreamExecutionPlan {
    let operator =
        ExpressionOperator::new("expression", "", select, filter.map(str::to_owned), vec![])
            .unwrap()
            .with_ports(
                Port::with_schema_ref("input", BatchKind::Table, true, Some(Arc::clone(schema)))
                    .unwrap(),
                Port::new("output", BatchKind::Table, true, None).unwrap(),
            )
            .unwrap();
    compile(
        PipelineBuilder::new("expression")
            .unwrap()
            .add_node("expression", Box::new(operator))
            .unwrap(),
    )
}

#[test]
fn test_sync_projection_reuses_its_ingress_snapshot_between_watermarks() {
    let plan = pass_through_plan(&input_schema());
    let input = source_id(&plan);
    let mut session = open_session(plan, BTreeMap::new());
    assert!(session.advance(&input, micros(10)).unwrap().is_empty());
    let identity = session.nodes[0].ingress_snapshot_identity();
    let measured = allocation_counter::measure(|| {
        assert!(session.advance(&input, micros(20)).unwrap().is_empty());
    });
    eprintln!("session watermark allocations: {measured:?}");
    assert_eq!(session.nodes[0].ingress_snapshot_identity(), identity);
    let output = session.output_ids()[0].to_owned();
    let emitted = session
        .push(&input, input_batch(&[(21, "k", 1, 7.0)]), None)
        .unwrap();
    assert_eq!(
        emitted
            .batches(&output)
            .iter()
            .map(Batch::num_rows)
            .sum::<usize>(),
        1
    );
    assert!(session.finish().unwrap().is_empty());
}

#[test]
fn test_staging_an_explicit_watermark_requires_no_allocation() {
    let plan = pass_through_plan(&input_schema());
    let input = source_id(&plan);
    let mut session = open_session(plan, BTreeMap::new());
    let mut staged = None;
    let measured = allocation_counter::measure(|| {
        staged = Some(session.input.stage_advance(&input, micros(10)).unwrap());
    });
    assert_eq!(measured.count_total, 0);
    assert!(session.run(staged.unwrap()).unwrap().is_empty());
    assert_eq!(session.watermark_for_test(), Some(micros(10)));
}

#[test]
fn test_collect_one_session_output_without_heap_allocation() {
    let (sender, receiver) = super::session_edge("result".into(), EdgeBudget::default()).unwrap();
    let mut outputs = [(
        "result".into(),
        super::OperatorEdgeReceiver::Session(receiver),
    )];
    sender.enqueue(super::StreamMessage::data(input_batch(&[(1, "a", 1, 9.0)])));
    sender.enqueue(super::StreamMessage::watermark(micros(1)));
    let mut emitted = None;
    let measured = allocation_counter::measure(|| {
        emitted = Some(super::collect_outputs(&mut outputs).unwrap());
    });
    let emitted = emitted.unwrap();
    assert_eq!(emitted.batches("result")[0].num_rows(), 1);
    assert!(emitted.batches("other").is_empty());
    assert_eq!(measured.count_total, 0, "{measured:?}");
    let mut owned = BTreeMap::new();
    let conversion = allocation_counter::measure(|| owned = emitted.into_batches());
    assert!(conversion.count_total <= 3, "{conversion:?}");
    assert_eq!(owned["result"].len(), 1);
    assert!(super::collect_outputs(&mut outputs).unwrap().is_empty());
}

#[test]
fn test_all_sync_expression_session_never_enters_async_drive() {
    let plan = pass_through_plan(&input_schema());
    let input = source_id(&plan);
    let mut session = open_session(plan, BTreeMap::new());
    let output = session.output_ids()[0].to_owned();
    assert!(session.runtime.is_none());
    super::reset_drive_counters_for_test();

    let pushed = session
        .push(&input, input_batch(&[(10, "a", 0, 1.0)]), Some(micros(10)))
        .unwrap();
    assert_eq!(pushed.batches(&output)[0].num_rows(), 1);
    assert!(session.advance(&input, micros(11)).unwrap().is_empty());
    assert!(session.finish().unwrap().is_empty());
    assert_eq!(super::drive_counters_for_test(), (0, 0));
}

#[test]
fn test_session_source_reuses_metadata_storage_for_large_and_empty_attributes() {
    let plan = pass_through_plan(&input_schema());
    let input = source_id(&plan);
    let mut session = open_session(plan, BTreeMap::new());
    let output = session.output_ids()[0].to_owned();
    let previous = BatchMetadata::new(
        "caller",
        99,
        BTreeMap::from([("large".into(), json!("x".repeat(64 * 1024)))]),
    )
    .unwrap();
    let mut stamped = Vec::new();
    for (sequence, original) in [previous, BatchMetadata::default()].into_iter().enumerate() {
        let batch = input_batch(&[(10, "a", 0, 1.0)]).with_metadata(original.clone());
        let result = session.push(&input, batch.clone(), None).unwrap();
        let metadata = result.batches(&output)[0].metadata().clone();
        assert_eq!(metadata.source(), input);
        assert_eq!(metadata.sequence(), sequence as u64);
        assert_eq!(batch.metadata(), &original);
        assert!(std::ptr::eq(metadata.attributes(), original.attributes()));
        let modified = metadata.clone().with_datafusion_active_entities(1).unwrap();
        assert_ne!(modified.attributes(), original.attributes());
        assert_eq!(metadata.attributes(), original.attributes());
        stamped.push(metadata);
    }
    assert!(std::ptr::eq(stamped[0].source(), stamped[1].source()));
}

fn assert_same_batches(actual: &[Batch], expected: &[Batch]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.metadata(), expected.metadata());
        assert_eq!(
            actual.table_payload().unwrap().schema(),
            expected.table_payload().unwrap().schema()
        );
        assert_eq!(
            actual.table_payload().unwrap().batches(),
            expected.table_payload().unwrap().batches()
        );
    }
}

#[test]
fn test_sync_expression_projection_and_filter_match_runner() {
    let rows = [(10, "a", 0, 1.0), (10, "b", 1, 3.0)];
    for (select, filter) in [
        (vec!["symbol".into(), "price".into()], None),
        (vec!["price + 1 AS next".into(), "symbol".into()], None),
        (vec!["symbol".into(), "price".into()], Some("price > 1")),
    ] {
        let mut session = open_session(
            expression_plan(&input_schema(), select.clone(), filter),
            BTreeMap::new(),
        );
        let input = session.input_ids()[0].to_owned();
        let output = session.output_ids()[0].to_owned();
        let actual = session
            .push(&input, input_batch(&rows), Some(micros(10)))
            .unwrap();
        assert_same_batches(
            actual.batches(&output),
            &runner_outputs(
                expression_plan(&input_schema(), select, filter),
                vec![data_event(&rows, 0), SourceEvent::Watermark(micros(10))],
                WatermarkPolicy::SourceProvided,
            )[&output],
        );
        assert!(session.finish().unwrap().is_empty());
    }
}

fn udf_plan() -> StreamExecutionPlan {
    let reference =
        crate::UdfReference::new("test", "trusted", "1", crate::UdfKind::DataFusionScalar).unwrap();
    let udf = Arc::new(datafusion::logical_expr::create_udf(
        "trusted",
        vec![DataType::Float64],
        DataType::Float64,
        datafusion::logical_expr::Volatility::Immutable,
        Arc::new(|arguments| Ok(arguments[0].clone())),
    ));
    let mut registry = UdfRegistry::new();
    registry
        .register_datafusion(reference.clone(), udf, 1)
        .unwrap();
    let operator = ExpressionOperator::new(
        "udf",
        "",
        vec![r#"trusted("price") AS trusted_price"#.into()],
        None,
        vec![reference],
    )
    .unwrap()
    .with_ports(
        Port::with_schema_ref("input", BatchKind::Table, true, Some(input_schema())).unwrap(),
        Port::new("output", BatchKind::Table, true, None).unwrap(),
    )
    .unwrap();
    PipelineBuilder::new("udf")
        .unwrap()
        .add_node("udf", Box::new(operator))
        .unwrap()
        .compile_stream(&registry.snapshot(), &StreamRequirements::default())
        .unwrap()
}

#[test]
fn test_udf_expression_uses_async_adapter_in_session_and_runner() {
    let rows = [(10, "a", 0, 1.0), (10, "b", 1, 3.0)];
    let plan = udf_plan();
    let input = source_id(&plan);
    let output = plan.sink_binding_ids()[0].to_owned();
    let mut session = open_session(plan, BTreeMap::new());
    assert!(session.runtime.is_some());
    let actual = session
        .push(&input, input_batch(&rows), Some(micros(10)))
        .unwrap();
    let expected = runner_outputs(
        udf_plan(),
        vec![data_event(&rows, 0), SourceEvent::Watermark(micros(10))],
        WatermarkPolicy::SourceProvided,
    );
    assert_same_batches(actual.batches(&output), &expected[&output]);
    assert!(session.finish().unwrap().is_empty());
}

#[test]
fn test_bounded_out_of_orderness_nanoseconds_matches_generated_watermark_state() {
    let plan = pass_through_plan(&nanosecond_schema());
    let input = source_id(&plan);
    let delay = std::time::Duration::from_nanos(1_500);
    let mut session = open_session(
        plan,
        BTreeMap::from([(
            input.clone(),
            SessionWatermark::BoundedOutOfOrderness {
                event_time_column: "ts".into(),
                max_out_of_orderness: delay,
            },
        )]),
    );
    let mut oracle = GeneratedWatermarkState::new(
        ResolvedEventTimeColumn {
            name: "ts".into(),
            index: 0,
            unit: ArrowTimestampUnit::Nanosecond,
        },
        delay,
    );
    let mut expected = None;
    for times in [
        vec![Some(5_000), Some(2_000)],
        vec![Some(9_999), None],
        vec![Some(3_000)],
        vec![],
        vec![Some(1_000_000_123)],
    ] {
        let batch = nanosecond_batch(&times);
        session.push(&input, batch.clone(), None).unwrap();
        oracle.observe_batch(&batch, &input).unwrap();
        if let Some(watermark) = oracle.on_timer(&input).unwrap() {
            expected = Some(watermark);
        }
        assert_eq!(session.watermark_for_test(), expected, "{times:?}");
    }
    // (1_000_000_123 ns - 1_500 ns).div_euclid(1_000)
    assert_eq!(expected, Some(micros(999_998)));
}

fn assert_invalid(result: crate::Result<super::SessionOutput>, expected_field: &str) {
    match result {
        Err(CalcFlowError::InvalidArgument { field, .. }) => assert_eq!(field, expected_field),
        other => panic!("expected InvalidArgument at {expected_field}, got {other:?}"),
    }
}

#[test]
fn test_validation_errors_leave_session_usable() {
    let plan = rolling_projection_plan();
    let input = source_id(&plan);
    let config = StreamRuntimeConfig {
        edge_budget: EdgeBudget {
            max_rows: 2,
            max_bytes: usize::MAX,
        },
        ..session_config()
    };
    let mut session = StreamSession::new(plan, &config, BTreeMap::new()).unwrap();
    let output = session.output_ids()[0].to_owned();
    let watermark_field = format!("session.inputs.{input}.watermark");
    let mut rows = rendered_rows(
        session
            .push(&input, input_batch(&[(10, "a", 0, 1.0)]), Some(micros(10)))
            .unwrap()
            .batches(&output),
    );
    assert!(matches!(
        session.push(&input, nanosecond_batch(&[Some(1)]), None),
        Err(CalcFlowError::Compile { .. })
    ));
    assert!(matches!(
        session.push(
            &input,
            input_batch(&[(11, "a", 1, 2.0), (11, "b", 2, 2.0), (11, "c", 3, 2.0)]),
            None,
        ),
        Err(CalcFlowError::EdgeBudgetExceeded { .. })
    ));
    assert_invalid(
        session.push(&input, input_batch(&[(11, "a", 1, 2.0)]), Some(micros(9))),
        &watermark_field,
    );
    assert_invalid(session.advance(&input, micros(9)), &watermark_field);
    assert_invalid(
        session.push("missing", input_batch(&[(11, "a", 1, 2.0)]), None),
        "session.inputs.missing",
    );
    assert_invalid(
        session.advance("missing", micros(11)),
        "session.inputs.missing",
    );
    assert!(
        session
            .push(&input, input_batch(&[]), None)
            .unwrap()
            .is_empty()
    );
    assert!(
        session
            .push(&input, input_batch(&[(11, "a", 1, 2.0)]), Some(micros(10)))
            .unwrap()
            .is_empty()
    );
    rows.extend(rendered_rows(
        session
            .push(&input, input_batch(&[(12, "a", 2, 3.0)]), Some(micros(12)))
            .unwrap()
            .batches(&output),
    ));
    assert_eq!(
        rows,
        [
            "1970-01-01T00:00:00.000010Z|a|1.0",
            "1970-01-01T00:00:00.000011Z|a|1.5",
            "1970-01-01T00:00:00.000012Z|a|2.5",
        ]
    );
}

#[test]
fn test_ordered_validation_rejects_decreasing_time_and_stays_usable() {
    let (mut session, input, output) = ordered_session();
    let time_field = format!("session.inputs.{input}.event_time");
    session
        .push(&input, input_batch(&[(10, "a", 0, 1.0)]), None)
        .unwrap();
    assert_invalid(
        session.push(
            &input,
            input_batch(&[(12, "a", 1, 2.0), (11, "a", 2, 3.0)]),
            None,
        ),
        &time_field,
    );
    assert_invalid(
        session.push(&input, input_batch(&[(9, "a", 1, 2.0)]), None),
        &time_field,
    );
    session
        .push(&input, input_batch(&[(10, "a", 1, 2.0)]), None)
        .unwrap();
    assert_eq!(
        rendered_rows(session.finish().unwrap().batches(&output)),
        [
            "1970-01-01T00:00:00.000010Z|a|1.0",
            "1970-01-01T00:00:00.000010Z|a|1.5",
        ]
    );
}

#[test]
fn test_disabled_watermarks_reject_explicit_watermarks_and_stay_usable() {
    let plan = rolling_projection_plan();
    let input = source_id(&plan);
    let watermark_field = format!("session.inputs.{input}.watermark");
    let mut session = open_session(
        plan,
        BTreeMap::from([(input.clone(), SessionWatermark::Disabled)]),
    );
    let output = session.output_ids()[0].to_owned();
    assert_invalid(
        session.push(&input, input_batch(&[(10, "a", 0, 1.0)]), Some(micros(10))),
        &watermark_field,
    );
    assert_invalid(session.advance(&input, micros(10)), &watermark_field);
    assert!(
        session
            .push(&input, input_batch(&[(10, "a", 0, 1.0)]), None)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        rendered_rows(session.finish().unwrap().batches(&output)),
        ["1970-01-01T00:00:00.000010Z|a|1.0"]
    );
}

fn assert_lifecycle_error(result: crate::Result<super::SessionOutput>, expected: &str) {
    match result {
        Err(CalcFlowError::InvalidArgument { field, message }) => {
            assert_eq!((field.as_str(), message.as_str()), ("session", expected));
        }
        other => panic!("expected session lifecycle error {expected:?}, got {other:?}"),
    }
}

fn assert_rejects_every_call(session: &mut StreamSession, input: &str, expected: &str) {
    assert_lifecycle_error(
        session.push(input, input_batch(&[(20, "a", 9, 1.0)]), Some(micros(20))),
        expected,
    );
    assert_lifecycle_error(session.advance(input, micros(21)), expected);
    assert_lifecycle_error(session.finish(), expected);
}

const FAILED: &str = "failed earlier; create a new session";
const FINISHED: &str = "already finished";

#[test]
fn test_operator_failure_fails_session_for_later_calls() {
    let mut session = open_session(rolling_projection_plan(), BTreeMap::new());
    let input = session.input_ids()[0].to_owned();
    session
        .push(&input, input_batch(&[(10, "a", 0, 1.0)]), Some(micros(10)))
        .unwrap();
    let error = session
        .push(&input, input_batch(&[(5, "a", 1, 2.0)]), None)
        .unwrap_err();
    assert!(
        !matches!(error, CalcFlowError::InvalidArgument { .. }),
        "{error:?}"
    );
    assert!(error.to_string().contains("late_row"), "{error}");
    assert!(session.is_failed());
    assert_rejects_every_call(&mut session, &input, FAILED);
}

#[test]
fn test_validation_error_does_not_mark_session_failed() {
    let mut session = open_session(rolling_projection_plan(), BTreeMap::new());
    let input = session.input_ids()[0].to_owned();
    assert!(!session.is_failed());
    assert!(session.push("missing", input_batch(&[]), None).is_err());
    assert!(!session.is_failed());
    session.finish().unwrap();
    assert!(!session.is_failed());
    assert_lifecycle_error(session.advance(&input, micros(1)), FINISHED);
}

#[test]
fn test_internal_edge_budget_after_validation_fails_session() {
    let operator = ExpressionOperator::new(
        "widen",
        "wide = repeat(symbol, 65536)",
        vec![],
        None,
        vec![],
    )
    .unwrap()
    .with_ports(
        Port::with_schema_ref("input", BatchKind::Table, true, Some(input_schema())).unwrap(),
        Port::with_schema_ref("output", BatchKind::Table, true, None).unwrap(),
    )
    .unwrap();
    let plan = compile(
        PipelineBuilder::new("widen")
            .unwrap()
            .add_node("widen", Box::new(operator))
            .unwrap(),
    );
    let input = source_id(&plan);
    let config = StreamRuntimeConfig {
        edge_budget: EdgeBudget {
            max_rows: 16,
            max_bytes: 16_384,
        },
        ..session_config()
    };
    let mut session = StreamSession::new(plan, &config, BTreeMap::new()).unwrap();
    let error = session
        .push(&input, input_batch(&[(10, "a", 0, 1.0)]), None)
        .unwrap_err();
    assert!(
        matches!(&error, CalcFlowError::EdgeBudgetExceeded { edge, .. } if edge.starts_with("sink/")),
        "{error:?}"
    );
    assert_rejects_every_call(&mut session, &input, FAILED);
}

#[test]
fn test_finish_without_pushes_then_rejects_later_calls() {
    let mut session = open_session(rolling_projection_plan(), BTreeMap::new());
    let input = session.input_ids()[0].to_owned();
    assert!(session.finish().unwrap().is_empty());
    assert_rejects_every_call(&mut session, &input, FINISHED);
}

#[test]
fn test_calls_inside_tokio_runtime_return_documented_error() {
    let mut session = open_session(rolling_projection_plan(), BTreeMap::new());
    let input = session.input_ids()[0].to_owned();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let expected = "cannot run inside a Tokio runtime; call it from a blocking thread";
    {
        let _context = runtime.enter();
        assert_lifecycle_error(
            session.push(&input, input_batch(&[(10, "a", 0, 1.0)]), Some(micros(10))),
            expected,
        );
        assert_lifecycle_error(session.advance(&input, micros(10)), expected);
        assert_lifecycle_error(session.finish(), expected);
        // Dropping a session inside a runtime context must not panic.
        drop(open_session(rolling_projection_plan(), BTreeMap::new()));
    }
    let inside = open_session(rolling_projection_plan(), BTreeMap::new());
    runtime.block_on(async move { drop(inside) });
    let output = session.output_ids()[0].to_owned();
    let pushed = session
        .push(&input, input_batch(&[(10, "a", 0, 1.0)]), Some(micros(10)))
        .unwrap();
    assert_eq!(rendered_rows(pushed.batches(&output)).len(), 1);
}

#[test]
fn test_operator_panic_is_contained_and_fails_session() {
    let mut session = open_session(rolling_projection_plan(), BTreeMap::new());
    let input = session.input_ids()[0].to_owned();
    session.panic_on_next_dispatch_for_test("rolling");
    let error = session
        .push(&input, input_batch(&[(10, "a", 0, 1.0)]), Some(micros(10)))
        .unwrap_err();
    assert!(
        matches!(&error, CalcFlowError::Operator { node_id, message }
            if node_id == "rolling" && message == "panicked: session dispatch test seam"),
        "{error:?}"
    );
    assert_rejects_every_call(&mut session, &input, FAILED);
}

#[test]
fn test_late_side_output_matches_runner_including_source_columns() {
    let plan = || {
        let side_output = json!({"kind": "side_output", "metrics_version": 1, "schema_version": 1});
        compile(
            PipelineBuilder::new("late")
                .unwrap()
                .add_node("rolling", rolling("rolling", &side_output))
                .unwrap(),
        )
    };
    let pushes = vec![
        (vec![(10, "a", 0, 1.0), (10, "b", 1, 2.0)], 10),
        (vec![(11, "a", 2, 3.0)], 11),
        (vec![(5, "a", 3, 9.0), (12, "b", 4, 4.0)], 12),
        (vec![(9, "b", 5, 8.0)], 12),
    ];
    let mut session = open_session(plan(), BTreeMap::new());
    let input = session.input_ids()[0].to_owned();
    let mut rows = BTreeMap::<String, Vec<String>>::new();
    for (batch_rows, watermark) in &pushes {
        let emitted = session
            .push(&input, input_batch(batch_rows), Some(micros(*watermark)))
            .unwrap();
        for (output, batches) in emitted.into_batches() {
            rows.entry(output)
                .or_default()
                .extend(rendered_rows(&batches));
        }
    }
    for (output, batches) in session.finish().unwrap().into_batches() {
        rows.entry(output)
            .or_default()
            .extend(rendered_rows(&batches));
    }
    let expected = runner_outputs(
        plan(),
        explicit_events(&pushes),
        WatermarkPolicy::SourceProvided,
    )
    .into_iter()
    .map(|(output, batches)| (output, rendered_rows(&batches)))
    .collect::<BTreeMap<_, _>>();
    assert_eq!(rows, expected);
    let late = &rows["late"];
    assert_eq!(late.len(), 2);
    assert!(late[0].contains(&format!("|{input}|2|")), "{late:?}");
    assert!(late[1].contains(&format!("|{input}|3|")), "{late:?}");
}

/// `input -> pass -> {fast, slow} -> {fast_out, slow_out}` plus `union(fast, slow)`.
fn fan_out_plan() -> StreamExecutionPlan {
    let rolling_with = |node_id: &str, size: usize| {
        let mut spec = serde_json::to_value(rolling_spec(&error_policy())).unwrap();
        spec["outputs"][0]["frame"]["size"] = json!(size);
        RollingOperator::new(
            node_id,
            input_schema(),
            serde_json::from_value(spec).unwrap(),
        )
        .unwrap()
    };
    let pass = ExpressionOperator::new(
        "pass",
        "",
        vec![
            "ts".into(),
            "symbol".into(),
            "sequence".into(),
            "price".into(),
        ],
        None,
        vec![],
    )
    .unwrap()
    .with_ports(
        Port::with_schema_ref("input", BatchKind::Table, true, Some(input_schema())).unwrap(),
        Port::with_schema_ref("output", BatchKind::Table, true, Some(input_schema())).unwrap(),
    )
    .unwrap();
    let fast = rolling_with("fast", 2);
    let rolled = fast.output_ports()[0].schema().unwrap().clone();
    let union_port = |name: &str| {
        Port::with_schema_ref(name, BatchKind::Table, true, Some(Arc::clone(&rolled))).unwrap()
    };
    let union =
        crate::UnionOperator::new("both", vec![union_port("left"), union_port("right")]).unwrap();
    let builder = PipelineBuilder::new("fan_out")
        .unwrap()
        .add_node("pass", Box::new(pass))
        .unwrap()
        .add_node("fast", fast)
        .unwrap()
        .add_node("slow", rolling_with("slow", 3))
        .unwrap()
        .add_node("fast_out", Box::new(projection("fast_out", &rolled)))
        .unwrap()
        .add_node("slow_out", Box::new(projection("slow_out", &rolled)))
        .unwrap()
        .add_node("both", Box::new(union))
        .unwrap();
    let builder = [
        (("pass", "output"), ("fast", "input")),
        (("pass", "output"), ("slow", "input")),
        (("fast", "output"), ("fast_out", "input")),
        (("slow", "output"), ("slow_out", "input")),
        (("fast", "output"), ("both", "left")),
        (("slow", "output"), ("both", "right")),
    ]
    .into_iter()
    .fold(builder, |builder, (source, target)| {
        connect(builder, source, target)
    });
    compile(builder)
}

#[test]
fn test_fan_out_outputs_and_union_match_runner() {
    let pushes = explicit_pushes();
    let mut session = open_session(fan_out_plan(), BTreeMap::new());
    let input = session.input_ids()[0].to_owned();
    assert_eq!(session.output_ids().len(), 3);
    let mut rows = BTreeMap::<String, Vec<String>>::new();
    let mut record = |output: super::SessionOutput| {
        assert!(output.iter_batches().map(|(name, _)| name).is_sorted());
        for (name, batches) in output.iter_batches() {
            assert!(std::ptr::eq(batches, output.batches(name)));
            rows.entry(name.to_owned())
                .or_default()
                .extend(rendered_rows(batches));
        }
    };
    for (batch_rows, watermark) in &pushes {
        record(
            session
                .push(&input, input_batch(batch_rows), Some(micros(*watermark)))
                .unwrap(),
        );
    }
    record(session.finish().unwrap());
    let mut expected = runner_outputs(
        fan_out_plan(),
        explicit_events(&pushes),
        WatermarkPolicy::SourceProvided,
    )
    .into_iter()
    .map(|(output, batches)| (output, rendered_rows(&batches)))
    .collect::<BTreeMap<_, _>>();
    // Union order across ingresses is a runner scheduling choice.
    for (_, union_rows) in rows
        .iter_mut()
        .chain(expected.iter_mut())
        .filter(|(output, _)| output.as_str() == "both.output")
    {
        union_rows.sort();
    }
    assert_eq!(
        rows.keys().collect::<Vec<_>>(),
        ["both.output", "fast_out.output", "slow_out.output"]
    );
    assert_eq!(rows["both.output"].len(), 10);
    assert_eq!(rows, expected);
}

#[test]
fn test_native_array_sequence_runs_on_the_calling_thread() {
    use crate::StreamOperatorFactory;

    let options = serde_json::from_value(json!({
        "ir_version": 1,
        "input_dtype": "float64",
        "kernel": {"function": "rolling_mean", "window": 2}
    }))
    .unwrap();
    let operator = crate::array::NativeArraySequenceFactory::default()
        .create(
            &crate::ExternalOperatorSpec::new("native", "array_sequence", "1", options).unwrap(),
            vec![Port::new("input", BatchKind::Array, true, None).unwrap()],
            vec![Port::new("output", BatchKind::Array, true, None).unwrap()],
        )
        .unwrap();
    let plan = PipelineBuilder::new("native-sequence")
        .unwrap()
        .add_node("sequence", operator)
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    let mut session = StreamSession::new(plan, &session_config(), BTreeMap::new()).unwrap();
    let (input, output) = (
        session.input_ids()[0].to_owned(),
        session.output_ids()[0].to_owned(),
    );
    for (value, expected) in [(1.0, 1.0), (3.0, 2.0)] {
        let array = crate::array::NativeArray::from_f64(vec![value], vec![1], None).unwrap();
        let emitted = session
            .push(
                &input,
                Batch::external(Arc::new(array), BatchMetadata::default()).unwrap(),
                None,
            )
            .unwrap();
        let batches = emitted.batches(&output);
        assert_eq!(batches.len(), 1);
        let payload = batches[0].external_payload().unwrap();
        let array = payload
            .as_any()
            .downcast_ref::<crate::array::NativeArray>()
            .unwrap();
        let values = array
            .as_arrow()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!((values.value(0) - expected).abs() < f64::EPSILON);
    }
    assert!(session.runtime.is_none());
    assert!(session.finish().unwrap().is_empty());
}

fn assert_rejected(
    plan: StreamExecutionPlan,
    watermarks: BTreeMap<String, SessionWatermark>,
    expected_field: &str,
) {
    match StreamSession::new(plan, &session_config(), watermarks) {
        Err(CalcFlowError::InvalidArgument { field, message }) => {
            assert_eq!(field, expected_field, "{message}");
        }
        Err(other) => panic!("expected InvalidArgument at {expected_field}, got {other:?}"),
        Ok(_) => panic!("expected InvalidArgument at {expected_field}, got a session"),
    }
}

fn table_port(name: &str) -> Port {
    Port::with_schema_ref(name, BatchKind::Table, true, Some(input_schema())).unwrap()
}

fn two_input_union() -> PipelineBuilder {
    PipelineBuilder::new("two")
        .unwrap()
        .add_node(
            "merge",
            Box::new(
                crate::UnionOperator::new("merge", vec![table_port("main"), table_port("other")])
                    .unwrap(),
            ),
        )
        .unwrap()
}

/// `input -> pass -> node`: one dynamic input fans out to `ingresses`, whose
/// ports declare `ingress_schema`.
fn behind_pass(
    name: &str,
    node: impl Into<crate::NodeOperator>,
    ingresses: &[&str],
    ingress_schema: Option<SchemaRef>,
) -> StreamExecutionPlan {
    let pass = ExpressionOperator::new("pass", "", vec!["*".into()], None, vec![])
        .unwrap()
        .with_ports(
            table_port("input"),
            Port::with_schema_ref("output", BatchKind::Table, true, ingress_schema).unwrap(),
        )
        .unwrap();
    let builder = PipelineBuilder::new(name)
        .unwrap()
        .add_node("pass", Box::new(pass))
        .unwrap()
        .add_node(name, node)
        .unwrap();
    compile(ingresses.iter().fold(builder, |builder, ingress| {
        connect(builder, ("pass", "output"), (name, ingress))
    }))
}

#[test]
fn test_admission_rejects_unsupported_inputs_with_field_paths() {
    assert_rejected(
        compile(two_input_union()),
        BTreeMap::new(),
        "session.inputs",
    );

    let with_static = compile(two_input_union())
        .with_static_input_specs(vec![crate::StaticInputSpec::Table {
            name: "other".into(),
            mutability: crate::StaticMutability::Static,
            schema: input_schema()
                .fields()
                .iter()
                .map(|field| crate::ArrowFieldSpec {
                    name: field.name().clone(),
                    data_type: match field.data_type() {
                        DataType::Timestamp(..) => "timestamp[us, tz=UTC]".into(),
                        DataType::Utf8 => "string".into(),
                        DataType::UInt64 => "uint64".into(),
                        _ => "float64".into(),
                    },
                    nullable: field.is_nullable(),
                })
                .collect(),
        }])
        .unwrap();
    assert_rejected(with_static, BTreeMap::new(), "session.static_inputs");
}

#[test]
fn test_admission_rejects_unsupported_nodes_with_field_paths() {
    let sql = crate::SqlOperator::new("sql", "SELECT * FROM input", vec!["input".into()], vec![])
        .unwrap();
    assert_rejected(
        behind_pass("sql", Box::new(sql), &["input"], None),
        BTreeMap::new(),
        "session.nodes.sql",
    );

    let join = crate::StreamJoinOperator::new(
        "join",
        input_schema(),
        input_schema(),
        crate::StreamJoinSpec::inner(
            ["symbol"],
            ["symbol"],
            "ts",
            "ts",
            crate::JoinTimeBounds::new(std::time::Duration::ZERO, std::time::Duration::ZERO)
                .unwrap(),
            crate::JoinStateLimits::new(100, 1_000_000, 1_000).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let join_ports = join
        .input_ports()
        .iter()
        .map(|port| port.name().to_owned())
        .collect::<Vec<_>>();
    let join_ports = join_ports.iter().map(String::as_str).collect::<Vec<_>>();
    assert_rejected(
        behind_pass("join", Box::new(join), &join_ports, Some(input_schema())),
        BTreeMap::new(),
        "session.nodes.join",
    );
}

#[test]
fn test_admission_rejects_unsupported_requirements_and_watermarks() {
    let output = rolling_projection_plan().sink_binding_ids()[0].to_owned();
    let rolling_builder = || {
        let rolling = rolling("rolling", &error_policy());
        let rolled = rolling.output_ports()[0].schema().unwrap().clone();
        let builder = PipelineBuilder::new("session")
            .unwrap()
            .add_node("rolling", rolling)
            .unwrap()
            .add_node("project", Box::new(projection("project", &rolled)))
            .unwrap();
        connect(builder, ("rolling", "output"), ("project", "input"))
    };
    let exactly_once = rolling_builder()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([(output.clone(), crate::DeliveryGuarantee::ExactlyOnce)]),
            },
        )
        .unwrap();
    assert_rejected(
        exactly_once,
        BTreeMap::new(),
        &format!("session.requirements.delivery.{output}"),
    );

    let input = source_id(&rolling_projection_plan());
    assert_rejected(
        rolling_projection_plan(),
        BTreeMap::from([("other".into(), SessionWatermark::Disabled)]),
        "session.watermarks.other",
    );
    assert_rejected(
        rolling_projection_plan(),
        BTreeMap::from([(
            input.clone(),
            SessionWatermark::Ordered {
                event_time_column: "price".into(),
            },
        )]),
        &format!("session.watermarks.{input}.event_time_column"),
    );
    assert_rejected(
        rolling_projection_plan(),
        BTreeMap::from([(
            input.clone(),
            SessionWatermark::BoundedOutOfOrderness {
                event_time_column: "ts".into(),
                max_out_of_orderness: std::time::Duration::ZERO,
            },
        )]),
        &format!("session.watermarks.{input}.max_out_of_orderness"),
    );
}

fn ordered_on(input: &str, column: &str) -> BTreeMap<String, SessionWatermark> {
    BTreeMap::from([(
        input.to_owned(),
        SessionWatermark::Ordered {
            event_time_column: column.into(),
        },
    )])
}

#[test]
fn test_ordered_push_reuses_the_validated_timestamp_maximum() {
    let records = [
        nanosecond_batch(&[Some(-2_100), Some(-1_900)]),
        nanosecond_batch(&[]),
        nanosecond_batch(&[Some(-1_900), Some(-101)]),
    ]
    .into_iter()
    .flat_map(|batch| batch.table_payload().unwrap().batches().to_vec())
    .collect();
    let batch = Batch::table(records, BatchMetadata::default()).unwrap();
    let mut scans = Vec::new();
    for mode in [
        SessionWatermark::Ordered {
            event_time_column: "ts".into(),
        },
        SessionWatermark::BoundedOutOfOrderness {
            event_time_column: "ts".into(),
            max_out_of_orderness: std::time::Duration::from_micros(1),
        },
    ] {
        let plan = pass_through_plan(&nanosecond_schema());
        let input = source_id(&plan);
        let output = plan.sink_binding_ids()[0].to_owned();
        let mut session = open_session(plan, BTreeMap::from([(input.clone(), mode)]));
        GeneratedWatermarkState::take_reduced_rows_for_test();
        let result = session.push(&input, batch.clone(), None).unwrap();
        scans.push(GeneratedWatermarkState::take_reduced_rows_for_test());
        assert_eq!(
            result
                .batches(&output)
                .iter()
                .map(Batch::num_rows)
                .sum::<usize>(),
            4
        );
        assert_generated_max(&session, -101, ArrowTimestampUnit::Nanosecond);
        assert_eq!(session.watermark_for_test(), Some(micros(-2)));
    }
    assert_eq!(
        scans,
        [0, 4],
        "generated reduction rows for ordered and bounded sessions"
    );
}

#[test]
fn test_ordered_rejects_nanosecond_row_inside_closed_microsecond() {
    let plan = pass_through_plan(&nanosecond_schema());
    let input = source_id(&plan);
    let mut session = open_session(plan, ordered_on(&input, "ts"));
    session
        .push(&input, nanosecond_batch(&[Some(5_000)]), Some(micros(5)))
        .unwrap();
    // 5 µs + 500 ns imports as event time 5 µs, which the watermark closed.
    assert_invalid(
        session.push(&input, nanosecond_batch(&[Some(5_500)]), None),
        &format!("session.inputs.{input}.event_time"),
    );
    session
        .push(&input, nanosecond_batch(&[Some(6_000)]), None)
        .unwrap();
}

fn second_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "ts",
        DataType::Timestamp(TimeUnit::Second, Some("UTC".into())),
        false,
    )]))
}

fn second_batch(times: &[i64]) -> Batch {
    let column: ArrayRef =
        Arc::new(TimestampSecondArray::from(times.to_vec()).with_timezone("UTC"));
    Batch::table(
        vec![RecordBatch::try_new(second_schema(), vec![column]).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

#[test]
fn test_generated_watermark_errors_use_session_watermark_paths() {
    let plan = pass_through_plan(&second_schema());
    let input = source_id(&plan);
    let mut session = open_session(
        plan,
        BTreeMap::from([(
            input.clone(),
            SessionWatermark::BoundedOutOfOrderness {
                event_time_column: "ts".into(),
                max_out_of_orderness: std::time::Duration::from_secs(1),
            },
        )]),
    );
    // Seconds far beyond the microsecond EventTime range.
    assert_invalid(
        session.push(&input, second_batch(&[i64::MAX / 2]), None),
        &format!("session.watermarks.{input}.event_time_column"),
    );
    session.push(&input, second_batch(&[10]), None).unwrap();
    assert_eq!(session.watermark_for_test(), Some(micros(9_000_000)));

    let plan = pass_through_plan(&second_schema());
    let input = source_id(&plan);
    let mut session = open_session(plan, ordered_on(&input, "ts"));
    assert_invalid(
        session.push(&input, second_batch(&[i64::MAX / 2]), None),
        &format!("session.watermarks.{input}.event_time_column"),
    );
    session.push(&input, second_batch(&[10]), None).unwrap();
}

#[test]
fn test_rejected_push_leaves_generated_watermark_state_unchanged() {
    let plan = rolling_projection_plan();
    let input = source_id(&plan);
    let config = StreamRuntimeConfig {
        edge_budget: EdgeBudget {
            max_rows: 2,
            max_bytes: usize::MAX,
        },
        ..session_config()
    };
    let watermarks = BTreeMap::from([(
        input.clone(),
        SessionWatermark::BoundedOutOfOrderness {
            event_time_column: "ts".into(),
            max_out_of_orderness: std::time::Duration::from_micros(1),
        },
    )]);
    let mut session = StreamSession::new(plan, &config, watermarks).unwrap();
    session
        .push(&input, input_batch(&[(10, "a", 0, 1.0)]), None)
        .unwrap();
    assert!(matches!(
        session.push(
            &input,
            input_batch(&[(20, "a", 1, 2.0), (20, "b", 2, 2.0), (20, "c", 3, 2.0)]),
            None,
        ),
        Err(CalcFlowError::EdgeBudgetExceeded { .. })
    ));
    session
        .push(&input, input_batch(&[(11, "a", 1, 2.0)]), None)
        .unwrap();
    assert_eq!(session.watermark_for_test(), Some(micros(10)));
}

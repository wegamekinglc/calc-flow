use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex},
};

use datafusion::arrow::{
    array::{
        ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, TimestampNanosecondArray,
        UInt64Array,
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
    Batch, BatchKind, BatchMetadata, CalcFlowError, Cursor, Edge, EdgeBudget, EventTime,
    ExpressionOperator, JobState, JsonMap, NativeWatermarkCapability, OperatorMetadata,
    PipelineBuilder, Port, PortEndpoint, ReplayPositioning, RollingOperator, RollingSpec,
    SinkBinding, SourceBinding, SourceCapabilities, SourceDeliveryCapability, SourceEvent,
    SourceSchema, StreamExecutionPlan, StreamRequirements, StreamRuntimeConfig, StreamSink,
    StreamSource, StreamingRunner, UdfRegistry, WatermarkPolicy,
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
        for (output, batches) in output.into_batches() {
            rows.entry(output)
                .or_default()
                .extend(rendered_rows(&batches));
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

    let indicator: RollingSpec = serde_json::from_value(json!({
        "configuration_version": 1, "state_layout_version": crate::ROLLING_INDICATOR_STATE_LAYOUT_VERSION,
        "partition_by": ["symbol"], "event_time": "ts", "sequence_by": ["sequence"],
        "outputs": [{"kind": "rsi", "primitive_version": 1, "input": "price", "output": "rsi", "period": 2}],
        "allowed_lateness_micros": 0, "late_policy": {"kind": "drop", "metrics_version": 1},
        "value_policy": "stateful_numeric_v1",
    }))
    .unwrap();
    let indicator = RollingOperator::new("rsi", input_schema(), indicator).unwrap();
    assert_rejected(
        compile(
            PipelineBuilder::new("rsi")
                .unwrap()
                .add_node("rsi", indicator)
                .unwrap(),
        ),
        BTreeMap::new(),
        "session.nodes.rsi",
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

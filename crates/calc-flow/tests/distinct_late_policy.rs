use calc_flow::{DistinctSpec, LatePolicySpec};
use serde_json::json;

fn window_wire() -> serde_json::Value {
    json!({"ingress_policy":"copy_visible_v2","history":{"kind":"closed_window",
        "event_time_column":"ts","start_micros":0,"end_micros":10},
        "state_budget":{"max_rows":2,"max_bytes":8_388_608}})
}

#[test]
fn test_local_policy_is_canonical_strict_and_persistent() {
    let old = window_wire();
    let mut drop = old.clone();
    drop["late_policy"] = json!({"kind":"drop","metrics_version":1});
    let spec: DistinctSpec = serde_json::from_value(drop.clone()).unwrap();
    assert_eq!(
        spec.late_policy(),
        LatePolicySpec::Drop { metrics_version: 1 }
    );
    assert!(
        DistinctSpec::bounded_source(calc_flow::StateBudget::new(2, 8_388_608).unwrap())
            .with_late_policy(spec.late_policy())
            .is_err()
    );
    assert_eq!(serde_json::to_value(spec).unwrap(), drop);
    let mut explicit = old.clone();
    explicit["late_policy"] = json!({"kind":"error","scope":"envelope"});
    let spec: DistinctSpec = serde_json::from_value(explicit).unwrap();
    assert_eq!(serde_json::to_value(spec).unwrap(), old);
    for invalid in [
        json!({"kind":"drop","metrics_version":2}),
        json!({"kind":"side_output","metrics_version":1,"schema_version":1}),
        json!({"kind":"error","scope":"row"}),
    ] {
        let mut wire = window_wire();
        wire["late_policy"] = invalid;
        assert!(serde_json::from_value::<DistinctSpec>(wire).is_err());
    }
}

#[test]
fn test_sql_window_local_drop_and_explicit_error() {
    let sql = |suffix: &str| {
        calc_flow::sql::distinct_query(
            &format!("SELECT DISTINCT q FROM CF_DISTINCT_WINDOW('a','ts',0,10,2,8388608{suffix})"),
            &["a".into()],
        )
        .unwrap()
        .unwrap()
        .spec
    };
    let drop = serde_json::to_value(sql(", 'drop'")).unwrap();
    assert_eq!(
        drop["late_policy"],
        json!({"kind":"drop","metrics_version":1})
    );
    assert_eq!(sql(""), sql(", 'error'"));
}

fn schema() -> std::sync::Arc<calc_flow::arrow::datatypes::Schema> {
    use calc_flow::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    std::sync::Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("q", DataType::Int64, true),
    ]))
}
fn data(rows: &[Vec<Option<i64>>]) -> calc_flow::Batch {
    use calc_flow::arrow::{
        array::{Int64Array, TimestampMicrosecondArray},
        record_batch::RecordBatch,
    };
    use std::sync::Arc;
    let records = rows
        .iter()
        .map(|r| {
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(TimestampMicrosecondArray::from(r.clone())),
                    Arc::new(Int64Array::from(vec![Some(1); r.len()])),
                ],
            )
            .unwrap()
        })
        .collect();
    calc_flow::Batch::table(records, calc_flow::BatchMetadata::default()).unwrap()
}
fn node(drop: bool) -> calc_flow::DistinctOperator {
    let mut wire = window_wire();
    if drop {
        wire["late_policy"] = json!({"kind":"drop","metrics_version":1});
    }
    calc_flow::DistinctOperator::new(
        "unique",
        schema(),
        vec!["q".into()],
        serde_json::from_value(wire).unwrap(),
    )
    .unwrap()
}
fn control(node: &mut calc_flow::DistinctOperator) -> serde_json::Value {
    use calc_flow::{Epoch, StreamOperator};
    serde_json::from_slice(node.checkpoint(Epoch::INITIAL).unwrap().segments["control"].bytes())
        .unwrap()
}

#[tokio::test]
async fn test_drop_at_full_history_budget_and_closed_capture_replay() {
    use calc_flow::{
        CancellationToken, EdgeCollector, Epoch, EventTime, OperatorMetadata, StreamJobContext,
        StreamOperator, StreamOperatorContext,
    };
    let job = StreamJobContext::new(
        1,
        "drop",
        calc_flow::JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let ctx = StreamOperatorContext::new(&job, "unique", None);
    let mut op = node(true);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("input", data(&[vec![Some(0), Some(9)]]), &ctx, &mut output)
        .await
        .unwrap();
    op.on_watermark(EventTime::from_micros(9), &ctx, &mut output)
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
    op.on_watermark(EventTime::from_micros(10), &ctx, &mut output)
        .await
        .unwrap();
    assert_eq!(output.drain("output").len(), 1);
    op.process_data(
        "input",
        data(&[vec![Some(0), Some(9)], vec![Some(10), None]]),
        &ctx,
        &mut output,
    )
    .await
    .unwrap();
    let cut = op.checkpoint(Epoch::INITIAL).unwrap();
    let c = control(&mut op);
    assert_eq!(c["state_layout"], 3);
    assert_eq!(c["accepted_rows"], 2);
    assert_eq!(
        c["late_metrics"],
        json!({"late_rows":2,"affected_batches":1,"max_lateness_micros":10,"null_event_time_rows":0,"null_event_time_batches":0})
    );
    for _ in 0..2 {
        let mut resumed = node(true);
        resumed.restore_async(&cut, &ctx).await.unwrap();
        resumed
            .on_watermark(EventTime::from_micros(12), &ctx, &mut output)
            .await
            .unwrap();
        resumed
            .process_data("input", data(&[vec![Some(1)]]), &ctx, &mut output)
            .await
            .unwrap();
        resumed.on_end(&ctx, &mut output).await.unwrap();
        assert!(output.drain("output").is_empty());
        let c = control(&mut resumed);
        assert_eq!(c["late_metrics"]["late_rows"], 3);
        assert_eq!(c["late_metrics"]["affected_batches"], 2);
        assert_eq!(c["late_metrics"]["max_lateness_micros"], 11);
    }
}

fn replace_control(
    mut snapshot: calc_flow::OperatorStateSnapshot,
    change: impl FnOnce(&mut serde_json::Value),
) -> calc_flow::OperatorStateSnapshot {
    let mut value: serde_json::Value =
        serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap();
    change(&mut value);
    let segment = calc_flow::StateSegment::new(serde_json::to_vec(&value).unwrap());
    snapshot
        .inline_metadata
        .insert("control_sha256".into(), json!(segment.sha256()));
    snapshot
        .inline_metadata
        .insert("state_layout".into(), value["state_layout"].clone());
    snapshot.segments.insert("control".into(), segment);
    snapshot
}

#[tokio::test]
async fn test_error_envelope_and_schema_or_cancel_failures_preserve_capture() {
    use calc_flow::{
        CancellationToken, EdgeCollector, Epoch, EventTime, OperatorMetadata, StreamJobContext,
        StreamOperator, StreamOperatorContext,
    };
    let token = CancellationToken::new();
    let job = StreamJobContext::new(1, "atomic", calc_flow::JsonMap::new(), None, token.clone());
    let context = StreamOperatorContext::new(&job, "unique", None);
    let mut op = node(false);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data(
        "input",
        data(&[vec![Some(0), Some(9)]]),
        &context,
        &mut output,
    )
    .await
    .unwrap();
    op.on_watermark(EventTime::from_micros(10), &context, &mut output)
        .await
        .unwrap();
    let before = op.checkpoint(Epoch::INITIAL).unwrap();
    let rejected = op
        .process_data(
            "input",
            data(&[vec![Some(0), None, Some(10)]]),
            &context,
            &mut output,
        )
        .await
        .unwrap_err();
    assert!(rejected.to_string().contains("late data"));
    assert_capture(&op.checkpoint(Epoch::INITIAL).unwrap(), &before);
    assert!(control(&mut op).get("late_metrics").is_none());
    let mut restored = node(false);
    restored.restore(&before).unwrap();
    assert_capture(&restored.checkpoint(Epoch::INITIAL).unwrap(), &before);
    token.cancel();
    assert!(
        op.process_data("input", data(&[vec![Some(0)]]), &context, &mut output)
            .await
            .is_err()
    );
    assert_capture(&op.checkpoint(Epoch::INITIAL).unwrap(), &before);
    assert_eq!(output.drain("output").len(), 1);
}

#[test]
fn test_policy_layout_and_metric_restore_mismatches_are_atomic() {
    use calc_flow::{Epoch, StreamOperator};
    let mut op = node(true);
    let before = op.checkpoint(Epoch::INITIAL).unwrap();
    let variants = [
        replace_control(before.clone(), |v| v["state_layout"] = json!(2)),
        replace_control(before.clone(), |v| {
            v.as_object_mut().unwrap().remove("late_metrics");
        }),
        replace_control(before.clone(), |v| {
            v["spec"]["late_policy"]["metrics_version"] = json!(2);
        }),
        replace_control(before.clone(), |v| {
            v["late_metrics"]["affected_batches"] = json!(1);
        }),
        replace_control(before.clone(), |v| {
            v["closed"] = json!(true);
            v["output_sequence"] = json!(1);
            v["late_metrics"] = json!({"late_rows":1,"affected_batches":1,"max_lateness_micros":10,"null_event_time_rows":0,"null_event_time_batches":0});
        }),
    ];
    for invalid in variants {
        assert!(op.restore(&invalid).is_err());
        assert_capture(&op.checkpoint(Epoch::INITIAL).unwrap(), &before);
    }
    assert!(node(false).restore(&before).is_err());
    assert!(
        op.restore(&node(false).checkpoint(Epoch::INITIAL).unwrap())
            .is_err()
    );
}

fn assert_capture(
    actual: &calc_flow::OperatorStateSnapshot,
    expected: &calc_flow::OperatorStateSnapshot,
) {
    assert_eq!(actual.inline_metadata, expected.inline_metadata);
    assert_eq!(actual.segments, expected.segments);
}

struct Source(std::collections::VecDeque<calc_flow::SourceEvent>);
#[async_trait::async_trait]
impl calc_flow::StreamSource for Source {
    fn capabilities(&self) -> calc_flow::SourceCapabilities {
        calc_flow::SourceCapabilities {
            replay_positioning: calc_flow::ReplayPositioning::Unsupported,
            delivery: calc_flow::SourceDeliveryCapability::Lossless,
            max_batch_rows: 4,
            max_batch_bytes: 4096,
            schema: calc_flow::SourceSchema::Exact(schema()),
            native_watermarks: calc_flow::NativeWatermarkCapability::EmitsNative,
        }
    }
    async fn open(&mut self, _: Option<calc_flow::Cursor>) -> calc_flow::Result<()> {
        Ok(())
    }
    async fn next(&mut self) -> calc_flow::Result<Option<calc_flow::SourceEvent>> {
        Ok(self.0.pop_front())
    }
    async fn close(&mut self) -> calc_flow::Result<()> {
        Ok(())
    }
}
#[derive(Default, Clone)]
struct Written(std::sync::Arc<std::sync::Mutex<Vec<calc_flow::Batch>>>);
#[async_trait::async_trait]
impl calc_flow::StreamSink for Written {
    async fn open(&mut self) -> calc_flow::Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &calc_flow::Batch) -> calc_flow::Result<()> {
        self.0.lock().unwrap().push(batch.clone());
        Ok(())
    }
    async fn close(&mut self) -> calc_flow::Result<()> {
        Ok(())
    }
}
fn script() -> Source {
    use calc_flow::{Cursor, EventTime, SourceEvent};
    let data_event = |position, batch| SourceEvent::Data {
        batch,
        cursor: Cursor::unbound(vec![position], calc_flow::JsonMap::new()).unwrap(),
    };
    Source(std::collections::VecDeque::from([
        data_event(1, data(&[vec![Some(0), Some(9)]])),
        SourceEvent::Watermark(EventTime::from_micros(9)),
        SourceEvent::Watermark(EventTime::from_micros(10)),
        data_event(2, data(&[vec![Some(0), Some(9)], vec![Some(10), None]])),
        SourceEvent::Watermark(EventTime::from_micros(12)),
        data_event(3, data(&[vec![Some(1)]])),
    ]))
}
#[tokio::test]
async fn test_public_rust_df_and_sql_drop_survive_lowering_and_stream_runtime() {
    use calc_flow::{
        CompileMode, CompileOptions, InputOptions, JobState, Program, SinkBinding, SourceBinding,
        StreamRequirements, StreamingRunner, UdfRegistry, table_input,
    };
    use std::collections::BTreeMap;
    let input = table_input("events", schema(), InputOptions::default()).unwrap();
    let drop: DistinctSpec = serde_json::from_value(window_wire()).unwrap();
    let df = input
        .distinct(
            vec!["q".into()],
            drop.with_late_policy(LatePolicySpec::Drop { metrics_version: 1 })
                .unwrap(),
        )
        .unwrap();
    let sql = calc_flow::sql::sql(
        "SELECT DISTINCT q FROM CF_DISTINCT_WINDOW('a','ts',0,10,2,8388608,'drop')",
        BTreeMap::from([("a".into(), input)]),
    )
    .unwrap();
    let program = Program::new(
        "drop",
        CompileMode::Stream,
        BTreeMap::from([("df".into(), df), ("sql".into(), sql)]),
    )
    .unwrap();
    let plan = program
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
            &CompileOptions::default(),
        )
        .await
        .unwrap();
    let written = Written::default();
    let sinks = ["df", "sql"]
        .into_iter()
        .map(|name| {
            (
                name.into(),
                vec![SinkBinding::ordinary(name, written.clone()).unwrap()],
            )
        })
        .collect();
    let job = StreamingRunner::without_checkpoints(
        plan,
        BTreeMap::from([("events".into(), SourceBinding::new(script()))]),
        sinks,
    )
    .unwrap()
    .start()
    .await
    .unwrap();
    assert_eq!(job.wait().await.state, JobState::Completed);
    assert_eq!(written.0.lock().unwrap().len(), 2);
    let project = program
        .to_project(&UdfRegistry::new().snapshot())
        .await
        .unwrap();
    let ids = project
        .graph
        .nodes
        .iter()
        .filter(|n| matches!(n.operator, calc_flow::OperatorSpec::Distinct { .. }))
        .map(|n| n.id.clone())
        .collect::<Vec<_>>();
    assert_eq!(ids.len(), 2);
    let status = job.status();
    for id in ids {
        let op = &status.operators[&id];
        assert_eq!(op.late_rows, 3);
        assert_eq!(op.late_affected_batches, 2);
        assert_eq!(op.max_lateness, Some(std::time::Duration::from_micros(11)));
    }
}

#[tokio::test]
async fn test_drop_schema_failure_does_not_change_closed_capture() {
    use calc_flow::arrow::{
        array::{Float64Array, TimestampMicrosecondArray},
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch,
    };
    use calc_flow::{
        Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, EventTime, OperatorMetadata,
        StreamJobContext, StreamOperator, StreamOperatorContext,
    };
    use std::sync::Arc;
    let mut op = node(true);
    let job = StreamJobContext::new(
        1,
        "schema",
        calc_flow::JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "unique", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.on_watermark(EventTime::from_micros(10), &context, &mut output)
        .await
        .unwrap();
    let before = op.checkpoint(Epoch::INITIAL).unwrap();
    let wrong = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("q", DataType::Float64, true),
    ]));
    let raw = Batch::table(
        vec![
            RecordBatch::try_new(
                wrong,
                vec![
                    Arc::new(TimestampMicrosecondArray::from(vec![0])),
                    Arc::new(Float64Array::from(vec![2.0])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    assert!(
        op.process_data("input", raw, &context, &mut output)
            .await
            .is_err()
    );
    assert_capture(&op.checkpoint(Epoch::INITIAL).unwrap(), &before);
    assert_eq!(control(&mut op)["late_metrics"]["late_rows"], 0);
    assert_eq!(output.drain("output").len(), 1);
}

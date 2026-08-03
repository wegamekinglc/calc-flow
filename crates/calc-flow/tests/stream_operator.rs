//! RED (M1.1): stream operator contract and the validating collector.
//!
//! Every test in this file fails to compile until the v3 stream surface
//! exists: `StreamOperator`, `StreamOperatorContext`, `StreamJobContext`,
//! `StreamCollector`, the runtime-owned validating collector,
//! `OperatorStateSnapshot`, and `UnionOperator` (plan task M1.1, API note
//! A1/A2). The expected RED reason is an unresolved import of these names
//! from `calc_flow`.

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchKind, BatchMetadata, BatchOperator, BatchOperatorContext, CalcFlowError,
    CancellationToken, EdgeCollector, Epoch, EventTime, ExpressionOperator, JsonMap,
    OperatorMetadata, OperatorStateSnapshot, Port, Result, SqlOperator, StreamCollector,
    StreamJobContext, StreamMessage, StreamMessageKind, StreamOperator, StreamOperatorContext,
    UnionOperator,
};
use chrono::DateTime;
use datafusion::arrow::{
    array::{Array, Int64Array},
    datatypes::{DataType, Field},
    record_batch::RecordBatch,
};
use serde_json::json;

fn table_batch(values: &[i64]) -> Batch {
    table_batch_named("value", values)
}

fn table_batch_named(column: &str, values: &[i64]) -> Batch {
    let record = RecordBatch::try_from_iter(vec![(
        column,
        Arc::new(Int64Array::from(values.to_vec())) as Arc<dyn Array>,
    )])
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn values(batch: &Batch, column: &str) -> Vec<i64> {
    batch.table_payload().unwrap().batches()[0]
        .column_by_name(column)
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "fingerprint",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

/// Records the ingress names and payload identity of every data batch.
struct RecordingOperator {
    input_ports: Vec<Port>,
    output_ports: Vec<Port>,
    events: Vec<String>,
}

impl RecordingOperator {
    fn new(inputs: &[&str]) -> Self {
        Self {
            input_ports: inputs
                .iter()
                .map(|name| Port::new(name, BatchKind::Table, true, None).unwrap())
                .collect(),
            output_ports: vec![Port::new("output", BatchKind::Table, true, None).unwrap()],
            events: Vec::new(),
        }
    }
}

impl OperatorMetadata for RecordingOperator {
    fn name(&self) -> &'static str {
        "recording"
    }

    fn input_ports(&self) -> &[Port] {
        &self.input_ports
    }

    fn output_ports(&self) -> &[Port] {
        &self.output_ports
    }

    fn configuration(&self) -> JsonMap {
        BTreeMap::new()
    }
}

#[async_trait]
impl StreamOperator for RecordingOperator {
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        _context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let first = values(&batch, "value")[0];
        self.events.push(format!("{ingress}:{first}"));
        output.emit("output", batch).await
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn stream_operator_receives_one_named_ingress_batch_per_call() {
    let mut operator = RecordingOperator::new(&["left", "right"]);
    let job = job();
    let context = StreamOperatorContext::new(&job, "merge", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

    let first = table_batch(&[1]);
    let second = table_batch(&[2]);
    let first_column = Arc::clone(first.table_payload().unwrap().batches()[0].column(0));
    operator
        .process_data("left", first, &context, &mut collector)
        .await
        .unwrap();
    operator
        .process_data("right", second, &context, &mut collector)
        .await
        .unwrap();

    assert_eq!(operator.events, ["left:1", "right:2"]);
    let drained = collector.drain("output");
    assert_eq!(drained.len(), 2);
    assert!(Arc::ptr_eq(
        drained[0]
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()[0]
            .column(0),
        &first_column
    ));
}

#[tokio::test]
async fn expression_stream_operator_projects_one_batch_per_input_batch() {
    let mut operator =
        ExpressionOperator::new("calc", "plus_one = value + 1", Vec::new(), None, Vec::new())
            .unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "calc", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

    operator
        .process_data("input", table_batch(&[1, 2]), &context, &mut collector)
        .await
        .unwrap();

    let drained = collector.drain("output");
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].kind(), StreamMessageKind::Data);
    assert_eq!(values(drained[0].as_data().unwrap(), "plus_one"), [2, 3]);
}

#[tokio::test]
async fn expression_stream_operator_rejects_an_unknown_ingress_without_emitting() {
    let mut operator =
        ExpressionOperator::new("calc", "plus_one = value + 1", Vec::new(), None, Vec::new())
            .unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "calc", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

    operator
        .process_data("wrong", table_batch(&[1]), &context, &mut collector)
        .await
        .unwrap_err();

    assert!(collector.drain("output").is_empty());
}

#[tokio::test]
async fn single_alias_sql_stream_operator_executes_per_batch() {
    let mut operator = SqlOperator::new(
        "project",
        "SELECT value * 2 AS doubled FROM events",
        vec!["events".into()],
        Vec::new(),
    )
    .unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "project", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

    operator
        .process_data("events", table_batch(&[2, 4]), &context, &mut collector)
        .await
        .unwrap();

    let drained = collector.drain("output");
    assert_eq!(drained.len(), 1);
    assert_eq!(values(drained[0].as_data().unwrap(), "doubled"), [4, 8]);
}

#[tokio::test]
async fn union_forwards_each_ingress_in_arrival_order_sharing_the_payload() {
    let mut operator = UnionOperator::new(
        "merge",
        vec![
            Port::new("left", BatchKind::Table, true, None).unwrap(),
            Port::new("right", BatchKind::Table, true, None).unwrap(),
        ],
    )
    .unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "merge", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

    let first = table_batch(&[1]);
    let second = table_batch(&[2]);
    let third = table_batch(&[3]);
    let first_column = Arc::clone(first.table_payload().unwrap().batches()[0].column(0));
    operator
        .process_data("left", first, &context, &mut collector)
        .await
        .unwrap();
    operator
        .process_data("right", second, &context, &mut collector)
        .await
        .unwrap();
    operator
        .process_data("left", third, &context, &mut collector)
        .await
        .unwrap();

    let drained = collector.drain("output");
    assert_eq!(drained.len(), 3);
    assert!(
        drained
            .iter()
            .all(|message| message.kind() == StreamMessageKind::Data)
    );
    assert_eq!(values(drained[0].as_data().unwrap(), "value"), [1]);
    assert_eq!(values(drained[1].as_data().unwrap(), "value"), [2]);
    assert_eq!(values(drained[2].as_data().unwrap(), "value"), [3]);
    // Fan-out clones share the immutable Arrow payload (spec S3).
    assert!(Arc::ptr_eq(
        drained[0]
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()[0]
            .column(0),
        &first_column
    ));
}

#[tokio::test]
async fn collector_rejects_an_unknown_output_port_before_enqueue() {
    let ports = vec![Port::new("output", BatchKind::Table, true, None).unwrap()];
    let mut collector = EdgeCollector::new(ports);

    let error = collector
        .emit("missing", table_batch(&[1]))
        .await
        .unwrap_err();

    assert!(matches!(error, CalcFlowError::Compile { .. }));
    assert!(error.to_string().contains("missing"));
    assert!(collector.drain("output").is_empty());
    assert!(collector.drain("missing").is_empty());
}

#[tokio::test]
async fn union_rejects_an_unknown_ingress_and_ignores_control_callbacks() {
    let mut operator = UnionOperator::new(
        "merge",
        vec![
            Port::new("left", BatchKind::Table, true, None).unwrap(),
            Port::new("right", BatchKind::Table, true, None).unwrap(),
        ],
    )
    .unwrap();
    assert_eq!(operator.name(), "merge");
    let debug = format!("{operator:?}");
    assert!(debug.contains("merge"));

    let job = job();
    let context = StreamOperatorContext::new(&job, "merge", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

    let error = operator
        .process_data("middle", table_batch(&[1]), &context, &mut collector)
        .await
        .unwrap_err();
    assert!(matches!(error, CalcFlowError::Operator { .. }));
    assert!(error.to_string().contains("middle"));
    assert!(collector.drain("output").is_empty());

    operator
        .on_watermark(EventTime::from_micros(3), &context, &mut collector)
        .await
        .unwrap();
    operator.on_end(&context, &mut collector).await.unwrap();
    assert!(collector.drain("output").is_empty());
}

#[test]
fn stream_job_context_exposes_accessors_and_reports_token_cancellation() {
    let cancellation = CancellationToken::new();
    // 2100-01-01T00:00:00Z: a fixed future deadline, no wall-clock dependence.
    let deadline = DateTime::from_timestamp_micros(4_102_444_800_000_000).unwrap();
    let job = StreamJobContext::new(
        42,
        "fingerprint",
        JsonMap::from([("key".into(), json!("value"))]),
        Some(deadline),
        cancellation.clone(),
    );

    assert_eq!(job.job_id(), 42);
    assert_eq!(job.fingerprint(), "fingerprint");
    assert_eq!(job.settings()["key"], json!("value"));
    assert_eq!(job.deadline(), Some(&deadline));
    assert!(!job.cancellation().is_cancelled());
    assert!(job.check_cancelled().is_ok());

    cancellation.cancel();
    assert!(matches!(
        job.check_cancelled(),
        Err(CalcFlowError::Cancelled { .. })
    ));
}

#[test]
fn stream_job_context_reports_an_elapsed_deadline_as_cancelled() {
    let job = StreamJobContext::new(
        1,
        "fingerprint",
        JsonMap::new(),
        Some(DateTime::from_timestamp_micros(0).unwrap()),
        CancellationToken::new(),
    );

    assert!(matches!(
        job.check_cancelled(),
        Err(CalcFlowError::Cancelled { .. })
    ));
}

#[tokio::test]
async fn collector_rejects_a_kind_mismatch_before_enqueue() {
    let ports = vec![Port::new("output", BatchKind::Array, true, None).unwrap()];
    let mut collector = EdgeCollector::new(ports);

    let error = collector
        .emit("output", table_batch(&[1]))
        .await
        .unwrap_err();

    assert!(matches!(error, CalcFlowError::Compile { .. }));
    assert!(collector.drain("output").is_empty());
}

#[tokio::test]
async fn collector_rejects_a_schema_mismatch_before_enqueue() {
    let fields = vec![Field::new("expected", DataType::Int64, false)];
    let ports = vec![Port::new("output", BatchKind::Table, true, Some(fields)).unwrap()];
    let mut collector = EdgeCollector::new(ports);

    // The batch carries column "value"; the port demands column "expected".
    let error = collector
        .emit("output", table_batch(&[1]))
        .await
        .unwrap_err();

    assert!(matches!(error, CalcFlowError::Compile { .. }));
    assert!(collector.drain("output").is_empty());

    let matching = table_batch_named("expected", &[1]);
    collector.emit("output", matching).await.unwrap();
    assert_eq!(collector.drain("output").len(), 1);
}

#[tokio::test]
async fn operators_cannot_emit_control_messages_through_the_collector() {
    // The collector is the only way an operator emits (API note A2), and it
    // accepts data batches only; `StreamMessage` control constructors are
    // crate-private, so no handler can forge a watermark, barrier, idle, or
    // end-of-input message (spec S1.3). Every message that can exist in the
    // outbox after any handler is therefore data.
    let mut operator = RecordingOperator::new(&["input"]);
    let job = job();
    let context = StreamOperatorContext::new(&job, "recording", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

    operator
        .process_data("input", table_batch(&[1]), &context, &mut collector)
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(10), &context, &mut collector)
        .await
        .unwrap();
    operator.on_end(&context, &mut collector).await.unwrap();

    let drained = collector.drain("output");
    assert_eq!(drained.len(), 1);
    assert!(
        drained
            .iter()
            .all(|message| message.kind() == StreamMessageKind::Data)
    );
}

struct MapAssertingOperator {
    input_ports: Vec<Port>,
    output_ports: Vec<Port>,
}

#[async_trait]
impl BatchOperator for MapAssertingOperator {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        _context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        assert_eq!(inputs.len(), 2);
        assert_eq!(values(&inputs["left"], "value"), [1]);
        assert_eq!(values(&inputs["right"], "value"), [2]);
        Ok(BTreeMap::from([("output".into(), inputs["left"].clone())]))
    }
}

impl OperatorMetadata for MapAssertingOperator {
    fn name(&self) -> &'static str {
        "map_asserting"
    }

    fn input_ports(&self) -> &[Port] {
        &self.input_ports
    }

    fn output_ports(&self) -> &[Port] {
        &self.output_ports
    }

    fn configuration(&self) -> JsonMap {
        BTreeMap::new()
    }
}

#[tokio::test]
async fn batch_operator_receives_the_complete_input_map() {
    let mut operator = MapAssertingOperator {
        input_ports: vec![
            Port::new("left", BatchKind::Table, true, None).unwrap(),
            Port::new("right", BatchKind::Table, true, None).unwrap(),
        ],
        output_ports: vec![Port::new("output", BatchKind::Table, true, None).unwrap()],
    };
    let run = calc_flow::RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    let context = BatchOperatorContext { run: &run };
    let inputs = BTreeMap::from([
        ("left".into(), table_batch(&[1])),
        ("right".into(), table_batch(&[2])),
    ]);

    let outputs = operator.process(&inputs, &context).await.unwrap();

    assert_eq!(values(&outputs["output"], "value"), [1]);
}

#[tokio::test]
async fn stream_operator_state_defaults_are_empty_and_reject_non_empty_state() {
    let mut operator = RecordingOperator::new(&["input"]);

    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert!(snapshot.inline_metadata.is_empty());
    assert!(snapshot.segments.is_empty());
    operator.restore(&OperatorStateSnapshot::default()).unwrap();
    operator.reset().unwrap();

    let mut metadata = JsonMap::new();
    metadata.insert("open_windows".into(), serde_json::json!(3));
    let non_empty = OperatorStateSnapshot {
        inline_metadata: metadata,
        segments: BTreeMap::new(),
    };
    let error = operator.restore(&non_empty).unwrap_err();
    assert!(matches!(error, CalcFlowError::Format { .. }));
}

#[tokio::test]
async fn stream_context_exposes_the_watermark_and_records_late_rows() {
    let job = job();
    let watermark = EventTime::from_micros(42);
    let context = StreamOperatorContext::new(&job, "window", Some(watermark));

    assert_eq!(context.job().fingerprint(), "fingerprint");
    assert_eq!(context.operator_id(), "window");
    assert_eq!(context.input_watermark(), Some(watermark));
    context.check_cancelled().unwrap();
    context
        .record_late_rows(2, Some(Duration::from_micros(5)))
        .unwrap();

    let unprimed = StreamOperatorContext::new(&job, "window", None);
    assert_eq!(unprimed.input_watermark(), None);
}

#[test]
fn stream_message_data_constructs_the_only_public_variant() {
    // `StreamMessage::data` is the sole public constructor; control variants
    // are crate-private (plan M1.2). This test exists so a future change that
    // publishes a control constructor fails review against a concrete anchor.
    let message = StreamMessage::data(table_batch(&[1]));
    assert_eq!(message.kind(), StreamMessageKind::Data);
}

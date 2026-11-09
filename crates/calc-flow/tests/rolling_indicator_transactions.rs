//! Indicator budgets and checkpoint preparation cover lifecycle boundaries.

use std::{collections::BTreeMap, sync::Arc};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, CancellationToken, EdgeCollector, Epoch, EventTime,
    ExecutionOptions, JsonMap, OperatorMetadata, PipelineBuilder, RollingOperator, RollingSpec,
    RunResult, StateBudget, StreamCollector, StreamJobContext, StreamOperator,
    StreamOperatorContext, UdfRegistry,
};
use datafusion::arrow::{
    array::{Array, ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::{Value, json};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("entity", DataType::Utf8, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::UInt64, false),
        Field::new("price", DataType::Float64, false),
    ]))
}

fn input() -> Batch {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec!["a"; 4])),
        Arc::new(TimestampMicrosecondArray::from(vec![1, 2, 3, 100]).with_timezone("UTC")),
        Arc::new(UInt64Array::from(vec![1, 2, 3, 4])),
        Arc::new(Float64Array::from(vec![1.0, 2.0, 1.0, 3.0])),
    ];
    Batch::table(
        vec![RecordBatch::try_new(schema(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn operator(max_rows: u64, with_window: bool) -> RollingOperator {
    operator_with_late_policy(
        max_rows,
        with_window,
        &json!({"kind":"error","scope":"envelope"}),
    )
}

fn operator_with_late_policy(
    max_rows: u64,
    with_window: bool,
    late_policy: &Value,
) -> RollingOperator {
    let mut outputs = vec![json!({
        "kind":"rsi","primitive_version":1,"input":"price","output":"rsi","period":2
    })];
    if with_window {
        outputs.push(json!({
            "kind":"mean","primitive_version":1,"input":"price","output":"mean",
            "frame":{"kind":"duration","micros":10},"min_periods":1
        }));
    }
    let spec: RollingSpec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":5,
        "partition_by":["entity"],"event_time":"ts","sequence_by":["seq"],
        "outputs":outputs,"allowed_lateness_micros":0,
        "late_policy":late_policy,
        "value_policy":"stateful_numeric_v1"
    }))
    .unwrap();
    let mut operator = RollingOperator::new("features", schema(), spec).unwrap();
    operator
        .set_state_budget(StateBudget::new(max_rows, 4 << 20).unwrap())
        .unwrap();
    operator
}

async fn execute(max_rows: u64, with_window: bool) -> calc_flow::Result<RunResult> {
    PipelineBuilder::new("indicator-prefix-budget")?
        .add_node("features", operator(max_rows, with_window))?
        .compile_batch(&UdfRegistry::new().snapshot())?
        .execute(
            BTreeMap::from([("input".into(), input())]),
            ExecutionOptions::default(),
        )
        .await
}

fn values(result: &RunResult, name: &str) -> Vec<Option<f64>> {
    batch_values(&result.outputs["output"], name)
}

fn batch_values(batch: &Batch, name: &str) -> Vec<Option<f64>> {
    batch.table_payload().unwrap().batches()[0]
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .iter()
        .collect()
}

#[tokio::test]
async fn test_finite_indicator_rejects_transient_retained_row_overflow_before_eviction() {
    let control = execute(32, true).await.unwrap();
    assert_eq!(
        values(&control, "mean"),
        [Some(1.0), Some(1.5), Some(4.0 / 3.0), Some(3.0)]
    );
    let error = execute(3, true).await.unwrap_err();
    assert!(error.to_string().contains("state_budget"), "{error}");
}

#[tokio::test]
async fn test_finite_indicator_output_rows_do_not_consume_retained_row_budget() {
    let result = execute(1, false).await.unwrap();
    assert_eq!(
        values(&result, "rsi"),
        [None, None, Some(50.0), Some(100.0 / 1.2)]
    );
}

async fn assert_empty_control_invalidates_preparation(ending: bool) {
    let mut operator = operator(32, false);
    let job = StreamJobContext::new(
        11,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator.prepare_checkpoint_async(&context).await.unwrap();
    operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    if ending {
        operator.on_end(&context, &mut output).await.unwrap();
    } else {
        operator
            .on_watermark(EventTime::from_micros(1), &context, &mut output)
            .await
            .unwrap();
    }
    assert!(output.drain("output").is_empty());
    let error = operator.checkpoint(Epoch::new(2).unwrap()).unwrap_err();
    assert!(error.to_string().contains("prepar"), "{error}");
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert_eq!(snapshot.inline_metadata["ended"], json!(ending));
}

#[tokio::test]
async fn test_zero_output_watermark_invalidates_prepared_indicator_revision() {
    assert_empty_control_invalidates_preparation(false).await;
}

#[tokio::test]
async fn test_empty_eof_invalidates_prepared_indicator_revision() {
    assert_empty_control_invalidates_preparation(true).await;
}

#[derive(Default)]
struct RejectingCollector {
    attempts: Vec<Batch>,
}

#[async_trait]
impl StreamCollector for RejectingCollector {
    async fn emit(&mut self, port: &str, batch: Batch) -> calc_flow::Result<()> {
        assert_eq!(port, "output");
        self.attempts.push(batch);
        Err(CalcFlowError::InvalidArgument {
            field: "test.collector".into(),
            message: "reject the first prepared emission".into(),
        })
    }
}

#[tokio::test]
async fn test_rejected_first_emission_keeps_pending_state_and_output_sequence_for_retry() {
    let mut operator = operator(32, false);
    let job = StreamJobContext::new(
        12,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", input(), &context, &mut output)
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
    let mut rejecting = RejectingCollector::default();
    let error = operator
        .on_watermark(EventTime::from_micros(101), &context, &mut rejecting)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("test.collector"), "{error}");
    assert_eq!(rejecting.attempts.len(), 1);
    assert_eq!(rejecting.attempts[0].metadata().sequence(), 0);
    operator
        .on_watermark(EventTime::from_micros(101), &context, &mut output)
        .await
        .unwrap();
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    let retry = messages[0].as_data().unwrap();
    assert_eq!(retry.metadata().sequence(), 0);
    assert_eq!(
        batch_values(retry, "rsi"),
        [None, None, Some(50.0), Some(100.0 / 1.2)]
    );
}

fn one_row(time: i64, sequence: u64, price: f64) -> Batch {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec!["a"])),
        Arc::new(TimestampMicrosecondArray::from(vec![time]).with_timezone("UTC")),
        Arc::new(UInt64Array::from(vec![sequence])),
        Arc::new(Float64Array::from(vec![price])),
    ];
    Batch::table(
        vec![RecordBatch::try_new(schema(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

#[tokio::test]
async fn test_late_drop_skips_indicator_domain_checks_and_preserves_recurrence() {
    let mut operator =
        operator_with_late_policy(32, false, &json!({"kind":"drop","metrics_version":1}));
    let job = StreamJobContext::new(
        13,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", input(), &context, &mut output)
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(101), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(output.drain("output").len(), 1);
    let context = StreamOperatorContext::new(&job, "features", Some(EventTime::from_micros(101)));
    operator
        .process_data(
            "input",
            one_row(50, 999, f64::INFINITY),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
    operator
        .process_data("input", one_row(102, 5, 2.0), &context, &mut output)
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(103), &context, &mut output)
        .await
        .unwrap();
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    assert_eq!(
        batch_values(messages[0].as_data().unwrap(), "rsi"),
        [Some(50.0)]
    );
}

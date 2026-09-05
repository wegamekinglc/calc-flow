//! Finality and recovery parity for the columnar ordered rolling path.

use std::sync::Arc;

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, EventTime, JsonMap,
    OperatorMetadata, RollingOperator, RollingSpec, StreamJobContext, StreamOperator,
    StreamOperatorContext,
};
use datafusion::arrow::{
    array::{ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

const FINGERPRINT: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
    ]))
}

fn operator(drop_late: bool) -> RollingOperator {
    let policy = if drop_late {
        serde_json::json!({"kind": "drop", "metrics_version": 1})
    } else {
        serde_json::json!({"kind": "error", "scope": "envelope"})
    };
    let spec: RollingSpec = serde_json::from_value(serde_json::json!({
        "configuration_version": 1, "state_layout_version": 1,
        "partition_by": ["symbol"], "event_time": "ts", "sequence_by": ["sequence"],
        "outputs": [{"kind": "mean", "primitive_version": 1, "input": "price",
                     "output": "mean", "frame": {"kind": "rows", "size": 2}, "min_periods": 1}],
        "allowed_lateness_micros": 0, "late_policy": policy, "value_policy": "stateful_numeric_v1"
    }))
    .unwrap();
    RollingOperator::new("rolling", schema(), spec).unwrap()
}

fn batch(times: &[i64]) -> Batch {
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(times.to_vec()).with_timezone("UTC"))
                as ArrayRef,
            Arc::new(StringArray::from(vec!["a"; times.len()])),
            Arc::new(UInt64Array::from(
                times
                    .iter()
                    .map(|&t| u64::try_from(t).unwrap())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                times
                    .iter()
                    .map(|&t| f64::from(i32::try_from(2 * t - 1).unwrap()))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

struct Stream {
    operator: RollingOperator,
    job: StreamJobContext,
    output: EdgeCollector,
    watermark: Option<EventTime>,
}

impl Stream {
    fn new(drop_late: bool) -> Self {
        let operator = operator(drop_late);
        let output = EdgeCollector::new(operator.output_ports().to_vec());
        Self {
            operator,
            output,
            watermark: None,
            job: StreamJobContext::new(
                1,
                FINGERPRINT,
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
        }
    }

    async fn push(&mut self, times: &[i64]) -> calc_flow::Result<()> {
        let context = StreamOperatorContext::new(&self.job, "rolling", self.watermark);
        self.operator
            .process_data("input", batch(times), &context, &mut self.output)
            .await
    }

    async fn advance(&mut self, time: i64) -> Vec<f64> {
        let watermark = EventTime::from_micros(time);
        let context = StreamOperatorContext::new(&self.job, "rolling", self.watermark);
        self.operator
            .on_watermark(watermark, &context, &mut self.output)
            .await
            .unwrap();
        self.watermark = Some(watermark);
        self.drain()
    }

    fn drain(&mut self) -> Vec<f64> {
        self.output
            .drain("output")
            .iter()
            .flat_map(|message| {
                message
                    .as_data()
                    .unwrap()
                    .table_payload()
                    .unwrap()
                    .batches()
                    .iter()
                    .flat_map(|record| {
                        record
                            .column(4)
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .unwrap()
                            .values()
                            .to_vec()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

#[tokio::test]
async fn partial_watermark_and_checkpoint_recovery_preserve_pending_rows() {
    let mut stream = Stream::new(false);
    stream.push(&[1, 2, 3]).await.unwrap();
    assert!(stream.drain().is_empty());
    assert_eq!(stream.advance(1).await, [1.0]);
    let snapshot = stream.operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let mut restored = Stream::new(false);
    restored.operator.restore(&snapshot).unwrap();
    restored.watermark = stream.watermark;
    restored.push(&[4, 5]).await.unwrap();
    assert_eq!(restored.advance(4).await, [2.0, 4.0, 6.0]);
    let context = StreamOperatorContext::new(&restored.job, "rolling", restored.watermark);
    restored
        .operator
        .on_end(&context, &mut restored.output)
        .await
        .unwrap();
    assert_eq!(restored.drain(), [8.0]);
}

#[tokio::test]
async fn out_of_order_arrivals_merge_with_the_ordered_pending_prefix() {
    let mut stream = Stream::new(false);
    stream.push(&[2, 4]).await.unwrap();
    stream.push(&[1, 3]).await.unwrap();
    assert_eq!(stream.advance(4).await, [1.0, 2.0, 4.0, 6.0]);
    stream.push(&[5, 6]).await.unwrap();
    assert_eq!(stream.advance(6).await, [8.0, 10.0]);
}

#[tokio::test]
async fn duplicate_envelope_cannot_install_its_nonduplicate_suffix() {
    let mut stream = Stream::new(false);
    stream.push(&[1, 2]).await.unwrap();
    assert!(
        stream
            .push(&[2, 3])
            .await
            .unwrap_err()
            .to_string()
            .contains("duplicate row identity")
    );
    assert_eq!(stream.advance(2).await, [1.0, 2.0]);
    stream.push(&[3]).await.unwrap();
    assert_eq!(stream.advance(3).await, [4.0]);
}

#[tokio::test]
async fn duplicate_late_rows_are_dropped_before_duplicate_validation() {
    let mut stream = Stream::new(true);
    stream.push(&[1, 2]).await.unwrap();
    assert_eq!(stream.advance(2).await, [1.0, 2.0]);
    stream.push(&[1, 1, 3]).await.unwrap();
    assert_eq!(stream.advance(3).await, [4.0]);
    let snapshot = stream.operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert_eq!(snapshot.inline_metadata["metrics"]["late_rows"], 2);
}

#[tokio::test]
async fn empty_envelopes_and_end_preserve_ordered_state() {
    let mut stream = Stream::new(false);
    stream.push(&[]).await.unwrap();
    stream.push(&[1, 2]).await.unwrap();
    stream.push(&[]).await.unwrap();
    let context = StreamOperatorContext::new(&stream.job, "rolling", None);
    stream
        .operator
        .on_end(&context, &mut stream.output)
        .await
        .unwrap();
    assert_eq!(stream.drain(), [1.0, 2.0]);
}

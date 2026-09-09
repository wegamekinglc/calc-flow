use super::*;
use crate::{BatchMetadata, CancellationToken, EdgeBudget, EdgeCollector, Epoch, StreamJobContext};
use datafusion::arrow::{
    array::{Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use std::{sync::Arc, time::Duration};

struct LateMetrics;
impl crate::operator::stream::LateMetricSink for LateMetrics {
    fn record(&self, _delta: crate::operator::stream::LateMetricDelta) -> Result<()> {
        Ok(())
    }
}
fn fixture() -> (StreamAsofJoinOperator, Batch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::Int64, false),
    ]));
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["seq".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::ZERO,
        AsofStateLimits::new(100, 1_048_576).unwrap(),
    )
    .unwrap();
    let operator =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let batch = Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(vec!["A"])),
                    Arc::new(TimestampMicrosecondArray::from(vec![100]).with_timezone("UTC")),
                    Arc::new(Int64Array::from(vec![1])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap();
    (operator, batch)
}

#[tokio::test]
async fn one_row_output_byte_limit_preserves_pending_state_and_releases_workspace() {
    let (mut op, batch) = fixture();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::for_task(
        &job,
        "asof",
        None,
        IngressProgressSnapshot::default(),
        EdgeBudget {
            max_bytes: 1,
            ..EdgeBudget::default()
        },
        Arc::new(LateMetrics),
    );
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", batch, &cx, &mut output)
        .await
        .unwrap();
    let before = op.capture(Epoch::INITIAL).unwrap();
    assert_eq!(op.runtime.pool.reserved(), 0);
    let error = op
        .on_watermark(EventTime::from_micros(101), &cx, &mut output)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofOutputLimitExceeded,
            ..
        }
    ));
    assert_eq!(op.status.pending_left_rows, 1);
    assert_eq!(op.status.output_limit_failures, 1);
    assert_eq!(
        op.capture(Epoch::INITIAL).unwrap().segments,
        before.segments
    );
    assert!(output.drain("output").is_empty());
    assert_eq!(op.runtime.pool.reserved(), 0);
    op.reset().unwrap();
    assert_eq!(op.runtime.pool.reserved(), 0);
}

#[tokio::test]
async fn restored_logical_counter_overflow_fails_before_admission_or_emit() {
    let (mut op, batch) = fixture();
    let mut snapshot = op.capture(Epoch::INITIAL).unwrap();
    let metrics = snapshot.inline_metadata.get_mut("metrics").unwrap();
    metrics["right"]["accepted_rows"] = u64::MAX.into();
    metrics["evicted_right_rows"] = u64::MAX.into();
    op.restore(&snapshot).unwrap();
    let before = op.status();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    let error = op
        .process_data("right", batch, &cx, &mut output)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofCounterOverflow,
            ..
        }
    ));
    assert_eq!(op.status(), before);
    assert!(output.drain("output").is_empty());
    assert_eq!(op.runtime.pool.reserved(), 0);
}

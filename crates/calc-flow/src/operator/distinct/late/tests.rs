use super::*;
use crate::{
    CancellationToken, DistinctSpec, EdgeCollector, EventTime, OperatorMetadata, StateBudget,
    StreamJobContext, StreamOperator,
};
use datafusion::arrow::{
    array::{Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
};
use datafusion::execution::memory_pool::MemoryPool;
use std::{cell::Cell, sync::Arc};
thread_local! {static ALLOCATIONS: Cell<usize> = const {Cell::new(0)};}
pub(super) fn before_metrics_allocation() {
    ALLOCATIONS.with(|n| n.set(n.get() + 1));
}
pub(in crate::operator::distinct) fn fixture(limit: u64) -> DistinctOperator {
    let schema = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("q", DataType::Int64, true),
    ]));
    let spec = DistinctSpec::closed_window(
        "ts",
        EventTime::from_micros(0),
        EventTime::from_micros(10),
        StateBudget::new(2, limit).unwrap(),
    )
    .unwrap()
    .with_late_policy(LatePolicySpec::Drop { metrics_version: 1 })
    .unwrap();
    DistinctOperator::new("quota", schema, vec!["q".into()], spec).unwrap()
}
fn data() -> Batch {
    let op = fixture(8 << 20);
    Batch::table(
        vec![
            RecordBatch::try_new(
                op.input_ports()[0].schema().unwrap().clone(),
                vec![
                    Arc::new(TimestampMicrosecondArray::from(vec![0, 9])),
                    Arc::new(Int64Array::from(vec![1, 1])),
                ],
            )
            .unwrap(),
        ],
        crate::BatchMetadata::default(),
    )
    .unwrap()
}
#[test]
fn test_drop_control_storage_inclusive_budget() {
    // Declaration=27016, stream control=4096+256; 256 is new Drop storage.
    for (limit, ok) in [(31_368, true), (31_367, false)] {
        let mut op = fixture(limit);
        assert_eq!(op.ensure_stream().is_ok(), ok);
        assert_eq!(op.stream.is_some(), ok);
    }
}
#[tokio::test]
async fn test_drop_metric_transaction_exact_512_and_one_byte_short() {
    for (limit, ok) in [(31_880, true), (31_879, false)] {
        let mut op = fixture(limit);
        op.ensure_stream().unwrap();
        let state = op.stream.as_mut().unwrap();
        state.closed = true;
        state.accepted = 2;
        state.watermark = Some(EventTime::from_micros(10));
        let job = StreamJobContext::new(
            1,
            "quota",
            crate::JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let ctx = StreamOperatorContext::new(&job, "quota", None);
        let mut output = EdgeCollector::new(op.output_ports().to_vec());
        ALLOCATIONS.with(|n| n.set(0));
        let result = op.process_data("input", data(), &ctx, &mut output).await;
        assert_eq!(result.is_ok(), ok);
        assert_eq!(ALLOCATIONS.with(Cell::get), usize::from(ok));
        assert_eq!(
            op.stream.as_ref().unwrap().metrics.late_rows,
            if ok { 2 } else { 0 }
        );
        assert!(output.drain("output").is_empty());
        assert_eq!(op.pool.reserved(), 31_368);
    }
}

#[derive(Default)]
struct Sink {
    totals: parking_lot::Mutex<LateMetricDelta>,
    cancel: Option<CancellationToken>,
    fail: bool,
}
impl crate::operator::stream::LateMetricSink for Sink {
    fn record(&self, _: LateMetricDelta) -> Result<()> {
        unreachable!("staged updates only")
    }
    fn prepare(self: Arc<Self>, delta: LateMetricDelta) -> Result<PreparedLateMetrics> {
        if self.fail {
            return Err(owned::error("metric preparation refused"));
        }
        let next = accumulate_late_metrics(*self.totals.lock(), delta)?;
        if let Some(token) = &self.cancel {
            token.cancel();
        }
        Ok(PreparedLateMetrics::new(move || *self.totals.lock() = next))
    }
    fn prepare_restore(self: Arc<Self>, totals: LateMetricDelta) -> Result<PreparedLateMetrics> {
        Ok(PreparedLateMetrics::new(move || {
            *self.totals.lock() = totals;
        }))
    }
}
fn mark_closed(op: &mut DistinctOperator) {
    op.ensure_stream().unwrap();
    let state = op.stream.as_mut().unwrap();
    state.closed = true;
    state.accepted = 2;
    state.watermark = Some(EventTime::from_micros(10));
}
#[tokio::test]
async fn test_late_sink_preparation_failure_and_cancel_do_not_commit() {
    for cancel in [false, true] {
        let mut op = fixture(8 << 20);
        mark_closed(&mut op);
        let capture = op.capture().unwrap();
        let token = CancellationToken::new();
        let job = StreamJobContext::new(1, "atomic", crate::JsonMap::new(), None, token.clone());
        let sink = Arc::new(Sink {
            cancel: cancel.then_some(token),
            fail: !cancel,
            ..Sink::default()
        });
        let context = StreamOperatorContext::for_task(
            &job,
            "quota",
            None,
            crate::IngressProgressSnapshot::default(),
            crate::EdgeBudget::default(),
            sink.clone(),
        );
        let mut output = EdgeCollector::new(op.output_ports().to_vec());
        let raw = data();
        let weak = Arc::downgrade(raw.table_payload().unwrap().batches()[0].column(0));
        assert!(
            op.process_data("input", raw, &context, &mut output)
                .await
                .is_err()
        );
        assert!(weak.upgrade().is_none());
        assert_eq!(op.capture().unwrap().segments, capture.segments);
        assert_eq!(sink.totals.lock().late_rows, 0);
        assert_eq!(
            op.stream.as_ref().unwrap().metrics,
            LateMetricDelta::default()
        );
    }
}
#[tokio::test]
async fn test_restore_installs_totals_once_and_counter_overflow_is_atomic() {
    let mut op = fixture(8 << 20);
    mark_closed(&mut op);
    op.stream.as_mut().unwrap().metrics = LateMetricDelta {
        late_rows: u64::MAX,
        affected_batches: 1,
        max_lateness_micros: Some(10),
        ..LateMetricDelta::default()
    };
    let before = op.capture().unwrap();
    let job = StreamJobContext::new(
        1,
        "metrics",
        crate::JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let sink = Arc::new(Sink::default());
    let context = StreamOperatorContext::for_task(
        &job,
        "quota",
        None,
        crate::IngressProgressSnapshot::default(),
        crate::EdgeBudget::default(),
        sink.clone(),
    );
    for _ in 0..2 {
        op.restore_with_context(&before, &context).unwrap();
    }
    assert_eq!(sink.totals.lock().late_rows, u64::MAX);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    assert!(
        op.process_data("input", data(), &context, &mut output)
            .await
            .is_err()
    );
    assert_eq!(op.capture().unwrap().segments, before.segments);
    assert_eq!(sink.totals.lock().late_rows, u64::MAX);
}

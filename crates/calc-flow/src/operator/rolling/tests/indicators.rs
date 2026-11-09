use super::*;

#[test]
fn test_indicator_controlled_history_copy_cancels_without_modifying_source_rows() {
    let rows: VecDeque<_> = (0..8193_u32)
        .map(|index| {
            vec![
                ScalarValue::Float64(Some(f64::from(index))),
                ScalarValue::Utf8(Some("immutable source".into())),
            ]
        })
        .collect();
    let mut calls = 0;
    let error = super::super::indicators::controlled::copy_rows(&rows, &mut || {
        calls += 1;
        if calls == 2 {
            Err(CalcFlowError::Cancelled {
                run_id: "copy".into(),
            })
        } else {
            Ok(())
        }
    })
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    assert_eq!(calls, 2);
    assert_eq!(rows.len(), 8193);
    assert_eq!(rows.front().unwrap()[0], ScalarValue::Float64(Some(0.0)));
    assert_eq!(rows.back().unwrap()[0], ScalarValue::Float64(Some(8192.0)));
}

struct RejectingIndicatorMetrics;

impl crate::operator::LateMetricSink for RejectingIndicatorMetrics {
    fn record(&self, _: LateMetricDelta) -> Result<()> {
        Err(internal_error("injected indicator metric failure"))
    }
}

#[tokio::test]
async fn test_indicator_metric_failure_preserves_unaccepted_rows_and_charge() {
    let mut spec = kernel_spec(json!([
        {"kind":"rsi", "primitive_version":1, "input":"price", "output":"rsi", "period":1}
    ]));
    spec.state_layout_version = 5;
    let mut operator = RollingOperator::new("features", Arc::new(kernel_schema()), spec).unwrap();
    let job = crate::StreamJobContext::new(
        1,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = StreamOperatorContext::for_task(
        &job,
        "features",
        None,
        crate::IngressProgressSnapshot::default(),
        crate::EdgeBudget::default(),
        Arc::new(RejectingIndicatorMetrics),
    );
    let input = Batch::table(
        vec![float64_fast_record(&[(1, "a", 1, Some(1.0))])],
        BatchMetadata::default(),
    )
    .unwrap();
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", input, &context, &mut output)
        .await
        .unwrap_err();
    assert!(
        operator.state.buffer.is_empty(),
        "failed metric admission committed rows"
    );
    assert_eq!(operator.state.charge, empty_indicator_charge(&operator));
    assert_eq!(operator.state.metrics, LateMetricDelta::default());
    assert!(operator.state.pipeline_fingerprint.is_none());
    assert!(output.drain("output").is_empty());
}

#[test]
fn test_indicator_checkpoint_uses_the_same_worker_pool_as_its_resources() {
    let mut spec = kernel_spec(json!([
        {"kind":"rsi", "primitive_version":1, "input":"price", "output":"rsi", "period":1}
    ]));
    spec.state_layout_version = 5;
    let operator = RollingOperator::new("features", Arc::new(kernel_schema()), spec).unwrap();
    let workers = operator.native_checkpoint_workers();
    assert!(Arc::ptr_eq(
        &workers,
        &operator.indicator_resources.as_ref().unwrap().workers()
    ));
    assert!(Arc::ptr_eq(&workers, &operator.native_checkpoint_workers()));
    let mut old = valid_spec();
    old.state_layout_version = 5;
    let old_only = RollingOperator::new("features", Arc::new(input_schema()), old).unwrap();
    assert!(old_only.indicator_resources.is_none());
    assert!(Arc::ptr_eq(
        &old_only.native_checkpoint_workers(),
        &old_only.native_checkpoint_workers()
    ));
}

fn empty_indicator_charge(operator: &RollingOperator) -> StateCharge {
    operator
        .indicator_resources
        .as_ref()
        .unwrap()
        .empty_state()
        .charge
}

async fn pending_indicator_fixture() -> (RollingOperator, crate::StreamJobContext) {
    let mut spec = kernel_spec(json!([
        {"kind":"rsi", "primitive_version":1, "input":"price", "output":"rsi", "period":1}
    ]));
    spec.state_layout_version = 5;
    let mut operator = RollingOperator::new("features", Arc::new(kernel_schema()), spec).unwrap();
    let job = crate::StreamJobContext::new(
        1,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "actual_node", None);
    let input = Batch::table(
        vec![float64_fast_record(&[(1, "a", 1, Some(1.0))])],
        BatchMetadata::default(),
    )
    .unwrap();
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", input, &context, &mut output)
        .await
        .unwrap();
    (operator, job)
}

#[tokio::test]
async fn test_pending_native_read_gate_rejects_input_until_settled() {
    let (mut operator, job) = pending_indicator_fixture().await;
    let guard = operator
        .native_checkpoint_guard("actual_node", &job)
        .await
        .unwrap();
    let context = StreamOperatorContext::new(&job, "actual_node", None);
    let input = Batch::table(
        vec![float64_fast_record(&[(2, "a", 2, Some(2.0))])],
        BatchMetadata::default(),
    )
    .unwrap();
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let error = operator
        .process_data("input", input, &context, &mut output)
        .await
        .expect_err("pending native read must settle before another input");
    let CalcFlowError::Operator { node_id, message } = error else {
        panic!("expected builtin pending error");
    };
    assert_eq!(node_id, "actual_node");
    assert!(message.contains("state.pending"));
    assert_eq!(operator.state.buffer.len(), 1);
    drop(guard);
    StreamOperator::reset(&mut operator).unwrap();
    assert!(operator.state.buffer.is_empty());
}

#[tokio::test]
async fn test_indicator_reset_clears_rows_without_erasing_pending_gate_or_capture_credit() {
    let (mut operator, job) = pending_indicator_fixture().await;
    let resources = operator.indicator_resources.as_ref().unwrap().clone();
    let capture = resources
        .retain(
            crate::StateSegment::new(vec![7; 1024]),
            operator.state.charge.bytes,
            operator.state_budget,
            "actual_node",
        )
        .unwrap();
    let guard = operator
        .native_checkpoint_guard("actual_node", &job)
        .await
        .unwrap();
    StreamOperator::reset(&mut operator)
        .expect("reset must invalidate pending work, not refuse reset");
    assert!(operator.state.buffer.is_empty());
    assert_eq!(operator.state.charge, empty_indicator_charge(&operator));
    assert!(operator.try_native_checkpoint_guard("actual_node").is_err());
    let baseline = empty_indicator_charge(&operator).bytes;
    let too_small = StateBudget::new(10, baseline.checked_add(1087).unwrap()).unwrap();
    assert!(operator.set_state_budget(too_small).is_err());
    let last_capture = capture.clone();
    drop(capture);
    drop(guard);
    assert!(operator.set_state_budget(too_small).is_err());
    drop(last_capture);
    operator
        .set_state_budget(StateBudget::new(10, baseline).unwrap())
        .unwrap();
}

#[tokio::test]
async fn test_indicator_state_packet_is_shared_without_copy_and_survives_reset() {
    let (mut operator, _) = pending_indicator_fixture().await;
    let source = operator.indicator_state_packet().unwrap();
    let mut shared = None;
    let measured = allocation_counter::measure(|| {
        shared = operator.indicator_state_packet();
    });
    let shared = shared.unwrap();
    assert!(Arc::ptr_eq(&source, &shared));
    assert_eq!(measured.bytes_total, 0);
    assert_eq!(source.buffer.len(), 1);
    StreamOperator::reset(&mut operator).unwrap();
    assert!(operator.state.buffer.is_empty());
    assert_eq!(source.buffer.len(), 1);
    assert!(!Arc::ptr_eq(
        &source,
        &operator.indicator_state_packet().unwrap()
    ));
}

#[tokio::test]
async fn test_tight_reset_reuses_the_funded_empty_packet_until_last_work_owner() {
    let (mut operator, job) = pending_indicator_fixture().await;
    StreamOperator::reset(&mut operator).unwrap();
    let empty = operator.indicator_state_packet().unwrap();
    let context = StreamOperatorContext::new(&job, "actual_node", None);
    let input = Batch::table(
        vec![float64_fast_record(&[(1, "a", 1, Some(1.0))])],
        BatchMetadata::default(),
    )
    .unwrap();
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", input, &context, &mut output)
        .await
        .unwrap();
    let bytes = operator
        .state
        .charge
        .bytes
        .checked_add(empty.charge.bytes)
        .unwrap();
    let limit = StateBudget::new(10, bytes).unwrap();
    operator.set_state_budget(limit).unwrap();
    let resources = operator.indicator_resources.as_ref().unwrap().clone();
    let pending = resources.hold_current_for_test();
    let old = Arc::downgrade(&operator.indicator_state_packet().unwrap());
    StreamOperator::reset(&mut operator).unwrap();
    let reset = operator.indicator_state_packet().unwrap();
    assert!(
        Arc::ptr_eq(&empty, &reset),
        "reset allocated a second empty packet"
    );
    assert!(reset.buffer.is_empty());
    assert!(old.upgrade().is_some(), "pending Work must still pin old S");
    resources.check(0, limit, "actual_node").unwrap();
    let fixed = Arc::downgrade(&empty);
    drop(empty);
    drop(reset);
    drop(operator);
    drop(resources);
    assert!(
        fixed.upgrade().is_some(),
        "last Work must keep fixed S alive"
    );
    drop(pending);
    assert!(old.upgrade().is_none());
    assert!(fixed.upgrade().is_none());
}

type BlockedIndicatorWorker = (
    std::sync::mpsc::Sender<()>,
    tokio::task::JoinHandle<Result<()>>,
);

async fn occupy_indicator_workers(
    workers: Arc<crate::array::RollingWorkers>,
    job: &crate::StreamJobContext,
) -> Vec<BlockedIndicatorWorker> {
    let mut active = Vec::new();
    for _ in 0..2 {
        let workers = Arc::clone(&workers);
        let scope = crate::array::RollingScope::stream(job);
        let (release, hold) = std::sync::mpsc::channel();
        let (started, ready) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            workers
                .run_rolling_scoped(scope, "features", move |_| {
                    started.send(()).unwrap();
                    hold.recv().unwrap();
                    Ok(())
                })
                .await
        });
        ready.await.unwrap();
        active.push((release, task));
    }
    active
}

#[tokio::test]
async fn test_indicator_input_preparation_waits_for_workers_and_abandoned_waiter_cannot_commit() {
    let mut spec = kernel_spec(json!([
        {"kind":"rsi", "primitive_version":1, "input":"price", "output":"rsi", "period":1}
    ]));
    spec.state_layout_version = 5;
    let mut operator = RollingOperator::new("features", Arc::new(kernel_schema()), spec).unwrap();
    let job = crate::StreamJobContext::new(
        15,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let blockers = occupy_indicator_workers(operator.native_checkpoint_workers(), &job).await;
    let context = StreamOperatorContext::new(&job, "features", None);
    let input = Batch::table(
        vec![float64_fast_record(&[(1, "a", 1, Some(1.0))])],
        BatchMetadata::default(),
    )
    .unwrap();
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let mut callback = Box::pin(operator.process_data("input", input, &context, &mut output));
    let completed = tokio::time::timeout(std::time::Duration::from_millis(1), &mut callback)
        .await
        .is_ok();
    drop(callback);
    for (release, worker) in blockers {
        release.send(()).unwrap();
        worker.await.unwrap().unwrap();
    }
    assert!(
        !completed,
        "indicator preparation bypassed its actual worker pool"
    );
    assert!(operator.state.buffer.is_empty());
    assert_eq!(operator.state.charge, empty_indicator_charge(&operator));
}

#[tokio::test]
async fn test_indicator_finalization_waits_for_workers_and_abandoned_waiter_cannot_commit() {
    let (mut operator, job) = pending_indicator_fixture().await;
    let original = operator.indicator_state_packet().unwrap();
    let blockers = occupy_indicator_workers(operator.native_checkpoint_workers(), &job).await;
    let context = StreamOperatorContext::new(&job, "actual_node", None);
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let mut callback =
        Box::pin(operator.on_watermark(EventTime::from_micros(2), &context, &mut output));
    let completed = tokio::time::timeout(std::time::Duration::from_millis(1), &mut callback)
        .await
        .is_ok();
    drop(callback);
    for (release, worker) in blockers {
        release.send(()).unwrap();
        worker.await.unwrap().unwrap();
    }
    assert!(
        !completed,
        "indicator finalization bypassed its actual worker pool"
    );
    assert!(Arc::ptr_eq(
        &original,
        &operator.indicator_state_packet().unwrap()
    ));
    assert_eq!(operator.state.buffer.len(), 1);
    assert_eq!(operator.state.last_input_watermark, None);
    assert!(output.drain("output").is_empty());
}

#[tokio::test]
async fn test_owned_indicator_preparation_runs_real_math_before_atomic_commit() {
    let (mut operator, job) = pending_indicator_fixture().await;
    let context = StreamOperatorContext::new(&job, "actual_node", None);
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .finalize_indicator_worker(Some(EventTime::from_micros(2)), &context, &mut output)
        .await
        .unwrap();
    assert!(operator.state.buffer.is_empty());
    assert_eq!(
        operator.state.last_input_watermark,
        Some(EventTime::from_micros(2))
    );
    assert_eq!(operator.state.histories.by_entity.len(), 1);
    assert_eq!(operator.state.next_output_sequence, 1);
    assert_eq!(output.drain("output").len(), 1);
    let resources = operator.indicator_resources.as_ref().unwrap();
    resources
        .check(
            operator.state.charge.bytes,
            operator.state_budget,
            "actual_node",
        )
        .unwrap();
}

#[tokio::test]
async fn test_owned_indicator_preparation_lost_waiter_keeps_old_packet() {
    let (mut operator, job) = pending_indicator_fixture().await;
    let original = operator.indicator_state_packet().unwrap();
    let blockers = occupy_indicator_workers(operator.native_checkpoint_workers(), &job).await;
    let context = StreamOperatorContext::new(&job, "actual_node", None);
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let mut callback = Box::pin(operator.finalize_indicator_worker(
        Some(EventTime::from_micros(2)),
        &context,
        &mut output,
    ));
    let completed = tokio::time::timeout(std::time::Duration::from_millis(1), &mut callback)
        .await
        .is_ok();
    drop(callback);
    for (release, worker) in blockers {
        release.send(()).unwrap();
        worker.await.unwrap().unwrap();
    }
    assert!(!completed);
    assert!(Arc::ptr_eq(
        &original,
        &operator.indicator_state_packet().unwrap()
    ));
    assert!(output.drain("output").is_empty());
}

#[tokio::test]
async fn test_indicator_capture_reuses_one_payload_across_distinct_epochs() {
    let (mut operator, job) = pending_indicator_fixture().await;
    let original = operator.indicator_state_packet().unwrap();
    let context = StreamOperatorContext::new(&job, "actual_node", None);
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let first = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let second = operator.checkpoint(Epoch::new(2).unwrap()).unwrap();
    let left = first.segments.values().next().unwrap();
    let right = second.segments.values().next().unwrap();
    assert!(Arc::ptr_eq(&left.bytes_arc(), &right.bytes_arc()));
    assert_eq!(left.sha256(), right.sha256());
    assert_ne!(first.segments.keys().next(), second.segments.keys().next());
    assert!(Arc::ptr_eq(
        &original,
        &operator.indicator_state_packet().unwrap()
    ));
    assert!(operator.checkpoint(Epoch::new(2).unwrap()).is_err());
}

#[tokio::test]
async fn test_indicator_reset_drops_cache_but_keeps_external_capture_credit() {
    let (mut operator, job) = pending_indicator_fixture().await;
    let context = StreamOperatorContext::new(&job, "actual_node", None);
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    StreamOperator::reset(&mut operator).unwrap();
    let resources = operator.indicator_resources.as_ref().unwrap();
    let tiny = StateBudget::new(10, empty_indicator_charge(&operator).bytes).unwrap();
    assert!(resources.check(0, tiny, "actual_node").is_err());
    drop(snapshot);
    resources.check(0, tiny, "actual_node").unwrap();
    assert!(operator.checkpoint(Epoch::new(2).unwrap()).is_err());
}

#[test]
fn test_indicator_cells_are_controlled_before_arrow_materialization() {
    let mut spec = kernel_spec(json!([
        {"kind":"rsi", "primitive_version":1, "input":"price", "output":"rsi", "period":1}
    ]));
    spec.state_layout_version = 5;
    let operator = RollingOperator::new("features", Arc::new(kernel_schema()), spec).unwrap();
    let input = Batch::table(
        vec![float64_fast_record(&[
            (1, "a", 1, Some(1.0)),
            (2, "a", 2, Some(2.0)),
            (3, "a", 3, Some(1.0)),
        ])],
        BatchMetadata::default(),
    )
    .unwrap();
    let rows = read_buffered_rows(
        input.table_payload().unwrap(),
        &operator.compiled,
        "features",
    )
    .unwrap();
    let histories = RollingHistories::default();
    let cells = compute_output_cells_checked(
        &rows,
        &histories,
        &operator.compiled,
        "features",
        None,
        &mut || Ok(()),
    )
    .unwrap();
    assert_eq!(
        cells.columns[0],
        [
            Some(ScalarValue::Float64(None)),
            Some(ScalarValue::Float64(Some(100.0))),
            Some(ScalarValue::Float64(Some(0.0))),
        ]
    );
    let error = compute_output_cells_checked(
        &rows,
        &histories,
        &operator.compiled,
        "features",
        None,
        &mut || {
            Err(CalcFlowError::Cancelled {
                run_id: "cells".into(),
            })
        },
    )
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { run_id } if run_id == "cells"));
    assert!(histories.by_entity.is_empty());
}

#[test]
fn test_indicator_lag_warmup_uses_representable_controlled_nested_null() {
    use datafusion::arrow::{
        array::{ListArray, UnionArray},
        buffer::{OffsetBuffer, ScalarBuffer},
        datatypes::{UnionFields, UnionMode},
    };
    let child = Arc::new(Field::new(
        "variant",
        DataType::Union(UnionFields::empty(), UnionMode::Sparse),
        true,
    ));
    let bag = ListArray::new(
        Arc::clone(&child),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 0])),
        Arc::new(
            UnionArray::try_new(
                UnionFields::empty(),
                ScalarBuffer::from(Vec::<i8>::new()),
                None,
                vec![],
            )
            .unwrap(),
        ),
        None,
    );
    let mut fields = kernel_schema().fields().iter().cloned().collect::<Vec<_>>();
    fields.push(Arc::new(Field::new("bag", bag.data_type().clone(), true)));
    let schema = Schema::new(fields);
    let mut spec = kernel_spec(json!([
        {"kind":"rsi","primitive_version":1,"input":"price","output":"rsi","period":1},
        {"kind":"lag","primitive_version":1,"input":"bag","output":"previous","periods":1}
    ]));
    spec.state_layout_version = 5;
    let compiled = compile_spec(&spec, &schema).unwrap();
    let record = float64_fast_record(&[(1, "a", 1, Some(1.0))]);
    let mut values = (0..record.num_columns())
        .map(|index| ScalarValue::try_from_array(record.column(index), 0).unwrap())
        .collect::<Vec<_>>();
    values.push(ScalarValue::List(Arc::new(bag)));
    let rows = vec![buffered_row_from_values(values, &compiled).unwrap()];
    let computed = compute_output_cells_checked(
        &rows,
        &RollingHistories::default(),
        &compiled,
        "features",
        None,
        &mut || Ok(()),
    )
    .unwrap();
    let Some(ScalarValue::List(result)) = &computed.columns[1][0] else {
        panic!("typed list")
    };
    assert!(result.is_null(0));
    assert_eq!(result.data_type(), &DataType::List(Arc::clone(&child)));
}

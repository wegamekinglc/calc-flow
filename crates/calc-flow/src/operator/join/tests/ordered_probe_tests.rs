use super::*;
use crate::runtime::streaming::gather_work::TestService;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

const ROWS: usize = 16_384;
type Observations = Arc<Mutex<Vec<(data_work::ProbePhase, usize, bool)>>>;

fn incoming() -> Batch {
    let record = RecordBatch::try_new(
        left_schema(),
        vec![
            Arc::new(Int64Array::from(vec![7; ROWS])),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(
                    (0..ROWS).map(|position| i64::try_from(position % 2).unwrap()),
                )
                .with_timezone("UTC"),
            ),
            Arc::new(Int64Array::from_iter_values(
                (0..ROWS).map(|position| i64::try_from(position).unwrap()),
            )),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

async fn restored_operator(limit: u64, context: &StreamOperatorContext<'_>) -> StreamJoinOperator {
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(20_000, 10_000_000, limit).unwrap();
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap();
    let record = RecordBatch::try_new(
        right_schema(),
        vec![
            Arc::new(Int64Array::from(vec![7; 5])),
            Arc::new(
                TimestampMicrosecondArray::from(vec![-400_000_000, 2, -400_000_000, 0, 0])
                    .with_timezone("UTC"),
            ),
            Arc::new(StringArray::from(vec![
                "expired",
                "two",
                "expired",
                "zero-first",
                "zero-second",
            ])),
        ],
    )
    .unwrap();
    let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", batch, context, &mut collector)
        .await
        .unwrap();
    let progress = progress_context(
        context.job(),
        (IngressState::Active, Some(-1)),
        (IngressState::Active, None),
    );
    operator
        .on_ingress_progress("left", &progress)
        .await
        .unwrap();
    assert_eq!(operator.state.right.len(), 3);
    let snapshot = operator.checkpoint_v1(Epoch::INITIAL).unwrap();
    operator.restore(&snapshot).unwrap();
    let mut ids = operator
        .state
        .right
        .iter()
        .map(|row| row.row_id)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    assert_eq!(ids, [1, 3, 4]);
    assert_eq!(operator.state.next_right_row_id, 5);
    assert!(collector.drain("output").is_empty());
    operator
}

fn observe_units(operator: &mut StreamJoinOperator, job: &StreamJobContext) -> Observations {
    let observations = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&observations);
    let counts = AtomicUsize::new(0);
    let owner = job.gather_owner().clone();
    let actor = std::thread::current().id();
    operator.probe_unit_test_hook = Some(Arc::new(move |phase, ordinal, completed| {
        assert_ne!(std::thread::current().id(), actor);
        assert!(owner.funding().2 > 0);
        recorded.lock().unwrap().push((phase, ordinal, completed));
        if phase == data_work::ProbePhase::Count && completed {
            counts.fetch_add(1, Ordering::SeqCst);
        }
        if phase == data_work::ProbePhase::Fill
            && ordinal == 0
            && !completed
            && counts.load(Ordering::SeqCst) == 2
        {
            assert!(owner.wait_parallel_unit(1, Duration::from_secs(10)));
        }
    }));
    observations
}

fn assert_literal_pairs(collector: &mut EdgeCollector) {
    let messages = collector.drain("output");
    assert_eq!(messages.len(), 5);
    let mut global = 0;
    for (sequence, message) in messages.iter().enumerate() {
        let batch = message.as_data().unwrap();
        assert_eq!(
            batch.metadata().sequence(),
            u64::try_from(sequence).unwrap()
        );
        let record = &batch.table_payload().unwrap().batches()[0];
        let positions = record
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let status = record
            .column(5)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let times = record
            .column(4)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        for row in 0..record.num_rows() {
            assert_eq!(positions.value(row), i64::try_from(global / 3).unwrap());
            assert_eq!(
                status.value(row),
                ["zero-first", "zero-second", "two"][global % 3]
            );
            assert_eq!(times.value(row), [0, 0, 2][global % 3]);
            global += 1;
        }
    }
    assert_eq!(global, 3 * ROWS);
}

fn completed_units(observations: &Observations, phase: data_work::ProbePhase) -> Vec<usize> {
    observations
        .lock()
        .unwrap()
        .iter()
        .filter_map(|&(actual, ordinal, completed)| {
            (actual == phase && completed).then_some(ordinal)
        })
        .collect()
}

async fn assert_ordered_success(service: &TestService) {
    let job = job().with_gather_owner(service.owner("ordered-probe".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut operator = restored_operator(49_152, &context).await;
    let observed = observe_units(&mut operator, &job);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    reset_join_work();
    operator
        .process_data("left", incoming(), &context, &mut collector)
        .await
        .unwrap();
    assert_literal_pairs(&mut collector);
    assert_eq!(
        completed_units(&observed, data_work::ProbePhase::Fill),
        [1, 0],
        "two paid fill units must complete in reverse order while output remains canonical"
    );
    let mut counts = completed_units(&observed, data_work::ProbePhase::Count);
    counts.sort_unstable();
    assert_eq!(counts, [0, 1]);
    assert_eq!(join_work().sql_probe_table_builds, 0);
    assert_eq!(join_work().native_range_visits, 0);
    assert_eq!(operator.state.next_left_row_id, 16_384);
    assert_eq!(operator.state.left.len(), ROWS);
    assert_eq!(operator.state.metrics.emitted_match_rows, 49_152);
    assert_eq!(operator.state.next_output_sequence, 5);
    assert_eq!(operator.state.deltas.pending.iter().count(), ROWS);
    assert_settled(&operator, &job);
    drop(context);
    finish(operator, job).await;
}

fn assert_settled(operator: &StreamJoinOperator, job: &StreamJobContext) {
    assert!(operator.compaction_release.is_none());
    assert!(operator.compaction_cleanup.is_none());
    assert!(operator.probe_control.is_none());
    assert_eq!(
        Arc::strong_count(operator.state.right.1.as_ref().unwrap()),
        1
    );
    let pool = operator
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool();
    native_lookup_tests::assert_resident_and_gather_funding(
        pool.as_ref(),
        job,
        native_lookup_tests::state_funding(operator),
    );
}

async fn finish(operator: StreamJoinOperator, job: StreamJobContext) {
    let pool = operator
        .runtime
        .runtime
        .as_ref()
        .unwrap()
        .incremental_memory_pool();
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

async fn assert_global_limit(service: &TestService) {
    let job = job().with_gather_owner(service.owner("global-limit-probe".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut operator = restored_operator(32_768, &context).await;
    let observed = observe_units(&mut operator, &job);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let before = operator.status();
    let error = operator
        .process_data("left", incoming(), &context, &mut collector)
        .await
        .unwrap_err();
    assert_eq!(
        reason_of(&error),
        Some(crate::StreamingFailureReason::JoinMatchLimitExceeded)
    );
    assert!(
        error
            .to_string()
            .contains("input batch match limit exceeded")
    );
    assert!(collector.drain("output").is_empty());
    assert!(completed_units(&observed, data_work::ProbePhase::Fill).is_empty());
    assert!(
        observed
            .lock()
            .unwrap()
            .iter()
            .all(|&(phase, _, _)| phase != data_work::ProbePhase::Fill)
    );
    let mut counts = completed_units(&observed, data_work::ProbePhase::Count);
    counts.sort_unstable();
    assert_eq!(counts, [0, 1]);
    assert_eq!(operator.state.metrics.match_limit_failures, 1);
    assert_eq!(operator.status().left, before.left);
    assert_eq!(operator.status().right, before.right);
    assert_eq!(operator.state.metrics.emitted_match_rows, 0);
    assert_eq!(operator.state.next_left_row_id, 0);
    assert_eq!(operator.state.next_output_sequence, 0);
    assert!(operator.state.deltas.pending.iter().next().is_none());
    assert_settled(&operator, &job);
    drop(context);
    finish(operator, job).await;
}

pub(super) async fn assert_bounded_order_and_global_limit(service: &TestService) {
    assert_ordered_success(service).await;
    assert_global_limit(service).await;
}

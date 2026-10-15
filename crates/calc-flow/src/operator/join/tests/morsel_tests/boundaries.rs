use super::{support::*, *};
use datafusion::execution::memory_pool::MemoryPool;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn test_morsel_ranges_preserve_sliced_order_projection_and_checkpoint() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    for incoming_left in [true, false] {
        for projected in [false, true] {
            runtime.block_on(ordered_case(&service, incoming_left, projected));
        }
    }
    drop(runtime);
    service.shutdown();
}

async fn ordered_case(service: &TestService, incoming_left: bool, projected: bool) {
    let mut operator = operator(projected);
    let job = job().with_gather_owner(service.owner("morsel-order".into()));
    preload(&mut operator, &job, incoming_left).await;
    let context = StreamOperatorContext::new(&job, "match", None);
    let rows = ROWS + 3;
    let source = input(incoming_left, rows, false);
    let before = source.table_payload().unwrap().batches().to_vec();
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    reset_join_work();
    operator
        .process_data(
            if incoming_left { "left" } else { "right" },
            source.clone(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    assert_eq!(
        join_work().output_column_takes + join_work().output_column_interleaves,
        0,
        "eligible gather must stay on morsel workers"
    );
    verify(&mut output, rows, incoming_left, projected);
    assert_eq!(source.table_payload().unwrap().batches(), &before);
    let status = operator.status();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    let mut restored = support::operator(projected);
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status(), status);
    assert_eq!(
        restored.state.next_output_sequence,
        operator.state.next_output_sequence
    );
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((operator, restored));
    drop(snapshot);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_morsel_limits_precede_output_and_state_commit() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    for failure in ["match", "state", "row", "sequence", "counter"] {
        runtime.block_on(limit_case(&service, failure));
    }
    drop(runtime);
    service.shutdown();
}

async fn limit_case(service: &TestService, failure: &str) {
    let mut operator = operator(false);
    let job = job().with_gather_owner(service.owner("morsel-limits".into()));
    preload(&mut operator, &job, true).await;
    let mut context = StreamOperatorContext::new(&job, "match", None);
    match failure {
        "match" => {
            operator.spec.limits.max_matches_per_input_batch = (ROWS * 8 - 1) as u64;
            operator.spec.limits.max_state_rows_per_side = 10;
        }
        "state" => operator.spec.limits.max_state_rows_per_side = 10,
        "row" => context = context.with_output_budget(EdgeBudget::new(10_000, 1).unwrap()),
        "sequence" => operator.state.next_output_sequence = u64::MAX,
        "counter" => operator.state.metrics.emitted_match_rows = u64::MAX,
        _ => unreachable!(),
    }
    let fills = Arc::new(AtomicUsize::new(0));
    let observed = fills.clone();
    operator.probe_unit_test_hook = Some(Arc::new(move |phase, _, completed| {
        if phase == data_work::ProbePhase::Fill && !completed {
            observed.fetch_add(1, Ordering::SeqCst);
        }
    }));
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let error = operator
        .process_data("left", input(true, ROWS, false), &context, &mut output)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains(match failure {
            "match" => "match",
            "state" => "state",
            "row" => "byte",
            "sequence" => "sequence",
            "counter" => "emitted_match_rows",
            _ => unreachable!(),
        }),
        "{failure}: {error}"
    );
    if matches!(failure, "match" | "state") {
        assert_eq!(fills.load(Ordering::SeqCst), 0);
    }
    assert!(output.drain("output").is_empty());
    assert_eq!(operator.state.next_left_row_id, 0);
    assert_eq!(operator.status().left.retained_rows, 0);
    assert_eq!(operator.status().right.retained_rows, 8);
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop(operator);
    let failures = job.gather_owner().close_and_drain().await;
    if failure == "row" {
        assert_eq!(
            failures.len(),
            3,
            "retain each secondary range preflight error"
        );
        assert!(
            failures
                .iter()
                .all(|failure| failure.error.to_string().contains("byte"))
        );
    } else {
        assert!(failures.is_empty(), "{failures:?}");
    }
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_morsel_scratch_and_result_refusals_use_prepaid_fallback() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    for reject_at in [1, 2] {
        for projected in [false, true] {
            runtime.block_on(refusal_case(&service, reject_at, projected));
        }
    }
    drop(runtime);
    service.shutdown();
}

async fn refusal_case(service: &TestService, reject_at: usize, projected: bool) {
    let pool = Pool::new(reject_at);
    let mut operator = operator(projected);
    operator
        .runtime
        .initialize_benchmark_pool(pool.clone())
        .unwrap();
    let job = job().with_gather_owner(service.owner("morsel-refusal".into()));
    preload(&mut operator, &job, true).await;
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", input(true, ROWS, false), &context, &mut output)
        .await
        .unwrap();
    assert!(pool.attempts.load(Ordering::SeqCst) >= reject_at);
    assert_eq!(
        pool.live.load(Ordering::SeqCst),
        0,
        "fallback retains no fused workspace"
    );
    verify(&mut output, ROWS, true, projected);
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_morsel_worker_admission_refusal_falls_back_or_propagates_cancel() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    for cancel in [false, true] {
        runtime.block_on(worker_refusal_case(&service, cancel));
    }
    drop(runtime);
    service.shutdown();
}

async fn worker_refusal_case(service: &TestService, cancel: bool) {
    use crate::runtime::streaming::gather_work::admission_probe::{AdmissionProbe, AdmissionStage};
    let pool = Pool::new(0);
    let mut operator = operator(true);
    operator
        .runtime
        .initialize_benchmark_pool(pool.clone())
        .unwrap();
    let job = job().with_gather_owner(service.owner("morsel-worker-refusal".into()));
    preload(&mut operator, &job, true).await;
    let owner = job.gather_owner().clone();
    let observed_pool = pool.clone();
    let token = job.cancellation().clone();
    let observed = Arc::new(parking_lot::Mutex::new(None));
    let installed = observed.clone();
    operator.probe_unit_test_hook = Some(Arc::new(move |phase, unit, completed| {
        if phase == data_work::ProbePhase::Count && unit == 0 && completed {
            let probe = AdmissionProbe::install(
                &owner,
                AdmissionStage::Attempt,
                observed_pool.clone(),
                1 << 30,
            );
            if cancel {
                probe.cancel_on_refusal(token.clone());
            }
            *installed.lock() = Some(probe);
        }
    }));
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let result = operator
        .process_data("left", input(true, ROWS, false), &context, &mut output)
        .await;
    let probe = observed.lock().take().unwrap();
    let event = probe.take_event().expect("real worker admission refusal");
    assert_eq!(event.available + 1, event.fee);
    if cancel {
        assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
        assert!(output.drain("output").is_empty());
        assert_eq!(operator.state.next_left_row_id, 0);
    } else {
        result.unwrap();
        verify(&mut output, ROWS, true, true);
    }
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

use super::*;
use crate::runtime::streaming::gather_work::admission_probe::{AdmissionProbe, AdmissionStage};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn test_join_column_gather_admission_refusal_does_not_emit_or_commit() {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    for stage in [AdmissionStage::Attempt, AdmissionStage::Generation] {
        for projected in [false, true] {
            runtime.block_on(assert_refusal(&service, stage, projected));
        }
    }
    drop(runtime);
    service.shutdown();
}

async fn assert_refusal(service: &TestService, stage: AdmissionStage, projected: bool) {
    let mut operator = with_one_gib_pool(
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap(),
    );
    if projected {
        operator.set_output_projection(vec![2, 5]).unwrap();
    }
    let source_job = job();
    let source_context = StreamOperatorContext::new(&source_job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "right",
            right_batch(vec![0]),
            &source_context,
            &mut collector,
        )
        .await
        .unwrap();
    let job = job().with_gather_owner(service.owner("refused-join-columns".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    assert_eq!(job.gather_owner().funding(), (0, 0, 0));
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let _configuration = output_gather::test_support::configure(
        true,
        Some(Arc::new(move |_, completed| {
            if !completed {
                observed.fetch_add(1, Ordering::SeqCst);
            }
        })),
    );
    let probe = AdmissionProbe::install(job.gather_owner(), stage, pool.clone(), 1 << 30);
    let error = operator
        .process_data("left", left_batch(vec![1, 2]), &context, &mut collector)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("credit admission"), "{error}");
    let event = probe
        .take_event()
        .expect("actual native admission was attempted");
    assert_eq!(event.stage, stage);
    assert_eq!(event.available + 1, event.fee);
    assert_eq!(event.operator, "match");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(operator.state.next_left_row_id, 0);
    assert_eq!(operator.state.next_output_sequence, 0);
    assert_eq!(operator.state.deltas.pending.iter().count(), 1);
    assert!(collector.drain("output").is_empty());
    operator
        .process_data("left", left_batch(vec![1, 2]), &context, &mut collector)
        .await
        .unwrap();
    assert_selected_literal_output(&mut collector, &operator, projected);
    assert_eq!(calls.load(Ordering::SeqCst), if projected { 2 } else { 6 });
    assert_eq!(operator.state.next_left_row_id, 2);
    assert_eq!(operator.state.next_output_sequence, 1);
    assert!(operator.compaction_cleanup.is_none());
    drop((operator, context, source_context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    assert!(source_job.gather_owner().close_and_drain().await.is_empty());
    drop((job, source_job));
    assert_eq!(pool.reserved(), 0);
}

use super::*;

#[path = "complete_expiry_boundary_tests.rs"]
mod boundary;

#[tokio::test]
async fn test_off_final_watermark_retires_whole_side_without_per_row_index_work() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job().with_checkpointing(false);
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let times = (0..65).map(|index| index * 37 % 65).collect();
    operator
        .process_data("left", left_batch(times), &context, &mut output)
        .await
        .unwrap();
    operator
        .process_data("right", right_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(output.drain("output")[0].as_data().unwrap().num_rows(), 65);
    assert!(
        operator.state.left.1.is_some(),
        "fixture includes a live native index"
    );
    let right = operator.state.metrics.right.clone();
    let sequence = operator.state.next_output_sequence;
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let progress = progress_context(
        &job,
        (IngressState::Active, None),
        (IngressState::Active, Some(60_000_065)),
    );
    reset_join_work();
    operator
        .on_ingress_progress("right", &progress)
        .await
        .unwrap();
    let work = join_work();
    assert!(operator.state.left.is_empty());
    assert_eq!(operator.state.metrics.left.retained_rows, 0);
    assert_eq!(operator.state.metrics.left.retained_bytes, 0);
    assert_eq!(operator.state.metrics.left.evicted_rows, 65);
    assert_eq!(operator.state.metrics.right, right);
    assert_eq!(operator.state.right.len(), 1);
    assert_eq!(operator.state.next_output_sequence, sequence);
    assert_eq!(operator.state.next_left_row_id, 65);
    assert_eq!(operator.state.metrics.emitted_match_rows, 65);
    assert!(!operator.state.ended);
    assert!(operator.state.deltas.pending.is_empty());
    assert!(output.drain("output").is_empty());
    assert!(!operator.has_sparse_candidates());
    drop((operator, context, progress));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(
        work.retained_visits, 0,
        "complete expiry must bypass heap pops"
    );
    assert_eq!(
        work.native_eviction_comparisons, 0,
        "complete expiry must retire the native index together"
    );
}

use super::*;

#[tokio::test]
async fn test_off_complete_expiry_preserves_strict_boundary_and_maximum_on_growth() {
    let mut declaration = spec();
    declaration.bounds =
        JoinTimeBounds::new(Duration::from_micros(10), Duration::from_micros(20)).unwrap();
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap();
    let job = job().with_checkpointing(false);
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    for times in [vec![90, 10, 40, 90], vec![20, 30, 50, 60, 70]] {
        operator
            .process_data("left", left_batch(times), &context, &mut output)
            .await
            .unwrap();
    }
    let equal = progress_context(
        &job,
        (IngressState::Active, None),
        (IngressState::Active, Some(110)),
    );
    operator.on_ingress_progress("right", &equal).await.unwrap();
    let mut ids = operator
        .state
        .left
        .iter()
        .map(|row| row.row_id)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    assert_eq!(
        ids,
        [0, 3],
        "the old maximum survives a lower-time capacity-growing append and an equal frontier"
    );
    let expired = progress_context(
        &job,
        (IngressState::Active, None),
        (IngressState::Active, Some(111)),
    );
    reset_join_work();
    operator
        .on_ingress_progress("right", &expired)
        .await
        .unwrap();
    assert_eq!(join_work().retained_visits, 0);
    assert_eq!(operator.state.metrics.left.evicted_rows, 9);
    assert!(operator.state.left.is_empty());
    operator
        .process_data("left", left_batch(vec![200, 180]), &expired, &mut output)
        .await
        .unwrap();
    let later = progress_context(
        &job,
        (IngressState::Active, None),
        (IngressState::Active, Some(221)),
    );
    reset_join_work();
    operator.on_ingress_progress("right", &later).await.unwrap();
    assert_eq!(join_work().retained_visits, 0);
    assert_eq!(operator.state.metrics.left.evicted_rows, 11);
    assert!(operator.state.left.is_empty());
    assert_maximum_time_expiry(&mut operator, &job, &later, &mut output).await;
    assert!(output.drain("output").is_empty());
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((operator, context, equal, expired, later));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

async fn assert_maximum_time_expiry(
    operator: &mut StreamJoinOperator,
    job: &StreamJobContext,
    context: &StreamOperatorContext<'_>,
    output: &mut EdgeCollector,
) {
    operator
        .process_data(
            "left",
            left_batch(vec![i64::MAX - 1, i64::MAX]),
            context,
            output,
        )
        .await
        .unwrap();
    let maximum = progress_context(
        job,
        (IngressState::Active, None),
        (IngressState::Active, Some(i64::MAX)),
    );
    operator
        .on_ingress_progress("right", &maximum)
        .await
        .unwrap();
    assert_eq!(
        operator.state.left.len(),
        2,
        "wide arithmetic must preserve unexpired extreme timestamps"
    );
    let ended = progress_context(
        job,
        (IngressState::Active, None),
        (IngressState::Ended, Some(i64::MAX)),
    );
    reset_join_work();
    operator.on_ingress_progress("right", &ended).await.unwrap();
    assert_eq!(join_work().retained_visits, 0);
    assert!(operator.state.left.is_empty());
    assert_eq!(operator.state.metrics.left.evicted_rows, 13);
    operator.on_ingress_progress("right", &ended).await.unwrap();
    assert_eq!(
        operator.state.metrics.left.evicted_rows, 13,
        "empty progress does not count twice"
    );
}

#[tokio::test]
async fn test_complete_expiry_rejects_cancellation_and_metric_overflow_before_mutation() {
    for cancelled in [false, true] {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job = job().with_checkpointing(false);
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", left_batch(vec![3, 0, 2, 1]), &context, &mut output)
            .await
            .unwrap();
        operator
            .process_data("right", right_batch(vec![0]), &context, &mut output)
            .await
            .unwrap();
        output.drain("output");
        if cancelled {
            job.cancellation().cancel();
        } else {
            operator.state.metrics.left.evicted_rows = u64::MAX;
        }
        let previous = operator.status();
        let rows = Arc::clone(&operator.state.left.0);
        let index = Arc::clone(operator.state.left.1.as_ref().unwrap());
        let progress = progress_context(
            &job,
            (IngressState::Active, None),
            (IngressState::Active, Some(60_000_004)),
        );
        let error = operator
            .on_ingress_progress("right", &progress)
            .await
            .unwrap_err();
        if cancelled {
            assert!(matches!(error, CalcFlowError::Cancelled { .. }));
        } else {
            assert!(error.to_string().contains("evicted_rows"));
        }
        assert_eq!(operator.status(), previous);
        assert!(Arc::ptr_eq(&operator.state.left.0, &rows));
        assert!(Arc::ptr_eq(operator.state.left.1.as_ref().unwrap(), &index));
        assert_eq!(
            operator
                .state
                .left
                .iter()
                .map(|row| row.row_id)
                .collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert!(output.drain("output").is_empty());
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop((rows, index, operator, context, progress));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(job);
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test]
async fn test_checkpoint_on_complete_expiry_preserves_tombstone_retention_order() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            left_batch(vec![3, 0, 4, 1, 5, 2]),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    operator.checkpoint(Epoch::INITIAL).unwrap();
    let partial = progress_context(
        &job,
        (IngressState::Active, None),
        (IngressState::Active, Some(60_000_003)),
    );
    operator
        .on_ingress_progress("right", &partial)
        .await
        .unwrap();
    operator.checkpoint(Epoch::new(2).unwrap()).unwrap();
    let complete = progress_context(
        &job,
        (IngressState::Active, None),
        (IngressState::Active, Some(60_000_006)),
    );
    reset_join_work();
    operator
        .on_ingress_progress("right", &complete)
        .await
        .unwrap();
    assert_eq!(
        join_work().retained_visits,
        3,
        "checkpoint-enabled expiry retains its ordered journal path"
    );
    assert_eq!(
        operator
            .state
            .deltas
            .pending
            .iter()
            .map(PendingOp::identity)
            .collect::<Vec<_>>(),
        [
            (JoinSide::Left, 0),
            (JoinSide::Left, 2),
            (JoinSide::Left, 4)
        ]
    );
    assert_eq!(operator.state.metrics.left.evicted_rows, 6);
    assert!(output.drain("output").is_empty());
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((operator, context, partial, complete));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

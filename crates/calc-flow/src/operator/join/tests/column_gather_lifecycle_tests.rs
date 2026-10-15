use super::*;

#[test]
fn test_join_column_gather_cancellation_keeps_owners_until_workers_settle() {
    assert_stopped_gather(true);
}

#[test]
fn test_join_column_gather_abandonment_keeps_owners_until_workers_settle() {
    assert_stopped_gather(false);
}

fn assert_stopped_gather(cancel: bool) {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let responsive = [false, true]
        .into_iter()
        .map(|projected| runtime.block_on(assert_cancelled_columns(&service, cancel, projected)))
        .collect::<Vec<_>>();
    drop(runtime);
    service.shutdown();
    assert!(
        responsive.into_iter().all(|responsive| responsive),
        "cancellation must return before the native column is released"
    );
}

async fn assert_cancelled_columns(service: &TestService, cancel: bool, projected: bool) -> bool {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    if projected {
        operator.set_output_projection(vec![2, 5]).unwrap();
    }
    let job = job().with_gather_owner(service.owner("join-cancel-columns".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right_batch(vec![0]), &context, &mut collector)
        .await
        .unwrap();
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let retained = Arc::downgrade(operator.state.right[0].record.column(0));
    let gates = ReleaseGates(vec![Arc::new(Gate::default())]);
    let worker_gate = Arc::clone(&gates.0[0]);
    let (started, mut observed) = tokio::sync::mpsc::unbounded_channel();
    let _configuration = output_gather::test_support::configure(
        true,
        Some(Arc::new(move |ordinal, completed| {
            if ordinal == 0 && !completed {
                started.send(()).unwrap();
                worker_gate.wait();
            }
        })),
    );
    let mut process =
        Box::pin(operator.process_data("left", left_batch(vec![1, 2]), &context, &mut collector));
    tokio::select! {
        entered = observed.recv() => assert_eq!(entered, Some(())),
        result = &mut process => panic!("Join finished before paused column: {result:?}"),
    }
    let responsive = if cancel {
        job.cancellation().cancel();
        matches!(
            futures::poll!(process.as_mut()),
            std::task::Poll::Ready(Err(CalcFlowError::Cancelled { .. }))
        )
    } else {
        true
    };
    drop(process);
    assert_eq!(operator.state.next_left_row_id, 0);
    assert_eq!(operator.state.next_output_sequence, 0);
    assert_eq!(operator.state.deltas.pending.iter().count(), 1);
    assert!(collector.drain("output").is_empty());
    assert!(operator.compaction_cleanup.is_some());
    operator.reset().unwrap();
    drop((operator, context));
    assert!(
        retained.upgrade().is_some(),
        "paused worker retains the old payload owner"
    );
    assert!(pool.reserved() > 0);
    let mut drain = Box::pin(job.gather_owner().close_and_drain());
    assert!(futures::poll!(drain.as_mut()).is_pending());
    gates.0[0].release();
    assert!(drain.await.is_empty());
    assert!(retained.upgrade().is_none());
    drop(job);
    assert_eq!(pool.reserved(), 0);
    responsive
}

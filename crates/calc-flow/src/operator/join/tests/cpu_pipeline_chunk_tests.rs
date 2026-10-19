use super::*;
use crate::runtime::streaming::gather_work::TestService;

fn operator() -> StreamJoinOperator {
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(100_000, 100_000_000, 100_000).unwrap();
    StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap()
}

#[test]
fn test_cpu_pipeline_retained_chunks_use_ordered_native_units() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut operator = operator();
        let rows = 65_539;
        let source = left_batch((0..rows).map(|row| i64::try_from(row).unwrap()).collect());
        let record = &source.table_payload().unwrap().batches()[0];
        let source = Batch::table(
            vec![record.slice(0, 33_001), record.slice(33_001, rows - 33_001)],
            BatchMetadata::default(),
        )
        .unwrap();
        let observations = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = Arc::clone(&observations);
        let owner_thread = std::thread::current().id();
        let _observe =
            retained::observe_preparation_for_test(Arc::new(move |phase, unit, rows| {
                captured
                    .lock()
                    .unwrap()
                    .push((phase, unit, rows, std::thread::current().id()));
            }));
        let job = job().with_gather_owner(service.owner("retained-chunks".into()));
        let context = progress_context(
            &job,
            (IngressState::Active, None),
            (IngressState::Active, Some(60_000_100)),
        );
        let prepared = operator
            .prepare_batch("left", &source, &context)
            .await
            .unwrap();
        assert_eq!(prepared.next_row_id, u64::try_from(rows).unwrap());
        assert_eq!(prepared.chunks.len(), rows - 100);
        for index in 0..rows - 100 {
            let row = prepared.chunks.row(index);
            assert_eq!(row.row_id, u64::try_from(index + 100).unwrap());
            assert_eq!(
                row.event_time,
                EventTime::from_micros(i64::try_from(index + 100).unwrap())
            );
        }
        {
            let observed = observations.lock().unwrap();
            for (phase, expected_rows) in [("selection", rows), ("chunks", rows - 100)] {
                let mut actual = observed
                    .iter()
                    .filter(|event| event.0 == phase)
                    .collect::<Vec<_>>();
                actual.sort_by_key(|event| event.1);
                assert_eq!(
                    actual.iter().map(|event| event.1).collect::<Vec<_>>(),
                    [0, 1, 2, 3]
                );
                assert_eq!(
                    actual.iter().map(|event| event.2).sum::<usize>(),
                    expected_rows
                );
                assert!(actual.iter().all(|event| event.3 != owner_thread));
            }
        }
        drop(prepared);
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop(operator);
        drop(context);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn test_cpu_pipeline_abandoned_retained_worker_keeps_credit_until_retirement() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut operator = operator();
        let source = left_batch(vec![0; 65_539]);
        let job = job().with_gather_owner(service.owner("retained-abandon".into()));
        let context = StreamOperatorContext::new(&job, "match", None);
        let (entered, started) = tokio::sync::oneshot::channel();
        let (release, waiting) = std::sync::mpsc::channel();
        let gate = std::sync::Mutex::new(Some((entered, waiting)));
        let _observe = retained::observe_preparation_for_test(Arc::new(move |phase, unit, _| {
            if phase == "chunks" && unit == 0 {
                let (entered, waiting) = gate.lock().unwrap().take().unwrap();
                entered.send(()).unwrap();
                waiting.recv_timeout(Duration::from_secs(10)).unwrap();
            }
        }));
        let mut pending = Box::pin(operator.prepare_batch("left", &source, &context));
        tokio::select! {
            result = &mut pending => panic!("retained worker must remain gated: {}", result.is_ok()),
            result = started => result.unwrap(),
        }
        drop(pending);
        assert_eq!(operator.state.next_left_row_id, 0);
        assert_eq!(operator.state.left.len(), 0);
        let pool = operator.runtime.runtime().unwrap().incremental_memory_pool();
        assert!(pool.reserved() > 0, "abandoned worker still owns prepaid buffers");
        release.send(()).unwrap();
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(operator);
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn test_cpu_pipeline_probe_only_batch_skips_retained_workers() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut operator = operator();
        let source = left_batch(vec![0; 65_536]);
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let _observe = retained::observe_preparation_for_test(Arc::new(move |_, _, _| {
            observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }));
        let job = job().with_gather_owner(service.owner("probe-only-retained".into()));
        let context = progress_context(
            &job,
            (IngressState::Active, None),
            (IngressState::Ended, None),
        );
        let prepared = operator
            .prepare_batch("left", &source, &context)
            .await
            .unwrap();
        assert!(prepared.chunks.is_empty());
        assert_eq!(prepared.next_row_id, 65_536);
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "retained worker count must follow retained work"
        );
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop(prepared);
        drop(operator);
        drop(context);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

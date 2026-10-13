use super::*;

struct GatedCollector {
    entered: Option<tokio::sync::oneshot::Sender<()>>,
    release: tokio::sync::oneshot::Receiver<bool>,
    accepted: Vec<Batch>,
}

#[async_trait]
impl StreamCollector for GatedCollector {
    async fn emit(&mut self, _port: &str, batch: Batch) -> Result<()> {
        self.entered.take().unwrap().send(()).unwrap();
        if (&mut self.release).await.unwrap() {
            Err(operator_error("match", "test collector rejected output"))
        } else {
            self.accepted.push(batch);
            Ok(())
        }
    }
}

#[test]
fn test_join_column_gather_output_credit_survives_blocked_and_failed_emit() {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    for reject in [false, true] {
        for projected in [false, true] {
            runtime.block_on(assert_emit_credit(&service, reject, projected));
        }
    }
    drop(runtime);
    service.shutdown();
}

async fn assert_emit_credit(service: &TestService, reject: bool, projected: bool) {
    let pool = GatherPool::new(0);
    let (mut operator, source_job) = preloaded(&pool).await;
    if projected {
        operator.set_output_projection(vec![2, 5]).unwrap();
    }
    let job = job().with_gather_owner(service.owner("join-emit-credit".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    let (entered, entering) = tokio::sync::oneshot::channel();
    let (release, released) = tokio::sync::oneshot::channel();
    let mut collector = GatedCollector {
        entered: Some(entered),
        release: released,
        accepted: Vec::new(),
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let column_calls = Arc::clone(&calls);
    let actor = std::thread::current().id();
    let _configuration = output_gather::test_support::configure(
        true,
        Some(Arc::new(move |_, completed| {
            if !completed {
                assert_ne!(std::thread::current().id(), actor);
                column_calls.fetch_add(1, Ordering::SeqCst);
            }
        })),
    );
    let mut process =
        Box::pin(operator.process_data("left", left_batch(vec![1, 2]), &context, &mut collector));
    tokio::select! {
        result = entering => result.unwrap(),
        result = &mut process => panic!("process finished before gated emit: {result:?}"),
    }
    assert_eq!(calls.load(Ordering::SeqCst), if projected { 2 } else { 6 });
    let growth = pool.growth.lock().clone();
    let input_bytes = growth[0];
    let output_bytes = growth[1];
    assert!(pool.refunds.lock().contains(&input_bytes));
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!(attempt, 0);
    assert!(home > 0 && generation > 0);
    assert_eq!(
        pool.current.load(Ordering::SeqCst),
        home + generation + output_bytes,
        "output allocation remains funded after worker/input cleanup"
    );
    release.send(reject).unwrap();
    let result = process.await;
    if reject {
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("test collector rejected output")
        );
        assert!(collector.accepted.is_empty());
        assert_eq!(operator.state.next_left_row_id, 0);
        assert_eq!(operator.state.next_output_sequence, 0);
    } else {
        result.unwrap();
        assert_eq!(collector.accepted.len(), 1);
        assert_eq!(collector.accepted[0].num_rows(), 2);
        assert_eq!(operator.state.next_left_row_id, 2);
        assert_eq!(operator.state.next_output_sequence, 1);
    }
    assert_eq!(pool.current.load(Ordering::SeqCst), home + generation);
    assert!(pool.refunds.lock().contains(&output_bytes));
    drop((operator, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    assert!(source_job.gather_owner().close_and_drain().await.is_empty());
    drop((job, source_job));
    assert_eq!(pool.reserved(), 0);
}

use super::{support::*, *};
use datafusion::execution::memory_pool::MemoryPool;
use std::sync::atomic::Ordering;

struct Gate {
    entered: parking_lot::Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    release: parking_lot::Mutex<bool>,
    changed: parking_lot::Condvar,
}

impl Gate {
    fn wait(&self) {
        if let Some(entered) = self.entered.lock().take() {
            entered.send(()).unwrap();
        }
        let mut released = self.release.lock();
        while !*released {
            assert!(
                !self
                    .changed
                    .wait_for(&mut released, Duration::from_secs(10))
                    .timed_out()
            );
        }
    }
    fn release(&self) {
        *self.release.lock() = true;
        self.changed.notify_all();
    }
}

struct Release(Arc<Gate>);
impl Drop for Release {
    fn drop(&mut self) {
        self.0.release();
    }
}

#[test]
fn test_morsel_cancel_deadline_panic_and_reset_keep_worker_leases_until_cleanup() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    for failure in ["cancel", "deadline", "abandon", "panic"] {
        runtime.block_on(stopped_case(&service, failure));
    }
    drop(runtime);
    service.shutdown();
}

async fn stopped_case(service: &TestService, failure: &str) {
    let pool = Pool::new(0);
    let mut operator = operator(false);
    operator
        .runtime
        .initialize_benchmark_pool(pool.clone())
        .unwrap();
    let owner = service.owner("morsel-stop".into());
    let initial = job().with_gather_owner(owner.clone());
    preload(&mut operator, &initial, true).await;
    let token = CancellationToken::new();
    let job = StreamJobContext::new(438, "morsel-stop", JsonMap::new(), None, token.clone())
        .with_gather_owner(owner.clone());
    let context = StreamOperatorContext::new(&job, "match", None);
    let (entered, entering) = tokio::sync::oneshot::channel();
    let gate = Arc::new(Gate {
        entered: parking_lot::Mutex::new(Some(entered)),
        release: parking_lot::Mutex::new(false),
        changed: parking_lot::Condvar::new(),
    });
    let _release = Release(gate.clone());
    let worker_gate = gate.clone();
    let panic = failure == "panic";
    operator.probe_unit_test_hook = Some(Arc::new(move |phase, unit, completed| {
        if phase == data_work::ProbePhase::Fill && unit == 0 && !completed {
            worker_gate.wait();
            assert!(!panic, "morsel test worker panic");
        }
    }));
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    if failure == "deadline" {
        let prepared = operator
            .prepare_batch("left", &input(true, ROWS, false), &context)
            .await
            .unwrap();
        data_work::stop_morsels_at_deadline(&mut operator, prepared, &job, entering).await;
    } else {
        let mut process = Box::pin(operator.process_data(
            "left",
            input(true, ROWS, false),
            &context,
            &mut output,
        ));
        tokio::select! {
            entered = entering => entered.unwrap(),
            result = &mut process => panic!("{failure}: stopped before worker gate: {result:?}"),
        }
        assert!(pool.live.load(Ordering::SeqCst) > 0);
        if failure == "panic" {
            gate.release();
            assert!(matches!(
                process.await,
                Err(CalcFlowError::TaskPanicked { .. })
            ));
        } else {
            if failure == "cancel" {
                token.cancel();
            }
            if failure != "abandon" {
                let result = tokio::time::timeout(Duration::from_secs(5), &mut process)
                    .await
                    .unwrap();
                assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
            }
            drop(process);
        }
    }
    if failure != "panic" {
        assert!(
            pool.live.load(Ordering::SeqCst) > 0,
            "a live worker retains scratch after handler stop"
        );
        operator.reset().unwrap();
        assert!(
            pool.live.load(Ordering::SeqCst) > 0,
            "reset cannot refund live worker scratch"
        );
        gate.release();
    }
    assert!(output.drain("output").is_empty());
    assert_eq!(operator.state.next_left_row_id, 0);
    assert_eq!(operator.state.next_output_sequence, 0);
    drop(operator);
    assert!(owner.close_and_drain().await.is_empty());
    drop(context);
    drop((job, initial, owner));
    assert_eq!(pool.live.load(Ordering::SeqCst), 0);
    assert_eq!(pool.reserved(), 0);
}

struct RejectSecond {
    entered: Option<tokio::sync::oneshot::Sender<()>>,
    release: tokio::sync::oneshot::Receiver<()>,
    accepted: usize,
}

#[async_trait]
impl StreamCollector for RejectSecond {
    async fn emit(&mut self, _: &str, _: Batch) -> Result<()> {
        if self.accepted > 0 {
            return Err(operator_error("match", "morsel downstream failure"));
        }
        self.entered.take().unwrap().send(()).unwrap();
        (&mut self.release).await.unwrap();
        self.accepted += 1;
        Ok(())
    }
}

#[test]
fn test_morsel_result_credit_survives_sink_wait_and_failure_preserves_accepted_sequence() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let pool = Pool::new(0);
        let mut operator = operator(false);
        operator
            .runtime
            .initialize_benchmark_pool(pool.clone())
            .unwrap();
        let job = job().with_gather_owner(service.owner("morsel-sink".into()));
        preload(&mut operator, &job, true).await;
        let context = StreamOperatorContext::new(&job, "match", None);
        let (entered, entering) = tokio::sync::oneshot::channel();
        let (release, released) = tokio::sync::oneshot::channel();
        let mut output = RejectSecond {
            entered: Some(entered),
            release: released,
            accepted: 0,
        };
        let mut process = Box::pin(operator.process_data(
            "left",
            input(true, ROWS, false),
            &context,
            &mut output,
        ));
        tokio::select! {
            entered = entering => entered.unwrap(),
            result = &mut process => panic!("stopped before sink wait: {result:?}"),
        }
        assert!(
            pool.live.load(Ordering::SeqCst) > 0,
            "result backing stays funded through sink acceptance"
        );
        release.send(()).unwrap();
        assert!(
            process
                .await
                .unwrap_err()
                .to_string()
                .contains("downstream failure")
        );
        assert_eq!(output.accepted, 1);
        assert_eq!(operator.state.next_output_sequence, 1);
        assert_eq!(operator.state.next_left_row_id, 0);
        assert_eq!(operator.status().left.retained_rows, 0);
        assert_eq!(pool.live.load(Ordering::SeqCst), 0);
        drop(operator);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

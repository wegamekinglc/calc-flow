use std::sync::atomic::{AtomicUsize, Ordering};

use datafusion::{
    arrow::array::Float64Array,
    execution::memory_pool::{GreedyMemoryPool, MemoryPool},
};

use super::*;
use crate::operator::join::checkpoint_v2::projected_copy::test_support;
use crate::{
    CalcFlowError,
    runtime::streaming::gather_work::{
        TestService,
        admission_probe::{AdmissionProbe, AdmissionStage},
    },
};

#[derive(Default)]
struct Gate {
    released: parking_lot::Mutex<bool>,
    changed: parking_lot::Condvar,
}

impl Gate {
    fn wait(&self) {
        let mut released = self.released.lock();
        while !*released {
            assert!(
                !self
                    .changed
                    .wait_for(&mut released, Duration::from_secs(10))
                    .timed_out(),
                "selected-copy worker was not released"
            );
        }
    }

    fn release(&self) {
        *self.released.lock() = true;
        self.changed.notify_all();
    }
}

struct Release(Arc<Gate>);

impl Drop for Release {
    fn drop(&mut self) {
        self.0.release();
    }
}

pub(super) async fn lifecycle_controls(service: &TestService) {
    for (copied, cancel) in [(false, true), (false, false), (true, true)] {
        Box::pin(stopped(service, copied, cancel)).await;
    }
    for (stage, cancel) in [
        (AdmissionStage::Attempt, false),
        (AdmissionStage::Generation, false),
        (AdmissionStage::Attempt, true),
    ] {
        Box::pin(refused(service, stage, cancel)).await;
    }
}

fn input() -> Batch {
    super::input(
        Arc::new(Float64Array::from(vec![Some(1.5), None, Some(3.5)])),
        Arc::new(Int64Array::from(vec![7, 8, 9])),
    )
}

fn operator(input: &Batch, pool: &Arc<dyn MemoryPool>) -> StreamJoinOperator {
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut operator = super::operator(&schema, true);
    operator
        .runtime
        .initialize_benchmark_pool(Arc::clone(pool))
        .unwrap();
    operator
}

fn job(service: &TestService) -> StreamJobContext {
    StreamJobContext::new(
        1,
        "selected-copy-worker",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
    .with_gather_owner(service.owner("selected-copy-worker".into()))
}

async fn stopped(service: &TestService, copied: bool, cancel: bool) {
    let input = input();
    let original = input.table_payload().unwrap().batches().to_vec();
    let retained = Arc::downgrade(original[0].column(3));
    let unused = Arc::downgrade(original[0].column(0));
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 30));
    let mut operator = operator(&input, &pool);
    let before = operator.status();
    let writer = operator.v2_writer.observation_for_test();
    let job = job(service);
    let context = StreamOperatorContext::new(&job, "selected-state", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let release = Release(Arc::new(Gate::default()));
    let gate = Arc::clone(&release.0);
    let actor = std::thread::current().id();
    let (entered, mut observed) = tokio::sync::mpsc::unbounded_channel();
    let _configuration = test_support::configure(Arc::new(move |phase| {
        assert_ne!(
            std::thread::current().id(),
            actor,
            "selected copy must leave the actor thread"
        );
        if phase == copied {
            entered.send(()).unwrap();
            gate.wait();
        }
    }));
    let mut process = Box::pin(operator.process_data("left", input.clone(), &context, &mut output));
    tokio::select! {
        entered = observed.recv() => assert_eq!(entered, Some(())),
        result = &mut process => panic!("selected copy finished before its worker gate: {result:?}"),
    }
    check_input(&input, &original);
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
    assert_eq!(operator.status(), before);
    assert_eq!(operator.v2_writer.observation_for_test(), writer);
    assert_eq!(
        (
            operator.state.next_left_row_id,
            operator.state.next_output_sequence
        ),
        (0, 0)
    );
    assert!(operator.state.deltas.pending.is_empty());
    assert!(output.drain("output").is_empty());
    assert!(operator.compaction_cleanup.is_some());
    drop((operator, context, input, original, output));
    assert!(
        retained.upgrade().is_some(),
        "paused copy lost its selected input"
    );
    assert!(
        unused.upgrade().is_none(),
        "copy worker retained an omitted full parent"
    );
    assert!(pool.reserved() > 0);
    let mut drain = Box::pin(job.gather_owner().close_and_drain());
    assert!(futures::poll!(drain.as_mut()).is_pending());
    release.0.release();
    assert!(drain.await.is_empty());
    assert!(retained.upgrade().is_none());
    drop(job);
    assert_eq!(pool.reserved(), 0);
    assert!(
        responsive,
        "cancellation must return before the copy worker is released"
    );
}

async fn refused(service: &TestService, stage: AdmissionStage, cancel: bool) {
    let input = input();
    let original = input.table_payload().unwrap().batches().to_vec();
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 30));
    let mut operator = operator(&input, &pool);
    let before = operator.status();
    let writer = operator.v2_writer.observation_for_test();
    let job = job(service);
    let context = StreamOperatorContext::new(&job, "selected-state", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let _configuration = test_support::configure(Arc::new(move |_| {
        observed.fetch_add(1, Ordering::SeqCst);
    }));
    let probe = AdmissionProbe::install(job.gather_owner(), stage, Arc::clone(&pool), 1 << 30);
    if cancel {
        probe.cancel_on_refusal(job.cancellation().clone());
    }
    let error = operator
        .process_data("left", input.clone(), &context, &mut output)
        .await
        .unwrap_err();
    check_refusal(&error, cancel);
    let event = probe
        .take_event()
        .expect("actual copy admission was attempted");
    assert_eq!(event.stage, stage);
    assert_eq!(event.operator, "selected-state");
    assert_eq!(event.available + 1, event.fee);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(operator.status(), before);
    assert_eq!(operator.v2_writer.observation_for_test(), writer);
    assert_eq!(
        (
            operator.state.next_left_row_id,
            operator.state.next_output_sequence
        ),
        (0, 0)
    );
    assert!(output.drain("output").is_empty());
    check_input(&input, &original);
    if !cancel {
        operator
            .process_data("left", input.clone(), &context, &mut output)
            .await
            .unwrap();
        assert_eq!(operator.state.left.len(), 3);
        assert!(calls.load(Ordering::SeqCst) > 0);
        assert!(operator.compaction_cleanup.is_none());
        check_input(&input, &original);
    }
    drop((operator, context, output, original, probe));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

fn check_refusal(error: &CalcFlowError, cancel: bool) {
    if cancel {
        assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    } else {
        let CalcFlowError::DataFusion { node_id, message } = error else {
            panic!("unexpected admission error: {error}")
        };
        assert_eq!(node_id.as_deref(), Some("selected-state"));
        assert!(message.contains("Resources exhausted:"), "{message}");
        assert!(
            message.contains("sql-incremental:stream-join-native"),
            "{message}"
        );
    }
}

fn check_input(input: &Batch, original: &[RecordBatch]) {
    assert_eq!(input.table_payload().unwrap().batches(), original);
    let values = input
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| {
            record
                .column(3)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .iter()
        })
        .collect::<Vec<_>>();
    assert_eq!(values, [Some(1.5), None, Some(3.5)]);
}

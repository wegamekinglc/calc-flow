use super::*;
use crate::runtime::streaming::gather_work::TestService;

#[path = "column_gather_admission_tests.rs"]
mod admission;
#[path = "column_gather_budget_tests.rs"]
mod budget;
#[path = "column_gather_lifecycle_tests.rs"]
mod lifecycle;
#[path = "column_gather_output_tests.rs"]
mod output;

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
                "column worker was not released"
            );
        }
    }

    fn release(&self) {
        *self.released.lock() = true;
        self.changed.notify_all();
    }
}

struct ReleaseGates(Vec<Arc<Gate>>);

impl Drop for ReleaseGates {
    fn drop(&mut self) {
        for gate in &self.0 {
            gate.release();
        }
    }
}

#[test]
fn test_join_column_gather_runs_concurrently_without_blocking_tokio() {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(assert_concurrent_columns(&service));
    drop(runtime);
    service.shutdown();
}

async fn assert_concurrent_columns(service: &TestService) {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job().with_gather_owner(service.owner("join-columns".into()));
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
    let gates = ReleaseGates(vec![Arc::new(Gate::default()), Arc::new(Gate::default())]);
    let worker_gates = gates.0.clone();
    let (observed, mut events) = tokio::sync::mpsc::unbounded_channel();
    let actor = std::thread::current().id();
    let _configuration = output_gather::test_support::configure(
        true,
        Some(Arc::new(move |ordinal, completed| {
            assert_ne!(
                std::thread::current().id(),
                actor,
                "Join column gather must leave the Tokio actor thread"
            );
            if ordinal < 2 {
                observed
                    .send((ordinal, completed, std::thread::current().id()))
                    .unwrap();
                if !completed {
                    worker_gates[ordinal].wait();
                }
            }
        })),
    );
    assert!(output_gather::test_support::force());
    let mut process =
        Box::pin(operator.process_data("left", left_batch(vec![1, 2]), &context, &mut collector));
    let mut entered = Vec::new();
    for _ in 0..2 {
        let event = tokio::select! {
            event = events.recv() => event.unwrap(),
            result = &mut process => panic!("Join finished before concurrent column entry: {result:?}"),
        };
        assert!(!event.1);
        assert_ne!(event.2, actor);
        entered.push(event);
    }
    assert_ne!(entered[0].0, entered[1].0);
    assert_ne!(entered[0].2, entered[1].2);
    tokio::select! {
        () = tokio::time::sleep(Duration::from_millis(1)) => {},
        result = &mut process => panic!("Join finished before worker release: {result:?}"),
    }
    gates.0[1].release();
    let completed = tokio::select! {
        event = events.recv() => event.unwrap(),
        result = &mut process => panic!("Join finished while column zero was gated: {result:?}"),
    };
    assert_eq!((completed.0, completed.1), (1, true));
    gates.0[0].release();
    process.await.unwrap();
    assert_literal_output(&mut collector, &operator);
    assert!(operator.compaction_cleanup.is_none());
    drop(operator);
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

fn assert_literal_output(collector: &mut EdgeCollector, operator: &StreamJoinOperator) {
    let output = collector.drain("output");
    assert_eq!(output.len(), 1);
    let batch = output[0].as_data().unwrap();
    assert_eq!(batch.metadata().sequence(), 0);
    let record = &batch.table_payload().unwrap().batches()[0];
    assert_eq!(
        record.schema_ref(),
        operator.output_ports[0].schema().unwrap()
    );
    assert_eq!(
        record.column(0).to_data(),
        Int64Array::from(vec![7, 7]).to_data()
    );
    assert_eq!(
        record.column(2).to_data(),
        Int64Array::from(vec![42, 42]).to_data()
    );
    assert_eq!(
        record.column(5).to_data(),
        StringArray::from(vec!["paid", "paid"]).to_data()
    );
}

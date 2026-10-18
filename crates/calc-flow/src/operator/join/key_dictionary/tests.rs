mod edge_cases;

use super::*;
use crate::runtime::streaming::gather_work::{
    TestService,
    admission_probe::{AdmissionProbe, AdmissionStage},
};
use crate::{CancellationToken, EventTime, JsonMap, StreamJobContext, StreamOperatorContext};
use datafusion::arrow::{
    array::StringArray,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
use std::{
    cell::RefCell,
    sync::{Condvar, Mutex},
    thread::ThreadId,
};

pub(super) type WorkerHook = Arc<dyn Fn(usize) + Send + Sync>;

thread_local! {
    static WORKER_HOOK: RefCell<Option<WorkerHook>> = const { RefCell::new(None) };
}

pub(super) fn worker_hook() -> Option<WorkerHook> {
    WORKER_HOOK.with(|hook| hook.borrow().clone())
}

struct Observe;

impl Observe {
    fn new(hook: WorkerHook) -> Self {
        WORKER_HOOK.with(|current| *current.borrow_mut() = Some(hook));
        Self
    }
}

impl Drop for Observe {
    fn drop(&mut self) {
        WORKER_HOOK.with(|hook| hook.borrow_mut().take());
    }
}

fn rows() -> Vec<super::super::AdmittedRow> {
    let count = 64_000;
    let record = Arc::new(
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("key", DataType::Utf8, false)])),
            vec![Arc::new(StringArray::from_iter_values(
                (0..count).map(|row| format!("symbol-{:02}", row % 64)),
            ))],
        )
        .unwrap(),
    );
    (0..count)
        .map(|row| super::super::AdmittedRow {
            record: super::super::columnar::RowPayload::Rowed {
                parent: Arc::clone(&record),
                row,
            },
            event_time: EventTime::from_micros(0),
            row_id: u64::try_from(row).unwrap(),
            // Keys first appear as probe-only rows, cross worker boundaries,
            // then birth in a different order from their first appearances.
            retain: row >= 32_017 && row % 3 != 0,
        })
        .collect()
}

fn snapshot(keys: &PreparedKeys) -> (Vec<u32>, Vec<Option<Vec<u8>>>) {
    (
        keys.ids.clone(),
        keys.keys
            .iter()
            .map(|key| key.as_ref().map(|key| key.as_slice().to_vec()))
            .collect(),
    )
}

fn commit(dictionary: &mut Dictionary, prepared: PreparedKeys) {
    dictionary.commit(prepared.dictionary);
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn job(service: &TestService) -> StreamJobContext {
    StreamJobContext::new(
        991,
        "parallel-keys",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
    .with_gather_owner(service.owner("parallel-keys".into()))
}

#[test]
fn batch_parallel_keys_dispatch_preserves_first_retained_identity_order() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 30));
        let master = MemoryConsumer::new("parallel-key-test").register(&pool);
        let rows = rows();
        let mut serial = Dictionary::default();
        let serial_keys = serial
            .prepare(&rows, &[0], master.new_empty(), None, "parallel-keys")
            .await
            .unwrap()
            .unwrap();
        let expected = snapshot(&serial_keys);
        assert!(expected.0[..32_017].iter().all(|&id| id == u32::MAX));
        commit(&mut serial, serial_keys);
        let job = job(&service);
        let context = StreamOperatorContext::new(&job, "parallel-keys", None);
        let workers = Arc::new((Mutex::new(Vec::<(usize, ThreadId)>::new()), Condvar::new()));
        let observed = Arc::clone(&workers);
        let _observe = Observe::new(Arc::new(move |ordinal| {
            let mut workers = observed.0.lock().unwrap();
            workers.push((ordinal, std::thread::current().id()));
            observed.1.notify_all();
            let (_workers, timeout) = observed
                .1
                .wait_timeout_while(workers, std::time::Duration::from_secs(5), |workers| {
                    workers.len() < 4
                })
                .unwrap();
            assert!(!timeout.timed_out(), "four key work units must overlap");
        }));
        let mut parallel = Dictionary::default();
        let keys = parallel
            .prepare(
                &rows,
                &[0],
                master.new_empty(),
                Some(&context),
                "parallel-keys",
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(snapshot(&keys), expected);
        commit(&mut parallel, keys);
        assert_eq!(parallel.inventory(), serial.inventory());
        {
            let workers = workers.0.lock().unwrap();
            assert_eq!(
                workers.len(),
                4,
                "64k keys must execute four native work units"
            );
            assert!(
                workers
                    .iter()
                    .all(|(_, thread)| *thread != std::thread::current().id())
            );
        }
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop((serial, parallel, master, job));
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn batch_parallel_keys_admission_refusal_falls_back_without_identity_drift() {
    let service = TestService::new(2, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 30));
        let master = MemoryConsumer::new("parallel-key-test").register(&pool);
        let rows = rows();
        let expected = Dictionary::default()
            .prepare(&rows, &[0], master.new_empty(), None, "parallel-keys")
            .await
            .unwrap()
            .unwrap();
        let job = job(&service);
        let context = StreamOperatorContext::new(&job, "parallel-keys", None);
        let probe = AdmissionProbe::install(
            job.gather_owner(),
            AdmissionStage::Attempt,
            Arc::clone(&pool),
            1 << 30,
        );
        let mut dictionary = Dictionary::default();
        let actual = dictionary
            .prepare(
                &rows,
                &[0],
                master.new_empty(),
                Some(&context),
                "parallel-keys",
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(snapshot(&actual), snapshot(&expected));
        assert!(
            probe.take_event().is_some(),
            "production key preparation must attempt native admission"
        );
        assert_eq!(dictionary.inventory(), (0, 0, 0, false));
        drop((actual, expected));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop((dictionary, master, job));
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn batch_parallel_keys_cancelled_admission_does_not_install_pending_births() {
    let service = TestService::new(2, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 30));
        let master = MemoryConsumer::new("parallel-key-test").register(&pool);
        let rows = rows();
        let job = job(&service);
        let context = StreamOperatorContext::new(&job, "parallel-keys", None);
        let probe = AdmissionProbe::install(
            job.gather_owner(),
            AdmissionStage::Attempt,
            Arc::clone(&pool),
            1 << 30,
        );
        probe.cancel_on_refusal(job.cancellation().clone());
        let mut dictionary = Dictionary::default();
        let result = dictionary
            .prepare(
                &rows,
                &[0],
                master.new_empty(),
                Some(&context),
                "parallel-keys",
            )
            .await;
        assert!(matches!(
            result,
            Err(crate::CalcFlowError::Cancelled { .. })
        ));
        assert!(probe.take_event().is_some());
        assert_eq!(dictionary.inventory(), (0, 0, 0, false));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop((result, dictionary, master, job));
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn batch_parallel_keys_cancelled_worker_retains_source_until_job_drain() {
    struct Release(std::sync::mpsc::Sender<()>);
    impl Drop for Release {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }

    let service = TestService::new(2, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 30));
        let master = MemoryConsumer::new("parallel-key-test").register(&pool);
        let rows = rows();
        let source = match &rows[0].record {
            super::super::columnar::RowPayload::Rowed { parent, .. } => Arc::downgrade(parent),
            _ => unreachable!(),
        };
        let job = job(&service);
        let context = StreamOperatorContext::new(&job, "parallel-keys", None);
        let (entered, started) = tokio::sync::oneshot::channel();
        let (release, wait) = std::sync::mpsc::channel();
        let release = Release(release);
        let gate = Mutex::new((Some(entered), wait));
        let _observe = Observe::new(Arc::new(move |unit| {
            if unit == 0 {
                let mut gate = gate.lock().unwrap();
                gate.0.take().unwrap().send(()).unwrap();
                gate.1
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
            }
        }));
        let mut dictionary = Dictionary::default();
        let mut prepare = Box::pin(dictionary.prepare(
            &rows,
            &[0],
            master.new_empty(),
            Some(&context),
            "parallel-keys",
        ));
        tokio::select! {
            result = started => result.unwrap(),
            _ = &mut prepare => panic!("key preparation must enter owned worker before completing"),
        }
        job.cancellation().cancel();
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), &mut prepare)
            .await
            .unwrap();
        assert!(matches!(
            result,
            Err(crate::CalcFlowError::Cancelled { .. })
        ));
        drop(prepare);
        assert_eq!(dictionary.inventory(), (0, 0, 0, false));
        drop(rows);
        assert!(
            source.upgrade().is_some(),
            "running key work must own its source"
        );
        assert!(pool.reserved() > 0);
        drop(release);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        assert!(source.upgrade().is_none());
        drop((result, dictionary, master, job));
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

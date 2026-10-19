use super::*;
use crate::{
    Batch, BatchMetadata, IngressProgress, IngressProgressSnapshot, IngressState, JoinStateLimits,
    JoinTimeBounds, StreamJoinOperator, StreamJoinSpec,
};
use datafusion::arrow::{array::TimestampMicrosecondArray, datatypes::TimeUnit};
use std::sync::atomic::{AtomicUsize, Ordering};

fn sparse_source() -> Batch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, true),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
    ]));
    let count = 72_013_usize;
    let keys = (0..count)
        .map(|row| (!row.is_multiple_of(97)).then(|| format!("symbol-{:02}", row % 64)))
        .collect::<StringArray>();
    let times = (0..count)
        .map(|row| {
            (!row.is_multiple_of(89)).then(|| {
                if row.is_multiple_of(83) {
                    -1
                } else {
                    i64::try_from(row).unwrap()
                }
            })
        })
        .collect::<TimestampMicrosecondArray>();
    let parent = RecordBatch::try_new(schema, vec![Arc::new(keys), Arc::new(times)]).unwrap();
    Batch::table(vec![parent.slice(13, 72_000)], BatchMetadata::default()).unwrap()
}

fn sparse_operator(source: &Batch, pool: &Arc<dyn MemoryPool>) -> StreamJoinOperator {
    let schema = source.table_payload().unwrap().batches()[0].schema();
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "time",
        "time",
        JoinTimeBounds::new(std::time::Duration::ZERO, std::time::Duration::ZERO).unwrap(),
        JoinStateLimits::new(100_000, 100_000_000, 1_000_000).unwrap(),
    )
    .unwrap();
    let mut operator =
        StreamJoinOperator::new("parallel-keys", Arc::clone(&schema), schema, spec).unwrap();
    operator
        .runtime
        .initialize_benchmark_pool(Arc::clone(pool))
        .unwrap();
    operator
}

fn sparse_context(job: &StreamJobContext, ended: bool) -> StreamOperatorContext<'_> {
    let progress = IngressProgressSnapshot::new(std::collections::BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(0))),
        ),
        (
            "right".into(),
            IngressProgress::new(
                if ended {
                    IngressState::Ended
                } else {
                    IngressState::Active
                },
                Some(EventTime::from_micros(36_000)),
            ),
        ),
    ]));
    StreamOperatorContext::with_ingress_progress(job, "parallel-keys", None, progress)
}

#[test]
fn batch_parallel_keys_actual_sparse_sliced_admission_preserves_retention() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 30));
        let master = MemoryConsumer::new("parallel-key-test").register(&pool);
        let source = sparse_source();
        for ended in [false, true] {
            let mut operator = sparse_operator(&source, &pool);
            let job = job(&service);
            let context = sparse_context(&job, ended);
            let plan = operator.begin_batch("left", &source).unwrap();
            let mut bundle = operator.admission_bundle(&plan, &source).unwrap();
            operator
                .admit_batch_records("left", &source, &plan, &context, &mut bundle)
                .await
                .unwrap();
            bundle.finish("parallel-keys").unwrap();
            assert!(bundle.admitted.len() >= 64_000);
            assert_eq!(bundle.admitted.record_count(), 1);
            assert!(bundle.admitted.sparse_index_len() > 0);
            let admitted = &bundle.admitted as &dyn AdmittedRows;
            let retained = admitted.iter().filter(|row| row.retain).count();
            assert_eq!(retained == 0, ended);
            assert!(retained < admitted.len());
            let expected = Dictionary::default()
                .prepare(admitted, &[0], master.new_empty(), None, "parallel-keys")
                .await
                .unwrap()
                .unwrap();
            let units = Arc::new(AtomicUsize::new(0));
            let observed = Arc::clone(&units);
            let _observe = Observe::new(Arc::new(move |_| {
                observed.fetch_add(1, Ordering::SeqCst);
            }));
            let mut dictionary = Dictionary::default();
            let actual = dictionary
                .prepare(
                    admitted,
                    &[0],
                    master.new_empty(),
                    Some(&context),
                    "parallel-keys",
                )
                .await
                .unwrap()
                .unwrap();
            assert_eq!(snapshot(&actual), snapshot(&expected));
            assert_eq!(units.load(Ordering::SeqCst), 4);
            if ended {
                assert!(actual.ids.iter().all(|&id| id == u32::MAX));
                assert!(actual.keys.is_empty());
            } else {
                let first = admitted.iter().position(|row| row.retain).unwrap();
                assert!(actual.ids[..first].iter().all(|&id| id == u32::MAX));
            }
            drop((actual, expected, bundle, dictionary, operator));
            assert!(job.gather_owner().close_and_drain().await.is_empty());
            drop(job);
        }
        drop(master);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn batch_parallel_keys_oversized_value_declines_with_serial_identity_parity() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 30));
        let master = MemoryConsumer::new("parallel-key-test").register(&pool);
        let mut rows = rows();
        let parent = Arc::new(
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("key", DataType::Utf8, false)])),
                vec![Arc::new(StringArray::from_iter_values(
                    (0..rows.len()).map(|row| {
                        if row == 32_020 {
                            "x".repeat(super::super::super::columnar::COPY_BYTES + 1)
                        } else {
                            format!("symbol-{:02}", row % 64)
                        }
                    }),
                ))],
            )
            .unwrap(),
        );
        for (row, admitted) in rows.iter_mut().enumerate() {
            admitted.record = super::super::super::columnar::RowPayload::Rowed {
                parent: Arc::clone(&parent),
                row,
            };
        }
        let expected = Dictionary::default()
            .prepare(&rows, &[0], master.new_empty(), None, "parallel-keys")
            .await
            .unwrap()
            .unwrap();
        let job = job(&service);
        let context = StreamOperatorContext::new(&job, "parallel-keys", None);
        let units = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&units);
        let _observe = Observe::new(Arc::new(move |_| {
            observed.fetch_add(1, Ordering::SeqCst);
        }));
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
        assert_eq!(units.load(Ordering::SeqCst), 4);
        assert_eq!(dictionary.inventory(), (0, 0, 0, false));
        drop((actual, expected, dictionary));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop((master, job));
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn batch_parallel_keys_single_worker_capacity_keeps_serial_route() {
    let service = TestService::new(1, 1).unwrap();
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
        let _observe = Observe::new(Arc::new(|_| panic!("single worker must use serial keys")));
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
        assert!(probe.take_event().is_none());
        drop((actual, expected, dictionary));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop((master, job));
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn batch_parallel_keys_dropped_future_retains_source_until_job_drain() {
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
            super::super::super::columnar::RowPayload::Rowed { parent, .. } => {
                Arc::downgrade(parent)
            }
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
        drop(prepare);
        assert!(!job.cancellation().is_cancelled());
        assert_eq!(dictionary.inventory(), (0, 0, 0, false));
        drop(rows);
        assert!(source.upgrade().is_some());
        assert!(pool.reserved() > 0);
        drop(release);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        assert!(source.upgrade().is_none());
        drop((dictionary, master, job));
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

fn cpu_budget_key_case(cpu_budget: usize, count: usize) {
    let service = TestService::new(cpu_budget, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 30));
        let master = MemoryConsumer::new("cpu-budget-key-test").register(&pool);
        let template = rows();
        let rows = (0..count)
            .map(|row| super::super::super::AdmittedRow {
                record: template[row % template.len()].record.clone(),
                event_time: EventTime::from_micros(0),
                row_id: u64::try_from(row).unwrap(),
                retain: row >= count / 2 + 17 && !row.is_multiple_of(3),
            })
            .collect::<Vec<_>>();
        let expected = Dictionary::default()
            .prepare(&rows, &[0], master.new_empty(), None, "parallel-keys")
            .await
            .unwrap()
            .unwrap();
        let job = job(&service);
        let context = StreamOperatorContext::new(&job, "parallel-keys", None);
        let units = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&units);
        let _observe = Observe::new(Arc::new(move |unit| {
            observed.lock().unwrap().push(unit);
        }));
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
            actual.ids[..count / 2 + 17]
                .iter()
                .all(|&id| id == u32::MAX)
        );
        {
            let mut units = units.lock().unwrap();
            units.sort_unstable();
            assert_eq!(*units, (0..cpu_budget).collect::<Vec<_>>());
        }
        drop((actual, expected, dictionary));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop((job, master));
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn cpu_budget_key_eight_workers_preserve_first_retained_identity() {
    cpu_budget_key_case(8, 128_000);
}

#[test]
fn cpu_budget_key_sixteen_workers_preserve_first_retained_identity() {
    cpu_budget_key_case(16, 256_000);
}

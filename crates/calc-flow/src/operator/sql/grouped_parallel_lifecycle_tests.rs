use super::*;
use crate::operator::{OperatorMetadata, StreamOperator};
use crate::{EdgeCollector, Epoch, SqlOperator};

type SavedGroups = Vec<(Arc<[u8]>, Vec<Vec<ScalarValue>>, Vec<ScalarValue>)>;

fn committed(plan: &IncrementalSql) -> SavedGroups {
    plan.groups
        .iter()
        .map(|group| {
            (
                group.key.clone(),
                group.states.clone(),
                group.results.clone(),
            )
        })
        .collect()
}

fn registration(name: &str, fault: Fault) -> (Registration, mpsc::Receiver<usize>) {
    let (entered, receiver) = mpsc::channel();
    let (completed, _) = mpsc::channel();
    let probe = Arc::new(Probe {
        entered,
        completed,
        gates: std::array::from_fn(|_| Gate::default()),
        active: AtomicUsize::new(0),
        maximum: AtomicUsize::new(0),
        fault: Mutex::new(Some(fault)),
        merge_calls: AtomicUsize::new(0),
    });
    PROBES.lock().unwrap().insert(name.into(), probe.clone());
    (
        Registration {
            name: name.into(),
            probe,
        },
        receiver,
    )
}

struct RunControl {
    token: CancellationToken,
    clock: tokio::sync::watch::Sender<chrono::DateTime<chrono::Utc>>,
    abort: tokio::sync::oneshot::Sender<()>,
    deadline: chrono::DateTime<chrono::Utc>,
}

async fn retry(plan: &mut IncrementalSql, input: &Batch, service: &TestService) {
    let job = StreamJobContext::new(
        948,
        "parallel-retry",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
    .with_gather_owner(service.owner("parallel-retry".into()));
    let context = StreamOperatorContext::new(&job, "parallel-retry", None);
    let transaction = plan
        .update(input, &context, "parallel-retry")
        .await
        .unwrap();
    for row in snapshot(&transaction.records) {
        assert_eq!(row[1], ScalarValue::Int64(Some(2049)));
    }
    plan.commit(transaction);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}

async fn interrupted_update(fault: Fault, service: &TestService, ready: mpsc::Sender<RunControl>) {
    let name = format!("parallel-fault-{fault:?}");
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let input = source(4 * CHUNK_ROWS);
    let query = crate::expression::parse_select_query(
        "SELECT key, SUM(value) AS total FROM events GROUP BY key",
    )
    .unwrap();
    let mut plan = IncrementalSql::plan(
        &runtime,
        &query,
        "events",
        input.table_payload().unwrap().schema().clone(),
        &name,
    )
    .await
    .unwrap()
    .unwrap();
    let now = chrono::Utc::now();
    let deadline = now + chrono::Duration::days(1);
    let (clock, receiver) = tokio::sync::watch::channel(now);
    let token = CancellationToken::new();
    let job = StreamJobContext::new(947, &name, JsonMap::new(), Some(deadline), token.clone())
        .with_deadline_clock_for_test(receiver)
        .with_gather_owner(service.owner(name.clone().into()));
    let context = StreamOperatorContext::new(&job, &name, None);
    let seed = plan.update(&source(16), &context, &name).await.unwrap();
    plan.commit(seed);
    let saved = committed(&plan);
    let (abort, dropped) = tokio::sync::oneshot::channel();
    ready
        .send(RunControl {
            token,
            clock,
            abort,
            deadline,
        })
        .unwrap();
    let result = tokio::select! {
        biased;
        _ = dropped => None,
        result = plan.update(&input, &context, &name) => Some(result),
    };
    if fault == Fault::Abandon {
        assert!(result.is_none());
    } else {
        assert!(result.unwrap().is_err());
    }
    assert_eq!(committed(&plan), saved);
    let failures = job.gather_owner().close_and_drain().await;
    assert!(failures.is_empty() || fault == Fault::Panic);
    drop(context);
    drop(job);
    retry(&mut plan, &input, service).await;
    drop(plan);
    assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
}

fn failure_case(fault: Fault) {
    let name = format!("parallel-fault-{fault:?}");
    let (registration, entered) = registration(&name, fault);
    let (ready, control) = mpsc::channel();
    let runner = std::thread::spawn(move || {
        let service = TestService::new(2, 1).unwrap();
        let executor = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        executor.block_on(interrupted_update(fault, &service, ready));
        drop(executor);
        service.shutdown();
    });
    let control = control.recv_timeout(Duration::from_secs(10)).unwrap();
    let started = (0..2)
        .filter_map(|_| entered.recv_timeout(Duration::from_secs(2)).ok())
        .count();
    match fault {
        Fault::Cancel => control.token.cancel(),
        Fault::Deadline => control.clock.send(control.deadline).unwrap(),
        Fault::Abandon => {
            let _ = control.abort.send(());
        }
        _ => {}
    }
    for gate in &registration.probe.gates {
        gate.release();
    }
    runner.join().unwrap();
    assert_eq!(started, 2);
}

#[test]
fn test_grouped_parallel_worker_error_and_panic_preserve_state_and_retry_once() {
    for fault in [Fault::Error, Fault::Panic, Fault::Merge] {
        failure_case(fault);
    }
}

#[test]
fn test_grouped_parallel_cancellation_deadline_and_future_drop_settle_owned_work() {
    for fault in [Fault::Cancel, Fault::Deadline, Fault::Abandon] {
        failure_case(fault);
    }
}

struct Reject;

#[async_trait::async_trait]
impl crate::StreamCollector for Reject {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        Err(df_error("parallel-output", "injected output refusal"))
    }
}

#[test]
fn test_grouped_parallel_output_refusal_checkpoint_restore_and_cpu_change() {
    let service = TestService::new(2, 1).unwrap();
    let one_cpu = TestService::new(1, 1).unwrap();
    let executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    executor.block_on(async {
        let name = "parallel-output";
        let job = StreamJobContext::new(949, name, JsonMap::new(), None, CancellationToken::new())
            .with_gather_owner(service.owner(name.into()));
        let context = StreamOperatorContext::new(&job, name, None);
        let mut operator =
            SqlOperator::new(name, PREFIX_SQL, vec!["events".into()], vec![]).unwrap();
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        let first = prefix_input(false, 0);
        let second = prefix_input(false, 1);
        operator
            .process_data("events", first.clone(), &context, &mut output)
            .await
            .unwrap();
        let saved = operator.checkpoint(Epoch::INITIAL).unwrap();
        assert!(
            operator
                .process_data("events", second.clone(), &context, &mut Reject)
                .await
                .is_err()
        );
        let refused = operator.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(refused.inline_metadata, saved.inline_metadata);
        assert_eq!(refused.segments, saved.segments);
        operator
            .process_data("events", second, &context, &mut output)
            .await
            .unwrap();
        let after = operator.checkpoint(Epoch::INITIAL).unwrap();
        let mut expected = Vec::new();
        extend_oracle(&mut expected, &first);
        extend_oracle(&mut expected, &prefix_input(false, 1));
        let messages = output.drain("output");
        assert_eq!(
            snapshot(
                messages[1]
                    .as_data()
                    .unwrap()
                    .table_payload()
                    .unwrap()
                    .batches()
            ),
            oracle_snapshot(&expected, false)
        );
        operator.reset().unwrap();
        operator.restore(&after).unwrap();
        let resumed =
            StreamJobContext::new(950, name, JsonMap::new(), None, CancellationToken::new())
                .with_gather_owner(one_cpu.owner("parallel-resumed".into()));
        let resumed_context = StreamOperatorContext::new(&resumed, name, None);
        let third = prefix_input(false, 2);
        operator
            .process_data("events", third.clone(), &resumed_context, &mut output)
            .await
            .unwrap();
        extend_oracle(&mut expected, &third);
        let messages = output.drain("output");
        assert_eq!(
            snapshot(
                messages[0]
                    .as_data()
                    .unwrap()
                    .table_payload()
                    .unwrap()
                    .batches()
            ),
            oracle_snapshot(&expected, false)
        );
        drop(operator);
        assert!(resumed.gather_owner().close_and_drain().await.is_empty());
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(resumed_context);
        drop(resumed);
    });
    drop(executor);
    one_cpu.shutdown();
    service.shutdown();
}

#[test]
fn test_grouped_parallel_global_group_limit_and_closed_launch_preserve_state() {
    let service = TestService::new(2, 1).unwrap();
    let executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    executor.block_on(async {
        let input = source(4 * CHUNK_ROWS);
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let query = crate::expression::parse_select_query(
            "SELECT key, SUM(value) AS total FROM events GROUP BY key",
        )
        .unwrap();
        let mut plan = IncrementalSql::plan(
            &runtime,
            &query,
            "events",
            input.table_payload().unwrap().schema().clone(),
            "parallel-limit",
        )
        .await
        .unwrap()
        .unwrap();
        plan.set_compact_budget(
            Some(crate::StateBudget::new(16, 1 << 30).unwrap()),
            "parallel-limit",
        )
        .unwrap();
        let job = StreamJobContext::new(
            951,
            "parallel-limit",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        )
        .with_gather_owner(service.owner("parallel-limit".into()));
        let context = StreamOperatorContext::new(&job, "parallel-limit", None);
        let first = plan
            .update(&input, &context, "parallel-limit")
            .await
            .unwrap();
        plan.commit(first);
        let saved = committed(&plan);
        let record = input.table_payload().unwrap().batches()[0].clone();
        let keys = StringArray::from_iter_values((0..4 * CHUNK_ROWS).map(|row| {
            if row == 0 {
                "new-key".into()
            } else {
                format!("key-{}", row % 16)
            }
        }));
        let new = Batch::table(
            vec![
                RecordBatch::try_new(
                    record.schema(),
                    vec![Arc::new(keys), record.column(1).clone()],
                )
                .unwrap(),
            ],
            crate::BatchMetadata::default(),
        )
        .unwrap();
        assert!(plan.update(&new, &context, "parallel-limit").await.is_err());
        assert_eq!(committed(&plan), saved);
        let closed = StreamJobContext::new(
            952,
            "parallel-closed",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        )
        .with_gather_owner(service.owner("parallel-closed".into()));
        closed.gather_owner().close_admission();
        let closed_context = StreamOperatorContext::new(&closed, "parallel-closed", None);
        assert!(
            plan.update(&input, &closed_context, "parallel-closed")
                .await
                .is_err()
        );
        assert_eq!(committed(&plan), saved);
        let replay = plan
            .update(&input, &context, "parallel-limit")
            .await
            .unwrap();
        for row in snapshot(&replay.records) {
            assert_eq!(row[1], ScalarValue::Int64(Some(4096)));
        }
        drop((replay, plan));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        assert!(closed.gather_owner().close_and_drain().await.is_empty());
    });
    drop(executor);
    service.shutdown();
}

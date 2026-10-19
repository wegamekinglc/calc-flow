use super::super::probe::tests::{assert_output, fixture_with_keys, input, oracle};
use super::*;
use crate::{
    CancellationToken, EdgeBudget, EdgeCollector, JsonMap, OperatorMetadata, StreamJobContext,
    StreamOperator, runtime::streaming::gather_work::TestService,
};
use datafusion::execution::memory_pool::MemoryConsumer;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

async fn admitted(
    job: &StreamJobContext,
    fragmented: bool,
) -> (StreamAsofJoinOperator, EdgeCollector) {
    let (mut operator, left, _) = fixture_with_keys(true, 32_768, 1);
    let right = left.iter().step_by(2).cloned().collect::<Vec<_>>();
    let context = StreamOperatorContext::new(job, "asof", None)
        .with_output_budget(EdgeBudget::new(32_768, 128 << 20).unwrap());
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    for (port, rows) in [("right", &right), ("left", &left)] {
        let batch = input(&operator, rows);
        let batch = if fragmented && port == "left" {
            crate::Batch::table(
                batch
                    .table_payload()
                    .unwrap()
                    .batches()
                    .iter()
                    .flat_map(|record| {
                        (0..record.num_rows())
                            .step_by(16)
                            .map(|offset| record.slice(offset, 16))
                    })
                    .collect(),
                crate::BatchMetadata::default(),
            )
            .unwrap()
        } else {
            batch
        };
        operator
            .process_data(port, batch, &context, &mut output)
            .await
            .unwrap();
    }
    (operator, output)
}

#[tokio::test]
async fn cpu_pipeline_fragment_snapshot_yields_during_many_run_capture() {
    let job = StreamJobContext::new(
        918,
        "fragment-capture",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let (operator, output) = admitted(&job, true).await;
    let context = StreamOperatorContext::new(&job, "asof", None);
    let prefix = Arc::new(operator.reserve_workspace(32_768 * 640 + 2_048).unwrap());
    let ran = Arc::new(AtomicBool::new(false));
    let observed = ran.clone();
    let task = tokio::spawn(async move {
        observed.store(true, Ordering::SeqCst);
    });
    let work = capture(&operator, 32_768, 2, prefix, &context)
        .await
        .unwrap()
        .unwrap();
    let yielded = ran.load(Ordering::SeqCst);
    drop(work);
    task.await.unwrap();
    drop((operator, output, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    assert!(
        yielded,
        "snapshot capture blocked the owner through all fragmented runs"
    );
}

#[derive(Default)]
struct Gate {
    open: parking_lot::Mutex<bool>,
    changed: parking_lot::Condvar,
}
impl Gate {
    fn wait(&self) {
        let mut open = self.open.lock();
        while !*open {
            self.changed.wait(&mut open);
        }
    }
    fn release(&self) {
        *self.open.lock() = true;
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
fn cpu_pipeline_cancelled_fragments_keep_snapshot_and_prefix_funded_until_exit() {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let job = StreamJobContext::new(919, "fragment-cancel", JsonMap::new(), None, CancellationToken::new()).with_gather_owner(service.owner("fragment-cancel".into()));
        let (mut operator, output) = admitted(&job, false).await;
        let pool = operator.runtime.pool.clone();
        let source = Arc::downgrade(operator.state.batches.values().next().unwrap().0.record.column(0));
        let context = StreamOperatorContext::new(&job, "asof", None);
        let prefix = Arc::new(operator.reserve_workspace(32_768 * 640 + 2_048).unwrap());
        let prefix_weak = Arc::downgrade(&prefix);
        let gate = Arc::new(Gate::default());
        let release = Release(gate.clone());
        let (entered, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        operator.match_hook = Some(Arc::new(move |ordinal| { entered.send(ordinal).unwrap(); gate.wait(); }));
        let mut pending = Box::pin(prepare(&operator, 32_768, prefix, &context));
        for _ in 0..2 {
            tokio::select! { _ = receiver.recv() => {}, _ = &mut pending => panic!("gated preparation completed early") }
        }
        drop(pending);
        drop((operator, output));
        assert!(source.upgrade().is_some());
        assert!(prefix_weak.upgrade().is_some());
        assert!(pool.reserved() > 32_768 * 640);
        assert!(tokio::time::timeout(Duration::from_millis(25), job.gather_owner().close_and_drain()).await.is_err());
        drop(release);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        assert!(source.upgrade().is_none());
        assert!(prefix_weak.upgrade().is_none());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn cpu_pipeline_competing_fragment_budget_refunds_before_fallback() {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let job = StreamJobContext::new(
            920,
            "fragment-budget",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        )
        .with_gather_owner(service.owner("fragment-budget".into()));
        let (mut operator, mut output) = admitted(&job, false).await;
        let pool = operator.runtime.pool.clone();
        let status = operator.status.clone();
        let context = StreamOperatorContext::new(&job, "asof", None)
            .with_output_budget(EdgeBudget::new(32_768, 128 << 20).unwrap());
        let prefix = Arc::new(operator.reserve_workspace(32_768 * 640 + 2_048).unwrap());
        let before = pool.reserved();
        let funding_before = job.gather_owner().funding();
        let gate = Arc::new(Gate::default());
        let release = Release(gate.clone());
        let (entered, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        operator.match_hook = Some(Arc::new(move |ordinal| {
            entered.send(ordinal).unwrap();
            gate.wait();
        }));
        let (done, finished) = tokio::sync::oneshot::channel();
        let prepare = async {
            let result = prepare(&operator, 32_768, prefix.clone(), &context).await;
            let reserved = pool.reserved();
            done.send(()).unwrap();
            (result, reserved)
        };
        let pressure = async {
            for _ in 0..2 {
                receiver.recv().await.unwrap();
            }
            let credit = MemoryConsumer::new("competing-fragment-work").register(&pool);
            let bytes = usize::try_from(operator.spec.limits().max_state_bytes()).unwrap()
                - pool.reserved()
                - 1_024;
            credit.try_grow(bytes).unwrap();
            drop(release);
            finished.await.unwrap();
            bytes
        };
        let ((result, reserved), pressure) = tokio::join!(prepare, pressure);
        assert!(result.unwrap().is_none());
        let funding = job.gather_owner().funding();
        assert_eq!(funding.2, 0);
        let controls = funding.0 + funding.1 - funding_before.0 - funding_before.1;
        assert_eq!(reserved, before + pressure + controls);
        assert_eq!(pool.reserved(), before + controls);
        assert_eq!(operator.status, status);
        drop(prefix);
        operator.match_hook = None;
        let (_, left, _) = fixture_with_keys(true, 32_768, 1);
        let right = left.iter().step_by(2).cloned().collect::<Vec<_>>();
        operator.on_end(&context, &mut output).await.unwrap();
        assert_output(&mut output, &oracle(&left, &right), true);
        drop((operator, output, context));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[tokio::test]
async fn cpu_pipeline_fragment_prefix_allocations_include_all_workers_and_owner() {
    for keys in [1, 65, 4_096] {
        let job = StreamJobContext::new(
            921,
            "prefix-allocations",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "asof", None);
        let (mut operator, rows, _) = fixture_with_keys(true, 4_096, keys);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", input(&operator, &rows), &context, &mut output)
            .await
            .unwrap();
        let prefix_credit = Arc::new(
            operator
                .reserve_workspace(operator.prefix_workspace_bytes(rows.len()).unwrap())
                .unwrap(),
        );
        let work = capture(&operator, rows.len(), 4, prefix_credit.clone(), &context)
            .await
            .unwrap()
            .unwrap();
        let credits = work
            .units
            .iter()
            .map(|runs| {
                work.prefix_credit(runs, runs.iter().map(|run| run.borrowed().len()).sum())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let funded =
            prefix_credit.size() + credits.iter().map(MemoryReservation::size).sum::<usize>();
        let mut owner = state::LeftPrefix::default();
        let allocation = allocation_counter::measure(|| {
            let summaries = work
                .units
                .iter()
                .map(|runs| {
                    let mut prefix = state::LeftPrefix::default();
                    for run in runs {
                        let left = run.borrowed().rows(1).next().unwrap().1;
                        run.visit_prefix(&mut prefix, work.view(left).batch.key, "asof")
                            .unwrap();
                    }
                    prefix
                })
                .collect::<Vec<_>>();
            for prefix in summaries {
                merge_prefix(&mut owner, prefix, "asof").unwrap();
            }
        });
        let mut expected = state::LeftPrefix::default();
        for run in operator.state.left.output_runs() {
            run.visit_prefix(run.len(), &mut expected, &operator.state.batches, "asof")
                .unwrap();
        }
        assert_eq!(owner.count, rows.len());
        assert_eq!(owner.batches, expected.batches);
        assert_eq!(owner.keys, expected.keys);
        assert_eq!(owner.owners, expected.owners);
        assert!(owner.sequence_owners.is_empty());
        assert!(
            allocation.bytes_max <= funded as u64,
            "keys={keys}, funded={funded}, actual={allocation:?}"
        );
        drop((
            owner,
            expected,
            credits,
            work,
            prefix_credit,
            operator,
            output,
            context,
        ));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
    }
}

#[test]
fn cpu_pipeline_optional_control_credit_refusal_preserves_serial_prefix() {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let job = StreamJobContext::new(
            922,
            "control-refusal",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        )
        .with_gather_owner(service.owner("control-refusal".into()));
        let (mut operator, mut output) = admitted(&job, false).await;
        let pool = operator.runtime.pool.clone();
        let context = StreamOperatorContext::new(&job, "asof", None)
            .with_output_budget(EdgeBudget::new(32_768, 128 << 20).unwrap());
        let prefix = Arc::new(
            operator
                .reserve_workspace(operator.prefix_workspace_bytes(32_768).unwrap())
                .unwrap(),
        );
        let work = capture(&operator, 32_768, 2, prefix.clone(), &context)
            .await
            .unwrap()
            .unwrap();
        let snapshot_bytes = work.credit.size();
        drop(work);
        let pressure = MemoryConsumer::new("snapshot-only-room").register(&pool);
        let limit = usize::try_from(operator.spec.limits().max_state_bytes()).unwrap();
        pressure
            .try_grow(limit - pool.reserved() - snapshot_bytes - 1)
            .unwrap();
        let reserved = pool.reserved();
        let prefix_bytes = prefix.size();
        let status = operator.status.clone();
        let result = prepare(&operator, 32_768, prefix.clone(), &context).await;
        assert!(
            matches!(result, Ok(None)),
            "optional control credit refusal did not select serial fallback"
        );
        assert_eq!(pool.reserved(), reserved);
        assert_eq!(prefix.size(), prefix_bytes);
        assert_eq!(operator.status, status);
        drop((pressure, prefix));
        operator.on_end(&context, &mut output).await.unwrap();
        let (_, left, _) = fixture_with_keys(true, 32_768, 1);
        let right = left.iter().step_by(2).cloned().collect::<Vec<_>>();
        assert_output(&mut output, &oracle(&left, &right), true);
        drop((operator, output, context));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[tokio::test]
async fn cpu_pipeline_snapshot_prepays_owned_operator_name_allocation() {
    use futures::FutureExt;

    let job = StreamJobContext::new(
        923,
        "long-name",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let (mut operator, output) = admitted(&job, false).await;
    let pool = operator.runtime.pool.clone();
    operator.name = "n".repeat(32 << 20);
    let context = StreamOperatorContext::new(&job, "asof", None);
    let prefix = Arc::new(
        operator
            .reserve_workspace(operator.prefix_workspace_bytes(32_768).unwrap())
            .unwrap(),
    );
    let mut work = None;
    let allocation = allocation_counter::measure(|| {
        work = capture(&operator, 32_768, 2, prefix, &context)
            .now_or_never()
            .expect("small source snapshot should not yield")
            .unwrap();
    });
    let work = work.unwrap();
    assert!(
        allocation.bytes_max <= work.credit.size() as u64,
        "snapshot funded={}, actual={allocation:?}",
        work.credit.size()
    );
    drop((work, operator, output, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

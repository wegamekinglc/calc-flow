use super::*;
use crate::{
    AsofStateLimits, BatchMetadata, EdgeCollector, Epoch, OperatorMetadata, StreamOperator,
    runtime::streaming::gather_work::TestService,
};
use datafusion::arrow::{
    array::{Int64Array, StringArray},
    datatypes::SchemaRef,
};
use std::sync::atomic::{AtomicUsize, Ordering};

const ROWS: usize = 8_192;

#[tokio::test]
async fn right_bucket_storage_is_sharded_before_and_after_cold_restore() {
    let mut operator = fixture();
    let pool = operator.runtime.pool.clone();
    let job = StreamJobContext::new(9, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", input(&operator, 0), &context, &mut output)
        .await
        .unwrap();
    let mut expected = [0; state::KEY_SHARDS];
    for (_, key) in operator.state.right.indexed_keys() {
        expected[state::key_shard(key)] += 1;
    }
    assert!(expected.iter().filter(|count| **count != 0).count() > 1);
    assert_eq!(operator.state.right.storage_shard_counts(), expected);
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    let mut restored = fixture();
    let restored_pool = restored.runtime.pool.clone();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.state.right.storage_shard_counts(), expected);
    assert_eq!(restored.status, operator.status);
    for candidate in [&mut operator, &mut restored] {
        candidate
            .process_data("left", input(candidate, 0), &context, &mut output)
            .await
            .unwrap();
        candidate.on_end(&context, &mut output).await.unwrap();
        assert_eq!(
            matches(&mut output, &candidate.schemas[2]),
            (0..ROWS)
                .map(|row| {
                    let sequence = i64::try_from(row).unwrap();
                    (sequence, sequence)
                })
                .collect::<Vec<_>>()
        );
    }
    drop((operator, restored, snapshot, output, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(restored_pool.reserved(), 0);
}

#[tokio::test]
async fn parallel_key_routes_do_not_hash_each_row() {
    let mut operator = fixture();
    let pool = operator.runtime.pool.clone();
    let job = StreamJobContext::new(8, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let batch = input(&operator, 0);
    let validated = operator.validate_admission("right", &batch).unwrap();
    let admission = operator
        .prepare_admission(validated, &batch, &context)
        .await
        .unwrap();
    state::take_encoding_hashes();
    let (work, references, credit) = parallel::capture(&operator, &admission, 2, &context)
        .await
        .unwrap();
    assert!(
        state::take_encoding_hashes() <= admission.right_capacities.len() * 2,
        "parallel routing must reuse handles instead of hashing each row"
    );
    assert_eq!(work.key_units().len(), admission.right_capacities.len());
    drop((
        work, references, credit, admission, operator, batch, context,
    ));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn right_key_routing_survives_new_keys_and_batch_reordering() {
    let mut operator = fixture();
    let pool = operator.runtime.pool.clone();
    let job = StreamJobContext::new(6, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let first = input(&operator, 0);
    let expected = key_routes(&mut operator, &first, 2, &context).await;
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", first, &context, &mut output)
        .await
        .unwrap();
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    let mut restored = fixture();
    let restored_pool = restored.runtime.pool.clone();
    restored.restore(&snapshot).unwrap();
    let extra = RecordBatch::try_new(
        operator.schemas[1].clone(),
        vec![
            Arc::new(StringArray::from(vec!["aaa-new-key"])),
            Arc::new(TimestampMicrosecondArray::from(vec![-1]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![-1])),
        ],
    )
    .unwrap();
    let next = input(&operator, ROWS);
    let mut records = vec![extra];
    records.extend(next.table_payload().unwrap().batches().iter().cloned());
    let next = Batch::table(records, BatchMetadata::default()).unwrap();
    let actual = key_routes(&mut operator, &next, 2, &context).await;
    for (key, unit) in expected {
        assert_eq!(
            actual[&key], unit,
            "new keys must not reroute retained keys"
        );
    }
    for units in [2, 3, 4, 8] {
        let expected = key_routes(&mut operator, &next, units, &context).await;
        assert_eq!(
            key_routes(&mut restored, &next, units, &context).await,
            expected
        );
    }
    drop((operator, restored, snapshot, output, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(restored_pool.reserved(), 0);
}

async fn key_routes(
    operator: &mut StreamAsofJoinOperator,
    input: &Batch,
    units: usize,
    context: &StreamOperatorContext<'_>,
) -> std::collections::BTreeMap<state::Encoding, usize> {
    let validated = operator.validate_admission("right", input).unwrap();
    let admission = operator
        .prepare_admission(validated, input, context)
        .await
        .unwrap();
    let (work, _, _credit) = parallel::capture(operator, &admission, units, context)
        .await
        .unwrap();
    work.key_units()
}

fn assert_snapshot(actual: &crate::OperatorStateSnapshot, expected: &crate::OperatorStateSnapshot) {
    assert_eq!(actual.inline_metadata, expected.inline_metadata);
    assert_eq!(actual.segments, expected.segments);
}

fn fixture() -> StreamAsofJoinOperator {
    let (template, schema) = identity_fixture();
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        std::time::Duration::from_micros(100_000),
        AsofStateLimits::new(100_000, 128 << 20).unwrap(),
    )
    .unwrap();
    StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap()
}

fn input(operator: &StreamAsofJoinOperator, start: usize) -> Batch {
    timed_input(operator, start, 0)
}

fn timed_input(operator: &StreamAsofJoinOperator, start: usize, offset: i64) -> Batch {
    keyed_input(operator, start, |row| {
        offset + i64::try_from(row % ROWS * 2).unwrap()
    })
}

fn keyed_input(
    operator: &StreamAsofJoinOperator,
    start: usize,
    time: impl Fn(usize) -> i64,
) -> Batch {
    let records = (start..start + ROWS)
        .collect::<Vec<_>>()
        .chunks(2_048)
        .map(|indices| {
            let rows = indices.iter().rev().copied().collect::<Vec<_>>();
            RecordBatch::try_new(
                operator.schemas[0].clone(),
                vec![
                    Arc::new(StringArray::from_iter_values(
                        rows.iter()
                            .map(|row| format!("key-{}-{}", row % 8, "x".repeat(32))),
                    )),
                    Arc::new(
                        TimestampMicrosecondArray::from_iter_values(
                            rows.iter().map(|row| time(*row)),
                        )
                        .with_timezone("UTC"),
                    ),
                    Arc::new(Int64Array::from_iter_values(
                        rows.iter().map(|row| i64::try_from(*row).unwrap()),
                    )),
                ],
            )
            .unwrap()
        })
        .collect();
    Batch::table(records, BatchMetadata::default()).unwrap()
}

fn matches(output: &mut EdgeCollector, schema: &SchemaRef) -> Vec<(i64, i64)> {
    let mut seen = Vec::new();
    for message in output.drain("output") {
        let batch = message.as_data().unwrap();
        assert_eq!(
            batch.metadata(),
            &BatchMetadata::new("asof", u64::try_from(seen.len()).unwrap(), JsonMap::new())
                .unwrap()
        );
        assert_eq!(batch.table_payload().unwrap().schema(), schema);
        for record in batch.table_payload().unwrap().batches() {
            let column = |index| {
                record
                    .column(index)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
            };
            seen.extend(
                column(2)
                    .values()
                    .iter()
                    .copied()
                    .zip(column(5).values().iter().copied()),
            );
        }
    }
    seen
}

#[test]
fn per_key_right_append_preserves_ties_and_overlap_recovery() {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut operator = fixture();
        let pool = operator.runtime.pool.clone();
        let job = StreamJobContext::new(5, "asof", JsonMap::new(), None, CancellationToken::new())
            .with_gather_owner(service.owner("per-key-append".into()));
        let context = StreamOperatorContext::new(&job, "asof", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        operator.admission_hook = Some(Arc::new(move |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        }));
        let parallel = std::thread::available_parallelism().map_or(1, usize::from) >= 2;
        for (ordinal, offset, expected_calls) in [(0, 0, 2), (1, 0, 2), (2, 1, 2), (3, 0, 4)] {
            let batch = keyed_input(&operator, ordinal * ROWS, |row| {
                i64::try_from(row % 8).unwrap() * 100_000 + offset
            });
            operator
                .process_data("right", batch, &context, &mut output)
                .await
                .unwrap();
            if parallel {
                assert_eq!(
                    calls.load(Ordering::Relaxed),
                    expected_calls,
                    "per-key append should preserve the in-place path at batch {ordinal}"
                );
            }
        }
        operator.prepare_checkpoint_async(&context).await.unwrap();
        let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
        let mut restored = fixture();
        let restored_pool = restored.runtime.pool.clone();
        restored.restore(&snapshot).unwrap();
        assert_eq!(restored.status, operator.status);
        assert_snapshot(&restored.checkpoint(Epoch::INITIAL).unwrap(), &snapshot);
        let expected = (0..8)
            .flat_map(|key| {
                (key..ROWS).step_by(8).map(move |row| {
                    (
                        i64::try_from(row).unwrap(),
                        i64::try_from(3 * ROWS - 8 + key).unwrap(),
                    )
                })
            })
            .collect::<Vec<_>>();
        for candidate in [&mut operator, &mut restored] {
            let left = keyed_input(candidate, 0, |row| {
                i64::try_from(row % 8).unwrap() * 100_000 + 1
            });
            candidate
                .process_data("left", left, &context, &mut output)
                .await
                .unwrap();
            candidate.on_end(&context, &mut output).await.unwrap();
            assert_eq!(matches(&mut output, &candidate.schemas[2]), expected);
        }
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop((operator, restored, snapshot, output, context));
        drop(job);
        assert_eq!(pool.reserved(), 0);
        assert_eq!(restored_pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn right_admission_runs_owned_key_units_and_preserves_recovery() {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut operator = fixture();
        let pool = operator.runtime.pool.clone();
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
            .with_gather_owner(service.owner("right-admission".into()));
        let context = StreamOperatorContext::new(&job, "asof", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        operator.admission_hook = Some(Arc::new(move |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        }));
        for start in [0, ROWS] {
            operator
                .process_data("right", input(&operator, start), &context, &mut output)
                .await
                .unwrap();
        }
        operator
            .process_data(
                "right",
                timed_input(&operator, 2 * ROWS, 100_000),
                &context,
                &mut output,
            )
            .await
            .unwrap();
        if std::thread::available_parallelism().map_or(1, usize::from) >= 2 {
            assert_eq!(
                calls.load(Ordering::Relaxed),
                4,
                "right admission ran no parallel key units"
            );
        }
        operator.prepare_checkpoint_async(&context).await.unwrap();
        let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
        let mut restored = fixture();
        let restored_pool = restored.runtime.pool.clone();
        restored.restore(&snapshot).unwrap();
        assert_eq!(restored.status, operator.status);
        let recaptured = restored.checkpoint(Epoch::INITIAL).unwrap();
        assert_snapshot(&recaptured, &snapshot);
        for candidate in [&mut operator, &mut restored] {
            candidate
                .process_data("left", input(candidate, 0), &context, &mut output)
                .await
                .unwrap();
            candidate.on_end(&context, &mut output).await.unwrap();
            assert_eq!(
                matches(&mut output, &candidate.schemas[2]),
                (0..ROWS)
                    .map(|row| (
                        i64::try_from(row).unwrap(),
                        i64::try_from(row + ROWS).unwrap()
                    ))
                    .collect::<Vec<_>>()
            );
        }
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop((operator, restored, snapshot, recaptured, output, context));
        drop(job);
        assert_eq!(pool.reserved(), 0);
        assert_eq!(restored_pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
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
fn abandoned_right_admission_keeps_owned_buckets_funded_until_exit() {
    use crate::runtime::streaming::gather_work::{GatherOperatorId, GatherStop};
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut operator = fixture();
        let pool = operator.runtime.pool.clone();
        let job = StreamJobContext::new(2, "asof", JsonMap::new(), None, CancellationToken::new())
            .with_gather_owner(service.owner("right-admission-abandon".into()));
        let context = StreamOperatorContext::new(&job, "asof", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("right", input(&operator, 0), &context, &mut output)
            .await
            .unwrap();
        let gate = Arc::new(Gate::default());
        let release = Release(gate.clone());
        let (entered, receiver) = std::sync::mpsc::channel();
        operator.admission_hook = Some(Arc::new(move |ordinal| {
            entered.send(ordinal).unwrap();
            gate.wait();
        }));
        let batch = input(&operator, ROWS);
        let validated = operator.validate_admission("right", &batch).unwrap();
        let admission = operator
            .prepare_admission(validated, &batch, &context)
            .await
            .unwrap();
        let bucket = operator
            .state
            .right
            .owned_bucket(&admission.right_capacities[0].0)
            .unwrap();
        let weak = Arc::downgrade(&bucket);
        drop(bucket);
        let reserved = pool.reserved();
        let (work, _, credit) = parallel::capture(&operator, &admission, 2, &context)
            .await
            .unwrap();
        let input_credit = pool.reserved() - reserved - credit.size();
        assert!(input_credit > 0);
        let scope = context
            .gather_client(GatherOperatorId::new("asof".into()))
            .scope()
            .unwrap();
        let ticket = scope
            .submit_parallel_work(Arc::new(work), credit, GatherStop::from_job(&job))
            .await
            .unwrap();
        let mut started = (0..2)
            .filter_map(|_| {
                receiver
                    .recv_timeout(std::time::Duration::from_secs(2))
                    .ok()
            })
            .collect::<Vec<_>>();
        started.sort_unstable();
        drop((ticket, admission, operator, batch, output));
        assert!(weak.upgrade().is_some());
        assert!(pool.reserved() >= input_credit);
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(25),
                job.gather_owner().close_and_drain()
            )
            .await
            .is_err()
        );
        drop(release);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        assert_eq!(started, [0, 1], "right admission units did not overlap");
        assert!(weak.upgrade().is_none());
        drop((scope, context));
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[tokio::test]
async fn right_admission_budget_refusal_preserves_state_and_refunds() {
    let mut operator = fixture();
    let job = StreamJobContext::new(3, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let batch = input(&operator, 0);
    let validated = operator.validate_admission("right", &batch).unwrap();
    let admission = operator
        .prepare_admission(validated, &batch, &context)
        .await
        .unwrap();
    let before = operator.status.clone();
    let free = operator.spec.limits().max_state_bytes() - operator.runtime.pool.reserved() as u64;
    let occupied = operator.reserve_workspace(free).unwrap();
    let reserved = operator.runtime.pool.reserved();
    assert!(matches!(
        parallel::capture(&operator, &admission, 2, &context).await,
        Err(crate::CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        })
    ));
    assert_eq!(operator.status, before);
    assert_eq!(operator.runtime.pool.reserved(), reserved);
    assert!(
        parallel::prepare(&operator, &admission, &context)
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(operator.runtime.pool.reserved(), reserved);
    if std::thread::available_parallelism().map_or(1, usize::from) >= 2 {
        job.cancellation().cancel();
        assert!(matches!(
            parallel::prepare(&operator, &admission, &context).await,
            Err(crate::CalcFlowError::Cancelled { .. })
        ));
        assert!(operator.state.right.is_empty());
        assert_eq!(operator.status, before);
        assert_eq!(operator.runtime.pool.reserved(), reserved);
    }
    drop((occupied, admission));
    assert_eq!(operator.runtime.pool.reserved(), 0);
}

#[test]
fn failed_right_admission_worker_preserves_checkpoint_and_state() {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut operator = fixture();
        let pool = operator.runtime.pool.clone();
        let job = StreamJobContext::new(4, "asof", JsonMap::new(), None, CancellationToken::new())
            .with_gather_owner(service.owner("right-admission-failure".into()));
        let context = StreamOperatorContext::new(&job, "asof", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        operator.prepare_checkpoint_async(&context).await.unwrap();
        let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
        let before = operator.status.clone();
        operator.admission_hook = Some(Arc::new(|ordinal| {
            assert_ne!(ordinal, 0, "right-admission-worker-failure");
        }));
        assert!(
            operator
                .process_data("right", input(&operator, 0), &context, &mut output)
                .await
                .is_err()
        );
        assert_eq!(operator.status, before);
        let after = operator.checkpoint(Epoch::INITIAL).unwrap();
        assert_snapshot(&after, &snapshot);
        assert!(output.drain("output").is_empty());
        let _failures = job.gather_owner().close_and_drain().await;
        drop((operator, snapshot, after, output, context));
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

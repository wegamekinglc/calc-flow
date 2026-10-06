use super::*;
use crate::{
    AsofJoinSide, AsofStateLimits, Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch,
    JsonMap, OperatorMetadata, StreamAsofJoinSpec, StreamJobContext, StreamOperator,
    runtime::streaming::gather_work::TestService,
};
use datafusion::arrow::{
    array::{Array, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

const ROWS: usize = 32_768;
type Row = (String, i64, i64, u64);

fn test_context(job: &StreamJobContext) -> StreamOperatorContext<'_> {
    StreamOperatorContext::new(job, "asof", None)
        .with_output_budget(crate::EdgeBudget::new(ROWS, 128 << 20).unwrap())
}

#[tokio::test]
async fn small_finalization_uses_serial_probe() {
    let (mut operator, left, right) = fixture(true, 10_000);
    let pool = operator.runtime.pool.clone();
    let job = StreamJobContext::new(904, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = test_context(&job);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    for (port, rows) in [("right", &right), ("left", &left)] {
        operator
            .process_data(port, input(&operator, rows), &context, &mut output)
            .await
            .unwrap();
    }
    assert!(
        parallel_matches(&operator, left.len(), &context)
            .await
            .unwrap()
            .is_none()
    );
    operator.on_end(&context, &mut output).await.unwrap();
    assert_output(&mut output, &oracle(&left, &right), true);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop((operator, output, context));
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

fn fixture(narrow: bool, count: usize) -> (StreamAsofJoinOperator, Vec<Row>, Vec<Row>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
    ]));
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["seq".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::from_micros(1),
        AsofStateLimits::new(100_000, 128 << 20).unwrap(),
    )
    .unwrap();
    let mut operator = StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap();
    if narrow {
        operator.set_output_projection(vec![2, 7]).unwrap();
    }
    let left = (0..count)
        .map(|index| {
            (
                format!("key-{:02}-{}", index % 16, "x".repeat(32)),
                i64::try_from(index / 16 * 3 + usize::from(index % 29 == 0) * 2).unwrap(),
                i64::try_from(index).unwrap(),
                f64::from(u32::try_from(index).unwrap()).to_bits(),
            )
        })
        .collect::<Vec<_>>();
    let right = left
        .iter()
        .enumerate()
        .filter(|(index, _)| index % 16 < 14)
        .flat_map(|(index, row)| {
            let time = i64::try_from(index / 16 * 3).unwrap();
            let bits = if index % 31 == 0 {
                0x7ff8_0000_0000_0001 + index as u64
            } else {
                (-f64::from_bits(row.3)).to_bits()
            };
            let first = (row.0.clone(), time, row.2, bits);
            std::iter::once(first)
                .chain((index % 31 == 0).then(|| (row.0.clone(), time, row.2 + 100_000, bits + 1)))
        })
        .collect();
    (operator, left, right)
}

fn input(operator: &StreamAsofJoinOperator, rows: &[Row]) -> Batch {
    let records = rows
        .chunks(2_048)
        .map(|rows| {
            let rows = rows.iter().rev().collect::<Vec<_>>();
            RecordBatch::try_new(
                operator.schemas[0].clone(),
                vec![
                    Arc::new(StringArray::from_iter_values(
                        rows.iter().map(|row| row.0.as_str()),
                    )),
                    Arc::new(
                        TimestampMicrosecondArray::from_iter_values(rows.iter().map(|row| row.1))
                            .with_timezone("UTC"),
                    ),
                    Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.2))),
                    Arc::new(Float64Array::from_iter_values(
                        rows.iter().map(|row| f64::from_bits(row.3)),
                    )),
                ],
            )
            .unwrap()
        })
        .collect();
    Batch::table(records, BatchMetadata::default()).unwrap()
}

fn oracle(left: &[Row], right: &[Row]) -> Vec<(i64, Option<i64>, Option<u64>)> {
    let mut groups = BTreeMap::<&str, BTreeMap<(i64, i64), u64>>::new();
    for row in right {
        groups
            .entry(&row.0)
            .or_default()
            .insert((row.1, row.2), row.3);
    }
    let mut left = left.iter().collect::<Vec<_>>();
    left.sort_by_key(|row| (row.1, &row.0, row.2));
    left.into_iter()
        .map(|row| {
            let candidate = groups
                .get(row.0.as_str())
                .and_then(|group| group.range(..=(row.1, i64::MAX)).next_back())
                .filter(|((time, _), _)| row.1 - time <= 1);
            (
                row.2,
                candidate.map(|((_, sequence), _)| *sequence),
                candidate.map(|(_, bits)| *bits),
            )
        })
        .collect()
}

fn assert_output(
    output: &mut EdgeCollector,
    expected: &[(i64, Option<i64>, Option<u64>)],
    narrow: bool,
) -> Vec<Vec<u8>> {
    let batches = output.drain("output");
    let mut rows = Vec::new();
    let mut encoded = Vec::new();
    for message in &batches {
        let batch = message.as_data().unwrap();
        assert_eq!(
            batch.metadata(),
            &BatchMetadata::new("asof", rows.len() as u64, JsonMap::new()).unwrap()
        );
        for record in batch.table_payload().unwrap().batches() {
            append_record_rows(record, narrow, &mut rows);
            encoded.push(
                crate::operator::asof::codec::encode_batch(record, 8 << 20, &mut Vec::new())
                    .unwrap(),
            );
        }
    }
    let expected = expected
        .iter()
        .map(|&(left, right, price)| (left, if narrow { None } else { right }, price))
        .collect::<Vec<_>>();
    assert_eq!(rows, expected);
    encoded
}

fn append_record_rows(
    record: &RecordBatch,
    narrow: bool,
    rows: &mut Vec<(i64, Option<i64>, Option<u64>)>,
) {
    let left = record
        .column(if narrow { 0 } else { 2 })
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let right = (!narrow).then(|| {
        record
            .column(6)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
    });
    let price = record
        .column(if narrow { 1 } else { 7 })
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for row in 0..record.num_rows() {
        rows.push((
            left.value(row),
            right
                .filter(|column| !column.is_null(row))
                .map(|column| column.value(row)),
            (!price.is_null(row)).then(|| price.value(row).to_bits()),
        ));
    }
}

#[test]
fn key_sharded_finalization_preserves_order_bits_and_cold_recovery() {
    for narrow in [false, true] {
        run_recovery(narrow);
    }
}

fn run_recovery(narrow: bool) {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (pool, restored_pool) = runtime.block_on(async {
        let (mut operator, left, right) = fixture(narrow, ROWS);
        let pool = operator.runtime.pool.clone();
        let job =
            StreamJobContext::new(902, "asof", JsonMap::new(), None, CancellationToken::new())
                .with_gather_owner(service.owner("key-shards".into()));
        let context = test_context(&job);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        for (port, rows) in [("right", &right), ("left", &left)] {
            operator
                .process_data(port, input(&operator, rows), &context, &mut output)
                .await
                .unwrap();
        }
        assert!(output.drain("output").is_empty());
        operator.prepare_checkpoint_async(&context).await.unwrap();
        let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
        let (mut restored, _, _) = fixture(narrow, 0);
        let restored_pool = restored.runtime.pool.clone();
        restored.restore(&snapshot).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        operator.match_hook = Some(Arc::new(move |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        }));
        operator.on_end(&context, &mut output).await.unwrap();
        if std::thread::available_parallelism().map_or(1, usize::from) >= 2 {
            assert_eq!(
                calls.load(Ordering::Relaxed),
                2,
                "ASOF never executed its matching shards"
            );
        }
        let expected = oracle(&left, &right);
        let encoded = assert_output(&mut output, &expected, narrow);
        let mut resumed = EdgeCollector::new(restored.output_ports().to_vec());
        restored.on_end(&context, &mut resumed).await.unwrap();
        assert_eq!(assert_output(&mut resumed, &expected, narrow), encoded);
        assert_eq!(operator.status, restored.status);
        assert_eq!(operator.status.emitted_left_rows, ROWS as u64);
        assert_eq!(operator.status.pending_left_rows, 0);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop((operator, restored, snapshot, output, resumed, context));
        drop(job);
        (pool, restored_pool)
    });
    drop(runtime);
    service.shutdown();
    assert_eq!(pool.reserved(), 0);
    assert_eq!(restored_pool.reserved(), 0);
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
fn payload_retirement_releases_owners_at_runtime_shutdown() {
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
    use std::sync::OnceLock;

    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    let gate = Arc::new(Gate::default());
    let release = Release(gate.clone());
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(8_192));
    let (owner, blocked) = runtime.block_on(async {
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let blocked = tokio::task::spawn_blocking(move || {
            started_tx.send(()).unwrap();
            gate.wait();
        });
        started_rx.await.unwrap();
        let workspace = MemoryConsumer::new("retirement-boundary").register(&pool);
        workspace.try_grow(4_096).unwrap();
        let payload = Arc::new(state::PayloadBatch {
            key: (0, 1),
            record: Arc::new(
                RecordBatch::try_new(
                    Arc::new(Schema::new(vec![Field::new(
                        "value",
                        DataType::Int64,
                        false,
                    )])),
                    vec![Arc::new(Int64Array::from(vec![7]))],
                )
                .unwrap(),
            ),
            encoded: OnceLock::new(),
            encoded_charge_bytes: 0,
            body_bytes: 8,
        });
        let owner = Arc::downgrade(&payload);
        let payloads = state::PayloadPool::default();
        let layout = payloads.project_remove(&BTreeMap::new(), "asof").unwrap();
        let mut removal = state::PreparedPayloadRemoval::capture(&payloads, &layout, workspace);
        removal.retain(1, &payload, 1);
        drop(payload);
        drop(removal);
        tokio::task::yield_now().await;
        let job = StreamJobContext::new(
            905,
            "retirement-boundary",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        )
        .with_gather_owner(service.owner("retirement-boundary".into()));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        // Payload retirement uses Tokio's blocking pool. Keep its input owner
        // and reservation observable until that separate lifecycle completes.
        assert_eq!(pool.reserved(), 4_096);
        assert!(owner.upgrade().is_some());
        (owner, blocked)
    });
    drop(release);
    drop(runtime);
    service.shutdown();
    assert!(blocked.is_finished());
    assert_eq!(pool.reserved(), 0);
    assert!(owner.upgrade().is_none());
}

#[test]
fn abandoned_matching_shards_keep_actual_buckets_funded_until_exit() {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let (mut operator, left, right) = fixture(false, 128);
        let pool = operator.runtime.pool.clone();
        let job =
            StreamJobContext::new(903, "asof", JsonMap::new(), None, CancellationToken::new())
                .with_gather_owner(service.owner("abandoned-matches".into()));
        let context = test_context(&job);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        for (port, rows) in [("right", &right), ("left", &left)] {
            operator
                .process_data(port, input(&operator, rows), &context, &mut output)
                .await
                .unwrap();
        }
        let gate = Arc::new(Gate::default());
        let release = Release(gate.clone());
        let (entered, receiver) = std::sync::mpsc::channel();
        operator.match_hook = Some(Arc::new(move |ordinal| {
            entered.send(ordinal).unwrap();
            gate.wait();
        }));
        let reserved = pool.reserved();
        let (work, credit) = capture(&operator, 128, 2, &context).await.unwrap();
        let weak = Arc::downgrade(&work.shards[0].buckets[0]);
        let input_credit = pool.reserved() - reserved - credit.size();
        assert!(
            input_credit as u64
                >= operator.state.right.metadata_bytes()
                    + operator.state.encoding_owner_allocation().0
        );
        let scope = context
            .gather_client(GatherOperatorId::new("asof".into()))
            .scope()
            .unwrap();
        let ticket = scope
            .submit_parallel_work(Arc::new(work), credit, GatherStop::from_job(&job))
            .await
            .unwrap();
        let mut started = (0..2)
            .filter_map(|_| receiver.recv_timeout(Duration::from_secs(2)).ok())
            .collect::<Vec<_>>();
        started.sort_unstable();
        let overlap = started == [0, 1];
        drop((ticket, operator, output));
        assert!(weak.upgrade().is_some());
        assert!(pool.reserved() >= input_credit);
        assert!(
            tokio::time::timeout(
                Duration::from_millis(25),
                job.gather_owner().close_and_drain()
            )
            .await
            .is_err()
        );
        drop(release);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        assert!(
            overlap,
            "ASOF key matching did not overlap on native workers"
        );
        assert!(weak.upgrade().is_none());
        drop((scope, context));
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[tokio::test]
async fn key_shard_budget_rejection_refunds_and_preserves_input() {
    let (mut operator, left, right) = fixture(false, 128);
    let job = StreamJobContext::new(904, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = test_context(&job);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    for (port, rows) in [("right", &right), ("left", &left)] {
        operator
            .process_data(port, input(&operator, rows), &context, &mut output)
            .await
            .unwrap();
    }
    let before = operator.status.clone();
    let capacity =
        operator.spec.limits().max_state_bytes() - operator.runtime.pool.reserved() as u64;
    let occupied = operator.reserve_workspace(capacity).unwrap();
    let reserved = operator.runtime.pool.reserved();
    assert!(matches!(
        capture(&operator, 128, 2, &context).await,
        Err(crate::CalcFlowError::OperatorReason {
            reason_code: crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        })
    ));
    assert_eq!(operator.runtime.pool.reserved(), reserved);
    assert_eq!(operator.status, before);
    drop(occupied);
    let (work, credit) = capture(&operator, 128, 2, &context).await.unwrap();
    let stop = GatherStop::from_job(&job);
    for ordinal in 0..work.unit_count() {
        for (position, row) in work.run(ordinal, &stop).unwrap() {
            let (key, _) = operator.state.left.output_iter().nth(position).unwrap();
            assert_eq!(row, operator.state.candidate(key.1, *key.0, 1).copied());
        }
    }
    drop((work, credit));
    assert_eq!(operator.status, before);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}

struct Reject;

#[async_trait::async_trait]
impl crate::StreamCollector for Reject {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        Err(crate::CalcFlowError::Internal {
            message: "rejected matching prefix".into(),
        })
    }
}

#[tokio::test]
async fn rejected_sharded_output_keeps_checkpoint_and_continues_exactly() {
    let (mut operator, left, right) = fixture(true, ROWS);
    let job = StreamJobContext::new(905, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = test_context(&job);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    for (port, rows) in [("right", &right), ("left", &left)] {
        operator
            .process_data(port, input(&operator, rows), &context, &mut output)
            .await
            .unwrap();
    }
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    let before = operator.status.clone();
    let reserved = operator.runtime.pool.reserved();
    let headroom = operator.checkpoint_workspace().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    operator.match_hook = Some(Arc::new(move |_| {
        counter.fetch_add(1, Ordering::Relaxed);
    }));
    let prepared = operator.output_attempt(ROWS, &context).await.unwrap();
    if std::thread::available_parallelism().map_or(1, usize::from) >= 2 {
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }
    let result = operator
        .commit_prefix_output(prepared, headroom, &context, &mut Reject)
        .await;
    assert!(
        matches!(result, Err(crate::CalcFlowError::Internal { message }) if message == "rejected matching prefix")
    );
    assert_eq!(operator.status, before);
    tokio::time::timeout(Duration::from_secs(2), async {
        while operator.runtime.pool.reserved() != reserved {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(operator.runtime.pool.reserved(), reserved);
    let repeated = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(repeated.inline_metadata, snapshot.inline_metadata);
    assert_eq!(repeated.segments, snapshot.segments);
    let (mut restored, _, _) = fixture(true, 0);
    restored.restore(&snapshot).unwrap();
    restored.on_end(&context, &mut output).await.unwrap();
    assert_output(&mut output, &oracle(&left, &right), true);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}

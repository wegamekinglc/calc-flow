use super::*;
use crate::runtime::streaming::gather_work::{GatherStop, TestService};
use crate::{CancellationToken, DataFusionConfig, JsonMap, StreamJobContext};
use datafusion::arrow::array::{Int64Array, StringArray};
use std::collections::BTreeMap;
use std::sync::{
    Condvar, LazyLock, Mutex,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};
use std::time::Duration;

static PROBES: LazyLock<Mutex<BTreeMap<String, Arc<Probe>>>> = LazyLock::new(Mutex::default);

#[derive(Default)]
struct Gate {
    released: Mutex<bool>,
    changed: Condvar,
}

impl Gate {
    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.changed.notify_all();
    }

    fn wait(&self, stop: &GatherStop) -> Result<()> {
        let mut released = self.released.lock().unwrap();
        while !*released {
            stop.check()?;
            released = self
                .changed
                .wait_timeout(released, Duration::from_millis(10))
                .unwrap()
                .0;
        }
        stop.check()
    }
}

struct Probe {
    entered: mpsc::Sender<usize>,
    completed: mpsc::Sender<usize>,
    gates: [Gate; 2],
    active: AtomicUsize,
    maximum: AtomicUsize,
    fault: Mutex<Option<Fault>>,
    merge_calls: AtomicUsize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    Error,
    Panic,
    Merge,
    Cancel,
    Deadline,
    Abandon,
}

struct Registration {
    name: String,
    probe: Arc<Probe>,
}

impl Drop for Registration {
    fn drop(&mut self) {
        for gate in &self.probe.gates {
            gate.release();
        }
        PROBES.lock().unwrap().remove(&self.name);
    }
}

pub(super) fn before_unit(name: &str, ordinal: usize, stop: &GatherStop) -> Result<()> {
    let probe = PROBES.lock().unwrap().get(name).cloned();
    if let Some(probe) = probe {
        let active = probe.active.fetch_add(1, Ordering::SeqCst) + 1;
        probe.maximum.fetch_max(active, Ordering::SeqCst);
        let _ = probe.entered.send(ordinal);
        let result = probe.gates[ordinal].wait(stop);
        if result.is_err() {
            probe.active.fetch_sub(1, Ordering::SeqCst);
        }
        result?;
        if ordinal == 0 {
            let mut fault = probe.fault.lock().unwrap();
            match *fault {
                Some(Fault::Error) => {
                    *fault = None;
                    return Err(df_error(name, "injected local SUM error"));
                }
                Some(Fault::Panic) => {
                    *fault = None;
                    drop(fault);
                    panic!("injected local SUM panic");
                }
                _ => {}
            }
        }
    }
    Ok(())
}

pub(super) fn before_merge(name: &str) -> Result<()> {
    if let Some(probe) = PROBES.lock().unwrap().get(name).cloned()
        && probe.merge_calls.fetch_add(1, Ordering::SeqCst) == 1
        && probe
            .fault
            .lock()
            .unwrap()
            .take_if(|fault| *fault == Fault::Merge)
            .is_some()
    {
        return Err(df_error(name, "injected coordinator SUM merge error"));
    }
    Ok(())
}

pub(super) fn after_unit(name: &str, ordinal: usize) {
    if let Some(probe) = PROBES.lock().unwrap().get(name).cloned() {
        probe.active.fetch_sub(1, Ordering::SeqCst);
        let _ = probe.completed.send(ordinal);
    }
}

pub(super) struct Completion<'a> {
    name: &'a str,
    ordinal: usize,
}

pub(super) fn completion(name: &str, ordinal: usize) -> Completion<'_> {
    Completion { name, ordinal }
}

impl Drop for Completion<'_> {
    fn drop(&mut self) {
        after_unit(self.name, self.ordinal);
    }
}

pub(super) fn source(rows: usize) -> Batch {
    let record = RecordBatch::try_from_iter(vec![
        (
            "key",
            Arc::new(StringArray::from_iter_values(
                (0..rows).map(|row| format!("key-{}", row % 16)),
            )) as ArrayRef,
        ),
        (
            "value",
            Arc::new(Int64Array::from(vec![1; rows])) as ArrayRef,
        ),
    ])
    .unwrap();
    Batch::table(vec![record], crate::BatchMetadata::default()).unwrap()
}

fn overlap_source() -> Batch {
    let original = source(4 * CHUNK_ROWS);
    let record = &original.table_payload().unwrap().batches()[0];
    let keys = StringArray::from_iter_values(
        (0..4 * CHUNK_ROWS).map(|row| format!("key-{}", row / (2 * CHUNK_ROWS) * 16 + row % 16)),
    );
    Batch::table(
        vec![
            RecordBatch::try_new(
                record.schema(),
                vec![Arc::new(keys), record.column(1).clone()],
            )
            .unwrap(),
        ],
        crate::BatchMetadata::default(),
    )
    .unwrap()
}

#[test]
fn test_grouped_parallel_integer_sum_executes_two_units_simultaneously() {
    let name = "parallel-sum-overlap";
    let (entered, receiver) = mpsc::channel();
    let (completed, finished) = mpsc::channel();
    let probe = Arc::new(Probe {
        entered,
        completed,
        gates: std::array::from_fn(|_| Gate::default()),
        active: AtomicUsize::new(0),
        maximum: AtomicUsize::new(0),
        fault: Mutex::new(None),
        merge_calls: AtomicUsize::new(0),
    });
    PROBES.lock().unwrap().insert(name.into(), probe.clone());
    let registration = Registration {
        name: name.into(),
        probe: probe.clone(),
    };
    let (ready, prepared) = mpsc::channel();
    let runner = std::thread::spawn(move || {
        let service = TestService::new(2, 1).unwrap();
        let executor = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        executor.block_on(async {
            let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
            let batch = overlap_source();
            let query = crate::expression::parse_select_query(
                "SELECT key, SUM(value) AS total FROM events GROUP BY key",
            )
            .unwrap();
            let mut plan = IncrementalSql::plan(
                &runtime,
                &query,
                "events",
                batch.table_payload().unwrap().schema().clone(),
                name,
            )
            .await
            .unwrap()
            .unwrap();
            let job =
                StreamJobContext::new(941, name, JsonMap::new(), None, CancellationToken::new())
                    .with_gather_owner(service.owner(name.into()));
            let context = StreamOperatorContext::new(&job, name, None);
            ready.send(()).unwrap();
            let transaction = plan.update(&batch, &context, name).await.unwrap();
            assert_eq!(
                transaction
                    .records
                    .iter()
                    .map(RecordBatch::num_rows)
                    .sum::<usize>(),
                32
            );
            let keys = snapshot(&transaction.records)
                .into_iter()
                .map(|row| row[0].clone())
                .collect::<Vec<_>>();
            assert_eq!(
                keys,
                (0..32)
                    .map(|key| ScalarValue::Utf8(Some(format!("key-{key}"))))
                    .collect::<Vec<_>>()
            );
            plan.commit(transaction);
            drop(plan);
            assert!(job.gather_owner().close_and_drain().await.is_empty());
        });
        drop(executor);
        service.shutdown();
    });
    prepared.recv_timeout(Duration::from_secs(10)).unwrap();
    let mut started = (0..2)
        .filter_map(|_| receiver.recv_timeout(Duration::from_secs(2)).ok())
        .collect::<Vec<_>>();
    started.sort_unstable();
    probe.gates[1].release();
    let first_completed = finished.recv_timeout(Duration::from_secs(2)).ok();
    probe.gates[0].release();
    runner.join().unwrap();
    drop(registration);
    assert_eq!(
        started,
        [0, 1],
        "integer SUM must execute two local CPU units before either unit is released"
    );
    assert_eq!(probe.maximum.load(Ordering::SeqCst), 2);
    assert_eq!(
        first_completed,
        Some(1),
        "unit completion must be reversed deliberately"
    );
}

#[path = "grouped_parallel_lifecycle_tests.rs"]
mod lifecycle;

#[path = "grouped_parallel_managed_tests.rs"]
mod managed;

const PREFIX_SQL: &str = "SELECT key, stock, SUM(value) FILTER (WHERE enabled) AS yes, SUM(value) FILTER (WHERE NOT enabled) AS no FROM events WHERE keep GROUP BY key, stock";

#[test]
fn test_grouped_parallel_sparse_state_slices_preserve_null_only_tail_groups() {
    let service = TestService::new(2, 1).unwrap();
    let executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    executor.block_on(async {
        for unsigned in [false, true] {
            let rows = 4 * CHUNK_ROWS;
            let valid = (0..rows).map(|row| row % (2 * CHUNK_ROWS) < CHUNK_ROWS).collect::<Vec<_>>();
            let value: ArrayRef = if unsigned {
                Arc::new(datafusion::arrow::array::UInt64Array::from(valid.iter().map(|valid| valid.then_some(1)).collect::<Vec<_>>()))
            } else {
                Arc::new(Int64Array::from(valid.iter().map(|valid| valid.then_some(1)).collect::<Vec<_>>()))
            };
            let record = RecordBatch::try_from_iter(vec![
                ("key", Arc::new(StringArray::from_iter_values((0..rows).map(|row| format!("key-{row}")))) as ArrayRef),
                ("value", value),
                ("enabled", Arc::new(BooleanArray::from(valid.clone())) as ArrayRef),
            ]).unwrap();
            let input = Batch::table(vec![record], crate::BatchMetadata::default()).unwrap();
            let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
            let query = crate::expression::parse_select_query("SELECT key, SUM(value), SUM(value) FILTER (WHERE enabled) FROM events GROUP BY key").unwrap();
            let mut plan = IncrementalSql::plan(&runtime, &query, "events", input.table_payload().unwrap().schema().clone(), "parallel-null-tail").await.unwrap().unwrap();
            let job = StreamJobContext::new(953, "parallel-null-tail", JsonMap::new(), None, CancellationToken::new()).with_gather_owner(service.owner("parallel-null-tail".into()));
            let context = StreamOperatorContext::new(&job, "parallel-null-tail", None);
            let transaction = plan.update(&input, &context, "parallel-null-tail").await.unwrap();
            for (row, values) in snapshot(&transaction.records).iter().enumerate() {
                let expected = if unsigned { ScalarValue::UInt64(valid[row].then_some(1)) } else { ScalarValue::Int64(valid[row].then_some(1)) };
                assert_eq!(values[1], expected, "unseen state-slice group {row}");
                assert_eq!(values[2], expected, "filtered state-slice group {row}");
            }
            drop((transaction, plan));
            assert!(job.gather_owner().close_and_drain().await.is_empty());
        }
    });
    drop(executor);
    service.shutdown();
}

fn prefix_input(unsigned: bool, phase: usize) -> Batch {
    let rows = 4 * CHUNK_ROWS;
    let keys = (0..rows)
        .map(|row| {
            let code = if row % 5 == 0 {
                0
            } else {
                row / CHUNK_ROWS * 4 + row % 4 + phase
            };
            (code % 4 != 3).then(|| format!("κ\0{code}"))
        })
        .collect::<Vec<_>>();
    let bits = (0..rows)
        .map(|row| (row % 11 != 0).then_some([i64::MAX.cast_unsigned(), 1, u64::MAX, 5][row % 4]))
        .collect::<Vec<_>>();
    let value: ArrayRef = if unsigned {
        Arc::new(datafusion::arrow::array::UInt64Array::from(bits))
    } else {
        Arc::new(Int64Array::from(
            bits.into_iter()
                .map(|value| value.map(u64::cast_signed))
                .collect::<Vec<_>>(),
        ))
    };
    let record = RecordBatch::try_from_iter(vec![
        ("key", Arc::new(StringArray::from(keys)) as ArrayRef),
        (
            "stock",
            Arc::new(StringArray::from_iter_values(
                (0..rows).map(|row| format!("stock-{}", row % 2)),
            )) as ArrayRef,
        ),
        ("value", value),
        (
            "keep",
            Arc::new(BooleanArray::from(
                (0..rows).map(|row| row % 7 != 0).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "enabled",
            Arc::new(BooleanArray::from(
                (0..rows)
                    .map(|row| (row % 13 != 0).then_some(row % 3 != 0))
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
    ])
    .unwrap();
    Batch::table(
        vec![
            record.slice(0, 2 * CHUNK_ROWS),
            record.slice(2 * CHUNK_ROWS, 2 * CHUNK_ROWS),
        ],
        crate::BatchMetadata::default(),
    )
    .unwrap()
}

fn snapshot(records: &[RecordBatch]) -> Vec<Vec<ScalarValue>> {
    records
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| {
                record
                    .columns()
                    .iter()
                    .map(|array| ScalarValue::try_from_array(array, row).unwrap())
                    .collect()
            })
        })
        .collect()
}

type PrefixOracle = Vec<(Vec<ScalarValue>, [Option<u64>; 2])>;

fn extend_oracle(expected: &mut PrefixOracle, input: &Batch) {
    for record in input.table_payload().unwrap().batches() {
        for row in 0..record.num_rows() {
            let values = record
                .columns()
                .iter()
                .map(|array| ScalarValue::try_from_array(array, row).unwrap())
                .collect::<Vec<_>>();
            if values[3] != ScalarValue::Boolean(Some(true)) {
                continue;
            }
            let key = values[..2].to_vec();
            let position = expected
                .iter()
                .position(|(previous, _)| previous == &key)
                .unwrap_or_else(|| {
                    expected.push((key, [None, None]));
                    expected.len() - 1
                });
            let selected = match values[4] {
                ScalarValue::Boolean(Some(true)) => Some(0),
                ScalarValue::Boolean(Some(false)) => Some(1),
                _ => None,
            };
            let value = match values[2] {
                ScalarValue::Int64(value) => value.map(i64::cast_unsigned),
                ScalarValue::UInt64(value) => value,
                _ => panic!("integer input"),
            };
            if let (Some(selected), Some(value)) = (selected, value) {
                let total = &mut expected[position].1[selected];
                *total = Some(total.unwrap_or(0).wrapping_add(value));
            }
        }
    }
}

fn oracle_snapshot(expected: &PrefixOracle, unsigned: bool) -> Vec<Vec<ScalarValue>> {
    expected
        .iter()
        .map(|(key, totals)| {
            key.iter()
                .cloned()
                .chain(totals.iter().map(|total| {
                    if unsigned {
                        ScalarValue::UInt64(*total)
                    } else {
                        ScalarValue::Int64(total.map(u64::cast_signed))
                    }
                }))
                .collect()
        })
        .collect()
}

#[test]
fn test_grouped_parallel_modular_prefixes_discard_and_held_snapshots() {
    let service = TestService::new(2, 1).unwrap();
    let executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    executor.block_on(async {
        for unsigned in [false, true] {
            let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
            let first = prefix_input(unsigned, 0);
            let query = crate::expression::parse_select_query(PREFIX_SQL).unwrap();
            let mut plan = IncrementalSql::plan(
                &runtime,
                &query,
                "events",
                first.table_payload().unwrap().schema().clone(),
                "parallel-prefix",
            )
            .await
            .unwrap()
            .unwrap();
            let job = StreamJobContext::new(
                942,
                "parallel-prefix",
                JsonMap::new(),
                None,
                CancellationToken::new(),
            )
            .with_gather_owner(service.owner("parallel-prefix".into()));
            let context = StreamOperatorContext::new(&job, "parallel-prefix", None);
            let mut expected = Vec::new();
            let mut held = Vec::new();
            for phase in 0..3 {
                let input = prefix_input(unsigned, phase);
                if phase == 1 {
                    drop(
                        plan.update(&input, &context, "parallel-prefix")
                            .await
                            .unwrap(),
                    );
                }
                let transaction = plan
                    .update(&input, &context, "parallel-prefix")
                    .await
                    .unwrap();
                extend_oracle(&mut expected, &input);
                assert_eq!(
                    snapshot(&transaction.records),
                    oracle_snapshot(&expected, unsigned)
                );
                held.push((transaction.records.clone(), snapshot(&transaction.records)));
                plan.commit(transaction);
            }
            for (records, original) in held {
                assert_eq!(snapshot(&records), original);
            }
            drop(plan);
            assert!(job.gather_owner().close_and_drain().await.is_empty());
            drop(context);
            drop(job);
            assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
        }
    });
    drop(executor);
    service.shutdown();
}

#[test]
fn test_grouped_parallel_where_and_filter_keep_distinct_group_existence() {
    let service = TestService::new(2, 1).unwrap();
    let executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    executor.block_on(async {
        let input = source(4 * CHUNK_ROWS);
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let job = StreamJobContext::new(
            943,
            "parallel-null",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        )
        .with_gather_owner(service.owner("parallel-null".into()));
        let context = StreamOperatorContext::new(&job, "parallel-null", None);
        for sql in [
            "SELECT key, SUM(value) FILTER (WHERE false) AS total FROM events GROUP BY key",
            "SELECT key, SUM(value) AS total FROM events WHERE false GROUP BY key",
        ] {
            let query = crate::expression::parse_select_query(sql).unwrap();
            let mut plan = IncrementalSql::plan(
                &runtime,
                &query,
                "events",
                input.table_payload().unwrap().schema().clone(),
                "parallel-null",
            )
            .await
            .unwrap()
            .unwrap();
            let transaction = plan
                .update(&input, &context, "parallel-null")
                .await
                .unwrap();
            let expected = runtime
                .sql(
                    sql,
                    &BTreeMap::from([("events".into(), input.clone())]),
                    Some("parallel-null-oracle"),
                )
                .await
                .unwrap();
            let mut actual = snapshot(&transaction.records);
            let mut oracle = snapshot(expected.table_payload().unwrap().batches());
            actual.sort_by_key(|row| format!("{row:?}"));
            oracle.sort_by_key(|row| format!("{row:?}"));
            assert_eq!(actual, oracle);
        }
        assert!(job.gather_owner().close_and_drain().await.is_empty());
    });
    drop(executor);
    service.shutdown();
}

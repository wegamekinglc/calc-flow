use std::{
    collections::BTreeMap,
    future::Future,
    sync::{
        Arc, Condvar, LazyLock, Mutex, Weak,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    task::{Context, Poll, Waker},
    thread::{self, JoinHandle, ThreadId},
    time::Duration,
};

use datafusion::{
    arrow::{
        array::{Array, ArrayRef, Int64Array},
        record_batch::RecordBatch,
    },
    execution::memory_pool::MemoryPool,
};

use super::super::*;
use crate::{CancellationToken, EdgeCollector, StreamJobContext};

const QUERY: &str = "SELECT key, SUM(value) AS total, COUNT(*) AS rows FROM events GROUP BY key";
const GROUPS: usize = 16_384;
static PROBES: LazyLock<Mutex<BTreeMap<String, Arc<Probe>>>> = LazyLock::new(Mutex::default);

#[derive(Default)]
struct Probe {
    ticks: AtomicUsize,
    census: Mutex<Vec<(usize, usize)>>,
    arrays: Mutex<Vec<Weak<dyn Array>>>,
    copied_bytes: Mutex<Vec<usize>>,
    encoder: Mutex<Option<(ThreadId, usize)>>,
    entered: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    gate: Option<Arc<(Mutex<bool>, Condvar)>>,
}

struct Registration(String);

impl Registration {
    fn new(name: &str, probe: &Arc<Probe>) -> Self {
        assert!(
            PROBES
                .lock()
                .unwrap()
                .insert(name.into(), probe.clone())
                .is_none()
        );
        Self(name.into())
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        PROBES.lock().unwrap().remove(&self.0);
    }
}

fn probe(name: &str) -> Option<Arc<Probe>> {
    PROBES.lock().unwrap().get(name).cloned()
}

pub(in crate::operator::sql) fn after_census(name: &str, groups: usize) {
    if let Some(probe) = probe(name) {
        let tick = probe.ticks.load(Ordering::SeqCst);
        probe.census.lock().unwrap().push((groups, tick));
    }
}

pub(in crate::operator::sql) fn after_array(name: &str, array: &ArrayRef) {
    if let Some(probe) = probe(name) {
        probe.arrays.lock().unwrap().push(Arc::downgrade(array));
    }
}

pub(in crate::operator::sql) fn after_bytes(name: &str, copied: usize) {
    if copied != 0
        && let Some(probe) = probe(name)
    {
        probe.copied_bytes.lock().unwrap().push(copied);
    }
}

pub(in crate::operator::sql) fn before_ipc(name: &str, records: &[RecordBatch], funded: usize) {
    let Some(probe) = probe(name) else { return };
    *probe.encoder.lock().unwrap() = Some((thread::current().id(), funded));
    probe.arrays.lock().unwrap().extend(
        records
            .iter()
            .flat_map(RecordBatch::columns)
            .map(Arc::downgrade),
    );
    if let Some(entered) = probe.entered.lock().unwrap().take() {
        let _ = entered.send(());
    }
    if let Some(gate) = &probe.gate {
        let (lock, ready) = gate.as_ref();
        let mut released = lock.lock().unwrap();
        while !*released {
            released = ready.wait(released).unwrap();
        }
    }
}

struct Release {
    release: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Release {
    fn new(gate: Arc<(Mutex<bool>, Condvar)>) -> Self {
        let (release, request) = mpsc::channel();
        let thread = thread::spawn(move || {
            let _ = request.recv_timeout(Duration::from_secs(2));
            *gate.0.lock().unwrap() = true;
            gate.1.notify_all();
        });
        Self {
            release: Some(release),
            thread: Some(thread),
        }
    }
}

impl Drop for Release {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            let _ = release.send(());
        }
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

fn input(name: &str, sequence: u64, start: usize, count: usize) -> Batch {
    let record = RecordBatch::try_from_iter(vec![
        (
            "key",
            Arc::new(Int64Array::from_iter_values(
                (start..start + count).map(|n| i64::try_from(n).unwrap()),
            )) as ArrayRef,
        ),
        (
            "value",
            Arc::new(Int64Array::from(
                (start..start + count)
                    .map(|n| (n % 17 != 0).then_some(i64::try_from(n % 101).unwrap()))
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
    ])
    .unwrap();
    Batch::table(
        vec![record],
        BatchMetadata::new(
            name,
            sequence,
            JsonMap::from([("case".into(), json!(name))]),
        )
        .unwrap(),
    )
    .unwrap()
}

fn rows(batch: &Batch) -> Vec<Vec<datafusion::common::ScalarValue>> {
    let mut rows = batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| {
                record
                    .columns()
                    .iter()
                    .map(|array| {
                        datafusion::common::ScalarValue::try_from_array(array, row).unwrap()
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.partial_cmp(right).unwrap());
    rows
}

async fn seeded(
    name: &str,
) -> (
    SqlOperator,
    StreamJobContext,
    OperatorStateSnapshot,
    Arc<dyn MemoryPool>,
) {
    seeded_batch(name, input(name, 0, 0, GROUPS)).await
}

async fn seeded_batch(
    name: &str,
    batch: Batch,
) -> (
    SqlOperator,
    StreamJobContext,
    OperatorStateSnapshot,
    Arc<dyn MemoryPool>,
) {
    let job = StreamJobContext::new(91, name, JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, name, None);
    let mut operator = SqlOperator::new(name, QUERY, vec!["events".into()], vec![]).unwrap();
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", batch.clone(), &context, &mut output)
        .await
        .unwrap();
    let emitted = output.drain("output");
    assert_eq!(emitted.len(), 1);
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(QUERY, &BTreeMap::from([("events".into(), batch)]), None)
        .await
        .unwrap();
    let observed = emitted[0].as_data().unwrap();
    assert_eq!(observed.metadata(), expected.metadata());
    assert_eq!(
        observed.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(rows(observed), rows(&expected));
    assert!(operator.retained.is_none());
    let prepared = operator.prepare_checkpoint_work(&|| Ok(())).unwrap();
    let snapshot = prepared.snapshot().unwrap();
    drop(prepared);
    assert!(operator.compact.as_ref().unwrap().capture.is_none());
    let pool = operator
        .retention_runtime()
        .unwrap()
        .incremental_memory_pool();
    (operator, job, snapshot, pool)
}

fn same_snapshot(left: &OperatorStateSnapshot, right: &OperatorStateSnapshot) {
    assert_eq!(left.inline_metadata, right.inline_metadata);
    assert_eq!(
        left.segments.keys().collect::<Vec<_>>(),
        right.segments.keys().collect::<Vec<_>>()
    );
    for (name, segment) in &left.segments {
        assert_eq!(segment.bytes(), right.segments[name].bytes());
    }
}

async fn wait_arrays_released(probe: &Probe) {
    tokio::time::timeout(Duration::from_secs(10), async {
        while probe
            .arrays
            .lock()
            .unwrap()
            .iter()
            .any(|array| array.upgrade().is_some())
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

async fn wait_encoder_released(probe: &Probe, pool: &dyn MemoryPool, baseline: usize) {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let arrays_released = probe
                .arrays
                .lock()
                .unwrap()
                .iter()
                .all(|array| array.upgrade().is_none());
            if arrays_released && pool.reserved() == baseline {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn test_direct_compact_async_heartbeat_advances_between_actual_export_work() {
    let name = "direct-compact-export-heartbeat";
    let (mut operator, job, expected, _) = seeded(name).await;
    let probe = Arc::new(Probe::default());
    let registration = Registration::new(name, &probe);
    let observed = probe.clone();
    let (pulse, mut requested) =
        tokio::sync::mpsc::unbounded_channel::<tokio::sync::oneshot::Sender<()>>();
    let heartbeat = tokio::spawn(async move {
        while let Some(completed) = requested.recv().await {
            observed.ticks.fetch_add(1, Ordering::SeqCst);
            let _ = completed.send(());
        }
    });
    let context = StreamOperatorContext::new(&job, name, None);
    let result = {
        let work = operator.prepare_checkpoint_async(&context);
        tokio::pin!(work);
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let mut context = Context::from_waker(Waker::noop());
                if let Poll::Ready(result) = work.as_mut().poll(&mut context) {
                    break result;
                }
                let (complete, completed) = tokio::sync::oneshot::channel();
                pulse.send(complete).unwrap();
                completed.await.unwrap();
            }
        })
        .await
    };
    drop(pulse);
    heartbeat.await.unwrap();
    drop(registration);
    let census = probe.census.lock().unwrap().clone();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    same_snapshot(&expected, &snapshot);
    result.unwrap().unwrap();
    assert!(census.len() >= 2);
    assert_eq!(census.last().unwrap().0, GROUPS);
    assert!(
        census.windows(2).all(|steps| steps[1].1 > steps[0].1),
        "heartbeat did not progress between real export chunks: {census:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_direct_compact_dropped_partial_export_refunds_actual_arrow_without_install() {
    let name = "direct-compact-partial-drop";
    let (mut operator, job, expected, pool) = seeded(name).await;
    let before = pool.reserved();
    let source = operator.compact.as_ref().unwrap().recovery_credit().1;
    let probe = Arc::new(Probe::default());
    let registration = Registration::new(name, &probe);
    let context = StreamOperatorContext::new(&job, name, None);
    let partial = {
        let work = operator.prepare_checkpoint_async(&context);
        tokio::pin!(work);
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let mut context = Context::from_waker(Waker::noop());
                match work.as_mut().poll(&mut context) {
                    Poll::Ready(result) => {
                        result.unwrap();
                        break false;
                    }
                    Poll::Pending if !probe.arrays.lock().unwrap().is_empty() => break true,
                    Poll::Pending => tokio::task::yield_now().await,
                }
            }
        })
        .await
        .unwrap()
    };
    wait_arrays_released(&probe).await;
    drop(registration);
    let paid = pool.reserved();
    let uninstalled = operator.compact.as_ref().unwrap().capture.is_none();
    let kept = source.iter().all(|owner| owner.upgrade().is_some());
    let after = operator.checkpoint(Epoch::INITIAL).unwrap();
    same_snapshot(&expected, &after);
    assert!(
        partial,
        "direct async export completed synchronously before a real partial Arrow boundary"
    );
    assert!(uninstalled);
    assert!(kept);
    assert_eq!(paid, before);
}

#[tokio::test(flavor = "current_thread")]
async fn test_direct_compact_encoder_drop_keeps_actual_arrow_credit_until_release() {
    let name = "direct-compact-encoder-drop";
    let (mut operator, job, expected, pool) = seeded(name).await;
    let before = pool.reserved();
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let release = Release::new(gate.clone());
    let (entered, started) = tokio::sync::oneshot::channel();
    let probe = Arc::new(Probe {
        gate: Some(gate),
        entered: Mutex::new(Some(entered)),
        ..Probe::default()
    });
    let registration = Registration::new(name, &probe);
    let observer = thread::current().id();
    let context = StreamOperatorContext::new(&job, name, None);
    let pending = {
        let work = operator.prepare_checkpoint_async(&context);
        tokio::pin!(work);
        tokio::select! {
            result = &mut work => { result.unwrap(); false },
            result = started => { result.unwrap(); true },
        }
    };
    let held = pool.reserved();
    let arrow_alive = probe
        .arrays
        .lock()
        .unwrap()
        .iter()
        .any(|array| array.upgrade().is_some());
    drop(release);
    if pending {
        wait_encoder_released(&probe, pool.as_ref(), before).await;
    } else {
        wait_arrays_released(&probe).await;
    }
    drop(registration);
    let refunded = pool.reserved();
    let uninstalled = operator.compact.as_ref().unwrap().capture.is_none();
    let encoder = probe.encoder.lock().unwrap().unwrap();
    same_snapshot(&expected, &operator.checkpoint(Epoch::INITIAL).unwrap());
    assert!(
        pending,
        "direct native encoder blocked its observer until cleanup release"
    );
    assert_ne!(encoder.0, observer);
    assert!(encoder.1 > 0 && held > before);
    assert!(arrow_alive);
    assert_eq!(refunded, before);
    assert!(uninstalled);
    assert!(!job.cancellation().is_cancelled());
}

#[tokio::test(flavor = "current_thread")]
async fn test_direct_compact_large_string_export_yields_inside_one_value() {
    let name = "direct-compact-string-fragment";
    let wide = "€\0".repeat(1 << 19);
    let record = RecordBatch::try_from_iter(vec![
        (
            "key",
            Arc::new(datafusion::arrow::array::StringArray::from(vec![
                Some(wide.as_str()),
                None,
            ])) as ArrayRef,
        ),
        (
            "value",
            Arc::new(Int64Array::from(vec![Some(7), None])) as ArrayRef,
        ),
    ])
    .unwrap();
    let batch = Batch::table(
        vec![record],
        BatchMetadata::new(name, 0, JsonMap::from([("case".into(), json!(name))])).unwrap(),
    )
    .unwrap();
    let (mut operator, job, expected, pool) = seeded_batch(name, batch).await;
    let before = pool.reserved();
    let probe = Arc::new(Probe::default());
    let registration = Registration::new(name, &probe);
    let context = StreamOperatorContext::new(&job, name, None);
    let partial = {
        let work = operator.prepare_checkpoint_async(&context);
        tokio::pin!(work);
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let mut context = Context::from_waker(Waker::noop());
                match work.as_mut().poll(&mut context) {
                    Poll::Ready(result) => {
                        result.unwrap();
                        break false;
                    }
                    Poll::Pending if !probe.copied_bytes.lock().unwrap().is_empty() => break true,
                    Poll::Pending => tokio::task::yield_now().await,
                }
            }
        })
        .await
        .unwrap()
    };
    wait_arrays_released(&probe).await;
    drop(registration);
    let paid = pool.reserved();
    let uninstalled = operator.compact.as_ref().unwrap().capture.is_none();
    let copied = probe.copied_bytes.lock().unwrap().clone();
    let encoding = probe.encoder.lock().unwrap().is_some();
    same_snapshot(&expected, &operator.checkpoint(Epoch::INITIAL).unwrap());
    assert!(
        partial,
        "whole variable-width value was copied before a cooperative boundary"
    );
    assert!(!copied.is_empty() && copied.iter().all(|&bytes| bytes <= 64 << 10));
    assert!(!encoding);
    assert!(uninstalled);
    assert_eq!(paid, before);
}

use super::*;
use crate::{StateSegment, operator::join::PendingLog};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryLimit, MemoryPool};
use serde_json::{Value, json};
use std::{collections::BTreeMap, sync::Mutex};

#[path = "rebase_admission_tests.rs"]
mod rebase_admission_tests;

#[derive(Clone, Copy, Debug)]
struct Admission {
    reservation: usize,
    before: usize,
    additional: usize,
}

#[derive(Debug, Default)]
struct Fault {
    armed: bool,
    reject: Option<usize>,
    admissions: Vec<Admission>,
}

#[derive(Debug)]
struct CapturePool {
    inner: GreedyMemoryPool,
    fault: Mutex<Fault>,
}

impl CapturePool {
    fn new() -> Self {
        Self {
            inner: GreedyMemoryPool::new(16 << 20),
            fault: Mutex::new(Fault::default()),
        }
    }

    fn arm(&self, reject: Option<usize>) {
        *self.fault.lock().unwrap() = Fault {
            armed: true,
            reject,
            admissions: Vec::new(),
        };
    }

    fn finish(&self) -> Vec<Admission> {
        std::mem::take(&mut *self.fault.lock().unwrap()).admissions
    }
}

impl std::fmt::Display for CapturePool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, formatter)
    }
}

impl MemoryPool for CapturePool {
    fn name(&self) -> &'static str {
        "pending-capture-fault"
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.inner.grow(reservation, additional);
    }

    fn shrink(&self, reservation: &MemoryReservation, bytes: usize) {
        self.inner.shrink(reservation, bytes);
    }

    fn try_grow(
        &self,
        reservation: &MemoryReservation,
        additional: usize,
    ) -> datafusion::common::Result<()> {
        let mut fault = self.fault.lock().unwrap();
        if fault.armed && additional > 0 {
            fault.admissions.push(Admission {
                reservation: std::ptr::from_ref(reservation) as usize,
                before: reservation.size(),
                additional,
            });
            if fault.reject == Some(fault.admissions.len()) {
                return Err(datafusion::common::DataFusionError::ResourcesExhausted(
                    "pending capture admission rejected".into(),
                ));
            }
        }
        drop(fault);
        self.inner.try_grow(reservation, additional)
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }

    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
}

async fn add_pending(join: &mut StreamJoinOperator, context: &StreamOperatorContext<'_>) {
    for side in ["left", "right"] {
        let mut output = EdgeCollector::new(join.output_ports().to_vec());
        join.process_data(
            side,
            Batch::table(vec![record(side)], BatchMetadata::default()).unwrap(),
            context,
            &mut output,
        )
        .await
        .unwrap();
    }
}

async fn pending_capture(
    context: &StreamOperatorContext<'_>,
    pool: Arc<dyn MemoryPool>,
) -> (StreamJoinOperator, OperatorStateSnapshot) {
    let mut join = operator();
    join.runtime.initialize_benchmark_pool(pool).unwrap();
    add_pending(&mut join, context).await;
    let old = join.checkpoint(Epoch::INITIAL).unwrap();
    assert!(join.state.deltas.pending.is_empty());
    add_pending(&mut join, context).await;
    assert_eq!(join.state.deltas.pending.iter().count(), 3);
    assert!(join.v2_writer.prepared.is_none());
    assert!(join.v2_writer.prepared_delta.is_none());
    (join, old)
}

fn segment_image(segment: &StateSegment) -> Value {
    json!({
        "owner": Arc::as_ptr(&segment.bytes_arc()) as usize,
        "bytes": segment.bytes(),
        "sha256": segment.sha256(),
    })
}

fn pending_op_image(operation: &PendingOp) -> Value {
    match operation {
        PendingOp::Upsert {
            side,
            row_id,
            event_time,
            encoded_key,
            record,
            charge,
        } => json!({
            "side": side.as_str(), "row_id": row_id, "at": event_time.as_micros(),
            "key": encoded_key.as_slice(), "key_owner": Arc::as_ptr(encoded_key) as usize,
            "columns": record.columns().as_ptr() as usize, "offset": record.offset(),
            "charge": charge,
        }),
        PendingOp::Tombstone {
            side,
            row_id,
            event_time,
            encoded_key,
        } => json!({
            "side": side.as_str(), "row_id": row_id, "at": event_time.as_micros(),
            "key": encoded_key.as_slice(), "key_owner": Arc::as_ptr(encoded_key) as usize,
        }),
    }
}

fn pending_image(pending: &PendingLog) -> Value {
    let slots = pending
        .slots
        .iter()
        .map(|entry| {
            entry.as_ref().map(|entry| json!({
        "op": pending_op_image(&entry.op), "previous": entry.previous, "next": entry.next,
    }))
        })
        .collect::<Vec<_>>();
    let upserts = pending
        .upserts
        .iter()
        .map(|((side, row), slot)| ((side.as_str(), *row), *slot))
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .collect::<Vec<_>>();
    json!({
        "slots": slots, "slots_owner": pending.slots.as_ptr() as usize,
        "free": pending.free, "head": pending.head, "tail": pending.tail,
        "upserts": upserts,
    })
}

fn row_image(row: &StoredRow) -> Value {
    json!({
        "row_id": row.row_id, "at": row.event_time.as_micros(), "charge": row.charge,
        "key": row.encoded_key.as_slice(), "key_owner": Arc::as_ptr(&row.encoded_key) as usize,
        "columns": row.record.columns().as_ptr() as usize, "offset": row.record.offset(),
    })
}

fn capture_image(join: &StreamJoinOperator) -> Value {
    let state = &join.state;
    let writer = &join.v2_writer;
    let base = writer.base.as_ref().unwrap();
    let segments = base
        .segments
        .iter()
        .map(|(name, segment)| (name, segment_image(segment)))
        .collect::<BTreeMap<_, _>>();
    let rows = [&state.left, &state.right].map(|rows| {
        json!({
            "owner": Arc::as_ptr(&rows.0) as usize,
            "index": rows.1.as_ref().map(|index| Arc::as_ptr(index) as usize),
            "rows": rows.iter().map(row_image).collect::<Vec<_>>(),
        })
    });
    json!({
        "rows": rows, "metrics": state.metrics, "ended": state.ended,
        "next": [state.next_left_row_id, state.next_right_row_id, state.next_output_sequence],
        "epoch": state.last_checkpoint_epoch.map(Epoch::as_u64),
        "tracker": [writer.tracker.generation, writer.tracker.revision, writer.tracker.dirty_revision],
        "cut": format!("{:?}", writer.tracker.snapshot(state.last_checkpoint_epoch)),
        "pending": pending_image(&state.deltas.pending),
        "base": segments, "base_credit": std::ptr::from_ref(base.credit()) as usize,
        "base_funded": base.credit().size(), "payloads": base.payloads,
        "base_epoch": writer.base_epoch, "deltas": writer.deltas, "dirty_epochs": writer.dirty_epochs,
        "migration_required": writer.migration_required,
        "prepared": writer.prepared.is_some(), "prepared_delta": writer.prepared_delta.is_some(),
        "v1_base": state.deltas.base.iter().map(|(name, segment)| (name, segment_image(segment))).collect::<Vec<_>>(),
        "v1_deltas": state.deltas.segments.iter().map(|(name, segment)| (name, segment_image(segment))).collect::<Vec<_>>(),
        "v1_count": state.deltas.segments_since_base, "v1_compaction": state.deltas.needs_compaction,
    })
}

fn capture_records(join: &StreamJoinOperator) -> Vec<RecordBatch> {
    join.state
        .left
        .iter()
        .chain(join.state.right.iter())
        .map(|row| row.record.view().clone())
        .chain(join.state.deltas.pending.iter().filter_map(|op| {
            if let PendingOp::Upsert { record, .. } = op {
                Some(record.view().clone())
            } else {
                None
            }
        }))
        .collect()
}

fn snapshot_wire(snapshot: &OperatorStateSnapshot) -> (JsonMap, BTreeMap<String, Vec<u8>>) {
    (
        snapshot.inline_metadata.clone(),
        snapshot
            .segments
            .iter()
            .map(|(name, segment)| (name.clone(), segment.bytes().to_vec()))
            .collect(),
    )
}

async fn capture_reference(
    service: &crate::runtime::streaming::gather_work::TestService,
) -> (Vec<Admission>, (JsonMap, BTreeMap<String, Vec<u8>>)) {
    let pool = Arc::new(CapturePool::new());
    let job = job().with_gather_owner(service.owner("pending-capture-reference".into()));
    let context = StreamOperatorContext::new(&job, "prepared-reuse", None);
    let (mut join, old) = pending_capture(&context, pool.clone()).await;
    let base_owner = std::ptr::from_ref(join.v2_writer.base.as_ref().unwrap().credit()) as usize;
    pool.arm(None);
    let snapshot = join.checkpoint(Epoch::new(2).unwrap()).unwrap();
    let admissions = pool.finish();
    assert!(
        admissions.len() > 8,
        "both-side selection and encoding must precede capture controls"
    );
    let [metadata, clone, merge] = &admissions[admissions.len() - 3..] else {
        unreachable!()
    };
    assert_eq!(metadata.reservation, clone.reservation);
    assert_eq!(clone.before, metadata.before + metadata.additional);
    assert_eq!(
        merge.reservation, base_owner,
        "last admission installs into the old Base"
    );
    assert_eq!(
        clone.additional,
        segment_copy_bytes(join.v2_writer.base.as_ref().unwrap(), None).unwrap()
    );
    let expected = snapshot_wire(&snapshot);
    drop((snapshot, old, join));
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
    (admissions, expected)
}

#[test]
fn test_pending_capture_admission_failures_preserve_state_and_allow_same_epoch_retry() {
    isolated_writer_test(|service, runtime| {
        runtime.block_on(assert_pending_capture_admission_failures(service));
    });
}

async fn assert_pending_capture_admission_failures(
    service: &crate::runtime::streaming::gather_work::TestService,
) {
    let (admissions, expected) = capture_reference(service).await;
    for reject in 1..=admissions.len() {
        let pool = Arc::new(CapturePool::new());
        let job = job().with_gather_owner(service.owner("pending-capture-rejection".into()));
        let context = StreamOperatorContext::new(&job, "prepared-reuse", None);
        let (mut join, old) = pending_capture(&context, pool.clone()).await;
        let before = capture_image(&join);
        let records = capture_records(&join);
        let status = join.status();
        let tracker = join.v2_writer.tracker;
        let published = snapshot_wire(&old);
        let reserved = pool.reserved();
        pool.arm(Some(reject));
        let result = join.checkpoint(Epoch::new(2).unwrap());
        let observed = pool.finish();
        assert!(result.is_err(), "admission {reject} must reject capture");
        assert_eq!(observed.len(), reject);
        assert_eq!(
            observed.last().unwrap().additional,
            admissions[reject - 1].additional
        );
        assert_eq!(capture_image(&join), before, "admission {reject}");
        assert_eq!(capture_records(&join), records, "admission {reject}");
        assert_eq!(join.status(), status);
        assert_eq!(snapshot_wire(&old), published);
        assert_eq!(
            pool.reserved(),
            reserved,
            "admission {reject} leaked scratch"
        );

        let retry = join.checkpoint(Epoch::new(2).unwrap()).unwrap();
        assert_eq!(
            snapshot_wire(&retry),
            expected,
            "retry after admission {reject}"
        );
        assert!(join.state.deltas.pending.is_empty());
        assert_eq!(
            join.state.last_checkpoint_epoch,
            Some(Epoch::new(2).unwrap())
        );
        assert_eq!(join.v2_writer.tracker, tracker.advanced(true).unwrap());
        assert_eq!(join.status(), status);
        assert_eq!(snapshot_wire(&old), published);
        drop((retry, old, join, records));
        drop(context);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(job);
        assert_eq!(
            pool.reserved(),
            0,
            "admission {reject} retained credit after teardown"
        );
    }
}

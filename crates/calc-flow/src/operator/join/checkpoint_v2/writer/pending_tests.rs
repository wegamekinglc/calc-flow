use std::cell::Cell;

use super::*;
use crate::operator::join::{PendingLog, StateRows, Tombstones, columnar::FramedKey};

fn log(operations: impl IntoIterator<Item = PendingOp>) -> PendingLog {
    let mut pending = PendingLog::default();
    for operation in operations {
        pending.push(operation);
    }
    pending
}

fn identities(pending: &PendingLog) -> Vec<(JoinSide, u64)> {
    pending.iter().map(PendingOp::identity).collect()
}

#[test]
fn test_pending_cancellation_precedes_encoder_admission() {
    let (rows, record) = fixture();
    let pending = log([pending_upsert(JoinSide::Left, &rows[0])]);
    let (pool, workspace) = reservation(0);
    let result = pending_ops(
        &pending,
        &[record.schema(), record.schema()],
        Epoch::INITIAL,
        &workspace,
        &|| {
            Err(CalcFlowError::Cancelled {
                run_id: "pending-first-check".into(),
            })
        },
    );
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(pool.reserved(), 0);
}

fn assert_same_segments(left: &Base, right: &Base) {
    assert_eq!(
        serde_json::to_value(&left.payloads).unwrap(),
        serde_json::to_value(&right.payloads).unwrap()
    );
    assert_eq!(left.segments.len(), right.segments.len());
    for (name, segment) in &left.segments {
        assert_eq!(segment.bytes(), right.segments[name].bytes(), "{name}");
        assert_eq!(segment.sha256(), right.segments[name].sha256());
    }
}

#[test]
fn test_pending_borrows_rows_within_compact_workspace() {
    let (rows, record) = flat_ipc_fixture(8_192);
    let pending = log(rows
        .iter()
        .map(|row| pending_upsert(JoinSide::Left, row))
        .chain(
            rows[..64]
                .iter()
                .map(|row| pending_upsert(JoinSide::Right, row)),
        ));
    let original = identities(&pending);
    // The old copies alone reserve 3 * 8_256 * 80 + 3 * 8_192 * 72 bytes.
    // The existing compact/IPC phase bound fits this unchanged 2.4 MB pool.
    let (pool, workspace) = reservation(2_400_000);
    let output = pending_ops(
        &pending,
        &[record.schema(), record.schema()],
        Epoch::INITIAL,
        &workspace,
        &|| Ok(()),
    )
    .expect("unprepared pending encoding must not copy its owned log and rows");
    assert_eq!(identities(&pending), original);
    assert_eq!(output.payloads.len(), 2);
    assert_eq!(output.segments.len(), 4);
    assert_eq!(
        workspace.size(),
        0,
        "side-local scratch must retire before return"
    );
    drop(output);
    assert_eq!(pool.reserved(), 0, "the caller workspace remains alive");
}

fn mixed_log(rows: &[StoredRow]) -> PendingLog {
    let mut pending = log([
        pending_upsert(JoinSide::Right, &rows[2]),
        pending_upsert(JoinSide::Left, &rows[1]),
        pending_upsert(JoinSide::Left, &rows[0]),
    ]);
    assert!(pending.remove_upsert((JoinSide::Left, rows[1].row_id)));
    pending.push(PendingOp::Tombstone {
        side: JoinSide::Right,
        row_id: 100,
        event_time: EventTime::from_micros(-41),
        encoded_key: Arc::clone(&rows[1].encoded_key),
    });
    pending.push(pending_upsert(JoinSide::Left, &rows[1]));
    pending
}

#[test]
fn test_pending_cancellation_releases_each_phase_without_consuming_log() {
    let (rows, record) = fixture();
    let pending = mixed_log(&rows);
    let schemas = [record.schema(), record.schema()];
    let original = identities(&pending);
    let epoch = Epoch::new(7).unwrap();
    let calls = Cell::new(0);
    let (_, reference_credit) = reservation(16 << 20);
    let expected = pending_ops(&pending, &schemas, epoch, &reference_credit, &|| {
        calls.set(calls.get() + 1);
        Ok(())
    })
    .unwrap();
    for stop in 1..=calls.get() {
        let (pool, workspace) = reservation(16 << 20);
        let checked = Cell::new(0);
        let result = pending_ops(&pending, &schemas, epoch, &workspace, &|| {
            checked.set(checked.get() + 1);
            if checked.get() == stop {
                Err(CalcFlowError::Cancelled {
                    run_id: "pending-borrow".into(),
                })
            } else {
                Ok(())
            }
        });
        assert!(
            matches!(result, Err(CalcFlowError::Cancelled { .. })),
            "check {stop}"
        );
        assert_eq!(identities(&pending), original);
        assert_eq!(pool.reserved(), 0, "scratch must retire at check {stop}");
    }
    let (pool, workspace) = reservation(16 << 20);
    let retry = pending_ops(&pending, &schemas, epoch, &workspace, &|| Ok(())).unwrap();
    assert_same_segments(&expected, &retry);
    drop(retry);
    assert_eq!(pool.reserved(), 0);
}

fn expected_segments(pending: &PendingLog, epoch: Epoch) -> BTreeMap<String, Vec<u8>> {
    let mut segments = BTreeMap::new();
    for side in [JoinSide::Left, JoinSide::Right] {
        let mut rows = pending
            .iter()
            .filter_map(|op| match op {
                PendingOp::Upsert {
                    side: actual,
                    row_id,
                    event_time,
                    encoded_key,
                    record,
                    charge,
                } if *actual == side => Some(StoredRow {
                    record: record.clone(),
                    event_time: *event_time,
                    row_id: *row_id,
                    charge: *charge,
                    encoded_key: Arc::clone(encoded_key),
                }),
                _ => None,
            })
            .collect::<Vec<_>>();
        rows.sort_unstable_by_key(|row| row.row_id);
        let mut tombstones = pending
            .iter()
            .filter(|op| matches!(op, PendingOp::Tombstone { side: actual, .. } if *actual == side))
            .collect::<Vec<_>>();
        tombstones.sort_unstable_by_key(|op| op.identity().1);
        if rows.is_empty() && tombstones.is_empty() {
            continue;
        }
        let digest = expected_payload(&rows, side, &mut segments);
        let (_, credit) = reservation(16 << 20);
        segments.insert(
            format!("{}-delta-{}", side.as_str(), epoch.as_u64()),
            geometric_delta(
                &rows.iter().collect::<Vec<_>>(),
                &tombstones,
                side,
                epoch,
                &digest,
                &credit,
            ),
        );
    }
    segments
}

fn expected_payload(
    rows: &[StoredRow],
    side: JoinSide,
    segments: &mut BTreeMap<String, Vec<u8>>,
) -> [u8; 32] {
    if rows.is_empty() {
        [0; 32]
    } else {
        let records = rows
            .iter()
            .map(|row| row.record.view().clone())
            .collect::<Vec<_>>();
        let record =
            datafusion::arrow::compute::concat_batches(&records[0].schema(), &records).unwrap();
        let ipc = stream_writer_bytes(&record);
        let mut payload = header(*PAYLOAD_MAGIC, side).to_vec();
        payload.extend(u64::try_from(rows.len()).unwrap().to_le_bytes());
        payload.extend(u64::try_from(ipc.len()).unwrap().to_le_bytes());
        for row in rows {
            payload.extend(row.row_id.to_le_bytes());
        }
        payload.extend(ipc);
        let digest: [u8; 32] = Sha256::digest(&payload).into();
        segments.insert(
            format!("{}-payload-{}", side.as_str(), hex::encode(digest)),
            payload,
        );
        digest
    }
}

fn sliced_rich_rows(legacy: bool) -> (Vec<StoredRow>, RecordBatch) {
    let (_, record) = rich_ipc_fixture();
    let padded =
        datafusion::arrow::compute::concat_batches(&record.schema(), [&record, &record]).unwrap();
    let parent = Arc::new(padded.slice(1, 4));
    let rows = [2, 0, 1]
        .into_iter()
        .enumerate()
        .map(|(id, index)| StoredRow {
            record: if legacy {
                RowPayload::Legacy(parent.slice(index, 1))
            } else {
                RowPayload::Rowed {
                    parent: Arc::clone(&parent),
                    row: index,
                }
            },
            event_time: EventTime::from_micros(i64::try_from(id).unwrap() - 20),
            row_id: u64::try_from(7 + id * 11).unwrap(),
            charge: state_row_charge(&parent, index, &[0, 2], "pending-rich").unwrap(),
            encoded_key: Arc::new(encode_join_key_v1(&parent, index, &[0, 2]).unwrap().into()),
        })
        .collect();
    (rows, parent.as_ref().clone())
}

#[test]
fn test_pending_borrowed_and_owned_paths_preserve_rich_wire_and_slot_order() {
    for legacy in [false, true] {
        let (rows, record) = sliced_rich_rows(legacy);
        for pending in [
            PendingLog::default(),
            log([pending_upsert(JoinSide::Left, &rows[1])]),
            log([PendingOp::Tombstone {
                side: JoinSide::Right,
                row_id: 81,
                event_time: EventTime::from_micros(-99),
                encoded_key: Arc::clone(&rows[2].encoded_key),
            }]),
            mixed_log(&rows),
        ] {
            let epoch = Epoch::new(17).unwrap();
            let expected = expected_segments(&pending, epoch);
            let (pool, workspace) = reservation(16 << 20);
            let schemas = [record.schema(), record.schema()];
            let output = pending_ops(&pending, &schemas, epoch, &workspace, &|| Ok(())).unwrap();
            assert_eq!(output.segments.len(), expected.len());
            for (name, bytes) in &expected {
                assert_eq!(
                    output.segments[name].bytes(),
                    bytes,
                    "{name}, legacy={legacy}"
                );
            }
            let owned = pending.iter().cloned().collect::<Vec<_>>();
            let owned_output =
                pending_owned(&owned, &schemas, epoch, &workspace, &|| Ok(())).unwrap();
            assert_same_segments(&output, &owned_output);
            drop((output, owned_output));
            assert_eq!(pool.reserved(), 0);
        }
    }
}

#[test]
fn test_pending_allocations_and_surviving_segment_backing_remain_funded() {
    let (rows, record) = flat_ipc_fixture(512);
    let pending = log(rows.iter().flat_map(|row| {
        [
            pending_upsert(JoinSide::Left, row),
            pending_upsert(JoinSide::Right, row),
        ]
    }));
    let tracking = Arc::new(IpcPool::new(2_400_000));
    let pool: Arc<dyn MemoryPool> = tracking.clone();
    let workspace = MemoryConsumer::new("pending-allocation").register(&pool);
    let mut output = None;
    let allocations = allocation_counter::measure(|| {
        output = Some(pending_ops(
            &pending,
            &[record.schema(), record.schema()],
            Epoch::INITIAL,
            &workspace,
            &|| Ok(()),
        ));
    });
    let output = output.unwrap().unwrap();
    assert!(usize::try_from(allocations.bytes_max).unwrap() <= tracking.peak());
    assert!(usize::try_from(allocations.bytes_current).unwrap() <= pool.reserved());
    assert_eq!(workspace.size(), 0);
    let segment = output.segments["left-delta-1"].clone();
    let expected = segment.bytes().to_vec();
    drop(output);
    assert_eq!(
        pool.reserved(),
        segment_charge(segment.bytes_arc().capacity()).unwrap()
    );
    assert_eq!(segment.bytes(), expected);
    drop(segment);
    assert_eq!(pool.reserved(), 0);
}

/// The byte-equivalence oracle (spec FR4): for identical admission, eviction,
/// and capture schedules the derived encoding must equal the old
/// pending-log encoding segment for segment, digest for digest.
#[test]
fn test_derived_delta_encoding_matches_the_pending_log_oracle() {
    let (rows, record) = fixture();
    let schemas = [record.schema(), record.schema()];
    let epoch = Epoch::new(11).unwrap();
    let check = &|| Ok(());
    for schedule in oracle_schedules(&rows) {
        let OracleSchedule {
            live,
            watermark,
            tombstones,
        } = schedule;
        // The shadow pending log the pre-change bookkeeping would have held:
        // one upsert per live row above the watermark, one tombstone per
        // recorded entry.
        let mut oracle = PendingLog::default();
        for row in live.iter().filter(|row| row.row_id >= watermark) {
            oracle.push(pending_upsert(JoinSide::Left, row));
        }
        for (row_id, event_time, key) in &tombstones {
            oracle.push(PendingOp::Tombstone {
                side: JoinSide::Left,
                row_id: *row_id,
                event_time: *event_time,
                encoded_key: Arc::clone(key),
            });
        }
        let mut list = Tombstones::default();
        for (row_id, event_time, key) in &tombstones {
            list.push(*row_id, *event_time, Arc::clone(key));
        }
        let state = StateRows::from(live.clone());
        let upserts = live.iter().filter(|row| row.row_id >= watermark).count();
        let left = SideDelta {
            rows: &state,
            watermark,
            upserts,
            tombstones: &list,
        };
        let empty = StateRows::default();
        let empty_list = Tombstones::default();
        let right = SideDelta {
            rows: &empty,
            watermark: u64::MAX,
            upserts: 0,
            tombstones: &empty_list,
        };
        let (pool, workspace) = reservation(16 << 20);
        let derived = dirty(&[left, right], &schemas, epoch, &workspace, check).unwrap();
        let (oracle_pool, oracle_workspace) = reservation(16 << 20);
        let expected = pending_ops(&oracle, &schemas, epoch, &oracle_workspace, check).unwrap();
        assert_same_segments(&derived, &expected);
        drop((derived, expected, list, state, empty, empty_list));
        assert_eq!(pool.reserved(), 0);
        assert_eq!(oracle_pool.reserved(), 0);
    }
}

struct OracleSchedule {
    live: Vec<StoredRow>,
    watermark: u64,
    tombstones: Vec<(u64, EventTime, Arc<FramedKey>)>,
}

/// Admission, eviction before and after capture, id gaps, and tombstone-only
/// windows (spec FR4/AC1).
fn oracle_schedules(rows: &[StoredRow]) -> Vec<OracleSchedule> {
    let tombstone = |row: &StoredRow| (row.row_id, row.event_time, Arc::clone(&row.encoded_key));
    vec![
        // Fresh admission: everything sits above the zero watermark.
        OracleSchedule {
            live: rows.to_vec(),
            watermark: 0,
            tombstones: Vec::new(),
        },
        // Captured rows below the watermark stay out of the derive while a
        // later batch remains dirty.
        OracleSchedule {
            live: rows.to_vec(),
            watermark: 21,
            tombstones: Vec::new(),
        },
        // A captured row was evicted: its tombstone rides the same delta as
        // the uncaptured upserts.
        OracleSchedule {
            live: rows[1..].to_vec(),
            watermark: 21,
            tombstones: vec![tombstone(&rows[0])],
        },
        // Only evictions since the capture: a tombstone-only delta.
        OracleSchedule {
            live: Vec::new(),
            watermark: 39,
            tombstones: vec![tombstone(&rows[0]), tombstone(&rows[2])],
        },
        // Monotone-with-gaps ids (spec FR15) above the watermark.
        OracleSchedule {
            live: vec![rows[2].clone()],
            watermark: 22,
            tombstones: vec![tombstone(&rows[0])],
        },
    ]
}

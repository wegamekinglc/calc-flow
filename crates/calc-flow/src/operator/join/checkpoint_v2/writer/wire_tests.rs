use super::*;
use crate::operator::join::{
    PendingOp, columnar::RowPayload, encode_join_key_v1, state_row_charge,
};
use crate::{CalcFlowError, Epoch, EventTime};
use datafusion::arrow::{
    array::{Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};

fn reservation(limit: usize) -> (Arc<dyn MemoryPool>, MemoryReservation) {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
    let credit = MemoryConsumer::new("exact-wire-test").register(&pool);
    (pool, credit)
}

fn fixture() -> (Vec<StoredRow>, RecordBatch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("tag", DataType::Utf8, false),
        Field::new("value", DataType::Int64, true),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![7, 19, 31])),
        Arc::new(TimestampMicrosecondArray::from(vec![-17, 0, 42])),
        Arc::new(StringArray::from(vec!["a", "seven", "longer 猫 key"])),
        Arc::new(Int64Array::from(vec![Some(1), None, Some(-4)])),
    ];
    let record = RecordBatch::try_new(schema, columns).unwrap();
    let parent = Arc::new(record.clone());
    let rows = [(7, -17), (21, 0), (39, 42)]
        .into_iter()
        .enumerate()
        .map(|(index, (row_id, time))| StoredRow {
            record: RowPayload::Rowed {
                parent: Arc::clone(&parent),
                row: index,
            },
            event_time: EventTime::from_micros(time),
            row_id,
            charge: state_row_charge(&record, index, &[0, 2], "exact-wire").unwrap(),
            encoded_key: Arc::new(encode_join_key_v1(&record, index, &[0, 2]).unwrap().into()),
        })
        .collect();
    (rows, record)
}

fn geometric_base(
    rows: &[&StoredRow],
    side: JoinSide,
    digest: &[u8; 32],
    credit: &MemoryReservation,
) -> Vec<u8> {
    let mut bytes = PaidBuffer::new(credit, 0);
    let mut fixed = [0; 32];
    fixed[..16].copy_from_slice(&header(*BASE_MAGIC, side));
    fixed[16..24].copy_from_slice(&integer(rows.len()).unwrap());
    bytes.append(&fixed).unwrap();
    for (position, row) in rows.iter().enumerate() {
        upsert(&mut bytes, row, digest, position).unwrap();
    }
    bytes.into_bytes().unwrap()
}

fn geometric_delta(
    rows: &[&StoredRow],
    tombstones: &[&PendingOp],
    side: JoinSide,
    epoch: Epoch,
    digest: &[u8; 32],
    credit: &MemoryReservation,
) -> Vec<u8> {
    let mut bytes = PaidBuffer::new(credit, 0);
    let mut fixed = [0; 40];
    fixed[..16].copy_from_slice(&header(*b"CFJDIX2\0", side));
    fixed[16..24].copy_from_slice(&epoch.as_u64().to_le_bytes());
    fixed[24..32].copy_from_slice(&integer(rows.len()).unwrap());
    fixed[32..40].copy_from_slice(&integer(tombstones.len()).unwrap());
    bytes.append(&fixed).unwrap();
    for (position, row) in rows.iter().enumerate() {
        upsert(&mut bytes, row, digest, position).unwrap();
    }
    for op in tombstones {
        let PendingOp::Tombstone {
            row_id,
            event_time,
            encoded_key,
            ..
        } = op
        else {
            panic!("fixture contains only tombstones");
        };
        bytes.append(&row_id.to_le_bytes()).unwrap();
        bytes.append(&event_time.as_micros().to_le_bytes()).unwrap();
        bytes.append(&integer(encoded_key.len()).unwrap()).unwrap();
        bytes.append(encoded_key).unwrap();
    }
    bytes.into_bytes().unwrap()
}

#[test]
fn test_base_index_exact_capacity_preserves_geometric_wire_bytes() {
    let (rows, _) = fixture();
    let references = rows.iter().collect::<Vec<_>>();
    let digest = [0xA5; 32];
    let length = 32
        + rows
            .iter()
            .map(|row| 72 + row.encoded_key.len())
            .sum::<usize>();
    assert!(length > 256 && length < 512);
    let (reference_pool, reference_credit) = reservation(4 << 20);
    let expected = geometric_base(&references, JoinSide::Right, &digest, &reference_credit);
    let retained = 73;
    let (pool, credit) = reservation(retained + length);
    credit.try_grow(retained).unwrap();
    let mut result = None;
    let allocations = allocation_counter::measure(|| {
        result = Some(encode_index(
            &references,
            JoinSide::Right,
            &digest,
            &credit,
            retained,
            &|| Ok(()),
        ));
    });
    let bytes = result
        .unwrap()
        .expect("one exact index allocation must fit where geometric growth cannot");
    assert_eq!(bytes, expected);
    assert_eq!((bytes.len(), bytes.capacity()), (length, length));
    assert_eq!(allocations.count_total, 1);
    assert!(allocations.bytes_max <= u64::try_from(credit.size()).unwrap());
    assert_eq!(credit.size(), retained + length);
    drop((bytes, expected));
    drop((credit, reference_credit));
    assert_eq!((pool.reserved(), reference_pool.reserved()), (0, 0));

    let (empty_pool, empty_credit) = reservation(32);
    let empty = encode_index(&[], JoinSide::Left, &[0; 32], &empty_credit, 0, &|| Ok(())).unwrap();
    assert_eq!((empty.len(), empty.capacity()), (32, 32));
    assert_eq!(&empty[..16], &header(*BASE_MAGIC, JoinSide::Left));
    assert_eq!(&empty[16..], &[0; 16]);
    drop((empty, empty_credit));
    assert_eq!(empty_pool.reserved(), 0);
}

#[test]
fn test_payload_frame_exact_capacity_preserves_ipc_and_segment_owner() {
    let (rows, record) = fixture();
    let references = rows.iter().collect::<Vec<_>>();
    let (reference_pool, reference_credit) = reservation(4 << 20);
    let scratch = super::super::budget::stream_scratch(&record).unwrap();
    ensure(&reference_credit, scratch).unwrap();
    let mut ipc = PaidBuffer::new(&reference_credit, scratch);
    write_stream(&mut ipc, &record, &|| Ok(())).unwrap();
    let ipc = ipc.into_bytes().unwrap();
    let length = 32 + 8 * rows.len() + ipc.len();
    let mut reference = PaidBuffer::new(&reference_credit, scratch + ipc.capacity());
    append_payload(
        &mut reference,
        &references,
        JoinSide::Left,
        &ipc,
        &|| Ok(()),
    )
    .unwrap();
    let expected = reference.into_bytes().unwrap();
    let (pool, workspace) = reservation(4 << 20);
    let mut encoder = Encoder::new(&workspace).unwrap();
    let live = encoder.retained + ipc.capacity();
    ensure(&encoder.funding, live).unwrap();
    let mut result = None;
    let allocations = allocation_counter::measure(|| {
        result = Some(encoder.frame_payload(&references, JoinSide::Left, ipc, &|| Ok(())));
    });
    let digest = result.unwrap().unwrap();
    let name = format!("left-payload-{}", hex::encode(digest));
    let segment = encoder.output.segments[&name].clone();
    assert_eq!(segment.bytes(), expected);
    assert_eq!(
        (segment.bytes().len(), segment.bytes_arc().capacity()),
        (length, length)
    );
    assert_eq!(segment.sha256(), hex::encode(Sha256::digest(&expected)));
    assert!(
        encoder.funding.size() >= live + length,
        "live IPC and exact output must be prepaid together"
    );
    assert!(allocations.bytes_max <= u64::try_from(encoder.funding.size()).unwrap());
    assert!(usize::try_from(allocations.bytes_current).unwrap() <= encoder.funding.size());
    let owner = Arc::downgrade(&encoder.funding);
    drop((encoder, workspace));
    assert!(owner.upgrade().is_some() && pool.reserved() > 0);
    assert_eq!(segment.bytes(), expected);
    drop(segment);
    assert!(owner.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
    drop((expected, reference_credit));
    assert_eq!(reference_pool.reserved(), 0);
}

fn pending_upsert(side: JoinSide, row: &StoredRow) -> PendingOp {
    PendingOp::Upsert {
        side,
        row_id: row.row_id,
        event_time: row.event_time,
        encoded_key: Arc::clone(&row.encoded_key),
        record: row.record.clone(),
        charge: row.charge,
    }
}

#[test]
fn test_delta_exact_capacity_preserves_mixed_wire_and_segment_owners() {
    let (rows, record) = fixture();
    let left_removal = PendingOp::Tombstone {
        side: JoinSide::Left,
        row_id: 8,
        event_time: EventTime::from_micros(-50),
        encoded_key: Arc::clone(&rows[1].encoded_key),
    };
    let right_removal = PendingOp::Tombstone {
        side: JoinSide::Right,
        row_id: 40,
        event_time: EventTime::from_micros(31),
        encoded_key: Arc::clone(&rows[2].encoded_key),
    };
    let pending = [
        right_removal.clone(),
        pending_upsert(JoinSide::Left, &rows[1]),
        pending_upsert(JoinSide::Right, &rows[2]),
        left_removal.clone(),
        pending_upsert(JoinSide::Left, &rows[0]),
    ];
    let (pool, workspace) = reservation(16 << 20);
    let (reference_pool, reference_credit) = reservation(4 << 20);
    let epoch = Epoch::new(7).unwrap();
    let output = pending_owned(
        &pending,
        &[record.schema(), record.schema()],
        epoch,
        &workspace,
        &|| Ok(()),
    )
    .unwrap();
    let selections = [
        (
            JoinSide::Left,
            vec![&rows[0], &rows[1]],
            vec![&left_removal],
        ),
        (JoinSide::Right, vec![&rows[2]], vec![&right_removal]),
    ];
    for (side, upserts, tombstones) in selections {
        let entry = output
            .payloads
            .iter()
            .find(|entry| entry.side == side.as_str())
            .unwrap();
        let digest = hex::decode(&entry.sha256).unwrap().try_into().unwrap();
        let expected = geometric_delta(
            &upserts,
            &tombstones,
            side,
            epoch,
            &digest,
            &reference_credit,
        );
        let segment = &output.segments[&format!("{}-delta-7", side.as_str())];
        assert_eq!(segment.bytes(), expected);
        assert_eq!(segment.bytes_arc().capacity(), expected.len());
    }
    let segments = output.segments.values().cloned().collect::<Vec<_>>();
    let owner = Arc::downgrade(&output.funding);
    drop((output, workspace));
    assert!(owner.upgrade().is_some() && pool.reserved() > 0);
    assert!(segments.iter().all(StateSegment::has_owner));
    drop(segments);
    assert!(owner.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
    drop(reference_credit);
    assert_eq!(reference_pool.reserved(), 0);
}

#[test]
fn test_wire_sizing_cancellation_precedes_admission() {
    let (rows, _) = fixture();
    let references = rows.iter().collect::<Vec<_>>();
    for (selected, stop) in [
        (references.as_slice(), 1),
        (references.as_slice(), 3),
        (&[][..], 1),
    ] {
        let (pool, credit) = reservation(0);
        let checks = std::cell::Cell::new(0);
        let result = encode_index(selected, JoinSide::Left, &[0; 32], &credit, 0, &|| {
            checks.set(checks.get() + 1);
            if checks.get() == stop {
                Err(CalcFlowError::Cancelled {
                    run_id: "wire-sizing".into(),
                })
            } else {
                Ok(())
            }
        });
        assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
        assert_eq!(checks.get(), stop);
        assert_eq!(pool.reserved(), 0);
    }
    let (pool, workspace) = reservation(64 << 10);
    let mut encoder = Encoder::new(&workspace).unwrap();
    let pressure = workspace.new_empty();
    pressure.try_grow((64 << 10) - pool.reserved()).unwrap();
    let result = encoder.frame_payload(&references, JoinSide::Left, Vec::new(), &|| {
        Err(CalcFlowError::Cancelled {
            run_id: "wire-frame".into(),
        })
    });
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert!(encoder.output.segments.is_empty());
    drop((encoder, pressure, workspace));
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_payload_frame_size_rejects_checked_overflow() {
    assert_eq!(payload_frame_bytes(3, 257).unwrap(), 313);
    assert!(payload_frame_bytes(usize::MAX / 8 + 1, 0).is_err());
    assert!(payload_frame_bytes(0, usize::MAX).is_err());
}

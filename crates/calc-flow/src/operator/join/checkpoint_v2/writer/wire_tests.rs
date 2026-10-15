use super::*;
use sha2::{Digest, Sha256};

#[path = "digest_tests.rs"]
pub(super) mod digest;
use crate::operator::join::{
    PendingOp, columnar::RowPayload, encode_join_key_v1, state_row_charge,
};
use crate::{CalcFlowError, Epoch, EventTime};
use datafusion::arrow::{
    array::{Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    ipc::writer::StreamWriter,
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

fn flat_ipc_fixture(length: usize) -> (Vec<StoredRow>, RecordBatch) {
    use datafusion::arrow::array::{Float64Array, UInt64Array};

    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
    ]));
    let symbols = (0..length)
        .map(|row| format!("S{:03}", row % 64))
        .collect::<Vec<_>>();
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![100; length]).with_timezone("UTC")),
            Arc::new(UInt64Array::from_iter_values(
                0..u64::try_from(length).unwrap(),
            )),
            Arc::new(StringArray::from(symbols)),
            Arc::new(Float64Array::from(vec![1.25; length])),
        ],
    )
    .unwrap();
    let parent = Arc::new(record.clone());
    let rows = (0..length)
        .map(|index| StoredRow {
            record: RowPayload::Rowed {
                parent: Arc::clone(&parent),
                row: index,
            },
            event_time: EventTime::from_micros(100),
            row_id: u64::try_from(index).unwrap(),
            charge: state_row_charge(&record, index, &[2], "ipc-owner").unwrap(),
            encoded_key: Arc::new(encode_join_key_v1(&record, index, &[2]).unwrap().into()),
        })
        .collect();
    (rows, record)
}

fn stream_writer_bytes(record: &RecordBatch) -> Vec<u8> {
    let options = IpcWriteOptions::try_new(8, false, MetadataVersion::V5).unwrap();
    let mut bytes = Vec::new();
    let mut writer =
        StreamWriter::try_new_with_options(&mut bytes, record.schema_ref(), options).unwrap();
    writer.write(record).unwrap();
    writer.finish().unwrap();
    drop(writer);
    bytes
}

fn rich_ipc_fixture() -> (Vec<StoredRow>, RecordBatch) {
    use datafusion::arrow::{
        array::{DictionaryArray, Int32Array, ListArray},
        datatypes::Int32Type,
    };

    let (_, original) = fixture();
    let dictionary = Arc::new(
        DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(vec![Some(1), None, Some(0)]),
            Arc::new(StringArray::from(vec!["blue", "猫"])),
        )
        .unwrap(),
    ) as ArrayRef;
    let nested = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
        Some(vec![Some(3), None]),
        None,
        Some(vec![]),
    ])) as ArrayRef;
    let mut fields = original
        .schema_ref()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields.extend([
        Field::new("dictionary", dictionary.data_type().clone(), true),
        Field::new("nested", nested.data_type().clone(), true),
    ]);
    let mut columns = original.columns().to_vec();
    columns.extend([dictionary, nested]);
    let record = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
    let parent = Arc::new(record.clone());
    let rows = (0..record.num_rows())
        .map(|index| StoredRow {
            record: RowPayload::Rowed {
                parent: Arc::clone(&parent),
                row: index,
            },
            event_time: EventTime::from_micros(100),
            row_id: u64::try_from(index).unwrap(),
            charge: state_row_charge(&record, index, &[0, 2], "ipc-owner").unwrap(),
            encoded_key: Arc::new(encode_join_key_v1(&record, index, &[0, 2]).unwrap().into()),
        })
        .collect();
    (rows, record)
}

#[derive(Debug)]
struct IpcPool {
    inner: GreedyMemoryPool,
    peak: std::sync::atomic::AtomicUsize,
    retiring: std::sync::Mutex<Option<std::sync::Weak<Vec<u8>>>>,
    retirements: std::sync::atomic::AtomicUsize,
}

impl IpcPool {
    fn new(limit: usize) -> Self {
        Self {
            inner: GreedyMemoryPool::new(limit),
            peak: std::sync::atomic::AtomicUsize::new(0),
            retiring: std::sync::Mutex::new(None),
            retirements: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn note_peak(&self) {
        self.peak
            .fetch_max(self.inner.reserved(), std::sync::atomic::Ordering::Relaxed);
    }

    fn peak(&self) -> usize {
        self.peak.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl std::fmt::Display for IpcPool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, formatter)
    }
}

impl MemoryPool for IpcPool {
    fn name(&self) -> &'static str {
        "ipc-ownership-test"
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.inner.grow(reservation, additional);
        self.note_peak();
    }

    fn shrink(&self, reservation: &MemoryReservation, bytes: usize) {
        if let Some(retiring) = self.retiring.lock().unwrap().as_ref() {
            assert_eq!(retiring.strong_count(), 0, "bytes must drop before credit");
            self.retirements
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.inner.shrink(reservation, bytes);
    }

    fn try_grow(
        &self,
        reservation: &MemoryReservation,
        additional: usize,
    ) -> datafusion::common::Result<()> {
        self.inner.try_grow(reservation, additional)?;
        self.note_peak();
        Ok(())
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }

    fn memory_limit(&self) -> datafusion::execution::memory_pool::MemoryLimit {
        self.inner.memory_limit()
    }
}

#[test]
fn test_owned_ipc_messages_drop_compacted_owners_and_preserve_stream_bytes() {
    for (rows, record) in [flat_ipc_fixture(65), rich_ipc_fixture()] {
        let references = rows.iter().collect::<Vec<_>>();
        let (reference_pool, reference_credit) = reservation(16 << 20);
        let compacted =
            compact::compact(&references, record.schema(), &reference_credit, &|| Ok(())).unwrap();
        let expected = stream_writer_bytes(&compacted.record);
        drop((compacted, reference_credit));
        assert_eq!(reference_pool.reserved(), 0);

        let tracking = Arc::new(IpcPool::new(16 << 20));
        let pool: Arc<dyn MemoryPool> = tracking.clone();
        let workspace = MemoryConsumer::new("ipc-owners").register(&pool);
        let credit = workspace.new_empty();
        let mut output = None;
        let allocations = allocation_counter::measure(|| {
            let compacted =
                compact::compact(&references, record.schema(), &workspace, &|| Ok(())).unwrap();
            let owner = compacted.funding_owner();
            let column = Arc::downgrade(compacted.record.columns().last().unwrap());
            let scratch = super::super::budget::stream_scratch(&compacted.record).unwrap();
            ensure(&credit, scratch).unwrap();
            let encoded = encode_stream(compacted, &|| Ok(())).unwrap();
            assert!(
                owner.upgrade().is_none(),
                "compaction credit must retire before output"
            );
            assert!(
                column.upgrade().is_none(),
                "compacted arrays must drop before their credit"
            );
            let encoded_bytes = std::iter::once(&encoded.schema)
                .chain(encoded.dictionaries.iter())
                .chain(std::iter::once(&encoded.batch))
                .map(|message| message.ipc_message.capacity() + message.arrow_data.capacity())
                .sum::<usize>()
                + encoded.dictionaries.capacity() * size_of::<EncodedData>();
            assert!(encoded_bytes > 0 && encoded_bytes <= credit.size());
            let mut bytes = PaidBuffer::new(&credit, scratch);
            write_stream(&mut bytes, encoded, &|| Ok(())).unwrap();
            output = Some(bytes.into_bytes().unwrap());
        });
        let output = output.unwrap();
        assert_eq!(output, expected);
        assert_eq!(
            &output[output.len() - 8..],
            &[255, 255, 255, 255, 0, 0, 0, 0]
        );
        assert!(allocations.bytes_max <= u64::try_from(tracking.peak()).unwrap());
        drop(workspace);
        assert!(usize::try_from(allocations.bytes_current).unwrap() <= pool.reserved());
        assert!(output.capacity() <= credit.size());
        drop((output, credit));
        assert_eq!(pool.reserved(), 0);
    }
}

#[test]
fn test_owned_ipc_cancellation_releases_every_phase() {
    let (rows, record) = rich_ipc_fixture();
    let references = rows.iter().collect::<Vec<_>>();
    assert_cancelled_ipc_generation(&references, &record);
    assert_cancelled_ipc_output(&references, &record);
}

fn assert_cancelled_ipc_generation(references: &[&StoredRow], record: &RecordBatch) {
    for cancel_after_generation in [false, true] {
        let (pool, workspace) = reservation(16 << 20);
        let compacted =
            compact::compact(references, record.schema(), &workspace, &|| Ok(())).unwrap();
        let owner = compacted.funding_owner();
        let scratch = super::super::budget::stream_scratch(&compacted.record).unwrap();
        let credit = workspace.new_empty();
        ensure(&credit, scratch).unwrap();
        let cancelled = std::cell::Cell::new(false);
        let result = encode_stream(compacted, &|| {
            if !cancel_after_generation || owner.upgrade().is_none() {
                cancelled.set(true);
                Err(CalcFlowError::Cancelled {
                    run_id: "ipc-generation".into(),
                })
            } else {
                Ok(())
            }
        });
        assert!(cancelled.get());
        assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
        assert!(owner.upgrade().is_none());
        drop((result, credit, workspace));
        assert_eq!(pool.reserved(), 0);
    }
}

fn assert_cancelled_ipc_output(references: &[&StoredRow], record: &RecordBatch) {
    for boundary in [0, 1, 2] {
        let (pool, workspace) = reservation(16 << 20);
        let compacted =
            compact::compact(references, record.schema(), &workspace, &|| Ok(())).unwrap();
        let expected = stream_writer_bytes(&compacted.record);
        let owner = compacted.funding_owner();
        let scratch = super::super::budget::stream_scratch(&compacted.record).unwrap();
        let credit = workspace.new_empty();
        ensure(&credit, scratch).unwrap();
        let encoded = encode_stream(compacted, &|| Ok(())).unwrap();
        assert!(owner.upgrade().is_none());
        let stop = match boundary {
            0 => 1,
            1 => 2,
            _ => encoded.dictionaries.len() + 3,
        };
        let checks = std::cell::Cell::new(0);
        let mut output = PaidBuffer::new(&credit, scratch);
        let result = write_stream(&mut output, encoded, &|| {
            checks.set(checks.get() + 1);
            if checks.get() == stop {
                Err(CalcFlowError::Cancelled {
                    run_id: "ipc-output".into(),
                })
            } else {
                Ok(())
            }
        });
        assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
        assert_eq!(checks.get(), stop);
        let output = output.into_bytes().unwrap();
        assert_partial_ipc_output(&expected, &output, boundary);
        assert!(output.capacity() <= credit.size());
        drop((output, credit, workspace));
        assert_eq!(pool.reserved(), 0);
    }
}

fn assert_partial_ipc_output(expected: &[u8], output: &[u8], boundary: usize) {
    assert!(expected.starts_with(output));
    if boundary == 0 {
        assert!(output.is_empty());
    } else if boundary == 2 {
        assert_eq!(output, &expected[..expected.len() - 8]);
    }
}

#[test]
fn test_payload_retires_compacted_batch_before_ipc_output_growth() {
    let (rows, record) = flat_ipc_fixture(8_192);
    let references = rows.iter().collect::<Vec<_>>();
    let ipc = stream_writer_bytes(&record);
    let mut expected = Vec::new();
    expected.extend_from_slice(&header(*PAYLOAD_MAGIC, JoinSide::Left));
    expected.extend_from_slice(&integer(rows.len()).unwrap());
    expected.extend_from_slice(&integer(ipc.len()).unwrap());
    for row in &rows {
        expected.extend_from_slice(&row.row_id.to_le_bytes());
    }
    expected.extend_from_slice(&ipc);

    let (pool, workspace) = reservation(2_400_000);
    let mut encoder = Encoder::new(&workspace).unwrap();
    encoder.install("held".into(), vec![0x59; 4_096]).unwrap();
    let held = encoder.output.segments["held"].clone();
    let digest = encoder
        .payload(
            &references,
            record.schema(),
            JoinSide::Left,
            &workspace,
            &|| Ok(()),
        )
        .expect("owned encoded messages must release compacted Arrow credit before IPC growth");
    let name = format!("left-payload-{}", hex::encode(digest));
    let segment = encoder.output.segments[&name].clone();
    assert_eq!(segment.bytes(), expected);
    assert_eq!(held.bytes(), &[0x59; 4_096]);
    let owner = Arc::downgrade(&encoder.funding);
    drop((encoder, workspace));
    assert!(owner.upgrade().is_none());
    let capacity = segment.bytes_arc().capacity();
    let both = pool.reserved();
    assert!(both >= capacity + held.bytes_arc().capacity());
    assert_eq!(segment.bytes(), expected);
    drop(segment);
    assert!(both - pool.reserved() >= capacity);
    assert!(pool.reserved() >= held.bytes_arc().capacity());
    assert_eq!(held.bytes(), &[0x59; 4_096]);
    drop(held);
    assert!(owner.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
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
    let empty = encode_index(
        &[] as &[StoredRow],
        JoinSide::Left,
        &[0; 32],
        &empty_credit,
        0,
        &|| Ok(()),
    )
    .unwrap();
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
    let ipc = stream_writer_bytes(&record);
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
    let tracking = Arc::new(IpcPool::new(4 << 20));
    let pool: Arc<dyn MemoryPool> = tracking.clone();
    let workspace = MemoryConsumer::new("exact-frame-test").register(&pool);
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
        tracking.peak() >= live + length,
        "live IPC and exact output must be prepaid together"
    );
    assert!(allocations.bytes_max <= u64::try_from(tracking.peak()).unwrap());
    assert!(usize::try_from(allocations.bytes_current).unwrap() <= pool.reserved());
    let owner = Arc::downgrade(&encoder.funding);
    drop((encoder, workspace));
    assert!(owner.upgrade().is_none());
    assert!(pool.reserved() >= segment.bytes_arc().capacity());
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
    assert!(owner.upgrade().is_none());
    let backing = segments
        .iter()
        .map(|segment| segment.bytes_arc().capacity())
        .sum::<usize>();
    assert!(pool.reserved() >= backing);
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

#[path = "funding_tests.rs"]
mod funding_tests;

#[path = "pending_tests.rs"]
mod pending_tests;

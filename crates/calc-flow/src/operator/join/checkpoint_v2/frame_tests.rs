use super::{
    super::JoinSide,
    frame::{Index, InvalidFrame, Payload, Record},
};
use datafusion::arrow::{
    array::{Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    ipc::{
        MetadataVersion,
        writer::{IpcWriteOptions, StreamWriter},
    },
    record_batch::RecordBatch,
};
use std::sync::Arc;

const KEY: [u8; 17] = [5, 0, 0, 0, 0, 8, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0];
const DIGEST: [u8; 32] = [0x42; 32];

fn header(magic: [u8; 8]) -> Vec<u8> {
    let mut bytes = magic.to_vec();
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&[0, 0, 0, 0]);
    bytes
}

fn ipc() -> Vec<u8> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![7, 7, 7])),
            Arc::new(TimestampMicrosecondArray::from(vec![95, 96, 97]).with_timezone("UTC")),
        ],
    )
    .unwrap();
    let mut bytes = Vec::new();
    let options = IpcWriteOptions::try_new(8, false, MetadataVersion::V5).unwrap();
    let mut writer = StreamWriter::try_new_with_options(&mut bytes, &schema, options).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();
    drop(writer);
    bytes
}

fn upsert(id: u64, time: i64, payload_row: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&id.to_le_bytes());
    bytes.extend_from_slice(&time.to_le_bytes());
    bytes.extend_from_slice(&115_u64.to_le_bytes());
    bytes.extend_from_slice(&DIGEST);
    bytes.extend_from_slice(&payload_row.to_le_bytes());
    bytes.extend_from_slice(&17_u64.to_le_bytes());
    bytes.extend_from_slice(&KEY);
    bytes
}

fn tombstone(id: u64, time: i64) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&id.to_le_bytes());
    bytes.extend_from_slice(&time.to_le_bytes());
    bytes.extend_from_slice(&17_u64.to_le_bytes());
    bytes.extend_from_slice(&KEY);
    bytes
}

fn with_u64(bytes: &[u8], offset: usize, value: u64) -> Vec<u8> {
    let mut changed = bytes.to_vec();
    changed[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    changed
}

fn consume(index: &Index<'_>) -> Result<(), InvalidFrame> {
    let mut records = index.records();
    while records.next()?.is_some() {}
    Ok(())
}

fn assert_valid_payload(payload: &[u8], ipc: &[u8]) {
    let decoded = Payload::decode(payload, JoinSide::Left).unwrap();
    assert_eq!(decoded.rows, 3);
    assert_eq!(decoded.row_id(0).unwrap(), 2);
    assert_eq!(decoded.row_id(1).unwrap(), 7);
    assert_eq!(decoded.row_id(2).unwrap(), 9);
    assert!(decoded.row_id(3).is_err());
    assert!(decoded.row_id(usize::MAX).is_err());
    assert_eq!(decoded.ipc, ipc);
    assert_eq!(decoded.ipc.as_ptr(), payload[56..].as_ptr());
}

fn assert_valid_base(base: &[u8]) {
    assert_eq!(base.len(), 210);
    let index = Index::base(base, JoinSide::Left).unwrap();
    assert_eq!((index.epoch, index.upserts, index.tombstones), (None, 2, 0));
    let mut records = index.records();
    for (id, time, payload_row) in [(2, 95, 0), (7, 96, 1)] {
        let Some(Record::Upsert(row)) = records.next().unwrap() else {
            panic!("expected base upsert");
        };
        assert_eq!((row.row_id, row.time, row.charge), (id, time, 115));
        assert_eq!((row.digest, row.payload_row), (DIGEST, payload_row));
        assert_eq!(row.key, KEY);
    }
    assert!(records.next().unwrap().is_none());
}

fn assert_valid_delta(delta: &[u8]) {
    assert_eq!(delta.len(), 211);
    let index = Index::delta(delta, JoinSide::Left, 2).unwrap();
    assert_eq!(
        (index.epoch, index.upserts, index.tombstones),
        (Some(2), 1, 2)
    );
    let mut records = index.records();
    let Some(Record::Upsert(row)) = records.next().unwrap() else {
        panic!("expected delta upsert");
    };
    assert_eq!((row.row_id, row.time, row.charge), (9, 97, 115));
    assert_eq!((row.digest, row.payload_row), (DIGEST, 2));
    assert_eq!(row.key, KEY);
    for (id, time) in [(2, 95), (7, 96)] {
        let Some(Record::Tombstone(row)) = records.next().unwrap() else {
            panic!("expected delta tombstone");
        };
        assert_eq!((row.row_id, row.time), (id, time));
        assert_eq!(row.key, KEY);
    }
    assert!(records.next().unwrap().is_none());
}

fn assert_invalid_payload(payload: &[u8]) {
    for (offset, value) in [(0, b'!'), (8, 1), (12, 1), (13, 1)] {
        let mut changed = payload.to_vec();
        changed[offset] = value;
        assert!(Payload::decode(&changed, JoinSide::Left).is_err());
    }
    assert!(Payload::decode(payload, JoinSide::Right).is_err());
    assert!(Payload::decode(&payload[..31], JoinSide::Left).is_err());
    assert!(Payload::decode(&payload[..payload.len() - 1], JoinSide::Left).is_err());
    let mut trailing_payload = payload.to_vec();
    trailing_payload.push(0);
    assert!(Payload::decode(&trailing_payload, JoinSide::Left).is_err());
    for (offset, value) in [(16, 0), (16, u64::MAX), (24, u64::MAX)] {
        assert!(Payload::decode(&with_u64(payload, offset, value), JoinSide::Left).is_err());
    }
}

fn assert_invalid_base(base: &[u8]) {
    assert!(Index::base(&base[..31], JoinSide::Left).is_err());
    assert!(
        Index::base(&base[..base.len() - 1], JoinSide::Left)
            .and_then(|index| consume(&index))
            .is_err()
    );
    let mut trailing_base = base.to_vec();
    trailing_base.push(0);
    assert!(
        Index::base(&trailing_base, JoinSide::Left)
            .and_then(|index| consume(&index))
            .is_err()
    );
    for (offset, value) in [(16, u64::MAX), (16, 3), (16, 1), (24, 1), (96, u64::MAX)] {
        assert!(
            Index::base(&with_u64(base, offset, value), JoinSide::Left)
                .and_then(|index| consume(&index))
                .is_err()
        );
    }
    for id in [2, 1] {
        assert!(
            Index::base(&with_u64(base, 121, id), JoinSide::Left)
                .and_then(|index| consume(&index))
                .is_err()
        );
    }
}

fn assert_invalid_delta(delta: &[u8]) {
    for id in [2, 1] {
        assert!(
            Index::delta(&with_u64(delta, 170, id), JoinSide::Left, 2)
                .and_then(|index| consume(&index))
                .is_err()
        );
    }
    assert!(Index::delta(delta, JoinSide::Left, 0).is_err());
    assert!(Index::delta(delta, JoinSide::Left, 3).is_err());
    let mut empty_delta = delta[..40].to_vec();
    empty_delta[24..40].fill(0);
    assert!(Index::delta(&empty_delta, JoinSide::Left, 2).is_err());
    for count in [u64::MAX, 3] {
        assert!(
            Index::delta(&with_u64(delta, 32, count), JoinSide::Left, 2)
                .and_then(|index| consume(&index))
                .is_err()
        );
    }
    assert!(
        Index::delta(&delta[..delta.len() - 1], JoinSide::Left, 2)
            .and_then(|index| consume(&index))
            .is_err()
    );
    let mut trailing_delta = delta.to_vec();
    trailing_delta.push(0);
    assert!(
        Index::delta(&trailing_delta, JoinSide::Left, 2)
            .and_then(|index| consume(&index))
            .is_err()
    );
}

#[test]
fn test_v2_frames_borrow_records_and_reject_noncanonical_wire() {
    let ipc = ipc();
    let mut payload = header(*b"CFJPAY2\0");
    payload.extend_from_slice(&3_u64.to_le_bytes());
    payload.extend_from_slice(&u64::try_from(ipc.len()).unwrap().to_le_bytes());
    for id in [2_u64, 7, 9] {
        payload.extend_from_slice(&id.to_le_bytes());
    }
    payload.extend_from_slice(&ipc);
    assert_valid_payload(&payload, &ipc);

    let mut base = header(*b"CFJIDX2\0");
    base.extend_from_slice(&2_u64.to_le_bytes());
    base.extend_from_slice(&0_u64.to_le_bytes());
    base.extend_from_slice(&upsert(2, 95, 0));
    base.extend_from_slice(&upsert(7, 96, 1));
    assert_valid_base(&base);

    let mut delta = header(*b"CFJDIX2\0");
    delta.extend_from_slice(&2_u64.to_le_bytes());
    delta.extend_from_slice(&1_u64.to_le_bytes());
    delta.extend_from_slice(&2_u64.to_le_bytes());
    delta.extend_from_slice(&upsert(9, 97, 2));
    delta.extend_from_slice(&tombstone(2, 95));
    delta.extend_from_slice(&tombstone(7, 96));
    assert_valid_delta(&delta);

    assert_invalid_payload(&payload);
    assert_invalid_base(&base);
    assert_invalid_delta(&delta);
}

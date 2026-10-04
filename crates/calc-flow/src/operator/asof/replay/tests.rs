use super::{Callback, Record, codec};
use crate::{Cursor, EventTime, IngressProgress, IngressState, JsonMap};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
use std::sync::Arc;

fn decode(bytes: &[u8], count: u64, records: &mut Vec<Record>) -> crate::Result<()> {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4 * 1024 * 1024));
    let mut credit = MemoryConsumer::new("replay-test").register(&pool);
    codec::decode_into(bytes, count, records, &mut credit, &|bytes| {
        let workspace = MemoryConsumer::new("replay-workspace-test").register(&pool);
        workspace
            .try_grow(
                usize::try_from(bytes).map_err(|_| super::mismatch("test workspace overflowed"))?,
            )
            .map_err(|_| super::mismatch("test workspace refused"))?;
        Ok(workspace)
    })
}

fn record() -> Record {
    Record {
        callback: Callback::Data {
            side: 1,
            sequence: u64::MAX - 1,
        },
        input_watermark: Some(EventTime::from_micros(i64::MIN)),
        progress: [
            IngressProgress::new(IngressState::Idle, Some(EventTime::from_micros(-1))),
            IngressProgress::new(IngressState::Ended, None),
        ],
        max_rows: 1000,
        max_bytes: 1024 * 1024,
        cursor: Some(Arc::new(
            Cursor::new("right-source", vec![0, 1, 255], JsonMap::new()).unwrap(),
        )),
        cursor_bytes: 4096,
    }
}

#[test]
fn callback_frame_preserves_full_coordinates_and_progress() {
    let records = [
        record(),
        Record {
            callback: Callback::Progress,
            cursor: None,
            cursor_bytes: 0,
            ..record()
        },
        Record {
            callback: Callback::End,
            input_watermark: None,
            cursor: None,
            cursor_bytes: 0,
            ..record()
        },
    ];
    let bytes = codec::encode(&records).unwrap();
    assert_eq!(bytes.len(), codec::encoded_len(&records).unwrap());
    let mut decoded = Vec::with_capacity(3);
    decode(&bytes, 3, &mut decoded).unwrap();
    assert_eq!(decoded, records);
}

#[test]
fn callback_frame_rejects_corrupt_tags_lengths_and_unfunded_records() {
    let original = codec::encode(&[record()]).unwrap();
    for (offset, value) in [
        (0, 0),
        (7, b'1'),
        (8, 2),
        (16, 3),
        (17, 2),
        (26, 2),
        (35, 3),
        (71, 0),
        (71, 2),
    ] {
        let mut bytes = original.clone();
        bytes[offset] = value;
        assert!(decode(&bytes, 1, &mut Vec::with_capacity(1)).is_err());
    }
    assert!(
        decode(
            &original[..original.len() - 1],
            1,
            &mut Vec::with_capacity(1)
        )
        .is_err()
    );
    assert!(decode(&original, 1, &mut Vec::new()).is_err());
    let mut extra = original;
    extra.push(0);
    assert!(decode(&extra, 1, &mut Vec::with_capacity(1)).is_err());
}

#[test]
fn callback_cursor_matches_complete_reader_position() {
    let saved = Cursor::new(
        "right-source",
        vec![1],
        JsonMap::from([("row".into(), serde_json::json!(7))]),
    )
    .unwrap();
    let same = Cursor::unbound(vec![1], saved.payload().clone()).unwrap();
    super::restore::checked_cursor(&saved, same, "right-source").unwrap();
    for cursor in [
        Cursor::new("foreign", vec![1], saved.payload().clone()).unwrap(),
        Cursor::unbound(vec![2], saved.payload().clone()).unwrap(),
        Cursor::unbound(
            vec![1],
            JsonMap::from([("row".into(), serde_json::json!(8))]),
        )
        .unwrap(),
    ] {
        assert!(super::restore::checked_cursor(&saved, cursor, "right-source").is_err());
    }
}

#[test]
fn callback_cursor_decode_refunds_understated_credit_and_refuses_workspace() {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024 * 1024));
    let mut credit = MemoryConsumer::new("replay-cursor-refund").register(&pool);
    let mut bytes = codec::encode(&[record()]).unwrap();
    bytes[72..80].copy_from_slice(&1_u64.to_le_bytes());
    let mut records = Vec::with_capacity(1);
    let reserve = |bytes| {
        let workspace = MemoryConsumer::new("replay-cursor-workspace").register(&pool);
        workspace
            .try_grow(
                usize::try_from(bytes).map_err(|_| super::mismatch("test workspace overflowed"))?,
            )
            .map_err(|_| super::mismatch("test workspace refused"))?;
        Ok(workspace)
    };
    assert!(codec::decode_into(&bytes, 1, &mut records, &mut credit, &reserve).is_err());
    assert!(records.is_empty());
    assert_eq!(credit.size(), 0);
    assert_eq!(pool.reserved(), 0);
    let bytes = codec::encode(&[record()]).unwrap();
    let refused = |_: u64| Err(super::mismatch("workspace refused"));
    assert!(codec::decode_into(&bytes, 1, &mut records, &mut credit, &refused).is_err());
    assert!(records.is_empty());
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn callback_cursor_rejects_duplicate_json_keys_and_invalid_payloads() {
    let original = codec::encode(&[record()]).unwrap();
    for payload in [b"[]".as_slice(), br#"{"row":1,"row":2}"#.as_slice()] {
        let mut bytes = original[..original.len() - 2].to_vec();
        bytes[96..104].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(payload);
        assert!(decode(&bytes, 1, &mut Vec::with_capacity(1)).is_err());
    }
}

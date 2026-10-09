use super::*;
use datafusion::arrow::{
    array::{Array, ArrayRef, ListArray},
    ipc::{
        MetadataVersion,
        reader::StreamReader,
        writer::{IpcWriteOptions, StreamWriter},
    },
};
use sha2::{Digest, Sha256};
use std::io::Cursor;

#[path = "checkpoint_v2_history_tests.rs"]
mod history_tests;

#[path = "checkpoint_v2_capture_tests.rs"]
mod capture_tests;

#[path = "checkpoint_v2_overlap_tests.rs"]
mod overlap_tests;

#[path = "checkpoint_v2_v1_reuse_tests.rs"]
mod v1_reuse_tests;

#[path = "checkpoint_v2_behavior_tests.rs"]
mod behavior_tests;

#[path = "checkpoint_v2_loan_tests.rs"]
mod loan_tests;

#[path = "checkpoint_v2_writer_tests.rs"]
mod writer_tests;

#[path = "checkpoint_v2_writer_cut_tests.rs"]
mod writer_cut_tests;

#[path = "checkpoint_v2_writer_restore_tests.rs"]
mod writer_restore_tests;

#[path = "checkpoint_v2_writer_lifecycle_tests.rs"]
mod writer_lifecycle_tests;

const KEY: [u8; 17] = [5, 0, 0, 0, 0, 8, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0];

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("text", DataType::Utf8, true),
        Field::new(
            "kind",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "nested",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        ),
    ]))
}

fn operator() -> StreamJoinOperator {
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "at",
        "at",
        JoinTimeBounds::new(Duration::ZERO, Duration::from_micros(10)).unwrap(),
        JoinStateLimits::new(100, 1_000_000, 100).unwrap(),
    )
    .unwrap();
    StreamJoinOperator::new("v2-match", schema(), schema(), spec).unwrap()
}

fn record(
    times: &[i64],
    text: &[Option<&str>],
    kinds: &[&str],
    nested: Vec<Option<Vec<Option<i32>>>>,
) -> RecordBatch {
    let dictionary: DictionaryArray<Int32Type> = kinds.iter().copied().collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(vec![7; times.len()])),
            Arc::new(TimestampMicrosecondArray::from(times.to_vec()).with_timezone("UTC")),
            Arc::new(StringArray::from(text.to_vec())),
            Arc::new(dictionary),
            Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(nested)),
        ],
    )
    .unwrap()
}

struct Payload {
    side: u8,
    ids: Vec<u64>,
    segment: StateSegment,
    digest: [u8; 32],
}

fn payload(side: u8, ids: &[u64], record: &RecordBatch, charges: &[u64]) -> Payload {
    assert_eq!(record.num_rows(), ids.len());
    for (row, expected) in charges.iter().enumerate() {
        assert_eq!(encode_join_key_v1(record, row, &[0]).unwrap(), KEY);
        assert_eq!(
            state_row_charge(record, row, &[0], "v2-match").unwrap(),
            *expected
        );
    }
    let options = IpcWriteOptions::try_new(8, false, MetadataVersion::V5).unwrap();
    let mut ipc = Vec::new();
    let mut writer = StreamWriter::try_new_with_options(&mut ipc, &schema(), options).unwrap();
    writer.write(record).unwrap();
    writer.finish().unwrap();
    drop(writer);
    let mut reader = StreamReader::try_new(Cursor::new(&ipc), None).unwrap();
    assert_eq!(reader.schema(), schema());
    assert_eq!(reader.next().unwrap().unwrap().num_rows(), ids.len());
    assert!(reader.next().is_none());
    let mut bytes = header(*b"CFJPAY2\0", side);
    bytes.extend_from_slice(&u64::try_from(ids.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&u64::try_from(ipc.len()).unwrap().to_le_bytes());
    for id in ids {
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    bytes.extend_from_slice(&ipc);
    let digest = Sha256::digest(&bytes).into();
    Payload {
        side,
        ids: ids.to_vec(),
        segment: StateSegment::new(bytes),
        digest,
    }
}

fn header(magic: [u8; 8], side: u8) -> Vec<u8> {
    let mut bytes = magic.to_vec();
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&[side, 0, 0, 0]);
    bytes
}

fn append_upsert(bytes: &mut Vec<u8>, payload: &Payload, row: usize, time: i64, charge: u64) {
    bytes.extend_from_slice(&payload.ids[row].to_le_bytes());
    bytes.extend_from_slice(&time.to_le_bytes());
    bytes.extend_from_slice(&charge.to_le_bytes());
    bytes.extend_from_slice(&payload.digest);
    bytes.extend_from_slice(&u64::try_from(row).unwrap().to_le_bytes());
    bytes.extend_from_slice(&17_u64.to_le_bytes());
    bytes.extend_from_slice(&KEY);
}

fn base(payload: &Payload, times: &[i64], charges: &[u64]) -> StateSegment {
    let mut bytes = header(*b"CFJIDX2\0", payload.side);
    bytes.extend_from_slice(&u64::try_from(payload.ids.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    for (row, (&time, &charge)) in times.iter().zip(charges).enumerate() {
        append_upsert(&mut bytes, payload, row, time, charge);
    }
    StateSegment::new(bytes)
}

fn delta(payload: &Payload) -> StateSegment {
    let mut bytes = header(*b"CFJDIX2\0", 0);
    bytes.extend_from_slice(&2_u64.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    append_upsert(&mut bytes, payload, 0, 97, 135);
    bytes.extend_from_slice(&2_u64.to_le_bytes());
    bytes.extend_from_slice(&95_i64.to_le_bytes());
    bytes.extend_from_slice(&17_u64.to_le_bytes());
    bytes.extend_from_slice(&KEY);
    StateSegment::new(bytes)
}

fn snapshot(operator: &StreamJoinOperator) -> OperatorStateSnapshot {
    let left = payload(
        0,
        &[2, 7],
        &record(
            &[95, 96],
            &[None, Some("猫")],
            &["red", "blue"],
            vec![Some(vec![Some(1), None]), Some(vec![Some(2), Some(3)])],
        ),
        &[136, 148],
    );
    let right = payload(
        1,
        &[4],
        &record(&[100], &[Some("ok")], &["red"], vec![Some(vec![Some(9)])]),
        &[141],
    );
    let added = payload(
        0,
        &[9],
        &record(&[97], &[Some("新")], &["green"], vec![None]),
        &[135],
    );
    let mut segments = BTreeMap::from([
        ("left-base".into(), base(&left, &[95, 96], &[136, 148])),
        ("right-base".into(), base(&right, &[100], &[141])),
        ("left-delta-2".into(), delta(&added)),
    ]);
    let mut payloads = [&left, &added, &right];
    payloads.sort_by_key(|payload| (payload.side, payload.digest));
    let inventory = payloads
        .iter()
        .map(|payload| {
            let side = if payload.side == 0 { "left" } else { "right" };
            let sha256 = hex::encode(payload.digest);
            segments.insert(format!("{side}-payload-{sha256}"), payload.segment.clone());
            serde_json::json!({
                "side": side, "sha256": sha256, "rows": payload.ids.len(),
                "bytes": payload.segment.bytes().len(),
            })
        })
        .collect::<Vec<_>>();
    let metadata = JoinCheckpointMetadata {
        layout_version: 2,
        spec: operator.spec.clone(),
        next_left_row_id: 10,
        next_right_row_id: 5,
        next_output_sequence: 6,
        ended: false,
        epoch: 2,
        metrics: JoinMetrics {
            left: SideMetrics {
                retained_rows: 2,
                retained_bytes: 283,
                evicted_rows: 1,
                ..SideMetrics::default()
            },
            right: SideMetrics {
                retained_rows: 1,
                retained_bytes: 141,
                ..SideMetrics::default()
            },
            emitted_match_rows: 5,
            ..JoinMetrics::default()
        },
    };
    let Value::Object(mut metadata) = serde_json::to_value(metadata).unwrap() else {
        panic!("object metadata");
    };
    metadata.insert(
        "v2_inventory".into(),
        serde_json::json!({
            "codec_version": 2, "base_epoch": 1,
            "deltas": [{ "epoch": 2, "sides": ["left"] }], "payloads": inventory,
        }),
    );
    OperatorStateSnapshot {
        inline_metadata: metadata.into_iter().collect(),
        segments,
    }
}

fn assert_row(
    row: &StoredRow,
    id: u64,
    time: i64,
    charge: u64,
    text: &str,
    kind: &str,
    nested: Option<&[i32]>,
) {
    assert_eq!(
        (row.row_id, row.event_time.as_micros(), row.charge),
        (id, time, charge)
    );
    assert_eq!(row.encoded_key.as_slice(), KEY);
    let record = row.record.view();
    assert_eq!(record.schema(), schema());
    assert_eq!(record.num_rows(), 1);
    assert_eq!(
        record
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        text
    );
    let dictionary = record
        .column(3)
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()
        .unwrap();
    let values = dictionary
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(
        values.value(usize::try_from(dictionary.keys().value(0)).unwrap()),
        kind
    );
    assert_nested(record.column(4), nested);
}

fn assert_nested(column: &ArrayRef, expected: Option<&[i32]>) {
    let array = column.as_any().downcast_ref::<ListArray>().unwrap();
    if let Some(expected) = expected {
        assert!(!array.is_null(0));
        let values = array.value(0);
        let values = values.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(values.values().as_ref(), expected);
    } else {
        assert!(array.is_null(0));
    }
}

fn assert_restored(operator: &StreamJoinOperator) {
    assert_eq!(operator.state.left.len(), 2);
    assert_eq!(operator.state.right.len(), 1);
    assert_row(
        &operator.state.left[0],
        7,
        96,
        148,
        "猫",
        "blue",
        Some(&[2, 3]),
    );
    assert_row(&operator.state.left[1], 9, 97, 135, "新", "green", None);
    assert_row(
        &operator.state.right[0],
        4,
        100,
        141,
        "ok",
        "red",
        Some(&[9]),
    );
    assert_eq!(
        (
            operator.state.next_left_row_id,
            operator.state.next_right_row_id,
            operator.state.next_output_sequence
        ),
        (10, 5, 6)
    );
    assert_eq!(operator.state.last_checkpoint_epoch, Epoch::new(2));
    let status = operator.status();
    assert_eq!(
        (status.left.retained_rows, status.left.retained_bytes),
        (2, 283)
    );
    assert_eq!(
        (status.right.retained_rows, status.right.retained_bytes),
        (1, 141)
    );
    assert_eq!(status.emitted_match_rows, 5);
    assert!(!operator.state.ended);
}

#[test]
fn test_join_restore_accepts_v2_payload_locator_history_atomically() {
    let mut operator = operator();
    let snapshot = snapshot(&operator);
    assert_eq!(snapshot.segments.len(), 6);
    let result = operator.restore(&snapshot);
    assert!(
        result.is_ok(),
        "valid layout2 locator history must restore: {result:?}"
    );
    assert_restored(&operator);
    let mut invalid = snapshot.clone();
    invalid
        .inline_metadata
        .insert("next_left_row_id".into(), 9.into());
    assert!(operator.restore(&invalid).is_err());
    assert_restored(&operator);
}

#[path = "checkpoint_v2_writer_cancel_tests.rs"]
mod writer_cancel_tests;

mod asof_support;

use std::{collections::HashMap, ops::Range, sync::Arc};

use asof_support::{batch, operator, schema, spec};
use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, CancellationToken, EdgeCollector, Epoch, JsonMap,
    OperatorMetadata, OperatorStateSnapshot, StateSegment, StreamAsofJoinOperator,
    StreamJobContext, StreamOperator, StreamOperatorContext,
};
use datafusion::arrow::{array::Int64Array, datatypes::SchemaRef, record_batch::RecordBatch};
use serde_json::json;

async fn populated_snapshot() -> OperatorStateSnapshot {
    let mut op = operator(10);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data(
        "right",
        batch(&[("A", 100, 1, 11), ("A", 100, 2, 22), ("B", 200, 3, 33)]),
        &cx,
        &mut out,
    )
    .await
    .unwrap();
    op.process_data(
        "left",
        batch(&[("A", 105, 10, 44), ("A", 105, 11, 55)]),
        &cx,
        &mut out,
    )
    .await
    .unwrap();
    op.checkpoint(Epoch::INITIAL).unwrap()
}

async fn seed_live_state(op: &mut StreamAsofJoinOperator, input_schema: &SchemaRef) {
    let job = StreamJobContext::new(2, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    for (side, rows) in [
        ("right", [("Z", 500, 99, 777)]),
        ("left", [("Z", 505, 100, 888)]),
    ] {
        let input = batch(&rows);
        let records = input
            .table_payload()
            .unwrap()
            .batches()
            .iter()
            .map(|record| {
                RecordBatch::try_new(input_schema.clone(), record.columns().to_vec()).unwrap()
            })
            .collect();
        op.process_data(
            side,
            Batch::table(records, BatchMetadata::default()).unwrap(),
            &cx,
            &mut out,
        )
        .await
        .unwrap();
    }
}

fn reject_without_replacing_state(
    op: &mut StreamAsofJoinOperator,
    damaged: &OperatorStateSnapshot,
    expected_message: &str,
    case: &str,
) {
    let before = op.checkpoint(Epoch::INITIAL).unwrap();
    let before_status = op.status();
    let error = op.restore(damaged).expect_err(case);
    assert!(
        matches!(error, CalcFlowError::CheckpointMismatch { ref message }
        if message.contains(expected_message)),
        "{case}: {error:?}"
    );
    assert_eq!(op.status(), before_status, "{case}");
    let after = op.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(after.inline_metadata, before.inline_metadata, "{case}");
    assert_eq!(after.segments, before.segments, "{case}");
}

async fn assert_original_answer(op: &mut StreamAsofJoinOperator) {
    let job = StreamJobContext::new(3, "asof", JsonMap::new(), None, CancellationToken::new());
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.on_end(&cx, &mut out).await.unwrap();
    let batches = out.drain("output");
    let record = &batches[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    assert_eq!(record.num_rows(), 1);
    for (name, expected) in [("left__value", 888), ("right__value", 777)] {
        let values = record
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(values.value(0), expected);
    }
    assert_eq!(op.status().emitted_left_rows, 1);
}

#[tokio::test]
async fn test_asof_restore_rejects_state_layout_accounting_and_encoding_versions_atomically() {
    let original = populated_snapshot().await;
    let mut target = operator(10);
    seed_live_state(&mut target, &schema()).await;
    for (field, value) in [
        ("state_version", json!(4)),
        ("layout_version", json!(3)),
        ("layout_version", json!(4)),
        ("layout_version", json!(7)),
        ("accounting_version", json!(3)),
        ("accounting_version", json!(4)),
        ("accounting_version", json!(7)),
        ("row_encoding", json!("arrow-row-unknown")),
    ] {
        let mut damaged = original.clone();
        damaged.inline_metadata.insert(field.into(), value);
        reject_without_replacing_state(&mut target, &damaged, "state version differs", field);
    }
    assert_original_answer(&mut target).await;
}

#[tokio::test]
async fn test_asof_restore_rejects_historical_versions_without_replacing_state() {
    let empty = operator(10).checkpoint(Epoch::INITIAL).unwrap();
    let mut target = operator(10);
    seed_live_state(&mut target, &schema()).await;
    for version in [1, 2] {
        let mut historical = empty.clone();
        for field in ["state_version", "layout_version", "accounting_version"] {
            historical
                .inline_metadata
                .insert(field.into(), json!(version));
        }
        if version == 1 {
            historical
                .inline_metadata
                .insert("row_encoding".into(), json!("arrow-row-58.3.0"));
        }
        reject_without_replacing_state(
            &mut target,
            &historical,
            "state version differs",
            "historical version",
        );
    }
    assert_original_answer(&mut target).await;
}

#[tokio::test]
async fn test_asof_restore_rejects_configuration_and_schema_fingerprint_changes_atomically() {
    let original = populated_snapshot().await;
    let changed_schema = Arc::new(schema().as_ref().clone().with_metadata(HashMap::from([(
        "revision".into(),
        "different-exact-schema".into(),
    )])));
    let changed = StreamAsofJoinOperator::new(
        "asof",
        changed_schema.clone(),
        changed_schema.clone(),
        spec(10),
    )
    .unwrap();
    for (case, mut target, input_schema) in [
        ("tolerance", operator(11), schema()),
        ("schema metadata", changed, changed_schema),
    ] {
        seed_live_state(&mut target, &input_schema).await;
        let own = target.checkpoint(Epoch::INITIAL).unwrap();
        assert_ne!(
            own.inline_metadata["fingerprint"],
            original.inline_metadata["fingerprint"]
        );
        reject_without_replacing_state(
            &mut target,
            &original,
            "configuration or state version differs",
            case,
        );
        assert_original_answer(&mut target).await;
    }
}

#[tokio::test]
async fn test_asof_restore_rejects_forged_gauges_counters_and_output_sequence_atomically() {
    let original = populated_snapshot().await;
    let mut target = operator(10);
    seed_live_state(&mut target, &schema()).await;
    for (path, value, message) in [
        (
            "/metrics/pending_left_rows",
            json!(1),
            "state charge or gauges differ",
        ),
        (
            "/metrics/retained_right_rows",
            json!(2),
            "state charge or gauges differ",
        ),
        (
            "/metrics/identity_only_rows",
            json!(1),
            "state charge or gauges differ",
        ),
        (
            "/metrics/state_rows",
            json!(4),
            "state charge or gauges differ",
        ),
        (
            "/metrics/state_bytes",
            json!(0),
            "state charge or gauges differ",
        ),
        (
            "/metrics/matched_rows",
            json!(1),
            "output sequence, counters",
        ),
        (
            "/metrics/unmatched_rows",
            json!(1),
            "output sequence, counters",
        ),
        (
            "/metrics/emitted_left_rows",
            json!(1),
            "output sequence, counters",
        ),
        (
            "/metrics/left/accepted_rows",
            json!(3),
            "output sequence, counters",
        ),
        (
            "/metrics/right/accepted_rows",
            json!(4),
            "output sequence, counters",
        ),
        (
            "/metrics/evicted_right_rows",
            json!(1),
            "output sequence, counters",
        ),
        (
            "/next_output_sequence",
            json!(1),
            "output sequence, counters",
        ),
        ("/terminal", json!(true), "output sequence, counters"),
        (
            "/metrics/output_watermark_micros",
            json!(100),
            "invalid fixed shape",
        ),
        (
            "/metrics/left/watermark_micros",
            json!(100),
            "invalid fixed shape",
        ),
        (
            "/metrics/left/idle",
            json!(true),
            "must not own runtime progress",
        ),
        (
            "/metrics/right/ended",
            json!(true),
            "must not own runtime progress",
        ),
    ] {
        let mut damaged = original.clone();
        let mut metadata = serde_json::to_value(&damaged.inline_metadata).unwrap();
        *metadata.pointer_mut(path).unwrap() = value;
        damaged.inline_metadata = serde_json::from_value(metadata).unwrap();
        reject_without_replacing_state(&mut target, &damaged, message, path);
    }
    assert_original_answer(&mut target).await;
}

fn read_count(bytes: &[u8], cursor: &mut usize) -> usize {
    let value = u64::from_le_bytes(bytes[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    usize::try_from(value).unwrap()
}

fn sequence_width(flag: u8) -> usize {
    match flag {
        0 => 16,
        1..=4 => 1 << (flag - 1),
        5..=8 => 1 << (flag - 5),
        _ => panic!("sequence codec"),
    }
}

fn columnar_entry_ranges(bytes: &[u8]) -> [Vec<Range<usize>>; 3] {
    assert_eq!(&bytes[..8], b"CFASDL09");
    assert_eq!(&bytes[128..136], b"CFASOF09");
    let mut cursor = 136;
    let chunks = read_count(bytes, &mut cursor);
    let buckets = read_count(bytes, &mut cursor);
    read_count(bytes, &mut cursor);
    cursor += 64;
    let owners = read_count(bytes, &mut cursor);
    for _ in 0..owners {
        let kind = bytes[cursor];
        cursor += 2;
        let rows = read_count(bytes, &mut cursor);
        read_count(bytes, &mut cursor);
        let length = read_count(bytes, &mut cursor);
        read_count(bytes, &mut cursor);
        cursor += length;
        if kind != 0 {
            cursor += (rows + 1) * if kind == 1 { 4 } else { 8 };
        }
    }
    let mut left = Vec::new();
    for _ in 0..chunks {
        read_count(bytes, &mut cursor);
        let rows = read_count(bytes, &mut cursor);
        let keys = read_count(bytes, &mut cursor);
        let width = sequence_width(bytes[cursor]);
        cursor += 1 + 48 + keys * 16 + rows * 12;
        left.extend((0..rows).map(|row| cursor + row * width..cursor + (row + 1) * width));
        cursor += rows * (width + 4);
    }
    let mut keys = Vec::new();
    let mut first_right = Vec::new();
    for key in 0..buckets {
        let start = cursor;
        cursor += 16;
        let rows = read_count(bytes, &mut cursor);
        let width = sequence_width(bytes[cursor]);
        cursor += 1 + 40 + rows * 8;
        if key == 0 {
            first_right
                .extend((0..rows).map(|row| cursor + row * width..cursor + (row + 1) * width));
        }
        cursor += rows * width;
        let payloads = bytes[cursor..cursor + rows]
            .iter()
            .fold(0, |count, &tag| count + usize::from(tag == 1));
        cursor += rows + payloads * 12;
        keys.push(start..cursor);
    }
    assert_eq!(cursor, bytes.len());
    [left, keys, first_right]
}

#[tokio::test]
async fn test_asof_v3_restore_rejects_serialized_duplicates_and_noncanonical_order_atomically() {
    let original = populated_snapshot().await;
    let bytes = original.segments["asof-log-v9-1-0-1"].bytes();
    let mut target = operator(10);
    seed_live_state(&mut target, &schema()).await;
    for (entries, message) in columnar_entry_ranges(bytes).into_iter().zip([
        "chunk identity order is not strict",
        "right key order is not strict",
        "right identity order is not strict",
    ]) {
        assert_eq!(entries.len(), 2);
        for duplicate in [false, true] {
            let [first, second] = [&entries[0], &entries[1]];
            let middle = if duplicate {
                [bytes[first.clone()].to_vec(), bytes[first.clone()].to_vec()].concat()
            } else {
                [
                    bytes[second.clone()].to_vec(),
                    bytes[first.clone()].to_vec(),
                ]
                .concat()
            };
            let mut replacement = [
                bytes[..first.start].to_vec(),
                middle,
                bytes[second.end..].to_vec(),
            ]
            .concat();
            let body_bytes = u64::try_from(replacement.len() - 128).unwrap();
            replacement[112..120].copy_from_slice(&body_bytes.to_le_bytes());
            let replacement = StateSegment::new(replacement);
            let mut damaged = original.clone();
            let frame =
                &mut damaged.inline_metadata.get_mut("checkpoint_log").unwrap()["frames"][0];
            frame["sha256"] = json!(replacement.sha256());
            frame["bytes"] = json!(replacement.bytes().len());
            damaged
                .segments
                .insert("asof-log-v9-1-0-1".into(), replacement);
            reject_without_replacing_state(
                &mut target,
                &damaged,
                message,
                if duplicate {
                    "v3 serialized duplicate"
                } else {
                    "v3 descending order"
                },
            );
        }
    }
    assert_original_answer(&mut target).await;
}

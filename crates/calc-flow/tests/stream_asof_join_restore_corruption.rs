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
    let before_status = op.status();
    let before = op.checkpoint(Epoch::INITIAL).unwrap();
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
        ("state_version", json!(2)),
        ("layout_version", json!(2)),
        ("accounting_version", json!(2)),
        ("row_encoding", json!("arrow-row-unknown")),
    ] {
        let mut damaged = original.clone();
        damaged.inline_metadata.insert(field.into(), value);
        reject_without_replacing_state(&mut target, &damaged, "state version differs", field);
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

fn skip_blob(bytes: &[u8], cursor: &mut usize) {
    let length = read_count(bytes, cursor);
    *cursor += length;
}

fn ordered_entry_ranges(bytes: &[u8]) -> [Vec<Range<usize>>; 3] {
    assert_eq!(&bytes[..8], b"CFASOF01");
    let mut cursor = 8;
    let left_count = read_count(bytes, &mut cursor);
    let key_count = read_count(bytes, &mut cursor);
    let mut left = Vec::new();
    for _ in 0..left_count {
        let start = cursor;
        cursor += 8;
        for _ in 0..3 {
            skip_blob(bytes, &mut cursor);
        }
        left.push(start..cursor);
    }
    let mut keys = Vec::new();
    let mut first_right_rows = Vec::new();
    for key_index in 0..key_count {
        let start = cursor;
        skip_blob(bytes, &mut cursor);
        let count = read_count(bytes, &mut cursor);
        for _ in 0..count {
            let row_start = cursor;
            cursor += 8;
            skip_blob(bytes, &mut cursor);
            skip_blob(bytes, &mut cursor);
            if key_index == 0 {
                first_right_rows.push(row_start..cursor);
            }
        }
        keys.push(start..cursor);
    }
    assert_eq!(cursor, bytes.len());
    [left, keys, first_right_rows]
}

#[tokio::test]
async fn test_asof_restore_rejects_serialized_duplicates_and_noncanonical_order_atomically() {
    let original = populated_snapshot().await;
    let bytes = original.segments["asof-state-v1"].bytes();
    let mut target = operator(10);
    seed_live_state(&mut target, &schema()).await;
    for (entries, message) in ordered_entry_ranges(bytes).into_iter().zip([
        "left identity order is not strict",
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
            let replacement = [
                bytes[..first.start].to_vec(),
                middle,
                bytes[second.end..].to_vec(),
            ]
            .concat();
            let mut damaged = original.clone();
            damaged
                .segments
                .insert("asof-state-v1".into(), StateSegment::new(replacement));
            reject_without_replacing_state(
                &mut target,
                &damaged,
                message,
                if duplicate {
                    "serialized duplicate"
                } else {
                    "serialized descending order"
                },
            );
        }
    }
    assert_original_answer(&mut target).await;
}

mod asof_support;

use std::{collections::HashMap, sync::Arc};

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

#[tokio::test]
async fn test_asof_columnar_restore_rejects_serialized_duplicate_atomically() {
    let original = populated_snapshot().await;
    let identity_id = original
        .segments
        .keys()
        .find(|name| name.contains("-1-") && name.ends_with("-identity"))
        .unwrap()
        .clone();
    let identity = read_ipc(original.segments[&identity_id].bytes());
    let sequences = identity
        .column(2)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::BinaryArray>()
        .unwrap();
    let mut values = sequences
        .iter()
        .map(|value| value.unwrap().to_vec())
        .collect::<Vec<_>>();
    values[1] = values[0].clone();
    let mut columns = identity.columns().to_vec();
    columns[2] = Arc::new(datafusion::arrow::array::BinaryArray::from_iter_values(
        values.iter().map(Vec::as_slice),
    ));
    let identity = RecordBatch::try_new(identity.schema(), columns).unwrap();

    let payload_id = identity_id.replace("-identity", "-payload");
    let payload = read_ipc(original.segments[&payload_id].bytes());
    let payload_sequences = payload
        .column_by_name("sequence")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let mut sequence_values = payload_sequences.values().to_vec();
    sequence_values[1] = sequence_values[0];
    let mut columns = payload.columns().to_vec();
    let sequence = payload.schema().index_of("sequence").unwrap();
    columns[sequence] = Arc::new(Int64Array::from(sequence_values));
    let payload = RecordBatch::try_new(payload.schema(), columns).unwrap();

    let mut damaged = original.clone();
    damaged
        .segments
        .insert(identity_id, StateSegment::new(write_ipc(&identity)));
    damaged
        .segments
        .insert(payload_id, StateSegment::new(write_ipc(&payload)));
    let mut target = operator(10);
    seed_live_state(&mut target, &schema()).await;
    reject_without_replacing_state(&mut target, &damaged, "duplicate", "columnar duplicate");
    assert_original_answer(&mut target).await;
}

fn read_ipc(bytes: &[u8]) -> RecordBatch {
    let mut reader =
        datafusion::arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes), None)
            .unwrap();
    reader.next().unwrap().unwrap()
}

fn write_ipc(record: &RecordBatch) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut writer =
        datafusion::arrow::ipc::writer::StreamWriter::try_new(&mut bytes, record.schema().as_ref())
            .unwrap();
    writer.write(record).unwrap();
    writer.finish().unwrap();
    drop(writer);
    bytes
}

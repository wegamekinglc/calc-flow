use std::{io::Cursor, path::PathBuf, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, JsonMap, OperatorMetadata,
    OperatorSpec, OperatorStateSnapshot, PipelineBuilder, StateSegment, StreamJobContext,
    StreamJoinOperator, StreamOperator, StreamOperatorContext, StreamRequirements, UdfRegistry,
    export_project_json, import_project_json,
};
use datafusion::arrow::{
    array::{Array, Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    ipc::reader::StreamReader,
    record_batch::RecordBatch,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/asof-inner-compat-v1")
        .join(name)
}

fn fixture_json(name: &str) -> Value {
    serde_json::from_slice(&std::fs::read(fixture(name)).unwrap()).unwrap()
}

fn schema(left: bool) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("value", DataType::Int64, left),
    ]))
}

fn operator() -> StreamJoinOperator {
    let project =
        import_project_json(&std::fs::read(fixture("raw-explicit.json")).unwrap()).unwrap();
    let OperatorSpec::StreamJoin { spec } = project.graph.nodes[0].operator.clone() else {
        panic!("historical fixture must describe an inner Join");
    };
    StreamJoinOperator::new("match", schema(true), schema(false), spec).unwrap()
}

fn batch(left: bool, rows: &[(&str, i64, i64)]) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema(left),
                vec![
                    Arc::new(StringArray::from(
                        rows.iter().map(|row| row.0).collect::<Vec<_>>(),
                    )),
                    Arc::new(TimestampMicrosecondArray::from(
                        rows.iter().map(|row| row.1).collect::<Vec<_>>(),
                    )),
                    Arc::new(Int64Array::from(
                        rows.iter().map(|row| row.2).collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn context_job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "legacy-inner-fixture",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

async fn captured_snapshot() -> OperatorStateSnapshot {
    let mut join = operator();
    let job = context_job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(join.output_ports().to_vec());
    join.process_data(
        "left",
        batch(true, &[("a", 100, 10), ("b", 200, 20)]),
        &context,
        &mut output,
    )
    .await
    .unwrap();
    join.checkpoint(Epoch::INITIAL).unwrap();
    join.process_data(
        "right",
        batch(false, &[("a", 105, 30)]),
        &context,
        &mut output,
    )
    .await
    .unwrap();
    assert_eq!(
        output
            .drain("output")
            .iter()
            .map(|message| message.as_data().unwrap().num_rows())
            .sum::<usize>(),
        1
    );
    join.checkpoint(Epoch::new(2).unwrap()).unwrap()
}

fn historical_snapshot() -> OperatorStateSnapshot {
    let description = fixture_json("checkpoint.json");
    OperatorStateSnapshot {
        inline_metadata: serde_json::from_value(description["inline_metadata"].clone()).unwrap(),
        segments: description["segments"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, file)| {
                (
                    name.clone(),
                    StateSegment::new(std::fs::read(fixture(file.as_str().unwrap())).unwrap()),
                )
            })
            .collect(),
    }
}

#[test]
fn test_inner_explicit_and_omitted_defaults_preserve_canonical_bytes() {
    let expected = std::fs::read(fixture("canonical-project.json")).unwrap();
    for name in ["raw-explicit.json", "raw-omitted-prefixes.json"] {
        let project = import_project_json(&std::fs::read(fixture(name)).unwrap()).unwrap();
        assert_eq!(
            export_project_json(&project).unwrap().as_bytes(),
            expected,
            "{name}"
        );
    }
    assert_eq!(
        serde_json::to_vec(&operator().configuration()).unwrap(),
        std::fs::read(fixture("operator-configuration.json")).unwrap()
    );
}

#[test]
fn test_inner_output_schema_preserves_right_non_nullability_and_naive_time() {
    let join = operator();
    let actual = join.output_ports()[0].schema().unwrap().as_ref();
    let expected = Schema::new(vec![
        Field::new("left__key", DataType::Utf8, false),
        Field::new(
            "left__ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("left__value", DataType::Int64, true),
        Field::new("right__key", DataType::Utf8, false),
        Field::new(
            "right__ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("right__value", DataType::Int64, false),
    ]);
    assert_eq!(actual, &expected);
}

#[test]
fn test_inner_direct_native_fingerprint_preserves_historical_identity() {
    let plan = PipelineBuilder::new("legacy-inner")
        .unwrap()
        .add_node("match", operator())
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    assert_eq!(
        plan.fingerprint(),
        fixture_json("fingerprints.json")["native"]
    );
}

fn wire_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn v2_header(magic: [u8; 8], left: bool) -> Vec<u8> {
    let mut bytes = magic.to_vec();
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&[u8::from(!left), 0, 0, 0]);
    bytes
}

fn expected_delta(left: bool, rows: &[(&str, i64, i64)], digest: &[u8; 32]) -> Vec<u8> {
    let mut bytes = v2_header(*b"CFJDIX2\0", left);
    bytes.extend_from_slice(&u64::from(if left { 1_u8 } else { 2_u8 }).to_le_bytes());
    bytes.extend_from_slice(&u64::try_from(rows.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    for (row, &(key, time, _)) in rows.iter().enumerate() {
        let position = u64::try_from(row).unwrap().to_le_bytes();
        let mut framed_key = vec![10, 0, 0, 0, 0];
        framed_key.extend_from_slice(&u32::try_from(key.len()).unwrap().to_le_bytes());
        framed_key.extend_from_slice(key.as_bytes());
        bytes.extend_from_slice(&position);
        bytes.extend_from_slice(&time.to_le_bytes());
        bytes.extend_from_slice(&114_u64.to_le_bytes());
        bytes.extend_from_slice(digest);
        bytes.extend_from_slice(&position);
        bytes.extend_from_slice(&u64::try_from(framed_key.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&framed_key);
    }
    bytes
}

fn assert_payload_rows(ipc: &[u8], left: bool, expected: &[(&str, i64, i64)]) {
    let mut reader = StreamReader::try_new(Cursor::new(ipc), None).unwrap();
    assert_eq!(reader.schema(), schema(left));
    let record = reader.next().unwrap().unwrap();
    assert!(reader.next().is_none());
    assert_eq!(record.num_rows(), expected.len());
    let keys = record
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let times = record
        .column(1)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    let values = record
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(
        (keys.null_count(), times.null_count(), values.null_count()),
        (0, 0, 0)
    );
    for (row, &value) in expected.iter().enumerate() {
        assert_eq!(
            (keys.value(row), times.value(row), values.value(row)),
            value
        );
    }
}

fn assert_v2_side(
    snapshot: &OperatorStateSnapshot,
    left: bool,
    rows: &[(&str, i64, i64)],
) -> Value {
    let side = if left { "left" } else { "right" };
    let payloads = snapshot
        .segments
        .iter()
        .filter(|(name, _)| name.starts_with(&format!("{side}-payload-")))
        .collect::<Vec<_>>();
    assert_eq!(payloads.len(), 1);
    let (name, segment) = payloads[0];
    let bytes = segment.bytes();
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    let digest_hex = hex::encode(digest);
    assert_eq!(name, &format!("{side}-payload-{digest_hex}"));
    assert_eq!(&bytes[..16], v2_header(*b"CFJPAY2\0", left));
    assert_eq!(wire_u64(bytes, 16), u64::try_from(rows.len()).unwrap());
    let ipc_start = 32 + rows.len() * 8;
    assert_eq!(
        wire_u64(bytes, 24),
        u64::try_from(bytes.len() - ipc_start).unwrap()
    );
    for row in 0..rows.len() {
        assert_eq!(wire_u64(bytes, 32 + row * 8), u64::try_from(row).unwrap());
    }
    assert_payload_rows(&bytes[ipc_start..], left, rows);
    let mut empty_base = v2_header(*b"CFJIDX2\0", left);
    empty_base.extend_from_slice(&[0; 16]);
    assert_eq!(
        snapshot.segments[&format!("{side}-base")].bytes(),
        empty_base
    );
    let epoch = if left { 1 } else { 2 };
    assert_eq!(
        snapshot.segments[&format!("{side}-delta-{epoch}")].bytes(),
        expected_delta(left, rows, &digest)
    );
    json!({"side": side, "sha256": digest_hex, "rows": rows.len(), "bytes": bytes.len()})
}

fn assert_v2_inventory(snapshot: &OperatorStateSnapshot) {
    let left = assert_v2_side(snapshot, true, &[("a", 100, 10), ("b", 200, 20)]);
    let right = assert_v2_side(snapshot, false, &[("a", 105, 30)]);
    assert_eq!(
        snapshot.inline_metadata["v2_inventory"],
        json!({
            "codec_version": 2,
            "base_epoch": 0,
            "deltas": [{"epoch": 1, "sides": ["left"]}, {"epoch": 2, "sides": ["right"]}],
            "payloads": [left, right]
        })
    );
    assert_eq!(snapshot.segments.len(), 6);
}

fn output_pairs(output: &mut EdgeCollector) -> Vec<(i64, i64)> {
    output
        .drain("output")
        .iter()
        .flat_map(|message| {
            message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
        })
        .flat_map(|record| {
            let left = record
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let right = record
                .column(5)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..record.num_rows())
                .map(|row| (left.value(row), right.value(row)))
                .collect::<Vec<_>>()
        })
        .collect()
}

async fn assert_v2_restores_matches(snapshot: &OperatorStateSnapshot) {
    let mut join = operator();
    let job = context_job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(join.output_ports().to_vec());
    join.restore(snapshot).unwrap();
    let status = join.status();
    assert_eq!(
        (status.left.retained_rows, status.left.retained_bytes),
        (2, 228)
    );
    assert_eq!(
        (status.right.retained_rows, status.right.retained_bytes),
        (1, 114)
    );
    assert_eq!(status.emitted_match_rows, 1);
    assert!(output_pairs(&mut output).is_empty());
    join.process_data(
        "right",
        batch(false, &[("a", 110, 40)]),
        &context,
        &mut output,
    )
    .await
    .unwrap();
    assert_eq!(output_pairs(&mut output), [(10, 40)]);
    join.process_data(
        "left",
        batch(true, &[("a", 103, 50)]),
        &context,
        &mut output,
    )
    .await
    .unwrap();
    assert_eq!(output_pairs(&mut output), [(50, 30), (50, 40)]);
    assert_eq!(join.status().emitted_match_rows, 4);
}

#[tokio::test]
async fn test_inner_default_v2_checkpoint_preserves_historical_semantics() {
    let actual = captured_snapshot().await;
    let expected = historical_snapshot();
    assert_eq!(
        serde_json::to_vec(&expected.inline_metadata).unwrap(),
        std::fs::read(fixture("checkpoint-metadata.json")).unwrap()
    );
    let mut actual_common = actual.inline_metadata.clone();
    assert_eq!(actual_common.remove("layout_version"), Some(json!(2)));
    assert!(actual_common.remove("v2_inventory").is_some());
    let mut expected_common = expected.inline_metadata;
    assert_eq!(expected_common.remove("layout_version"), Some(json!(1)));
    assert_eq!(actual_common, expected_common);
    assert_v2_inventory(&actual);
    assert_v2_restores_matches(&actual).await;
}

#[tokio::test]
async fn test_inner_restores_historical_segments_and_keeps_all_matches() {
    let mut join = operator();
    let job = context_job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(join.output_ports().to_vec());
    join.restore(&historical_snapshot()).unwrap();
    assert!(output.drain("output").is_empty());
    assert_eq!(join.status().emitted_match_rows, 1);
    join.process_data(
        "right",
        batch(false, &[("a", 110, 40)]),
        &context,
        &mut output,
    )
    .await
    .unwrap();
    let right_outputs = output.drain("output");
    assert_eq!(
        right_outputs
            .iter()
            .map(|message| message.as_data().unwrap().num_rows())
            .sum::<usize>(),
        1
    );
    join.process_data(
        "left",
        batch(true, &[("a", 103, 50)]),
        &context,
        &mut output,
    )
    .await
    .unwrap();
    let outputs = output.drain("output");
    let actual = outputs
        .iter()
        .flat_map(|message| {
            message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
        })
        .flat_map(|batch| {
            let left = batch
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let right = batch
                .column(5)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..batch.num_rows())
                .map(|row| (left.value(row), right.value(row)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, [(50, 30), (50, 40)]);
    assert_eq!(join.status().emitted_match_rows, 4);
}

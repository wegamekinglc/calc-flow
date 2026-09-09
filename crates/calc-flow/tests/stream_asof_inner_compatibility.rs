use std::{path::PathBuf, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, JsonMap, OperatorMetadata,
    OperatorSpec, OperatorStateSnapshot, PipelineBuilder, StateSegment, StreamJobContext,
    StreamJoinOperator, StreamOperator, StreamOperatorContext, StreamRequirements, UdfRegistry,
    export_project_json, import_project_json,
};
use datafusion::arrow::{
    array::{Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::Value;

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

#[tokio::test]
async fn test_inner_checkpoint_metadata_and_segment_bytes_preserve_layout_one() {
    let actual = captured_snapshot().await;
    let expected = historical_snapshot();
    assert_eq!(
        serde_json::to_vec(&actual.inline_metadata).unwrap(),
        std::fs::read(fixture("checkpoint-metadata.json")).unwrap()
    );
    assert_eq!(actual.inline_metadata, expected.inline_metadata);
    assert_eq!(actual.segments, expected.segments);
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

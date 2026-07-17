mod support;

use std::{collections::BTreeMap, fs, path::PathBuf};

use calc_flow::{Batch, BatchKind, BatchMetadata, ExecutionOptions};
use datafusion::arrow::{compute::concat_batches, record_batch::RecordBatch};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize, Eq, PartialEq)]
struct Manifest {
    format_version: u32,
    arrow_files: Vec<String>,
    cases: Vec<FixtureCase>,
}

#[derive(Debug, Deserialize, Eq, PartialEq)]
struct FixtureCase {
    name: String,
    input: Value,
    operation: String,
    expected: Value,
    invariants: Vec<String>,
}

fn fixture_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/v1")
}

fn manifest() -> Manifest {
    serde_json::from_slice(&fs::read(fixture_directory().join("manifest.json")).unwrap()).unwrap()
}

fn case<'a>(manifest: &'a Manifest, name: &str) -> &'a FixtureCase {
    manifest
        .cases
        .iter()
        .find(|case| case.name == name)
        .unwrap()
}

fn assert_arrow_eq(actual: &Batch, expected: &[RecordBatch]) {
    let actual = actual.table_payload().unwrap();
    let expected_schema = expected.first().unwrap().schema();
    assert_eq!(actual.schema(), &expected_schema);
    assert_eq!(
        concat_batches(actual.schema(), actual.batches()).unwrap(),
        concat_batches(&expected_schema, expected).unwrap()
    );
}

#[test]
fn manifest_exactly_describes_the_frozen_core_corpus() {
    let actual = manifest();
    assert_eq!(actual.format_version, 1);
    assert_eq!(
        actual.arrow_files,
        [
            "expression.arrow",
            "expression_expected.arrow",
            "sql_left.arrow",
            "sql_right.arrow",
            "sql_expected.arrow",
            "empty.arrow",
        ]
    );
    assert_eq!(
        actual.cases,
        [
            FixtureCase {
                name: "expression_assignment".into(),
                input: json!("expression.arrow"),
                operation: "total = a + b".into(),
                expected: json!("expression_expected.arrow"),
                invariants: vec!["table_only".into(), "metadata_preserved".into()],
            },
            FixtureCase {
                name: "sql_join".into(),
                input: json!(["sql_left.arrow", "sql_right.arrow"]),
                operation: "join".into(),
                expected: json!("sql_expected.arrow"),
                invariants: vec!["single_select".into()],
            },
            FixtureCase {
                name: "empty_table".into(),
                input: json!("empty.arrow"),
                operation: "identity".into(),
                expected: json!("empty.arrow"),
                invariants: vec!["schema_preserved".into()],
            },
            FixtureCase {
                name: "metadata_round_trip".into(),
                input: json!("expression.arrow"),
                operation: "identity".into(),
                expected: json!("expression.arrow"),
                invariants: vec!["deeply_immutable_json".into()],
            },
            FixtureCase {
                name: "state_rollback".into(),
                input: json!("expression.arrow"),
                operation: "fail_after_state".into(),
                expected: json!({"state": {}}),
                invariants: vec!["rollback".into()],
            },
        ]
    );
    for name in &actual.arrow_files {
        assert!(!support::read_v1_fixture(name).is_empty());
    }
}

#[tokio::test]
async fn expression_assignment_matches_frozen_arrow_and_metadata_semantics() {
    let manifest = manifest();
    let fixture = case(&manifest, "expression_assignment");
    let metadata = BatchMetadata::new(
        "v1-fixture",
        7,
        BTreeMap::from([("nested".into(), json!({"preserved": [true, 3]}))]),
    )
    .unwrap();
    let input = Batch::table(
        support::read_v1_fixture(fixture.input.as_str().unwrap()),
        metadata.clone(),
    )
    .unwrap();
    let result = support::v1_expression_plan(&fixture.operation)
        .execute(
            BTreeMap::from([("input".into(), input.clone())]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    let output = &result.outputs["output"];

    assert_eq!(output.kind(), BatchKind::Table);
    assert_arrow_eq(
        output,
        &support::read_v1_fixture(fixture.expected.as_str().unwrap()),
    );
    assert_eq!(output.metadata(), &metadata);
    assert_eq!(input.metadata(), &metadata);
}

#[tokio::test]
async fn sql_join_matches_frozen_multi_input_semantics() {
    let manifest = manifest();
    let fixture = case(&manifest, "sql_join");
    let inputs = fixture.input.as_array().unwrap();
    let result = support::v1_sql_plan()
        .execute(
            BTreeMap::from([
                (
                    "l".into(),
                    Batch::table(
                        support::read_v1_fixture(inputs[0].as_str().unwrap()),
                        BatchMetadata::default(),
                    )
                    .unwrap(),
                ),
                (
                    "r".into(),
                    Batch::table(
                        support::read_v1_fixture(inputs[1].as_str().unwrap()),
                        BatchMetadata::default(),
                    )
                    .unwrap(),
                ),
            ]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();

    assert_arrow_eq(
        &result.outputs["output"],
        &support::read_v1_fixture(fixture.expected.as_str().unwrap()),
    );
}

#[tokio::test]
async fn empty_identity_preserves_the_frozen_schema() {
    let manifest = manifest();
    let fixture = case(&manifest, "empty_table");
    let input_batches = support::read_v1_fixture(fixture.input.as_str().unwrap());
    let input_schema = input_batches[0].schema();
    let result = support::v1_identity_plan("empty-identity")
        .execute(
            BTreeMap::from([(
                "input".into(),
                Batch::table(input_batches, BatchMetadata::default()).unwrap(),
            )]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    let output = &result.outputs["output"];

    assert_eq!(output.num_rows(), 0);
    assert_eq!(output.table_payload().unwrap().schema(), &input_schema);
    assert_arrow_eq(
        output,
        &support::read_v1_fixture(fixture.expected.as_str().unwrap()),
    );
}

#[tokio::test]
async fn metadata_identity_is_a_deeply_immutable_json_round_trip() {
    let manifest = manifest();
    let fixture = case(&manifest, "metadata_round_trip");
    let mut caller_attributes = BTreeMap::from([(
        "nested".into(),
        json!({"items": [{"id": 1}], "enabled": true}),
    )]);
    let metadata = BatchMetadata::new("metadata", 9, caller_attributes.clone()).unwrap();
    caller_attributes.get_mut("nested").unwrap()["items"][0]["id"] = json!(99);
    let wire = serde_json::to_string(&metadata).unwrap();
    let round_trip: BatchMetadata = serde_json::from_str(&wire).unwrap();
    assert_eq!(round_trip, metadata);
    assert_eq!(metadata.attributes()["nested"]["items"][0]["id"], 1);

    let result = support::v1_identity_plan("metadata-identity")
        .execute(
            BTreeMap::from([(
                "input".into(),
                Batch::table(
                    support::read_v1_fixture(fixture.input.as_str().unwrap()),
                    metadata.clone(),
                )
                .unwrap(),
            )]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    let output = &result.outputs["output"];

    assert_eq!(output.metadata(), &metadata);
    assert_arrow_eq(
        output,
        &support::read_v1_fixture(fixture.expected.as_str().unwrap()),
    );
}

#[tokio::test]
async fn failed_stateful_execution_restores_the_frozen_empty_state() {
    let manifest = manifest();
    let fixture = case(&manifest, "state_rollback");
    let plan = support::v1_rollback_plan();
    let error = plan
        .execute(
            BTreeMap::from([(
                "input".into(),
                Batch::table(
                    support::read_v1_fixture(fixture.input.as_str().unwrap()),
                    BatchMetadata::default(),
                )
                .unwrap(),
            )]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap_err();

    assert!(error.to_string().contains("fail_after_state"));
    assert_eq!(
        plan.snapshot().await.unwrap()["state"],
        fixture.expected["state"]
    );
}

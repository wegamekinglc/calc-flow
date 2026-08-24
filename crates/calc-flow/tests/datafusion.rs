use std::{collections::BTreeMap, fs::File, path::PathBuf, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, DataFusionConfig, DataFusionRuntime, ExternalPayload,
};
use datafusion::arrow::{
    array::Int64Array, compute::concat_batches, ipc::reader::FileReader, record_batch::RecordBatch,
};
use serde_json::json;

#[derive(Debug)]
struct TestArray;

impl ExternalPayload for TestArray {
    fn backend(&self) -> &'static str {
        "test"
    }

    fn len(&self) -> usize {
        1
    }

    fn estimated_bytes(&self) -> usize {
        8
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn input(values: Vec<i64>) -> Batch {
    let record =
        RecordBatch::try_from_iter(vec![("a", Arc::new(Int64Array::from(values)) as _)]).unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/v1")
        .join(name)
}

fn read_fixture(name: &str) -> Vec<RecordBatch> {
    FileReader::try_new(File::open(fixture_path(name)).unwrap(), None)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

fn fixture_batch(name: &str, metadata: BatchMetadata) -> Batch {
    Batch::table(read_fixture(name), metadata).unwrap()
}

fn assert_matches_fixture(actual: &Batch, expected_name: &str) {
    let actual = actual.table_payload().unwrap();
    let expected = read_fixture(expected_name);
    let expected_schema = expected.first().unwrap().schema();

    assert_eq!(actual.schema(), &expected_schema);
    assert_eq!(
        concat_batches(actual.schema(), actual.batches()).unwrap(),
        concat_batches(&expected_schema, &expected).unwrap()
    );
}

#[tokio::test]
async fn runtime_evaluates_v1_assignment_and_collects_metrics() {
    let metadata = BatchMetadata::new(
        "fixture",
        7,
        BTreeMap::from([("nested".into(), json!({"ok": true}))]),
    )
    .unwrap();
    let input = fixture_batch("expression.arrow", metadata.clone());
    let original_batches = input.table_payload().unwrap().batches().to_vec();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();

    let output = runtime
        .evaluate("total = a + b", &input, Some("calculate"))
        .await
        .unwrap();

    assert_matches_fixture(&output, "expression_expected.arrow");
    assert_eq!(output.metadata(), &metadata);
    assert_eq!(input.metadata(), &metadata);
    assert_eq!(input.table_payload().unwrap().batches(), original_batches);
    let metrics = runtime.metrics();
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].query_id, 1);
    assert_eq!(metrics[0].node_id.as_deref(), Some("calculate"));
    assert_eq!(metrics[0].output_rows, 2);
    assert!(metrics[0].planning_ns > 0);
    assert!(metrics[0].execution_ns > 0);
    assert!(metrics[0].logical_plan.contains("total"));
    assert!(!metrics[0].physical_plan.is_empty());
}

#[tokio::test]
async fn runtime_matches_v1_sql_join_and_preserves_the_input_map() {
    let left = fixture_batch("sql_left.arrow", BatchMetadata::default());
    let right = fixture_batch("sql_right.arrow", BatchMetadata::default());
    let left_before = left.table_payload().unwrap().batches().to_vec();
    let right_before = right.table_payload().unwrap().batches().to_vec();
    let tables = BTreeMap::from([("l".into(), left), ("r".into(), right)]);
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();

    let output = runtime
        .sql(
            "SELECT l.id, l.amount * r.rate AS total FROM l JOIN r ON l.id = r.id",
            &tables,
            Some("join"),
        )
        .await
        .unwrap();

    assert_matches_fixture(&output, "sql_expected.arrow");
    assert_eq!(output.metadata(), &BatchMetadata::default());
    assert_eq!(
        tables.keys().map(String::as_str).collect::<Vec<_>>(),
        ["l", "r"]
    );
    assert_eq!(tables["l"].table_payload().unwrap().batches(), left_before);
    assert_eq!(tables["r"].table_payload().unwrap().batches(), right_before);
    assert_eq!(runtime.metrics()[0].output_rows, 2);
}

#[tokio::test]
async fn failed_query_deregisters_aliases_before_the_next_query() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let tables = BTreeMap::from([("input".into(), input(vec![1]))]);

    let error = runtime
        .sql("SELECT * FROM missing", &tables, Some("broken"))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::DataFusion {
            node_id: Some(node_id),
            ..
        } if node_id == "broken"
    ));

    let output = runtime
        .sql("SELECT * FROM input", &tables, None)
        .await
        .unwrap();
    assert_eq!(output.num_rows(), 1);
    assert!(
        runtime
            .sql("SELECT * FROM input", &tables, None)
            .await
            .is_ok()
    );
    assert_eq!(runtime.metrics().len(), 2);
}

#[tokio::test]
async fn registration_error_deregisters_aliases_registered_so_far() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let tables = BTreeMap::from([
        ("first".into(), input(vec![1])),
        (
            "second".into(),
            Batch::external(Arc::new(TestArray), BatchMetadata::default()).unwrap(),
        ),
    ]);

    assert!(
        runtime
            .sql("SELECT * FROM first", &tables, None)
            .await
            .is_err()
    );

    let reusable = BTreeMap::from([("first".into(), input(vec![2]))]);
    assert!(
        runtime
            .sql("SELECT * FROM first", &reusable, None)
            .await
            .is_ok()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overlapping_evaluations_isolate_the_input_alias() {
    let runtime = Arc::new(
        DataFusionRuntime::new(DataFusionConfig {
            batch_size: 8_192,
            target_partitions: 4,
        })
        .unwrap(),
    );
    let first_input = Arc::new(input((0..524_288).rev().collect()));
    let second_input = Arc::new(input((0..131_072).rev().collect()));
    let start = Arc::new(tokio::sync::Barrier::new(3));

    // The bounded window sorts keep both real DataFusion calls active after a
    // barrier releases their separate runtime workers onto the shared alias.
    let evaluate = |input: Arc<Batch>, node_id| {
        let runtime = Arc::clone(&runtime);
        let start = Arc::clone(&start);
        tokio::spawn(async move {
            start.wait().await;
            runtime
                .evaluate("row_number() OVER (ORDER BY a)", &input, Some(node_id))
                .await
        })
    };
    let first = evaluate(Arc::clone(&first_input), "first");
    let second = evaluate(Arc::clone(&second_input), "second");

    start.wait().await;
    let (first, second) = tokio::join!(first, second);
    let first = first.unwrap().unwrap();
    let second = second.unwrap().unwrap();

    assert_eq!(first.num_rows(), 524_288);
    assert_eq!(second.num_rows(), 131_072);
    assert_eq!(runtime.metrics().len(), 2);
}

#[tokio::test]
async fn registration_error_names_the_alias_and_node_without_input_data() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let tables = BTreeMap::from([(
        "sensitive_alias".into(),
        Batch::external(Arc::new(TestArray), BatchMetadata::default()).unwrap(),
    )]);

    let error = runtime
        .sql(
            "SELECT * FROM sensitive_alias",
            &tables,
            Some("array_reader"),
        )
        .await
        .unwrap_err();

    assert!(matches!(
        &error,
        CalcFlowError::DataFusion {
            node_id: Some(node_id),
            message,
        } if node_id == "array_reader" && message.contains("sensitive_alias")
    ));
    assert!(!error.to_string().contains("TestArray"));
}

#[tokio::test]
async fn runtime_rejects_mutation_sql_empty_inputs_and_invalid_aliases() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let tables = BTreeMap::from([("input".into(), input(vec![1, 2]))]);

    assert!(
        runtime
            .sql("DELETE FROM input", &tables, None)
            .await
            .is_err()
    );
    assert!(
        runtime
            .sql("SELECT 1", &BTreeMap::new(), None)
            .await
            .is_err()
    );
    let invalid_alias = BTreeMap::from([("not-valid".into(), input(vec![1]))]);
    assert!(runtime.sql("SELECT 1", &invalid_alias, None).await.is_err());

    assert!(
        runtime
            .sql("SELECT * FROM input", &tables, None)
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn close_is_idempotent_and_rejects_later_queries() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let tables = BTreeMap::from([("input".into(), input(vec![1]))]);

    runtime.close();
    runtime.close();

    assert!(matches!(
        runtime.sql("SELECT * FROM input", &tables, None).await,
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "runtime"
    ));
}

#[test]
fn config_rejects_zero_batch_size_and_target_partitions() {
    for config in [
        DataFusionConfig {
            batch_size: 0,
            target_partitions: 1,
        },
        DataFusionConfig {
            batch_size: 8_192,
            target_partitions: 0,
        },
    ] {
        assert!(matches!(
            DataFusionRuntime::new(config),
            Err(CalcFlowError::InvalidArgument { field, .. }) if field.starts_with("datafusion.")
        ));
    }
}

#[tokio::test]
async fn sql_normalizes_a_zero_row_result_to_one_empty_batch() {
    // A zero-row INNER JOIN collects to zero RecordBatches; the runtime-level
    // contract represents an empty table as one zero-row batch instead of an
    // error (the same wording Batch::table's error message prescribes).
    let tables = BTreeMap::from([("l".into(), input(vec![1])), ("r".into(), input(vec![2]))]);
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();

    let output = runtime
        .sql(
            "SELECT l.a FROM l JOIN r ON l.a = r.a",
            &tables,
            Some("join"),
        )
        .await
        .unwrap();

    assert_eq!(output.num_rows(), 0);
    let payload = output.table_payload().unwrap();
    assert_eq!(payload.batches().len(), 1);
    assert_eq!(payload.batches()[0].num_rows(), 0);
}

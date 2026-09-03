use std::{collections::BTreeMap, fs::File, path::PathBuf, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, DataFusionConfig, DataFusionRuntime, ExternalPayload,
};
use datafusion::arrow::{
    array::{Float64Array, Int64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    compute::concat_batches,
    datatypes::{DataType, Field, Schema, TimeUnit},
    ipc::reader::FileReader,
    record_batch::RecordBatch,
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

fn rolling_input() -> Batch {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("price", DataType::Float64, true),
    ]));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![1, 1, 2, 2, 3])),
            Arc::new(UInt64Array::from(vec![1, 1, 2, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "a", "b", "a"])),
            Arc::new(Float64Array::from(vec![1.0, 10.0, 3.0, 14.0, 5.0])),
        ],
    )
    .unwrap();
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
    assert_eq!(metrics[0].physical_planning_count, 1);
    assert!(metrics[0].sql_parse_ns > 0);
    assert!(metrics[0].logical_planning_ns > 0);
    assert!(metrics[0].physical_planning_ns > 0);
    assert!(metrics[0].planning_ns > 0);
    assert!(metrics[0].stream_open_ns > 0);
    assert!(metrics[0].execution_ns > 0);
    assert!(metrics[0].collect_ns > 0);
    assert_eq!(
        metrics[0].planning_ns,
        metrics[0]
            .sql_parse_ns
            .saturating_add(metrics[0].logical_planning_ns)
            .saturating_add(metrics[0].physical_planning_ns)
    );
    assert_eq!(
        metrics[0].execution_ns,
        metrics[0]
            .stream_open_ns
            .saturating_add(metrics[0].collect_ns)
    );
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
async fn bounded_float64_avg_uses_the_shared_rolling_physical_kernel() {
    let runtime = DataFusionRuntime::new(DataFusionConfig {
        batch_size: 2,
        target_partitions: 4,
    })
    .unwrap();
    let tables = BTreeMap::from([("input".to_owned(), rolling_input())]);

    let output = runtime
        .sql(
            "SELECT event_time, sequence, symbol, price, \
             avg(price) OVER (PARTITION BY symbol ORDER BY event_time, sequence \
             ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS sma_2 FROM input",
            &tables,
            Some("rolling_sql"),
        )
        .await
        .unwrap();

    let table = output.table_payload().unwrap();
    let record = concat_batches(table.schema(), table.batches()).unwrap();
    let symbol = record
        .column_by_name("symbol")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let sequence = record
        .column_by_name("sequence")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let actual = record
        .column_by_name("sma_2")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let rows = (0..record.num_rows())
        .map(|index| {
            (
                (symbol.value(index).to_owned(), sequence.value(index)),
                actual.value(index),
            )
        })
        .collect::<BTreeMap<_, _>>();
    for (key, expected) in [
        (("a".to_owned(), 1), 1.0_f64),
        (("a".to_owned(), 2), 2.0),
        (("a".to_owned(), 3), 4.0),
        (("b".to_owned(), 1), 10.0),
        (("b".to_owned(), 2), 12.0),
    ] {
        assert_eq!(rows[&key].to_bits(), expected.to_bits());
    }

    let metrics = runtime.metrics();
    assert_eq!(metrics[0].configured_target_partitions, 4);
    assert_eq!(metrics[0].effective_target_partitions, 1);
    assert_eq!(metrics[0].rolling_candidate_windows, 1);
    assert_eq!(metrics[0].rolling_rewritten_windows, 1);
    assert!(metrics[0].rolling_fallback_reasons.is_empty());
    assert!(metrics[0].physical_plan.contains("CalcFlowRollingExec"));
}

#[tokio::test]
async fn unsupported_sql_window_stays_on_the_datafusion_fallback() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let tables = BTreeMap::from([("input".to_owned(), rolling_input())]);

    let output = runtime
        .sql(
            "SELECT sum(price) OVER (PARTITION BY symbol ORDER BY event_time, sequence \
             ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS rolling_sum FROM input",
            &tables,
            Some("rolling_fallback"),
        )
        .await
        .unwrap();

    assert_eq!(output.num_rows(), 5);
    let metrics = runtime.metrics();
    assert_eq!(metrics[0].rolling_candidate_windows, 1);
    assert_eq!(metrics[0].rolling_rewritten_windows, 0);
    assert_eq!(
        metrics[0].rolling_fallback_reasons,
        ["window_aggregate_is_not_avg"]
    );
    assert!(!metrics[0].physical_plan.contains("CalcFlowRollingExec"));
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

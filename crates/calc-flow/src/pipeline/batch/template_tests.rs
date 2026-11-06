use std::{collections::BTreeMap, sync::Arc};

use datafusion::arrow::{
    array::{Int64Array, StringArray},
    record_batch::RecordBatch,
};

use crate::{
    Batch, BatchExecutionPlan, BatchMetadata, ExecutionOptions, PipelineBuilder, SqlOperator,
    UdfRegistry,
};

fn plan(query: &str, aliases: &[&str]) -> BatchExecutionPlan {
    PipelineBuilder::new("reusable finite query")
        .unwrap()
        .add_node(
            "query",
            Box::new(
                SqlOperator::new(
                    "query",
                    query,
                    aliases.iter().map(|alias| (*alias).to_owned()).collect(),
                    vec![],
                )
                .unwrap(),
            ),
        )
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap()
}

fn keyed(keys: &[&str], values: &[i64]) -> Batch {
    let record = RecordBatch::try_from_iter(vec![
        ("key", Arc::new(StringArray::from(keys.to_vec())) as _),
        ("value", Arc::new(Int64Array::from(values.to_vec())) as _),
    ])
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

/// The fed-back state carries the aggregate output's nullable schema.
fn empty_state() -> Batch {
    let record = RecordBatch::try_from_iter_with_nullable(vec![
        (
            "key",
            Arc::new(StringArray::from(Vec::<&str>::new())) as _,
            true,
        ),
        (
            "value",
            Arc::new(Int64Array::from(Vec::<i64>::new())) as _,
            true,
        ),
    ])
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn rows(batch: &Batch) -> Vec<(String, i64)> {
    let mut rows = batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| {
            let keys = record
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let values = record
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..record.num_rows()).map(move |row| (keys.value(row).to_owned(), values.value(row)))
        })
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

#[tokio::test]
async fn cached_batch_preparation_keeps_no_input_rows_or_operator_state() {
    let plan = plan(
        "SELECT value + 1 AS result FROM input WHERE value > 0",
        &["input"],
    );
    for (index, value) in [1_i64, 9].into_iter().enumerate() {
        let record = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![value])) as _,
        )])
        .unwrap();
        let input = Batch::table(vec![record.clone()], BatchMetadata::default()).unwrap();
        let result = plan
            .execute(
                BTreeMap::from([("input".into(), input.clone())]),
                ExecutionOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(result.datafusion_metrics.len(), 1);
        assert_eq!(
            result.datafusion_metrics[0].physical_planning_count,
            u32::from(index == 0)
        );
        assert_eq!(plan.query_templates.len(), 1);
        assert_eq!(plan.query_templates.retained_rows(), 0);
        assert_eq!(input.table_payload().unwrap().batches(), &[record]);
        assert_eq!(
            plan.snapshot().await.unwrap()["query"],
            serde_json::Value::Null
        );
        plan.reset().await.unwrap();
    }
}

#[tokio::test]
async fn multi_input_aggregate_query_reuses_its_template() {
    let plan = plan(
        "SELECT key, SUM(value) AS value FROM (
            SELECT key, value FROM previous UNION ALL SELECT key, value FROM input
        ) GROUP BY key",
        &["input", "previous"],
    );
    let mut previous = empty_state();
    let inputs = [
        keyed(&["a", "b"], &[1, 2]),
        keyed(&["a", "c"], &[10, 30]),
        keyed(&["b"], &[200]),
    ];
    for (index, input) in inputs.iter().enumerate() {
        let mut result = plan
            .execute(
                BTreeMap::from([
                    ("input".into(), input.clone()),
                    ("previous".into(), previous.clone()),
                ]),
                ExecutionOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(result.datafusion_metrics.len(), 1);
        assert_eq!(
            result.datafusion_metrics[0].physical_planning_count,
            u32::from(index == 0),
            "run {index} must bind the captured aggregate template"
        );
        assert_eq!(plan.query_templates.len(), 1);
        assert_eq!(plan.query_templates.retained_rows(), 0);
        previous = result.outputs.remove("output").unwrap();
    }
    assert_eq!(
        rows(&previous),
        vec![
            ("a".to_owned(), 11),
            ("b".to_owned(), 202),
            ("c".to_owned(), 30),
        ]
    );
}

#[tokio::test]
async fn same_schema_inputs_bind_to_their_own_aliases() {
    let plan = plan(
        "SELECT 'left' AS key, SUM(value) AS value FROM left_input
         UNION ALL
         SELECT 'right' AS key, SUM(value) AS value FROM right_input",
        &["left_input", "right_input"],
    );
    for (index, (left, right)) in [(1_i64, 2_i64), (10, 20), (300, 400)]
        .into_iter()
        .enumerate()
    {
        let mut result = plan
            .execute(
                BTreeMap::from([
                    ("left_input".into(), keyed(&["x"], &[left])),
                    ("right_input".into(), keyed(&["x"], &[right])),
                ]),
                ExecutionOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(
            result.datafusion_metrics[0].physical_planning_count,
            u32::from(index == 0)
        );
        assert_eq!(
            rows(&result.outputs.remove("output").unwrap()),
            vec![("left".to_owned(), left), ("right".to_owned(), right)],
            "run {index} must read each alias from its own rows"
        );
    }
}

#[tokio::test]
async fn join_queries_plan_every_run() {
    let plan = plan(
        "SELECT facts.key, SUM(facts.value) AS value FROM facts
         JOIN dimension ON facts.key = dimension.key GROUP BY facts.key",
        &["facts", "dimension"],
    );
    for _ in 0..2 {
        let result = plan
            .execute(
                BTreeMap::from([
                    ("facts".into(), keyed(&["a", "a"], &[1, 2])),
                    ("dimension".into(), keyed(&["a"], &[0])),
                ]),
                ExecutionOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(result.datafusion_metrics[0].physical_planning_count, 1);
        assert_eq!(plan.query_templates.len(), 0);
    }
}

#[tokio::test]
async fn limited_queries_plan_every_run() {
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    fn nullable(rows: &[(Option<&str>, Option<i64>)]) -> Batch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, true),
            Field::new("value", DataType::Int64, true),
        ]));
        let record = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(
                    rows.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
                )) as _,
                Arc::new(Int64Array::from(
                    rows.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        Batch::table(vec![record], BatchMetadata::default()).unwrap()
    }
    // Top-k sorts carry execution-updated dynamic filters, so a bound tree
    // would prune the second history against the first run's bounds.
    let query = "SELECT key, SUM(value) AS total FROM events GROUP BY key \
                 HAVING SUM(value) > 1 ORDER BY key DESC NULLS FIRST LIMIT 4";
    let first = nullable(&[
        (Some("A\0B"), Some(1)),
        (Some("\u{e9}"), None),
        (Some(""), Some(0)),
        (None, Some(2)),
        (Some("e\u{301}"), Some(3)),
    ]);
    let mut history = first.table_payload().unwrap().batches().to_vec();
    history.extend(
        nullable(&[
            (Some("A\0B"), Some(2)),
            (Some("\u{e9}"), Some(4)),
            (Some("\u{65b0}\u{1f600}"), Some(5)),
            (None, None),
        ])
        .table_payload()
        .unwrap()
        .batches()
        .to_vec(),
    );
    let history = Batch::table(history, BatchMetadata::default()).unwrap();
    let reused = plan(query, &["events"]);
    for input in [first, history.clone()] {
        let result = reused
            .execute(
                BTreeMap::from([("events".into(), input)]),
                ExecutionOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(result.datafusion_metrics[0].physical_planning_count, 1);
        assert_eq!(reused.query_templates.len(), 0);
    }
    let mut second = reused
        .execute(
            BTreeMap::from([("events".into(), history.clone())]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    let mut fresh = plan(query, &["events"])
        .execute(
            BTreeMap::from([("events".into(), history)]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        rows(&second.outputs.remove("output").unwrap()),
        rows(&fresh.outputs.remove("output").unwrap())
    );
}

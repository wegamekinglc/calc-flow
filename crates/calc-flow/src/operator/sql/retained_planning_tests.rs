use std::sync::atomic::Ordering;

use datafusion::arrow::{
    array::ArrayRef,
    datatypes::{Field, Schema},
};

use super::*;

fn planner_input(sequence: u64) -> Batch {
    let (keys, values) = if sequence == 0 {
        (
            vec![Some(1), Some(2), None, Some(1)],
            vec![Some(5), None, Some(7), Some(-2)],
        )
    } else {
        (
            vec![Some(2), None, Some(1), Some(3)],
            vec![Some(11), None, Some(13), Some(17)],
        )
    };
    let mut columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(keys))];
    columns.extend(
        (1..7)
            .map(|column| Arc::new(Int64Array::from(vec![Some(100_000 + column); 4])) as ArrayRef),
    );
    columns.push(Arc::new(Int64Array::from(values)));
    let fields = std::iter::once("key".into())
        .chain((1..8).map(|column| format!("v{column}")))
        .map(|name: String| {
            Field::new(name, DataType::Int64, true)
                .with_metadata([("origin".into(), "planning-input".into())].into())
        })
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        [("owner".into(), "planning-input".into())].into(),
    ));
    Batch::table(
        vec![RecordBatch::try_new(schema, columns).unwrap()],
        BatchMetadata::new(
            "planning-input",
            sequence,
            JsonMap::from([
                ("sequence".into(), json!(sequence)),
                ("nested".into(), json!({"labels":["first","second"]})),
            ]),
        )
        .unwrap(),
    )
    .unwrap()
}

async fn assert_two_batches_plan_once(query: &str) {
    let mut operator = SqlOperator::new("totals", query, vec!["events".into()], vec![]).unwrap();
    let schema = planner_input(0).table_payload().unwrap().schema().clone();
    operator = operator
        .with_ports(
            vec![Port::with_schema_ref("events", BatchKind::Table, true, Some(schema)).unwrap()],
            Port::new("output", BatchKind::Table, true, None).unwrap(),
        )
        .unwrap();
    let job = StreamJobContext::new(
        1,
        "retained-planning",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "totals", None);
    let oracle = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let mut prefix = Vec::new();
    for sequence in 0..2 {
        let input = planner_input(sequence);
        let metadata = input.metadata().clone();
        let before = input.table_payload().unwrap().batches().to_vec();
        prefix.extend(before.iter().cloned());
        let cumulative = Batch::table(prefix.clone(), metadata.clone()).unwrap();
        let expected = oracle
            .sql(
                query,
                &BTreeMap::from([("events".into(), cumulative)]),
                Some("oracle"),
            )
            .await
            .unwrap();
        operator
            .process_data("events", input.clone(), &context, &mut collector)
            .await
            .unwrap();
        let output = collector.drain("output");
        assert_eq!(output.len(), 1);
        let actual = output[0].as_data().unwrap();
        assert_eq!(actual.metadata(), &metadata);
        assert_eq!(
            actual.table_payload().unwrap().schema(),
            expected.table_payload().unwrap().schema()
        );
        assert_eq!(rows(actual), rows(&expected));
        assert_eq!(input.table_payload().unwrap().batches(), before);
        let retained = operator.retained.as_ref().unwrap();
        assert_eq!(retained.rows, 4 * (sequence + 1));
        assert_eq!(retained.records[0].num_columns(), 2);
        assert_eq!(retained.records[0].schema().field(0).name(), "key");
        assert_eq!(retained.records[0].schema().field(1).name(), "v7");
    }
    assert_eq!(oracle.incremental_sql_plan_calls.load(Ordering::Relaxed), 0,);
    let runtime = operator.retention_runtime().unwrap();
    assert_eq!(
        runtime.incremental_sql_plan_calls.load(Ordering::Relaxed),
        1,
        "dependency proof and physical binding repeated SQL planning",
    );
    let independent = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    independent
        .incremental_sql_plan(
            &parse_select_query(query).unwrap(),
            "events",
            planner_input(0).table_payload().unwrap().schema().clone(),
            "independent",
        )
        .await
        .unwrap();
    assert_eq!(
        independent
            .incremental_sql_plan_calls
            .load(Ordering::Relaxed),
        1
    );
    assert_eq!(
        runtime.incremental_sql_plan_calls.load(Ordering::Relaxed),
        1
    );
}

#[tokio::test]
async fn test_sql_retained_nonadjacent_columns_reuse_original_planning() {
    assert_two_batches_plan_once("SELECT key, SUM(v7) AS total FROM events GROUP BY key").await;
}

#[tokio::test]
async fn test_sql_retained_qualified_columns_reuse_original_planning() {
    assert_two_batches_plan_once(
        "SELECT events.key, SUM(events.v7) AS total FROM events GROUP BY events.key",
    )
    .await;
}

#[tokio::test]
async fn test_sql_retained_table_alias_reuses_original_planning() {
    assert_two_batches_plan_once(
        "SELECT e.key, SUM(e.v7) AS total FROM events AS e GROUP BY e.key",
    )
    .await;
}

use calc_flow::{
    Batch, BatchKind, BatchMetadata, BatchOperator, BatchOperatorContext, CancellationToken,
    EdgeCollector, JsonMap, Port, RunContext, SnapshotSqlOperator, SqlOperator, StreamJobContext,
};
use datafusion::arrow::{array::Int64Array, compute::concat_batches, record_batch::RecordBatch};
use std::{collections::BTreeMap, sync::Arc};

pub const QUERY: &str = "SELECT p.value AS key, SUM(q.value) AS total FROM p LEFT JOIN q ON p.value = q.value GROUP BY p.value ORDER BY p.value";

pub fn batch(parts: &[&[i64]]) -> Batch {
    Batch::table(
        parts
            .iter()
            .map(|values| {
                RecordBatch::try_from_iter(vec![(
                    "value",
                    Arc::new(Int64Array::from(values.to_vec()))
                        as Arc<dyn datafusion::arrow::array::Array>,
                )])
                .unwrap()
            })
            .collect(),
        BatchMetadata::new(
            "source",
            3,
            JsonMap::from([("tag".into(), serde_json::json!("preserved"))]),
        )
        .unwrap(),
    )
    .unwrap()
}

pub async fn operator(query: &str) -> SnapshotSqlOperator {
    let aliases = vec!["p".into(), "q".into()];
    let sql = SqlOperator::new("oracle", query, aliases.clone(), vec![]).unwrap();
    let schema = batch(&[&[]]).table_payload().unwrap().schema().clone();
    let output = sql
        .infer_schema(&BTreeMap::from([
            ("p".into(), schema.clone()),
            ("q".into(), schema.clone()),
        ]))
        .await
        .unwrap();
    SnapshotSqlOperator::new("valuation", query, aliases, vec![])
        .unwrap()
        .with_ports(
            ["p", "q"]
                .into_iter()
                .map(|name| {
                    Port::new(
                        name,
                        BatchKind::Table,
                        true,
                        Some(schema.fields().iter().map(|f| f.as_ref().clone()).collect()),
                    )
                    .unwrap()
                })
                .collect(),
            Port::new(
                "output",
                BatchKind::Table,
                true,
                Some(output.fields().iter().map(|f| f.as_ref().clone()).collect()),
            )
            .unwrap(),
        )
        .unwrap()
}

pub fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "snapshot-test",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

pub async fn oracle(query: &str, p: Batch, q: Batch) -> Batch {
    let mut sql = SqlOperator::new("oracle", query, vec!["p".into(), "q".into()], vec![]).unwrap();
    let run = RunContext::new(JsonMap::new(), None, CancellationToken::new()).unwrap();
    sql.process(
        &BTreeMap::from([("p".into(), p), ("q".into(), q)]),
        &BatchOperatorContext { run: &run },
    )
    .await
    .unwrap()
    .remove("output")
    .unwrap()
}

pub fn assert_output(collector: &mut EdgeCollector, expected: &Batch) {
    let messages = collector.drain("output");
    assert_eq!(messages.len(), 1);
    let actual = messages[0].as_data().unwrap().table_payload().unwrap();
    let expected = expected.table_payload().unwrap();
    assert_eq!(
        concat_batches(actual.schema(), actual.batches()).unwrap(),
        concat_batches(expected.schema(), expected.batches()).unwrap()
    );
}

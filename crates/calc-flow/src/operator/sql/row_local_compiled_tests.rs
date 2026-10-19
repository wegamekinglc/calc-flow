use super::*;
use std::sync::Arc;

use datafusion::arrow::{
    array::{Float64Array, Int64Array},
    datatypes::{DataType, Field, Schema},
};

use crate::{CancellationToken, EdgeCollector, StreamJobContext};

fn input(offset: i64) -> Batch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("sequence", DataType::Int64, false),
        Field::new("value", DataType::Float64, true),
    ]));
    let record = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from_iter_values(
                (0..1_024).map(|row| row + offset),
            )),
            Arc::new(
                (0..1_024)
                    .map(|row| (row % 7 != 0).then_some(f64::from(row) / 4.0))
                    .collect::<Float64Array>(),
            ),
        ],
    )
    .unwrap();
    Batch::table(
        vec![
            record.slice(0, 127),
            RecordBatch::new_empty(schema),
            record.slice(127, 897),
        ],
        BatchMetadata::new("row-local-input", 17, JsonMap::new()).unwrap(),
    )
    .unwrap()
}

fn operator(query: &str) -> SqlOperator {
    SqlOperator::new("row-local", query, vec!["input".into()], vec![]).unwrap()
}

fn job(token: CancellationToken) -> StreamJobContext {
    StreamJobContext::new(42, "row-local-query", JsonMap::new(), None, token)
}

async fn collect(
    operator: &mut SqlOperator,
    input: Batch,
    job: &StreamJobContext,
) -> Result<Batch> {
    let context = StreamOperatorContext::new(job, "row-local", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    if let Err(error) = operator
        .process_data("input", input, &context, &mut output)
        .await
    {
        assert!(output.drain("output").is_empty());
        return Err(error);
    }
    let batches = output.drain("output");
    assert_eq!(batches.len(), 1);
    Ok(batches[0].as_data().unwrap().clone())
}

async fn sql_output(operator: &mut SqlOperator, input: Batch) -> Result<Batch> {
    let inputs = BTreeMap::from([("input".into(), input)]);
    let tables = operator.collect_tables(&inputs, None)?;
    operator
        .stream_state
        .runtime()?
        .sql_validated(&operator.validated, &tables, Some(&operator.name))
        .await
}

async fn compare(
    operator: &mut SqlOperator,
    reference: &mut SqlOperator,
    offset: i64,
    job: &StreamJobContext,
) {
    let batch = input(offset);
    let before = batch.table_payload().unwrap().batches().to_vec();
    let expected = sql_output(reference, batch.clone()).await.unwrap();
    let actual = collect(operator, batch.clone(), job).await.unwrap();
    assert_eq!(actual.metadata(), expected.metadata());
    assert_eq!(actual.source_cursor(), expected.source_cursor());
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(
        actual.table_payload().unwrap().batches(),
        expected.table_payload().unwrap().batches()
    );
    assert_eq!(batch.table_payload().unwrap().batches(), before);
}

#[tokio::test]
async fn stream_cpu_sql_reuses_immutable_row_local_execution() {
    let mut operator = operator("SELECT sequence, value + 1 AS next FROM input WHERE value > 3.5");
    let mut reference = operator.clone();
    let job = job(CancellationToken::new());
    assert!(!operator.stream_aggregate);
    for offset in [0, 3, 9] {
        compare(&mut operator, &mut reference, offset, &job).await;
    }
    assert_eq!(
        operator
            .stream_state
            .runtime
            .as_ref()
            .unwrap()
            .metrics()
            .len(),
        1,
        "repeated immutable row-local SQL must avoid binding and scheduling a fresh plan"
    );
    assert!(operator.retained.is_none());
}

#[tokio::test]
async fn stream_cpu_sql_order_and_limit_keep_sql_execution() {
    let mut operator =
        operator("SELECT sequence, value FROM input ORDER BY value NULLS FIRST, sequence LIMIT 5");
    let mut reference = operator.clone();
    let job = job(CancellationToken::new());
    for offset in [0, 3] {
        compare(&mut operator, &mut reference, offset, &job).await;
    }
    assert_eq!(
        operator
            .stream_state
            .runtime
            .as_ref()
            .unwrap()
            .metrics()
            .len(),
        2
    );
}

#[tokio::test]
async fn stream_cpu_sql_cancelled_hot_call_emits_nothing_and_can_retry() {
    let mut operator = operator("SELECT sequence, value + 1 AS next FROM input");
    let mut reference = operator.clone();
    let token = CancellationToken::new();
    let cancelled = job(token.clone());
    compare(&mut operator, &mut reference, 0, &cancelled).await;
    token.cancel();
    assert!(matches!(
        collect(&mut operator, input(3), &cancelled).await,
        Err(CalcFlowError::Cancelled { .. })
    ));
    compare(
        &mut operator,
        &mut reference,
        9,
        &job(CancellationToken::new()),
    )
    .await;
    assert_eq!(
        operator
            .stream_state
            .runtime
            .as_ref()
            .unwrap()
            .metrics()
            .len(),
        1
    );
}

#[tokio::test]
async fn stream_cpu_sql_hot_error_preserves_error_and_cache() {
    let mut operator = operator("SELECT sequence, 10 / sequence AS ratio FROM input");
    let mut reference = operator.clone();
    let job = job(CancellationToken::new());
    compare(&mut operator, &mut reference, 1, &job).await;
    let expected = sql_output(&mut reference, input(0)).await.unwrap_err();
    let actual = collect(&mut operator, input(0), &job).await.unwrap_err();
    assert_eq!(actual.to_string(), expected.to_string());
    compare(&mut operator, &mut reference, 9, &job).await;
    assert_eq!(
        operator
            .stream_state
            .runtime
            .as_ref()
            .unwrap()
            .metrics()
            .len(),
        1
    );
}

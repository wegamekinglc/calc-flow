use calc_flow::arrow::{
    array::{Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use calc_flow::{
    Batch, BatchExecutionPlan, BatchMetadata, CancellationToken, ExecutionOptions, JoinStateLimits,
    JoinTimeBounds, PipelineBuilder, Result, StreamJoinOperator, StreamJoinSpec, UdfRegistry,
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

type Row = (Option<i64>, Option<i64>, i64);

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("id", DataType::Int64, false),
    ]))
}

fn data(chunks: &[&[Row]]) -> Batch {
    let records = chunks
        .iter()
        .map(|rows| {
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(rows.iter().map(|row| row.0).collect::<Int64Array>()),
                    Arc::new(
                        rows.iter()
                            .map(|row| row.1)
                            .collect::<TimestampMicrosecondArray>(),
                    ),
                    Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.2))),
                ],
            )
            .unwrap()
        })
        .collect();
    Batch::table(records, BatchMetadata::default()).unwrap()
}

fn plan(limits: JoinStateLimits) -> Result<BatchExecutionPlan> {
    PipelineBuilder::new("independent")?
        .add_node(
            "pairs",
            StreamJoinOperator::new("pairs", schema(), schema(), join_spec(limits)?)?,
        )?
        .compile_batch(&UdfRegistry::new().snapshot())
}

fn join_spec(limits: JoinStateLimits) -> Result<StreamJoinSpec> {
    StreamJoinSpec::inner(
        ["k"],
        ["k"],
        "ts",
        "ts",
        JoinTimeBounds::new(Duration::from_micros(2), Duration::from_micros(3))?,
        limits,
    )
}

fn inputs(left: Batch, right: Batch) -> BTreeMap<String, Batch> {
    BTreeMap::from([("left".into(), left), ("right".into(), right)])
}

async fn execute(plan: &BatchExecutionPlan, left: Batch, right: Batch) -> Result<Batch> {
    Ok(plan
        .execute(inputs(left, right), ExecutionOptions::default())
        .await?
        .outputs["output"]
        .clone())
}

fn pairs(batch: &Batch) -> Vec<(i64, i64)> {
    let mut result = Vec::new();
    for record in batch.table_payload().unwrap().batches() {
        let left = record
            .column_by_name("left__id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let right = record
            .column_by_name("right__id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        result.extend(
            left.values()
                .iter()
                .copied()
                .zip(right.values().iter().copied()),
        );
    }
    result.sort_unstable();
    result
}

async fn assert_fresh(plan: &BatchExecutionPlan) -> Result<()> {
    for time in [10, 100] {
        let probe = data(&[&[(Some(1), Some(time), 500)]]);
        for (left, right) in [(data(&[&[]]), probe.clone()), (probe, data(&[&[]]))] {
            let output = execute(plan, left, right).await?;
            assert!(pairs(&output).is_empty());
            assert_eq!(output.table_payload()?.schema().fields().len(), 6);
        }
    }
    assert!(pairs(&execute(plan, data(&[&[]]), data(&[&[]])).await?).is_empty());
    let fresh = data(&[&[(Some(1), Some(100), 500)]]);
    assert_eq!(
        pairs(&execute(plan, fresh.clone(), fresh).await?),
        vec![(500, 500)]
    );
    Ok(())
}

#[tokio::test]
async fn test_native_null_time_and_key_drop_preserves_exact_six_pairs() -> Result<()> {
    let plan = plan(JoinStateLimits::new(10, 10_000, 10)?)?;
    let left = data(&[
        &[(Some(1), Some(10), 1), (Some(1), None, 3)],
        &[(Some(1), Some(10), 2), (None, Some(10), 4)],
    ]);
    let right = data(&[
        &[
            (Some(1), Some(7), 7),
            (Some(1), Some(8), 80),
            (None, Some(10), 90),
        ],
        &[
            (Some(1), Some(8), 81),
            (Some(1), Some(13), 13),
            (Some(1), Some(14), 14),
            (Some(1), None, 91),
            (Some(2), Some(10), 92),
        ],
    ]);
    let output = execute(&plan, left, right).await?;
    assert_eq!(
        pairs(&output),
        vec![(1, 13), (1, 80), (1, 81), (2, 13), (2, 80), (2, 81)]
    );
    let expected = Arc::new(Schema::new(
        ["left", "right"]
            .into_iter()
            .flat_map(|prefix| {
                schema()
                    .fields()
                    .iter()
                    .map(move |field| {
                        Field::new(
                            format!("{prefix}__{}", field.name()),
                            field.data_type().clone(),
                            field.is_nullable(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    ));
    assert_eq!(output.table_payload()?.schema(), &expected);
    Ok(())
}

#[tokio::test]
async fn test_match_budget_spans_records_and_failure_does_not_poison_reuse() -> Result<()> {
    let plan = plan(JoinStateLimits::new(10, 10_000, 3)?)?;
    let left = data(&[&[(Some(1), Some(10), 1)], &[(Some(1), Some(10), 2)]]);
    let first = data(&[&[(Some(1), Some(8), 80)]]);
    assert_eq!(
        pairs(&execute(&plan, left.clone(), first).await?),
        vec![(1, 80), (2, 80)]
    );
    let right = data(&[&[(Some(1), Some(8), 80)], &[(Some(1), Some(13), 13)]]);
    let error = execute(&plan, left, right).await.unwrap_err();
    assert!(error.to_string().contains("match limit"), "{error}");
    assert_fresh(&plan).await
}

#[tokio::test]
async fn test_both_side_budgets_span_records_even_when_opposite_is_empty() -> Result<()> {
    // V1 charge: 64 overhead + (1 tag + 4 timezone length + 4 value length + 8 key bytes)
    // + 16 time/row-id bytes + three (1 validity + 8 value) payload cells.
    let row_bytes = 64 + 17 + 16 + 3 * 9;
    for limits in [
        JoinStateLimits::new(1, 10_000, 10)?,
        JoinStateLimits::new(10, row_bytes, 10)?,
    ] {
        let plan = plan(limits)?;
        assert_side_limit_retries(&plan).await?;
    }
    Ok(())
}

async fn assert_side_limit_retries(plan: &BatchExecutionPlan) -> Result<()> {
    let one = data(&[&[(Some(1), Some(10), 1)]]);
    let two = data(&[&[(Some(1), Some(10), 1)], &[(Some(1), Some(10), 2)]]);
    for (left, right) in [(one.clone(), data(&[&[]])), (data(&[&[]]), one)] {
        assert!(pairs(&execute(plan, left, right).await?).is_empty());
    }
    for (left, right) in [(two.clone(), data(&[&[]])), (data(&[&[]]), two)] {
        let error = execute(plan, left, right).await.unwrap_err();
        assert!(error.to_string().contains("state limit"), "{error}");
        assert_fresh(plan).await?;
    }
    Ok(())
}

#[tokio::test]
async fn test_schema_and_cancel_failures_each_allow_immediate_empty_retry() -> Result<()> {
    let plan = plan(JoinStateLimits::new(10, 10_000, 10)?)?;
    assert_fresh(&plan).await?;
    let fresh = data(&[&[(Some(1), Some(100), 500)]]);
    let wrong = Batch::table(
        vec![RecordBatch::new_empty(Arc::new(Schema::new(vec![
            Field::new("wrong", DataType::Int64, false),
        ])))],
        BatchMetadata::default(),
    )?;
    assert!(execute(&plan, wrong, fresh.clone()).await.is_err());
    assert_fresh(&plan).await?;
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert!(
        plan.execute(
            inputs(fresh.clone(), fresh),
            ExecutionOptions {
                cancellation,
                ..ExecutionOptions::default()
            },
        )
        .await
        .is_err()
    );
    assert_fresh(&plan).await
}

fn public_source(name: &str, nullable_time: bool) -> Result<calc_flow::TableExpr> {
    calc_flow::table_input(
        name,
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                nullable_time,
            ),
            Field::new("id", DataType::Int64, false),
        ])),
        calc_flow::InputOptions {
            entity_by: vec!["id".into()],
            event_time: Some("ts".into()),
            sequence_by: vec!["id".into()],
        },
    )
}

#[test]
fn test_rust_public_source_rejects_nullable_time_before_join_declaration() {
    for name in ["a", "b"] {
        let error = public_source(name, true).unwrap_err();
        assert!(
            error.to_string().contains("non-null timestamp[us, UTC]"),
            "{error}"
        );
        assert!(
            error
                .to_string()
                .contains(&format!("inputs.{name}.event_time")),
            "{error}"
        );
    }
}

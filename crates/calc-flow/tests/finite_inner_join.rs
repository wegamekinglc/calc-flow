use calc_flow::arrow::{
    array::{Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use calc_flow::{
    Batch, BatchMetadata, ExecutionOptions, JoinStateLimits, JoinTimeBounds, PipelineBuilder,
    Result, StreamJoinOperator, StreamJoinSpec, UdfRegistry,
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
    ]))
}
fn data(times: Vec<Option<i64>>) -> Batch {
    let n = times.len();
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int64Array::from(vec![1; n])),
                    Arc::new(TimestampMicrosecondArray::from(times)),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}
fn plan(limits: JoinStateLimits) -> Result<calc_flow::BatchExecutionPlan> {
    let spec = StreamJoinSpec::inner(
        ["k"],
        ["k"],
        "ts",
        "ts",
        JoinTimeBounds::new(Duration::from_micros(2), Duration::from_micros(3))?,
        limits,
    )?;
    PipelineBuilder::new("finite")?
        .add_node(
            "pairs",
            StreamJoinOperator::new("pairs", schema(), schema(), spec)?,
        )?
        .compile_batch(&UdfRegistry::new().snapshot())
}
#[tokio::test]
async fn finite_join_boundaries_duplicates_nulls_and_same_plan_isolation() -> Result<()> {
    let plan = plan(JoinStateLimits::new(100, 1_000_000, 100)?)?;
    let left = vec![Some(10), Some(10), None];
    let right = vec![Some(7), Some(8), Some(8), Some(13), Some(14), None];
    for (l, r, expected) in [
        (left.clone(), right.clone(), 6),
        (vec![], right.clone(), 0),
        (left, vec![], 0),
        (vec![], vec![], 0),
        (
            vec![Some(-10)],
            vec![Some(-12), Some(-7), Some(-13), Some(-6)],
            2,
        ),
    ] {
        let out = plan
            .execute(
                BTreeMap::from([("left".into(), data(l)), ("right".into(), data(r))]),
                ExecutionOptions::default(),
            )
            .await?;
        let batch = &out.outputs["output"];
        assert_eq!(batch.num_rows(), expected);
        assert_eq!(batch.table_payload()?.schema().fields().len(), 4);
        assert_eq!(batch.metadata().source(), "pairs");
        assert_eq!(batch.metadata().sequence(), 0);
    }
    Ok(())
}

#[tokio::test]
async fn complete_batch_limits_failure_retry_schema_and_cancellation() -> Result<()> {
    let plan = plan(JoinStateLimits::new(4, 1_000_000, 3)?)?;
    let left = data(vec![Some(10), Some(10)]);
    let right_one = data(vec![Some(8)]);
    let right = Batch::table(
        vec![
            right_one.table_payload()?.batches()[0].clone(),
            right_one.table_payload()?.batches()[0].clone(),
        ],
        BatchMetadata::default(),
    )?;
    assert!(
        plan.execute(
            BTreeMap::from([("left".into(), left.clone()), ("right".into(), right)]),
            ExecutionOptions::default()
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("match limit")
    );
    let oversized = data(vec![Some(10); 5]);
    assert!(
        plan.execute(
            BTreeMap::from([("left".into(), oversized), ("right".into(), data(vec![]))]),
            ExecutionOptions::default()
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("state limit")
    );
    let wrong = Batch::table(
        vec![
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "other",
                    DataType::Int64,
                    false,
                )])),
                vec![Arc::new(Int64Array::from(vec![1]))],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )?;
    assert!(
        plan.execute(
            BTreeMap::from([("left".into(), wrong), ("right".into(), right_one.clone())]),
            ExecutionOptions::default()
        )
        .await
        .is_err()
    );
    let token = calc_flow::CancellationToken::new();
    token.cancel();
    assert!(
        plan.execute(
            BTreeMap::from([("left".into(), left), ("right".into(), right_one.clone())]),
            ExecutionOptions {
                cancellation: token,
                ..ExecutionOptions::default()
            }
        )
        .await
        .is_err()
    );
    let result = plan
        .execute(
            BTreeMap::from([("left".into(), data(vec![])), ("right".into(), right_one)]),
            ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(result.outputs["output"].num_rows(), 0);
    assert!(
        plan.snapshot()
            .await?
            .values()
            .all(serde_json::Value::is_null)
    );
    assert!(
        plan.restore(&BTreeMap::from([(
            "pairs".into(),
            serde_json::json!({"rows": []})
        )]))
        .await
        .is_err()
    );
    Ok(())
}

#[tokio::test]
async fn native_null_keys_and_extreme_time_bounds() -> Result<()> {
    let plan = plan(JoinStateLimits::new(100, 1_000_000, 100)?)?;
    for time in [i64::MIN, i64::MAX] {
        let side = data(vec![Some(time)]);
        let out = plan
            .execute(
                BTreeMap::from([("left".into(), side.clone()), ("right".into(), side)]),
                ExecutionOptions::default(),
            )
            .await?;
        assert_eq!(out.outputs["output"].num_rows(), 1);
    }
    let null = Batch::table(
        vec![
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int64Array::from(vec![None])),
                    Arc::new(TimestampMicrosecondArray::from(vec![10])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )?;
    let out = plan
        .execute(
            BTreeMap::from([
                ("left".into(), null),
                ("right".into(), data(vec![Some(10)])),
            ]),
            ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(out.outputs["output"].num_rows(), 0);
    Ok(())
}

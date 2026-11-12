mod asof_support;

use calc_flow::{
    Batch, BatchExecutionPlan, ExecutionOptions, PipelineBuilder, Result, UdfRegistry,
};
use datafusion::arrow::array::Int64Array;
use std::collections::BTreeMap;

fn plan() -> Result<BatchExecutionPlan> {
    PipelineBuilder::new("finite_asof")?
        .add_node("asof", asof_support::operator(2))?
        .compile_batch(&UdfRegistry::new().snapshot())
}

fn values(batch: &Batch) -> Vec<Option<i64>> {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| {
            record
                .column_by_name("right__value")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>()
        })
        .collect()
}

#[tokio::test]
async fn finite_asof_complete_admission_and_warm_plan_isolation() -> Result<()> {
    let plan = plan()?;
    for (left, right, expected) in [
        (vec![("A", 10, 1, 99)], vec![("A", 8, 1, 7)], vec![Some(7)]),
        (vec![("A", 10, 1, 99)], vec![], vec![None]),
        (vec![], vec![("A", 8, 1, 7)], vec![]),
        (vec![], vec![], vec![]),
        (
            vec![("A", 10, 1, 99), ("A", 10, 2, 98)],
            vec![("A", 9, 2, 2), ("A", 9, 10, 10), ("A", 11, 1, 11)],
            vec![Some(10), Some(10)],
        ),
        (
            vec![("A", i64::MAX, 1, 99)],
            vec![("A", i64::MAX, 1, 42)],
            vec![Some(42)],
        ),
    ] {
        let output = plan
            .execute(
                BTreeMap::from([
                    ("left".into(), asof_support::batch(&left)),
                    ("right".into(), asof_support::batch(&right)),
                ]),
                ExecutionOptions::default(),
            )
            .await?;
        let batch = &output.outputs["output"];
        assert_eq!(values(batch), expected);
        assert_eq!(batch.table_payload()?.schema().fields().len(), 8);
        assert_eq!(batch.metadata().source(), "asof");
        assert_eq!(batch.metadata().sequence(), 0);
    }
    Ok(())
}

#[tokio::test]
async fn finite_asof_rejects_cross_record_duplicates_and_total_budget_then_retries() -> Result<()> {
    let plan = plan()?;
    let left = asof_support::batch(&[("A", 10, 1, 99)]);
    let one = asof_support::batch(&[("A", 8, 1, 7)]);
    let duplicate = Batch::table(
        vec![
            one.table_payload()?.batches()[0].clone(),
            asof_support::batch(&[("A", 8, 1, 777)])
                .table_payload()?
                .batches()[0]
                .clone(),
        ],
        calc_flow::BatchMetadata::default(),
    )?;
    for (left, right) in [(left.clone(), duplicate.clone()), (duplicate, one.clone())] {
        let error = plan
            .execute(
                BTreeMap::from([("left".into(), left), ("right".into(), right)]),
                ExecutionOptions::default(),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("duplicate"), "{error}");
    }
    assert_fresh_after_cancel(&plan, left, one).await
}

async fn assert_fresh_after_cancel(
    plan: &BatchExecutionPlan,
    left: Batch,
    one: Batch,
) -> Result<()> {
    let token = calc_flow::CancellationToken::new();
    token.cancel();
    assert!(
        plan.execute(
            BTreeMap::from([("left".into(), left.clone()), ("right".into(), one.clone())]),
            ExecutionOptions {
                cancellation: token,
                ..Default::default()
            }
        )
        .await
        .is_err()
    );
    let output = plan
        .execute(
            BTreeMap::from([("left".into(), left), ("right".into(), one)]),
            ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(values(&output.outputs["output"]), [Some(7)]);
    assert!(
        plan.snapshot()
            .await?
            .values()
            .all(serde_json::Value::is_null)
    );
    assert!(
        plan.restore(&BTreeMap::from([(
            "asof".into(),
            serde_json::json!({"rows":[]})
        )]))
        .await
        .is_err()
    );
    Ok(())
}

#[tokio::test]
async fn finite_asof_total_budget_is_shared_and_retry_is_fresh() -> Result<()> {
    let small = small_plan()?;
    let side = asof_support::batch(&[("A", 10, 1, 99)]);
    let error = small
        .execute(
            BTreeMap::from([("left".into(), side.clone()), ("right".into(), side)]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("max_state_rows"), "{error}");
    let output = small
        .execute(
            BTreeMap::from([
                ("left".into(), asof_support::batch(&[("B", 10, 1, 1)])),
                ("right".into(), asof_support::batch(&[])),
            ]),
            ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(values(&output.outputs["output"]), [None]);
    Ok(())
}

#[tokio::test]
async fn finite_asof_inclusive_tolerance_zero_negative_and_extreme_times() -> Result<()> {
    for (left, right, tolerance, expected) in [
        (10, 8, 2, Some(7)),
        (10, 7, 2, None),
        (10, 11, 2, None),
        (10, 10, 0, Some(7)),
        (10, 9, 0, None),
        (-10, -12, 2, Some(7)),
        (i64::MIN, i64::MIN, 0, Some(7)),
        (i64::MAX, i64::MAX, 0, Some(7)),
        (i64::MAX, i64::MIN, 9_007_199_254_740_991, None),
    ] {
        let plan = PipelineBuilder::new("boundaries")?
            .add_node("asof", asof_support::operator(tolerance))?
            .compile_batch(&UdfRegistry::new().snapshot())?;
        let out = plan
            .execute(
                BTreeMap::from([
                    ("left".into(), asof_support::batch(&[("A", left, 1, 99)])),
                    ("right".into(), asof_support::batch(&[("A", right, 1, 7)])),
                ]),
                ExecutionOptions::default(),
            )
            .await?;
        assert_eq!(values(&out.outputs["output"]), [expected]);
    }
    Ok(())
}

fn small_plan() -> Result<BatchExecutionPlan> {
    let small_spec = calc_flow::StreamAsofJoinSpec::new(
        asof_support::spec(2).left().clone(),
        asof_support::spec(2).right().clone(),
        std::time::Duration::from_micros(2),
        calc_flow::AsofStateLimits::new(1, 1_048_576)?,
    )?;
    PipelineBuilder::new("budget")?
        .add_node(
            "asof",
            calc_flow::StreamAsofJoinOperator::new(
                "asof",
                asof_support::schema(),
                asof_support::schema(),
                small_spec,
            )?,
        )?
        .compile_batch(&UdfRegistry::new().snapshot())
}

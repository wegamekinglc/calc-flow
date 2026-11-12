use calc_flow::arrow::{
    array::{Int64Array, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use calc_flow::{
    AggregateFunction, Batch, BatchMetadata, ExecutionOptions, PipelineBuilder, Result,
    UdfRegistry, WindowAggregateOperator, WindowSpec,
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("q", DataType::Int64, true),
    ]))
}
fn data(ts: Vec<Option<i64>>, q: Vec<Option<i64>>) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(TimestampMicrosecondArray::from(ts)),
                    Arc::new(Int64Array::from(q)),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}
#[tokio::test]
async fn finite_window_reuses_plan_and_finalizes_complete_input() -> Result<()> {
    let spec = WindowSpec::tumbling("ts", Duration::from_micros(10))?
        .aggregate(AggregateFunction::Count, "q", "n")?
        .aggregate(AggregateFunction::Sum, "q", "s")?;
    let plan = PipelineBuilder::new("finite")?
        .add_node(
            "bars",
            WindowAggregateOperator::new("bars", schema(), spec)?,
        )?
        .compile_batch(&UdfRegistry::new().snapshot())?;
    for (ts, q, starts, counts, sums) in [
        (
            vec![Some(-1), Some(0), Some(9), Some(10), None],
            vec![Some(2), Some(3), None, Some(5), Some(99)],
            vec![-10, 0, 10],
            vec![1, 1, 1],
            vec![Some(2), Some(3), Some(5)],
        ),
        (
            vec![Some(1)],
            vec![Some(8)],
            vec![0],
            vec![1],
            vec![Some(8)],
        ),
        (vec![], vec![], vec![], vec![], vec![]),
    ] {
        let out = plan
            .execute(
                BTreeMap::from([("input".into(), data(ts, q))]),
                ExecutionOptions::default(),
            )
            .await?;
        let batch = &out.outputs["output"];
        let record = &batch.table_payload()?.batches()[0];
        assert_eq!(
            record
                .column(0)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap()
                .values()
                .as_ref(),
            starts
        );
        assert_eq!(
            record
                .column(2)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .values()
                .as_ref(),
            counts
        );
        assert_eq!(
            record
                .column(3)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            sums
        );
        assert_eq!(batch.metadata().source(), "bars");
        assert_eq!(batch.metadata().sequence(), 0);
    }
    Ok(())
}

#[tokio::test]
async fn finite_window_errors_and_cancellation_do_not_leak_state() -> Result<()> {
    let spec = WindowSpec::tumbling("ts", Duration::from_micros(10))?.aggregate(
        AggregateFunction::Sum,
        "q",
        "s",
    )?;
    let plan = PipelineBuilder::new("failures")?
        .add_node(
            "bars",
            WindowAggregateOperator::new("bars", schema(), spec)?,
        )?
        .compile_batch(&UdfRegistry::new().snapshot())?;
    let invalid = [
        data(vec![Some(0), Some(1)], vec![Some(i64::MAX), Some(1)]),
        data(vec![Some(i64::MAX)], vec![Some(1)]),
        data(vec![Some(i64::MIN)], vec![Some(1)]),
    ];
    for batch in invalid {
        assert!(
            plan.execute(
                BTreeMap::from([("input".into(), batch)]),
                ExecutionOptions::default()
            )
            .await
            .is_err()
        );
        let output = plan
            .execute(
                BTreeMap::from([("input".into(), data(vec![Some(1)], vec![Some(8)]))]),
                ExecutionOptions::default(),
            )
            .await?;
        let record = &output.outputs["output"].table_payload()?.batches()[0];
        assert_eq!(record.num_rows(), 1);
        assert_eq!(
            record
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            8
        );
    }
    let cancellation = calc_flow::CancellationToken::new();
    cancellation.cancel();
    assert!(
        plan.execute(
            BTreeMap::from([("input".into(), data(vec![Some(1)], vec![Some(100)]))]),
            ExecutionOptions {
                cancellation,
                ..Default::default()
            }
        )
        .await
        .is_err()
    );
    let output = plan
        .execute(
            BTreeMap::from([("input".into(), data(vec![], vec![]))]),
            ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(output.outputs["output"].num_rows(), 0);
    let state = plan.snapshot().await?;
    assert!(state.values().all(serde_json::Value::is_null));
    assert!(
        plan.restore(&BTreeMap::from([(
            "bars".into(),
            serde_json::json!({"accumulators": []})
        )]))
        .await
        .is_err()
    );
    Ok(())
}

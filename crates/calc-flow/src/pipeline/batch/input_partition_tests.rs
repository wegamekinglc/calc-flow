use std::{collections::BTreeMap, sync::Arc};

use datafusion::arrow::{
    array::{Array, Int64Array},
    record_batch::RecordBatch,
};

use crate::{
    Batch, BatchMetadata, PipelineBuilder, SqlOperator, UdfRegistry,
    datafusion::split_record_batches,
};

fn record(rows: usize, start: i64) -> RecordBatch {
    RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from_iter_values(
            (0..rows).map(|row| i64::try_from(row).unwrap() + start),
        )) as Arc<dyn Array>,
    )])
    .unwrap()
}

fn input(rows: usize, batches: usize) -> Batch {
    let width = rows / batches;
    let records = (0..batches)
        .map(|batch| {
            let start = i64::try_from(batch * width).unwrap();
            record(width, start)
        })
        .collect();
    Batch::table(records, BatchMetadata::default()).unwrap()
}

fn plan() -> crate::BatchExecutionPlan {
    PipelineBuilder::new("input partitions")
        .unwrap()
        .add_node(
            "query",
            Box::new(
                SqlOperator::new(
                    "query",
                    "SELECT value + 1 AS result FROM input WHERE value > 0",
                    vec!["input".into()],
                    Vec::new(),
                )
                .unwrap(),
            ),
        )
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap()
}

fn values(batch: &Batch) -> Vec<i64> {
    let table = batch.table_payload().unwrap();
    let mut values = table
        .batches()
        .iter()
        .flat_map(|batch| {
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            column.values().to_vec()
        })
        .collect::<Vec<_>>();
    values.sort_unstable();
    values
}

fn split_widths(batches: &[RecordBatch], partitions: usize) -> Vec<usize> {
    split_record_batches(batches, partitions)
        .iter()
        .map(Vec::len)
        .collect()
}

#[test]
fn split_record_batches_keeps_existing_single_partition_shape() {
    let records: Vec<RecordBatch> = (0..5).map(|index| record(2, index * 2)).collect();
    assert_eq!(split_record_batches(&records, 1).len(), 1);
    assert_eq!(split_record_batches(&records, 1)[0].len(), 5);
    assert_eq!(split_record_batches(&records, 0).len(), 1);
}

#[test]
fn split_record_batches_is_contiguous_and_balanced() {
    let records: Vec<RecordBatch> = (0..5).map(|index| record(2, index * 2)).collect();
    assert_eq!(split_widths(&records, 3), vec![2, 2, 1]);
    assert_eq!(split_widths(&records, 5), vec![1, 1, 1, 1, 1]);
    // A request above the batch count is capped at the batch count.
    assert_eq!(split_widths(&records, 9), vec![1, 1, 1, 1, 1]);
    let values: Vec<i64> = split_record_batches(&records, 3)
        .into_iter()
        .flatten()
        .flat_map(|batch| {
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            column.values().to_vec()
        })
        .collect();
    assert_eq!(values, (0..10).collect::<Vec<_>>());
}

#[test]
fn split_record_batches_keeps_an_empty_input_single() {
    assert_eq!(
        split_record_batches(&[], 4),
        vec![Vec::<RecordBatch>::new()]
    );
}

#[test]
fn input_partitions_reject_zero() {
    assert!(plan().with_input_partitions(0).is_err());
}

#[tokio::test]
async fn partitioned_input_preserves_results_and_reports_partitions() {
    let mut reports = Vec::new();
    let mut outputs = Vec::new();
    for partitions in [1_usize, 4] {
        let plan = plan().with_input_partitions(partitions).unwrap();
        let input = input(800, 8);
        let result = plan
            .execute(
                BTreeMap::from([("input".to_owned(), input)]),
                crate::ExecutionOptions::default(),
            )
            .await
            .unwrap();
        let metric = result
            .datafusion_metrics
            .first()
            .expect("partitioned execution records a DataFusion metric");
        reports.push(metric.output_partition_count);
        outputs.push(values(&result.outputs["output"]));
    }
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(reports[0], 1, "single-partition input keeps one partition");
    assert_eq!(
        reports[1], 4,
        "four input partitions must reach the physical plan"
    );
}

#[tokio::test]
async fn changing_input_partitions_invalidates_a_warmed_template() {
    let plan = plan();
    let inputs = BTreeMap::from([("input".to_owned(), input(800, 8))]);
    let initial = plan
        .execute(inputs.clone(), crate::ExecutionOptions::default())
        .await
        .unwrap();
    let warmed = plan
        .execute(inputs.clone(), crate::ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(warmed.datafusion_metrics[0].physical_planning_count, 0);
    let unchanged = plan.with_input_partitions(1).unwrap();
    let unchanged_result = unchanged
        .execute(inputs.clone(), crate::ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(
        unchanged_result.datafusion_metrics[0].physical_planning_count,
        0
    );

    let partitioned = unchanged.with_input_partitions(4).unwrap();
    let result = partitioned
        .execute(inputs, crate::ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(
        values(&initial.outputs["output"]),
        values(&result.outputs["output"])
    );
    assert_eq!(result.datafusion_metrics[0].output_partition_count, 4);
}

#[tokio::test]
async fn configured_input_partitions_follow_each_input_batch_layout() {
    let plan = plan().with_input_partitions(4).unwrap();
    let initial = plan
        .execute(
            BTreeMap::from([("input".to_owned(), input(800, 1))]),
            crate::ExecutionOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(initial.datafusion_metrics[0].output_partition_count, 1);
    let changed = plan
        .execute(
            BTreeMap::from([("input".to_owned(), input(800, 8))]),
            crate::ExecutionOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        values(&initial.outputs["output"]),
        values(&changed.outputs["output"])
    );
    assert_eq!(changed.datafusion_metrics[0].output_partition_count, 4);
}

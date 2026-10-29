use std::{collections::BTreeMap, sync::Arc};

use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

use crate::{Batch, BatchMetadata, ExecutionOptions, PipelineBuilder, SqlOperator, UdfRegistry};

#[tokio::test]
async fn cached_batch_preparation_keeps_no_input_rows_or_operator_state() {
    let plan = PipelineBuilder::new("reusable finite query")
        .unwrap()
        .add_node(
            "query",
            Box::new(
                SqlOperator::new(
                    "query",
                    "SELECT value + 1 AS result FROM input WHERE value > 0",
                    vec!["input".into()],
                    vec![],
                )
                .unwrap(),
            ),
        )
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap();
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

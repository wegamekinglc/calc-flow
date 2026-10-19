use std::{collections::BTreeMap, sync::Arc};

use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

use crate::{
    Batch, BatchExecutionPlan, BatchMetadata, DataFusionConfig, DataFusionParallelismMode, Edge,
    PipelineBuilder, PortEndpoint, SqlOperator, UdfRegistry,
};

fn batch(rows: usize, start: i64) -> Batch {
    let record = RecordBatch::try_from_iter(vec![
        ("key", Arc::new(Int64Array::from(vec![1; rows])) as _),
        (
            "value",
            Arc::new(Int64Array::from_iter_values(
                (0..rows).map(|row| i64::try_from(row).unwrap() + start),
            )) as _,
        ),
    ])
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn config(mode: DataFusionParallelismMode, rewrite: bool) -> DataFusionConfig {
    DataFusionConfig {
        parallelism_mode: mode,
        target_partitions: 4,
        max_partitions: 4,
        min_rows_per_partition: 16,
        small_rows_threshold: 4,
        enable_rolling_rewrite: rewrite,
        ..DataFusionConfig::default()
    }
}

fn plan(config: DataFusionConfig, window: bool) -> BatchExecutionPlan {
    let mut builder = PipelineBuilder::new("batch parallel sql")
        .unwrap()
        .with_datafusion_config(config)
        .add_node(
            "z_join",
            Box::new(
                SqlOperator::new(
                    "join",
                    "SELECT facts.key, facts.value + dimension.value AS total \
                 FROM facts JOIN dimension ON facts.key = dimension.key ORDER BY total",
                    vec!["facts".into(), "dimension".into()],
                    vec![],
                )
                .unwrap(),
            ),
        )
        .unwrap();
    if !window {
        builder = builder
            .add_node(
                "a_small",
                Box::new(
                    SqlOperator::new(
                        "small",
                        "SELECT value FROM tiny",
                        vec!["tiny".into()],
                        vec![],
                    )
                    .unwrap(),
                ),
            )
            .unwrap();
    }
    if window {
        builder = builder
            .add_node(
                "zz_window",
                Box::new(
                    SqlOperator::new(
                        "window",
                        "SELECT key, total, AVG(total) OVER (PARTITION BY key ORDER BY total \
                     ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS average FROM input \
                     ORDER BY total",
                        vec!["input".into()],
                        vec![],
                    )
                    .unwrap(),
                ),
            )
            .unwrap()
            .connect(Edge::new(
                PortEndpoint::new("z_join", "output").unwrap(),
                PortEndpoint::new("zz_window", "input").unwrap(),
            ))
            .unwrap();
    }
    builder
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap()
}

fn inputs(plan: &BatchExecutionPlan, rows: usize) -> BTreeMap<String, Batch> {
    plan.external_inputs()
        .iter()
        .map(|(name, port)| {
            let value = match port.port.as_str() {
                "facts" => batch(rows, 0),
                "dimension" => batch(1, 10),
                "tiny" => batch(1, -1),
                port => panic!("unexpected input {port}"),
            };
            (name.clone(), value)
        })
        .collect()
}

fn records(batch: &Batch) -> RecordBatch {
    let table = batch.table_payload().unwrap();
    datafusion::arrow::compute::concat_batches(table.schema(), table.batches()).unwrap()
}

#[tokio::test]
async fn batch_parallel_sql_auto_uses_run_inputs_before_the_first_small_node() {
    let plan = plan(config(DataFusionParallelismMode::Auto, true), false);
    assert_eq!(plan.topological_order()[0], "a_small");
    let result = plan
        .execute(inputs(&plan, 64), crate::ExecutionOptions::default())
        .await
        .unwrap();
    let available = std::thread::available_parallelism().unwrap().get();
    for metric in result.datafusion_metrics {
        assert_eq!(metric.effective_target_partitions, available.min(4));
        assert_eq!(metric.decision_input_rows, 66);
        assert_eq!(metric.decision_active_entities, None);
        assert_eq!(
            metric.decision_active_entities_source,
            "batch_multi_input_sql"
        );
    }
}

#[tokio::test]
async fn batch_parallel_sql_auto_recalculates_for_each_execution() {
    let plan = plan(config(DataFusionParallelismMode::Auto, true), false);
    let small = plan
        .execute(inputs(&plan, 1), crate::ExecutionOptions::default())
        .await
        .unwrap();
    assert!(
        small
            .datafusion_metrics
            .iter()
            .all(|m| m.effective_target_partitions == 1)
    );
    let large = plan
        .execute(inputs(&plan, 64), crate::ExecutionOptions::default())
        .await
        .unwrap();
    let available = std::thread::available_parallelism().unwrap().get();
    assert!(large.datafusion_metrics.iter().all(|m| {
        m.effective_target_partitions == available.min(4) && m.decision_input_rows == 66
    }));
}

#[tokio::test]
async fn batch_parallel_sql_mixed_window_matches_fixed_with_both_rewrite_modes() {
    for rewrite in [true, false] {
        let auto = plan(config(DataFusionParallelismMode::Auto, rewrite), true);
        let actual = auto
            .execute(inputs(&auto, 64), crate::ExecutionOptions::default())
            .await
            .unwrap();
        assert_eq!(
            actual.datafusion_metrics[0].effective_target_partitions,
            std::thread::available_parallelism().unwrap().get().min(4)
        );
        for partitions in [1, 4] {
            let fixed = plan(
                DataFusionConfig {
                    target_partitions: partitions,
                    ..config(DataFusionParallelismMode::Fixed, rewrite)
                },
                true,
            );
            let expected = fixed
                .execute(inputs(&fixed, 64), crate::ExecutionOptions::default())
                .await
                .unwrap();
            assert_eq!(
                expected.datafusion_metrics[0].effective_target_partitions,
                partitions
            );
            assert_eq!(
                actual.outputs.keys().collect::<Vec<_>>(),
                expected.outputs.keys().collect::<Vec<_>>()
            );
            for (name, value) in &actual.outputs {
                assert_eq!(records(value), records(&expected.outputs[name]));
            }
        }
    }
}

#[tokio::test]
async fn batch_parallel_sql_single_input_retains_conservative_auto() {
    let plan = PipelineBuilder::new("single input")
        .unwrap()
        .with_datafusion_config(config(DataFusionParallelismMode::Auto, true))
        .add_node(
            "query",
            Box::new(
                SqlOperator::new(
                    "query",
                    "SELECT value FROM input ORDER BY value",
                    vec!["input".into()],
                    vec![],
                )
                .unwrap(),
            ),
        )
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap();
    let result = plan
        .execute(
            BTreeMap::from([("input".into(), batch(64, 0))]),
            crate::ExecutionOptions::default(),
        )
        .await
        .unwrap();
    let metric = &result.datafusion_metrics[0];
    assert_eq!(metric.effective_target_partitions, 1);
    assert_eq!(metric.partition_limit_reason, "missing_active_entities");
}

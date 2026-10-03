use super::*;
use datafusion::{
    arrow::{
        array::{Float32Array, Float64Array, Int64Array},
        datatypes::{DataType, Field, Schema},
    },
    datasource::memory::{DataSourceExec, MemorySourceConfig},
    physical_plan::{
        ExecutionPlan, InputOrderMode,
        aggregates::{AggregateExec, AggregateMode},
        empty::EmptyExec,
        projection::ProjectionExec,
    },
};
use serde_json::{Value, json};

const QUERY: &str =
    "SELECT key, MIN(value32), MAX(value32), MIN(value64), MAX(value64) FROM events GROUP BY key";

fn input(rows: usize, record_rows: usize, distinct: bool) -> Batch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value32", DataType::Float32, true),
        Field::new("value64", DataType::Float64, true),
    ]));
    let ranges = if rows == 0 {
        vec![(0, 0)]
    } else {
        (0..rows)
            .step_by(record_rows)
            .map(|start| (start, (start + record_rows).min(rows)))
            .collect()
    };
    let records = ranges
        .into_iter()
        .map(|(start, end)| {
            let keys = (start..end)
                .map(|row| i64::try_from(if distinct { row } else { row % 31 }).unwrap())
                .collect::<Vec<_>>();
            let values = (start..end)
                .map(|row| (row % 7 != 0).then_some(1.0))
                .collect::<Vec<_>>();
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(keys)),
                    Arc::new(Float32Array::from(values.clone())),
                    Arc::new(Float64Array::from(
                        values
                            .into_iter()
                            .map(|value| value.map(f64::from))
                            .collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap()
        })
        .collect();
    Batch::table(records, BatchMetadata::default()).unwrap()
}

#[derive(Default)]
struct Shape {
    aggregates: usize,
    non_single_linear: usize,
    scans: usize,
    non_fifo_nodes: usize,
    empty: usize,
}

fn inspect(plan: &dyn ExecutionPlan, shape: &mut Shape) -> Value {
    let mut node = json!({"node": plan.name(), "partitions": plan.output_partitioning().partition_count(),
        "statistics": format!("{:?}", plan.partition_statistics(None))});
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        shape.aggregates += 1;
        shape.non_single_linear += usize::from(
            *aggregate.mode() != AggregateMode::Single
                || *aggregate.input_order_mode() != InputOrderMode::Linear
                || aggregate.input().output_partitioning().partition_count() != 1
                || aggregate.limit_options().is_some()
                || aggregate.group_expr().groups().len() != 1
                || aggregate.filter_expr().iter().any(Option::is_some),
        );
        node["aggregate"] = json!({"mode": format!("{:?}", aggregate.mode()),
            "order": format!("{:?}", aggregate.input_order_mode()),
            "input_partitions": aggregate.input().output_partitioning().partition_count(),
            "limit": format!("{:?}", aggregate.limit_options()), "grouping_exprs": aggregate.group_expr().expr().len()});
    } else if let Some(scan) = plan
        .downcast_ref::<DataSourceExec>()
        .and_then(|source| source.data_source().downcast_ref::<MemorySourceConfig>())
    {
        shape.scans += 1;
        node["memory"] = json!({"partitions": scan.partitions().len(), "records": scan.partitions().iter()
            .map(|partition| partition.iter().map(RecordBatch::num_rows).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "sort_information": format!("{:?}", scan.sort_information())});
    } else if plan.is::<EmptyExec>() {
        shape.empty += 1;
    } else if !plan.is::<ProjectionExec>() {
        shape.non_fifo_nodes += 1;
    }
    node["children"] = Value::Array(
        plan.children()
            .iter()
            .map(|child| inspect(child.as_ref(), shape))
            .collect(),
    );
    node
}

fn assert_fifo_source(plan: &dyn ExecutionPlan, expected: &Batch) {
    if let Some(source) = plan.downcast_ref::<DataSourceExec>() {
        let scan = source
            .data_source()
            .downcast_ref::<MemorySourceConfig>()
            .unwrap();
        assert_eq!(scan.partitions().len(), 1);
        assert!(scan.sort_information().is_empty());
        let original = expected.table_payload().unwrap();
        assert_eq!(scan.original_schema(), original.schema().clone());
        let records = &scan.partitions()[0];
        assert_eq!(records.len(), original.batches().len());
        for (actual, expected) in records.iter().zip(original.batches()) {
            assert_eq!(actual, expected);
        }
    }
    for child in plan.children() {
        assert_fifo_source(child.as_ref(), expected);
    }
}

struct Observation {
    shape: Shape,
    spill_bytes: usize,
    failure: Option<String>,
    resource_refusal: bool,
    rows: usize,
}

async fn observe(
    runtime: &DataFusionRuntime,
    batch: Batch,
    label: &str,
    require_fifo: bool,
) -> Observation {
    let _guard = runtime.query_lock.lock().await;
    let context = runtime.context_for_rows(batch.num_rows(), None, "not_evaluated");
    let query = parse_select_query(QUERY).unwrap();
    let tables = BTreeMap::from([("events".into(), batch.clone())]);
    let planned = runtime
        .prepare_query(context, &query, &tables, Some(label))
        .await
        .unwrap();
    let plan = planned.physical_plan.clone();
    let mut shape = Shape::default();
    let tree = inspect(plan.as_ref(), &mut shape);
    if require_fifo {
        assert_fifo_source(plan.as_ref(), &batch);
    }
    let started = Instant::now();
    let stream = execute_stream(plan.clone(), planned.task_ctx.clone())
        .map_err(|error| datafusion_error(Some(label), error));
    let result = match stream {
        Ok(stream) => collect_bounded(stream, plan.schema(), started, Some(label)).await,
        Err(error) => Err(error),
    };
    let rows = result.as_ref().map_or(0, |output| {
        output.batches.iter().map(RecordBatch::num_rows).sum()
    });
    let failure = result.as_ref().err().map(ToString::to_string);
    let resource_refusal = matches!(result.as_ref(), Err(CalcFlowError::DataFusion { message, .. }) if message.contains("Resources exhausted: "));
    let statistics = physical_plan_statistics(plan.as_ref(), rows);
    println!(
        "GROUPED_FLOAT_DIAGNOSTIC {}",
        json!({"label": label,
        "input_rows": batch.num_rows(), "input_records": batch.table_payload().unwrap().batches().len(),
        "requested_partitions": runtime.config.target_partitions,
        "effective_partitions": runtime.effective_target_partitions.load(Ordering::Acquire),
        "decision_input_rows": runtime.decision().unwrap().input_rows,
        "tree": tree, "output_rows": rows, "error": failure,
        "spill_bytes": statistics.spill_bytes, "resource_refusal": resource_refusal, "metric_count": statistics.metric_count,
        "repartitions": statistics.repartition_operator_count, "sorts": statistics.sort_operator_count,
        "coalesces": statistics.coalesce_operator_count,
        "partition_rows": statistics.partition_rows})
    );
    Observation {
        shape,
        spill_bytes: statistics.spill_bytes,
        failure,
        resource_refusal,
        rows,
    }
}

fn assert_default(observed: &Observation, input_rows: usize) {
    assert!(observed.failure.is_none(), "{:?}", observed.failure);
    assert_eq!(observed.rows, input_rows.min(31));
    assert_eq!(observed.spill_bytes, 0);
    assert_eq!(observed.shape.non_single_linear, 0);
    assert_eq!(observed.shape.non_fifo_nodes, 0);
    if input_rows > 0 {
        assert_eq!(observed.shape.aggregates, 1);
        assert_eq!(observed.shape.scans, 1);
    } else {
        assert!(observed.shape.aggregates > 0 || observed.shape.empty > 0);
    }
}

#[tokio::test]
async fn test_grouped_float_default_prefix_physical_mode_and_fifo() {
    for record_rows in [257, 8192, 20_000] {
        for first in [0, 1] {
            let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
            for rows in [first, 0, 100_003] {
                let observed = observe(
                    &runtime,
                    input(rows, record_rows, false),
                    &format!("default-first{first}-rows{rows}-records{record_rows}"),
                    true,
                )
                .await;
                assert_default(&observed, rows);
                assert_eq!(
                    runtime.effective_target_partitions.load(Ordering::Acquire),
                    1
                );
            }
            assert_eq!(runtime.runtime_env.memory_pool.reserved(), 0);
        }
    }
}

#[tokio::test]
async fn test_grouped_float_nondefault_actual_partition_topology_control() {
    for first in [1, 131_073] {
        let runtime = DataFusionRuntime::new(DataFusionConfig {
            target_partitions: 4,
            ..DataFusionConfig::default()
        })
        .unwrap();
        for rows in [first, 131_073] {
            let observed = observe(
                &runtime,
                input(rows, 257, false),
                &format!("requested4-first{first}-rows{rows}"),
                false,
            )
            .await;
            assert!(observed.failure.is_none(), "{:?}", observed.failure);
            assert_eq!(observed.rows, rows.min(31));
            let effective = runtime.effective_target_partitions.load(Ordering::Acquire);
            assert_eq!(effective, if first == 1 { 1 } else { 3 });
            if effective > 1 {
                assert!(observed.shape.non_single_linear > 0 || observed.shape.non_fifo_nodes > 0);
            }
        }
        assert_eq!(runtime.runtime_env.memory_pool.reserved(), 0);
    }
}

#[tokio::test]
async fn test_grouped_float_low_pool_single_mode_spill_or_refusal_control() {
    let mut runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    runtime.runtime_env = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(GreedyMemoryPool::new(8 << 20)))
        .build_arc()
        .unwrap();
    let observed = observe(
        &runtime,
        input(200_003, 8192, true),
        "low-pool8MiB-distinct200003",
        true,
    )
    .await;
    assert_eq!(observed.shape.aggregates, 1);
    assert_eq!(observed.shape.non_single_linear, 0);
    assert_eq!(observed.shape.non_fifo_nodes, 0);
    assert!(
        observed.spill_bytes > 0 || observed.resource_refusal,
        "low pool did not force spill/refusal"
    );
    if observed.failure.is_none() {
        assert_eq!(observed.rows, 200_003);
    }
    assert_eq!(runtime.runtime_env.memory_pool.reserved(), 0);
}

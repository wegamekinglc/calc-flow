use super::super::ScalarValue;
use super::*;
use crate::{Batch, CancellationToken, DataFusionConfig, JsonMap, StreamJobContext};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};

fn unique_source() -> Batch {
    let strings = (0..MIN_ROWS)
        .map(|row| (row % 7 != 0).then(|| format!("λ\0{row:08}-{}", "x".repeat(row % 65))))
        .collect::<Vec<_>>();
    let record = RecordBatch::try_from_iter(vec![
        (
            "key",
            Arc::new(datafusion::arrow::array::StringArray::from(strings)) as ArrayRef,
        ),
        (
            "value",
            Arc::new(datafusion::arrow::array::Int64Array::from(vec![
                1;
                MIN_ROWS
            ])) as ArrayRef,
        ),
    ])
    .unwrap();
    Batch::table(vec![record], crate::BatchMetadata::default()).unwrap()
}

async fn plan(runtime: &DataFusionRuntime, input: &Batch, sql: &str) -> Option<IncrementalSql> {
    let query = crate::expression::parse_select_query(sql).unwrap();
    IncrementalSql::plan(
        runtime,
        &query,
        "events",
        input.table_payload().unwrap().schema().clone(),
        "parallel-admission",
    )
    .await
    .unwrap()
}

#[test]
fn test_parallel_sum_threshold_partition_and_checked_capacity() {
    assert_eq!(
        (0..4)
            .map(|ordinal| partition(10, 4, ordinal).unwrap())
            .collect::<Vec<_>>(),
        [0..2, 2..5, 5..7, 7..10]
    );
    assert_eq!(units(MIN_ROWS - 1, 4, 8), 1);
    assert_eq!(units(MIN_ROWS, 4, 1), 1);
    assert_eq!(units(MIN_ROWS, 4, 8), 2);
    assert_eq!(units(16 * CHUNK_ROWS, 16, 128), 8);
    for (chunks, count) in [(4, 2), (17, 8), (usize::MAX, 8)] {
        let ranges = (0..count)
            .map(|ordinal| partition(chunks, count, ordinal).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges.last().unwrap().end, chunks);
        assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
    }
    assert!(checked_bytes(0, [(usize::MAX, 256)], "parallel-capacity").is_err());
}

#[tokio::test]
async fn test_parallel_sum_physical_shape_and_predicate_admission_is_fail_closed() {
    let input = grouped_parallel_tests::source(MIN_ROWS);
    let records = input.table_payload().unwrap().batches();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    for sql in [
        "SELECT key, SUM(value), COUNT(*) FROM events GROUP BY key",
        "SELECT key, SUM(DISTINCT value) FROM events GROUP BY key",
        "SELECT key, SUM(value + 1) FROM events GROUP BY key",
        "SELECT key, SUM(CAST(value AS DOUBLE)) FROM events GROUP BY key",
        "SELECT key, SUM(CAST(value AS DECIMAL(20,0))) FROM events GROUP BY key",
        "SELECT key, SUM(value) FROM events WHERE CAST(key AS BIGINT) > 0 GROUP BY key",
        "SELECT key, SUM(value) FILTER (WHERE key = 0) FROM events GROUP BY key",
        "SELECT lower(key), SUM(value) FROM events GROUP BY lower(key)",
        "SELECT value, SUM(value) FROM events GROUP BY value",
        "SELECT key, SUM(value) FROM events WHERE key = 'key-0' OR value > 0 GROUP BY key",
        "SELECT key, SUM(value) FROM events WHERE key = 'key-0' AND value > 0 GROUP BY key",
    ] {
        if let Some(plan) = plan(&runtime, &input, sql).await {
            assert!(
                prepare(&plan, records, None, 2, "parallel-admission").is_none(),
                "{sql}"
            );
        }
    }
}

#[tokio::test]
async fn test_parallel_sum_distinct_record_schema_heap_is_prepaid() {
    let input = grouped_parallel_tests::source(MIN_ROWS);
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let plan = plan(
        &runtime,
        &input,
        "SELECT key, SUM(value) FROM events GROUP BY key",
    )
    .await
    .unwrap();
    let schema = input
        .table_payload()
        .unwrap()
        .schema()
        .as_ref()
        .clone()
        .with_metadata(std::collections::HashMap::from([(
            "large".into(),
            "schema".repeat(8192),
        )]));
    let record = input.table_payload().unwrap().batches()[0].slice(0, 0);
    let mut retained = None;
    let allocated = allocation_counter::measure(|| {
        retained = Some(record.with_schema(Arc::new(schema.clone())).unwrap());
    });
    let charge = record_charge(&plan, &retained.unwrap(), true, "parallel-schema").unwrap();
    assert!(allocated.bytes_current > 32768);
    assert!(
        u64::try_from(charge).unwrap() >= u64::try_from(allocated.bytes_current).unwrap(),
        "each equal-but-separately-owned schema must remain funded: paid={charge}, retained={allocated:?}"
    );
}

#[tokio::test]
async fn test_parallel_sum_bound_physical_payload_is_paid_independently_of_rows() {
    let input = grouped_parallel_tests::source(MIN_ROWS);
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let mut plan = plan(
        &runtime,
        &input,
        "SELECT key, SUM(value) FROM events GROUP BY key",
    )
    .await
    .unwrap();
    let dimensions = Dimensions {
        rows: 0,
        chunks: 0,
        units: 0,
        nodes: 0,
    };
    let before = scratch_bytes(&plan, &[], &dimensions, true, "parallel-payload").unwrap();
    let name = "source-column".repeat(4096);
    let literal = "literal".repeat(8192);
    let mut expression = None;
    let allocated = allocation_counter::measure(|| {
        expression = Some(Arc::new(BinaryExpr::new(
            Arc::new(Column::new(&name, 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some(literal.clone())))),
        )) as Arc<dyn PhysicalExpr>);
    });
    plan.aggregate_filters.push(expression);
    let dimensions = Dimensions {
        nodes: filter_nodes(&plan).unwrap(),
        ..dimensions
    };
    let after = scratch_bytes(&plan, &[], &dimensions, true, "parallel-payload").unwrap();
    assert!(allocated.bytes_current > 65536);
    assert!(
        u64::try_from(after - before).unwrap() >= u64::try_from(allocated.bytes_current).unwrap(),
        "bound names/literals require their own retained payload charge: delta={}, retained={allocated:?}",
        after - before
    );
}

#[tokio::test]
async fn test_parallel_sum_cold_bounded_pool_falls_back_without_gather_fees() {
    let input = grouped_parallel_tests::source(MIN_ROWS);
    let sql = "SELECT key, SUM(value) FROM events GROUP BY key";
    let unlimited = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let original = plan(&unlimited, &input, sql).await.unwrap();
    let dimensions = Dimensions {
        rows: MIN_ROWS,
        chunks: 4,
        units: 2,
        nodes: 0,
    };
    let optional = scratch_bytes(
        &original,
        input.table_payload().unwrap().batches(),
        &dimensions,
        false,
        "parallel-admission",
    )
    .unwrap();
    let pool = Arc::new(GreedyMemoryPool::new(optional - 1));
    let runtime =
        DataFusionRuntime::with_benchmark_pool(DataFusionConfig::default(), pool.clone()).unwrap();
    let mut plan = plan(&runtime, &input, sql).await.unwrap();
    let job = StreamJobContext::new(
        944,
        "parallel-bounded",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "parallel-bounded", None);
    let transaction = plan
        .update(&input, &context, "parallel-bounded")
        .await
        .unwrap();
    assert_eq!(
        transaction
            .records
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        16
    );
    assert_eq!(job.gather_owner().funding(), (0, 0, 0));
    drop((transaction, plan));
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_parallel_sum_local_peak_and_detached_results_stay_funded() {
    let input = unique_source();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let plan = plan(
        &runtime,
        &input,
        "SELECT key, SUM(value) FROM events GROUP BY key",
    )
    .await
    .unwrap();
    let work = prepare(
        &plan,
        input.table_payload().unwrap().batches(),
        None,
        2,
        "parallel-cost",
    )
    .unwrap();
    let paid = runtime.incremental_memory_pool().reserved();
    let job = StreamJobContext::new(
        945,
        "parallel-cost",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let mut fragment = None;
    let allocations = allocation_counter::measure(|| {
        fragment = Some(work.run(0, &GatherStop::from_job(&job)).unwrap());
    });
    eprintln!("Parallel SUM lane heap: {allocations:?}; prepaid={paid}");
    assert!(allocations.bytes_max > 0);
    assert!(allocations.bytes_max <= u64::try_from(paid).unwrap());
    let fragment = fragment.unwrap();
    let retained = work.scratch.size()
        + fragment
            .states
            .iter()
            .flatten()
            .map(Array::get_array_memory_size)
            .sum::<usize>();
    drop((work, plan, input));
    assert!(runtime.incremental_memory_pool().reserved() >= retained);
    drop(fragment);
    assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
}

#[tokio::test]
async fn test_parallel_sum_variable_literal_evaluation_peak_is_funded() {
    let input = grouped_parallel_tests::source(1);
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let mut plan = plan(
        &runtime,
        &input,
        "SELECT key, SUM(value) FROM events GROUP BY key",
    )
    .await
    .unwrap();
    let expression: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("key", 0)),
        Operator::NotEq,
        Arc::new(Literal::new(ScalarValue::Utf8(Some(
            "literal".repeat(16384),
        )))),
    ));
    plan.aggregate_filters.push(Some(expression.clone()));
    let charge = filter_charge(&plan, 1, "parallel-literal").unwrap();
    let mut output = None;
    let allocations = allocation_counter::measure(|| {
        output = Some(
            boolean(
                expression.as_ref(),
                &input.table_payload().unwrap().batches()[0],
                "parallel-literal",
            )
            .unwrap(),
        );
    });
    assert_eq!(output.unwrap().len(), 1);
    eprintln!("Parallel literal evaluation heap: {allocations:?}; paid={charge}");
    assert!(allocations.bytes_max > 200_000);
    assert!(
        u64::try_from(charge).unwrap() >= allocations.bytes_max,
        "literal scalar copies must be funded per concurrently evaluating unit"
    );
}

#[tokio::test]
async fn test_parallel_sum_keeps_large_plan_schema_and_input_owner_funded() {
    let original = grouped_parallel_tests::source(MIN_ROWS);
    let record = original.table_payload().unwrap().batches()[0].clone();
    let schema = Arc::new(record.schema().as_ref().clone().with_metadata(
        std::collections::HashMap::from([("large".into(), "schema".repeat(32768))]),
    ));
    let input = Batch::table(
        vec![record.with_schema(schema).unwrap()],
        crate::BatchMetadata::default(),
    )
    .unwrap();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let sql = format!(
        "SELECT key, SUM(value) FROM events WHERE key <> '{}' GROUP BY key",
        "literal".repeat(32768)
    );
    let plan = plan(&runtime, &input, &sql).await.unwrap();
    let owner = Arc::new(runtime.incremental_reservation("parallel-input-owner"));
    owner
        .try_grow(input.table_payload().unwrap().batches()[0].get_array_memory_size())
        .unwrap();
    let weak_owner = Arc::downgrade(&owner);
    let work = prepare(
        &plan,
        input.table_payload().unwrap().batches(),
        Some(owner),
        2,
        "parallel-input-owner",
    )
    .unwrap();
    let prepaid = work.scratch.size();
    assert!(
        prepaid >= plan.plan_bytes + schema_charge(&plan.schema, "parallel-input-owner").unwrap()
    );
    drop((plan, input, original));
    assert!(weak_owner.upgrade().is_some());
    assert!(runtime.incremental_memory_pool().reserved() >= prepaid);
    drop(work);
    assert!(weak_owner.upgrade().is_none());
    assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
}

#[tokio::test]
async fn test_parallel_sum_checks_stop_inside_local_key_work() {
    let input = unique_source();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let plan = plan(
        &runtime,
        &input,
        "SELECT key, SUM(value) FROM events GROUP BY key",
    )
    .await
    .unwrap();
    let work = prepare(
        &plan,
        input.table_payload().unwrap().batches(),
        None,
        2,
        "parallel-stop",
    )
    .unwrap();
    let token = CancellationToken::new();
    let job = StreamJobContext::new(946, "parallel-stop", JsonMap::new(), None, token.clone());
    let checks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed = checks.clone();
    let stop = GatherStop::from_job(&job).with_check_hook(Arc::new(move |_| {
        if observed.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 5 {
            token.cancel();
        }
    }));
    assert!(work.run(0, &stop).is_err());
    assert_eq!(checks.load(std::sync::atomic::Ordering::SeqCst), 6);
    drop((work, plan));
    assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
}

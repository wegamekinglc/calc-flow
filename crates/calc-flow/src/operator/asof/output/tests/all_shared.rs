use super::*;
use crate::runtime::streaming::gather_work::TestService;

fn shared_plan() -> (OutputPlan, SchemaRef, std::sync::Weak<dyn Array>) {
    let column: ArrayRef = Arc::new(Int64Array::from(vec![0, 1, 2, 3, 4, 5, 6, 7]));
    let source = Arc::downgrade(&column);
    let schema = Arc::new(Schema::new(vec![
        Field::new("left__value", DataType::Int64, false)
            .with_metadata(HashMap::from([("ownership".into(), "paid-shared".into())])),
    ]));
    let plan = OutputPlan {
        left: OutputSide {
            batches: vec![vec![column]],
            positions: (0..8).map(|row| (0, row)).collect(),
            spans: vec![Span {
                source: 0,
                start: 0,
                end: 8,
            }],
            has_nulls: false,
        },
        right: OutputSide {
            batches: Vec::new(),
            positions: Vec::new(),
            spans: Vec::new(),
            has_nulls: false,
        },
        len: 8,
        matched: 0,
        raw_bytes: 64,
    };
    (plan, schema, source)
}

#[test]
fn all_shared_output_keeps_source_and_credit_without_launching_native_worker() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let observed = runtime.block_on(async {
        let job = crate::StreamJobContext::new(
            91,
            "all-shared",
            crate::JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        )
        .with_gather_owner(service.owner("91".into()));
        let context = StreamOperatorContext::new(&job, "asof", None);
        let mut output = OutputRuntime::new(1_048_576, "asof");
        let pool = output.pool.clone();
        let workspace = MemoryConsumer::new("shared-owned-input-output").register(&pool);
        workspace.try_grow(32_768).unwrap();
        let (plan, schema, source) = shared_plan();
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            output.materialize_plan(plan, &schema, workspace, "asof", &context),
        )
        .await;
        let funding = job.gather_owner().funding();
        let failures = job.gather_owner().close_and_drain().await;
        drop(context);
        drop(job);
        let native_joins = service.joined_workers();
        let capacity = service.available_capacity();
        let (batch, workspace) = result.unwrap().unwrap();
        let record = &batch.table_payload().unwrap().batches()[0];
        let alive = source.upgrade().unwrap();
        let same_array = Arc::ptr_eq(&alive, record.column(0));
        let values = record
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec();
        let schema_exact =
            record.schema() == schema && batch.metadata() == &BatchMetadata::default();
        let credit_live = workspace.size() >= 32_768 && pool.reserved() >= workspace.size();
        drop(alive);
        let source_live_with_output = source.upgrade().is_some();
        drop((batch, workspace));
        let source_released = source.upgrade().is_none();
        let reserved_after_drop = pool.reserved();
        (
            native_joins,
            capacity,
            funding,
            failures.is_empty(),
            same_array,
            values,
            schema_exact,
            credit_live,
            source_live_with_output,
            source_released,
            reserved_after_drop,
        )
    });
    drop(runtime);
    service.shutdown();
    assert!(observed.3);
    assert!(observed.4);
    assert_eq!(observed.5, (0..8).collect::<Vec<i64>>());
    assert!(observed.6 && observed.7 && observed.8 && observed.9);
    assert_eq!(observed.10, 0);
    assert_eq!(
        observed.0, 0,
        "shared-only Arrow output launched and joined a native CPU worker"
    );
    assert_eq!(observed.1, (1, 1, 0));
    assert_eq!(observed.2, (0, 0, 0));
}

#[test]
fn all_shared_output_cancellation_releases_source_credit_without_submission() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let observed = runtime.block_on(async {
        let token = crate::CancellationToken::new();
        let job = crate::StreamJobContext::new(
            92,
            "all-shared-cancel",
            crate::JsonMap::new(),
            None,
            token.clone(),
        )
        .with_gather_owner(service.owner("92".into()));
        let context = StreamOperatorContext::new(&job, "asof", None);
        let mut output = OutputRuntime::new(1_048_576, "asof");
        let pool = output.pool.clone();
        let workspace = MemoryConsumer::new("shared-cancel-input-output").register(&pool);
        workspace.try_grow(32_768).unwrap();
        let (plan, schema, source) = shared_plan();
        token.cancel();
        let result = output
            .materialize_plan(plan, &schema, workspace, "asof", &context)
            .await;
        let failures = job.gather_owner().close_and_drain().await;
        drop(context);
        drop(job);
        (
            matches!(result, Err(crate::CalcFlowError::Cancelled { .. })),
            failures.is_empty(),
            service.joined_workers(),
            source.upgrade().is_none(),
            pool.reserved(),
        )
    });
    drop(runtime);
    service.shutdown();
    assert!(observed.0 && observed.1 && observed.3);
    assert_eq!((observed.2, observed.4), (0, 0));
}

#[test]
fn all_shared_output_budget_rejection_keeps_domain_and_releases_input() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let observed = runtime.block_on(async {
        let job = crate::StreamJobContext::new(
            93,
            "all-shared-budget",
            crate::JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        )
        .with_gather_owner(service.owner("93".into()));
        let context = StreamOperatorContext::new(&job, "asof", None);
        let mut output = OutputRuntime::new(32_768, "asof");
        let pool = output.pool.clone();
        let workspace = MemoryConsumer::new("shared-budget-input-output").register(&pool);
        workspace.try_grow(32_768).unwrap();
        let (plan, schema, source) = shared_plan();
        let result = output
            .materialize_plan(plan, &schema, workspace, "asof", &context)
            .await;
        let failures = job.gather_owner().close_and_drain().await;
        drop(context);
        drop(job);
        (
            matches!(
                result,
                Err(crate::CalcFlowError::OperatorReason {
                    reason_code: crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
                    ..
                })
            ),
            failures.is_empty(),
            service.joined_workers(),
            source.upgrade().is_none(),
            pool.reserved(),
        )
    });
    drop(runtime);
    service.shutdown();
    assert!(observed.0 && observed.1 && observed.3);
    assert_eq!((observed.2, observed.4), (0, 0));
}

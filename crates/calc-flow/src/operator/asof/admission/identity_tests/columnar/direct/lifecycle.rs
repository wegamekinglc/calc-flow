use super::*;
use crate::{EdgeCollector, Epoch, OperatorMetadata, StreamOperator};

fn nullable_payload_input() -> Batch {
    let input = string_unsigned_input(1_024);
    let record = &input.table_payload().unwrap().batches()[0];
    let mut fields = record.schema().fields().iter().cloned().collect::<Vec<_>>();
    fields.push(Arc::new(Field::new("payload", DataType::Utf8, true)));
    let mut columns = record.columns().to_vec();
    let payload = "nullable界".repeat(128);
    columns.push(Arc::new(
        (0..1_024)
            .map(|row| (row % 3 != 0).then_some(payload.as_str()))
            .collect::<StringArray>(),
    ));
    batch(vec![
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap(),
    ])
}

#[test]
fn test_direct_left_nullable_projection_compacts_slices_without_mutation() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let input = nullable_payload_input();
        let original = input.table_payload().unwrap().batches()[0].clone();
        let owners = original
            .columns()
            .iter()
            .map(Arc::strong_count)
            .collect::<Vec<_>>();
        for projection in [None, Some(vec![3]), Some(Vec::new())] {
            let mut operator = fixture(&original.schema());
            if let Some(columns) = projection {
                operator.set_output_projection(columns).unwrap();
            }
            compare_preparation(&mut operator, &input, None, true, &service).await;
            let sliced = batch(vec![original.slice(17, 3)]);
            compare_preparation(&mut operator, &sliced, None, true, &service).await;
        }
        assert_eq!(input.table_payload().unwrap().batches()[0], original);
        assert_eq!(
            original
                .columns()
                .iter()
                .map(Arc::strong_count)
                .collect::<Vec<_>>(),
            owners
        );
    });
    drop(runtime);
    service.shutdown();
}

async fn snapshot_input(
    operator: &mut StreamAsofJoinOperator,
    input: &Batch,
    context: &StreamOperatorContext<'_>,
    legacy: bool,
) -> crate::OperatorStateSnapshot {
    let guard = LegacyGuard::new(legacy);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    for port in ["right", "left"] {
        operator
            .process_data(port, input.clone(), context, &mut output)
            .await
            .unwrap();
    }
    operator.prepare_checkpoint_async(context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert!(output.drain("output").is_empty());
    drop(guard);
    snapshot
}

async fn continuation(service: &TestService) {
    let input = string_unsigned_input(4_097);
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut source = fixture(&schema);
    let mut legacy = fixture(&schema);
    let mut restored = fixture(&schema);
    let pools = [&source, &legacy, &restored].map(|operator| operator.runtime.pool.clone());
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(service.owner("direct-left-recovery".into()));
    let context = StreamOperatorContext::new(&job, "asof", None);
    let snapshot = snapshot_input(&mut source, &input, &context, false).await;
    let old = snapshot_input(&mut legacy, &input, &context, true).await;
    assert_eq!(snapshot.inline_metadata, old.inline_metadata);
    assert_eq!(snapshot.segments, old.segments);
    restored.restore(&snapshot).unwrap();
    assert_eq!(source.status, restored.status);
    assert_eq!(source.status, legacy.status);
    let appended = batch(vec![unsigned_record(
        &schema,
        vec!["S000"],
        vec![9_000],
        vec![64],
    )]);
    for candidate in [&mut source, &mut restored, &mut legacy] {
        let mut output = EdgeCollector::new(candidate.output_ports().to_vec());
        candidate
            .process_data("left", appended.clone(), &context, &mut output)
            .await
            .unwrap();
        candidate.on_end(&context, &mut output).await.unwrap();
        let pairs = output
            .drain("output")
            .into_iter()
            .flat_map(|message| {
                message
                    .as_data()
                    .unwrap()
                    .table_payload()
                    .unwrap()
                    .batches()
                    .iter()
                    .flat_map(|record| {
                        let left = record
                            .column(2)
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .unwrap();
                        let right = record
                            .column(5)
                            .as_any()
                            .downcast_ref::<UInt64Array>()
                            .unwrap();
                        assert_eq!(right.null_count(), 0);
                        left.values()
                            .iter()
                            .copied()
                            .zip(right.values().iter().copied())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            pairs,
            (0..4_097)
                .map(|row| (row, row))
                .chain([(9_000, 4_096)])
                .collect::<Vec<_>>()
        );
        assert_eq!(candidate.status.left.accepted_rows, 4_098);
        assert_eq!(candidate.status.emitted_left_rows, 4_098);
    }
    assert_eq!(source.status, restored.status);
    assert_eq!(source.status, legacy.status);
    drop((source, legacy, restored, snapshot, old));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
}

#[test]
fn test_direct_left_checkpoint_matches_legacy_and_continues_after_restore() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(continuation(&service));
    drop(runtime);
    service.shutdown();
}

async fn competing_limits(
    legacy: bool,
    duplicate: bool,
    tight: bool,
) -> (StreamingFailureReason, StreamAsofJoinStatus) {
    let input = string_unsigned_input(1);
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let template = fixture(&schema);
    let spec = StreamAsofJoinSpec::new(
        template.spec.left().clone(),
        template.spec.right().clone(),
        std::time::Duration::ZERO,
        crate::AsofStateLimits::new(1, 128 << 20).unwrap(),
    )
    .unwrap();
    let mut operator =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    let pool = operator.runtime.pool.clone();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", input.clone(), &context, &mut output)
        .await
        .unwrap();
    let hold = tight.then(|| {
        operator
            .reserve_workspace((128 << 20) - pool.reserved() as u64)
            .unwrap()
    });
    let next = if duplicate {
        input
    } else {
        batch(vec![unsigned_record(
            &schema,
            vec!["S000"],
            vec![1],
            vec![1],
        )])
    };
    let guard = LegacyGuard::new(legacy);
    let error = operator
        .process_data("left", next, &context, &mut output)
        .await
        .unwrap_err();
    drop(guard);
    let crate::CalcFlowError::OperatorReason { reason_code, .. } = error else {
        panic!("unexpected error: {error}")
    };
    assert_eq!(operator.status.state_rows, 1);
    assert_eq!(operator.status.left.accepted_rows, 1);
    let status = operator.status.clone();
    drop((operator, output, hold));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    assert_eq!(pool.reserved(), 0);
    (reason_code, status)
}

#[tokio::test(flavor = "current_thread")]
async fn test_direct_left_preserves_duplicate_workspace_and_state_error_priority() {
    for duplicate in [false, true] {
        for tight in [false, true] {
            let old = competing_limits(true, duplicate, tight).await;
            let fast = competing_limits(false, duplicate, tight).await;
            assert_eq!(fast, old);
            assert_eq!(
                fast.0,
                if duplicate {
                    StreamingFailureReason::AsofDuplicateIdentity
                } else if tight {
                    StreamingFailureReason::AsofWorkspaceLimitExceeded
                } else {
                    StreamingFailureReason::AsofStateLimitExceeded
                }
            );
        }
    }
}

use super::*;

mod lifecycle;

thread_local! {
    static CANONICAL_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(in super::super::super) fn record_canonical_rows(rows: usize) {
    CANONICAL_ROWS.set(CANONICAL_ROWS.get() + rows);
}

pub(in super::super::super::super) fn string_unsigned_input(count: usize) -> Batch {
    let schema = schema(DataType::Utf8, DataType::UInt64);
    batch(
        (0..count)
            .step_by(2_048)
            .map(|start| {
                let end = (start + 2_048).min(count);
                record(
                    &schema,
                    Arc::new(StringArray::from_iter_values(
                        (start..end).map(|row| format!("S{:03}", row % 64)),
                    )),
                    Arc::new(UInt64Array::from_iter_values(
                        (start..end).map(|row| row as u64),
                    )),
                    (start..end)
                        .map(|row| i64::try_from(row / 64).unwrap())
                        .collect(),
                )
            })
            .collect(),
    )
}

#[test]
fn test_direct_left_admission_avoids_per_row_canonical_descriptors() {
    use crate::{OperatorMetadata, StreamOperator};

    let input = string_unsigned_input(4_097);
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let canonical = runtime.block_on(async {
        let mut operator = fixture(&schema);
        let pool = operator.runtime.pool.clone();
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
            .with_gather_owner(service.owner("direct-left".into()));
        let context = StreamOperatorContext::new(&job, "asof", None);
        let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
        CANONICAL_ROWS.set(0);
        operator
            .process_data("left", input.clone(), &context, &mut output)
            .await
            .unwrap();
        let canonical = CANONICAL_ROWS.replace(0);
        assert_eq!(operator.status.left.accepted_rows, 4_097);
        assert_eq!(operator.status.state_rows, 4_097);
        assert_eq!(operator.status.pending_left_rows, 4_097);
        assert_eq!(operator.state.left.len(), 4_097);
        let first = &input.table_payload().unwrap().batches()[0];
        let keys = (0..64)
            .map(|row| state::encoded_columns(first, row, operator.spec.left().keys()).unwrap())
            .collect::<Vec<_>>();
        for (position, ((time, key, sequence), _)) in operator.state.left.iter().enumerate() {
            assert_eq!(*time, i64::try_from(position / 64).unwrap());
            assert_eq!(key, &keys[position % 64]);
            let mut encoded = [1; 9];
            encoded[1..].copy_from_slice(&(position as u64).to_be_bytes());
            assert_eq!(sequence.as_ref().as_slice(), encoded);
        }
        drop((operator, output));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
        canonical
    });
    drop(runtime);
    service.shutdown();
    assert_eq!(
        canonical, 0,
        "eligible input built canonical row descriptors"
    );
}

async fn compare_preparation(
    operator: &mut StreamAsofJoinOperator,
    input: &Batch,
    watermark: Option<i64>,
    direct: bool,
    service: &TestService,
) {
    let pool = operator.runtime.pool.clone();
    let reserved = pool.reserved();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(service.owner("direct-prepare".into()));
    let context = StreamOperatorContext::new(&job, "asof", None);
    let validated = ValidatedInput {
        index: 0,
        watermark,
    };
    let (mut legacy, _) = prepared(operator, input, validated, &context, true).await;
    let (mut fast, _) = prepared(operator, input, validated, &context, false).await;
    assert_eq!(fast.len(), legacy.len());
    assert_eq!(fast.accepted, legacy.accepted);
    assert_eq!(fast.rows.is_empty(), direct || legacy.rows.is_empty());
    if !direct {
        let identities = |admission: &Admission| {
            admission
                .rows
                .iter()
                .map(|(order, row)| (order.clone(), row.batch_index, row.row, row.key_index))
                .collect::<Vec<_>>()
        };
        assert_eq!(identities(&fast), identities(&legacy));
    }
    let fast_chunks = fast.left_chunks.take().unwrap();
    let legacy_chunks = legacy.left_chunks.take().unwrap();
    assert_eq!(fast_chunks.len(), legacy_chunks.len());
    for (fast, legacy) in fast_chunks.into_iter().zip(legacy_chunks) {
        assert_chunks_equal(fast, legacy);
    }
    drop((fast, legacy));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), reserved);
}

fn unsigned_record(
    schema: &SchemaRef,
    keys: Vec<&str>,
    sequences: Vec<u64>,
    times: Vec<i64>,
) -> RecordBatch {
    record(
        schema,
        Arc::new(StringArray::from(keys)),
        Arc::new(UInt64Array::from(sequences)),
        times,
    )
}

#[test]
fn test_direct_left_dictionary_boundaries_and_fallback_match_legacy() {
    let schema = schema(DataType::Utf8, DataType::UInt64);
    let empty = RecordBatch::new_empty(schema.clone());
    let first = unsigned_record(
        &schema,
        vec!["", "\0", "a", "a\0", "é"],
        vec![0; 5],
        vec![10; 5],
    );
    let last = unsigned_record(
        &schema,
        vec!["é"; 3],
        vec![1, u64::MAX - 1, u64::MAX],
        vec![10; 3],
    );
    let cases = vec![
        (string_unsigned_input(4_097), None, true),
        (
            batch(vec![empty.clone(), first, empty.clone(), last]),
            None,
            true,
        ),
        (batch(vec![empty]), None, false),
        (
            batch(vec![unsigned_record(
                &schema,
                vec!["long-key-outside-inline"; 3],
                vec![0, 1, 2],
                vec![1, 2, 3],
            )]),
            None,
            false,
        ),
        (
            batch(vec![unsigned_record(
                &schema,
                vec!["k"; 3],
                vec![0, 1, 2],
                vec![3, 1, 2],
            )]),
            None,
            false,
        ),
        (
            batch(vec![unsigned_record(
                &schema,
                vec!["k"; 3],
                vec![0, 1, 2],
                vec![9, 10, 11],
            )]),
            Some(10),
            false,
        ),
    ];
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    for (input, watermark, direct) in cases {
        runtime.block_on(compare_preparation(
            &mut fixture(&schema),
            &input,
            watermark,
            direct,
            &service,
        ));
    }
    drop(runtime);
    service.shutdown();
}

async fn restored_boundary(service: &TestService) {
    use crate::StreamOperator;
    let schema = schema(DataType::Utf8, DataType::UInt64);
    let mut operator = fixture(&schema);
    let mut restored = fixture(&schema);
    let pools = [operator.runtime.pool.clone(), restored.runtime.pool.clone()];
    let seed = batch(vec![unsigned_record(
        &schema,
        vec!["k"; 2],
        vec![0, 1],
        vec![100, 0],
    )]);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(service.owner("direct-restore".into()));
    let context = StreamOperatorContext::new(&job, "asof", None);
    admit(&mut operator, &seed, &context, false).await;
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(crate::Epoch::INITIAL).unwrap();
    restored.restore(&snapshot).unwrap();
    for candidate in [&mut operator, &mut restored] {
        assert_eq!(candidate.status.left.accepted_rows, 2);
        for (time, sequence, direct) in [(50, 2, false), (100, 2, true)] {
            let input = batch(vec![unsigned_record(
                &schema,
                vec!["k"],
                vec![sequence],
                vec![time],
            )]);
            compare_preparation(candidate, &input, None, direct, service).await;
        }
        let duplicate = batch(vec![unsigned_record(
            &schema,
            vec!["k"],
            vec![0],
            vec![100],
        )]);
        let error = candidate
            .prepare_admission(
                ValidatedInput {
                    index: 0,
                    watermark: None,
                },
                &duplicate,
                &context,
            )
            .await
            .err()
            .unwrap();
        assert!(matches!(
            error,
            crate::CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofDuplicateIdentity,
                ..
            }
        ));
        assert_eq!(candidate.status.state_rows, 2);
        assert_eq!(candidate.status.left.duplicate_rows, 1);
    }
    drop((operator, restored, snapshot));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
}

#[test]
fn test_direct_left_uses_actual_restored_maximum() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(restored_boundary(&service));
    drop(runtime);
    service.shutdown();
}

#[test]
fn test_direct_left_accounts_smaller_worker_credit_and_preserves_limits() {
    let input = string_unsigned_input(4_097);
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let operator = fixture(&schema);
    let validated = ValidatedInput {
        index: 0,
        watermark: None,
    };
    let identity = operator.identity_workspace(&input, validated).unwrap();
    let payload = operator.input_workspace(&input, validated).unwrap();
    let descriptor = operator.reserve_left_work(3).unwrap();
    let thresholds = [
        identity.reservation.size(),
        identity.reservation.size() + payload.size(),
        identity.reservation.size() + payload.size() + descriptor.size(),
    ];
    drop((identity, payload, descriptor, operator));
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let baseline = runtime.block_on(budget_case(&input, 128 << 20, true, &service));
    let fast_baseline = runtime.block_on(budget_case(&input, 128 << 20, false, &service));
    let delta = size_of::<LeftChunkWork>()
        - size_of::<crate::operator::asof::admission::direct::ChunkWork>();
    assert_eq!(baseline.0, None);
    assert_eq!(fast_baseline.0, None);
    assert_eq!(baseline.1, fast_baseline.1);
    assert_eq!(baseline.2, fast_baseline.2);
    let old_credit = fixture(&schema)
        .cpu_work_credit(size_of::<LeftChunkWork>())
        .unwrap();
    let direct_credit = fixture(&schema)
        .cpu_work_credit(size_of::<crate::operator::asof::admission::direct::ChunkWork>())
        .unwrap();
    assert_eq!(old_credit.size() - direct_credit.size(), delta);
    drop((old_credit, direct_credit));
    let old_fee = runtime.block_on(worker_admission_fee(&input, true, &service));
    let direct_fee = runtime.block_on(worker_admission_fee(&input, false, &service));
    assert_eq!(old_fee - direct_fee, delta);
    for limit in thresholds.into_iter().flat_map(|limit| [limit - 1, limit]) {
        let old = runtime.block_on(budget_case(&input, limit, true, &service));
        let direct = runtime.block_on(budget_case(&input, limit, false, &service));
        assert_eq!(
            (&direct.0, &direct.1),
            (&old.0, &old.1),
            "pool limit={limit}"
        );
        assert_eq!(direct.2, old.2, "pre-worker pool limit={limit}");
    }
    for limit in [baseline.2 - 1, baseline.2] {
        let old = runtime.block_on(budget_case(&input, limit, true, &service));
        let direct = runtime.block_on(budget_case(&input, limit, false, &service));
        assert_eq!((&direct.0, &direct.1), (&old.0, &old.1));
        assert_eq!(
            old.0,
            (limit < baseline.2).then_some(StreamingFailureReason::AsofWorkspaceLimitExceeded)
        );
        // Worker admission and its ASOF wrapper each fund the smaller work
        // type. Successful state installation has a later, unchanged peak.
        assert_eq!(
            old.2 - direct.2,
            if old.0.is_some() { 2 * delta } else { 0 }
        );
    }
    drop(runtime);
    service.shutdown();
}

async fn worker_admission_fee(input: &Batch, legacy: bool, service: &TestService) -> usize {
    use crate::runtime::streaming::gather_work::admission_probe::{AdmissionProbe, AdmissionStage};
    use crate::{OperatorMetadata, StreamOperator};

    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut operator = fixture(&schema);
    let pool = Arc::new(ObservedPool::new(128 << 20));
    operator.runtime.pool = pool.clone();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(service.owner("direct-worker-fee".into()));
    let context = StreamOperatorContext::new(&job, "asof", None);
    let probe = AdmissionProbe::install(
        job.gather_owner(),
        AdmissionStage::Attempt,
        pool.clone(),
        128 << 20,
    );
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let guard = LegacyGuard::new(legacy);
    let error = operator
        .process_data("left", input.clone(), &context, &mut output)
        .await
        .unwrap_err();
    drop(guard);
    assert!(matches!(
        error,
        crate::CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        }
    ));
    let event = probe.take_event().expect("actual owned work admission");
    assert_eq!(event.available + 1, event.fee);
    assert_eq!(operator.status.state_rows, 0);
    assert_eq!(operator.status.left.accepted_rows, 0);
    drop((operator, output, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
    event.fee
}

fn dictionary_plan_inputs() -> Vec<Batch> {
    let schema = schema(DataType::Utf8, DataType::UInt64);
    let keys = (0..512).map(|row| format!("K{row:04}")).collect::<Vec<_>>();
    let unique = record(
        &schema,
        Arc::new(StringArray::from_iter_values(&keys)),
        Arc::new(UInt64Array::from_iter_values(0..512)),
        vec![10; 512],
    );
    let many = (0..65)
        .flat_map(|row| {
            [
                RecordBatch::new_empty(schema.clone()),
                unsigned_record(&schema, vec!["k"], vec![row], vec![10]),
            ]
        })
        .collect();
    vec![batch(vec![unique]), batch(many)]
}

#[test]
fn test_direct_left_key_plans_fit_existing_identity_credit() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    for input in dictionary_plan_inputs() {
        let schema = input.table_payload().unwrap().batches()[0].schema();
        let mut operator = fixture(&schema);
        let pool = Arc::new(ObservedPool::new(128 << 20));
        operator.runtime.pool = pool.clone();
        let validated = ValidatedInput {
            index: 0,
            watermark: None,
        };
        let identity = operator.identity_workspace(&input, validated).unwrap();
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "asof", None);
        let mut plan = None;
        let allocation = allocation_counter::measure(|| {
            plan = crate::operator::asof::admission::direct::plan(
                &operator, &input, validated, &context,
            )
            .unwrap();
        });
        assert!(plan.is_some());
        assert!(
            u64::try_from(allocation.bytes_current).unwrap() <= identity.reservation.size() as u64
        );
        assert!(allocation.bytes_max <= pool.peak.load(Ordering::SeqCst) as u64);
        drop((plan, identity, context));
        drop(job);
        runtime.block_on(compare_preparation(
            &mut operator,
            &input,
            None,
            true,
            &service,
        ));
        drop(operator);
        assert_eq!(pool.reserved(), 0);
    }
    drop(runtime);
    service.shutdown();
}

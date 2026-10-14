use super::*;
use datafusion::execution::memory_pool::{
    MemoryLimit, MemoryPool, MemoryReservation, UnboundedMemoryPool,
};
use std::{
    fmt,
    sync::atomic::{AtomicU8, AtomicUsize, Ordering},
};

#[derive(Debug, Default)]
pub(super) struct DictionaryPool {
    inner: UnboundedMemoryPool,
    pub(super) mode: AtomicU8,
    pub(super) refusals: AtomicUsize,
}

impl fmt::Display for DictionaryPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("dictionary-control-pool")
    }
}

impl MemoryPool for DictionaryPool {
    fn name(&self) -> &'static str {
        "dictionary-control-pool"
    }
    fn grow(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.grow(credit, bytes);
    }
    fn shrink(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.shrink(credit, bytes);
    }
    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
    fn try_grow(&self, credit: &MemoryReservation, bytes: usize) -> datafusion::common::Result<()> {
        let mode = self.mode.load(Ordering::SeqCst);
        let selected = match mode {
            1 | 2 => credit
                .consumer()
                .name()
                .ends_with(":stream-join-dictionary"),
            3 | 4 => credit.consumer().name().ends_with(":stream-join-native"),
            _ => false,
        };
        if selected && bytes > 0 {
            self.refusals.fetch_add(1, Ordering::SeqCst);
            return Err(if mode == 1 || mode == 3 {
                datafusion::common::DataFusionError::ResourcesExhausted(
                    "dictionary budget refused".into(),
                )
            } else {
                datafusion::common::DataFusionError::Execution(
                    "dictionary non-budget failure".into(),
                )
            });
        }
        self.inner.try_grow(credit, bytes)
    }
}

fn fixture() -> (StreamJoinOperator, Arc<DictionaryPool>) {
    fixture_spec(spec())
}

pub(super) fn fixture_spec(
    declaration: StreamJoinSpec,
) -> (StreamJoinOperator, Arc<DictionaryPool>) {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap();
    let pool = Arc::new(DictionaryPool::default());
    operator
        .runtime
        .initialize_benchmark_pool(pool.clone())
        .unwrap();
    (operator, pool)
}

fn keyed_batch(left: bool, key: i64) -> Batch {
    keyed_values(left, vec![key])
}

fn keyed_values(left: bool, keys: Vec<i64>) -> Batch {
    let original = if left {
        left_batch(vec![0; keys.len()])
    } else {
        right_batch(vec![0; keys.len()])
    };
    let record = &original.table_payload().unwrap().batches()[0];
    let mut columns = record.columns().to_vec();
    columns[0] = Arc::new(Int64Array::from(keys));
    Batch::table(
        vec![RecordBatch::try_new(record.schema(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

#[tokio::test]
async fn test_dictionary_refusal_mixed_legacy_state_matches_until_explicit_rebuild() {
    let (mut operator, pool) = fixture();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", keyed_batch(false, 7), &context, &mut output)
        .await
        .unwrap();
    let known = Arc::clone(&operator.state.right[0].encoded_key);
    pool.mode.store(1, Ordering::SeqCst);
    operator
        .process_data("left", keyed_batch(true, 8), &context, &mut output)
        .await
        .unwrap();
    pool.mode.store(0, Ordering::SeqCst);
    reset_join_work();
    operator
        .process_data("right", keyed_batch(false, 8), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(
        operator.status().emitted_match_rows,
        1,
        "third production operation must match the legacy new key"
    );
    assert_eq!(
        (operator.state.left.len(), operator.state.right.len()),
        (1, 2)
    );
    assert!(Arc::ptr_eq(&known, &operator.state.right[0].encoded_key));
    assert_eq!(
        pool.refusals.load(Ordering::SeqCst),
        1,
        "needs-rebuild is not a fresh pool failure"
    );
    assert!(join_work().sql_probe_table_builds > 0);
    assert_eq!(
        operator.key_dictionary.inventory().1,
        1,
        "incomplete ordinary batches must not birth an identity that omits old equal legacy rows"
    );
    let progress = progress_context(
        &job,
        (IngressState::Active, Some(0)),
        (IngressState::Active, None),
    );
    operator
        .on_ingress_progress("left", &progress)
        .await
        .unwrap();
    assert!(
        Arc::ptr_eq(
            operator
                .key_bindings
                .left
                .canonical(&operator.state.left[0]),
            operator
                .key_bindings
                .right
                .canonical(&operator.state.right[1])
        ),
        "explicit funded progress rebuild must coalesce all live legacy keys"
    );
    assert!(
        Arc::ptr_eq(&known, &operator.state.right[0].encoded_key),
        "rebuild must preserve active known identities"
    );
    assert_eq!(operator.key_dictionary.inventory().1, 3);
    reset_join_work();
    operator
        .process_data("left", keyed_left_batch(&[8], &[10]), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.status().emitted_match_rows, 2);
    assert_eq!(join_work().key_encodings, 0);
    assert_eq!(join_work().sql_probe_table_builds, 0);
    assert_rebuilt_known_append_and_expiry(&mut operator, &pool, &context, &mut output).await;
    drop((operator, output, context, known, progress));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

async fn assert_rebuilt_known_append_and_expiry(
    operator: &mut StreamJoinOperator,
    pool: &DictionaryPool,
    context: &StreamOperatorContext<'_>,
    output: &mut EdgeCollector,
) {
    let bindings_paid = operator.key_bindings.left.bytes() + operator.key_bindings.right.bytes();
    assert!(bindings_paid > 0);
    let old_backing = Arc::as_ptr(&operator.state.left.0);
    pool.mode.store(1, Ordering::SeqCst);
    operator
        .process_data("left", keyed_batch(true, 7), context, output)
        .await
        .unwrap();
    assert_eq!(operator.status().emitted_match_rows, 3);
    assert_eq!(operator.key_dictionary.inventory().1, 5);
    assert_eq!(
        Arc::as_ptr(&operator.state.left.0),
        old_backing,
        "ordinary append preserves the existing legacy row owner"
    );
    assert_eq!(
        operator.key_bindings.left.bytes() + operator.key_bindings.right.bytes(),
        bindings_paid,
        "known identified append does not grow the legacy binding view"
    );
    operator
        .evict_progress(
            "right",
            IngressProgress::new(
                IngressState::Active,
                Some(EventTime::from_micros(60_000_001)),
            ),
            true,
        )
        .unwrap();
    assert_eq!(operator.state.left.len(), 1);
    assert_eq!(
        operator.state.left[0].event_time,
        EventTime::from_micros(10)
    );
    assert_eq!(operator.key_dictionary.inventory().1, 3);
    assert_eq!(operator.key_bindings.left.bytes(), 0);
    assert!(operator.key_bindings.right.bytes() > 0);
}

#[tokio::test]
async fn test_dictionary_refusal_preserves_known_canonical_owner_and_references() {
    let (mut operator, pool) = fixture();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    let canonical = Arc::clone(&operator.state.right[0].encoded_key);
    pool.mode.store(1, Ordering::SeqCst);
    operator
        .process_data("left", left_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.status().emitted_match_rows, 1);
    assert!(pool.refusals.load(Ordering::SeqCst) > 0);
    assert!(
        Arc::ptr_eq(&canonical, &operator.state.left[0].encoded_key),
        "genuine optional refusal must preserve known canonical owners"
    );
    assert_eq!(operator.key_dictionary.inventory().1, 2);
}

#[tokio::test]
async fn test_dictionary_nonbudget_error_propagates_without_sql_retention() {
    let (mut operator, pool) = fixture();
    pool.mode.store(2, Ordering::SeqCst);
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let before = operator.status();
    let error = operator
        .process_data("left", left_batch(vec![0]), &context, &mut output)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("dictionary non-budget failure"));
    assert_eq!(operator.status(), before);
    assert!(output.drain("output").is_empty());
    assert_eq!(pool.refusals.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_dictionary_refusal_commits_batched_sql_retention_with_full_fees() {
    let (mut operator, pool) = fixture();
    pool.mode.store(1, Ordering::SeqCst);
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let left = left_batch(vec![0; 2]);
    let right = right_batch(vec![0]);
    let fees = [(&left, 0), (&left, 1), (&right, 0)].map(|(batch, row)| {
        state_row_charge(
            &batch.table_payload().unwrap().batches()[0],
            row,
            &[0],
            "match",
        )
        .unwrap()
    });
    reset_join_work();
    operator
        .process_data("left", left, &context, &mut output)
        .await
        .unwrap();
    operator
        .process_data("right", right, &context, &mut output)
        .await
        .unwrap();
    assert_eq!(
        (operator.state.left.len(), operator.state.right.len()),
        (2, 1)
    );
    assert_eq!(
        operator
            .state
            .left
            .iter()
            .map(|row| row.charge)
            .sum::<u64>(),
        fees[0] + fees[1]
    );
    assert_eq!(operator.state.right[0].charge, fees[2]);
    assert_eq!(operator.status().emitted_match_rows, 2);
    assert_eq!(join_work().key_encodings, 3);
    assert!(join_work().sql_probe_table_builds > 0);
    assert_eq!(
        pool.refusals.load(Ordering::SeqCst),
        1,
        "the first real refusal leaves a persistent needs-rebuild route, not a second fake failure"
    );
    drop((operator, output, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_native_budget_refusal_reuses_prepared_keys_through_sql_commit() {
    let (mut operator, pool) = fixture();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    let known = Arc::clone(&operator.state.right[0].encoded_key);
    pool.mode.store(3, Ordering::SeqCst);
    let input = keyed_values(true, vec![7, 8, 7]);
    let record = &input.table_payload().unwrap().batches()[0];
    let charges = (0..3)
        .map(|row| state_row_charge(record, row, &[0], "match").unwrap())
        .collect::<Vec<_>>();
    reset_join_work();
    operator
        .process_data("left", input, &context, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.status().emitted_match_rows, 2);
    assert_eq!(
        (operator.state.left.len(), operator.state.right.len()),
        (3, 1)
    );
    assert_eq!(
        operator
            .state
            .left
            .iter()
            .map(|row| row.charge)
            .collect::<Vec<_>>(),
        charges
    );
    assert!(Arc::ptr_eq(&known, &operator.state.left[0].encoded_key));
    assert!(Arc::ptr_eq(&known, &operator.state.left[2].encoded_key));
    assert!(pool.refusals.load(Ordering::SeqCst) > 0);
    assert!(join_work().sql_probe_table_builds > 0);
    assert_eq!(
        join_work().key_encodings,
        1,
        "native budget refusal must retain the already prepared distinct canonical keys"
    );
    assert_eq!(
        (
            operator.key_dictionary.inventory().0,
            operator.key_dictionary.inventory().1
        ),
        (2, 4)
    );
    drop((known, operator, output, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_native_nonbudget_error_propagates_without_sql_commit() {
    let (mut operator, pool) = fixture();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    pool.mode.store(4, Ordering::SeqCst);
    reset_join_work();
    let error = operator
        .process_data("left", left_batch(vec![0]), &context, &mut output)
        .await
        .unwrap_err();
    assert!(matches!(error, CalcFlowError::DataFusion { .. }));
    assert!(operator.state.left.is_empty());
    assert_eq!(operator.key_dictionary.inventory().1, 1);
    assert_eq!(join_work().sql_probe_table_builds, 0);
    assert!(output.drain("output").is_empty());
    drop((operator, output, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_dictionary_refusal_known_references_do_not_scan_unrelated_entries() {
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(1024, 16 << 20, 4096).unwrap();
    let (mut operator, pool) = fixture_spec(declaration);
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "right",
            keyed_values(false, (0..128).collect()),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    pool.mode.store(1, Ordering::SeqCst);
    reset_join_work();
    operator
        .process_data(
            "left",
            keyed_values(true, vec![0, 1, 0]),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    assert_eq!(operator.status().emitted_match_rows, 3);
    assert_eq!(operator.key_dictionary.inventory().1, 131);
    assert!(
        (3..=6).contains(&join_work().dictionary_reference_visits),
        "known-ref preflight and touched cleanup visit only batch hits: {}",
        join_work().dictionary_reference_visits
    );
    drop((operator, output, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

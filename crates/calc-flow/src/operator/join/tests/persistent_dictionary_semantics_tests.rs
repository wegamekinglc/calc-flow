use super::*;
use datafusion::arrow::{array::ListArray, datatypes::Int64Type};
use datafusion::execution::memory_pool::{
    MemoryLimit, MemoryPool, MemoryReservation, UnboundedMemoryPool,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[tokio::test]
async fn test_persistent_dictionary_composite_sparse_slices_keep_nested_fees_and_order() {
    let pairs = key_pairs();
    let schema = nested_schema();
    let declaration = StreamJoinSpec::inner(
        ["a", "b"],
        ["a", "b"],
        "time",
        "time",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(100, 1_000_000, 100).unwrap(),
    )
    .unwrap();
    let mut operator = StreamJoinOperator::new(
        "match",
        Arc::clone(&schema),
        Arc::clone(&schema),
        declaration,
    )
    .unwrap();
    operator.set_output_projection(vec![3, 8]).unwrap();
    operator.key_dictionary.force_collisions();
    let job = job();
    let context = progress_context(
        &job,
        (IngressState::Active, Some(0)),
        (IngressState::Active, None),
    );
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let dense = dense_record(&schema, &pairs, 0);
    reset_join_work();
    operator
        .process_data(
            "right",
            table(vec![dense.slice(0, 4), dense.slice(4, 4)]),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    assert_eq!(join_work().key_encodings, pairs.len());
    let repeated = dense_record(&schema, &pairs[1..2], 8);
    operator
        .process_data("right", table(vec![repeated]), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(
        join_work().key_encodings,
        pairs.len(),
        "known cross-batch composite keys do not frame again"
    );
    let records = sparse_records(&schema, &pairs);
    let omitted = records
        .iter()
        .map(|record| Arc::downgrade(record.column(4)))
        .collect::<Vec<_>>();
    assert_eq!(records[0].num_rows(), 1024);
    assert_eq!(
        records[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value_offsets()[0],
        6,
        "sliced key column retains a nonzero underlying value offset"
    );
    reset_join_work();
    operator
        .process_data("left", table(records.clone()), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(join_work().key_encodings, 0);
    assert_eq!(join_work().sql_probe_table_builds, 0);
    assert_state(&operator, &pairs);
    assert_nested_output(&mut output);
    assert_eq!(
        records[0]
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(1023),
        "removed"
    );
    drop(records);
    assert!(
        omitted.iter().all(|owner| owner.upgrade().is_none()),
        "omitted caller columns must not escape through identity owners or projected state"
    );
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((operator, output, context, dense));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

fn key_pairs() -> Vec<(String, String)> {
    [
        ("", ""),
        ("x", "suffix"),
        ("12345678", "s"),
        ("123456789", "s"),
        ("é\0", "s"),
        ("a", "bc"),
        ("ab", "c"),
    ]
    .into_iter()
    .map(|(left, right)| (left.to_owned(), right.to_owned()))
    .chain(std::iter::once(("z".repeat(16384), "s".into())))
    .collect()
}

fn nested_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("a", DataType::Utf8, true),
        Field::new("b", DataType::Utf8, true),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
        Field::new(
            "payload",
            ListArray::from_iter_primitive::<Int64Type, _, _>([Some(vec![Some(0)])])
                .data_type()
                .clone(),
            true,
        ),
        Field::new("unused", DataType::Utf8, false),
    ]))
}

type InputRow<'a> = (Option<&'a str>, Option<&'a str>, Option<i64>, i64);

fn record(schema: &SchemaRef, rows: &[InputRow<'_>]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            Arc::new(StringArray::from(
                rows.iter().map(|row| row.0).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter().map(|row| row.1).collect::<Vec<_>>(),
            )),
            Arc::new(
                TimestampMicrosecondArray::from(rows.iter().map(|row| row.2).collect::<Vec<_>>())
                    .with_timezone("UTC"),
            ),
            Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(
                rows.iter().map(|row| Some(vec![Some(row.3)])),
            )),
            Arc::new(StringArray::from(vec!["removed"; rows.len()])),
        ],
    )
    .unwrap()
}

fn dense_record(schema: &SchemaRef, pairs: &[(String, String)], first: i64) -> RecordBatch {
    record(
        schema,
        &pairs
            .iter()
            .enumerate()
            .map(|(index, (a, b))| {
                (
                    Some(a.as_str()),
                    Some(b.as_str()),
                    Some(0),
                    first + i64::try_from(index).unwrap(),
                )
            })
            .collect::<Vec<_>>(),
    )
}

fn sparse_records(schema: &SchemaRef, pairs: &[(String, String)]) -> Vec<RecordBatch> {
    let mut first = vec![(None, Some(""), Some(0), -1); 1025];
    first[0] = (Some("prefix"), Some(""), Some(0), -1);
    first[1024] = (Some(""), Some(""), Some(0), 1023);
    vec![
        record(schema, &first).slice(1, 1024),
        record(
            schema,
            &[
                (None, Some(""), None, 1024),
                (None, Some(""), Some(0), 1025),
                (Some("late"), Some(""), Some(-1), 1026),
            ],
        ),
        dense_record(schema, &pairs[1..], 1027),
    ]
}

fn table(records: Vec<RecordBatch>) -> Batch {
    Batch::table(records, BatchMetadata::default()).unwrap()
}

fn expected_frame(a: &str, b: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in [a, b] {
        bytes.push(10);
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(value.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
    bytes
}

fn assert_state(operator: &StreamJoinOperator, pairs: &[(String, String)]) {
    let ids = [1023, 1027, 1028, 1029, 1030, 1031, 1032, 1033];
    assert_eq!(
        operator
            .state
            .left
            .iter()
            .map(|row| row.row_id)
            .collect::<Vec<_>>(),
        ids
    );
    assert_eq!(
        (
            operator.state.next_left_row_id,
            operator.state.next_right_row_id
        ),
        (1034, 9)
    );
    assert_eq!(
        (
            operator.status().left.null_event_time_rows,
            operator.status().left.null_key_rows,
            operator.status().left.late_rows
        ),
        (1, 1024, 1)
    );
    assert_eq!(operator.key_dictionary.inventory().0, 8);
    assert_eq!(operator.key_dictionary.inventory().1, 17);
    let mut fees = 0;
    for (row, (a, b)) in operator.state.left.iter().zip(pairs) {
        assert_eq!(row.encoded_key.as_slice(), expected_frame(a, b));
        let fee = 143 + 2 * u64::try_from(a.len() + b.len()).unwrap();
        assert_eq!(row.charge, fee);
        fees += fee;
    }
    assert_eq!(operator.status().left.retained_bytes, fees);
    assert_eq!(operator.status().right.retained_bytes, fees + 157);
}

fn assert_nested_output(output: &mut EdgeCollector) {
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    let batch = &messages[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    assert_eq!(
        batch
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        ["left__payload", "right__payload"]
    );
    let values = |column| {
        let lists = batch
            .column(column)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        (0..lists.len())
            .map(|row| {
                lists
                    .value(row)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .value(0)
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        values(0),
        [1023, 1027, 1027, 1028, 1029, 1030, 1031, 1032, 1033]
    );
    assert_eq!(values(1), [0, 1, 8, 2, 3, 4, 5, 6, 7]);
}

struct FailSecond {
    accepted: Vec<Batch>,
}

#[async_trait]
impl StreamCollector for FailSecond {
    async fn emit(&mut self, _port: &str, batch: Batch) -> Result<()> {
        if self.accepted.len() == 1 {
            return Err(operator_error("match", "partial collector failure"));
        }
        self.accepted.push(batch);
        Ok(())
    }
}

#[tokio::test]
async fn test_persistent_dictionary_partial_emit_failure_does_not_publish_keys_or_rows() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None)
        .with_output_budget(EdgeBudget::new(1, 4096).unwrap());
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    let canonical = Arc::clone(operator.state.right.key_at(0));
    let inventory = operator.key_dictionary.inventory();
    let pending = operator.state.deltas.pending_count_for_test();
    let mut failed = FailSecond {
        accepted: Vec::new(),
    };
    let error = operator
        .process_data(
            "left",
            keyed_left_batch(&[7, 8, 7], &[0, 0, 0]),
            &context,
            &mut failed,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("partial collector failure"));
    assert_eq!(failed.accepted.len(), 1);
    assert_eq!(failed.accepted[0].metadata().sequence(), 0);
    assert_eq!(operator.state.next_output_sequence, 1);
    assert_eq!(operator.state.next_left_row_id, 0);
    assert!(operator.state.left.is_empty());
    assert_eq!(operator.key_dictionary.inventory(), inventory);
    assert_eq!(operator.state.deltas.pending_count_for_test(), pending);
    operator
        .process_data(
            "left",
            keyed_left_batch(&[7, 8, 7], &[0, 0, 0]),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    assert_eq!(operator.key_dictionary.inventory().1, 4);
    assert_eq!(operator.state.left.len(), 3);
    assert!(Arc::ptr_eq(&canonical, operator.state.left.key_at(0)));
    assert!(operator.state.left.iter().all(|row| row.charge == 124));
    let messages = output.drain("output");
    assert_eq!(
        messages
            .iter()
            .map(|message| message.as_data().unwrap().metadata().sequence())
            .collect::<Vec<_>>(),
        [1, 2]
    );
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((operator, output, failed, context, canonical, messages));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[derive(Debug, Default)]
struct PublicationPool {
    inner: UnboundedMemoryPool,
    refuse: AtomicBool,
    refusals: AtomicUsize,
    cancel: std::sync::Mutex<Option<CancellationToken>>,
    cancellations: AtomicUsize,
}

impl fmt::Display for PublicationPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("restore-publication-pool")
    }
}

impl MemoryPool for PublicationPool {
    fn name(&self) -> &'static str {
        "restore-publication-pool"
    }
    fn grow(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.grow(credit, bytes);
    }
    fn shrink(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.shrink(credit, bytes);
        if credit
            .consumer()
            .name()
            .ends_with(":stream-join-dictionary")
            && let Some(cancel) = self.cancel.lock().unwrap().take()
        {
            self.cancellations.fetch_add(1, Ordering::SeqCst);
            cancel.cancel();
        }
    }
    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
    fn try_grow(&self, credit: &MemoryReservation, bytes: usize) -> datafusion::common::Result<()> {
        if self.refuse.load(Ordering::SeqCst)
            && credit
                .consumer()
                .name()
                .ends_with(":stream-join-dictionary")
            && bytes > 0
        {
            self.refusals.fetch_add(1, Ordering::SeqCst);
            return Err(datafusion::common::DataFusionError::ResourcesExhausted(
                "restore dictionary grant refused".into(),
            ));
        }
        self.inner.try_grow(credit, bytes)
    }
}

fn fixed_operator() -> StreamJoinOperator {
    let declaration = StreamJoinSpec::inner(
        ["account_id"],
        ["account_id"],
        "authorized_at",
        "authorized_at",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(100, 1_000_000, 100).unwrap(),
    )
    .unwrap();
    StreamJoinOperator::new("match", left_schema(), left_schema(), declaration).unwrap()
}

#[test]
fn test_persistent_dictionary_refused_capture_has_one_atomic_publication_boundary() {
    checkpoint_compaction_tests::isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(refused_capture_publication(service));
    });
}

async fn refused_capture_publication(
    service: &crate::runtime::streaming::gather_work::TestService,
) {
    let owner = job().with_gather_owner(service.owner("dictionary-publication".into()));
    let context = StreamOperatorContext::new(&owner, "match", None);
    let mut source = fixed_operator();
    source.checkpoint_v1_test_producer = true;
    let mut output = EdgeCollector::new(source.output_ports().to_vec());
    source
        .process_data("left", left_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    source
        .process_data("right", left_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    source.state.deltas.needs_compaction = true;
    source.prepare_checkpoint_async(&context).await.unwrap();
    let mut snapshot = source.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(
        snapshot
            .segments
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["left-base", "right-base"]
    );
    let mut target = fixed_operator();
    let pool = Arc::new(PublicationPool::default());
    target
        .runtime
        .initialize_benchmark_pool(pool.clone())
        .unwrap();
    target
        .process_data("right", keyed_left_batch(&[9], &[0]), &context, &mut output)
        .await
        .unwrap();
    let before = target.status();
    let inventory = target.key_dictionary.inventory();
    let writer = target.v2_writer.observation_for_test();
    let pending = target.state.deltas.pending_count_for_test();
    pool.refuse.store(true, Ordering::SeqCst);
    *pool.cancel.lock().unwrap() = Some(owner.cancellation().clone());
    let result = target
        .restore_managed_snapshot(&mut snapshot, &owner, None)
        .await;
    assert_eq!(
        pool.cancellations.load(Ordering::SeqCst),
        1,
        "actual retirement of old dictionary credit must trigger cancellation"
    );
    assert!(pool.refusals.load(Ordering::SeqCst) > 0);
    let atomic = match &result {
        Ok(()) => target.status() == source.status() && target.key_dictionary_needs_rebuild,
        Err(CalcFlowError::Cancelled { .. }) => {
            target.status() == before
                && target.key_dictionary.inventory() == inventory
                && target.v2_writer.observation_for_test() == writer
                && target.state.deltas.pending_count_for_test() == pending
        }
        Err(error) => panic!("unexpected restore error: {error}"),
    };
    let source_pool = source.runtime.runtime().unwrap().incremental_memory_pool();
    drop((source, target, snapshot, output, context, writer));
    assert!(owner.gather_owner().close_and_drain().await.is_empty());
    drop(owner);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(source_pool.reserved(), 0);
    assert!(
        atomic,
        "restore returned {result:?} after publishing a different state"
    );
}

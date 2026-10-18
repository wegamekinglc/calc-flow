use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[tokio::test]
async fn test_persistent_dictionary_long_single_value_stops_before_canonical_frame() {
    for phase in [typed_key::KeyPhase::LocalHash, typed_key::KeyPhase::Hash] {
        assert_long_single_value_stops_before_canonical_frame(phase).await;
    }
}

async fn assert_long_single_value_stops_before_canonical_frame(stop_phase: typed_key::KeyPhase) {
    let schema = |source: SchemaRef| {
        Arc::new(Schema::new(
            source
                .fields()
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    if index == 0 {
                        field.as_ref().clone().with_data_type(DataType::Utf8)
                    } else {
                        field.as_ref().clone()
                    }
                })
                .collect::<Vec<_>>(),
        ))
    };
    let left = schema(left_schema());
    let right = schema(right_schema());
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(100, 64 << 20, 1000).unwrap();
    let mut operator =
        StreamJoinOperator::new("match", Arc::clone(&left), right, declaration).unwrap();
    let value = "x".repeat(8 << 20);
    let record = RecordBatch::try_new(
        left,
        vec![
            Arc::new(StringArray::from(vec![value])),
            Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![42])),
        ],
    )
    .unwrap();
    let cancellation = CancellationToken::new();
    let token = cancellation.clone();
    let hook = typed_key::observe(Arc::new(move |phase, bytes| {
        if phase == stop_phase && bytes == 4096 {
            token.cancel();
        }
    }));
    let job = StreamJobContext::new(1, "long-key", JsonMap::new(), None, cancellation);
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    reset_join_work();
    let error = operator
        .process_data(
            "left",
            Batch::table(vec![record], BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    assert!(operator.state.left.is_empty());
    assert_eq!(operator.key_dictionary.inventory().0, 0);
    assert_eq!(
        join_work().key_encodings,
        0,
        "a cancelled wide hash must stop before preparing its full canonical frame"
    );
    drop((hook, operator, context, output));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}

#[tokio::test]
async fn test_persistent_dictionary_collision_chain_checks_cancellation_quantum() {
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(1024, 16 << 20, 4096).unwrap();
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap();
    operator.key_dictionary.force_collisions();
    let source = left_batch(vec![0; 256]);
    let record = &source.table_payload().unwrap().batches()[0];
    let mut columns = record.columns().to_vec();
    columns[0] = Arc::new(Int64Array::from((0..256).collect::<Vec<i64>>()));
    let input = Batch::table(
        vec![RecordBatch::try_new(record.schema(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap();
    let healthy = job();
    let context = StreamOperatorContext::new(&healthy, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", input, &context, &mut output)
        .await
        .unwrap();
    let key = operator.key_dictionary.last_collision_key();
    let value = i64::from_le_bytes(key[key.len() - 8..].try_into().unwrap());
    let raw = right_batch(vec![0]);
    let record = &raw.table_payload().unwrap().batches()[0];
    let mut columns = record.columns().to_vec();
    columns[0] = Arc::new(Int64Array::from(vec![value]));
    let input = Batch::table(
        vec![RecordBatch::try_new(record.schema(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap();
    let cancellation = CancellationToken::new();
    let token = cancellation.clone();
    let comparisons = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&comparisons);
    let hook = typed_key::observe(Arc::new(move |phase, _| {
        if phase == typed_key::KeyPhase::Comparison {
            observed.fetch_add(1, Ordering::SeqCst);
            token.cancel();
        }
    }));
    let stopped = StreamJobContext::new(2, "collision-chain", JsonMap::new(), None, cancellation);
    let context = StreamOperatorContext::new(&stopped, "match", None);
    let before = operator.status();
    let error = operator
        .process_data("right", input, &context, &mut output)
        .await
        .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    assert_eq!(operator.status(), before);
    assert!(
        comparisons.load(Ordering::SeqCst) <= 64,
        "collision traversal must stop within the bounded visit quantum"
    );
    drop((hook, operator, context, output));
    assert!(healthy.gather_owner().close_and_drain().await.is_empty());
    assert!(stopped.gather_owner().close_and_drain().await.is_empty());
}

#[tokio::test]
async fn test_persistent_dictionary_collision_tail_eviction_has_bounded_unlink_work() {
    let mut declaration = spec();
    declaration.bounds = JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap();
    declaration.limits = JoinStateLimits::new(1024, 16 << 20, 4096).unwrap();
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap();
    operator.key_dictionary.force_collisions();
    let mut times = vec![10; 256];
    times[0] = 0;
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "right",
            keyed_right_batch(&(0..256).collect::<Vec<_>>(), &times),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    reset_join_work();
    operator
        .evict_progress(
            "left",
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1))),
            true,
        )
        .unwrap();
    assert_eq!(operator.state.right.len(), 255);
    assert_eq!(operator.key_dictionary.inventory().1, 255);
    assert!(
        join_work().dictionary_unlink_visits <= 3,
        "tail unlink must touch only removed/predecessor/successor slots, observed {}",
        join_work().dictionary_unlink_visits
    );
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((operator, output, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_persistent_dictionary_small_keys_hash_once_without_per_chunk_quantum_futures() {
    let short = prepare_small_keys(256).await;
    let long = prepare_small_keys(512).await;
    for work in [&short, &long] {
        assert_eq!(
            work.hashed, work.distinct_frames,
            "each distinct small key is charged and hashed once with its whole frame"
        );
        assert!(
            work.boundaries * 16 <= work.frames.len(),
            "small-key cooperation stays amortized per quantum, observed {} boundaries",
            work.boundaries
        );
    }
    let extra = long.frames.len() - short.frames.len();
    assert!(
        long.steps - short.steps <= extra + 4,
        "known small keys must pay at most one cooperative step per row, observed {} steps for {extra} extra rows",
        long.steps - short.steps
    );
}

/// Frame lengths of the fixture's distinct keys in first-appearance order.
fn distinct_small_frames(frames: &[usize]) -> Vec<usize> {
    let mut distinct = Vec::new();
    for frame in frames {
        if !distinct.contains(frame) {
            distinct.push(*frame);
        }
    }
    distinct
}

struct SmallKeyWork {
    frames: Vec<usize>,
    distinct_frames: Vec<usize>,
    hashed: Vec<usize>,
    steps: usize,
    boundaries: usize,
}

async fn prepare_small_keys(count: usize) -> SmallKeyWork {
    use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, UnboundedMemoryPool};

    let wide = "x".repeat(100);
    let values = (0..count)
        .map(|row| match row % 4 {
            0 => Some(""),
            1 => Some("AAPL"),
            2 => Some(wide.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Utf8, true)]));
    let parent =
        Arc::new(RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(values))]).unwrap());
    let encoded = (0..count)
        .map(|row| encode_join_key_columns_v1(parent.columns(), row, &[0]).unwrap())
        .collect::<Vec<_>>();
    let rows = (0..count)
        .map(|row| AdmittedRow {
            record: columnar::RowPayload::Rowed {
                parent: Arc::clone(&parent),
                row,
            },
            event_time: EventTime::from_micros(0),
            row_id: row as u64,
            retain: true,
        })
        .collect::<Vec<_>>();
    let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
    let master = MemoryConsumer::new("small-keys").register(&pool);
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut dictionary = key_dictionary::Dictionary::default();
    let hashed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed = Arc::clone(&hashed);
    let hook = typed_key::observe(Arc::new(move |phase, bytes| {
        if phase == typed_key::KeyPhase::Hash {
            observed.lock().unwrap().push(bytes);
        }
    }));
    reset_join_work();
    let prepared = dictionary
        .prepare(&rows, &[0], master, Some(&context), "match")
        .await
        .unwrap()
        .unwrap();
    let work = join_work();
    drop(hook);
    for (row, &id) in prepared.ids.iter().enumerate() {
        let key = prepared.keys[id as usize].as_ref().unwrap();
        assert_eq!(key.as_slice(), encoded[row], "row {row} identity");
    }
    dictionary.commit(prepared.dictionary);
    for frame in &encoded {
        assert!(
            dictionary.canonical(frame).is_some(),
            "stored hashes must equal the canonical framed hash"
        );
    }
    let hashed = hashed.lock().unwrap().clone();
    let frames = encoded.iter().map(Vec::len).collect::<Vec<_>>();
    let distinct_frames = distinct_small_frames(&frames);
    SmallKeyWork {
        frames,
        distinct_frames,
        hashed,
        steps: work.quantum_steps,
        boundaries: work.quantum_boundaries,
    }
}

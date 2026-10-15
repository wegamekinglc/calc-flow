use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[tokio::test]
async fn test_persistent_dictionary_long_single_value_stops_before_canonical_frame() {
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
        if phase == typed_key::KeyPhase::Hash && bytes == 4096 {
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

use super::*;

#[tokio::test]
async fn test_persistent_dictionary_slot_exhaustion_preserves_committed_identity() {
    unchanged_failure(
        Arc::new(Int64Array::from(vec![7])),
        Arc::new(Int64Array::from(vec![8])),
        key_dictionary::TestLimits {
            max_slot: 0,
            ..Default::default()
        },
        0,
    )
    .await;
}

#[tokio::test]
async fn test_persistent_dictionary_reference_exhaustion_precedes_match_publication() {
    unchanged_failure(
        Arc::new(Int64Array::from(vec![7])),
        Arc::new(Int64Array::from(vec![7])),
        key_dictionary::TestLimits {
            max_references: 1,
            ..Default::default()
        },
        1,
    )
    .await;
}

#[tokio::test]
async fn test_persistent_dictionary_framed_byte_exhaustion_preserves_prior_key() {
    unchanged_failure(
        Arc::new(StringArray::from(vec!["a"])),
        Arc::new(StringArray::from(vec!["aa"])),
        key_dictionary::TestLimits {
            max_key_bytes: 10,
            ..Default::default()
        },
        0,
    )
    .await;
}

async fn unchanged_failure(
    initial: ArrayRef,
    failing: ArrayRef,
    limits: key_dictionary::TestLimits,
    retry_matches: usize,
) {
    let mut operator = keyed_operator(initial.data_type());
    operator.key_dictionary.set_test_limits(limits);
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "right",
            keyed_batch(false, Arc::clone(&initial)),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    let original = operator
        .key_dictionary
        .canonical(operator.state.right.key_at(0))
        .unwrap();
    let lease = operator.key_dictionary.lease(&original).unwrap();
    let address = lease.address();
    let before = Snapshot::capture(&operator);
    operator.key_dictionary.set_test_limits(limits);
    let error = operator
        .process_data(
            "left",
            keyed_batch(true, Arc::clone(&failing)),
            &context,
            &mut output,
        )
        .await
        .expect_err("dictionary domain exhaustion must reject before publishing any row");
    assert_eq!(
        reason_of(&error),
        Some(crate::StreamingFailureReason::JoinCounterOverflow)
    );
    before.assert_unchanged(&operator);
    assert!(output.drain("output").is_empty());
    assert_eq!(
        operator.key_dictionary.lease(&original).unwrap().address(),
        address
    );
    operator
        .key_dictionary
        .set_test_limits(key_dictionary::TestLimits::default());
    operator
        .process_data("left", keyed_batch(true, failing), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(drain_rows(&mut output), retry_matches);
    assert_eq!(
        (operator.state.left.len(), operator.state.right.len()),
        (1, 1)
    );
    let retry = operator
        .key_dictionary
        .canonical(operator.state.left.key_at(0))
        .unwrap();
    assert_eq!(Arc::ptr_eq(&original, &retry), retry_matches == 1);
    operator
        .process_data("left", keyed_batch(true, initial), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(drain_rows(&mut output), 1);
    assert_eq!(operator.key_dictionary.inventory().1, 3);
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((operator, output, context, before, retry, lease, original));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_persistent_dictionary_namespace_exhaustion_keeps_retired_lease_distinct() {
    let mut operator = keyed_operator(&DataType::Int64);
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    let original = operator
        .key_dictionary
        .canonical(operator.state.right.key_at(0))
        .unwrap();
    let lease = operator.key_dictionary.lease(&original).unwrap();
    let address = lease.address();
    let bytes = original.to_vec();
    operator
        .evict_progress(
            "left",
            IngressProgress::new(IngressState::Ended, None),
            true,
        )
        .unwrap();
    assert_eq!(operator.key_dictionary.inventory(), (0, 0, 0, false));
    assert!(operator.key_dictionary.lease(&original).is_none());
    let before = Snapshot::capture(&operator);
    operator
        .key_dictionary
        .set_test_limits(key_dictionary::TestLimits {
            reject_new_namespace: true,
            ..Default::default()
        });
    let error = operator
        .process_data("left", left_batch(vec![0]), &context, &mut output)
        .await
        .expect_err("exhausted fresh namespace must not alias a surviving retired lease");
    assert_eq!(
        reason_of(&error),
        Some(crate::StreamingFailureReason::JoinCounterOverflow)
    );
    before.assert_unchanged(&operator);
    assert!(output.drain("output").is_empty());
    assert_eq!(lease.address(), address);
    assert_eq!(lease.owner().as_slice(), bytes);
    operator
        .key_dictionary
        .set_test_limits(key_dictionary::TestLimits::default());
    operator
        .process_data("left", left_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
    let retry = operator
        .key_dictionary
        .canonical(operator.state.left.key_at(0))
        .unwrap();
    let current = operator.key_dictionary.lease(&retry).unwrap();
    assert_ne!(current.address(), address);
    assert!(!Arc::ptr_eq(&original, &retry));
    assert_eq!(retry.as_slice(), bytes);
    assert!(operator.key_dictionary.lease(&original).is_none());
    operator
        .process_data("right", right_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(drain_rows(&mut output), 1);
    assert_eq!(operator.key_dictionary.inventory().1, 2);
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((
        operator, output, context, before, current, retry, lease, original,
    ));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

fn keyed_operator(key_type: &DataType) -> StreamJoinOperator {
    StreamJoinOperator::new(
        "match",
        keyed_schema(&left_schema(), key_type),
        keyed_schema(&right_schema(), key_type),
        spec(),
    )
    .unwrap()
}

fn keyed_schema(schema: &SchemaRef, key_type: &DataType) -> SchemaRef {
    let mut fields = schema
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields[0] = fields[0].clone().with_data_type(key_type.clone());
    Arc::new(Schema::new(fields))
}

fn keyed_batch(left: bool, keys: ArrayRef) -> Batch {
    let source = if left {
        left_batch(vec![0; keys.len()])
    } else {
        right_batch(vec![0; keys.len()])
    };
    let record = &source.table_payload().unwrap().batches()[0];
    let schema = keyed_schema(&record.schema(), keys.data_type());
    let mut columns = record.columns().to_vec();
    columns[0] = keys;
    Batch::table(
        vec![RecordBatch::try_new(schema, columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn drain_rows(output: &mut EdgeCollector) -> usize {
    output
        .drain("output")
        .iter()
        .map(|message| message.as_data().unwrap().num_rows())
        .sum()
}

/// One observed dirty identity: a derived upsert carries its charge, a
/// recorded tombstone does not (spec FR3).
type PendingIdentity = (JoinSide, u64, EventTime, Vec<u8>, Option<u64>);

struct Snapshot {
    status: StreamJoinStatus,
    inventory: (usize, u64, usize, bool),
    ids: (u64, u64, u64),
    left: Vec<StoredRow>,
    right: Vec<StoredRow>,
    pending: Vec<PendingIdentity>,
}

impl Snapshot {
    fn capture(operator: &StreamJoinOperator) -> Self {
        Self {
            status: operator.status(),
            inventory: operator.key_dictionary.inventory(),
            ids: (
                operator.state.next_left_row_id,
                operator.state.next_right_row_id,
                operator.state.next_output_sequence,
            ),
            left: operator.state.left.to_vec(),
            right: operator.state.right.to_vec(),
            pending: pending(operator),
        }
    }

    fn assert_unchanged(&self, operator: &StreamJoinOperator) {
        assert_eq!(operator.status(), self.status);
        assert_eq!(operator.key_dictionary.inventory(), self.inventory);
        assert_eq!(
            (
                operator.state.next_left_row_id,
                operator.state.next_right_row_id,
                operator.state.next_output_sequence
            ),
            self.ids
        );
        assert_rows(&operator.state.left, &self.left);
        assert_rows(&operator.state.right, &self.right);
        assert_eq!(pending(operator), self.pending);
    }
}

fn assert_rows(actual: &dyn Rows, expected: &[StoredRow]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(
            (actual.row_id, actual.event_time, actual.charge),
            (expected.row_id, expected.event_time, expected.charge)
        );
        assert!(Arc::ptr_eq(&actual.encoded_key, &expected.encoded_key));
        assert_eq!(&*actual.record.view(), &*expected.record.view());
    }
}

fn pending(operator: &StreamJoinOperator) -> Vec<PendingIdentity> {
    let deltas = &operator.state.deltas;
    let mut ops = Vec::new();
    for (index, rows) in [
        (0usize, &operator.state.left),
        (1usize, &operator.state.right),
    ] {
        let side = if index == 0 {
            JoinSide::Left
        } else {
            JoinSide::Right
        };
        let watermark = deltas.watermark[index];
        let mut upserts = rows.references_from(watermark).collect::<Vec<_>>();
        upserts.sort_unstable_by_key(|&position| rows.identity(position).1);
        for position in upserts {
            let (event_time, row_id) = rows.identity(position);
            ops.push((
                side,
                row_id,
                event_time,
                rows.key(position).as_slice().to_vec(),
                Some(rows.charge(position)),
            ));
        }
        let list = &deltas.tombstones[index];
        let mut order = (0..list.len()).collect::<Vec<_>>();
        order.sort_unstable_by_key(|&entry| list.row_ids[entry]);
        for entry in order {
            let (row_id, event_time, key) = list.entry(entry);
            ops.push((side, row_id, event_time, key.as_slice().to_vec(), None));
        }
    }
    ops
}

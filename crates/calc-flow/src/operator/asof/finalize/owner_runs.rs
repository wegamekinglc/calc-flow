//! Differential and complexity checks for finalized prefix ownership.

use super::*;
use crate::{
    AsofJoinSide, AsofStateLimits, CancellationToken, StreamAsofJoinSpec, StreamJobContext,
};
use datafusion::{
    arrow::{
        array::{ArrayRef, Int64Array, StringArray, TimestampMicrosecondArray, UInt8Array},
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch,
    },
    execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool},
};
use std::{sync::OnceLock, time::Duration};

fn operator(
    key: ArrayRef,
    sequence: ArrayRef,
    parts: &[std::ops::Range<usize>],
) -> StreamAsofJoinOperator {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", key.data_type().clone(), false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", sequence.data_type().clone(), false),
    ]));
    let len = key.len();
    let record = RecordBatch::try_new(
        schema,
        vec![
            key,
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(
                    (0..len).map(|i| i64::try_from(i).unwrap()),
                )
                .with_timezone("UTC"),
            ),
            sequence,
        ],
    )
    .unwrap();
    operator_for(&record, &["key".into()], &["seq".into()], parts)
}

fn operator_for(
    record: &RecordBatch,
    keys: &[String],
    sequences: &[String],
    parts: &[std::ops::Range<usize>],
) -> StreamAsofJoinOperator {
    let schema = record.schema();
    let side = |prefix: &str| {
        AsofJoinSide::new(
            keys.to_vec(),
            "time".into(),
            sequences.to_vec(),
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::ZERO,
        AsofStateLimits::new(100_000, 100_000_000).unwrap(),
    )
    .unwrap();
    let mut operator =
        StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap();
    for (id, part) in parts.iter().enumerate() {
        let len = part.end - part.start;
        let record = record.slice(part.start, len);
        let keys = state::encode_columns(&record, operator.spec.left().keys()).unwrap();
        let sequences = (operator.spec.left().keys() != operator.spec.left().sequence_by())
            .then(|| state::encode_columns(&record, operator.spec.left().sequence_by()).unwrap());
        let times = record
            .column_by_name("time")
            .unwrap()
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        let rows = (0..len)
            .map(|row| {
                (
                    (
                        times.value(row),
                        keys.row(row),
                        sequences.as_ref().unwrap_or(&keys).row(row),
                    ),
                    state::AdmissionRef {
                        batch_index: 0,
                        row: u32::try_from(row).unwrap(),
                        key_index: 0,
                    },
                )
            })
            .collect::<Vec<_>>();
        let owner = Arc::new(state::PayloadBatch {
            key: (0, id as u64),
            record: Arc::new(record),
            encoded: OnceLock::new(),
            encoded_charge_bytes: 0,
            body_bytes: 0,
        });
        let chunks =
            state::PreparedLeftChunk::prepare(&rows, &[owner], operator.spec.left(), "asof")
                .unwrap();
        operator
            .state
            .left
            .install(chunks, &mut operator.state.batches);
    }
    operator.state.rebuild_encoding_owners();
    operator
}

fn composite_operator(sequence: ArrayRef, overlap: bool) -> StreamAsofJoinOperator {
    let count = sequence.len();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("key2", DataType::UInt8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", sequence.data_type().clone(), false),
        Field::new("seq2", DataType::Int64, false),
    ]));
    let times = (0..count).map(|i| {
        if overlap {
            i64::try_from((i % (count / 2)) * 2 + i / (count / 2)).unwrap()
        } else {
            i64::try_from(i).unwrap()
        }
    });
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from_iter_values(
                (0..count).map(|i| format!("long key {}", i % 3)),
            )),
            Arc::new(UInt8Array::from_iter_values(
                (0..count).map(|i| u8::try_from(i % 2).unwrap()),
            )),
            Arc::new(TimestampMicrosecondArray::from_iter_values(times).with_timezone("UTC")),
            sequence,
            Arc::new(Int64Array::from_iter_values(
                (0..count).map(|i| i64::try_from(i).unwrap()),
            )),
        ],
    )
    .unwrap();
    operator_for(
        &record,
        &["key".into(), "key2".into()],
        &["seq".into(), "seq2".into()],
        &[count / 2..count, 0..count / 2],
    )
}

#[tokio::test]
async fn every_prefix_cut_matches_rowwise_owner_inventory_after_partial_consumption() {
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 24));
    let selected = [Vec::new(), Vec::new()];
    for overlap in [false, true] {
        for consumed in [0, 1, 7, 12, 19] {
            for kind in 0..3 {
                let sequences = Arc::new(StringArray::from_iter_values(
                    (0..24).map(|i| format!("canonical sequence {i:04}")),
                )) as ArrayRef;
                let mut operator = match kind {
                    0 => operator(
                        Arc::new(StringArray::from(vec!["long key"; 24])),
                        Arc::new(Int64Array::from_iter_values(-12..12)),
                        &[12..24, 0..12],
                    ),
                    1 => composite_operator(sequences, overlap),
                    _ => operator(
                        Arc::new(StringArray::from_iter_values(
                            (0..24).map(|i| format!("key-{i:04}")),
                        )),
                        sequences,
                        &[12..24, 0..12],
                    ),
                };
                operator.state.commit_left_prefix(consumed);
                for count in 0..=operator.state.left.len() {
                    let expected = rowwise_prefix(&operator, count);
                    let mut reservation = MemoryConsumer::new("owner-test").register(&pool);
                    let mut plan =
                        OutputPlanBuilder::new(count, Some(&selected), &mut reservation, "asof")
                            .unwrap();
                    let actual = binary_search_candidate_rows(
                        &operator,
                        count,
                        &context,
                        &mut plan,
                        &mut reservation,
                    )
                    .await
                    .unwrap();
                    assert_prefix(&actual, &expected);
                    assert_projected_inventory_and_journal(&operator, &actual, &expected);
                }
            }
        }
    }
}

fn rowwise_prefix(operator: &StreamAsofJoinOperator, count: usize) -> LeftPrefix {
    let mut rows = operator
        .state
        .left
        .unordered_iter()
        .map(|((time, key, sequence), row)| ((*time, key.clone(), sequence.into_owned()), row))
        .collect::<Vec<_>>();
    rows.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    let mut prefix = LeftPrefix::default();
    for ((_, key, sequence), row) in rows.into_iter().take(count) {
        prefix
            .visit_owners(
                &key,
                Some(&sequence),
                operator.state.batches.key(row),
                "asof",
            )
            .unwrap();
    }
    prefix
}

fn assert_prefix(actual: &LeftPrefix, expected: &LeftPrefix) {
    assert_eq!(actual.count, expected.count);
    assert_eq!(actual.batches, expected.batches);
    assert_eq!(actual.keys, expected.keys);
    assert_eq!(actual.owners, expected.owners);
    assert_eq!(actual.sequence_owners, expected.sequence_owners);
}

fn assert_projected_inventory_and_journal(
    operator: &StreamAsofJoinOperator,
    actual: &LeftPrefix,
    expected: &LeftPrefix,
) {
    let state = &operator.state;
    let drain = state
        .left
        .drain_input(&actual.batches, &state.batches)
        .prepare();
    assert_eq!(
        state.left.journal_prefix(actual, &drain, &state.batches),
        state.left.journal_prefix(expected, &drain, &state.batches),
    );
    let snapshot = state.capacity_snapshot("asof");
    assert_eq!(
        state
            .project_capacity_prefix(snapshot, actual, &drain, "asof")
            .unwrap(),
        state
            .project_capacity_prefix(snapshot, expected, &drain, "asof")
            .unwrap(),
    );
}

fn aggregated_prefix(operator: &StreamAsofJoinOperator, count: usize) -> LeftPrefix {
    let mut prefix = LeftPrefix::default();
    for run in operator.state.left.output_runs() {
        let amount = run.len().min(count - prefix.count);
        run.visit_prefix(amount, &mut prefix, &operator.state.batches, "asof")
            .unwrap();
        if prefix.count == count {
            break;
        }
    }
    prefix
}

#[test]
fn typed_integer_prefix_updates_scale_with_distinct_keys_and_skip_sequence_owners() {
    let count = 1_024;
    let key = Arc::new(StringArray::from_iter_values(
        (0..count).map(|i| format!("long key {:04}", i % 64)),
    )) as ArrayRef;
    let sequence = Arc::new(Int64Array::from_iter_values(
        (0..count).map(|i| i64::try_from(i).unwrap()),
    )) as ArrayRef;
    let operator = operator(key, sequence, std::slice::from_ref(&(0..count)));
    let expected = rowwise_prefix(&operator, count);
    state::take_prefix_updates();
    let prefix = aggregated_prefix(&operator, count);
    assert_eq!(state::take_prefix_updates(), (1, 64, 0));
    assert_prefix(&prefix, &expected);
}

#[test]
fn unique_keys_and_tiny_cuts_fit_the_unchanged_prefix_workspace() {
    for canonical in [false, true] {
        let count = 10_000;
        let key = Arc::new(StringArray::from_iter_values(
            (0..count).map(|i| format!("unique long key {i:08}")),
        )) as ArrayRef;
        let sequence: ArrayRef = if canonical {
            Arc::new(StringArray::from_iter_values(
                (0..count).map(|i| format!("shared canonical sequence {i:08}")),
            ))
        } else {
            Arc::new(Int64Array::from_iter_values(
                (0..count).map(|i| i64::try_from(i).unwrap()),
            ))
        };
        let operator = operator(key, sequence, std::slice::from_ref(&(0..count)));
        for selected in [0, 1, 2, 5, 17, 127, 1_024, count] {
            let expected = rowwise_prefix(&operator, selected);
            let mut prefix = LeftPrefix::default();
            let allocation = allocation_counter::measure(|| {
                prefix = aggregated_prefix(&operator, selected);
            });
            assert_prefix(&prefix, &expected);
            let charge =
                prefix_workspace_bytes(selected) + operator.state.left.iter_workspace_bytes();
            assert!(
                allocation.bytes_max <= charge,
                "canonical={canonical}, selected={selected}, charge={charge}, actual={allocation:?}"
            );
        }
    }
}

#[test]
fn shared_key_and_sequence_buffer_removals_preserve_partial_counts() {
    let count = 24;
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
    ]));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from_iter_values(
                (0..count).map(|i| format!("one shared key and sequence {i:08}")),
            )),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(
                    (0..count).map(|i| i64::try_from(i).unwrap()),
                )
                .with_timezone("UTC"),
            ),
        ],
    )
    .unwrap();
    let mut operator = operator_for(
        &record,
        &["key".into()],
        &["key".into()],
        std::slice::from_ref(&(0..count)),
    );
    for consumed in [0, 7] {
        if consumed != 0 {
            operator.state.commit_left_prefix(consumed);
        }
        for selected in 1..=operator.state.left.len() {
            let expected = rowwise_prefix(&operator, selected);
            let prefix = aggregated_prefix(&operator, selected);
            assert_prefix(&prefix, &expected);
            assert_eq!(prefix.owners.len(), 1);
            assert_projected_inventory_and_journal(&operator, &prefix, &expected);
            assert_eq!(prefix.owners.values().sum::<usize>(), selected * 2);
            assert_eq!(
                prefix
                    .sequence_owners
                    .values()
                    .flat_map(|owners| owners.values())
                    .sum::<usize>(),
                selected
            );
        }
    }
}

#[tokio::test]
async fn matching_aggregates_batch_key_and_shared_sequence_owners_by_run() {
    let key = Arc::new(StringArray::from(vec!["a long shared key encoding"; 512])) as ArrayRef;
    let sequence = Arc::new(StringArray::from_iter_values(
        (0..512).map(|i| format!("long sequence encoding {i:04}")),
    )) as ArrayRef;
    let operator = operator(key, sequence, &[256..512, 0..256]);
    let expected = rowwise_prefix(&operator, 512);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 24));
    let selected = [Vec::new(), Vec::new()];
    for mode in 0..3 {
        let mut reservation = MemoryConsumer::new("owner-test").register(&pool);
        let mut plan =
            OutputPlanBuilder::new(512, Some(&selected), &mut reservation, "asof").unwrap();
        state::take_prefix_updates();
        let actual = match mode {
            0 => {
                binary_search_candidate_rows(&operator, 512, &context, &mut plan, &mut reservation)
                    .await
                    .unwrap()
            }
            1 => monotonic_candidate_rows(&operator, 512, &context, &mut plan, &mut reservation)
                .await
                .unwrap(),
            _ => parallel_candidate_rows(
                &operator,
                512,
                &context,
                &vec![None; 512],
                &mut plan,
                &mut reservation,
            )
            .await
            .unwrap(),
        };
        assert_prefix(&actual, &expected);
        assert_eq!(
            state::take_prefix_updates(),
            (2, 2, 2),
            "matching mode {mode}"
        );
    }
}

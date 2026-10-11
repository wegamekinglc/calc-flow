use super::*;

thread_local! {
    static INTERNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(in super::super::super) fn record_intern() {
    INTERNS.set(INTERNS.get() + 1);
}

fn assert_canonical(
    operator: &StreamAsofJoinOperator,
    records: &[RecordBatch],
    identities: &InputIdentities<'_>,
) {
    assert_eq!(identities.duplicates, 0);
    for record in records {
        let keys = state::encode_columns(record, operator.spec.right().keys()).unwrap();
        let sequences = state::encode_columns(record, operator.spec.right().sequence_by()).unwrap();
        for (identity, parent, row, key_index) in &identities.rows {
            if !std::ptr::eq(*parent, record) {
                continue;
            }
            assert_eq!(identity.1, keys.row(*row));
            assert_eq!(identity.2, sequences.row(*row));
            assert_eq!(
                identity.1,
                identities.right_capacities[*key_index as usize].0
            );
        }
    }
}

#[test]
fn repeated_right_keys_resolve_each_dictionary_entry_once() {
    for wide in [false, true] {
        check_repeated_keys(wide);
    }
}

fn check_repeated_keys(wide: bool) {
    let input = key_batch(4_096, 64, wide);
    let records = input.table_payload().unwrap().batches();
    let operator = fixture(&records[0].schema());
    let pool = operator.runtime.pool.clone();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    INTERNS.set(0);
    let identities = operator
        .admission_identities(
            records,
            ValidatedInput {
                index: 1,
                watermark: None,
            },
            input.num_rows(),
            &context,
        )
        .unwrap();
    assert_eq!(identities.rows.len(), 4_096);
    assert_eq!(identities.right_capacities.len(), 64);
    assert!(
        identities
            .right_capacities
            .iter()
            .all(|(_, rows)| *rows == 64)
    );
    assert_canonical(&operator, records, &identities);
    let interns = INTERNS.replace(0);
    drop(identities);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(
        interns, 64,
        "repeated dictionary IDs re-probe the key interner"
    );
}

#[test]
fn dictionary_ids_do_not_escape_their_record_batch() {
    let schema = schema(DataType::Utf8, DataType::Int64);
    let records = [false, true]
        .into_iter()
        .enumerate()
        .map(|(index, reversed)| {
            let start = index as i64 * 64;
            record(
                &schema,
                Arc::new(StringArray::from_iter_values((0..64).map(|row| {
                    if (row % 2 == 0) == reversed {
                        "first"
                    } else {
                        "second"
                    }
                }))),
                Arc::new(Int64Array::from_iter_values(start..start + 64)),
                (start..start + 64).collect(),
            )
        })
        .collect::<Vec<_>>();
    let operator = fixture(&schema);
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    INTERNS.set(0);
    let identities = operator
        .admission_identities(
            &records,
            ValidatedInput {
                index: 1,
                watermark: None,
            },
            128,
            &context,
        )
        .unwrap();
    assert_canonical(&operator, &records, &identities);
    assert_eq!(identities.right_capacities.len(), 2);
    assert!(
        identities
            .right_capacities
            .iter()
            .all(|(_, count)| *count == 64)
    );
    assert_eq!(INTERNS.replace(0), 4);
    drop(identities);
    assert_eq!(operator.runtime.pool.reserved(), 0);
}

#[test]
fn mixed_late_right_rows_preserve_capacity_counts_and_identity() {
    let input = key_batch(64, 2, false);
    let records = input.table_payload().unwrap().batches();
    let operator = fixture(&records[0].schema());
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let identities = operator
        .admission_identities(
            records,
            ValidatedInput {
                index: 1,
                watermark: Some(32),
            },
            32,
            &context,
        )
        .unwrap();
    assert_canonical(&operator, records, &identities);
    assert_eq!(identities.rows.len(), 32);
    assert!(
        identities
            .rows
            .iter()
            .all(|(identity, _, _, _)| identity.0 >= 32)
    );
    assert!(
        identities
            .right_capacities
            .iter()
            .all(|(_, count)| *count == 16)
    );
    drop(identities);
    assert_eq!(operator.runtime.pool.reserved(), 0);
}

#[test]
fn optional_dictionary_credit_refusal_preserves_fallback() {
    let input = key_batch(64, 2, false);
    let record = &input.table_payload().unwrap().batches()[0];
    let mut operator = fixture(&record.schema());
    operator.runtime.pool = Arc::new(GreedyMemoryPool::new(24));
    let encoded = InputEncodings::new(
        &operator,
        record,
        ValidatedInput {
            index: 1,
            watermark: None,
        },
    )
    .unwrap()
    .unwrap();
    assert!(encoded.key_cache.is_none());
    assert!(encoded.hashes.is_none());
    drop(encoded);
    assert_eq!(operator.runtime.pool.reserved(), 0);
}

#[test]
fn distinct_keys_keep_the_existing_per_row_hash_path() {
    let input = key_batch(64, 64, false);
    let record = &input.table_payload().unwrap().batches()[0];
    let operator = fixture(&record.schema());
    let encoded = InputEncodings::new(
        &operator,
        record,
        ValidatedInput {
            index: 1,
            watermark: None,
        },
    )
    .unwrap()
    .unwrap();
    assert!(encoded.key_cache.is_none());
    assert_eq!(encoded.hashes.as_ref().unwrap().len(), 64);
    drop(encoded);
    assert_eq!(operator.runtime.pool.reserved(), 0);
}

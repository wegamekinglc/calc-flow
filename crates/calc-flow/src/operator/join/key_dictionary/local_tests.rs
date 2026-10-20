use super::*;
use crate::operator::join::{
    AdmittedRow, admitted::AdmittedGroup, columnar::RowPayload, typed_key,
};
use crate::{CalcFlowError, CancellationToken, EventTime, StreamJobContext, StreamOperatorContext};
use datafusion::arrow::{
    array::ArrayRef,
    datatypes::{Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, UnboundedMemoryPool};
use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn admitted(columns: Vec<ArrayRef>) -> Vec<AdmittedRow> {
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .enumerate()
            .map(|(index, column)| Field::new(index.to_string(), column.data_type().clone(), false))
            .collect::<Vec<_>>(),
    ));
    let record = Arc::new(RecordBatch::try_new(schema, columns).unwrap());
    (0..record.num_rows())
        .map(|row| AdmittedRow {
            record: RowPayload::Rowed {
                parent: Arc::clone(&record),
                row,
            },
            row_id: row as u64,
            event_time: EventTime::from_micros(0),
            retain: true,
        })
        .collect()
}

fn local_table(rows: usize) -> LocalIntern {
    let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
    LocalIntern::new(rows, MemoryConsumer::new("local-stop-test").register(&pool))
}

fn seed_distinct(local: &mut LocalIntern, hash: u64, row: usize) {
    let mask = local.table.len() - 1;
    let mut bucket = usize::try_from(hash & u64::try_from(mask).unwrap()).unwrap();
    while local.table[bucket] != u32::MAX {
        bucket = (bucket + 1) & mask;
    }
    local.table[bucket] = local.insert_distinct(hash, row, "match").unwrap();
}

#[tokio::test]
async fn test_parallel_merge_preserves_explicit_first_appearance_without_owner_rows() {
    let values = [7_i64, 9, 7];
    let columns = [LocalColumn::I64(&values)];
    let mut local = local_table(values.len());
    let mut key_flow = KeyFlow::new(None);
    let mut flow = LocalFlow::new(&mut key_flow);
    for (row, appearance, expected) in [(0, 31, 0), (1, 77, 1), (2, 99, 0)] {
        assert_eq!(
            local
                .merge_hashed(&columns, row, 42, appearance, "match", &mut flow)
                .await
                .unwrap(),
            expected
        );
    }
    assert_eq!(local.appearances, [31, 77]);
    assert!(local.locals.is_empty());
}

#[tokio::test]
async fn test_parallel_owner_merge_reuses_allowance_across_distinct_keys() {
    use crate::operator::join::{join_work, reset_join_work};

    let values = (0..32).collect::<Vec<i64>>();
    let columns = [LocalColumn::I64(&values)];
    let mut local = local_table(values.len());
    let job = StreamJobContext::new(
        1,
        "owner-merge-allowance",
        BTreeMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut key_flow = KeyFlow::new(Some(&context));
    let mut flow = LocalFlow::new(&mut key_flow);
    reset_join_work();
    for row in 0..values.len() {
        local
            .merge_hashed(
                &columns,
                row,
                row as u64,
                u32::try_from(row + 1_024).unwrap(),
                "match",
                &mut flow,
            )
            .await
            .unwrap();
        flow.step(1, 0).await.unwrap();
    }
    assert!(
        join_work().quantum_boundaries <= 1,
        "32 distinct keys must share the probe and metadata allowance"
    );
    assert_eq!(local.appearances, (1_024..1_056).collect::<Vec<_>>());
    assert!(local.locals.is_empty());
}

#[tokio::test]
async fn test_parallel_worker_short_hashes_match_cooperative_column_blocks() {
    let strings = ["", "é\0tail", "", "other"];
    let cases: Vec<Vec<ArrayRef>> = vec![
        vec![Arc::new(StringArray::from(strings.to_vec()))],
        vec![Arc::new(LargeStringArray::from(strings.to_vec()))],
        vec![Arc::new(PrimitiveArray::<Int64Type>::from(vec![
            -1, 7, -1, 9,
        ]))],
        vec![
            Arc::new(StringArray::from(strings.to_vec())),
            Arc::new(PrimitiveArray::<Int8Type>::from(vec![-1, 2, -1, 3])),
        ],
        vec![Arc::new(BooleanArray::from(vec![true, false, true, false]))],
    ];
    for arrays in cases {
        let rows = admitted(arrays);
        let indices = (0..rows[0].record.columns().len()).collect::<Vec<_>>();
        let columns = bind_columns((&rows[0].record).into(), &indices).unwrap();
        let mut serial = local_table(rows.len());
        serial
            .intern_group(
                &columns,
                AdmittedGroup::new(&rows, 0..rows.len()),
                "match",
                &mut KeyFlow::new(None),
            )
            .await
            .unwrap();
        let job = StreamJobContext::new(
            1,
            "worker-hash-parity",
            BTreeMap::new(),
            None,
            CancellationToken::new(),
        );
        let stop = GatherStop::from_job(&job);
        let mut flow = WorkerFlow::new(&stop).unwrap();
        let mut worker = local_table(rows.len());
        for row in 0..rows.len() {
            let (id, _) = worker
                .intern_worker_row(&columns, row, "match", &mut flow)
                .unwrap()
                .unwrap();
            worker.locals.push(id);
        }
        assert_eq!(worker.locals, serial.locals);
        assert_eq!(worker.hashes, serial.hashes);
        assert_eq!(worker.appearances, serial.appearances);
    }
}

#[test]
fn test_parallel_worker_collision_checks_cancel_and_deadline_within_quantum() {
    let values = (0..257).collect::<Vec<i64>>();
    let columns = [LocalColumn::I64(&values)];
    for deadline in [false, true] {
        let mut local = local_table(values.len());
        for row in 0..256 {
            seed_distinct(&mut local, 42, row);
        }
        let (job, processed, _hook) = stop_after_work(typed_key::KeyPhase::LocalProbe, deadline);
        let stop = GatherStop::from_job(&job);
        let mut flow = WorkerFlow::new(&stop).unwrap();
        let error = local
            .intern_worker_hashed(42, (256, 8), &columns, "match", &mut flow)
            .unwrap_err();
        assert!(matches!(error, CalcFlowError::Cancelled { .. }));
        assert!((1..=COPY_VISITS).contains(&processed.load(Ordering::SeqCst)));
        assert_eq!(local.distinct, 256);
    }
}

#[tokio::test]
async fn test_parallel_owner_merge_collision_stays_cooperative() {
    let values = (0..257).collect::<Vec<i64>>();
    let columns = [LocalColumn::I64(&values)];
    let mut local = local_table(values.len());
    for row in 0..256 {
        seed_distinct(&mut local, 42, row);
    }
    let (job, processed, _hook) = stop_after_work(typed_key::KeyPhase::LocalProbe, false);
    let context = StreamOperatorContext::new(&job, "match", None);
    let error = local
        .merge_hashed(
            &columns,
            256,
            42,
            1234,
            "match",
            &mut LocalFlow::new(&mut KeyFlow::new(Some(&context))),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    assert!((1..=COPY_VISITS).contains(&processed.load(Ordering::SeqCst)));
    assert_eq!(local.distinct, 256);
}

fn string_column(large: bool, count: usize, width: usize) -> ArrayRef {
    let values = vec!["x".repeat(width); count];
    if large {
        Arc::new(LargeStringArray::from(values))
    } else {
        Arc::new(StringArray::from(values))
    }
}

fn stop_after_work(
    phase: typed_key::KeyPhase,
    deadline: bool,
) -> (StreamJobContext, Arc<AtomicUsize>, typed_key::Observation) {
    let cancellation = CancellationToken::new();
    let token = cancellation.clone();
    let now = chrono::Utc::now();
    let limit = now + chrono::Duration::seconds(60);
    let (clock, time) = tokio::sync::watch::channel(now);
    let job = StreamJobContext::new(
        1,
        "local-stop",
        BTreeMap::new(),
        deadline.then_some(limit),
        cancellation,
    )
    .with_deadline_clock_for_test(time);
    let processed = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&processed);
    let hook = typed_key::observe(Arc::new(move |actual, bytes| {
        if actual == phase {
            observed.fetch_add(bytes.max(1), Ordering::SeqCst);
            if deadline {
                clock.send_replace(limit);
            } else {
                token.cancel();
            }
        }
    }));
    (job, processed, hook)
}

#[tokio::test]
async fn test_long_string_hash_stops_within_byte_quantum() {
    for large in [false, true] {
        for (count, width) in [(64, COPY_BYTES), (1, 3 * COPY_BYTES + 7)] {
            for deadline in [false, true] {
                let column = string_column(large, count, width);
                let columns = [LocalColumn::bind(column.as_ref()).unwrap()];
                let rows = admitted(vec![Arc::clone(&column)]);
                let mut local = local_table(count);
                let (job, processed, _hook) =
                    stop_after_work(typed_key::KeyPhase::LocalHash, deadline);
                let context = StreamOperatorContext::new(&job, "match", None);
                let error = local
                    .intern_group(
                        &columns,
                        AdmittedGroup::new(&rows, 0..count),
                        "match",
                        &mut KeyFlow::new(Some(&context)),
                    )
                    .await
                    .unwrap_err();
                assert!(matches!(error, CalcFlowError::Cancelled { .. }));
                let bytes = processed.load(Ordering::SeqCst);
                assert!(
                    bytes > 0 && bytes <= COPY_BYTES,
                    "large={large}, rows={count}: processed {bytes}"
                );
                assert!(
                    local.locals.is_empty(),
                    "stop hashing before interning the block"
                );
            }
        }
    }
}

#[tokio::test]
async fn test_long_and_composite_equality_stops_within_byte_quantum() {
    for large in [false, true] {
        // The second case crosses the quantum through two individually small
        // components, so resetting the byte allowance per column would fail.
        for (components, width) in [(1, 3 * COPY_BYTES + 7), (2, 3 * COPY_BYTES / 4)] {
            let arrays = (0..components)
                .map(|_| string_column(large, 2, width))
                .collect::<Vec<_>>();
            let columns = arrays
                .iter()
                .map(|array| LocalColumn::bind(array.as_ref()).unwrap())
                .collect::<Vec<_>>();
            for deadline in [false, true] {
                let mut local = local_table(2);
                seed_distinct(&mut local, 42, 0);
                let (job, processed, _hook) =
                    stop_after_work(typed_key::KeyPhase::LocalComparison, deadline);
                let context = StreamOperatorContext::new(&job, "match", None);
                let error = local
                    .intern_hashed(
                        42,
                        (1, components * width),
                        &columns,
                        "match",
                        &mut LocalFlow::new(&mut KeyFlow::new(Some(&context))),
                        |left, right| values_equal(&columns, left, right),
                    )
                    .await
                    .unwrap_err();
                assert!(matches!(error, CalcFlowError::Cancelled { .. }));
                let bytes = processed.load(Ordering::SeqCst);
                assert!(
                    bytes > 0 && bytes <= COPY_BYTES,
                    "processed {bytes} equality bytes"
                );
                assert_eq!(
                    local.distinct, 1,
                    "stopped equality must not insert a candidate"
                );
            }
        }
    }
}

#[tokio::test]
async fn test_collision_probe_stops_within_visit_quantum() {
    let values = (0..257).collect::<Vec<i64>>();
    let columns = [LocalColumn::I64(&values)];
    let mut local = local_table(values.len());
    for offset in 0..256 {
        seed_distinct(&mut local, 42, offset);
    }
    let (job, processed, _hook) = stop_after_work(typed_key::KeyPhase::LocalProbe, false);
    let context = StreamOperatorContext::new(&job, "match", None);
    let error = local
        .intern_positions(
            &columns,
            &[(256, true)],
            &[(42, 8)],
            "match",
            &mut LocalFlow::new(&mut KeyFlow::new(Some(&context))),
            |left, right| values[left] == values[right],
        )
        .await
        .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    assert!((1..=COPY_VISITS).contains(&processed.load(Ordering::SeqCst)));
    assert_eq!(local.distinct, 256);
}

#[tokio::test]
async fn test_small_collision_chain_resumes_after_allowance_exhaustion() {
    use std::{future::Future, task::Poll};
    let values = (0..512).collect::<Vec<i64>>();
    let columns = [LocalColumn::I64(&values)];
    let positions = (0..values.len()).map(|row| (row, true)).collect::<Vec<_>>();
    let hashes = vec![(42, 8); values.len()];
    let mut local = local_table(values.len());
    let job = StreamJobContext::new(
        1,
        "local-resume",
        BTreeMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut key_flow = KeyFlow::new(Some(&context));
    let mut flow = LocalFlow::new(&mut key_flow);
    let pending = {
        let mut future = Box::pin(local.intern_positions(
            &columns,
            &positions,
            &hashes,
            "match",
            &mut flow,
            |left, right| values[left] == values[right],
        ));
        let mut pending = 0;
        futures::future::poll_fn(|context| match future.as_mut().poll(context) {
            Poll::Pending => {
                pending += 1;
                Poll::Pending
            }
            Poll::Ready(result) => Poll::Ready(result),
        })
        .await
        .unwrap();
        pending
    };
    assert!(
        pending > 0,
        "exercise real cooperative Pending and continuation"
    );
    assert_eq!(local.locals, (0..512).collect::<Vec<u32>>());
    assert_eq!(local.retains, vec![1; 512]);
    assert_eq!(local.first_retained, (0..512).collect::<Vec<u32>>());
}

#[tokio::test]
async fn test_wide_composite_hash_stops_within_visit_quantum() {
    let values = [1_i8];
    let columns = (0..130)
        .map(|_| LocalColumn::I8(&values))
        .collect::<Vec<_>>();
    let (job, processed, _hook) = stop_after_work(typed_key::KeyPhase::LocalHash, false);
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut hashes = [(0, 0)];
    let mut key_flow = KeyFlow::new(Some(&context));
    let mut flow = LocalFlow::new(&mut key_flow);
    let result = async {
        for column in &columns {
            column
                .hash_column(&[(0, true)], &mut hashes, &mut flow)
                .await?;
        }
        Result::Ok(())
    }
    .await;
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert!((1..=COPY_VISITS).contains(&processed.load(Ordering::SeqCst)));
}

#[tokio::test]
async fn test_long_collision_equality_checks_tail_and_shares_identical_keys() {
    for large in [false, true] {
        let same = "é".repeat(COPY_BYTES + 1);
        let different = format!("{}xx", &same[..same.len() - 2]);
        let values = vec![same.clone(), different, same];
        let array: ArrayRef = if large {
            Arc::new(LargeStringArray::from(values))
        } else {
            Arc::new(StringArray::from(values))
        };
        let columns = [LocalColumn::bind(array.as_ref()).unwrap()];
        let mut local = local_table(3);
        let mut key_flow = KeyFlow::new(None);
        let mut flow = LocalFlow::new(&mut key_flow);
        seed_distinct(&mut local, 42, 0);
        for (row, expected) in [(1, 1), (2, 0)] {
            let id = local
                .intern_hashed(
                    42,
                    (row, 2 * COPY_BYTES + 2),
                    &columns,
                    "match",
                    &mut flow,
                    |left, right| values_equal(&columns, left, right),
                )
                .await
                .unwrap();
            assert_eq!(id, expected);
        }
        assert_eq!(local.distinct, 2);
    }
}

#[tokio::test]
async fn test_single_strings_compare_bytes_after_a_plain_hash_collision() {
    for large in [false, true] {
        let values = vec!["other", "right", "right"];
        let column: ArrayRef = if large {
            Arc::new(LargeStringArray::from(values))
        } else {
            Arc::new(StringArray::from(values))
        };
        let columns = [LocalColumn::bind(column.as_ref()).unwrap()];
        let rows = admitted(vec![Arc::clone(&column)]);
        let mut local = local_table(rows.len());
        let collision = hash_words(mix(0x243F_6A88_85A3_08D3, 5), b"right");
        // A different equal-width representative occupies the same full-hash bucket.
        seed_distinct(&mut local, collision, 0);
        local
            .intern_group(
                &columns,
                AdmittedGroup::new(&rows, 1..3),
                "match",
                &mut KeyFlow::new(None),
            )
            .await
            .unwrap();
        assert_eq!(local.locals, [1, 1]);
        assert_eq!(local.distinct, 2);
        assert_eq!(local.retains, [0, 2]);
    }
}

#[tokio::test]
async fn test_single_string_blocks_keep_generic_work_and_cancellation_boundaries() {
    use crate::operator::join::{join_work, reset_join_work};

    let wide = "x".repeat(COPY_BYTES + 1);
    let mut values = vec![""; COPY_VISITS + 1];
    values.extend(["é\0", &wide, "tail"]);
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(values.clone())),
        Arc::new(LargeStringArray::from(values)),
    ];
    for array in arrays {
        for cancelled in [false, true] {
            let token = CancellationToken::new();
            if cancelled {
                token.cancel();
            }
            let job = StreamJobContext::new(1, "intern", BTreeMap::new(), None, token);
            let context = StreamOperatorContext::new(&job, "intern", None);
            let mut rows = admitted(vec![Arc::clone(&array)]);
            for (index, row) in rows.iter_mut().enumerate() {
                row.retain = index % 3 == 0;
            }
            let columns = [LocalColumn::bind(array.as_ref()).unwrap()];
            let mut outcomes = Vec::new();
            for specialized in [false, true] {
                let mut local = local_table(rows.len());
                let mut flow = KeyFlow::new(Some(&context));
                let group = AdmittedGroup::new(&rows, 0..rows.len());
                reset_join_work();
                let result = if specialized {
                    local
                        .intern_group(&columns, group, "match", &mut flow)
                        .await
                } else {
                    local
                        .intern_blocks(
                            &columns,
                            group,
                            "match",
                            &mut LocalFlow::new(&mut flow),
                            |left, right| values_equal(&columns, left, right),
                        )
                        .await
                };
                assert_eq!(result.is_err(), cancelled);
                let work = join_work();
                outcomes.push((
                    local.locals,
                    local.rows,
                    local.hashes,
                    local.first_retained,
                    local.retains,
                    local.appearances,
                    work.quantum_steps,
                    work.quantum_boundaries,
                ));
            }
            assert_eq!(outcomes[0], outcomes[1]);
        }
    }
}

fn string_array(large: bool, values: Vec<String>) -> ArrayRef {
    if large {
        Arc::new(LargeStringArray::from(values))
    } else {
        Arc::new(StringArray::from(values))
    }
}

#[tokio::test]
async fn test_local_string_hash_charges_bounded_blocks_instead_of_values() {
    for large in [false, true] {
        for width in [0, 4, 15, 64] {
            let array = string_array(large, vec!["x".repeat(width); COPY_VISITS]);
            let column = LocalColumn::bind(array.as_ref()).unwrap();
            let positions = (0..COPY_VISITS).map(|row| (row, true)).collect::<Vec<_>>();
            let mut hashes = [(0, 0); COPY_VISITS];
            let mut key_flow = KeyFlow::new(None);
            let mut flow = LocalFlow::new(&mut key_flow);
            flow.visits = COPY_VISITS;
            flow.bytes = COPY_BYTES;
            column
                .hash_column(&positions, &mut hashes, &mut flow)
                .await
                .unwrap();
            assert!(
                flow.debits <= 2,
                "large={large}, width={width}: {} accounting calls for one bounded block",
                flow.debits
            );
        }
    }
}

fn reference_string_hash(seed: u64, value: &str) -> u64 {
    let mut hash = mix(seed, value.len() as u64);
    for chunk in value.as_bytes().chunks(8) {
        let mut word = [0; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        hash = mix(hash, u64::from_le_bytes(word));
    }
    hash
}

async fn check_string_hash_reference(array: ArrayRef, values: &[String]) {
    let column = LocalColumn::bind(array.as_ref()).unwrap();
    let positions = (0..values.len())
        .rev()
        .map(|row| (row, row % 2 == 0))
        .collect::<Vec<_>>();
    let mut hashes = positions
        .iter()
        .map(|&(row, _)| (row as u64, 17))
        .collect::<Vec<_>>();
    column
        .hash_column(
            &positions,
            &mut hashes,
            &mut LocalFlow::new(&mut KeyFlow::new(None)),
        )
        .await
        .unwrap();
    let expected = positions
        .iter()
        .map(|&(row, _)| {
            (
                reference_string_hash(row as u64, &values[row]),
                17 + values[row].len(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(hashes, expected);
}

#[tokio::test]
async fn test_local_string_hash_matches_word_reference_for_short_and_mixed_slices() {
    let mut values = (0..=16).map(|width| "x".repeat(width)).collect::<Vec<_>>();
    values.extend([
        "\0".to_owned(),
        "é\0tail".to_owned(),
        "12345678\0tail".to_owned(),
    ]);
    for large in [false, true] {
        let short = string_array(large, values.clone());
        check_string_hash_reference(
            short.slice(1, values.len() - 2),
            &values[1..values.len() - 1],
        )
        .await;
        let mut mixed = values.clone();
        mixed.extend(["x".repeat(COPY_BYTES), "é".repeat(COPY_BYTES + 1)]);
        check_string_hash_reference(string_array(large, mixed.clone()), &mixed).await;
    }
}

#[tokio::test]
async fn test_local_string_block_hash_stop_stays_within_visit_and_byte_quanta() {
    for large in [false, true] {
        for width in [15, 64] {
            for deadline in [false, true] {
                let array = string_column(large, COPY_VISITS * 2, width);
                let columns = [LocalColumn::bind(array.as_ref()).unwrap()];
                let rows = admitted(vec![Arc::clone(&array)]);
                let mut local = local_table(rows.len());
                let (job, processed, _hook) =
                    stop_after_work(typed_key::KeyPhase::LocalHash, deadline);
                let context = StreamOperatorContext::new(&job, "match", None);
                let error = local
                    .intern_group(
                        &columns,
                        AdmittedGroup::new(&rows, 0..rows.len()),
                        "match",
                        &mut KeyFlow::new(Some(&context)),
                    )
                    .await
                    .unwrap_err();
                assert!(matches!(error, CalcFlowError::Cancelled { .. }));
                assert_eq!(processed.load(Ordering::SeqCst), COPY_VISITS * width);
                assert!(local.locals.is_empty());
            }
        }
    }
}

#[tokio::test]
async fn test_local_string_hash_byte_refill_cannot_reuse_block_visit_allowance() {
    for large in [false, true] {
        let array = string_column(large, COPY_VISITS, 4);
        let columns = [LocalColumn::bind(array.as_ref()).unwrap()];
        let (job, _, _hook) = stop_after_work(typed_key::KeyPhase::LocalHash, false);
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut key_flow = KeyFlow::new(Some(&context));
        let mut flow = LocalFlow::new(&mut key_flow);
        flow.step(0, COPY_BYTES).await.unwrap();
        let positions = (0..COPY_VISITS).map(|row| (row, true)).collect::<Vec<_>>();
        let mut hashes = [(0x243F_6A88_85A3_08D3, 0); COPY_VISITS];
        columns[0]
            .hash_column(&positions, &mut hashes, &mut flow)
            .await
            .unwrap();
        let mut local = local_table(COPY_VISITS);
        let error = local
            .intern_positions(
                &columns,
                &positions,
                &hashes,
                "match",
                &mut flow,
                |left, right| values_equal(&columns, left, right),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, CalcFlowError::Cancelled { .. }));
        assert!(
            local.locals.len() <= 1,
            "a byte refill must not grant a second block of visits after hashing"
        );
    }
}

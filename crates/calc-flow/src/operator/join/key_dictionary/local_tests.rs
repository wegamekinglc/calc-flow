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

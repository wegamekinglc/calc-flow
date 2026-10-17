use super::super::{AdmittedBatch, JoinStateLimits, StreamJoinSpec};
use super::*;
use crate::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, JsonMap, OperatorMetadata,
    StreamJobContext, StreamOperator, StreamOperatorContext,
};
use datafusion::arrow::{
    array::{Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
use std::time::Duration;

fn window_left_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("account_id", DataType::Int64, false),
        Field::new(
            "authorized_at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("amount", DataType::Int64, true),
    ]))
}

fn window_right_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("account_id", DataType::Int64, false),
        Field::new(
            "paid_at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("status", DataType::Int64, true),
    ]))
}

fn window_spec() -> StreamJoinSpec {
    StreamJoinSpec::inner(
        ["account_id"],
        ["account_id"],
        "authorized_at",
        "paid_at",
        JoinTimeBounds::new(Duration::from_micros(10), Duration::from_micros(20)).unwrap(),
        JoinStateLimits::new(100, 1_000_000, 1_000).unwrap(),
    )
    .unwrap()
}

fn window_left_batch(times: &[i64]) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                window_left_schema(),
                vec![
                    Arc::new(Int64Array::from(vec![7; times.len()])),
                    Arc::new(TimestampMicrosecondArray::from(times.to_vec()).with_timezone("UTC")),
                    Arc::new(Int64Array::from(vec![42; times.len()])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn window_right_batch(times: &[i64]) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                window_right_schema(),
                vec![
                    Arc::new(Int64Array::from(vec![7; times.len()])),
                    Arc::new(TimestampMicrosecondArray::from(times.to_vec()).with_timezone("UTC")),
                    Arc::new(Int64Array::from(vec![9; times.len()])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn window_job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "fingerprint",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

async fn window_admitted_probe(
    operator: &mut StreamJoinOperator,
    ingress: &str,
    batch: &Batch,
    context: &StreamOperatorContext<'_>,
) -> (SidePlan, AdmittedBatch) {
    let plan = operator.begin_batch(ingress, batch).unwrap();
    let mut bundle = operator.admission_bundle(&plan, batch).unwrap();
    operator
        .admit_record(
            &batch.table_payload().unwrap().batches()[0],
            &plan,
            ingress,
            context,
            &mut bundle,
            0,
        )
        .await
        .unwrap();
    bundle.finish(&operator.name).unwrap();
    (plan, bundle.admitted)
}

fn pair_identities(pairs: impl Iterator<Item = MatchedPair>) -> Vec<(usize, usize)> {
    pairs.map(|pair| (pair.pos, pair.opposite_index)).collect()
}

// One key run with duplicate times, event-time-ordered probe rows, then one
// out-of-order row, then one row that only advances from the out-of-order
// cursor: the count and fill passes together must perform full-run binary
// searches only for the first touch and the out-of-order fallback, not four
// searches per probed row.
#[tokio::test]
async fn test_serial_window_probe_reuses_count_searches_for_fill() {
    let mut operator = StreamJoinOperator::new(
        "match",
        window_left_schema(),
        window_right_schema(),
        window_spec(),
    )
    .unwrap();
    let job = window_job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let mut right_times = Vec::new();
    for index in 0..40_i64 {
        let time = index * 3;
        right_times.push(time);
        if index % 4 == 0 {
            right_times.push(time);
        }
    }
    operator
        .process_data(
            "right",
            window_right_batch(&right_times),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let mut probe_times: Vec<i64> = (0..31).map(|index| index * 10).collect();
    probe_times.extend([5, 15]);
    let (plan, admitted) = window_admitted_probe(
        &mut operator,
        "left",
        &window_left_batch(&probe_times),
        &context,
    )
    .await;
    let mut admitted = admitted;
    let reference = operator
        .legacy_matches(&plan, &mut admitted, &context)
        .await
        .unwrap();
    let expected = pair_identities(reference.iter().copied());
    assert!(!expected.is_empty());
    reset_window_run_searches();
    let native = operator
        .native_matches(&plan, &admitted, None)
        .unwrap()
        .expect("single Int64 key stays native");
    assert_eq!(pair_identities(native.pairs.iter()), expected);
    assert_eq!(
        window_run_searches(),
        4,
        "one first touch plus one out-of-order fallback must be the only full-run searches"
    );
}

fn fixture_index(times_per_key: &[&[i64]]) -> (NativeIndex, Vec<u32>) {
    let record = RecordBatch::new_empty(Arc::new(Schema::empty()));
    let mut rows = Vec::new();
    for (key, times) in times_per_key.iter().enumerate() {
        for (index, &time) in times.iter().enumerate() {
            rows.push(StoredRow {
                record: record.clone().into(),
                event_time: EventTime::from_micros(time),
                row_id: u64::try_from(index).expect("fixture row ID") + (key as u64) * 1_000,
                charge: 0,
                encoded_key: Arc::new(vec![u8::try_from(key).expect("fixture key")].into()),
            });
        }
    }
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let credit = MemoryConsumer::new("window-probe-fixture").register(&pool);
    credit
        .try_grow(NativeIndex::build_charge(rows.len()).expect("fixture charge"))
        .unwrap();
    let index = NativeIndex::new(&rows, credit);
    let ids = times_per_key
        .iter()
        .scan(0_usize, |offset, times| {
            let id = index
                .key_id(&rows[*offset].encoded_key)
                .expect("fixture key ID");
            *offset += times.len();
            Some(id)
        })
        .collect::<Vec<_>>();
    (index, ids)
}

fn assert_advanced_bounds_match_plain_window(
    index: &NativeIndex,
    id: u32,
    cursors: &mut [Option<WindowCursor>],
    key: usize,
    low: i64,
    high: i64,
) {
    let range = (EventTime::from_micros(low), EventTime::from_micros(high));
    let bounds = index.window_bounds_by_id(id, range, cursors[key]);
    cursors[key] = Some(WindowCursor::new(range, bounds));
    let advanced = index.range_by_bounds(id, bounds).collect::<Vec<_>>();
    let plain = index.range_by_id(id, range).collect::<Vec<_>>();
    assert_eq!(advanced, plain, "key {key} range {low}..={high}");
}

#[test]
fn test_window_cursor_bounds_match_plain_windows_on_edge_probe_sequences() {
    let key0 = [
        0, 3, 3, 7, 7, 7, 11, 15, 15, 19, 22, 22, 26, 30, 30, 34, 34, 38, 41, 45, 45, 48, 52, 52,
    ];
    let key1 = [
        1, 2, 9, 9, 13, 17, 17, 21, 25, 28, 28, 33, 33, 36, 40, 44, 44, 47, 51, 55, 55, 58, 62, 62,
    ];
    let (index, ids) = fixture_index(&[&key0, &key1]);
    let mut cursors = [None, None];
    // Ordered advances, duplicate windows, boundary-inclusive lows and highs,
    // interleaved keys with independent cursors, out-of-order regressions,
    // both run edges, and empty windows inside gaps and past the run end.
    let probes: &[(usize, i64, i64)] = &[
        (0, 2, 8),
        (1, 0, 10),
        (0, 3, 9),
        (0, 3, 9),
        (1, 8, 30),
        (0, 4, 12),
        (0, 0, 3),
        (0, 26, 52),
        (1, 10, 63),
        (0, 23, 25),
        (0, 53, 60),
        (1, 0, 1),
        (0, 30, 30),
        (0, 5, 53),
        (1, 20, 21),
    ];
    for &(key, low, high) in probes {
        assert_advanced_bounds_match_plain_window(&index, ids[key], &mut cursors, key, low, high);
    }
}

#[test]
fn test_window_cursor_bounds_match_plain_windows_on_seeded_probe_sweeps() {
    let first: Vec<i64> = (0..97)
        .map(|index: i64| 8 * (index / 2) + i64::from(index % 7 == 3) * 4)
        .collect();
    let second: Vec<i64> = first.iter().map(|&time| time + 3).collect();
    let (index, ids) = fixture_index(&[&first, &second]);
    let mut cursors = [None, None];
    let mut seed = 0x7769_6e64_6f77_u64;
    for _ in 0..512 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let key = usize::try_from(seed & 1).unwrap();
        let low = i64::try_from((seed >> 17) % 400).unwrap() - 8;
        let high = low + i64::try_from((seed >> 33) % 40).unwrap();
        assert_advanced_bounds_match_plain_window(&index, ids[key], &mut cursors, key, low, high);
    }
}

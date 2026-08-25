//! Stage-3 property tests for the bounded event-time stream Join (spec AC7,
//! FR18/FR19): random batch partitions and legal ingress interleavings must
//! yield the same on-time output pair multiset as an offline reference.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, EventTime, IngressProgress,
    IngressProgressSnapshot, IngressState, JoinStateLimits, JoinTimeBounds, JsonMap,
    OperatorMetadata, StreamJobContext, StreamJoinOperator, StreamJoinSpec, StreamOperator,
    StreamOperatorContext,
};
use datafusion::arrow::array::{StringArray, TimestampMicrosecondArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use datafusion::arrow::record_batch::RecordBatch;
use proptest::collection::vec;
use proptest::prelude::*;

const PROPERTY_CASES: u32 = 32;
const PROPERTY_SEED: u64 = 0xCA1C_F10A_0000_0015;
const SECOND: i64 = 1_000_000;

fn property_config() -> ProptestConfig {
    ProptestConfig {
        cases: PROPERTY_CASES,
        failure_persistence: None,
        rng_algorithm: proptest::test_runner::RngAlgorithm::ChaCha,
        rng_seed: proptest::test_runner::RngSeed::Fixed(PROPERTY_SEED),
        ..ProptestConfig::default()
    }
}

/// One generated side row: a small key alphabet keeps collisions and duplicate
/// multiplicities likely, so partition invariance is not only tested on
/// disjoint keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SideRow {
    key: char,
    ts: i64,
}

fn side_rows() -> BoxedStrategy<Vec<SideRow>> {
    vec(
        (
            prop_oneof![Just('a'), Just('b'), Just('c'), Just('d')],
            0..40_i64,
        ),
        0..12,
    )
    .prop_map(|rows| {
        rows.into_iter()
            .map(|(key, ts)| SideRow { key, ts })
            .collect()
    })
    .boxed()
}

fn bounds_seconds() -> BoxedStrategy<(u64, u64)> {
    (0..=8_u64, 0..=8_u64).boxed()
}

fn chunk_sizes() -> BoxedStrategy<Vec<usize>> {
    vec(1_usize..=3, 0..12).boxed()
}

fn permutations() -> BoxedStrategy<Vec<Vec<u32>>> {
    vec(vec(any::<u32>(), 0..3), 0..12).boxed()
}

fn picks() -> BoxedStrategy<Vec<bool>> {
    vec(any::<bool>(), 0..24).boxed()
}

/// Splits one sorted side log into random consecutive batches and shuffles
/// rows within each batch; a batch's emitted watermark is its maximum event
/// time, which keeps every row on-time because batches partition a sorted log.
fn partition_side(
    rows: &[SideRow],
    chunk_sizes: &[usize],
    within_batch_sort_keys: &[Vec<u32>],
) -> Vec<(Vec<SideRow>, i64)> {
    let mut batches = Vec::new();
    let mut index = 0;
    let sizes = chunk_sizes.iter().copied().chain(std::iter::repeat(1));
    for (ordinal, size) in sizes.enumerate() {
        if index >= rows.len() {
            break;
        }
        let end = (index + size.max(1)).min(rows.len());
        let mut batch: Vec<SideRow> = rows[index..end].to_vec();
        // Sort keys permute rows by their relative order; rows beyond the
        // generated keys keep their input order, so every shuffle stays a
        // valid permutation of the batch (FR18 physical-order freedom).
        if let Some(keys) = within_batch_sort_keys.get(ordinal) {
            let mut keyed = batch
                .iter()
                .zip(keys.iter().chain(std::iter::repeat(&0)))
                .collect::<Vec<_>>();
            keyed.sort_by_key(|(row, key)| (**key, row.ts));
            batch = keyed.into_iter().map(|(row, _)| *row).collect();
        }
        let watermark = batch.iter().map(|row| row.ts).max().unwrap_or(0);
        batches.push((batch, watermark));
        index = end;
    }
    batches
}

fn row_batch(rows: &[SideRow]) -> Batch {
    let schema = join_schema();
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.key.to_string())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(TimestampMicrosecondArray::from(
                rows.iter().map(|row| row.ts * SECOND).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn join_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
    ]))
}

/// One legal interleaving step: a data batch on one side followed by that
/// side's watermark update.
#[derive(Debug)]
enum Step {
    Left(Vec<SideRow>, i64),
    Right(Vec<SideRow>, i64),
}

/// Merges the two per-side batch sequences into one legal schedule while
/// keeping each side's internal order.
fn interleave(
    left: Vec<(Vec<SideRow>, i64)>,
    right: Vec<(Vec<SideRow>, i64)>,
    picks_left: &[bool],
) -> Vec<Step> {
    let mut steps = Vec::new();
    let mut left = left.into_iter().peekable();
    let mut right = right.into_iter().peekable();
    let mut picks = picks_left.iter().copied();
    loop {
        let take_left = match picks.next() {
            Some(take_left) => take_left,
            None => match (left.peek(), right.peek()) {
                (Some(_), _) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            },
        };
        match (take_left, left.peek().is_some(), right.peek().is_some()) {
            (true, true, _) | (_, true, false) => {
                let (batch, watermark) = left.next().unwrap();
                steps.push(Step::Left(batch, watermark));
            }
            (_, _, true) => {
                let (batch, watermark) = right.next().unwrap();
                steps.push(Step::Right(batch, watermark));
            }
            (_, false, false) => break,
        }
    }
    steps
}

/// The offline reference (spec AC7): every pair of rows with equal keys and
/// `right.ts` inside the inclusive `[left.ts - before, left.ts + after]`
/// interval, as a multiset of `(key, left.ts, right.ts)`.
fn reference_pairs(
    left: &[SideRow],
    right: &[SideRow],
    before_seconds: u64,
    after_seconds: u64,
) -> BTreeMap<(char, i64, i64), usize> {
    let mut pairs = BTreeMap::new();
    for left_row in left {
        for right_row in right {
            if left_row.key != right_row.key {
                continue;
            }
            let delta = right_row.ts - left_row.ts;
            let before = i64::try_from(before_seconds).expect("bounds fit i64");
            let after = i64::try_from(after_seconds).expect("bounds fit i64");
            if delta >= -before && delta <= after {
                *pairs
                    .entry((left_row.key, left_row.ts, right_row.ts))
                    .or_insert(0) += 1;
            }
        }
    }
    pairs
}

/// Both-side progress snapshot with the given watermarks in whole seconds.
fn both_sides(left: Option<i64>, right: Option<i64>) -> IngressProgressSnapshot {
    let to_progress = |watermark: Option<i64>| {
        IngressProgress::new(
            IngressState::Active,
            watermark.map(|value| EventTime::from_micros(value * SECOND)),
        )
    };
    IngressProgressSnapshot::new(BTreeMap::from([
        ("left".into(), to_progress(left)),
        ("right".into(), to_progress(right)),
    ]))
}

/// Drives one generated schedule through a real `StreamJoinOperator` and
/// returns its emitted pair multiset.
async fn emitted_pairs(bounds: (u64, u64), steps: &[Step]) -> BTreeMap<(char, i64, i64), usize> {
    let schema = join_schema();
    let mut operator = StreamJoinOperator::new(
        "match",
        Arc::clone(&schema),
        schema,
        StreamJoinSpec::inner(
            ["key"],
            ["key"],
            "ts",
            "ts",
            JoinTimeBounds::new(Duration::from_secs(bounds.0), Duration::from_secs(bounds.1))
                .unwrap(),
            JoinStateLimits::new(100_000, 134_217_728, 1_000_000).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let job = StreamJobContext::new(
        1,
        "fingerprint",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let mut left_watermark: Option<i64> = None;
    let mut right_watermark: Option<i64> = None;
    for step in steps {
        let (ingress, rows, watermark) = match step {
            Step::Left(rows, watermark) => ("left", rows, watermark),
            Step::Right(rows, watermark) => ("right", rows, watermark),
        };
        let pre_data = both_sides(left_watermark, right_watermark);
        let context = StreamOperatorContext::with_ingress_progress(&job, "match", None, pre_data);
        operator
            .process_data(ingress, row_batch(rows), &context, &mut collector)
            .await
            .unwrap();
        match step {
            Step::Left(_, _) => left_watermark = Some(*watermark),
            Step::Right(_, _) => right_watermark = Some(*watermark),
        }
        let post_data = both_sides(left_watermark, right_watermark);
        let context = StreamOperatorContext::with_ingress_progress(&job, "match", None, post_data);
        operator
            .on_ingress_progress(ingress, &context)
            .await
            .unwrap();
    }
    let mut emitted = BTreeMap::new();
    for message in collector.drain("output") {
        let batch = message.as_data().expect("the Join emits data batches only");
        for record in batch.table_payload().unwrap().batches() {
            let keys = record
                .column_by_name("left__key")
                .expect("prefixed left key column")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("utf8 key column");
            let left_times = record
                .column_by_name("left__ts")
                .expect("prefixed left event-time column")
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .expect("microsecond event-time column");
            let right_times = record
                .column_by_name("right__ts")
                .expect("prefixed right event-time column")
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .expect("microsecond event-time column");
            for index in 0..record.num_rows() {
                *emitted
                    .entry((
                        keys.value(index).chars().next().unwrap_or('?'),
                        left_times.value(index) / SECOND,
                        right_times.value(index) / SECOND,
                    ))
                    .or_insert(0) += 1;
            }
        }
    }
    emitted
}

fn sorted_log(rows: &[SideRow]) -> Vec<SideRow> {
    let mut sorted = rows.to_vec();
    sorted.sort_by_key(|row| row.ts);
    sorted
}

proptest! {
    #![proptest_config(property_config())]
    #[test]
    fn random_partitions_and_legal_interleavings_match_the_offline_reference(
        left_rows in side_rows(),
        right_rows in side_rows(),
        bounds in bounds_seconds(),
        left_chunks in chunk_sizes(),
        right_chunks in chunk_sizes(),
        left_permutations in permutations(),
        right_permutations in permutations(),
        picks_left in picks(),
    ) {
        // Arrival order per side is the sorted event-time order, so the
        // per-batch maximum watermark keeps every generated row on-time and
        // every interval-compatible pair probeable (FR19).
        let left_sorted = sorted_log(&left_rows);
        let right_sorted = sorted_log(&right_rows);
        let steps = interleave(
            partition_side(&left_sorted, &left_chunks, &left_permutations),
            partition_side(&right_sorted, &right_chunks, &right_permutations),
            &picks_left,
        );
        let expected = reference_pairs(&left_sorted, &right_sorted, bounds.0, bounds.1);

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let emitted = runtime.block_on(emitted_pairs(bounds, &steps));

        prop_assert_eq!(
            emitted,
            expected,
            "steps: {:?} sorted-left: {:?} sorted-right: {:?}",
            steps,
            left_sorted,
            right_sorted
        );
    }

    #[test]
    fn partition_invariance_holds_across_independent_repartitions(
        left_rows in side_rows(),
        right_rows in side_rows(),
        bounds in bounds_seconds(),
        first_left_chunks in chunk_sizes(),
        first_right_chunks in chunk_sizes(),
        second_left_chunks in chunk_sizes(),
        second_right_chunks in chunk_sizes(),
        first_left_permutations in permutations(),
        first_right_permutations in permutations(),
        second_left_permutations in permutations(),
        second_right_permutations in permutations(),
        first_picks in picks(),
        second_picks in picks(),
    ) {
        let left_sorted = sorted_log(&left_rows);
        let right_sorted = sorted_log(&right_rows);
        let first = interleave(
            partition_side(&left_sorted, &first_left_chunks, &first_left_permutations),
            partition_side(&right_sorted, &first_right_chunks, &first_right_permutations),
            &first_picks,
        );
        let second = interleave(
            partition_side(&left_sorted, &second_left_chunks, &second_left_permutations),
            partition_side(&right_sorted, &second_right_chunks, &second_right_permutations),
            &second_picks,
        );

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let first_emitted = runtime.block_on(emitted_pairs(bounds, &first));
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let second_emitted = runtime.block_on(emitted_pairs(bounds, &second));

        prop_assert_eq!(first_emitted, second_emitted);
    }
}

/// Pins the offline reference so the property above cannot pass against a
/// degenerate oracle: inclusive boundaries match and duplicates multiply.
#[test]
fn reference_pairs_count_inclusive_boundaries_and_duplicates() {
    let left = vec![SideRow { key: 'a', ts: 10 }, SideRow { key: 'a', ts: 10 }];
    let right = vec![
        SideRow { key: 'a', ts: 5 },
        SideRow { key: 'a', ts: 15 },
        SideRow { key: 'b', ts: 10 },
    ];
    // before=5, after=5: every left/right `a` pair lies inside the inclusive
    // interval, so duplicates multiply to 2 * 2 = 4 pairs; `b` never matches.
    let pairs = reference_pairs(&left, &right, 5, 5);
    assert_eq!(pairs.get(&('a', 10, 5)), Some(&2));
    assert_eq!(pairs.get(&('a', 10, 15)), Some(&2));
    assert_eq!(pairs.values().sum::<usize>(), 4);
    let keys: BTreeSet<_> = pairs.keys().collect();
    assert!(keys.iter().all(|(key, _, _)| *key == 'a'));
}

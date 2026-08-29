//! Cross-section segmentation/interleaving/tie/null property matrix: every
//! result is independent of micro-batch boundaries, arrival order, and the
//! batch vs stream lifecycle (SCE-09 acceptance).

use std::sync::Arc;

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, CrossSectionOperator, CrossSectionSpec, EdgeCollector,
    EventTime, JsonMap, OperatorMetadata, StreamJobContext, StreamOperator, StreamOperatorContext,
};
use datafusion::arrow::{
    array::{Array, ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

const FINGERPRINT: &str = "1032547698badcfe1032547698badcfe1032547698badcfe1032547698badcfe";

fn input_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("industry", DataType::Utf8, true),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("momentum_20", DataType::Float64, true),
    ]))
}

type InputRow = (i64, &'static str, Option<&'static str>, u64, Option<f64>);

fn input_batch(rows: &[InputRow]) -> Batch {
    let record = RecordBatch::try_new(
        input_schema(),
        vec![
            Arc::new(
                TimestampMicrosecondArray::from(rows.iter().map(|row| row.0).collect::<Vec<_>>())
                    .with_timezone("UTC"),
            ) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter().map(|row| row.1).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter().map(|row| row.2).collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from(
                rows.iter().map(|row| row.3).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|row| row.4).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn spec() -> CrossSectionSpec {
    serde_json::from_value(serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "event_time": "ts",
        "entity_by": ["symbol"],
        "partition_by": ["industry"],
        "sequence_by": ["sequence"],
        "grouping": {"kind": "exact_time"},
        "outputs": [
            {
                "kind": "rank",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "momentum_rank",
                "direction": "ascending",
                "tie_method": "average",
                "null_placement": "exclude",
                "min_samples": 1
            },
            {
                "kind": "percentile",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "momentum_pct",
                "direction": "ascending",
                "tie_method": "max",
                "null_placement": "last",
                "min_samples": 1
            },
            {
                "kind": "demean",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "momentum_demean",
                "min_samples": 2
            },
            {
                "kind": "zscore",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "momentum_z",
                "min_samples": 2,
                "ddof": 1
            }
        ],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "nan_exclude_preserve_v1"
    }))
    .unwrap()
}

#[derive(Debug, Default, Clone)]
struct Observed {
    symbols: Vec<String>,
    industries: Vec<bool>,
    ranks: Vec<Option<f64>>,
    percentiles: Vec<Option<f64>>,
    demeans: Vec<Option<f64>>,
    zscores: Vec<Option<f64>>,
}

/// Float comparison under the frozen 1e-10 cross-section tolerance with NaN
/// equal to NaN (SCE-00 tolerance table); batch and stream lifecycles share
/// one kernel, so any difference exceeds it.
fn same_column(left: &[Option<f64>], right: &[Option<f64>]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| match (left, right) {
                (Some(left), Some(right)) => {
                    (left - right).abs() < 1e-10 || (left.is_nan() && right.is_nan())
                }
                (None, None) => true,
                _ => false,
            })
}

impl PartialEq for Observed {
    fn eq(&self, other: &Self) -> bool {
        self.symbols == other.symbols
            && self.industries == other.industries
            && same_column(&self.ranks, &other.ranks)
            && same_column(&self.percentiles, &other.percentiles)
            && same_column(&self.demeans, &other.demeans)
            && same_column(&self.zscores, &other.zscores)
    }
}

fn observe(batch: &Batch, observed: &mut Observed) {
    for record in batch.table_payload().unwrap().batches() {
        let symbols = record
            .column_by_name("symbol")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let industries = record
            .column_by_name("industry")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for (name, target) in [
            ("momentum_rank", &mut observed.ranks),
            ("momentum_pct", &mut observed.percentiles),
            ("momentum_demean", &mut observed.demeans),
            ("momentum_z", &mut observed.zscores),
        ] {
            let column = record
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            for index in 0..column.len() {
                target.push(if column.is_null(index) {
                    None
                } else {
                    Some(column.value(index))
                });
            }
        }
        for index in 0..symbols.len() {
            observed.symbols.push(symbols.value(index).to_owned());
            observed.industries.push(!industries.is_null(index));
        }
    }
}

/// The property-matrix fixture: ties, nulls, NaNs, interleaved partitions,
/// duplicate values across groups, and singleton groups.
fn matrix_rows() -> Vec<InputRow> {
    vec![
        (100, "a", Some("tech"), 1, Some(2.0)),
        (100, "b", Some("tech"), 2, Some(2.0)),
        (100, "c", Some("tech"), 3, Some(1.0)),
        (100, "d", Some("tech"), 4, None),
        (100, "e", Some("tech"), 5, Some(3.0)),
        (100, "f", Some("fin"), 1, Some(2.0)),
        (100, "g", Some("fin"), 2, None),
        (200, "a", Some("tech"), 1, Some(5.0)),
        (200, "b", Some("tech"), 2, Some(5.0)),
        (200, "c", Some("tech"), 3, Some(7.0)),
        (200, "d", Some("tech"), 4, Some(f64::NAN)),
        (300, "h", Some("fin"), 1, Some(1.0)),
    ]
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

async fn batch_reference(rows: &[InputRow]) -> Observed {
    let mut operator = CrossSectionOperator::new("cross_section", input_schema(), spec()).unwrap();
    let job = job();
    let mut collected = EdgeCollector::new(operator.output_ports().to_vec());
    let context = StreamOperatorContext::new(&job, "cross_section", None);
    let mut observed = Observed::default();
    // The stream lifecycle with one whole envelope and an immediate
    // end-of-input is the complete-group batch evaluation.
    operator
        .process_data("input", input_batch(rows), &context, &mut collected)
        .await
        .unwrap();
    operator.on_end(&context, &mut collected).await.unwrap();
    for message in collected.drain("output") {
        observe(message.as_data().unwrap(), &mut observed);
    }
    observed
}

async fn stream_reference(segments: &[&[InputRow]]) -> Observed {
    let mut operator = CrossSectionOperator::new("cross_section", input_schema(), spec()).unwrap();
    let job = job();
    let mut collected = EdgeCollector::new(operator.output_ports().to_vec());
    let mut observed = Observed::default();
    let mut watermark: Option<i64> = None;
    for (index, segment) in segments.iter().enumerate() {
        let context = StreamOperatorContext::new(
            &job,
            "cross_section",
            watermark.map(EventTime::from_micros),
        );
        operator
            .process_data("input", input_batch(segment), &context, &mut collected)
            .await
            .unwrap();
        // Advance only far enough to keep every later segment's rows on
        // time: one microsecond below the earliest remaining event time.
        let remaining = segments[index + 1..]
            .iter()
            .flat_map(|rows| rows.iter().map(|row| row.0))
            .min()
            .map(|earliest| earliest - 1);
        if let Some(target) = remaining {
            let context = StreamOperatorContext::new(
                &job,
                "cross_section",
                Some(EventTime::from_micros(target)),
            );
            operator
                .on_watermark(EventTime::from_micros(target), &context, &mut collected)
                .await
                .unwrap();
            watermark = Some(target);
        }
    }
    let context =
        StreamOperatorContext::new(&job, "cross_section", watermark.map(EventTime::from_micros));
    operator.on_end(&context, &mut collected).await.unwrap();
    for message in collected.drain("output") {
        observe(message.as_data().unwrap(), &mut observed);
    }
    observed
}

#[tokio::test]
async fn matrix_results_are_identical_across_segmentations_interleavings_and_lifecycles() {
    let rows = matrix_rows();
    let reference = batch_reference(&rows).await;

    // Deterministic arrival permutations of the same logical input.
    let mut reversed = rows.clone();
    reversed.reverse();
    let rotated = (0..rows.len())
        .map(|index| rows[(index + 5) % rows.len()])
        .collect::<Vec<_>>();
    for permutation in [rows.clone(), reversed, rotated] {
        assert_eq!(batch_reference(&permutation).await, reference);
    }

    // Stream segmentation: one logical envelope, per-time segmentation, and
    // split groups all converge on the reference.
    let whole: Vec<&[InputRow]> = vec![&rows];
    assert_eq!(stream_reference(&whole).await, reference);

    let grouped: Vec<Vec<InputRow>> = vec![
        rows.iter().filter(|row| row.0 == 100).copied().collect(),
        rows.iter().filter(|row| row.0 == 200).copied().collect(),
        rows.iter().filter(|row| row.0 == 300).copied().collect(),
    ];
    let grouped: Vec<&[InputRow]> = grouped.iter().map(Vec::as_slice).collect();
    assert_eq!(stream_reference(&grouped).await, reference);

    let split: Vec<&[InputRow]> = vec![&rows[..5], &rows[5..9], &rows[9..]];
    assert_eq!(stream_reference(&split).await, reference);
}

#[tokio::test]
async fn matrix_reference_values_match_the_frozen_semantics() {
    let reference = batch_reference(&matrix_rows()).await;
    // Canonical order: group (100, fin) first, then (100, tech), then
    // (200, tech), then (300, fin); rows within a group stay canonical.
    assert_eq!(
        reference.symbols,
        vec!["f", "g", "a", "b", "c", "d", "e", "a", "b", "c", "d", "h"]
    );

    // rank (average tie, exclude): (100, fin): f=1, g=null;
    // (100, tech): c=1, a=b=2.5, e=4, d=null; (200, tech): a=b=1.5, c=3,
    // the NaN row carries NaN; (300, fin): h=1.
    let expected_ranks = [
        Some(1.0),
        None,
        Some(2.5),
        Some(2.5),
        Some(1.0),
        None,
        Some(4.0),
        Some(1.5),
        Some(1.5),
        Some(3.0),
        Some(f64::NAN),
        Some(1.0),
    ];
    assert!(same_column(&reference.ranks, &expected_ranks));
    assert!(reference.ranks[10].is_some_and(f64::is_nan));

    // percentile (max tie, nulls last): (100, fin) count 2: f=1 -> 0,
    // g=2 -> 1; (100, tech) count 5: c=1 -> 0, a=b=3 -> 1/2, e=4 -> 3/4,
    // d=5 -> 1; (200, tech) count 3: a=b=2 -> 1/2, c=3 -> 1, NaN row NaN;
    // (300, fin) count 1: h -> 0.5.
    let expected_percentiles = [
        Some(0.0),
        Some(1.0),
        Some(0.5),
        Some(0.5),
        Some(0.0),
        Some(1.0),
        Some(0.75),
        Some(0.5),
        Some(0.5),
        Some(1.0),
        Some(f64::NAN),
        Some(0.5),
    ];
    assert!(same_column(&reference.percentiles, &expected_percentiles));
    assert!(reference.percentiles[10].is_some_and(f64::is_nan));

    // demean/zscore with min_samples 2: groups below two valid samples stay
    // null; the (200, tech) NaN row preserves NaN for both statistics.
    let expected_demean = [
        (0_usize, None), // f: one valid sample
        (1, None),       // g: null preserved
        (2, Some(0.0)),
        (3, Some(0.0)),
        (4, Some(-1.0)),
        (5, None), // d: null preserved
        (6, Some(1.0)),
        (7, Some(-2.0 / 3.0)),
        (8, Some(-2.0 / 3.0)),
        (9, Some(4.0 / 3.0)),
        (11, None), // h: one valid sample
    ];
    for (index, expected) in expected_demean {
        match (reference.demeans[index], expected) {
            (Some(actual), Some(expected)) => assert!(
                (actual - expected).abs() < 1e-10,
                "demeans[{index}] = {actual}, expected {expected}"
            ),
            (actual, expected) => assert_eq!(
                actual.is_none(),
                expected.is_none(),
                "demeans[{index}] = {actual:?}"
            ),
        }
    }
    assert!(reference.demeans[10].is_some_and(f64::is_nan));
    let sample_std = (4.0_f64 / 3.0).sqrt();
    for (index, expected) in [
        (7_usize, -1.0 / (3.0_f64).sqrt()),
        (8, -1.0 / (3.0_f64).sqrt()),
        (9, 2.0 / (3.0_f64).sqrt()),
    ] {
        let value = reference.zscores[index].expect("ddof=1 keeps the group eligible");
        assert!(
            (value - expected).abs() < 1e-12,
            "zscores[{index}] = {value}, expected {expected} (std {sample_std})"
        );
    }
    for index in [0_usize, 1, 5, 11] {
        assert_eq!(
            reference.zscores[index], None,
            "zscores[{index}] should be null"
        );
    }
    for index in [2_usize, 3, 4, 6] {
        let value = reference.zscores[index]
            .unwrap_or_else(|| panic!("zscores[{index}] should carry a value"));
        assert!(value.is_finite(), "zscores[{index}] = {value}");
    }
    assert!(reference.zscores[10].is_some_and(f64::is_nan));
}

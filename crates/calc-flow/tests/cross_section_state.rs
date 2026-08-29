//! Cross-section checkpoint semantics: half-built groups survive the aligned
//! epoch cut and restored execution is equivalent to uninterrupted execution
//! (SCE-00 D11, SCE-09).

use std::sync::Arc;

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, CrossSectionOperator, CrossSectionSpec, EdgeCollector,
    Epoch, EventTime, JsonMap, OperatorMetadata, StreamJobContext, StreamOperator,
    StreamOperatorContext,
};
use datafusion::arrow::{
    array::{Array, ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

const FINGERPRINT: &str = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

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
                "kind": "zscore",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "momentum_z",
                "min_samples": 1,
                "ddof": 0
            }
        ],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "nan_exclude_preserve_v1"
    }))
    .unwrap()
}

fn new_job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn context(job: &StreamJobContext, watermark: Option<i64>) -> StreamOperatorContext<'_> {
    StreamOperatorContext::new(job, "cross_section", watermark.map(EventTime::from_micros))
}

fn new_operator() -> CrossSectionOperator {
    CrossSectionOperator::new("cross_section", input_schema(), spec()).unwrap()
}

fn symbols(batch: &Batch) -> Vec<String> {
    let mut values = Vec::new();
    for record in batch.table_payload().unwrap().batches() {
        let array = record
            .column_by_name("symbol")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for index in 0..array.len() {
            values.push(array.value(index).to_owned());
        }
    }
    values
}

fn float_column(batch: &Batch, name: &str) -> Vec<Option<f64>> {
    let mut values = Vec::new();
    for record in batch.table_payload().unwrap().batches() {
        let array = record
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for index in 0..array.len() {
            values.push(if array.is_null(index) {
                None
            } else {
                Some(array.value(index))
            });
        }
    }
    values
}

#[derive(Default)]
struct Observed {
    symbols: Vec<String>,
    ranks: Vec<Option<f64>>,
    zscores: Vec<Option<f64>>,
}

fn drain(collector: &mut EdgeCollector, observed: &mut Observed) {
    for message in collector.drain("output") {
        let batch = message.as_data().unwrap();
        observed.symbols.extend(symbols(batch));
        observed.ranks.extend(float_column(batch, "momentum_rank"));
        observed.zscores.extend(float_column(batch, "momentum_z"));
    }
}

/// The shared scenario: two interleaved groups, one closing before the cut
/// and one still half-built across it.
const FIRST_BATCH: &[InputRow] = &[
    (100, "a", Some("tech"), 1, Some(2.0)),
    (100, "e", Some("fin"), 1, Some(10.0)),
    (200, "a", Some("tech"), 1, Some(5.0)),
];

const SECOND_BATCH: &[InputRow] = &[
    (100, "b", Some("tech"), 2, Some(2.0)),
    (100, "c", Some("tech"), 3, Some(1.0)),
    (100, "d", Some("tech"), 4, None),
];

#[tokio::test]
async fn half_built_group_checkpoint_recovery_matches_uninterrupted_execution() {
    // Uninterrupted reference execution.
    let job = new_job();
    let mut reference = new_operator();
    let mut reference_collector = EdgeCollector::new(reference.output_ports().to_vec());
    let mut reference_observed = Observed::default();
    reference
        .process_data(
            "input",
            input_batch(FIRST_BATCH),
            &context(&job, None),
            &mut reference_collector,
        )
        .await
        .unwrap();
    reference
        .process_data(
            "input",
            input_batch(SECOND_BATCH),
            &context(&job, Some(99)),
            &mut reference_collector,
        )
        .await
        .unwrap();
    reference
        .on_watermark(
            EventTime::from_micros(100),
            &context(&job, Some(100)),
            &mut reference_collector,
        )
        .await
        .unwrap();
    reference
        .on_watermark(
            EventTime::from_micros(200),
            &context(&job, Some(200)),
            &mut reference_collector,
        )
        .await
        .unwrap();
    reference
        .on_end(&context(&job, Some(200)), &mut reference_collector)
        .await
        .unwrap();
    drain(&mut reference_collector, &mut reference_observed);
    assert_eq!(
        reference_observed.symbols,
        vec!["e", "a", "b", "c", "d", "a"]
    );

    // Recovered execution: checkpoint after the first batch while the
    // (100, tech) group is half-built, then restore into a fresh operator.
    let recovered_job = new_job();
    let mut original = new_operator();
    let mut collected = EdgeCollector::new(original.output_ports().to_vec());
    original
        .process_data(
            "input",
            input_batch(FIRST_BATCH),
            &context(&job, None),
            &mut collected,
        )
        .await
        .unwrap();
    let snapshot = original.checkpoint(Epoch::new(1).unwrap()).unwrap();

    let mut restored = new_operator();
    restored.restore(&snapshot).unwrap();
    let mut restored_collector = EdgeCollector::new(restored.output_ports().to_vec());
    let mut restored_observed = Observed::default();
    restored
        .process_data(
            "input",
            input_batch(SECOND_BATCH),
            &context(&recovered_job, Some(99)),
            &mut restored_collector,
        )
        .await
        .unwrap();
    restored
        .on_watermark(
            EventTime::from_micros(100),
            &context(&recovered_job, Some(100)),
            &mut restored_collector,
        )
        .await
        .unwrap();
    restored
        .on_watermark(
            EventTime::from_micros(200),
            &context(&recovered_job, Some(200)),
            &mut restored_collector,
        )
        .await
        .unwrap();
    restored
        .on_end(&context(&recovered_job, Some(200)), &mut restored_collector)
        .await
        .unwrap();
    drain(&mut restored_collector, &mut restored_observed);

    assert_eq!(restored_observed.symbols, reference_observed.symbols);
    assert_eq!(restored_observed.ranks, reference_observed.ranks);
    assert_eq!(restored_observed.zscores, reference_observed.zscores);
}

#[tokio::test]
async fn empty_snapshot_restores_a_fresh_operator() {
    let mut operator = new_operator();
    let job = new_job();
    let mut collected = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            input_batch(FIRST_BATCH),
            &context(&job, None),
            &mut collected,
        )
        .await
        .unwrap();
    operator.reset().unwrap();

    let mut restored = new_operator();
    let empty = calc_flow::OperatorStateSnapshot::default();
    restored.restore(&empty).unwrap();
    let mut observed = Observed::default();
    restored
        .process_data(
            "input",
            input_batch(FIRST_BATCH),
            &context(&job, None),
            &mut collected,
        )
        .await
        .unwrap();
    restored
        .on_end(&context(&job, None), &mut collected)
        .await
        .unwrap();
    drain(&mut collected, &mut observed);
    assert_eq!(observed.symbols, vec!["e", "a", "a"]);
}

#[tokio::test]
async fn checkpoint_epochs_must_advance_strictly() {
    let mut operator = new_operator();
    operator.checkpoint(Epoch::new(5).unwrap()).unwrap();
    let error = operator.checkpoint(Epoch::new(5).unwrap()).unwrap_err();
    assert!(error.to_string().contains("did not advance"), "{error}");
}

#[tokio::test]
async fn an_empty_state_checkpoint_carries_no_segments() {
    let mut operator = new_operator();
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert!(snapshot.segments.is_empty());
}

#[tokio::test]
async fn restore_rejects_state_from_a_different_configuration() {
    let job = new_job();
    let mut operator = new_operator();
    let mut collected = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            input_batch(FIRST_BATCH),
            &context(&job, None),
            &mut collected,
        )
        .await
        .unwrap();
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert!(!snapshot.segments.is_empty());

    let mut altered_document = serde_json::to_value(spec()).unwrap();
    altered_document["outputs"][0]["tie_method"] = serde_json::json!("min");
    let altered: CrossSectionSpec = serde_json::from_value(altered_document).unwrap();
    let mut other = CrossSectionOperator::new("cross_section", input_schema(), altered).unwrap();
    let error = other.restore(&snapshot).unwrap_err();
    assert!(error.to_string().contains("does not match"), "{error}");
}

#[tokio::test]
async fn restore_rejects_a_corrupted_segment() {
    let job = new_job();
    let mut operator = new_operator();
    let mut collected = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            input_batch(FIRST_BATCH),
            &context(&job, None),
            &mut collected,
        )
        .await
        .unwrap();
    let mut snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    for segment in snapshot.segments.values_mut() {
        let mut bytes = segment.bytes().to_vec();
        bytes[0] ^= 0xff;
        *segment = calc_flow::StateSegment::new(bytes);
    }
    let mut restored = new_operator();
    let error = restored.restore(&snapshot).unwrap_err();
    assert!(error.to_string().contains("does not match"), "{error}");
}

#[tokio::test]
async fn restored_state_rejects_a_different_pipeline_identity() {
    let job = new_job();
    let mut operator = new_operator();
    let mut collected = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            input_batch(FIRST_BATCH),
            &context(&job, None),
            &mut collected,
        )
        .await
        .unwrap();
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();

    let mut restored = new_operator();
    restored.restore(&snapshot).unwrap();
    let other_job = StreamJobContext::new(
        1,
        "0f0e0d0c0b0a09080706050403020100ffeeddccbbaa99887766554433221100",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let mut other_collector = EdgeCollector::new(restored.output_ports().to_vec());
    let error = restored
        .process_data(
            "input",
            input_batch(SECOND_BATCH),
            &context(&other_job, None),
            &mut other_collector,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("fingerprint"), "{error}");
}

#[tokio::test]
async fn terminal_flush_after_restore_releases_state_for_a_duplicate_identity() {
    let job = new_job();
    let mut operator = new_operator();
    let mut collected = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            input_batch(FIRST_BATCH),
            &context(&job, None),
            &mut collected,
        )
        .await
        .unwrap();
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();

    let mut restored = new_operator();
    restored.restore(&snapshot).unwrap();
    let mut restored_collector = EdgeCollector::new(restored.output_ports().to_vec());
    restored
        .on_end(&context(&job, None), &mut restored_collector)
        .await
        .unwrap();
    let empty = restored.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert!(empty.segments.is_empty(), "flushed state was not released");
}

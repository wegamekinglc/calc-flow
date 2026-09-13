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
    array::{
        Array, ArrayRef, BooleanArray, Float64Array, StringArray, TimestampMicrosecondArray,
        UInt64Array,
    },
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
            },
            {
                "kind": "winsorize",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "momentum_winsorized",
                "min_samples": 1,
                "lower": 0.25,
                "upper": 0.75
            },
            {
                "kind": "top",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "is_top",
                "count": 2,
                "include_ties": true,
                "min_samples": 1
            },
            {
                "kind": "bottom",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "is_bottom",
                "count": 2,
                "include_ties": false,
                "min_samples": 1
            },
            {
                "kind": "mean_fill",
                "primitive_version": 1,
                "input": "momentum_20",
                "output": "momentum_filled",
                "min_samples": 1
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

fn late_operator() -> CrossSectionOperator {
    let mut spec = spec();
    spec.late_policy = calc_flow::LatePolicySpec::SideOutput {
        metrics_version: 1,
        schema_version: 1,
    };
    CrossSectionOperator::new("cross_section", input_schema(), spec).unwrap()
}

#[test]
fn test_late_snapshot_extension_restores_sequence_and_reset() {
    let mut operator = late_operator();
    let mut snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert_eq!(
        snapshot.inline_metadata["late_output"],
        serde_json::json!({
            "version": 1, "schema_version": 1, "next_sequence": 0
        })
    );
    snapshot.inline_metadata.get_mut("late_output").unwrap()["next_sequence"] = 7.into();
    StreamOperator::restore(&mut operator, &snapshot).unwrap();
    let restored = operator.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert_eq!(restored.inline_metadata["late_output"]["next_sequence"], 7);
    StreamOperator::reset(&mut operator).unwrap();
    let reset = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert_eq!(reset.inline_metadata["late_output"]["next_sequence"], 0);
    let legacy = new_operator().checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert!(!legacy.inline_metadata.contains_key("late_output"));
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
    winsorized: Vec<Option<f64>>,
    top: Vec<Option<bool>>,
    bottom: Vec<Option<bool>>,
    filled: Vec<Option<f64>>,
}

fn drain(collector: &mut EdgeCollector, observed: &mut Observed) {
    for message in collector.drain("output") {
        let batch = message.as_data().unwrap();
        observed.symbols.extend(symbols(batch));
        observed.ranks.extend(float_column(batch, "momentum_rank"));
        observed.zscores.extend(float_column(batch, "momentum_z"));
        observed
            .winsorized
            .extend(float_column(batch, "momentum_winsorized"));
        observed
            .filled
            .extend(float_column(batch, "momentum_filled"));
        for (name, target) in [
            ("is_top", &mut observed.top),
            ("is_bottom", &mut observed.bottom),
        ] {
            for record in batch.table_payload().unwrap().batches() {
                let array = record
                    .column_by_name(name)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .unwrap();
                target.extend(array.iter());
            }
        }
    }
}

fn assert_observed_eq(restored: &Observed, reference: &Observed) {
    assert_eq!(restored.symbols, reference.symbols);
    assert_eq!(restored.ranks, reference.ranks);
    assert_eq!(restored.zscores, reference.zscores);
    assert_eq!(restored.winsorized, reference.winsorized);
    assert_eq!(restored.top, reference.top);
    assert_eq!(restored.bottom, reference.bottom);
    assert_eq!(restored.filled, reference.filled);
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

    assert_observed_eq(&restored_observed, &reference_observed);
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

#[test]
fn test_late_snapshot_rejects_corruption_without_installing_state() {
    let mut operator = late_operator();
    let mut snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    snapshot.inline_metadata.get_mut("late_output").unwrap()["next_sequence"] = 7.into();
    StreamOperator::restore(&mut operator, &snapshot).unwrap();
    let valid = serde_json::json!({"version": 1, "schema_version": 1, "next_sequence": 7});
    let malformed = [
        None,
        Some(serde_json::Value::Null),
        Some(serde_json::json!({})),
        Some(serde_json::json!({"version": 2, "schema_version": 1, "next_sequence": 7})),
        Some(serde_json::json!({"version": 1, "schema_version": 2, "next_sequence": 7})),
        Some(serde_json::json!({"version": 1, "schema_version": 1, "next_sequence": -1})),
        Some(serde_json::json!({"version": 1, "schema_version": 1, "next_sequence": "7"})),
        Some(serde_json::json!({"version": 1, "schema_version": 1, "next_sequence": 7.5})),
        Some(serde_json::json!({"version": 1, "schema_version": 1, "next_sequence": 1e30})),
        Some(
            serde_json::json!({"version": 1, "schema_version": 1, "next_sequence": 7, "outbox": []}),
        ),
    ];
    for extension in malformed {
        let mut corrupt = snapshot.clone();
        corrupt.inline_metadata.remove("late_output");
        if let Some(extension) = extension {
            corrupt
                .inline_metadata
                .insert("late_output".into(), extension);
        }
        assert!(
            StreamOperator::restore(&mut operator, &corrupt).is_err(),
            "{corrupt:?}"
        );
    }
    for field in ["configuration_hash", "state_schema_fingerprint"] {
        let mut corrupt = snapshot.clone();
        corrupt.inline_metadata.insert(field.into(), "wrong".into());
        assert!(StreamOperator::restore(&mut operator, &corrupt).is_err());
    }
    let after = operator.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert_eq!(after.inline_metadata["late_output"], valid);
    let mut old = new_operator();
    let legacy = old.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert!(StreamOperator::restore(&mut operator, &legacy).is_err());
    for extension in [valid, serde_json::Value::Null] {
        let mut corrupt = legacy.clone();
        corrupt
            .inline_metadata
            .insert("late_output".into(), extension);
        assert!(StreamOperator::restore(&mut old, &corrupt).is_err());
    }
}

#[tokio::test]
async fn test_late_restored_sequence_overflow_rejects_envelope_without_output_or_state_change() {
    let mut operator = late_operator();
    let mut snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    snapshot.inline_metadata.get_mut("late_output").unwrap()["next_sequence"] = u64::MAX.into();
    StreamOperator::restore(&mut operator, &snapshot).unwrap();
    let job = new_job();
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let error = operator
        .process_data(
            "input",
            input_batch(&[(5, "a", None, 1, Some(5.0)), (20, "a", None, 2, Some(20.0))]),
            &context(&job, Some(10)),
            &mut output,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("sequence"), "{error}");
    assert!(output.drain("late").is_empty());
    assert!(output.drain("output").is_empty());
    let after = operator.checkpoint(Epoch::new(2).unwrap()).unwrap();
    for field in [
        "late_output",
        "next_output_sequence",
        "metrics",
        "last_input_watermark",
        "segment_inventory",
    ] {
        assert_eq!(
            after.inline_metadata[field],
            snapshot.inline_metadata[field]
        );
    }
}

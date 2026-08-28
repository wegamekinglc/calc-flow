//! Stream cross-section semantics: cross-batch group assembly, watermark
//! finality gating, late policies, final flush, and state release
//! (SCE-00 D6/D7, SCE-09).

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

const FINGERPRINT: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

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

fn spec(
    grouping: &serde_json::Value,
    allowed_lateness_micros: u64,
    late_policy: &serde_json::Value,
) -> CrossSectionSpec {
    serde_json::from_value(serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "event_time": "ts",
        "entity_by": ["symbol"],
        "partition_by": ["industry"],
        "sequence_by": ["sequence"],
        "grouping": grouping,
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
            }
        ],
        "allowed_lateness_micros": allowed_lateness_micros,
        "late_policy": late_policy,
        "value_policy": "nan_exclude_preserve_v1"
    }))
    .unwrap()
}

fn error_operator() -> CrossSectionOperator {
    CrossSectionOperator::new(
        "cross_section",
        input_schema(),
        spec(
            &serde_json::json!({"kind": "exact_time"}),
            0,
            &serde_json::json!({"kind": "error", "scope": "envelope"}),
        ),
    )
    .unwrap()
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

fn context(job: &StreamJobContext, watermark: Option<i64>) -> StreamOperatorContext<'_> {
    StreamOperatorContext::new(job, "cross_section", watermark.map(EventTime::from_micros))
}

fn collector(operator: &CrossSectionOperator) -> EdgeCollector {
    EdgeCollector::new(operator.output_ports().to_vec())
}

#[derive(Default)]
struct Observed {
    event_times: Vec<i64>,
    symbols: Vec<String>,
    industries: Vec<Option<String>>,
    sequences: Vec<u64>,
    ranks: Vec<Option<f64>>,
    batch_sequences: Vec<u64>,
}

fn drain(collector: &mut EdgeCollector, observed: &mut Observed) {
    for message in collector.drain("output") {
        let batch = message.as_data().unwrap();
        observed.batch_sequences.push(batch.metadata().sequence());
        for record in batch.table_payload().unwrap().batches() {
            let event_times = record
                .column_by_name("ts")
                .unwrap()
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
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
            let sequences = record
                .column_by_name("sequence")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let ranks = record
                .column_by_name("momentum_rank")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            for index in 0..record.num_rows() {
                observed.event_times.push(event_times.value(index));
                observed.symbols.push(symbols.value(index).to_owned());
                observed.industries.push(if industries.is_null(index) {
                    None
                } else {
                    Some(industries.value(index).to_owned())
                });
                observed.sequences.push(sequences.value(index));
                observed.ranks.push(if ranks.is_null(index) {
                    None
                } else {
                    Some(ranks.value(index))
                });
            }
        }
    }
}

#[tokio::test]
async fn group_split_across_batches_emits_once_at_the_closing_watermark() {
    let job = job();
    let mut operator = error_operator();
    let mut collected = collector(&operator);
    let mut observed = Observed::default();

    // First micro-batch carries half of the (100, tech) group.
    operator
        .process_data(
            "input",
            input_batch(&[
                (100, "a", Some("tech"), 1, Some(2.0)),
                (100, "c", Some("tech"), 3, Some(1.0)),
            ]),
            &context(&job, None),
            &mut collected,
        )
        .await
        .unwrap();
    drain(&mut collected, &mut observed);
    assert!(
        observed.event_times.is_empty(),
        "incomplete group emitted early"
    );

    // A watermark below the closing coordinate still emits nothing.
    operator
        .on_watermark(
            EventTime::from_micros(99),
            &context(&job, Some(99)),
            &mut collected,
        )
        .await
        .unwrap();
    drain(&mut collected, &mut observed);
    assert!(
        observed.event_times.is_empty(),
        "group closed before its coordinate"
    );

    // The remaining rows of the same group arrive in a second batch.
    operator
        .process_data(
            "input",
            input_batch(&[
                (100, "b", Some("tech"), 2, Some(2.0)),
                (100, "d", Some("tech"), 4, None),
            ]),
            &context(&job, Some(99)),
            &mut collected,
        )
        .await
        .unwrap();

    // Equality closes: t + L == W with L = 0.
    operator
        .on_watermark(
            EventTime::from_micros(100),
            &context(&job, Some(100)),
            &mut collected,
        )
        .await
        .unwrap();
    drain(&mut collected, &mut observed);
    assert_eq!(observed.event_times, vec![100, 100, 100, 100]);
    assert_eq!(observed.symbols, vec!["a", "b", "c", "d"]);
    assert_eq!(observed.ranks, vec![Some(2.5), Some(2.5), Some(1.0), None]);

    // The closed group released its state: a duplicate identity is a fresh
    // duplicate of a closed group and still fails loudly.
    let error = operator
        .process_data(
            "input",
            input_batch(&[(100, "a", Some("tech"), 1, Some(2.0))]),
            &context(&job, Some(100)),
            &mut collected,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("late_row"), "{error}");
}

#[tokio::test]
async fn groups_close_independently_and_stream_in_finality_order() {
    let job = job();
    let mut operator = error_operator();
    let mut collected = collector(&operator);
    let mut observed = Observed::default();

    operator
        .process_data(
            "input",
            input_batch(&[
                (300, "x", Some("tech"), 1, Some(9.0)),
                (100, "e", Some("fin"), 1, Some(10.0)),
                (100, "a", Some("tech"), 1, Some(1.0)),
                (200, "b", Some("tech"), 1, Some(5.0)),
            ]),
            &context(&job, None),
            &mut collected,
        )
        .await
        .unwrap();

    operator
        .on_watermark(
            EventTime::from_micros(100),
            &context(&job, Some(100)),
            &mut collected,
        )
        .await
        .unwrap();
    drain(&mut collected, &mut observed);
    // Group (100, fin) precedes (100, tech) in group-key order.
    assert_eq!(observed.symbols, vec!["e", "a"]);
    assert_eq!(
        observed.industries,
        vec![Some("fin".to_owned()), Some("tech".to_owned())]
    );

    operator
        .on_watermark(
            EventTime::from_micros(200),
            &context(&job, Some(200)),
            &mut collected,
        )
        .await
        .unwrap();
    drain(&mut collected, &mut observed);
    assert_eq!(observed.symbols, vec!["e", "a", "b"]);

    operator
        .on_watermark(
            EventTime::from_micros(300),
            &context(&job, Some(300)),
            &mut collected,
        )
        .await
        .unwrap();
    drain(&mut collected, &mut observed);
    assert_eq!(observed.symbols, vec!["e", "a", "b", "x"]);
    assert_eq!(
        observed.batch_sequences,
        vec![0, 1, 2],
        "each closed group emits in its own ordered output batch"
    );
}

#[tokio::test]
async fn allowed_lateness_defers_group_closing() {
    let job = job();
    let operator = CrossSectionOperator::new(
        "cross_section",
        input_schema(),
        spec(
            &serde_json::json!({"kind": "exact_time"}),
            10,
            &serde_json::json!({"kind": "error", "scope": "envelope"}),
        ),
    )
    .unwrap();
    let mut operator = operator;
    let mut collected = collector(&error_operator());
    let mut observed = Observed::default();

    operator
        .process_data(
            "input",
            input_batch(&[(100, "a", Some("tech"), 1, Some(1.0))]),
            &context(&job, None),
            &mut collected,
        )
        .await
        .unwrap();
    operator
        .on_watermark(
            EventTime::from_micros(109),
            &context(&job, Some(109)),
            &mut collected,
        )
        .await
        .unwrap();
    drain(&mut collected, &mut observed);
    assert!(observed.event_times.is_empty(), "closed at 100 + 10 > 109");

    // A row for the still-open group keeps accumulating.
    operator
        .process_data(
            "input",
            input_batch(&[(100, "b", Some("tech"), 1, Some(2.0))]),
            &context(&job, Some(109)),
            &mut collected,
        )
        .await
        .unwrap();
    operator
        .on_watermark(
            EventTime::from_micros(110),
            &context(&job, Some(110)),
            &mut collected,
        )
        .await
        .unwrap();
    drain(&mut collected, &mut observed);
    assert_eq!(observed.symbols, vec!["a", "b"]);
    assert_eq!(observed.ranks, vec![Some(1.0), Some(2.0)]);
}

#[tokio::test]
async fn bucketed_groups_close_on_bucket_end() {
    let job = job();
    let operator = CrossSectionOperator::new(
        "cross_section",
        input_schema(),
        spec(
            &serde_json::json!({"kind": "fixed_bucket", "width_micros": 100}),
            0,
            &serde_json::json!({"kind": "error", "scope": "envelope"}),
        ),
    )
    .unwrap();
    let mut operator = operator;
    let mut collected = collector(&error_operator());
    let mut observed = Observed::default();

    operator
        .process_data(
            "input",
            input_batch(&[
                (100, "a", Some("tech"), 1, Some(1.0)),
                (199, "b", Some("tech"), 1, Some(2.0)),
            ]),
            &context(&job, None),
            &mut collected,
        )
        .await
        .unwrap();
    // The bucket [100, 200) closes at 200, not at its first row's event time.
    operator
        .on_watermark(
            EventTime::from_micros(199),
            &context(&job, Some(199)),
            &mut collected,
        )
        .await
        .unwrap();
    drain(&mut collected, &mut observed);
    assert!(observed.event_times.is_empty());

    operator
        .on_watermark(
            EventTime::from_micros(200),
            &context(&job, Some(200)),
            &mut collected,
        )
        .await
        .unwrap();
    drain(&mut collected, &mut observed);
    assert_eq!(observed.symbols, vec!["a", "b"]);
}

#[tokio::test]
async fn late_error_policy_rejects_the_envelope_transactionally() {
    let job = job();
    let mut operator = error_operator();
    let mut collected = collector(&operator);
    let mut observed = Observed::default();

    operator
        .process_data(
            "input",
            input_batch(&[(100, "a", Some("tech"), 1, Some(1.0))]),
            &context(&job, None),
            &mut collected,
        )
        .await
        .unwrap();
    operator
        .on_watermark(
            EventTime::from_micros(150),
            &context(&job, Some(150)),
            &mut collected,
        )
        .await
        .unwrap();
    drain(&mut collected, &mut observed);
    assert_eq!(observed.symbols, vec!["a"]);

    let error = operator
        .process_data(
            "input",
            input_batch(&[
                (100, "b", Some("tech"), 2, Some(2.0)),
                (300, "c", Some("tech"), 1, Some(3.0)),
            ]),
            &context(&job, Some(150)),
            &mut collected,
        )
        .await
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("late_row"), "{message}");
    assert!(message.contains("row_index=0"), "{message}");

    // Nothing from the rejected envelope was installed: the valid (300, tech)
    // row never appears, and the duplicate-free group state is unchanged.
    operator
        .on_watermark(
            EventTime::from_micros(400),
            &context(&job, Some(400)),
            &mut collected,
        )
        .await
        .unwrap();
    drain(&mut collected, &mut observed);
    assert_eq!(observed.symbols, vec!["a"]);
}

#[tokio::test]
async fn drop_policy_drops_only_the_late_rows_and_records_metrics() {
    let job = job();
    let operator = CrossSectionOperator::new(
        "cross_section",
        input_schema(),
        spec(
            &serde_json::json!({"kind": "exact_time"}),
            0,
            &serde_json::json!({"kind": "drop", "metrics_version": 1}),
        ),
    )
    .unwrap();
    let mut operator = operator;
    let mut collected = collector(&error_operator());
    let mut observed = Observed::default();

    operator
        .process_data(
            "input",
            input_batch(&[(100, "a", Some("tech"), 1, Some(1.0))]),
            &context(&job, None),
            &mut collected,
        )
        .await
        .unwrap();
    operator
        .on_watermark(
            EventTime::from_micros(200),
            &context(&job, Some(200)),
            &mut collected,
        )
        .await
        .unwrap();
    drain(&mut collected, &mut observed);
    assert_eq!(observed.symbols, vec!["a"]);

    // The late (100, tech) row is dropped; the (300, tech) row survives.
    operator
        .process_data(
            "input",
            input_batch(&[
                (100, "b", Some("tech"), 2, Some(2.0)),
                (300, "c", Some("tech"), 1, Some(3.0)),
            ]),
            &context(&job, Some(200)),
            &mut collected,
        )
        .await
        .unwrap();
    operator
        .on_watermark(
            EventTime::from_micros(300),
            &context(&job, Some(300)),
            &mut collected,
        )
        .await
        .unwrap();
    drain(&mut collected, &mut observed);
    assert_eq!(observed.symbols, vec!["a", "c"]);
}

#[tokio::test]
async fn end_of_input_flushes_every_open_group_once_in_canonical_order() {
    let job = job();
    let mut operator = error_operator();
    let mut collected = collector(&operator);
    let mut observed = Observed::default();

    operator
        .process_data(
            "input",
            input_batch(&[
                (500, "a", Some("tech"), 1, Some(1.0)),
                (100, "b", Some("tech"), 1, Some(2.0)),
                (100, "c", Some("fin"), 1, Some(3.0)),
            ]),
            &context(&job, None),
            &mut collected,
        )
        .await
        .unwrap();
    operator
        .on_end(&context(&job, None), &mut collected)
        .await
        .unwrap();
    drain(&mut collected, &mut observed);
    // Groups ordered by finality coordinate then group key; both (100, *)
    // groups flush before (500, tech).
    assert_eq!(observed.symbols, vec!["c", "b", "a"]);
    assert_eq!(observed.industries.first(), Some(&Some("fin".to_owned())));

    // The flush released all state: a repeated end is a no-op and data after
    // end-of-input is rejected.
    operator
        .on_end(&context(&job, None), &mut collected)
        .await
        .unwrap();
    drain(&mut collected, &mut observed);
    assert_eq!(observed.symbols.len(), 3);
    let error = operator
        .process_data(
            "input",
            input_batch(&[(600, "z", Some("tech"), 1, Some(1.0))]),
            &context(&job, None),
            &mut collected,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("after end-of-input"), "{error}");
}

#[tokio::test]
async fn duplicate_identity_within_one_envelope_fails_transactionally() {
    let job = job();
    let mut operator = error_operator();
    let mut collected = collector(&operator);

    let error = operator
        .process_data(
            "input",
            input_batch(&[
                (100, "a", Some("tech"), 1, Some(1.0)),
                (100, "a", Some("tech"), 1, Some(2.0)),
                (100, "b", Some("tech"), 1, Some(3.0)),
            ]),
            &context(&job, None),
            &mut collected,
        )
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("duplicate row identity"),
        "{error}"
    );
}

#[tokio::test]
async fn watermarks_must_advance_strictly() {
    let job = job();
    let mut operator = error_operator();
    let mut collected = collector(&operator);

    operator
        .on_watermark(
            EventTime::from_micros(100),
            &context(&job, Some(100)),
            &mut collected,
        )
        .await
        .unwrap();
    let error = operator
        .on_watermark(
            EventTime::from_micros(100),
            &context(&job, Some(100)),
            &mut collected,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("did not advance"), "{error}");
}

#[tokio::test]
async fn unknown_ingress_is_rejected() {
    let job = job();
    let mut operator = error_operator();
    let mut collected = collector(&operator);
    let error = operator
        .process_data(
            "other",
            input_batch(&[]),
            &context(&job, None),
            &mut collected,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("unknown ingress"), "{error}");
}

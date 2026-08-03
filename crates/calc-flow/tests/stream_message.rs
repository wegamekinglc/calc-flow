//! RED (M1.2): strongly typed stream messages.
//!
//! Every test in this file fails to compile until `EventTime`, `Epoch`, and
//! `StreamMessage` exist with the frozen v3 surface (spec D1/D9, API note A7,
//! plan task M1.2). The expected RED reason is an unresolved import of
//! `calc_flow::{EventTime, Epoch, StreamMessage, StreamMessageKind}`.

use std::sync::Arc;

use calc_flow::{Batch, BatchMetadata, Epoch, EventTime, StreamMessage, StreamMessageKind};
use datafusion::arrow::{
    array::{Array, Int64Array},
    datatypes::{DataType, TimeUnit},
    record_batch::RecordBatch,
};

fn table_batch(values: &[i64]) -> Batch {
    let record = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from(values.to_vec())) as Arc<dyn Array>,
    )])
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn timestamp(unit: TimeUnit) -> DataType {
    DataType::Timestamp(unit, None)
}

#[test]
fn event_time_orders_values_across_the_unix_epoch() {
    let before = EventTime::from_micros(-1);
    let epoch = EventTime::from_micros(0);
    let after = EventTime::from_micros(1);

    assert!(before < epoch);
    assert!(epoch < after);
    assert_eq!(before, EventTime::from_micros(-1));
    assert!(EventTime::from_micros(i64::MIN) < before);
    assert!(after < EventTime::from_micros(i64::MAX));
}

#[test]
fn event_time_imports_seconds_and_milliseconds_with_checked_multiplication() {
    let column = "events.commit_time";
    let seconds = EventTime::import_timestamp(42, &timestamp(TimeUnit::Second), column).unwrap();
    assert_eq!(seconds.as_micros(), 42_000_000);
    let millis =
        EventTime::import_timestamp(-42, &timestamp(TimeUnit::Millisecond), column).unwrap();
    assert_eq!(millis.as_micros(), -42_000);
    let micros =
        EventTime::import_timestamp(i64::MAX, &timestamp(TimeUnit::Microsecond), column).unwrap();
    assert_eq!(micros.as_micros(), i64::MAX);
}

#[test]
fn event_time_import_nanoseconds_floors_toward_negative_infinity() {
    let column = "events.commit_time";
    let nanos = timestamp(TimeUnit::Nanosecond);

    // Spec D1.5 worked examples: -1,500 ns -> -2 us; -1 ns -> -1 us; 1,999 ns -> 1 us.
    assert_eq!(
        EventTime::import_timestamp(-1_500, &nanos, column)
            .unwrap()
            .as_micros(),
        -2
    );
    assert_eq!(
        EventTime::import_timestamp(-1, &nanos, column)
            .unwrap()
            .as_micros(),
        -1
    );
    assert_eq!(
        EventTime::import_timestamp(1_999, &nanos, column)
            .unwrap()
            .as_micros(),
        1
    );
    // Nanosecond import cannot overflow: division by 1,000 always fits.
    assert_eq!(
        EventTime::import_timestamp(i64::MAX, &nanos, column)
            .unwrap()
            .as_micros(),
        i64::MAX / 1_000
    );
    assert_eq!(
        EventTime::import_timestamp(i64::MIN, &nanos, column)
            .unwrap()
            .as_micros(),
        i64::MIN.div_euclid(1_000)
    );
}

#[test]
fn event_time_import_overflow_is_a_typed_error_with_the_column_path() {
    let column = "events.commit_time";
    let seconds =
        EventTime::import_timestamp(i64::MAX, &timestamp(TimeUnit::Second), column).unwrap_err();
    assert!(seconds.to_string().contains(column));

    let millis = EventTime::import_timestamp(
        i64::MAX / 1_000 + 1,
        &timestamp(TimeUnit::Millisecond),
        column,
    )
    .unwrap_err();
    assert!(millis.to_string().contains(column));

    let negative =
        EventTime::import_timestamp(i64::MIN, &timestamp(TimeUnit::Second), column).unwrap_err();
    assert!(negative.to_string().contains(column));
}

#[test]
fn event_time_import_rejects_non_timestamp_types_with_the_column_path() {
    let error = EventTime::import_timestamp(1, &DataType::Int64, "events.value").unwrap_err();
    assert!(error.to_string().contains("events.value"));
}

#[test]
fn event_time_import_accepts_naive_and_explicit_utc_only() {
    let column = "events.commit_time";
    let utc = DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC")));
    assert!(EventTime::import_timestamp(7, &utc, column).is_ok());

    let zoned = DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("America/New_York")));
    let error = EventTime::import_timestamp(7, &zoned, column).unwrap_err();
    assert!(error.to_string().contains(column));
}

#[test]
fn event_time_export_floors_coarser_units_toward_negative_infinity() {
    // Spec D1.5 worked examples: -1 us -> -1 s; 999,999 us -> 0 s.
    assert_eq!(
        EventTime::from_micros(-1)
            .export_timestamp(TimeUnit::Second)
            .unwrap(),
        -1
    );
    assert_eq!(
        EventTime::from_micros(999_999)
            .export_timestamp(TimeUnit::Second)
            .unwrap(),
        0
    );
    assert_eq!(
        EventTime::from_micros(-1_000_001)
            .export_timestamp(TimeUnit::Second)
            .unwrap(),
        -2
    );
    assert_eq!(
        EventTime::from_micros(-1)
            .export_timestamp(TimeUnit::Millisecond)
            .unwrap(),
        -1
    );
    assert_eq!(
        EventTime::from_micros(42)
            .export_timestamp(TimeUnit::Microsecond)
            .unwrap(),
        42
    );
}

#[test]
fn event_time_export_nanoseconds_is_checked() {
    assert_eq!(
        EventTime::from_micros(42)
            .export_timestamp(TimeUnit::Nanosecond)
            .unwrap(),
        42_000
    );
    assert!(
        EventTime::from_micros(i64::MAX / 1_000 + 1)
            .export_timestamp(TimeUnit::Nanosecond)
            .is_err()
    );
}

#[test]
fn event_time_serializes_the_exact_microsecond_value() {
    let time = EventTime::from_micros(-1_234_567_890);
    let encoded = serde_json::to_string(&time).unwrap();
    assert_eq!(encoded, "-1234567890");
    let decoded: EventTime = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, time);
}

#[test]
fn epoch_starts_at_one_and_increments_by_one() {
    assert_eq!(Epoch::INITIAL.as_u64(), 1);
    let second = Epoch::INITIAL.next().unwrap();
    assert_eq!(second.as_u64(), 2);
    assert!(Epoch::INITIAL < second);
}

#[test]
fn epoch_zero_is_unconstructable() {
    assert!(Epoch::new(0).is_none());
    assert_eq!(Epoch::new(1).unwrap(), Epoch::INITIAL);
}

#[test]
fn epoch_increment_is_checked() {
    let max = Epoch::new(u64::MAX).unwrap();
    assert!(max.next().is_err());
}

#[test]
fn epoch_serializes_the_exact_value() {
    let epoch = Epoch::new(7).unwrap();
    let encoded = serde_json::to_string(&epoch).unwrap();
    assert_eq!(encoded, "7");
    let decoded: Epoch = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, epoch);
}

#[test]
fn stream_message_data_clone_shares_the_immutable_payload() {
    let batch = table_batch(&[1, 2, 3]);
    let column = Arc::clone(batch.table_payload().unwrap().batches()[0].column(0));
    let schema = Arc::clone(batch.table_payload().unwrap().schema());

    let message = StreamMessage::data(batch);
    let clone = message.clone();

    for fan_out in [message, clone] {
        let data = fan_out.as_data().unwrap();
        assert!(Arc::ptr_eq(
            data.table_payload().unwrap().batches()[0].column(0),
            &column
        ));
        assert!(Arc::ptr_eq(data.table_payload().unwrap().schema(), &schema));
        assert_eq!(fan_out.kind(), StreamMessageKind::Data);
    }
}

#[test]
fn stream_message_accessors_distinguish_kinds() {
    let data = StreamMessage::data(table_batch(&[1]));
    assert!(data.as_data().is_some());
    assert_eq!(data.as_watermark(), None);
    assert_eq!(data.as_barrier(), None);
    assert!(!data.is_idle());
    assert!(!data.is_end_of_input());
}

#[test]
fn stream_message_debug_never_prints_payload_or_secret_attributes() {
    let mut attributes = calc_flow::JsonMap::new();
    attributes.insert(
        "password".into(),
        serde_json::Value::String("hunter2-canary".into()),
    );
    let metadata = BatchMetadata::new("secret-source", 0, attributes).unwrap();
    let record = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from(vec![42_424_242])) as Arc<dyn Array>,
    )])
    .unwrap();
    let batch = Batch::table(vec![record], metadata).unwrap();

    let debug = format!("{:?}", StreamMessage::data(batch));

    assert!(
        !debug.contains("hunter2-canary"),
        "secret attribute leaked: {debug}"
    );
    assert!(!debug.contains("42424242"), "row payload leaked: {debug}");
    assert!(
        !debug.contains("secret-source"),
        "source identity leaked: {debug}"
    );
}

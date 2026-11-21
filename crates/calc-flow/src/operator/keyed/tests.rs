use super::keyed::{GroupId, KeySchema, KeyStore, MORSEL_ROWS};
use crate::{CalcFlowError, DataFusionConfig, DataFusionRuntime};
use datafusion::arrow::{
    array::{Array, ArrayRef},
    datatypes::{DataType, Field, FieldRef, TimeUnit},
    record_batch::RecordBatch,
};
use datafusion::common::ScalarValue;
use std::{collections::BTreeSet, sync::Arc};

fn runtime() -> DataFusionRuntime {
    DataFusionRuntime::new(DataFusionConfig::default()).unwrap()
}

fn field(data_type: DataType) -> FieldRef {
    Arc::new(Field::new("key", data_type, true))
}

fn store(runtime: &DataFusionRuntime, fields: Vec<FieldRef>) -> KeyStore {
    KeyStore::new(
        KeySchema::groupable(fields).unwrap(),
        runtime.incremental_reservation("keyed-core-test"),
    )
}

fn cancelled() -> CalcFlowError {
    CalcFlowError::Cancelled {
        run_id: "keyed-core-test".into(),
    }
}

fn route(store: &mut KeyStore, values: &[ScalarValue]) -> Vec<GroupId> {
    let array = ScalarValue::iter_to_array(values.iter().cloned()).unwrap();
    store.route(&[array], &mut || Ok(())).unwrap()
}

fn scalar_oracle(rows: &[Vec<ScalarValue>]) -> Vec<usize> {
    let mut keys: Vec<Vec<ScalarValue>> = Vec::new();
    rows.iter()
        .map(|row| {
            if let Some(group) = keys.iter().position(|key| key == row) {
                group
            } else {
                keys.push(row.clone());
                keys.len() - 1
            }
        })
        .collect()
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the supported-type matrix is intentionally one exhaustive test table"
)]
fn test_keyed_typed_routing_matches_naive_oracle_for_every_supported_type() {
    let runtime = runtime();
    let timezone: Arc<str> = Arc::from("America/New_York");
    let cases = vec![
        vec![
            ScalarValue::Boolean(Some(true)),
            ScalarValue::Boolean(None),
            ScalarValue::Boolean(Some(false)),
            ScalarValue::Boolean(Some(true)),
        ],
        vec![
            ScalarValue::Int8(Some(i8::MIN)),
            ScalarValue::Int8(None),
            ScalarValue::Int8(Some(i8::MAX)),
            ScalarValue::Int8(Some(i8::MIN)),
        ],
        vec![
            ScalarValue::Int16(Some(i16::MIN)),
            ScalarValue::Int16(None),
            ScalarValue::Int16(Some(i16::MAX)),
        ],
        vec![
            ScalarValue::Int32(Some(i32::MIN)),
            ScalarValue::Int32(None),
            ScalarValue::Int32(Some(i32::MAX)),
        ],
        vec![
            ScalarValue::Int64(Some(i64::MIN)),
            ScalarValue::Int64(None),
            ScalarValue::Int64(Some(i64::MAX)),
        ],
        vec![
            ScalarValue::UInt8(Some(0)),
            ScalarValue::UInt8(None),
            ScalarValue::UInt8(Some(u8::MAX)),
        ],
        vec![
            ScalarValue::UInt16(Some(0)),
            ScalarValue::UInt16(None),
            ScalarValue::UInt16(Some(u16::MAX)),
        ],
        vec![
            ScalarValue::UInt32(Some(0)),
            ScalarValue::UInt32(None),
            ScalarValue::UInt32(Some(u32::MAX)),
        ],
        vec![
            ScalarValue::UInt64(Some(0)),
            ScalarValue::UInt64(None),
            ScalarValue::UInt64(Some(u64::MAX)),
        ],
        vec![
            ScalarValue::Utf8(Some("A\0B".into())),
            ScalarValue::Utf8(None),
            ScalarValue::Utf8(Some(String::new())),
            ScalarValue::Utf8(Some("é新😀".into())),
            ScalarValue::Utf8(Some("A\0B".into())),
        ],
        vec![
            ScalarValue::LargeUtf8(Some("large\0key".into())),
            ScalarValue::LargeUtf8(None),
            ScalarValue::LargeUtf8(Some("large\0key".into())),
        ],
        vec![
            ScalarValue::Date32(Some(i32::MIN)),
            ScalarValue::Date32(None),
            ScalarValue::Date32(Some(i32::MAX)),
        ],
        vec![
            ScalarValue::Date64(Some(i64::MIN)),
            ScalarValue::Date64(None),
            ScalarValue::Date64(Some(i64::MAX)),
        ],
        vec![
            ScalarValue::TimestampSecond(Some(1), None),
            ScalarValue::TimestampSecond(None, None),
            ScalarValue::TimestampSecond(Some(1), None),
        ],
        vec![
            ScalarValue::TimestampMillisecond(Some(-2), Some(timezone.clone())),
            ScalarValue::TimestampMillisecond(None, Some(timezone.clone())),
            ScalarValue::TimestampMillisecond(Some(3), Some(timezone.clone())),
        ],
        vec![
            ScalarValue::TimestampMicrosecond(Some(-4), Some(timezone.clone())),
            ScalarValue::TimestampMicrosecond(None, Some(timezone.clone())),
            ScalarValue::TimestampMicrosecond(Some(-4), Some(timezone)),
        ],
        vec![
            ScalarValue::TimestampNanosecond(Some(i64::MIN), None),
            ScalarValue::TimestampNanosecond(None, None),
            ScalarValue::TimestampNanosecond(Some(i64::MAX), None),
        ],
    ];
    for values in cases {
        let array = ScalarValue::iter_to_array(values.iter().cloned()).unwrap();
        let mut store = store(&runtime, vec![field(array.data_type().clone())]);
        let expected = scalar_oracle(
            &values
                .into_iter()
                .map(|value| vec![value])
                .collect::<Vec<_>>(),
        );
        let actual = store
            .route_colliding(std::slice::from_ref(&array), &mut || Ok(()))
            .unwrap();
        let data_type = array.data_type().clone();
        assert_eq!(
            actual.iter().map(|id| id.index()).collect::<Vec<_>>(),
            expected
        );
        let replay = store.route(&[array], &mut || Ok(())).unwrap();
        assert_eq!(actual, replay, "replay type {data_type:?}");
        let canonical = (0..store.len())
            .map(|group| store.canonical(GroupId::new(group).unwrap()).to_vec())
            .collect::<BTreeSet<_>>();
        assert_eq!(canonical.len(), store.len());
    }
}

fn composite_rows() -> Vec<Vec<ScalarValue>> {
    (0..513)
        .map(|row| {
            let number = match row % 11 {
                0 => ScalarValue::Int64(None),
                1 => ScalarValue::Int64(Some(i64::MIN)),
                2 => ScalarValue::Int64(Some(i64::MAX)),
                value => ScalarValue::Int64(Some(value)),
            };
            let string = match row % 7 {
                0 => ScalarValue::Utf8(None),
                1 => ScalarValue::Utf8(Some(String::new())),
                2 => ScalarValue::Utf8(Some("NUL\0KEY".into())),
                value => ScalarValue::Utf8(Some(format!("key-{}", value % 3))),
            };
            let flag = match row % 5 {
                0 => ScalarValue::Boolean(None),
                value => ScalarValue::Boolean(Some(value % 2 == 0)),
            };
            vec![number, string, flag]
        })
        .collect()
}

fn composite_columns(rows: &[Vec<ScalarValue>]) -> Vec<ArrayRef> {
    (0..3)
        .map(|column| {
            ScalarValue::iter_to_array(rows.iter().map(|row| row[column].clone())).unwrap()
        })
        .collect()
}

fn route_in_splits(store: &mut KeyStore, columns: &[ArrayRef], step: usize) -> Vec<GroupId> {
    let rows = columns[0].len();
    let mut groups = Vec::new();
    for start in (0..rows).step_by(step) {
        let end = (start + step).min(rows);
        let chunk = columns
            .iter()
            .map(|array| array.slice(start, end - start))
            .collect::<Vec<_>>();
        groups.extend(store.route(&chunk, &mut || Ok(())).unwrap());
    }
    groups
}

fn assert_same_canonical(left: &KeyStore, right: &KeyStore, distinct: usize) {
    for group in 0..distinct {
        let id = GroupId::new(group).unwrap();
        assert_eq!(left.canonical(id), right.canonical(id));
    }
}

#[test]
fn test_keyed_composite_null_collisions_and_batch_splits_preserve_identity() {
    let runtime = runtime();
    let schema = vec![
        field(DataType::Int64),
        field(DataType::Utf8),
        field(DataType::Boolean),
    ];
    let rows = composite_rows();
    let expected = scalar_oracle(&rows);
    let columns = composite_columns(&rows);
    let mut whole = store(&runtime, schema.clone());
    let actual = whole.route_colliding(&columns, &mut || Ok(())).unwrap();
    assert_eq!(
        actual.iter().map(|id| id.index()).collect::<Vec<_>>(),
        expected
    );
    let distinct = expected.iter().copied().max().unwrap() + 1;
    assert_eq!(whole.len(), distinct);
    assert_eq!(whole.canonical_births(), distinct);

    let mut split = store(&runtime, schema);
    let split_ids = route_in_splits(&mut split, &columns, 37);
    assert_eq!(actual, split_ids);
    assert_eq!(split.canonical_births(), distinct);
    assert_same_canonical(&whole, &split, distinct);
    let births = split.canonical_births();
    split.route(&columns, &mut || Ok(())).unwrap();
    assert_eq!(split.canonical_births(), births);
}

#[test]
fn test_keyed_cancellation_at_morsel_and_long_string_boundaries_is_atomic() {
    let runtime = runtime();
    let mut store = store(&runtime, vec![field(DataType::Utf8)]);
    route(&mut store, &[ScalarValue::Utf8(Some("committed".into()))]);
    let before = (
        store.len(),
        store.next_id(),
        store.canonical_births(),
        store.reserved_bytes(),
        store.canonical(GroupId::new(0).unwrap()),
    );
    let long = "x".repeat(200_000);
    let values = vec![
        ScalarValue::Utf8(Some("birth-a".into())),
        ScalarValue::Utf8(Some(long)),
        ScalarValue::Utf8(Some("birth-b".into())),
    ];
    let array = ScalarValue::iter_to_array(values).unwrap();
    let mut checks = 0;
    let result = store.route(&[array], &mut || {
        checks += 1;
        if checks == 6 {
            Err(cancelled())
        } else {
            Ok(())
        }
    });
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert!(checks >= 3);
    assert_eq!(store.len(), before.0);
    assert_eq!(store.next_id(), before.1);
    assert_eq!(store.canonical_births(), before.2);
    assert_eq!(store.reserved_bytes(), before.3);
    assert_eq!(store.canonical(GroupId::new(0).unwrap()), before.4);

    let oversized = ScalarValue::iter_to_array(
        (0..=MORSEL_ROWS).map(|_| ScalarValue::Utf8(Some("row".into()))),
    )
    .unwrap();
    assert!(store.route(&[oversized], &mut || Ok(())).is_err());
    assert_eq!(store.len(), before.0);
    assert_eq!(store.reserved_bytes(), before.3);
}

#[test]
fn test_keyed_layout_fingerprint_is_deterministic_and_schema_exact() {
    let runtime = runtime();
    let utc: Arc<str> = Arc::from("UTC");
    let other: Arc<str> = Arc::from("Europe/Paris");
    let schema = KeySchema::groupable(vec![field(DataType::Timestamp(
        TimeUnit::Microsecond,
        Some(utc.clone()),
    ))])
    .unwrap();
    let left = KeyStore::new(
        schema.clone(),
        runtime.incremental_reservation("fingerprint-left"),
    );
    let right = KeyStore::new(schema, runtime.incremental_reservation("fingerprint-right"));
    assert_eq!(left.layout_fingerprint(), right.layout_fingerprint());
    let changed = KeyStore::new(
        KeySchema::groupable(vec![field(DataType::Timestamp(
            TimeUnit::Microsecond,
            Some(other),
        ))])
        .unwrap(),
        runtime.incremental_reservation("fingerprint-changed"),
    );
    assert_ne!(left.layout_fingerprint(), changed.layout_fingerprint());
}

#[test]
fn test_keyed_rejects_wrong_input_type_without_mutation() {
    let runtime = runtime();
    let mut store = store(&runtime, vec![field(DataType::Int64)]);
    let wrong: ArrayRef = Arc::new(datafusion::arrow::array::StringArray::from(vec!["1"]));
    let result = store.route(&[wrong], &mut || Ok(()));
    assert!(result.is_err());
    assert_eq!(store.len(), 0);
    assert_eq!(store.reserved_bytes(), 0);
}

#[test]
fn test_keyed_restore_preserves_group_ids_and_rolls_back_duplicates() {
    let runtime = runtime();
    let array: ArrayRef = Arc::new(datafusion::arrow::array::Int64Array::from(vec![9, 7, 9, 8]));
    let mut original = store(&runtime, vec![field(DataType::Int64)]);
    assert_eq!(
        original
            .route(std::slice::from_ref(&array), &mut || Ok(()))
            .unwrap()
            .iter()
            .map(|group| group.index())
            .collect::<Vec<_>>(),
        [0, 1, 0, 2]
    );
    let groups = (0..3)
        .map(GroupId::new)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let record =
        RecordBatch::try_new(original.schema(), original.columns(&groups).unwrap()).unwrap();
    let mut restored = store(&runtime, vec![field(DataType::Int64)]);
    restored.restore(&record).unwrap();
    assert_eq!(restored.canonical_bytes(), original.canonical_bytes());
    assert_eq!(
        restored
            .route(std::slice::from_ref(&array), &mut || Ok(()))
            .unwrap()
            .iter()
            .map(|group| group.index())
            .collect::<Vec<_>>(),
        [0, 1, 0, 2]
    );

    let duplicated = RecordBatch::try_new(
        original.schema(),
        original
            .columns(&[GroupId::new(0).unwrap(), GroupId::new(0).unwrap()])
            .unwrap(),
    )
    .unwrap();
    let mut rejected = store(&runtime, vec![field(DataType::Int64)]);
    assert!(rejected.restore(&duplicated).is_err());
    assert_eq!(rejected.len(), 0);
    assert_eq!(rejected.reserved_bytes(), 0);
}

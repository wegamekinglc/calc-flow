use datafusion::arrow::{
    array::{
        Array, BooleanArray, Date32Array, Date64Array, Int8Array, Int16Array, Int32Array,
        Int64Array, LargeStringArray, StringArray, TimestampMicrosecondArray,
        TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt8Array,
        UInt16Array, UInt32Array, UInt64Array,
    },
    datatypes::{DataType, TimeUnit},
    record_batch::RecordBatch,
};

#[derive(Clone, Copy, PartialEq)]
enum Value<'a> {
    Signed(i64, usize),
    Unsigned(u64, usize),
    Text(&'a str),
    Boolean(bool),
}

fn value(array: &dyn Array, row: usize) -> Value<'_> {
    macro_rules! signed {
        ($array:ty, $width:expr) => {
            Value::Signed(
                i64::from(
                    array
                        .as_any()
                        .downcast_ref::<$array>()
                        .expect("validated type")
                        .value(row),
                ),
                $width,
            )
        };
    }
    macro_rules! unsigned {
        ($array:ty, $width:expr) => {
            Value::Unsigned(
                u64::from(
                    array
                        .as_any()
                        .downcast_ref::<$array>()
                        .expect("validated type")
                        .value(row),
                ),
                $width,
            )
        };
    }
    match array.data_type() {
        DataType::Int8 => signed!(Int8Array, 1),
        DataType::Int16 => signed!(Int16Array, 2),
        DataType::Int32 => signed!(Int32Array, 4),
        DataType::Int64 => signed!(Int64Array, 8),
        DataType::UInt8 => unsigned!(UInt8Array, 1),
        DataType::UInt16 => unsigned!(UInt16Array, 2),
        DataType::UInt32 => unsigned!(UInt32Array, 4),
        DataType::UInt64 => unsigned!(UInt64Array, 8),
        DataType::Date32 => signed!(Date32Array, 4),
        DataType::Date64 => signed!(Date64Array, 8),
        DataType::Timestamp(TimeUnit::Second, _) => signed!(TimestampSecondArray, 8),
        DataType::Timestamp(TimeUnit::Millisecond, _) => signed!(TimestampMillisecondArray, 8),
        DataType::Timestamp(TimeUnit::Microsecond, _) => signed!(TimestampMicrosecondArray, 8),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => signed!(TimestampNanosecondArray, 8),
        DataType::Boolean => Value::Boolean(
            array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("validated type")
                .value(row),
        ),
        DataType::Utf8 => Value::Text(
            array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("validated type")
                .value(row),
        ),
        DataType::LargeUtf8 => Value::Text(
            array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("validated type")
                .value(row),
        ),
        _ => unreachable!("validated total-order identity type"),
    }
}

pub(super) fn equal(
    a: (&RecordBatch, usize),
    b: (&RecordBatch, usize),
    columns: &[String],
) -> bool {
    columns.iter().all(|name| {
        let index = a.0.schema().index_of(name).expect("validated schema");
        value(a.0.column(index).as_ref(), a.1) == value(b.0.column(index).as_ref(), b.1)
    })
}

pub(super) fn encoded_equal(
    batch: &RecordBatch,
    row: usize,
    columns: &[String],
    mut bytes: &[u8],
) -> bool {
    columns.iter().all(|name| {
        let index = batch.schema().index_of(name).expect("validated schema");
        encoded_value(value(batch.column(index).as_ref(), row), &mut bytes)
    }) && bytes.is_empty()
}

fn prefix(bytes: &mut &[u8], expected: &[u8]) -> bool {
    if !bytes.starts_with(expected) {
        return false;
    }
    *bytes = &bytes[expected.len()..];
    true
}
fn encoded_value(value: Value<'_>, bytes: &mut &[u8]) -> bool {
    match value {
        Value::Boolean(value) => prefix(bytes, &[1, u8::from(value)]),
        Value::Signed(value, width) => {
            let mut encoded = value.to_be_bytes();
            encoded[8 - width] ^= 128;
            prefix(bytes, &[1]) && prefix(bytes, &encoded[8 - width..])
        }
        Value::Unsigned(value, width) => {
            prefix(bytes, &[1]) && prefix(bytes, &value.to_be_bytes()[8 - width..])
        }
        Value::Text(value) => encoded_text(value.as_bytes(), bytes),
    }
}
fn encoded_text(mut value: &[u8], bytes: &mut &[u8]) -> bool {
    if value.is_empty() {
        return prefix(bytes, &[1]);
    }
    if !prefix(bytes, &[2]) {
        return false;
    }
    let mut index = 0;
    while !value.is_empty() {
        let width = if index < 4 { 8 } else { 32 };
        let length = width.min(value.len());
        let mut block = [0_u8; 33];
        block[..length].copy_from_slice(&value[..length]);
        block[width] = if value.len() > width {
            255
        } else {
            u8::try_from(length).expect("bounded block")
        };
        if !prefix(bytes, &block[..=width]) {
            return false;
        }
        value = &value[length..];
        index += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::{
        array::ArrayRef,
        datatypes::{Field, Schema},
        row::{RowConverter, SortField},
    };
    use std::sync::Arc;

    #[test]
    fn asof_fallback_encoded_equality_matches_arrow_typed_rows() {
        let texts = [
            "".to_owned(),
            "é".repeat(4),
            "a".repeat(32),
            "é".repeat(16) + "x",
            "x".repeat(64),
            "y".repeat(65),
        ];
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![i64::MIN, -1, 0, 1, i64::MAX, -999])),
            Arc::new(UInt64Array::from(vec![u64::MAX, 1_u64 << 63, 0, 1, 9, 10])),
            Arc::new(StringArray::from(texts.to_vec())),
        ];
        let names: Vec<String> = vec!["signed".into(), "unsigned".into(), "text".into()];
        let schema = Arc::new(Schema::new(
            names
                .iter()
                .zip(&columns)
                .map(|(name, column)| Field::new(name, column.data_type().clone(), false))
                .collect::<Vec<_>>(),
        ));
        let record = RecordBatch::try_new(schema, columns.clone()).unwrap();
        let converter = RowConverter::new(
            columns
                .iter()
                .map(|column| SortField::new(column.data_type().clone()))
                .collect(),
        )
        .unwrap();
        let rows = converter.convert_columns(&columns).unwrap();
        for incoming in 0..record.num_rows() {
            for existing in 0..record.num_rows() {
                assert_eq!(
                    encoded_equal(&record, incoming, &names, rows.row(existing).as_ref()),
                    incoming == existing
                );
            }
        }
    }
}

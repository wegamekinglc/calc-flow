use super::fail;
use arrow::{
    array::{
        Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, Float32Array, Float64Array,
        Int8Array, Int16Array, Int32Array, Int64Array, StringArray, TimestampMicrosecondArray,
        UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    },
    datatypes::{DataType, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use calc_flow::Result;
use chrono::{Datelike, NaiveDate, Timelike};
use mysql_async::{Row, Value, from_value_opt};
use std::sync::Arc;

pub(super) fn data_type(name: &str, column_type: &str) -> Result<DataType> {
    let unsigned = column_type.contains("unsigned");
    Ok(match (name, unsigned) {
        ("tinyint", false) => DataType::Int8,
        ("tinyint", true) => DataType::UInt8,
        ("smallint", false) => DataType::Int16,
        ("smallint", true) | ("year", _) => DataType::UInt16,
        ("mediumint" | "int", false) => DataType::Int32,
        ("mediumint" | "int", true) => DataType::UInt32,
        ("bigint", false) => DataType::Int64,
        ("bigint", true) => DataType::UInt64,
        ("float", _) => DataType::Float32,
        ("double", _) => DataType::Float64,
        (
            "char" | "varchar" | "tinytext" | "text" | "mediumtext" | "longtext" | "json"
            | "decimal" | "enum" | "set" | "time",
            _,
        ) => DataType::Utf8,
        ("binary" | "varbinary" | "tinyblob" | "blob" | "mediumblob" | "longblob" | "bit", _) => {
            DataType::Binary
        }
        ("date", _) => DataType::Date32,
        ("datetime" | "timestamp", _) => DataType::Timestamp(TimeUnit::Microsecond, None),
        _ => return Err(fail("schema", &format!("unsupported MySQL type {name:?}"))),
    })
}

fn date(value: &Value) -> Result<chrono::NaiveDateTime> {
    if let Value::Date(year, month, day, hour, minute, second, micros) = value {
        return NaiveDate::from_ymd_opt(i32::from(*year), u32::from(*month), u32::from(*day))
            .and_then(|date| {
                date.and_hms_micro_opt(
                    u32::from(*hour),
                    u32::from(*minute),
                    u32::from(*second),
                    *micros,
                )
            })
            .ok_or_else(|| fail("decode", "invalid or zero MySQL date"));
    }
    Err(fail("decode", "expected a MySQL date value"))
}

fn text(value: Value) -> Result<String> {
    match value {
        Value::Bytes(bytes) => {
            String::from_utf8(bytes).map_err(|_| fail("decode", "invalid UTF-8 text"))
        }
        Value::Time(negative, days, hours, minutes, seconds, micros) => Ok(format!(
            "{}{:02}:{minutes:02}:{seconds:02}.{micros:06}",
            if negative { "-" } else { "" },
            u64::from(days) * 24 + u64::from(hours)
        )),
        _ => Err(fail("decode", "expected text or TIME value")),
    }
}

fn values<T>(
    rows: &[Row],
    index: usize,
    convert: impl Fn(Value) -> Result<T>,
) -> Result<Vec<Option<T>>> {
    rows.iter()
        .map(|row| match row.as_ref(index) {
            Some(Value::NULL) => Ok(None),
            Some(value) => convert(value.clone()).map(Some),
            None => Err(fail("decode", "row width differs from schema")),
        })
        .collect()
}

fn column(rows: &[Row], index: usize, kind: &DataType) -> Result<ArrayRef> {
    macro_rules! primitive {
        ($array:ty, $value:ty) => {
            Arc::new(<$array>::from(values(rows, index, |value| {
                from_value_opt::<$value>(value)
                    .map_err(|_| fail("decode", "value is outside its Arrow type"))
            })?))
        };
    }
    Ok(match kind {
        DataType::Int8 => primitive!(Int8Array, i8),
        DataType::Int16 => primitive!(Int16Array, i16),
        DataType::Int32 => primitive!(Int32Array, i32),
        DataType::Int64 => primitive!(Int64Array, i64),
        DataType::UInt8 => primitive!(UInt8Array, u8),
        DataType::UInt16 => primitive!(UInt16Array, u16),
        DataType::UInt32 => primitive!(UInt32Array, u32),
        DataType::UInt64 => primitive!(UInt64Array, u64),
        DataType::Float32 => primitive!(Float32Array, f32),
        DataType::Float64 => primitive!(Float64Array, f64),
        _ => return structured_column(rows, index, kind),
    })
}

fn structured_column(rows: &[Row], index: usize, kind: &DataType) -> Result<ArrayRef> {
    Ok(match kind {
        DataType::Utf8 => Arc::new(StringArray::from(values(rows, index, text)?)),
        DataType::Binary => {
            let bytes = values(rows, index, |value| {
                from_value_opt::<Vec<u8>>(value)
                    .map_err(|_| fail("decode", "expected binary bytes"))
            })?;
            Arc::new(bytes.iter().map(Option::as_deref).collect::<BinaryArray>())
        }
        DataType::Date32 => Arc::new(Date32Array::from(values(rows, index, |value| {
            let days = date(&value)?
                .date()
                .signed_duration_since(NaiveDate::from_ymd_opt(1970, 1, 1).expect("epoch"))
                .num_days();
            i32::try_from(days).map_err(|_| fail("decode", "date out of range"))
        })?)),
        DataType::Timestamp(TimeUnit::Microsecond, None) => Arc::new(
            TimestampMicrosecondArray::from(values(rows, index, |value| {
                Ok(date(&value)?.and_utc().timestamp_micros())
            })?),
        ),
        _ => return Err(fail("decode", "unsupported Arrow type")),
    })
}

pub(super) fn record(rows: &[Row], schema: SchemaRef) -> Result<RecordBatch> {
    let columns = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| {
            let array = column(rows, index, field.data_type())?;
            if !field.is_nullable() && array.null_count() > 0 {
                return Err(fail("decode", "NULL in a non-null column"));
            }
            Ok(array)
        })
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(schema, columns).map_err(|_| fail("decode", "invalid Arrow record batch"))
}

fn mysql_date(value: chrono::NaiveDateTime) -> Result<Value> {
    let year = u16::try_from(value.year())
        .ok()
        .filter(|year| (1000..=9999).contains(year))
        .ok_or_else(|| fail("write", "date outside MySQL range"))?;
    Ok(Value::Date(
        year,
        u8::try_from(value.month()).expect("month"),
        u8::try_from(value.day()).expect("day"),
        u8::try_from(value.hour()).expect("hour"),
        u8::try_from(value.minute()).expect("minute"),
        u8::try_from(value.second()).expect("second"),
        value.nanosecond() / 1000,
    ))
}

pub(super) fn validate_type(kind: &DataType) -> Result<()> {
    match kind {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64
        | DataType::Utf8
        | DataType::Binary
        | DataType::Date32
        | DataType::Timestamp(TimeUnit::Microsecond, _)
        | DataType::Decimal128(_, _) => Ok(()),
        _ => Err(fail("write", "unsupported Arrow column type")),
    }
}

fn typed_array<T: Array + 'static>(array: &ArrayRef) -> Result<&T> {
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| fail("write", "Arrow type mismatch"))
}

pub(super) fn cell(array: &ArrayRef, row: usize) -> Result<Value> {
    validate_type(array.data_type())?;
    if array.is_null(row) {
        return Ok(Value::NULL);
    }
    if array.data_type().is_integer() {
        return integer_cell(array, row);
    }
    noninteger_cell(array, row)
}

fn integer_cell(array: &ArrayRef, row: usize) -> Result<Value> {
    if array.data_type().is_unsigned_integer() {
        return unsigned_cell(array, row);
    }
    signed_cell(array, row)
}

fn signed_cell(array: &ArrayRef, row: usize) -> Result<Value> {
    let value = match array.data_type() {
        DataType::Int8 => i64::from(typed_array::<Int8Array>(array)?.value(row)),
        DataType::Int16 => i64::from(typed_array::<Int16Array>(array)?.value(row)),
        DataType::Int32 => i64::from(typed_array::<Int32Array>(array)?.value(row)),
        DataType::Int64 => typed_array::<Int64Array>(array)?.value(row),
        _ => return Err(fail("write", "expected a signed integer")),
    };
    Ok(Value::Int(value))
}

fn unsigned_cell(array: &ArrayRef, row: usize) -> Result<Value> {
    let value = match array.data_type() {
        DataType::UInt8 => u64::from(typed_array::<UInt8Array>(array)?.value(row)),
        DataType::UInt16 => u64::from(typed_array::<UInt16Array>(array)?.value(row)),
        DataType::UInt32 => u64::from(typed_array::<UInt32Array>(array)?.value(row)),
        DataType::UInt64 => typed_array::<UInt64Array>(array)?.value(row),
        _ => return Err(fail("write", "expected an unsigned integer")),
    };
    Ok(Value::UInt(value))
}

fn noninteger_cell(array: &ArrayRef, row: usize) -> Result<Value> {
    match array.data_type() {
        DataType::Boolean => Ok(Value::Int(i64::from(
            typed_array::<BooleanArray>(array)?.value(row),
        ))),
        DataType::Float32 | DataType::Float64 => float_cell(array, row),
        DataType::Utf8 | DataType::Binary => bytes_cell(array, row),
        DataType::Date32 | DataType::Timestamp(TimeUnit::Microsecond, _) => {
            temporal_cell(array, row)
        }
        DataType::Decimal128(_, _) => {
            arrow::util::display::array_value_to_string(array.as_ref(), row)
                .map(|value| Value::Bytes(value.into_bytes()))
                .map_err(|_| fail("write", "invalid decimal"))
        }
        _ => Err(fail("write", "unsupported Arrow column type")),
    }
}

fn float_cell(array: &ArrayRef, row: usize) -> Result<Value> {
    if array.data_type() == &DataType::Float32 {
        let value = typed_array::<Float32Array>(array)?.value(row);
        require_finite(f64::from(value))?;
        return Ok(Value::Float(value));
    }
    let value = typed_array::<Float64Array>(array)?.value(row);
    require_finite(value)?;
    Ok(Value::Double(value))
}

fn require_finite(value: f64) -> Result<()> {
    if !value.is_finite() {
        return Err(fail("write", "MySQL cannot store non-finite floats"));
    }
    Ok(())
}

fn bytes_cell(array: &ArrayRef, row: usize) -> Result<Value> {
    let bytes = if array.data_type() == &DataType::Utf8 {
        typed_array::<StringArray>(array)?
            .value(row)
            .as_bytes()
            .to_vec()
    } else {
        typed_array::<BinaryArray>(array)?.value(row).to_vec()
    };
    Ok(Value::Bytes(bytes))
}

fn temporal_cell(array: &ArrayRef, row: usize) -> Result<Value> {
    if array.data_type() == &DataType::Date32 {
        let days = typed_array::<Date32Array>(array)?.value(row);
        let value = NaiveDate::from_ymd_opt(1970, 1, 1)
            .expect("epoch")
            .checked_add_signed(chrono::Duration::days(i64::from(days)))
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .ok_or_else(|| fail("write", "date out of range"))?;
        return mysql_date(value);
    }
    let micros = typed_array::<TimestampMicrosecondArray>(array)?.value(row);
    let value = chrono::DateTime::from_timestamp_micros(micros)
        .ok_or_else(|| fail("write", "timestamp out of range"))?;
    mysql_date(value.naive_utc())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_database_values_fail_without_echoing_payloads() {
        assert!(data_type("geometry", "geometry").is_err());
        assert!(date(&Value::Date(0, 0, 0, 0, 0, 0, 0)).is_err());
        assert!(date(&Value::Int(1)).is_err());
        assert!(text(Value::Bytes(vec![255])).is_err());
        assert!(text(Value::Int(1)).is_err());
        assert_eq!(
            text(Value::Time(false, 0, 1, 2, 3, 4)).unwrap(),
            "01:02:03.000004"
        );
        let too_early = NaiveDate::from_ymd_opt(999, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        assert!(mysql_date(too_early).is_err());
    }

    #[test]
    fn sink_rejects_nonfinite_unsupported_and_out_of_range_values() {
        for array in [
            Arc::new(Float32Array::from(vec![f32::NAN])) as ArrayRef,
            Arc::new(Float64Array::from(vec![f64::INFINITY])),
            Arc::new(Date32Array::from(vec![i32::MAX])),
            Arc::new(TimestampMicrosecondArray::from(vec![i64::MAX])),
            Arc::new(arrow::array::LargeStringArray::from(vec!["secret"])),
        ] {
            let error = cell(&array, 0).unwrap_err().to_string();
            assert!(!error.contains("secret"));
        }
        assert_eq!(
            cell(&(Arc::new(BooleanArray::from(vec![true])) as ArrayRef), 0).unwrap(),
            Value::Int(1)
        );
        let decimal: ArrayRef = Arc::new(
            arrow::array::Decimal128Array::from(vec![123_456_789_012_345_678_901_234_567_890_i128])
                .with_precision_and_scale(38, 10)
                .unwrap(),
        );
        assert_eq!(
            cell(&decimal, 0).unwrap(),
            Value::Bytes(b"12345678901234567890.1234567890".to_vec())
        );
    }
}

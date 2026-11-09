use super::{invalid, typed};
use crate::Result;
use datafusion::arrow::{
    array::{
        Array, ArrayData, BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Date64Array,
        Decimal32Array, Decimal64Array, Decimal128Array, Decimal256Array, DurationMicrosecondArray,
        DurationMillisecondArray, DurationNanosecondArray, DurationSecondArray,
        FixedSizeBinaryArray, Float16Array, Float32Array, Float64Array, Int8Array, Int16Array,
        Int32Array, Int64Array, IntervalDayTimeArray, IntervalMonthDayNanoArray,
        IntervalYearMonthArray, LargeBinaryArray, LargeStringArray, NullArray, PrimitiveArray,
        StringArray, StringViewArray, Time32MillisecondArray, Time32SecondArray,
        Time64MicrosecondArray, Time64NanosecondArray, UInt8Array, UInt16Array, UInt32Array,
        UInt64Array,
    },
    datatypes::{DataType as D, IntervalUnit as I, TimeUnit as U},
};

type Converted = Option<Result<ArrayData>>;
fn value<T: Array + 'static>(
    array: &dyn Array,
    offset: usize,
    len: usize,
    node: &str,
) -> Result<ArrayData> {
    let array = typed::<T>(array, node)?;
    if offset.checked_add(len).is_none_or(|end| end > array.len()) {
        return Err(invalid(node));
    }
    Ok(array.to_data().slice(offset, len))
}
macro_rules! convert {
    ($array:expr,$offset:expr,$len:expr,$node:expr,$ty:ty) => {
        Some(value::<$ty>($array, $offset, $len, $node))
    };
}

pub(super) fn data(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &D,
    node: &str,
) -> Converted {
    match array.data_type() {
        D::Int8 => convert!(array, offset, len, node, Int8Array),
        D::Int16 => convert!(array, offset, len, node, Int16Array),
        D::Int32 => convert!(array, offset, len, node, Int32Array),
        D::Int64 => convert!(array, offset, len, node, Int64Array),
        _ => unsigned(array, offset, len, expected_dtype, node),
    }
}
fn unsigned(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &D,
    node: &str,
) -> Converted {
    match array.data_type() {
        D::UInt8 => convert!(array, offset, len, node, UInt8Array),
        D::UInt16 => convert!(array, offset, len, node, UInt16Array),
        D::UInt32 => convert!(array, offset, len, node, UInt32Array),
        D::UInt64 => convert!(array, offset, len, node, UInt64Array),
        _ => floating(array, offset, len, expected_dtype, node),
    }
}
fn floating(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &D,
    node: &str,
) -> Converted {
    match array.data_type() {
        D::Float16 => convert!(array, offset, len, node, Float16Array),
        D::Float32 => convert!(array, offset, len, node, Float32Array),
        D::Float64 => convert!(array, offset, len, node, Float64Array),
        D::Date32 => convert!(array, offset, len, node, Date32Array),
        D::Date64 => convert!(array, offset, len, node, Date64Array),
        _ => time(array, offset, len, expected_dtype, node),
    }
}
fn time(array: &dyn Array, offset: usize, len: usize, expected_dtype: &D, node: &str) -> Converted {
    match array.data_type() {
        D::Time32(U::Second) => convert!(array, offset, len, node, Time32SecondArray),
        D::Time32(U::Millisecond) => convert!(array, offset, len, node, Time32MillisecondArray),
        D::Time64(U::Microsecond) => convert!(array, offset, len, node, Time64MicrosecondArray),
        D::Time64(U::Nanosecond) => convert!(array, offset, len, node, Time64NanosecondArray),
        _ => timestamp(array, offset, len, expected_dtype, node),
    }
}
fn timestamp(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &D,
    node: &str,
) -> Converted {
    match array.data_type() {
        D::Timestamp(U::Second, _) => Some(timestamp_data::<
            datafusion::arrow::datatypes::TimestampSecondType,
        >(array, offset, len, expected_dtype, node)),
        D::Timestamp(U::Millisecond, _) => Some(timestamp_data::<
            datafusion::arrow::datatypes::TimestampMillisecondType,
        >(array, offset, len, expected_dtype, node)),
        D::Timestamp(U::Microsecond, _) => Some(timestamp_data::<
            datafusion::arrow::datatypes::TimestampMicrosecondType,
        >(array, offset, len, expected_dtype, node)),
        D::Timestamp(U::Nanosecond, _) => Some(timestamp_data::<
            datafusion::arrow::datatypes::TimestampNanosecondType,
        >(array, offset, len, expected_dtype, node)),
        _ => duration(array, offset, len, expected_dtype, node),
    }
}
fn duration(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &D,
    node: &str,
) -> Converted {
    match array.data_type() {
        D::Duration(U::Second) => convert!(array, offset, len, node, DurationSecondArray),
        D::Duration(U::Millisecond) => convert!(array, offset, len, node, DurationMillisecondArray),
        D::Duration(U::Microsecond) => convert!(array, offset, len, node, DurationMicrosecondArray),
        D::Duration(U::Nanosecond) => convert!(array, offset, len, node, DurationNanosecondArray),
        _ => interval(array, offset, len, expected_dtype, node),
    }
}
fn interval(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &D,
    node: &str,
) -> Converted {
    match array.data_type() {
        D::Interval(I::YearMonth) => convert!(array, offset, len, node, IntervalYearMonthArray),
        D::Interval(I::DayTime) => convert!(array, offset, len, node, IntervalDayTimeArray),
        D::Interval(I::MonthDayNano) => {
            convert!(array, offset, len, node, IntervalMonthDayNanoArray)
        }
        _ => bytes(array, offset, len, expected_dtype, node),
    }
}
fn bytes(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &D,
    node: &str,
) -> Converted {
    match array.data_type() {
        D::Utf8 => convert!(array, offset, len, node, StringArray),
        D::LargeUtf8 => convert!(array, offset, len, node, LargeStringArray),
        D::Utf8View => convert!(array, offset, len, node, StringViewArray),
        D::Binary => convert!(array, offset, len, node, BinaryArray),
        D::LargeBinary => convert!(array, offset, len, node, LargeBinaryArray),
        D::BinaryView => convert!(array, offset, len, node, BinaryViewArray),
        _ => decimal(array, offset, len, expected_dtype, node),
    }
}
fn decimal(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected_dtype: &D,
    node: &str,
) -> Converted {
    match array.data_type() {
        D::Decimal32(..) => convert!(array, offset, len, node, Decimal32Array),
        D::Decimal64(..) => convert!(array, offset, len, node, Decimal64Array),
        D::Decimal128(..) => convert!(array, offset, len, node, Decimal128Array),
        D::Decimal256(..) => convert!(array, offset, len, node, Decimal256Array),
        _ => special(array, offset, len, expected_dtype, node),
    }
}
fn special(
    array: &dyn Array,
    offset: usize,
    len: usize,
    _expected_dtype: &D,
    node: &str,
) -> Converted {
    match array.data_type() {
        D::Null => convert!(array, offset, len, node, NullArray),
        D::Boolean => convert!(array, offset, len, node, BooleanArray),
        D::FixedSizeBinary(_) => convert!(array, offset, len, node, FixedSizeBinaryArray),
        _ => None,
    }
}

fn timestamp_data<T: datafusion::arrow::datatypes::ArrowTimestampType>(
    array: &dyn Array,
    offset: usize,
    len: usize,
    expected: &D,
    node: &str,
) -> Result<ArrayData> {
    let array = typed::<PrimitiveArray<T>>(array, node)?;
    let array = array.clone().with_data_type(expected.clone());
    Ok(array.to_data().slice(offset, len))
}

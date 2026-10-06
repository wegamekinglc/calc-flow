use super::{
    AccumulatorValue, AggregateFunction, DataType, Ordering, ScalarColumn, ScalarValue,
    TypedValues, update_accumulator, update_count, update_extreme,
};

pub(super) type UpdateKernel =
    fn(&ScalarColumn<'_>, usize, &mut AccumulatorValue) -> Result<(), String>;

const SUM: u8 = 0;
const MIN: u8 = 1;
const MAX: u8 = 2;
const AVG: u8 = 3;

#[inline]
fn apply<const FUNCTION: u8>(
    accumulator: &mut AccumulatorValue,
    value: ScalarValue,
) -> Result<(), String> {
    let function = match FUNCTION {
        SUM => AggregateFunction::Sum,
        MIN => AggregateFunction::Min,
        MAX => AggregateFunction::Max,
        AVG => AggregateFunction::Avg,
        _ => unreachable!("compiled aggregate function mismatch"),
    };
    update_accumulator(accumulator, function, value)
}

macro_rules! numeric_kernel {
    ($name:ident, $array:ident, $scalar:ident, $convert:expr) => {
        fn $name<const FUNCTION: u8>(
            column: &ScalarColumn<'_>,
            row: usize,
            accumulator: &mut AccumulatorValue,
        ) -> Result<(), String> {
            let TypedValues::$array(array) = column.values else {
                return Err("compiled aggregate column type mismatch".into());
            };
            apply::<FUNCTION>(
                accumulator,
                ScalarValue::$scalar(($convert)(array.value(row))),
            )
        }
    };
}

numeric_kernel!(int8, Int8, Signed, i64::from);
numeric_kernel!(int16, Int16, Signed, i64::from);
numeric_kernel!(int32, Int32, Signed, i64::from);
numeric_kernel!(int64, Int64, Signed, std::convert::identity);
numeric_kernel!(uint8, UInt8, Unsigned, u64::from);
numeric_kernel!(uint16, UInt16, Unsigned, u64::from);
numeric_kernel!(uint32, UInt32, Unsigned, u64::from);
numeric_kernel!(uint64, UInt64, Unsigned, std::convert::identity);
numeric_kernel!(float32, Float32, Float32, f32::to_bits);
numeric_kernel!(float64, Float64, Float64, f64::to_bits);

fn count(_: &ScalarColumn<'_>, _: usize, accumulator: &mut AccumulatorValue) -> Result<(), String> {
    update_count(accumulator)
}

fn string_extreme<const FUNCTION: u8>(
    column: &ScalarColumn<'_>,
    row: usize,
    accumulator: &mut AccumulatorValue,
) -> Result<(), String> {
    let value = match column.values {
        TypedValues::Utf8(array) => array.value(row),
        TypedValues::LargeUtf8(array) => array.value(row),
        _ => return Err("compiled aggregate column type mismatch".into()),
    };
    let (AccumulatorValue::Min(current) | AccumulatorValue::Max(current)) = accumulator else {
        return Err("compiled aggregate accumulator type mismatch".into());
    };
    let ordering = if FUNCTION == MIN {
        Ordering::Less
    } else {
        Ordering::Greater
    };
    let replace = match current {
        None => true,
        Some(ScalarValue::String(current)) => value.cmp(current.as_str()) == ordering,
        Some(_) => return Err("compiled aggregate scalar type mismatch".into()),
    };
    if replace {
        *current = Some(ScalarValue::String(value.into()));
    }
    Ok(())
}

fn scalar_extreme<const FUNCTION: u8>(
    column: &ScalarColumn<'_>,
    row: usize,
    accumulator: &mut AccumulatorValue,
) -> Result<(), String> {
    update_extreme(
        accumulator,
        column.value(row),
        if FUNCTION == MIN {
            Ordering::Less
        } else {
            Ordering::Greater
        },
    )
}

pub(super) fn select(function: AggregateFunction, data_type: &DataType) -> UpdateKernel {
    if function == AggregateFunction::Count {
        return count;
    }
    macro_rules! select_numeric {
        ($kernel:ident) => {
            match function {
                AggregateFunction::Sum => $kernel::<SUM>,
                AggregateFunction::Min => $kernel::<MIN>,
                AggregateFunction::Max => $kernel::<MAX>,
                AggregateFunction::Avg => $kernel::<AVG>,
                AggregateFunction::Count => unreachable!("count kernel selected already"),
            }
        };
    }
    match data_type {
        DataType::Int8 => select_numeric!(int8),
        DataType::Int16 => select_numeric!(int16),
        DataType::Int32 => select_numeric!(int32),
        DataType::Int64 => select_numeric!(int64),
        DataType::UInt8 => select_numeric!(uint8),
        DataType::UInt16 => select_numeric!(uint16),
        DataType::UInt32 => select_numeric!(uint32),
        DataType::UInt64 => select_numeric!(uint64),
        DataType::Float32 => select_numeric!(float32),
        DataType::Float64 => select_numeric!(float64),
        DataType::Utf8 | DataType::LargeUtf8 => match function {
            AggregateFunction::Min => string_extreme::<MIN>,
            AggregateFunction::Max => string_extreme::<MAX>,
            _ => unreachable!("string aggregate matrix validated at construction"),
        },
        _ => match function {
            AggregateFunction::Min => scalar_extreme::<MIN>,
            AggregateFunction::Max => scalar_extreme::<MAX>,
            _ => unreachable!("aggregate matrix validated at construction"),
        },
    }
}

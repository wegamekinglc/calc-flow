use super::{
    AccumulatorValue, AggregateFunction, CompensatedSum, DataType, Ordering, ScalarColumn,
    ScalarValue, SlotColumn, TypedValues, increment_average_count, scalar_total_cmp,
    update_accumulator, update_count, update_extreme, update_signed_average, update_signed_sum,
    update_unsigned_average, update_unsigned_sum,
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

// ---------------------------------------------------------------------------
// Columnar fast path: pass one resolves (slot, row) pairs; these loops run
// once per aggregate with dense per-slot accumulators. Per-slot updates apply
// in row order and reuse the row-wise scalar helpers, so accumulator state is
// bit-identical to the reference path.
// ---------------------------------------------------------------------------

/// One tight loop per aggregate over the (slot, row) pairs pass one resolved.
pub(super) fn update_column(
    column: &mut SlotColumn,
    values: &ScalarColumn<'_>,
    pairs: &[(u32, u32)],
) -> Result<(), String> {
    match column {
        SlotColumn::Count(counts) => update_count_column(counts, values, pairs),
        SlotColumn::SignedSum(sums) => update_signed_sum_column(sums, values, pairs),
        SlotColumn::UnsignedSum(sums) => update_unsigned_sum_column(sums, values, pairs),
        SlotColumn::FloatSum(sums) => update_float_sum_column(sums, values, pairs),
        SlotColumn::Min(extremes) => {
            update_extreme_column(extremes, values, pairs, Ordering::Less);
            Ok(())
        }
        SlotColumn::Max(extremes) => {
            update_extreme_column(extremes, values, pairs, Ordering::Greater);
            Ok(())
        }
        SlotColumn::SignedAverage(entries) => update_signed_average_column(entries, values, pairs),
        SlotColumn::UnsignedAverage(entries) => {
            update_unsigned_average_column(entries, values, pairs)
        }
        SlotColumn::FloatAverage(entries) => update_float_average_column(entries, values, pairs),
    }
}

fn update_count_column(
    counts: &mut [u64],
    values: &ScalarColumn<'_>,
    pairs: &[(u32, u32)],
) -> Result<(), String> {
    for &(slot, row) in pairs {
        if values.is_null(row as usize) {
            continue;
        }
        let count = &mut counts[slot as usize];
        *count = count
            .checked_add(1)
            .ok_or_else(|| "count overflowed UInt64".to_string())?;
    }
    Ok(())
}

/// Tight numeric loop over typed arrays: the type match is hoisted out of the
/// pair loop, and the per-slot update reuses the row-wise scalar helper.
macro_rules! signed_column_loop {
    ($values:expr, $pairs:expr, $variant:ident, $convert:expr, $body:expr) => {{
        let TypedValues::$variant(array) = $values.values else {
            return Err("compiled aggregate column type mismatch".into());
        };
        for &(slot, row) in $pairs {
            if $values.is_null(row as usize) {
                continue;
            }
            $body(slot, ($convert)(array.value(row as usize)))?;
        }
        Ok::<(), String>(())
    }};
}

fn update_signed_sum_column(
    sums: &mut [Option<i128>],
    values: &ScalarColumn<'_>,
    pairs: &[(u32, u32)],
) -> Result<(), String> {
    match values.values {
        TypedValues::Int8(_) => {
            signed_column_loop!(values, pairs, Int8, i64::from, |slot, value| {
                update_signed_sum(&mut sums[slot as usize], &ScalarValue::Signed(value))
            })
        }
        TypedValues::Int16(_) => {
            signed_column_loop!(values, pairs, Int16, i64::from, |slot, value| {
                update_signed_sum(&mut sums[slot as usize], &ScalarValue::Signed(value))
            })
        }
        TypedValues::Int32(_) => {
            signed_column_loop!(values, pairs, Int32, i64::from, |slot, value| {
                update_signed_sum(&mut sums[slot as usize], &ScalarValue::Signed(value))
            })
        }
        TypedValues::Int64(_) => {
            signed_column_loop!(
                values,
                pairs,
                Int64,
                std::convert::identity,
                |slot, value| {
                    update_signed_sum(&mut sums[slot as usize], &ScalarValue::Signed(value))
                }
            )
        }
        _ => Err("compiled aggregate column type mismatch".into()),
    }
}

fn update_unsigned_sum_column(
    sums: &mut [Option<u128>],
    values: &ScalarColumn<'_>,
    pairs: &[(u32, u32)],
) -> Result<(), String> {
    match values.values {
        TypedValues::UInt8(_) => {
            signed_column_loop!(values, pairs, UInt8, u64::from, |slot, value| {
                update_unsigned_sum(&mut sums[slot as usize], &ScalarValue::Unsigned(value))
            })
        }
        TypedValues::UInt16(_) => {
            signed_column_loop!(values, pairs, UInt16, u64::from, |slot, value| {
                update_unsigned_sum(&mut sums[slot as usize], &ScalarValue::Unsigned(value))
            })
        }
        TypedValues::UInt32(_) => {
            signed_column_loop!(values, pairs, UInt32, u64::from, |slot, value| {
                update_unsigned_sum(&mut sums[slot as usize], &ScalarValue::Unsigned(value))
            })
        }
        TypedValues::UInt64(_) => {
            signed_column_loop!(
                values,
                pairs,
                UInt64,
                std::convert::identity,
                |slot, value| {
                    update_unsigned_sum(&mut sums[slot as usize], &ScalarValue::Unsigned(value))
                }
            )
        }
        _ => Err("compiled aggregate column type mismatch".into()),
    }
}

fn update_float_sum_column(
    sums: &mut [Option<CompensatedSum>],
    values: &ScalarColumn<'_>,
    pairs: &[(u32, u32)],
) -> Result<(), String> {
    match values.values {
        TypedValues::Float32(_) => {
            signed_column_loop!(values, pairs, Float32, f64::from, |slot, value| {
                sums[slot as usize]
                    .get_or_insert(CompensatedSum::new(0.0))
                    .add(value);
                Ok::<(), String>(())
            })
        }
        TypedValues::Float64(_) => {
            signed_column_loop!(
                values,
                pairs,
                Float64,
                std::convert::identity,
                |slot, value| {
                    sums[slot as usize]
                        .get_or_insert(CompensatedSum::new(0.0))
                        .add(value);
                    Ok::<(), String>(())
                }
            )
        }
        _ => Err("compiled aggregate column type mismatch".into()),
    }
}

fn update_extreme_column(
    extremes: &mut [Option<ScalarValue>],
    values: &ScalarColumn<'_>,
    pairs: &[(u32, u32)],
    ordering: Ordering,
) {
    for &(slot, row) in pairs {
        if values.is_null(row as usize) {
            continue;
        }
        let value = values.value(row as usize);
        let current = &mut extremes[slot as usize];
        if current
            .as_ref()
            .is_none_or(|current| scalar_total_cmp(&value, current) == ordering)
        {
            *current = Some(value);
        }
    }
}

fn update_signed_average_column(
    entries: &mut [(i128, u64)],
    values: &ScalarColumn<'_>,
    pairs: &[(u32, u32)],
) -> Result<(), String> {
    match values.values {
        TypedValues::Int8(_) => {
            signed_column_loop!(values, pairs, Int8, i64::from, |slot, value| {
                let (sum, count) = &mut entries[slot as usize];
                update_signed_average(sum, count, &ScalarValue::Signed(value))
            })
        }
        TypedValues::Int16(_) => {
            signed_column_loop!(values, pairs, Int16, i64::from, |slot, value| {
                let (sum, count) = &mut entries[slot as usize];
                update_signed_average(sum, count, &ScalarValue::Signed(value))
            })
        }
        TypedValues::Int32(_) => {
            signed_column_loop!(values, pairs, Int32, i64::from, |slot, value| {
                let (sum, count) = &mut entries[slot as usize];
                update_signed_average(sum, count, &ScalarValue::Signed(value))
            })
        }
        TypedValues::Int64(_) => {
            signed_column_loop!(
                values,
                pairs,
                Int64,
                std::convert::identity,
                |slot, value| {
                    let (sum, count) = &mut entries[slot as usize];
                    update_signed_average(sum, count, &ScalarValue::Signed(value))
                }
            )
        }
        _ => Err("compiled aggregate column type mismatch".into()),
    }
}

fn update_unsigned_average_column(
    entries: &mut [(u128, u64)],
    values: &ScalarColumn<'_>,
    pairs: &[(u32, u32)],
) -> Result<(), String> {
    match values.values {
        TypedValues::UInt8(_) => {
            signed_column_loop!(values, pairs, UInt8, u64::from, |slot, value| {
                let (sum, count) = &mut entries[slot as usize];
                update_unsigned_average(sum, count, &ScalarValue::Unsigned(value))
            })
        }
        TypedValues::UInt16(_) => {
            signed_column_loop!(values, pairs, UInt16, u64::from, |slot, value| {
                let (sum, count) = &mut entries[slot as usize];
                update_unsigned_average(sum, count, &ScalarValue::Unsigned(value))
            })
        }
        TypedValues::UInt32(_) => {
            signed_column_loop!(values, pairs, UInt32, u64::from, |slot, value| {
                let (sum, count) = &mut entries[slot as usize];
                update_unsigned_average(sum, count, &ScalarValue::Unsigned(value))
            })
        }
        TypedValues::UInt64(_) => {
            signed_column_loop!(
                values,
                pairs,
                UInt64,
                std::convert::identity,
                |slot, value| {
                    let (sum, count) = &mut entries[slot as usize];
                    update_unsigned_average(sum, count, &ScalarValue::Unsigned(value))
                }
            )
        }
        _ => Err("compiled aggregate column type mismatch".into()),
    }
}

fn update_float_average_column(
    entries: &mut [(CompensatedSum, u64)],
    values: &ScalarColumn<'_>,
    pairs: &[(u32, u32)],
) -> Result<(), String> {
    match values.values {
        TypedValues::Float32(_) => {
            signed_column_loop!(values, pairs, Float32, f64::from, |slot, value| {
                let (sum, count) = &mut entries[slot as usize];
                sum.add(value);
                increment_average_count(count)
            })
        }
        TypedValues::Float64(_) => {
            signed_column_loop!(
                values,
                pairs,
                Float64,
                std::convert::identity,
                |slot, value| {
                    let (sum, count) = &mut entries[slot as usize];
                    sum.add(value);
                    increment_average_count(count)
                }
            )
        }
        _ => Err("compiled aggregate column type mismatch".into()),
    }
}

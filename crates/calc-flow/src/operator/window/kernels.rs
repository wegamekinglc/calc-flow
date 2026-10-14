use super::{
    AccumulatorValue, AggregateFunction, CompensatedSum, DataType, ExtremeColumn, NullBuffer,
    Ordering, ScalarColumn, ScalarValue, SlotColumn, TypedValues, increment_average_count,
    scalar_total_cmp, update_accumulator, update_count, update_extreme, update_signed_average,
    update_signed_sum, update_unsigned_average, update_unsigned_sum,
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
        SlotColumn::Count(counts) => for_each_pair(
            values.nulls,
            pairs,
            |_| (),
            |slot, ()| {
                let count = &mut counts[slot];
                *count = count
                    .checked_add(1)
                    .ok_or_else(|| "count overflowed UInt64".to_string())?;
                Ok(())
            },
        ),
        SlotColumn::SignedSum(sums) => signed_pairs(values, pairs, |slot, value| {
            update_signed_sum(&mut sums[slot], &ScalarValue::Signed(value))
        }),
        SlotColumn::UnsignedSum(sums) => unsigned_pairs(values, pairs, |slot, value| {
            update_unsigned_sum(&mut sums[slot], &ScalarValue::Unsigned(value))
        }),
        SlotColumn::FloatSum(sums) => float_pairs(values, pairs, |slot, value| {
            sums[slot]
                .get_or_insert(CompensatedSum::new(0.0))
                .add(value);
            Ok(())
        }),
        SlotColumn::Min(extremes) => update_extreme_column(extremes, values, pairs, Ordering::Less),
        SlotColumn::Max(extremes) => {
            update_extreme_column(extremes, values, pairs, Ordering::Greater)
        }
        SlotColumn::SignedAverage(entries) => signed_pairs(values, pairs, |slot, value| {
            let (sum, count) = &mut entries[slot];
            update_signed_average(sum, count, &ScalarValue::Signed(value))
        }),
        SlotColumn::UnsignedAverage(entries) => unsigned_pairs(values, pairs, |slot, value| {
            let (sum, count) = &mut entries[slot];
            update_unsigned_average(sum, count, &ScalarValue::Unsigned(value))
        }),
        SlotColumn::FloatAverage(entries) => float_pairs(values, pairs, |slot, value| {
            let (sum, count) = &mut entries[slot];
            sum.add(value);
            increment_average_count(count)
        }),
    }
}

/// Applies `update` to every pair whose row is valid, in pair order. The
/// validity check is hoisted out of the loop for null-free columns.
fn for_each_pair<T>(
    nulls: Option<&NullBuffer>,
    pairs: &[(u32, u32)],
    read: impl Fn(usize) -> T,
    mut update: impl FnMut(usize, T) -> Result<(), String>,
) -> Result<(), String> {
    match nulls {
        None => pairs
            .iter()
            .try_for_each(|&(slot, row)| update(slot as usize, read(row as usize))),
        Some(nulls) => pairs
            .iter()
            .filter(|&&(_, row)| nulls.is_valid(row as usize))
            .try_for_each(|&(slot, row)| update(slot as usize, read(row as usize))),
    }
}

fn type_mismatch() -> Result<(), String> {
    Err("compiled aggregate column type mismatch".into())
}

/// Runs `update` over a signed integer column widened to `i64`.
fn signed_pairs(
    values: &ScalarColumn<'_>,
    pairs: &[(u32, u32)],
    update: impl FnMut(usize, i64) -> Result<(), String>,
) -> Result<(), String> {
    let nulls = values.nulls;
    match values.values {
        TypedValues::Int8(array) => {
            for_each_pair(nulls, pairs, |row| i64::from(array.value(row)), update)
        }
        TypedValues::Int16(array) => {
            for_each_pair(nulls, pairs, |row| i64::from(array.value(row)), update)
        }
        TypedValues::Int32(array) => {
            for_each_pair(nulls, pairs, |row| i64::from(array.value(row)), update)
        }
        TypedValues::Int64(array) => for_each_pair(nulls, pairs, |row| array.value(row), update),
        _ => type_mismatch(),
    }
}

/// Runs `update` over an unsigned integer column widened to `u64`.
fn unsigned_pairs(
    values: &ScalarColumn<'_>,
    pairs: &[(u32, u32)],
    update: impl FnMut(usize, u64) -> Result<(), String>,
) -> Result<(), String> {
    let nulls = values.nulls;
    match values.values {
        TypedValues::UInt8(array) => {
            for_each_pair(nulls, pairs, |row| u64::from(array.value(row)), update)
        }
        TypedValues::UInt16(array) => {
            for_each_pair(nulls, pairs, |row| u64::from(array.value(row)), update)
        }
        TypedValues::UInt32(array) => {
            for_each_pair(nulls, pairs, |row| u64::from(array.value(row)), update)
        }
        TypedValues::UInt64(array) => for_each_pair(nulls, pairs, |row| array.value(row), update),
        _ => type_mismatch(),
    }
}

/// Runs `update` over a floating-point column widened to `f64`.
fn float_pairs(
    values: &ScalarColumn<'_>,
    pairs: &[(u32, u32)],
    update: impl FnMut(usize, f64) -> Result<(), String>,
) -> Result<(), String> {
    let nulls = values.nulls;
    match values.values {
        TypedValues::Float32(array) => {
            for_each_pair(nulls, pairs, |row| f64::from(array.value(row)), update)
        }
        TypedValues::Float64(array) => for_each_pair(nulls, pairs, |row| array.value(row), update),
        _ => type_mismatch(),
    }
}

/// Replaces a witness when the new value orders strictly before (min) or
/// after (max) it under `compare`, matching `scalar_total_cmp`.
fn replace_extreme<T: Copy>(
    current: &mut Option<T>,
    value: T,
    ordering: Ordering,
    compare: impl Fn(T, T) -> Ordering,
) {
    if current.is_none_or(|current| compare(value, current) == ordering) {
        *current = Some(value);
    }
}

fn update_extreme_column(
    extremes: &mut ExtremeColumn,
    values: &ScalarColumn<'_>,
    pairs: &[(u32, u32)],
    ordering: Ordering,
) -> Result<(), String> {
    let nulls = values.nulls;
    match (extremes, values.values) {
        (ExtremeColumn::Signed(extremes), _) => signed_pairs(values, pairs, |slot, value| {
            replace_extreme(&mut extremes[slot], value, ordering, |left, right| {
                left.cmp(&right)
            });
            Ok(())
        }),
        (ExtremeColumn::Unsigned(extremes), _) => unsigned_pairs(values, pairs, |slot, value| {
            replace_extreme(&mut extremes[slot], value, ordering, |left, right| {
                left.cmp(&right)
            });
            Ok(())
        }),
        (ExtremeColumn::Float32(extremes), TypedValues::Float32(array)) => for_each_pair(
            nulls,
            pairs,
            |row| array.value(row).to_bits(),
            |slot, bits| {
                replace_extreme(&mut extremes[slot], bits, ordering, |left, right| {
                    f32::from_bits(left).total_cmp(&f32::from_bits(right))
                });
                Ok(())
            },
        ),
        (ExtremeColumn::Float64(extremes), TypedValues::Float64(array)) => for_each_pair(
            nulls,
            pairs,
            |row| array.value(row).to_bits(),
            |slot, bits| {
                replace_extreme(&mut extremes[slot], bits, ordering, |left, right| {
                    f64::from_bits(left).total_cmp(&f64::from_bits(right))
                });
                Ok(())
            },
        ),
        (ExtremeColumn::Scalar(extremes), _) => for_each_pair(
            nulls,
            pairs,
            |row| values.value(row),
            |slot, value| {
                let current = &mut extremes[slot];
                if current
                    .as_ref()
                    .is_none_or(|current| scalar_total_cmp(&value, current) == ordering)
                {
                    *current = Some(value);
                }
                Ok(())
            },
        ),
        _ => type_mismatch(),
    }
}

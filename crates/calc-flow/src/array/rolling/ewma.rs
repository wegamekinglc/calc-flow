use datafusion::arrow::buffer::{BooleanBuffer, Buffer, NullBuffer};

use crate::{
    Result,
    array::{ArrayExecutionOptions, FloatValues, FloatView, NativeArray},
};

/// Parameters for a first-sample-seeded, unadjusted exponential average.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EwmaOptions {
    span: u64,
    min_periods: u64,
}

impl EwmaOptions {
    /// Creates an average with `alpha = 2 / (span + 1)` and `min_periods = 1`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CalcFlowError::InvalidArgument`] if `span` is zero.
    pub fn new(span: u64) -> Result<Self> {
        if span == 0 {
            return Err(super::super::invalid("options.span", "must be positive"));
        }
        Ok(Self {
            span,
            min_periods: 1,
        })
    }

    /// Returns parameters with a new positive valid-sample warm-up count.
    ///
    /// Counts above `span` are accepted. Null and NaN inputs do not advance it.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CalcFlowError::InvalidArgument`] if `min_periods` is zero.
    pub fn with_min_periods(self, min_periods: u64) -> Result<Self> {
        if min_periods == 0 {
            return Err(super::super::invalid(
                "options.min_periods",
                "must be positive",
            ));
        }
        Ok(Self {
            min_periods,
            ..self
        })
    }

    /// Returns the exponential span.
    pub const fn span(&self) -> u64 {
        self.span
    }

    /// Returns the minimum number of valid samples before producing values.
    pub const fn min_periods(&self) -> u64 {
        self.min_periods
    }
}

/// Computes an unadjusted exponential average using Calc Flow's table semantics.
///
/// The first valid sample seeds the recurrence. Null and NaN inputs leave the
/// state unchanged; after warm-up their output carries the preceding value.
/// Infinity is a valid sample, and IEEE arithmetic can produce valid NaN output.
/// Finite-input arithmetic retains the same table recurrence's overflow behavior.
/// Inputs remain read-only and float32 inputs produce float64 output.
///
/// ```
/// use calc_flow::{Result, array::{ArrayExecutionOptions, EwmaOptions, FloatView, ewma}};
/// # fn main() -> Result<()> {
/// let output = ewma(
///     FloatView::from_f64(&[10.0, 14.0, 18.0]),
///     EwmaOptions::new(3)?,
///     &ArrayExecutionOptions::default(),
/// )?;
/// assert_eq!(output.len(), 3);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`crate::CalcFlowError::InvalidArgument`] before execution when the
/// output elements or allocated output-and-state bytes exceed the limits,
/// size or byte counts overflow, or a result buffer cannot be allocated.
#[allow(
    clippy::cast_precision_loss,
    reason = "alpha uses the existing table Float64 span conversion"
)]
pub fn ewma(
    input: FloatView<'_>,
    options: EwmaOptions,
    execution: &ArrayExecutionOptions,
) -> Result<NativeArray> {
    let len = input.len();
    let bitmap_bytes = len.div_ceil(8);
    let output_bytes = len
        .checked_mul(size_of::<f64>())
        .and_then(|bytes| bytes.checked_add(bitmap_bytes))
        .and_then(|bytes| bytes.checked_add(size_of::<usize>()))
        .ok_or_else(|| {
            super::super::invalid("outputs.output.bytes", "byte count overflowed usize")
        })?;
    let scratch_bytes = size_of::<(u64, f64)>();
    execution.validate_allocation(len, output_bytes, scratch_bytes)?;
    let alpha = 2.0 / (options.span as f64 + 1.0);
    match input.values() {
        FloatValues::Float32(values) => calculate(
            values.iter().map(|&value| f64::from(value)),
            input.validity(),
            len,
            alpha,
            options.min_periods,
            execution,
        ),
        FloatValues::Float64(values) => calculate(
            values.iter().copied(),
            input.validity(),
            len,
            alpha,
            options.min_periods,
            execution,
        ),
    }
}

fn calculate(
    input: impl Iterator<Item = f64>,
    validity: Option<&NullBuffer>,
    len: usize,
    alpha: f64,
    min_periods: u64,
    execution: &ArrayExecutionOptions,
) -> Result<NativeArray> {
    let (mut values, mut bitmap) = allocate_output(len, execution)?;
    let mut valid_count = 0;
    let mut previous = 0.0;
    for (index, sample) in input.enumerate() {
        let sample = if validity.is_none_or(|mask| mask.is_valid(index)) && !sample.is_nan() {
            Some(sample)
        } else {
            None
        };
        (valid_count, previous) = next_ewma(valid_count, previous, sample, alpha, false)
            .ok_or_else(|| super::super::invalid("input.rows", "EWMA sample count overflowed"))?;
        if valid_count >= min_periods {
            values[index] = previous;
            bitmap[index / 8] |= 1 << (index % 8);
        }
    }
    let validity = NullBuffer::new(BooleanBuffer::new(Buffer::from(bitmap), 0, len));
    NativeArray::from_f64(values, vec![len], Some(validity))
}

fn allocate_output(len: usize, execution: &ArrayExecutionOptions) -> Result<(Vec<f64>, Vec<u8>)> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|error| super::super::invalid("outputs.output.bytes", error.to_string()))?;
    values.resize(len, 0.0);
    let mut bitmap = Vec::new();
    bitmap
        .try_reserve_exact(len.div_ceil(8))
        .map_err(|error| super::super::invalid("outputs.output.bytes", error.to_string()))?;
    bitmap.resize(len.div_ceil(8), 0_u8);
    let allocated = values
        .capacity()
        .checked_mul(size_of::<f64>())
        .and_then(|bytes| bytes.checked_add(bitmap.capacity()))
        .and_then(|bytes| bytes.checked_add(size_of::<usize>()))
        .ok_or_else(|| {
            super::super::invalid("outputs.output.bytes", "byte count overflowed usize")
        })?;
    execution.validate_allocation(len, allocated, size_of::<(u64, f64)>())?;
    Ok((values, bitmap))
}

#[inline]
pub(crate) fn next_ewma(
    valid_count: u64,
    previous: f64,
    sample: Option<f64>,
    alpha: f64,
    cumulative: bool,
) -> Option<(u64, f64)> {
    let Some(sample) = sample else {
        return Some((valid_count, previous));
    };
    let next_count = valid_count.checked_add(1)?;
    let value = if valid_count == 0 || alpha.to_bits() == 1.0_f64.to_bits() {
        sample
    } else if cumulative {
        next_cumulative_mean(previous, sample, next_count)
    } else {
        previous + alpha * (sample - previous)
    };
    Some((next_count, value))
}

#[allow(
    clippy::cast_precision_loss,
    reason = "cumulative mean uses the released Float64 sample count conversion"
)]
#[inline]
fn next_cumulative_mean(previous: f64, sample: f64, next_count: u64) -> f64 {
    if previous.is_infinite() && sample.is_finite() {
        previous
    } else if sample.is_infinite() {
        sample + previous
    } else {
        let difference = sample - previous;
        if difference.is_finite() {
            previous + difference / next_count as f64
        } else {
            let weight = 1.0 / next_count as f64;
            previous * (1.0 - weight) + sample * weight
        }
    }
}

#[cfg(test)]
mod tests;

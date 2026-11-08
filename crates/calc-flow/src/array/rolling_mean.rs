use datafusion::arrow::buffer::{BooleanBuffer, Buffer, NullBuffer};

use super::{
    ArrayExecutionOptions, FloatValues, FloatView, NativeArray, invalid,
    rolling_mean_math::{self, ArithmeticError, FloatSummary},
};
use crate::{Result, RollingNumericalProfile};

/// Parameters for an ordered, single-sequence row-window mean.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub struct RollingMeanOptions {
    window: u64,
    min_periods: u64,
    numerical_profile: RollingNumericalProfile,
}

impl RollingMeanOptions {
    /// Selects a positive window, one minimum valid sample, and `StableV1`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CalcFlowError::InvalidArgument`] for a zero window.
    pub fn new(window: u64) -> Result<Self> {
        if window == 0 {
            return Err(invalid(
                "options.window",
                "expected a positive window, received 0",
            ));
        }
        Ok(Self {
            window,
            min_periods: 1,
            numerical_profile: RollingNumericalProfile::StableV1,
        })
    }

    /// Selects the minimum number of valid samples required for a result.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CalcFlowError::InvalidArgument`] unless the count is
    /// between one and the window size, inclusive.
    pub fn with_min_periods(self, min_periods: u64) -> Result<Self> {
        if min_periods == 0 || min_periods > self.window {
            return Err(invalid(
                "options.min_periods",
                format!("expected 1..={}, received {min_periods}", self.window),
            ));
        }
        Ok(Self {
            min_periods,
            ..self
        })
    }

    /// Explicitly selects an existing native rolling numerical profile.
    pub const fn with_numerical_profile(self, numerical_profile: RollingNumericalProfile) -> Self {
        Self {
            numerical_profile,
            ..self
        }
    }

    /// Returns the number of input positions retained by the window.
    pub const fn window(&self) -> u64 {
        self.window
    }

    /// Returns the minimum valid-sample count.
    pub const fn min_periods(&self) -> u64 {
        self.min_periods
    }

    /// Returns the explicitly selected numerical profile.
    pub const fn numerical_profile(&self) -> RollingNumericalProfile {
        self.numerical_profile
    }
}

/// Computes a row-aligned mean using the native stateful table transitions.
///
/// Nulls and NaNs occupy window positions but do not count as valid samples.
/// Infinities remain valid. Output values are `Float64`, including for `Float32`
/// input. The default preserves the released `StableV1` operation order and
/// its documented extreme-value behavior; it does not use the exact ordinary
/// mean reduction or the SQL AVG NaN-value compatibility path.
///
/// ```
/// use calc_flow::{Result, array::{ArrayExecutionOptions, FloatView, RollingMeanOptions, rolling_mean}};
///
/// # fn main() -> Result<()> {
/// let result = rolling_mean(
///     FloatView::from_f32(&[1.0, 3.0, 5.0]),
///     RollingMeanOptions::new(2)?.with_min_periods(2)?,
///     &ArrayExecutionOptions::default(),
/// )?;
/// assert_eq!(result.shape(), &[3]);
/// assert!(result.float_view()?.validity().unwrap().is_null(0));
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`crate::CalcFlowError::InvalidArgument`] if output or scratch
/// requirements exceed the execution limits or a size count overflows.
pub fn rolling_mean(
    input: FloatView<'_>,
    options: RollingMeanOptions,
    execution: &ArrayExecutionOptions,
) -> Result<NativeArray> {
    let (window, capacity) = allocation_plan(input.len(), options, execution)?;
    if input.is_empty() {
        return NativeArray::from_f64(Vec::new(), vec![0], None);
    }
    evaluate(input, options, window, capacity)
}

fn evaluate(
    input: FloatView<'_>,
    options: RollingMeanOptions,
    window: usize,
    capacity: usize,
) -> Result<NativeArray> {
    let mut history = RowsHistory::new(capacity);
    let mut summary = FloatSummary::default();
    let mut output = Vec::with_capacity(input.len());
    let mut bitmap = vec![0_u8; input.len().div_ceil(8)];
    for index in 0..input.len() {
        advance(&mut summary, &mut history, sample_at(input, index), window)?;
        let count = u64::try_from(index + 1)
            .map_err(|_| invalid("input.shape", "sequence position exceeds u64"))?;
        repair(&mut summary, &history, options.numerical_profile, count)?;
        if summary.valid_count >= options.min_periods {
            output.push(summary.read_mean());
            bitmap[index / 8] |= 1 << (index % 8);
        } else {
            output.push(0.0);
        }
    }
    let validity = NullBuffer::new(BooleanBuffer::new(Buffer::from(bitmap), 0, input.len()));
    NativeArray::from_f64(output, vec![input.len()], Some(validity))
}

fn advance(
    summary: &mut FloatSummary,
    history: &mut RowsHistory,
    sample: Option<f64>,
    window: usize,
) -> Result<()> {
    if let Some(sample) = sample {
        summary.as_mut().add(sample).map_err(arithmetic_error)?;
    }
    history.push(sample);
    if history.len > window
        && let Some(sample) = history.pop_front()
    {
        summary.as_mut().remove(sample).map_err(arithmetic_error)?;
    }
    Ok(())
}

fn allocation_plan(
    len: usize,
    options: RollingMeanOptions,
    execution: &ArrayExecutionOptions,
) -> Result<(usize, usize)> {
    u64::try_from(len).map_err(|_| invalid("input.shape", "sequence length exceeds u64"))?;
    let window = usize::try_from(options.window).unwrap_or(usize::MAX);
    let capacity = len.min(window.saturating_add(1));
    let output = len
        .checked_mul(size_of::<f64>())
        .and_then(|bytes| bytes.checked_add(len.div_ceil(8)))
        .and_then(|bytes| bytes.checked_add(size_of::<usize>()))
        .ok_or_else(|| invalid("outputs.output.bytes", "output byte count overflowed usize"))?;
    let scratch = scratch_bytes(len, capacity, options.numerical_profile)?;
    execution.validate_allocation(len, output, scratch)?;
    Ok((window, capacity))
}

fn scratch_bytes(len: usize, capacity: usize, profile: RollingNumericalProfile) -> Result<usize> {
    if len == 0 {
        return Ok(0);
    }
    let state = size_of::<FloatSummary>();
    let fold = if profile == RollingNumericalProfile::StableV2Preview {
        rolling_mean_math::stable_scratch_bytes()
    } else {
        0
    };
    capacity
        .checked_mul(size_of::<Option<f64>>())
        .and_then(|bytes| bytes.checked_add(state))
        .and_then(|bytes| bytes.checked_add(fold))
        .ok_or_else(|| invalid("scratch.bytes", "history byte count overflowed usize"))
}

fn sample_at(input: FloatView<'_>, index: usize) -> Option<f64> {
    if input
        .validity()
        .is_some_and(|validity| validity.is_null(index))
    {
        return None;
    }
    let value = match input.values() {
        FloatValues::Float32(values) => f64::from(values[index]),
        FloatValues::Float64(values) => values[index],
    };
    (!value.is_nan()).then_some(value)
}

fn repair(
    summary: &mut FloatSummary,
    history: &RowsHistory,
    profile: RollingNumericalProfile,
    count: u64,
) -> Result<()> {
    if profile == RollingNumericalProfile::StableV2Preview
        && rolling_mean_math::rebase_due(count, history.len, summary.is_non_finite())
    {
        *summary =
            rolling_mean_math::stable_fold(history.iter().flatten()).map_err(arithmetic_error)?;
    } else if summary.is_non_finite() {
        *summary = FloatSummary::default();
        for value in history.iter().flatten() {
            summary.as_mut().add(value).map_err(arithmetic_error)?;
        }
    }
    Ok(())
}

fn arithmetic_error(error: ArithmeticError) -> crate::CalcFlowError {
    invalid(
        "input.values",
        format!("rolling arithmetic invariant failed: {error:?}"),
    )
}

struct RowsHistory {
    values: Box<[Option<f64>]>,
    start: usize,
    len: usize,
}

impl RowsHistory {
    fn new(capacity: usize) -> Self {
        Self {
            values: vec![None; capacity].into_boxed_slice(),
            start: 0,
            len: 0,
        }
    }

    fn index(&self, offset: usize) -> usize {
        let tail = self.values.len() - self.start;
        if offset < tail {
            self.start + offset
        } else {
            offset - tail
        }
    }

    fn push(&mut self, value: Option<f64>) {
        let index = self.index(self.len);
        self.values[index] = value;
        self.len += 1;
    }

    fn pop_front(&mut self) -> Option<f64> {
        let value = self.values[self.start].take();
        self.start += 1;
        if self.start == self.values.len() {
            self.start = 0;
        }
        self.len -= 1;
        value
    }

    fn iter(&self) -> impl Iterator<Item = Option<f64>> + '_ {
        (0..self.len).map(|index| self.values[self.index(index)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_history_wrap_preserves_missing_positions_without_growth() {
        let mut history = RowsHistory::new(3);
        history.push(Some(1.0));
        history.push(None);
        history.push(Some(3.0));
        assert_eq!(
            history.pop_front().map(f64::to_bits),
            Some(1.0_f64.to_bits())
        );
        history.push(Some(5.0));
        assert_eq!(
            history
                .iter()
                .map(|value| value.map(f64::to_bits))
                .collect::<Vec<_>>(),
            [None, Some(3.0_f64.to_bits()), Some(5.0_f64.to_bits())]
        );
        assert!(history.pop_front().is_none());
        assert_eq!(history.len, 2);
        assert_eq!(history.values.len(), 3);
    }

    #[test]
    fn test_history_and_output_allocation_are_checked_at_the_boundary() {
        let options = RollingMeanOptions::new(u64::MAX).unwrap();
        let defaults = ArrayExecutionOptions::default();
        let (window, capacity) = allocation_plan(2, options, &defaults).unwrap();
        assert_eq!(window, usize::MAX);
        assert_eq!(capacity, 2);
        let output = 2 * size_of::<f64>() + 1 + size_of::<usize>();
        let history = RowsHistory::new(capacity);
        let scratch = history.values.len() * size_of::<Option<f64>>() + size_of::<FloatSummary>();
        assert!(
            allocation_plan(
                2,
                options,
                &defaults.with_max_allocation_bytes(output + scratch)
            )
            .is_ok()
        );
        assert!(matches!(
            allocation_plan(2, options, &defaults.with_max_allocation_bytes(output + scratch - 1)),
            Err(crate::CalcFlowError::InvalidArgument { field, .. }) if field == "scratch.bytes"
        ));
        let unlimited = defaults
            .with_max_elements(usize::MAX)
            .with_max_allocation_bytes(usize::MAX);
        assert!(matches!(
            allocation_plan(usize::MAX, options, &unlimited),
            Err(crate::CalcFlowError::InvalidArgument { field, .. }) if field == "outputs.output.bytes"
        ));
    }
}

use datafusion::arrow::buffer::{BooleanBuffer, Buffer, NullBuffer};

use super::control;
use super::{ArrayExecutionOptions, FloatValues, FloatView, NativeArray, byte_width, invalid};
use crate::Result;

mod exact;

use exact::ExactSum;

/// Missing-value handling for native whole-array reductions.
///
/// By default any null input makes the result null. Valid NaNs remain values
/// and produce a canonical NaN, including when nulls are skipped.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[must_use]
pub struct ReductionOptions {
    skip_nulls: bool,
}

impl ReductionOptions {
    /// Returns options selecting whether null input positions are skipped.
    pub const fn with_skip_nulls(self, skip_nulls: bool) -> Self {
        Self { skip_nulls }
    }

    /// Returns whether null input positions are skipped.
    pub const fn skip_nulls(&self) -> bool {
        self.skip_nulls
    }
}

/// Sums valid values exactly before one float64 ties-to-even rounding.
///
/// Returns a scalar float64 array. Empty input and input with no accepted
/// values return a null scalar. Exact zero is positive zero; valid NaNs and
/// mixed-sign infinities produce a canonical NaN.
///
/// ```
/// use calc_flow::{Result, array::{self, ArrayExecutionOptions, FloatValues, FloatView, ReductionOptions}};
///
/// # fn main() -> Result<()> {
/// let total = array::sum(
///     FloatView::from_f64(&[1e308, 1.0, -1e308]),
///     ReductionOptions::default(),
///     &ArrayExecutionOptions::default(),
/// )?;
/// let FloatValues::Float64(values) = total.float_view()?.values() else {
///     unreachable!("sum returns float64");
/// };
/// assert_eq!(values[0].to_bits(), 1.0_f64.to_bits());
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`crate::CalcFlowError::InvalidArgument`] when the scalar output
/// or fixed kernel scratch exceeds `execution`, or the valid count overflows.
pub fn sum(
    input: FloatView<'_>,
    options: ReductionOptions,
    execution: &ArrayExecutionOptions,
) -> Result<NativeArray> {
    sum_checked(input, options, execution, &mut || Ok(()))
}

pub(super) fn sum_checked(
    input: FloatView<'_>,
    options: ReductionOptions,
    execution: &ArrayExecutionOptions,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    exact_reduction(input, options, execution, false, check)
}

/// Computes an exactly accumulated mean before float64 ties-to-even rounding.
///
/// The exact total is divided before conversion to float64, so finite inputs
/// do not overflow solely because their unnormalized sum exceeds float64.
/// Empty input and input with no accepted values return a null scalar.
///
/// # Errors
///
/// Returns [`crate::CalcFlowError::InvalidArgument`] when the scalar output
/// or fixed kernel scratch exceeds `execution`, or the valid count overflows.
pub fn mean(
    input: FloatView<'_>,
    options: ReductionOptions,
    execution: &ArrayExecutionOptions,
) -> Result<NativeArray> {
    mean_checked(input, options, execution, &mut || Ok(()))
}

pub(super) fn mean_checked(
    input: FloatView<'_>,
    options: ReductionOptions,
    execution: &ArrayExecutionOptions,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    exact_reduction(input, options, execution, true, check)
}

/// Returns a scalar minimum, preserving the input dtype.
///
/// Valid NaNs propagate as a canonical NaN. Mixed signed zeros select negative
/// zero; a single zero sign is preserved. Empty or unaccepted input is null.
///
/// # Errors
///
/// Returns [`crate::CalcFlowError::InvalidArgument`] when the scalar output
/// exceeds the element or allocation limits in `execution`.
pub fn min(
    input: FloatView<'_>,
    options: ReductionOptions,
    execution: &ArrayExecutionOptions,
) -> Result<NativeArray> {
    min_checked(input, options, execution, &mut || Ok(()))
}

pub(super) fn min_checked(
    input: FloatView<'_>,
    options: ReductionOptions,
    execution: &ArrayExecutionOptions,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    extremum(input, options, execution, false, check)
}

/// Returns a scalar maximum, preserving the input dtype.
///
/// Valid NaNs propagate as a canonical NaN. Mixed signed zeros select positive
/// zero; a single zero sign is preserved. Empty or unaccepted input is null.
///
/// # Errors
///
/// Returns [`crate::CalcFlowError::InvalidArgument`] when the scalar output
/// exceeds the element or allocation limits in `execution`.
pub fn max(
    input: FloatView<'_>,
    options: ReductionOptions,
    execution: &ArrayExecutionOptions,
) -> Result<NativeArray> {
    max_checked(input, options, execution, &mut || Ok(()))
}

pub(super) fn max_checked(
    input: FloatView<'_>,
    options: ReductionOptions,
    execution: &ArrayExecutionOptions,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    extremum(input, options, execution, true, check)
}

pub(super) const fn exact_scratch_bytes() -> usize {
    ExactSum::scratch_bytes()
}

fn exact_reduction(
    input: FloatView<'_>,
    options: ReductionOptions,
    execution: &ArrayExecutionOptions,
    average: bool,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    check()?;
    if null_result(input, options) {
        execution.validate_allocation(1, size_of::<f64>() + 1, 0)?;
        return control::finish(scalar_f64(None), check);
    }
    execution.validate_allocation(1, size_of::<f64>(), ExactSum::scratch_bytes())?;
    let mut total = ExactSum::new();
    visit_valid(input, check, |_, value| {
        total
            .push(value)
            .map_err(|_| invalid("input.elements", "valid element count overflowed usize"))
    })?;
    control::finish(
        scalar_f64(if average { total.mean() } else { total.sum() }),
        check,
    )
}

fn extremum(
    input: FloatView<'_>,
    options: ReductionOptions,
    execution: &ArrayExecutionOptions,
    maximum: bool,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    check()?;
    let is_null = null_result(input, options);
    execution.validate_allocation(1, byte_width(input.dtype()) + usize::from(is_null), 0)?;
    if is_null {
        return control::finish(scalar_from_position(input, None), check);
    }
    let mut selected = None;
    visit_valid(input, check, |index, value| {
        if selected.is_none_or(|(_, current)| better_extremum(value, current, maximum)) {
            selected = Some((index, value));
        }
        Ok(())
    })?;
    control::finish(
        scalar_from_position(input, selected.map(|(index, _)| index)),
        check,
    )
}

fn better_extremum(next: f64, current: f64, maximum: bool) -> bool {
    if current.is_nan() {
        false
    } else if next.is_nan() {
        true
    } else if maximum {
        next.total_cmp(&current).is_gt()
    } else {
        next.total_cmp(&current).is_lt()
    }
}

fn null_result(input: FloatView<'_>, options: ReductionOptions) -> bool {
    input.is_empty()
        || input.validity().is_some_and(|validity| {
            validity.null_count() == input.len()
                || (!options.skip_nulls && validity.null_count() != 0)
        })
}

fn visit_valid(
    input: FloatView<'_>,
    check: &mut dyn FnMut() -> Result<()>,
    visit: impl FnMut(usize, f64) -> Result<()>,
) -> Result<()> {
    match input.values() {
        FloatValues::Float32(values) => visit_values(input, values, check, visit),
        FloatValues::Float64(values) => visit_values(input, values, check, visit),
    }
}

fn visit_values<T: Copy + Into<f64>>(
    input: FloatView<'_>,
    values: &[T],
    check: &mut dyn FnMut() -> Result<()>,
    mut visit: impl FnMut(usize, f64) -> Result<()>,
) -> Result<()> {
    for (index, value) in values.iter().enumerate() {
        control::checkpoint(index, check)?;
        if input.validity().is_none_or(|mask| mask.is_valid(index)) {
            visit(index, (*value).into())?;
        }
    }
    check()
}

fn scalar_from_position(input: FloatView<'_>, selected: Option<usize>) -> Result<NativeArray> {
    match input.values() {
        FloatValues::Float32(values) => {
            let value = selected.map(|index| {
                let value = values[index];
                if value.is_nan() {
                    f32::from_bits(0x7fc0_0000)
                } else {
                    value
                }
            });
            NativeArray::from_f32(
                vec![value.unwrap_or(0.0)],
                vec![],
                value.is_none().then(null_scalar_bitmap),
            )
        }
        FloatValues::Float64(values) => scalar_f64(selected.map(|index| {
            let value = values[index];
            if value.is_nan() {
                f64::from_bits(0x7ff8_0000_0000_0000)
            } else {
                value
            }
        })),
    }
}

fn scalar_f64(value: Option<f64>) -> Result<NativeArray> {
    NativeArray::from_f64(
        vec![value.unwrap_or(0.0)],
        vec![],
        value.is_none().then(null_scalar_bitmap),
    )
}

fn null_scalar_bitmap() -> NullBuffer {
    NullBuffer::new(BooleanBuffer::new(Buffer::from_vec(vec![0_u8]), 0, 1))
}

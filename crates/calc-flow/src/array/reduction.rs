use super::{ArrayDType, ArrayExecutionOptions, ArrayView, FloatView, NativeArray};
use crate::Result;

mod axis;
mod exact;

pub(super) use axis::reduce_checked;

pub(super) fn reduction_requirements(
    dtype: ArrayDType,
    shape: &[usize],
    nullable: bool,
    operation: Reduction,
    options: ReductionOptions,
) -> Result<(usize, usize, usize)> {
    axis::descriptor_requirements(dtype, shape, nullable, operation, options)
}

/// Missing-value handling for native whole-array reductions.
///
/// By default any null input makes the result null. Valid NaNs remain values
/// and produce a canonical NaN, including when nulls are skipped.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[must_use]
pub struct ReductionOptions {
    skip_nulls: bool,
    axis: Option<usize>,
    keepdims: bool,
}

impl ReductionOptions {
    /// Returns options selecting whether null input positions are skipped.
    pub const fn with_skip_nulls(self, skip_nulls: bool) -> Self {
        Self { skip_nulls, ..self }
    }

    /// Returns whether null input positions are skipped.
    pub const fn skip_nulls(&self) -> bool {
        self.skip_nulls
    }

    /// Selects a single nonnegative axis, or all positions with `None`.
    pub const fn with_axis(self, axis: Option<usize>) -> Self {
        Self { axis, ..self }
    }

    /// Selects whether reduced dimensions remain with extent one.
    pub const fn with_keepdims(self, keepdims: bool) -> Self {
        Self { keepdims, ..self }
    }

    /// Returns the selected axis, or `None` for all positions.
    pub const fn axis(&self) -> Option<usize> {
        self.axis
    }

    /// Returns whether reduced dimensions remain in the result.
    pub const fn keepdims(&self) -> bool {
        self.keepdims
    }
}

/// A shape-aware floating-point reduction.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reduction {
    /// Exact finite accumulation and one float64 rounding.
    Sum,
    /// Exact finite accumulation and division before float64 rounding.
    Mean,
    /// Minimum, retaining input dtype and signed-zero rules.
    Min,
    /// Maximum, retaining input dtype and signed-zero rules.
    Max,
}

/// Reduces all positions or one axis of a scalar, vector or packed matrix.
///
/// Each lane preserves the exact sum/mean, canonical NaN, null and signed-zero
/// contracts of the original whole-vector functions. Empty lanes are null.
///
/// ```
/// use calc_flow::{Result, array::{self, ArrayExecutionOptions, NativeArray, Reduction, ReductionOptions}};
/// # fn main() -> Result<()> {
/// let input = NativeArray::from_f64(vec![1.0, 3.0, 5.0, 7.0], vec![2, 2], None)?;
/// let means = array::reduce(input.view(), Reduction::Mean,
///     ReductionOptions::default().with_axis(Some(1)), &ArrayExecutionOptions::default())?;
/// assert_eq!(means.shape(), &[2]);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns `InvalidArgument` for Boolean inputs, an axis outside the rank,
/// overflow or exceeded result/scratch limits.
pub fn reduce(
    input: ArrayView<'_>,
    operation: Reduction,
    options: ReductionOptions,
    execution: &ArrayExecutionOptions,
) -> Result<NativeArray> {
    reduce_checked(input, operation, options, execution, &mut || Ok(()))
}

/// Sums valid values exactly before one float64 ties-to-even rounding.
///
/// Default options return a scalar float64 array. Empty input and input with no accepted
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
    reduce_checked(
        ArrayView::from_float(input),
        Reduction::Sum,
        options,
        execution,
        check,
    )
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
    reduce_checked(
        ArrayView::from_float(input),
        Reduction::Mean,
        options,
        execution,
        check,
    )
}

/// Returns a minimum, preserving the input dtype; defaults produce a scalar.
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
    reduce_checked(
        ArrayView::from_float(input),
        Reduction::Min,
        options,
        execution,
        check,
    )
}

/// Returns a maximum, preserving the input dtype; defaults produce a scalar.
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
    reduce_checked(
        ArrayView::from_float(input),
        Reduction::Max,
        options,
        execution,
        check,
    )
}

pub(super) const fn exact_scratch_bytes() -> usize {
    exact::ExactSum::scratch_bytes()
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

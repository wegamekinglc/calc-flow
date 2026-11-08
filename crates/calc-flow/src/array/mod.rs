//! Read-only native floating-point and Boolean array storage and computation.
//!
//! ```
//! use calc_flow::{Result, array::NativeArray};
//!
//! # fn main() -> Result<()> {
//! let array = NativeArray::from_f64(vec![1.0, 2.0], vec![2], None)?;
//! assert_eq!(array.shape(), &[2]);
//! assert_eq!(array.float_view()?.len(), 2);
//! # Ok(())
//! # }
//! ```

use std::fmt;

use datafusion::arrow::buffer::{BooleanBuffer, NullBuffer};

use crate::{CalcFlowError, Result};

mod control;
mod elementwise;
mod execution;
mod matrix;
mod provider;
mod reduction;
pub(crate) mod rolling;
mod rolling_mean;
pub(crate) mod rolling_mean_math;
mod sequence;
mod storage;

fn shape_elements(shape: &[usize]) -> Result<usize> {
    if shape.len() > 2 {
        return Err(invalid("input.shape", "expected rank zero, one or two"));
    }
    shape.iter().try_fold(1_usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| invalid("input.shape", "element count overflowed usize"))
    })
}

pub use elementwise::{
    ArrayView, abs, add, divide, equal, exp, greater, greater_equal, less, less_equal, log,
    logical_and, logical_not, logical_or, logical_xor, multiply, neg, not_equal, pow, sqrt,
    subtract, where_,
};
pub use execution::ArrayExecutionOptions;
pub use matrix::{matmul, reshape, transpose};
pub(crate) use provider::preparation::settlement::{Preparation, channel as preparation_channel};
pub(crate) use provider::preparation::{NativeScope, NativeStaticPlan};
pub use provider::{NativeArrayFactory, NativeTableArrayFactory, register_native};
pub use reduction::{Reduction, ReductionOptions, max, mean, min, reduce, sum};
pub use rolling::{EwmaOptions, ewma};
pub use rolling_mean::{RollingMeanOptions, rolling_mean};
pub use sequence::{ArrayStateOptions, EwmaState, RollingMeanState};
pub use storage::NativeArray;

/// Supported native element types.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ArrayDType {
    /// IEEE 754 single-precision values.
    Float32,
    /// IEEE 754 double-precision values.
    Float64,
    /// Packed Boolean values.
    Boolean,
}

/// A borrowed, read-only sequence of typed floating-point values.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub enum FloatValues<'a> {
    /// Single-precision values.
    Float32(&'a [f32]),
    /// Double-precision values.
    Float64(&'a [f64]),
}

/// A contiguous sequence borrowed for synchronous native computation.
///
/// The values remain read-only and belong to the caller. Transfer owned array
/// storage instead when work must outlive the input borrow.
#[derive(Clone, Copy)]
pub struct FloatView<'a> {
    values: FloatValues<'a>,
    validity: Option<&'a NullBuffer>,
}

impl<'a> FloatView<'a> {
    /// Borrows typed values with an optional validity bitmap.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] if the validity length
    /// differs from the value length or visible byte accounting overflows.
    pub fn try_new(values: FloatValues<'a>, validity: Option<&'a NullBuffer>) -> Result<Self> {
        let view = Self { values, validity };
        validate_validity(view.len(), validity)?;
        visible_bytes(view.len(), view.dtype(), validity, 0)?;
        Ok(view)
    }

    /// Borrows single-precision values with every position valid.
    pub const fn from_f32(values: &'a [f32]) -> Self {
        Self {
            values: FloatValues::Float32(values),
            validity: None,
        }
    }

    /// Borrows double-precision values with every position valid.
    pub const fn from_f64(values: &'a [f64]) -> Self {
        Self {
            values: FloatValues::Float64(values),
            validity: None,
        }
    }

    /// Returns the exact element type.
    pub const fn dtype(&self) -> ArrayDType {
        match self.values {
            FloatValues::Float32(_) => ArrayDType::Float32,
            FloatValues::Float64(_) => ArrayDType::Float64,
        }
    }

    /// Returns the number of positions, including null positions.
    pub const fn len(&self) -> usize {
        match self.values {
            FloatValues::Float32(values) => values.len(),
            FloatValues::Float64(values) => values.len(),
        }
    }

    /// Returns whether the view has no positions.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the borrowed typed values without copying.
    pub const fn values(&self) -> FloatValues<'a> {
        self.values
    }

    /// Returns the optional validity bitmap; set bits denote valid positions.
    pub const fn validity(&self) -> Option<&'a NullBuffer> {
        self.validity
    }
}

fn invalid(field: &str, message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: field.into(),
        message: message.into(),
    }
}

fn bytes_overflow() -> CalcFlowError {
    invalid("input.bytes", "byte count overflowed usize")
}

const fn byte_width(dtype: ArrayDType) -> usize {
    match dtype {
        ArrayDType::Float32 => size_of::<f32>(),
        ArrayDType::Float64 => size_of::<f64>(),
        // Packed storage uses boolean_visible_bytes instead.
        ArrayDType::Boolean => size_of::<bool>(),
    }
}

fn validate_validity(len: usize, validity: Option<&NullBuffer>) -> Result<()> {
    if let Some(validity) = validity
        && validity.len() != len
    {
        return Err(invalid(
            "input.validity",
            format!("expected length {len}, received {}", validity.len()),
        ));
    }
    Ok(())
}

fn visible_bytes(
    len: usize,
    dtype: ArrayDType,
    validity: Option<&NullBuffer>,
    rank: usize,
) -> Result<usize> {
    let values = len
        .checked_mul(byte_width(dtype))
        .ok_or_else(bytes_overflow)?;
    visible_bytes_with_values(values, len, validity, rank)
}

fn boolean_visible_bytes(
    values: &BooleanBuffer,
    validity: Option<&NullBuffer>,
    rank: usize,
) -> Result<usize> {
    let bytes = bit_span_bytes(values.len(), values.offset())?;
    visible_bytes_with_values(bytes, values.len(), validity, rank)
}

fn bit_span_bytes(len: usize, offset: usize) -> Result<usize> {
    if len == 0 {
        return Ok(0);
    }
    (offset % 8)
        .checked_add(len)
        .map(|bits| bits.div_ceil(8))
        .ok_or_else(bytes_overflow)
}

fn visible_bytes_with_values(
    values: usize,
    len: usize,
    validity: Option<&NullBuffer>,
    rank: usize,
) -> Result<usize> {
    let bitmap = validity.map_or(Ok(0), |mask| bit_span_bytes(len, mask.offset()))?;
    let shape = rank
        .checked_mul(size_of::<usize>())
        .ok_or_else(bytes_overflow)?;
    values
        .checked_add(bitmap)
        .and_then(|bytes| bytes.checked_add(shape))
        .ok_or_else(bytes_overflow)
}

impl fmt::Debug for FloatView<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FloatView")
            .field("dtype", &self.dtype())
            .field("len", &self.len())
            .field("nullable", &self.validity.is_some())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod matrix_tests;

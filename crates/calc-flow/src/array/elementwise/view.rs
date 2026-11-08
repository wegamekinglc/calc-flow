use std::fmt;

use datafusion::arrow::buffer::{BooleanBuffer, NullBuffer};

use super::super::{ArrayDType, FloatValues, FloatView, boolean_visible_bytes, validate_validity};
use crate::Result;

#[derive(Clone, Copy)]
enum Values<'a> {
    Float(FloatView<'a>),
    Boolean(&'a BooleanBuffer, Option<&'a NullBuffer>),
    ScalarFloat32(Option<f32>),
    ScalarFloat64(Option<f64>),
    ScalarBoolean(Option<bool>),
}

/// A borrowed vector or inline nullable scalar for native elementwise work.
///
/// Floating and Boolean vector buffers stay read-only. Scalars have shape
/// `[]`; a one-element vector keeps shape `[1]` and is not broadcast as a scalar.
#[derive(Clone, Copy)]
#[must_use]
pub struct ArrayView<'a> {
    values: Values<'a>,
    shape: [usize; 1],
    scalar: bool,
}

impl<'a> ArrayView<'a> {
    /// Borrows a floating vector, including a vector of length one.
    pub const fn from_float(values: FloatView<'a>) -> Self {
        Self {
            shape: [values.len()],
            values: Values::Float(values),
            scalar: false,
        }
    }

    /// Borrows packed Boolean values and optional validity as a vector.
    ///
    /// Values and validity retain their independent bit offsets.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CalcFlowError::InvalidArgument`] for a validity length
    /// mismatch or visible byte-count overflow.
    pub fn from_boolean(
        values: &'a BooleanBuffer,
        validity: Option<&'a NullBuffer>,
    ) -> Result<Self> {
        validate_validity(values.len(), validity)?;
        boolean_visible_bytes(values, validity, 0)?;
        Ok(Self {
            values: Values::Boolean(values, validity),
            shape: [values.len()],
            scalar: false,
        })
    }

    /// Creates an inline nullable float32 scalar without allocating.
    pub const fn scalar_f32(value: Option<f32>) -> Self {
        Self {
            values: Values::ScalarFloat32(value),
            shape: [1],
            scalar: true,
        }
    }

    /// Creates an inline nullable float64 scalar without allocating.
    pub const fn scalar_f64(value: Option<f64>) -> Self {
        Self {
            values: Values::ScalarFloat64(value),
            shape: [1],
            scalar: true,
        }
    }

    /// Creates an inline nullable Boolean scalar without allocating.
    pub const fn scalar_boolean(value: Option<bool>) -> Self {
        Self {
            values: Values::ScalarBoolean(value),
            shape: [1],
            scalar: true,
        }
    }

    /// Returns the exact element type.
    pub const fn dtype(&self) -> ArrayDType {
        match self.values {
            Values::Float(values) => values.dtype(),
            Values::Boolean(..) | Values::ScalarBoolean(_) => ArrayDType::Boolean,
            Values::ScalarFloat32(_) => ArrayDType::Float32,
            Values::ScalarFloat64(_) => ArrayDType::Float64,
        }
    }

    /// Returns `[]` for a scalar or `[n]` for a vector.
    pub const fn shape(&self) -> &[usize] {
        if self.scalar { &[] } else { &self.shape }
    }

    /// Returns the number of positions; a nullable scalar still has length one.
    pub const fn len(&self) -> usize {
        self.shape[0]
    }

    /// Returns whether this is an empty vector.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(in crate::array) const fn with_scalar(self, scalar: bool) -> Self {
        Self { scalar, ..self }
    }

    pub(super) const fn is_scalar(&self) -> bool {
        self.scalar
    }

    pub(super) fn has_nulls(&self) -> bool {
        match self.values {
            Values::Float(values) => values.validity().is_some_and(|mask| mask.null_count() != 0),
            Values::Boolean(values, mask) => {
                debug_assert_eq!(values.len(), self.len());
                mask.is_some_and(|mask| mask.null_count() != 0)
            }
            Values::ScalarFloat32(value) => value.is_none(),
            Values::ScalarFloat64(value) => value.is_none(),
            Values::ScalarBoolean(value) => value.is_none(),
        }
    }

    pub(super) fn number32(&self, index: usize) -> Option<f32> {
        let index = if self.scalar { 0 } else { index };
        match self.values {
            Values::Float(values) => {
                if values.validity().is_some_and(|mask| mask.is_null(index)) {
                    return None;
                }
                let FloatValues::Float32(values) = values.values() else {
                    unreachable!("float32 execution requires float32 values");
                };
                Some(values[index])
            }
            Values::ScalarFloat32(value) => value,
            _ => unreachable!("float32 execution requires float32 values"),
        }
    }

    pub(super) fn number64(&self, index: usize) -> Option<f64> {
        let index = if self.scalar { 0 } else { index };
        match self.values {
            Values::Float(values) => {
                if values.validity().is_some_and(|mask| mask.is_null(index)) {
                    return None;
                }
                Some(match values.values() {
                    FloatValues::Float32(values) => f64::from(values[index]),
                    FloatValues::Float64(values) => values[index],
                })
            }
            Values::ScalarFloat32(value) => value.map(f64::from),
            Values::ScalarFloat64(value) => value,
            _ => unreachable!("numeric execution requires floating values"),
        }
    }

    pub(super) fn boolean(&self, index: usize) -> Option<bool> {
        let index = if self.scalar { 0 } else { index };
        match self.values {
            Values::Boolean(values, validity) => {
                if validity.is_some_and(|mask| mask.is_null(index)) {
                    None
                } else {
                    Some(values.value(index))
                }
            }
            Values::ScalarBoolean(value) => value,
            _ => unreachable!("Boolean execution requires Boolean values"),
        }
    }
}

impl fmt::Debug for ArrayView<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArrayView")
            .field("dtype", &self.dtype())
            .field("shape", &self.shape())
            .field("nullable", &self.has_nulls())
            .finish_non_exhaustive()
    }
}

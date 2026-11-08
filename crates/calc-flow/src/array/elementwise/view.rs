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

/// A borrowed packed rank-0/1/2 array or inline nullable scalar.
///
/// Floating and Boolean vector buffers stay read-only. Scalars have shape
/// `[]`; a one-element vector keeps shape `[1]` and is not broadcast as a scalar.
#[derive(Clone, Copy)]
#[must_use]
pub struct ArrayView<'a> {
    values: Values<'a>,
    shape: [usize; 2],
    rank: usize,
    len: usize,
}

impl<'a> ArrayView<'a> {
    /// Borrows a floating vector, including a vector of length one.
    pub const fn from_float(values: FloatView<'a>) -> Self {
        Self {
            shape: [values.len(), 0],
            values: Values::Float(values),
            rank: 1,
            len: values.len(),
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
            shape: [values.len(), 0],
            rank: 1,
            len: values.len(),
        })
    }

    /// Creates an inline nullable float32 scalar without allocating.
    pub const fn scalar_f32(value: Option<f32>) -> Self {
        Self {
            values: Values::ScalarFloat32(value),
            shape: [0, 0],
            rank: 0,
            len: 1,
        }
    }

    /// Creates an inline nullable float64 scalar without allocating.
    pub const fn scalar_f64(value: Option<f64>) -> Self {
        Self {
            values: Values::ScalarFloat64(value),
            shape: [0, 0],
            rank: 0,
            len: 1,
        }
    }

    /// Creates an inline nullable Boolean scalar without allocating.
    pub const fn scalar_boolean(value: Option<bool>) -> Self {
        Self {
            values: Values::ScalarBoolean(value),
            shape: [0, 0],
            rank: 0,
            len: 1,
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

    /// Returns scalar `[]`, vector `[n]` or matrix `[rows, columns]` shape.
    pub const fn shape(&self) -> &[usize] {
        match self.rank {
            0 => &[],
            1 => std::slice::from_ref(&self.shape[0]),
            _ => &self.shape,
        }
    }

    /// Returns the number of positions; a nullable scalar still has length one.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the view has zero physical positions.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Changes only the shape descriptor, retaining the original borrow.
    ///
    /// ```
    /// use calc_flow::{Result, array::{ArrayView, FloatView}};
    /// # fn main() -> Result<()> {
    /// let values = [1.0, 2.0, 3.0, 4.0];
    /// let view = ArrayView::from_float(FloatView::from_f64(&values)).reshape(&[2, 2])?;
    /// assert_eq!(view.shape(), &[2, 2]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` for unsupported rank, overflow or changed count.
    pub fn reshape(self, shape: &[usize]) -> Result<Self> {
        let count = super::super::shape_elements(shape)?;
        if count != self.len {
            return Err(super::super::invalid(
                "input.shape",
                "reshape must preserve element count",
            ));
        }
        let mut dimensions = [0; 2];
        dimensions[..shape.len()].copy_from_slice(shape);
        Ok(Self {
            shape: dimensions,
            rank: shape.len(),
            ..self
        })
    }

    pub(in crate::array) const fn is_scalar(&self) -> bool {
        self.rank == 0
    }

    pub(in crate::array) fn has_nulls(&self) -> bool {
        self.null_count() != 0
    }

    pub(in crate::array) fn null_count(&self) -> usize {
        match self.values {
            Values::Float(values) => values.validity().map_or(0, NullBuffer::null_count),
            Values::Boolean(_, validity) => validity.map_or(0, NullBuffer::null_count),
            Values::ScalarFloat32(value) => usize::from(value.is_none()),
            Values::ScalarFloat64(value) => usize::from(value.is_none()),
            Values::ScalarBoolean(value) => usize::from(value.is_none()),
        }
    }

    pub(in crate::array) fn number32(&self, index: usize) -> Option<f32> {
        let index = if self.is_scalar() { 0 } else { index };
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

    pub(in crate::array) fn number64(&self, index: usize) -> Option<f64> {
        let index = if self.is_scalar() { 0 } else { index };
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

    pub(in crate::array) fn boolean(&self, index: usize) -> Option<bool> {
        let index = if self.is_scalar() { 0 } else { index };
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

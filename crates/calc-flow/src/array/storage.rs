use std::{any::Any, fmt, sync::Arc};

use datafusion::arrow::{
    array::{Array, ArrayRef, BooleanArray, Float32Array, Float64Array},
    buffer::{BooleanBuffer, NullBuffer, ScalarBuffer},
};

use super::{
    ArrayDType, ArrayView, FloatValues, FloatView, boolean_visible_bytes, byte_width,
    bytes_overflow, invalid, validate_validity, visible_bytes,
};
use crate::{ExternalPayload, Result};

/// Immutable native floating-point or Boolean packed rank-0/1/2 storage.
///
/// Values and validity use Arrow's shared read-only buffers. Clones retain
/// the same storage; accessors never expose mutable caller-owned data.
#[derive(Clone)]
pub struct NativeArray {
    values: ArrayRef,
    dtype: ArrayDType,
    shape: Box<[usize]>,
    visible_bytes: usize,
    retained_bytes: usize,
}

impl NativeArray {
    /// Takes ownership of a float32/float64/Boolean Arrow array without copying.
    ///
    /// Preserves values, validity, and their slice offsets. Shape `[]` denotes
    /// one scalar; `[n]` denotes a vector, and `[rows, columns]` a packed matrix.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CalcFlowError::InvalidArgument`] for an unsupported
    /// concrete Arrow type, invalid shape or validity, or byte-count overflow.
    pub fn from_arrow(values: ArrayRef, shape: Vec<usize>) -> Result<Self> {
        if let Some(array) = values.as_any().downcast_ref::<BooleanArray>() {
            validate_shape(array.len(), &shape)?;
            validate_validity(array.len(), array.nulls())?;
            let visible_bytes = boolean_visible_bytes(array.values(), array.nulls(), shape.len())?;
            let retained_bytes = retained_bytes(
                array.values().inner().capacity(),
                array.nulls(),
                shape.len(),
            )?;
            return Ok(Self {
                values,
                dtype: ArrayDType::Boolean,
                shape: shape.into_boxed_slice(),
                visible_bytes,
                retained_bytes,
            });
        }
        let (view, capacity) = arrow_view(&values)?;
        let (visible_bytes, retained_bytes) =
            validate_parts(view.len(), view.dtype(), &shape, view.validity(), capacity)?;
        Ok(Self {
            dtype: view.dtype(),
            values,
            shape: shape.into_boxed_slice(),
            visible_bytes,
            retained_bytes,
        })
    }

    /// Takes ownership of packed Boolean values and optional validity.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CalcFlowError::InvalidArgument`] for an invalid shape,
    /// mismatched validity length, or byte-count overflow before construction.
    pub fn from_boolean(
        values: BooleanBuffer,
        shape: Vec<usize>,
        validity: Option<NullBuffer>,
    ) -> Result<Self> {
        validate_shape(values.len(), &shape)?;
        validate_validity(values.len(), validity.as_ref())?;
        boolean_visible_bytes(&values, validity.as_ref(), shape.len())?;
        retained_bytes(values.inner().capacity(), validity.as_ref(), shape.len())?;
        Self::from_arrow(Arc::new(BooleanArray::new(values, validity)), shape)
    }

    /// Takes ownership of single-precision values and optional validity.
    ///
    /// All descriptors are validated before constructing new Arrow storage.
    /// The values retain their exact IEEE bits and original precision.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CalcFlowError::InvalidArgument`] for invalid shape,
    /// mismatched validity length, or byte-count overflow.
    pub fn from_f32(
        values: Vec<f32>,
        shape: Vec<usize>,
        validity: Option<NullBuffer>,
    ) -> Result<Self> {
        validate_vec(
            values.len(),
            values.capacity(),
            ArrayDType::Float32,
            &shape,
            validity.as_ref(),
        )?;
        let array = Float32Array::try_new(ScalarBuffer::from(values), validity)
            .map_err(|error| invalid("input.validity", error.to_string()))?;
        Self::from_arrow(Arc::new(array), shape)
    }

    /// Takes ownership of double-precision values and optional validity.
    ///
    /// All descriptors are validated before constructing new Arrow storage.
    /// NaNs, infinities, and signed zeros are preserved without normalization.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CalcFlowError::InvalidArgument`] for invalid shape,
    /// mismatched validity length, or byte-count overflow.
    pub fn from_f64(
        values: Vec<f64>,
        shape: Vec<usize>,
        validity: Option<NullBuffer>,
    ) -> Result<Self> {
        validate_vec(
            values.len(),
            values.capacity(),
            ArrayDType::Float64,
            &shape,
            validity.as_ref(),
        )?;
        let array = Float64Array::try_new(ScalarBuffer::from(values), validity)
            .map_err(|error| invalid("input.validity", error.to_string()))?;
        Self::from_arrow(Arc::new(array), shape)
    }

    /// Returns the exact element type.
    pub const fn dtype(&self) -> ArrayDType {
        self.dtype
    }

    /// Returns the scalar, vector or packed row-major matrix shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the logical row count, including null positions.
    pub fn len(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }

    /// Returns the number of physical positions, including null values.
    pub fn element_count(&self) -> usize {
        self.values.len()
    }

    /// Returns whether the logical row count is zero.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the shared read-only Arrow array without copying.
    pub const fn as_arrow(&self) -> &ArrayRef {
        &self.values
    }

    /// Borrows values without copying and preserves the stored scalar shape.
    #[expect(
        clippy::missing_panics_doc,
        reason = "Private storage is validated by every constructor before it can be borrowed."
    )]
    pub fn view(&self) -> ArrayView<'_> {
        let view = if let Some(array) = self.values.as_any().downcast_ref::<BooleanArray>() {
            ArrayView::from_boolean(array.values(), array.nulls())
                .expect("native Boolean storage was validated at construction")
        } else {
            ArrayView::from_float(
                arrow_view(&self.values)
                    .map(|(view, _)| view)
                    .expect("native floating storage was validated at construction"),
            )
        };
        view.reshape(&self.shape)
            .expect("native shape was validated at construction")
    }

    /// Borrows the stored primitive values and validity for synchronous work.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CalcFlowError::InvalidArgument`] if the storage is not
    /// floating-point, including Boolean storage, or has matrix shape.
    pub fn float_view(&self) -> Result<FloatView<'_>> {
        if self.shape.len() == 2 {
            return Err(invalid(
                "input.shape",
                "matrix sequence requires an explicit vector reshape",
            ));
        }
        arrow_view(&self.values).map(|(view, _)| view)
    }

    /// Returns visible typed bytes, bitmap byte span, and shape entries.
    ///
    /// This is the logical per-consumer queue charge; it excludes fixed
    /// object/allocator overhead and is not a process RSS measurement.
    pub const fn estimated_bytes(&self) -> usize {
        self.visible_bytes
    }

    /// Returns exposed backing-buffer capacities plus shape entries.
    ///
    /// Slices can retain a larger allocation than their visible charge. This
    /// does not measure external owners' other allocations or process RSS.
    pub const fn retained_buffer_bytes(&self) -> usize {
        self.retained_bytes
    }
}

impl fmt::Debug for NativeArray {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeArray")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("null_count", &self.values.null_count())
            .finish_non_exhaustive()
    }
}

impl ExternalPayload for NativeArray {
    fn backend(&self) -> &'static str {
        "native"
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn estimated_bytes(&self) -> usize {
        self.estimated_bytes()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn arrow_view(values: &ArrayRef) -> Result<(FloatView<'_>, usize)> {
    if let Some(array) = values.as_any().downcast_ref::<Float32Array>() {
        return Ok((
            FloatView::try_new(FloatValues::Float32(array.values()), array.nulls())?,
            array.values().inner().capacity(),
        ));
    }
    if let Some(array) = values.as_any().downcast_ref::<Float64Array>() {
        return Ok((
            FloatView::try_new(FloatValues::Float64(array.values()), array.nulls())?,
            array.values().inner().capacity(),
        ));
    }
    Err(invalid(
        "input.dtype",
        format!(
            "expected Float32/Float64 primitive array, received {}",
            values.data_type()
        ),
    ))
}

fn validate_parts(
    len: usize,
    dtype: ArrayDType,
    shape: &[usize],
    validity: Option<&NullBuffer>,
    capacity_bytes: usize,
) -> Result<(usize, usize)> {
    validate_shape(len, shape)?;
    validate_validity(len, validity)?;
    let visible = visible_bytes(len, dtype, validity, shape.len())?;
    Ok((
        visible,
        retained_bytes(capacity_bytes, validity, shape.len())?,
    ))
}

fn retained_bytes(
    capacity_bytes: usize,
    validity: Option<&NullBuffer>,
    rank: usize,
) -> Result<usize> {
    let bitmap = validity.map_or(0, |validity| validity.buffer().capacity());
    let shape_bytes = rank
        .checked_mul(size_of::<usize>())
        .ok_or_else(bytes_overflow)?;
    capacity_bytes
        .checked_add(bitmap)
        .and_then(|bytes| bytes.checked_add(shape_bytes))
        .ok_or_else(bytes_overflow)
}

fn validate_vec(
    len: usize,
    capacity: usize,
    dtype: ArrayDType,
    shape: &[usize],
    validity: Option<&NullBuffer>,
) -> Result<()> {
    validate_shape(len, shape)?;
    validate_validity(len, validity)?;
    let capacity_bytes = capacity
        .checked_mul(byte_width(dtype))
        .ok_or_else(bytes_overflow)?;
    validate_parts(len, dtype, shape, validity, capacity_bytes)?;
    Ok(())
}

fn validate_shape(len: usize, shape: &[usize]) -> Result<()> {
    let expected = super::shape_elements(shape)?;
    if expected != len {
        return Err(invalid(
            "input.shape",
            format!("expected {expected} elements, received {len}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;

use super::{Batch, BatchMetadata, BatchPayload, LatchedArrayPayload, LatchedArrayValues};
use crate::{CalcFlowError, Result};
use std::sync::Arc;

pub(crate) enum NativeStaticValues<'a> {
    Bool(&'a [bool]),
    Float(&'a [f64]),
}
pub(crate) enum NativeStaticOwned {
    Bool(Vec<bool>),
    Float(Vec<f64>),
}
impl LatchedArrayPayload {
    pub(crate) fn native_values(&self) -> Result<NativeStaticValues<'_>> {
        match &self.values {
            LatchedArrayValues::Bool(values) => Ok(NativeStaticValues::Bool(values)),
            LatchedArrayValues::Float(values) => Ok(NativeStaticValues::Float(values)),
            _ => Err(invalid(
                "dtype",
                "native arrays require bool, float32 or float64 values",
            )),
        }
    }
    pub(crate) fn null_flags(&self) -> Option<&[bool]> {
        self.nulls.as_deref()
    }
    pub(crate) fn checked_retained_bytes(&self) -> Result<usize> {
        let values = match &self.values {
            LatchedArrayValues::Bool(values) => bytes(values),
            LatchedArrayValues::Float(values) => bytes(values),
            LatchedArrayValues::Int(values) => bytes(values),
            LatchedArrayValues::Uint(values) => bytes(values),
        }?;
        let parts = [
            values,
            bytes(&self.shape)?,
            self.nulls.as_ref().map_or(Ok(0), bytes)?,
            self.backend.capacity(),
            self.dtype.capacity(),
        ];
        parts.into_iter().try_fold(0_usize, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or_else(|| invalid("bytes", "retained carrier byte count overflowed"))
        })
    }
}
impl Batch {
    // Native preparation validates descriptors and values before moving the owned carrier.
    pub(crate) fn from_prepared_native(
        dtype: &str,
        shape: Vec<u64>,
        nulls: Option<Vec<bool>>,
        values: NativeStaticOwned,
    ) -> Self {
        let values = match values {
            NativeStaticOwned::Bool(values) => LatchedArrayValues::Bool(values),
            NativeStaticOwned::Float(values) => LatchedArrayValues::Float(values),
        };
        let payload = LatchedArrayPayload {
            backend: "native".into(),
            dtype: dtype.into(),
            shape,
            nulls,
            values,
        };
        Self {
            payload: BatchPayload::External(Arc::new(payload)),
            metadata: BatchMetadata::default(),
            source_cursor: None,
        }
    }
}
fn bytes<T>(values: &Vec<T>) -> Result<usize> {
    values
        .capacity()
        .checked_mul(size_of::<T>())
        .ok_or_else(|| invalid("bytes", "retained buffer capacity overflowed"))
}
fn invalid(field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: format!("static_inputs.{field}"),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Batch;

    #[test]
    fn test_borrowed_native_latched_access_reports_actual_capacities_without_snapshot_copies() {
        let mut values = Vec::with_capacity(16);
        values.extend([1.0, -0.0]);
        let mut shape = Vec::with_capacity(4);
        shape.extend([2, 1]);
        let mut nulls = Vec::with_capacity(32);
        nulls.extend([false, false]);
        let batch =
            Batch::static_array_float("native", "float32", shape, Some(nulls), values).unwrap();
        let payload = batch.latched_array_payload().unwrap();
        assert_eq!(payload.checked_retained_bytes().unwrap(), 205);
        let NativeStaticValues::Float(values) = payload.native_values().unwrap() else {
            panic!("floating values expected")
        };
        assert_eq!(values[1].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(payload.null_flags().unwrap(), [false, false]);
        let NativeStaticValues::Float(second) = payload.native_values().unwrap() else {
            panic!("floating values expected")
        };
        assert!(std::ptr::eq(values.as_ptr(), second.as_ptr()));
    }

    #[test]
    fn test_prepared_native_carrier_moves_validated_owned_buffers_without_another_copy() {
        let shape = vec![2];
        let values = vec![f64::from_bits(0x7ff0_0000_0000_0001), -0.0];
        let pointer = values.as_ptr();
        let batch =
            Batch::from_prepared_native("float64", shape, None, NativeStaticOwned::Float(values));
        let payload = batch.latched_array_payload().unwrap();
        let NativeStaticValues::Float(values) = payload.native_values().unwrap() else {
            panic!("floating values expected")
        };
        assert_eq!(pointer, values.as_ptr());
        assert_eq!(
            payload.cell(0).unwrap(),
            0x7ff8_0000_0000_0000_u64.to_be_bytes()
        );
        assert_eq!(payload.checked_retained_bytes().unwrap(), 37);
    }
}

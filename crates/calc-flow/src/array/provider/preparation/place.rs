use super::source::{self, Latched, Source, Values};
use crate::{
    Result,
    array::{ArrayDType, NativeArray, control},
};
use datafusion::arrow::buffer::{BooleanBuffer, Buffer, NullBuffer};

pub(super) fn place(
    view: &Latched<'_>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    check()?;
    let profile = Source::Latched(*view).profile()?;
    let array = typed(view, &profile, check)?;
    verify(&array, profile.placement)?;
    check()?;
    Ok(array)
}
fn typed(
    view: &Latched<'_>,
    profile: &source::Profile,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    let shape = view.shape_usize()?;
    match view.dtype {
        ArrayDType::Float32 => float32(view, profile.elements, shape, check),
        ArrayDType::Float64 => float64(view, profile.elements, shape, check),
        ArrayDType::Boolean => boolean(view, profile.elements, shape, check),
    }
}
fn verify(array: &NativeArray, admitted: usize) -> Result<()> {
    if array.retained_buffer_bytes() > admitted {
        return Err(source::invalid(
            "bytes",
            "placement capacity exceeds admitted bytes",
        ));
    }
    Ok(())
}
#[allow(
    clippy::cast_possible_truncation,
    reason = "float32 carriers use the existing digest-v1 width conversion"
)]
fn float32(
    view: &Latched<'_>,
    len: usize,
    shape: Vec<usize>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    let mut values = vec![0_f32; len];
    let nulls = positions(view, len, check, |position, next| {
        values[position] = floating(view, next)? as f32;
        Ok(())
    })?;
    NativeArray::from_f32(values, shape, nulls)
}
fn float64(
    view: &Latched<'_>,
    len: usize,
    shape: Vec<usize>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    let mut values = vec![0_f64; len];
    let nulls = positions(view, len, check, |position, next| {
        values[position] = floating(view, next)?;
        Ok(())
    })?;
    NativeArray::from_f64(values, shape, nulls)
}
fn floating(view: &Latched<'_>, index: usize) -> Result<f64> {
    let Values::Float(values) = view.values else {
        return Err(source::invalid("dtype", "expected a floating carrier"));
    };
    values
        .get(index)
        .copied()
        .ok_or_else(|| source::invalid("values", "carrier does not cover valid cells"))
}
fn boolean(
    view: &Latched<'_>,
    len: usize,
    shape: Vec<usize>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<NativeArray> {
    let Values::Boolean(source) = view.values else {
        return Err(source::invalid("dtype", "expected a boolean carrier"));
    };
    let mut values = vec![0; len.div_ceil(8)];
    let nulls = positions(view, len, check, |position, next| {
        let value = source
            .get(next)
            .ok_or_else(|| source::invalid("values", "carrier does not cover valid cells"))?;
        values[position / 8] |= u8::from(*value) << (position % 8);
        Ok(())
    })?;
    NativeArray::from_boolean(bits(values, len), shape, nulls)
}
fn positions(
    view: &Latched<'_>,
    len: usize,
    check: &mut dyn FnMut() -> Result<()>,
    mut copy: impl FnMut(usize, usize) -> Result<()>,
) -> Result<Option<NullBuffer>> {
    view.validate_flags(len)?;
    let mut validity = view.nulls.map(|_| vec![0; len.div_ceil(8)]);
    let mut next = 0;
    for index in 0..len {
        control::checkpoint(index, check)?;
        if view.is_null(index) {
            continue;
        }
        copy_valid(index, next, &mut validity, &mut copy)?;
        next += 1;
    }
    view.validate_count(next)?;
    Ok(validity.map(|bytes| NullBuffer::new(bits(bytes, len))))
}
fn copy_valid(
    position: usize,
    next: usize,
    validity: &mut Option<Vec<u8>>,
    copy: &mut impl FnMut(usize, usize) -> Result<()>,
) -> Result<()> {
    copy(position, next)?;
    if let Some(mask) = validity {
        mask[position / 8] |= 1 << (position % 8);
    }
    Ok(())
}
fn bits(values: Vec<u8>, len: usize) -> BooleanBuffer {
    BooleanBuffer::new(Buffer::from_vec(values), 0, len)
}

#[cfg(test)]
mod tests {
    use super::super::source::{Latched, Values};
    use super::*;
    use crate::array::ArrayDType;

    #[test]
    fn test_latched_float32_placement_keeps_nulls_and_typed_storage_in_the_admitted_bytes() {
        let view = Latched {
            backend: "native",
            dtype: ArrayDType::Float32,
            shape: &[4, 1],
            nulls: Some(&[false, true, false, false]),
            values: Values::Float(&[1.25, -0.0, f64::INFINITY]),
            retained: 0,
        };
        let array = place(&view, &mut || Ok(())).unwrap();
        assert_eq!(array.shape(), [4, 1]);
        assert_eq!(array.retained_buffer_bytes(), 33);
        let values = array.view();
        assert_eq!(values.number32(0).unwrap().to_bits(), 1.25_f32.to_bits());
        assert!(values.number32(1).is_none());
        assert_eq!(values.number32(2).unwrap().to_bits(), (-0.0_f32).to_bits());
        assert!(values.number32(3).unwrap().is_infinite());
    }
    #[test]
    fn test_boolean_placement_keeps_explicit_mask_and_independent_bit_buffers() {
        let view = Latched {
            backend: "native",
            dtype: ArrayDType::Boolean,
            shape: &[9],
            nulls: Some(&[false; 9]),
            values: Values::Boolean(&[true, false, true, false, true, false, true, false, true]),
            retained: 0,
        };
        let array = place(&view, &mut || Ok(())).unwrap();
        assert_eq!(array.retained_buffer_bytes(), 12);
        assert!(array.as_arrow().nulls().is_some());
        assert_eq!(array.view().boolean(8), Some(true));
    }
    #[test]
    fn test_placement_cancellation_checks_null_positions_without_reading_compact_carriers() {
        let nulls = vec![true; 8193];
        let view = Latched {
            backend: "native",
            dtype: ArrayDType::Float64,
            shape: &[8193],
            nulls: Some(&nulls),
            values: Values::Float(&[]),
            retained: 0,
        };
        let mut checks = 0;
        let result = place(&view, &mut || {
            checks += 1;
            if checks == 3 {
                Err(crate::CalcFlowError::Cancelled {
                    run_id: "placement".into(),
                })
            } else {
                Ok(())
            }
        });
        assert!(matches!(
            result,
            Err(crate::CalcFlowError::Cancelled { .. })
        ));
        assert_eq!(checks, 3);
    }
}

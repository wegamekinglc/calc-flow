use crate::{
    CalcFlowError, Result,
    array::{ArrayDType, NativeArray, elementwise},
};

pub(super) enum Source<'a> {
    Native(&'a NativeArray),
    Latched(Latched<'a>),
}
#[derive(Clone, Copy)]
pub(super) enum Values<'a> {
    Boolean(&'a [bool]),
    Float(&'a [f64]),
}
#[derive(Clone, Copy)]
pub(super) struct Latched<'a> {
    pub backend: &'a str,
    pub dtype: ArrayDType,
    pub shape: &'a [u64],
    pub nulls: Option<&'a [bool]>,
    pub values: Values<'a>,
    pub retained: usize,
}
pub(super) struct Profile {
    pub frozen: usize,
    pub placement: usize,
    pub elements: usize,
}
impl Latched<'_> {
    pub(super) fn shape_usize(&self) -> Result<Vec<usize>> {
        let mut shape = Vec::with_capacity(self.shape.len());
        for extent in self.shape {
            shape.push(
                usize::try_from(*extent).map_err(|_| invalid("shape", "shape exceeds usize"))?,
            );
        }
        Ok(shape)
    }

    pub(super) fn validate_flags(&self, len: usize) -> Result<()> {
        if self.nulls.is_some_and(|nulls| nulls.len() != len) {
            return Err(invalid("nulls", "null flags do not cover the shape"));
        }
        Ok(())
    }
    pub(super) fn validate_count(&self, count: usize) -> Result<()> {
        let expected = match self.values {
            Values::Boolean(values) => values.len(),
            Values::Float(values) => values.len(),
        };
        if expected != count {
            return Err(invalid("values", "carrier length differs from valid cells"));
        }
        Ok(())
    }
    pub(super) fn is_null(&self, position: usize) -> bool {
        self.nulls.is_some_and(|nulls| nulls[position])
    }
}
impl Source<'_> {
    pub(super) fn profile(&self) -> Result<Profile> {
        match self {
            Self::Native(array) => native(array),
            Self::Latched(view) => latched(view),
        }
    }
}
fn native(array: &NativeArray) -> Result<Profile> {
    let elements = array.element_count();
    let valid = elements
        .checked_sub(array.as_arrow().null_count())
        .ok_or_else(|| invalid("values", "native null count exceeds length"))?;
    let rank = array.shape().len();
    let dtype = array.dtype();
    let nullable = array.as_arrow().nulls().is_some();
    let frozen = frozen_bytes(dtype, rank, elements, valid, nullable)?;
    let frozen = add(array.retained_buffer_bytes(), frozen)?;
    Ok(Profile {
        frozen,
        placement: 0,
        elements,
    })
}
fn latched(view: &Latched<'_>) -> Result<Profile> {
    if view.backend != "native" {
        return Err(invalid(
            "backend",
            "native preparation requires a native payload",
        ));
    }
    let elements = elements(view.shape)?;
    let rank = view.shape.len();
    let nullable = view.nulls.is_some();
    let placement = elementwise::required_shape_bytes(view.dtype, elements, rank, nullable)?;
    Ok(Profile {
        frozen: view.retained,
        placement,
        elements,
    })
}
pub(super) fn elements(shape: &[u64]) -> Result<usize> {
    if shape.len() > 2 {
        return Err(invalid(
            "shape",
            "native static arrays require rank zero through two",
        ));
    }
    shape.iter().try_fold(1_usize, |count, extent| {
        usize::try_from(*extent)
            .ok()
            .and_then(|extent| count.checked_mul(extent))
            .ok_or_else(|| invalid("shape", "native static shape exceeds usize"))
    })
}
fn frozen_bytes(
    dtype: ArrayDType,
    rank: usize,
    elements: usize,
    valid: usize,
    nullable: bool,
) -> Result<usize> {
    let width = if dtype == ArrayDType::Boolean { 1 } else { 8 };
    let values = valid
        .checked_mul(width)
        .ok_or_else(|| invalid("bytes", "frozen carrier byte count overflowed"))?;
    let shape = rank
        .checked_mul(size_of::<u64>())
        .ok_or_else(|| invalid("bytes", "frozen shape byte count overflowed"))?;
    let flags = if nullable { elements } else { 0 };
    add(add(values, shape)?, add(flags, 6 + name(dtype).len())?)
}
pub(super) const fn name(dtype: ArrayDType) -> &'static str {
    match dtype {
        ArrayDType::Boolean => "bool",
        ArrayDType::Float32 => "float32",
        ArrayDType::Float64 => "float64",
    }
}
fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid("bytes", "native static byte count overflowed"))
}
pub(super) fn invalid(field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: format!("static_inputs.{field}"),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{ArrayDType, NativeArray};
    use datafusion::arrow::buffer::NullBuffer;

    #[test]
    fn test_native_float32_profile_keeps_source_capacity_and_widened_frozen_overlap() {
        let mut values = Vec::with_capacity(16);
        values.extend([1.0_f32, 2.0]);
        let array =
            NativeArray::from_f32(values, vec![2, 1], Some(NullBuffer::new_valid(2))).unwrap();
        let profile = Source::Native(&array).profile().unwrap();
        assert_eq!(
            profile.frozen,
            array.retained_buffer_bytes() + 13 + 16 + 2 + 16
        );
        assert_eq!(profile.placement, 0);
        assert_eq!(profile.elements, 2);
    }
    #[test]
    fn test_existing_float32_carrier_is_retained_while_narrow_placement_is_allocated() {
        let mut values = Vec::with_capacity(16);
        values.extend([1.0, 2.0]);
        let mut shape = Vec::with_capacity(4);
        shape.extend([2, 1]);
        let mut nulls = Vec::with_capacity(32);
        nulls.extend([false, false]);
        let view = Latched {
            backend: "native",
            dtype: ArrayDType::Float32,
            shape: &shape,
            nulls: Some(&nulls),
            values: Values::Float(&values),
            retained: 205,
        };
        let profile = Source::Latched(view).profile().unwrap();
        assert_eq!(profile.frozen, 205);
        assert_eq!(profile.placement, 25);
        assert_eq!(profile.elements, 2);
    }
    #[test]
    fn test_boolean_placement_charges_packed_values_and_explicit_validity() {
        let values = [true; 8];
        let view = Latched {
            backend: "native",
            dtype: ArrayDType::Boolean,
            shape: &[9],
            nulls: Some(&[false, true, false, false, false, false, false, false, false]),
            values: Values::Boolean(&values),
            retained: 35,
        };
        let profile = Source::Latched(view).profile().unwrap();
        assert_eq!(profile.frozen, 35);
        assert_eq!(profile.placement, 12);
    }
    #[test]
    fn test_latched_shape_and_capacity_costs_reject_overflow_without_narrowing() {
        let view = Latched {
            backend: "native",
            dtype: ArrayDType::Float64,
            shape: &[u64::MAX, 2],
            nulls: None,
            values: Values::Float(&[]),
            retained: 0,
        };
        assert!(Source::Latched(view).profile().is_err());
    }
}

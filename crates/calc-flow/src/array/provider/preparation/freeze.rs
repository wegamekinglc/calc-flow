use super::source::{self, Latched, Values};
use crate::{
    Result,
    array::{ArrayDType, ArrayView, NativeArray, control},
};

pub(super) enum OwnedValues {
    Boolean(Vec<bool>),
    Float(Vec<f64>),
}
pub(super) struct Frozen {
    pub dtype: ArrayDType,
    pub shape: Vec<u64>,
    pub nulls: Option<Vec<bool>>,
    pub values: OwnedValues,
}
impl Frozen {
    pub(super) fn view(&self) -> Latched<'_> {
        let values = match &self.values {
            OwnedValues::Boolean(values) => Values::Boolean(values),
            OwnedValues::Float(values) => Values::Float(values),
        };
        Latched {
            backend: "native",
            dtype: self.dtype,
            shape: &self.shape,
            nulls: self.nulls.as_deref(),
            values,
            retained: 0,
        }
    }
}

pub(super) fn freeze(array: &NativeArray, check: &mut dyn FnMut() -> Result<()>) -> Result<Frozen> {
    check()?;
    let view = array.view();
    let valid = array.element_count() - array.as_arrow().null_count();
    let mut nulls = array
        .as_arrow()
        .nulls()
        .map(|_| Vec::with_capacity(array.element_count()));
    let values = match array.dtype() {
        ArrayDType::Boolean => {
            OwnedValues::Boolean(collect(view, valid, &mut nulls, check, ArrayView::boolean)?)
        }
        ArrayDType::Float32 | ArrayDType::Float64 => OwnedValues::Float(collect(
            view,
            valid,
            &mut nulls,
            check,
            ArrayView::number64,
        )?),
    };
    let shape = shape(array)?;
    check()?;
    Ok(Frozen {
        dtype: array.dtype(),
        shape,
        nulls,
        values,
    })
}
fn shape(array: &NativeArray) -> Result<Vec<u64>> {
    let mut output = Vec::with_capacity(array.shape().len());
    for extent in array.shape() {
        output.push(
            u64::try_from(*extent)
                .map_err(|_| source::invalid("shape", "shape exceeds the digest range"))?,
        );
    }
    Ok(output)
}
fn collect<'a, T>(
    input: ArrayView<'a>,
    valid: usize,
    nulls: &mut Option<Vec<bool>>,
    check: &mut dyn FnMut() -> Result<()>,
    read: impl Fn(&ArrayView<'a>, usize) -> Option<T>,
) -> Result<Vec<T>> {
    let mut values = Vec::with_capacity(valid);
    for index in 0..input.len() {
        control::checkpoint(index, check)?;
        let value = read(&input, index);
        if let Some(nulls) = nulls {
            nulls.push(value.is_none());
        }
        if let Some(value) = value {
            values.push(value);
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::NativeArray;
    use datafusion::arrow::buffer::NullBuffer;

    #[test]
    fn test_freeze_widens_only_valid_float32_cells_and_preserves_explicit_mask() {
        let values = vec![1.25_f32, f32::from_bits(0x7f80_0001), -0.0];
        let array = NativeArray::from_f32(
            values,
            vec![3, 1],
            Some(NullBuffer::from(vec![true, false, true])),
        )
        .unwrap();
        let frozen = freeze(&array, &mut || Ok(())).unwrap();
        assert_eq!(frozen.shape, [3, 1]);
        assert_eq!(frozen.nulls, Some(vec![false, true, false]));
        let OwnedValues::Float(values) = frozen.values else {
            panic!("float carrier expected")
        };
        assert_eq!(
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [1.25_f64.to_bits(), (-0.0_f64).to_bits()]
        );
        assert_eq!(array.as_arrow().null_count(), 1);
    }

    #[test]
    fn test_freeze_cancellation_scans_null_slots_and_leaves_input_unchanged() {
        let array = NativeArray::from_f64(
            vec![42.0; 8193],
            vec![8193],
            Some(NullBuffer::new_null(8193)),
        )
        .unwrap();
        let mut checks = 0;
        let result = freeze(&array, &mut || {
            checks += 1;
            if checks == 3 {
                Err(crate::CalcFlowError::Cancelled {
                    run_id: "freeze".into(),
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
        assert_eq!(array.as_arrow().null_count(), 8193);
    }

    #[test]
    fn test_freeze_boolean_scalar_and_empty_array_keep_exact_shape_and_mask_presence() {
        use datafusion::arrow::buffer::BooleanBuffer;
        let scalar =
            NativeArray::from_boolean(BooleanBuffer::from(vec![true]), vec![], None).unwrap();
        let frozen = freeze(&scalar, &mut || Ok(())).unwrap();
        assert!(frozen.shape.is_empty());
        assert!(matches!(frozen.values, OwnedValues::Boolean(values) if values == [true]));
        let array =
            NativeArray::from_f64(vec![], vec![0, 2], Some(NullBuffer::new_valid(0))).unwrap();
        let frozen = freeze(&array, &mut || Ok(())).unwrap();
        assert_eq!(frozen.shape, [0, 2]);
        assert_eq!(frozen.nulls, Some(vec![]));
    }
}

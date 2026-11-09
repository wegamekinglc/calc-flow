use super::{Check, Piece, Result, Sources, invalid, size};
use crate::operator::rolling::indicators::scalar_output_plan::single;
use datafusion::{
    arrow::{
        array::{Array, ArrayData, AsArray},
        buffer::MutableBuffer,
        datatypes::DataType,
    },
    common::ScalarValue,
};

#[derive(Clone, Copy)]
pub(super) enum Value<'a> {
    Array(&'a dyn Array),
    Scalar(&'a ScalarValue),
    Empty,
}
impl<'a> Value<'a> {
    pub(super) fn length(self) -> Result<u64> {
        match self {
            Self::Array(array) => size(array.len()),
            Self::Scalar(_) => Ok(1),
            Self::Empty => Ok(0),
        }
    }
    pub(super) fn scalar(self) -> Option<&'a ScalarValue> {
        if let Self::Scalar(value) = self {
            Some(value)
        } else {
            None
        }
    }
    pub(super) fn piece(self) -> Result<Piece<'a>> {
        Ok(Piece {
            array: if let Self::Array(array) = self {
                Some(array)
            } else {
                None
            },
            start: 0,
            len: usize::try_from(self.length()?)
                .map_err(|_| invalid("fresh dictionary values length"))?,
        })
    }
}
pub(super) fn at<'a>(sources: &mut dyn Sources<'a>, index: usize) -> Result<Value<'a>> {
    if let Some(value) = sources.scalar(index)? {
        let ScalarValue::Dictionary(_, value) = value else {
            return Err(invalid("fresh dictionary scalar"));
        };
        return Ok(single::array(value).map_or(Value::Scalar(value), Value::Array));
    }
    let Some(array) = sources.source(index)?.array else {
        return Ok(Value::Empty);
    };
    let dictionary = array
        .as_any_dictionary_opt()
        .ok_or_else(|| invalid("mutable dictionary array"))?;
    Ok(Value::Array(dictionary.values().as_ref()))
}
pub(super) fn equal(
    dtype: &DataType,
    left: Value<'_>,
    right: Value<'_>,
    check: Check<'_>,
) -> Result<bool> {
    check()?;
    match (left, right) {
        (Value::Scalar(left), Value::Scalar(right)) => {
            Ok(fresh(left).is_some_and(|kind| fresh(right) == Some(kind)))
        }
        (Value::Scalar(value), other) | (other, Value::Scalar(value)) => {
            fresh_array(value, other, check)
        }
        _ => {
            let left = data(dtype, left, check)?;
            let right = data(dtype, right, check)?;
            Ok(left.ptr_eq(&right))
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fresh {
    Null,
    EmptyBinary,
}
fn fresh(value: &ScalarValue) -> Option<Fresh> {
    match value {
        ScalarValue::Null => Some(Fresh::Null),
        ScalarValue::FixedSizeBinary(0, Some(value)) if value.is_empty() => {
            Some(Fresh::EmptyBinary)
        }
        _ => None,
    }
}
fn fresh_array(value: &ScalarValue, other: Value<'_>, check: Check<'_>) -> Result<bool> {
    let Some(kind) = fresh(value) else {
        return Ok(false);
    };
    let Value::Array(array) = other else {
        return Ok(false);
    };
    check()?;
    let data = array.to_data();
    check()?;
    if !single_shape(&data) {
        return Ok(false);
    }
    match kind {
        Fresh::Null => Ok(empty_null(&data)),
        Fresh::EmptyBinary => Ok(empty_binary(&data)),
    }
}
fn single_shape(data: &ArrayData) -> bool {
    data.len() == 1 && data.offset() == 0 && data.nulls().is_none() && data.child_data().is_empty()
}

fn empty_null(data: &ArrayData) -> bool {
    data.data_type() == &DataType::Null && data.buffers().is_empty()
}
fn empty_binary(data: &ArrayData) -> bool {
    if data.data_type() != &DataType::FixedSizeBinary(0) || data.buffers().len() != 1 {
        return false;
    }
    let sentinel = MutableBuffer::from_len_zeroed(0);
    data.buffers()[0].is_empty() && data.buffers()[0].as_ptr() == sentinel.as_ptr()
}
fn data(dtype: &DataType, value: Value<'_>, check: Check<'_>) -> Result<ArrayData> {
    if let Value::Array(array) = value {
        check()?;
        let data = array.to_data();
        check()?;
        return Ok(data);
    }
    let array = crate::operator::rolling::indicators::materialize::canonical::null_array(
        dtype,
        0,
        "rolling-output-plan",
        check,
    )?;
    Ok(array.to_data())
}

#[cfg(test)]
mod tests {
    use super::{Value, equal};
    use datafusion::{
        arrow::{
            array::{ArrayData, FixedSizeBinaryArray, make_array},
            buffer::Buffer,
            datatypes::DataType,
        },
        common::ScalarValue,
    };
    #[test]
    fn test_virtual_fresh_empty_binary_matches_only_the_actual_constructor_pointer() {
        let value = ScalarValue::FixedSizeBinary(0, Some(Vec::new()));
        let actual = crate::operator::rolling::indicators::scalar_output_plan::producer::single(
            &value,
            &mut || Ok(()),
        )
        .unwrap();
        let old = make_array(
            ArrayData::builder(DataType::FixedSizeBinary(0))
                .len(1)
                .add_buffer(Buffer::from_vec(Vec::<u8>::new()))
                .build()
                .unwrap(),
        );
        assert!(
            equal(
                &DataType::FixedSizeBinary(0),
                Value::Scalar(&value),
                Value::Array(actual.as_ref()),
                &mut || Ok(())
            )
            .unwrap()
        );
        assert!(
            !equal(
                &DataType::FixedSizeBinary(0),
                Value::Scalar(&value),
                Value::Array(old.as_ref()),
                &mut || Ok(())
            )
            .unwrap()
        );
        assert!(actual.as_any().is::<FixedSizeBinaryArray>());
    }
}

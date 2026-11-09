use crate::{CalcFlowError, Result};
use datafusion::{
    arrow::{array::Array, datatypes::DataType},
    common::ScalarValue,
};

#[path = "materialize/canonical.rs"]
pub(super) mod canonical;
#[path = "materialize/encoded.rs"]
mod encoded;
#[path = "materialize/plain.rs"]
mod plain;

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

pub(in super::super) fn read_scalar(
    array: &dyn Array,
    row: usize,
    node: &str,
    borrowed: bool,
) -> Result<ScalarValue> {
    if borrowed {
        return read_scalar_checked(array, row, node, &mut || Ok(()));
    }
    ScalarValue::try_from_array(array, row).map_err(|error| CalcFlowError::Operator {
        node_id: node.into(),
        message: format!("rolling input row could not be read: {error}"),
    })
}

pub(in super::super) fn read_scalar_checked(
    array: &dyn Array,
    row: usize,
    node: &str,
    check: Check<'_>,
) -> Result<ScalarValue> {
    read_scalar_as(array, row, array.data_type(), node, check)
}

pub(in super::super) fn read_scalar_as(
    array: &dyn Array,
    row: usize,
    expected: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ScalarValue> {
    check()?;
    let scalar = match canonical::nested_scalar_as(array, row, expected, node, check)? {
        Some(scalar) => scalar,
        None => encoded_or_plain(array, row, expected, node, check)?,
    };
    check()?;
    Ok(scalar)
}
fn encoded_or_plain(
    array: &dyn Array,
    row: usize,
    expected: &DataType,
    node: &str,
    check: Check<'_>,
) -> Result<ScalarValue> {
    if matches!(
        expected,
        DataType::Dictionary(..) | DataType::RunEndEncoded(..) | DataType::Union(..)
    ) {
        encoded::read(array, row, expected, node, check)
    } else {
        plain::read_as(array, row, expected, node)
    }
}

fn typed<'a, T: Array + 'static>(array: &'a dyn Array, node: &str) -> Result<&'a T> {
    array.as_any().downcast_ref().ok_or_else(|| invalid(node))
}
fn invalid(node: &str) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: node.into(),
        message: "rolling inputs.input selected scalar representation is invalid".into(),
    }
}
fn null_scalar(dtype: &DataType, node: &str, check: Check<'_>) -> Result<ScalarValue> {
    canonical::null_scalar(dtype, node, check).map_err(|error| match error {
        CalcFlowError::Format { .. } | CalcFlowError::InvalidArgument { .. } => invalid(node),
        other => other,
    })
}

#[cfg(test)]
mod tests {
    use datafusion::{
        arrow::{
            array::{Array, FixedSizeListArray, Float64Array},
            buffer::NullBuffer,
            datatypes::{DataType, Field},
        },
        common::ScalarValue,
    };
    use std::{collections::HashMap, sync::Arc};

    #[test]
    fn test_null_offset_nested_scalar_keeps_shared_backing_and_child_metadata() {
        let field = Arc::new(
            Field::new("closing value", DataType::Float64, true)
                .with_metadata(HashMap::from([("unit".into(), "USD".into())])),
        );
        let source = FixedSizeListArray::new(
            field,
            4,
            Arc::new(Float64Array::from_iter_values((0..12_u32).map(f64::from))),
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let offset = source.slice(1, 2);
        let null = super::read_scalar_checked(&offset, 0, "features", &mut || Ok(())).unwrap();
        let valid = super::read_scalar_checked(&offset, 1, "features", &mut || Ok(())).unwrap();
        assert!(null.is_null());
        assert_eq!(valid.data_type(), source.data_type().clone());
        let ScalarValue::FixedSizeList(array) = &null else {
            panic!("expected fixed list");
        };
        assert_eq!(array.len(), 1);
        assert_eq!(
            array.values().to_data().buffers()[0].data_ptr(),
            source.values().to_data().buffers()[0].data_ptr()
        );
        let output = ScalarValue::iter_to_array(vec![null, valid]).unwrap();
        assert!(output.is_null(0));
        assert!(!output.is_null(1));
        assert_eq!(output.data_type(), source.data_type());
        let expected = source.slice(1, 2);
        assert_eq!(output.to_data(), expected.to_data());
    }
    #[test]
    fn test_dictionary_typed_null_shape_is_rejected_without_panicking() {
        use datafusion::arrow::{
            array::{DictionaryArray, Int8Array, new_empty_array},
            datatypes::{Field, Int8Type},
        };
        let run = DataType::RunEndEncoded(
            Arc::new(Field::new("ends", DataType::Int16, false)),
            Arc::new(Field::new("value", DataType::Int32, true)),
        );
        let shape = DataType::FixedSizeList(Arc::new(Field::new("item", run, true)), 32768);
        let array = DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(vec![None]),
            new_empty_array(&shape),
        )
        .unwrap();
        let result = super::read_scalar(&array, 0, "features", true);
        assert!(
            matches!(result, Err(crate::CalcFlowError::Operator { node_id, .. }) if node_id == "features"),
            "an unrepresentable selected row must return its operator field error"
        );
    }
    #[test]
    fn test_decoded_timestamp_scalar_reuses_expected_timezone_owner() {
        use datafusion::arrow::{
            array::TimestampMicrosecondArray,
            datatypes::{Schema, TimeUnit},
            ipc::{reader::FileReader, writer::FileWriter},
            record_batch::RecordBatch,
        };
        let timezone: Arc<str> = Arc::from("UTC");
        let dtype = DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::clone(&timezone)));
        let schema = Arc::new(Schema::new(vec![Field::new("ts", dtype.clone(), true)]));
        let record = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(
                TimestampMicrosecondArray::from(vec![Some(1), None])
                    .with_timezone(Arc::clone(&timezone)),
            )],
        )
        .unwrap();
        let mut bytes = Vec::new();
        {
            let mut writer = FileWriter::try_new(&mut bytes, &schema).unwrap();
            writer.write(&record).unwrap();
            writer.finish().unwrap();
        }
        let decoded = FileReader::try_new(std::io::Cursor::new(bytes), None)
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        let DataType::Timestamp(_, Some(decoded_timezone)) = decoded.column(0).data_type() else {
            panic!("expected timestamp");
        };
        assert!(!Arc::ptr_eq(&timezone, decoded_timezone));
        for row in 0..2 {
            let scalar = super::read_scalar_as(
                decoded.column(0).as_ref(),
                row,
                &dtype,
                "features",
                &mut || Ok(()),
            )
            .unwrap();
            let ScalarValue::TimestampMicrosecond(value, Some(expected_timezone)) = scalar else {
                panic!("expected timestamp scalar");
            };
            assert!(Arc::ptr_eq(&timezone, &expected_timezone));
            assert_eq!(value, [Some(1), None][row]);
        }
    }
    #[test]
    fn test_decoded_encoded_scalars_reuse_compiled_field_and_timezone_owners() {
        use datafusion::arrow::{
            array::{
                ArrayRef, DictionaryArray, Int8Array, Int16Array, RunArray,
                TimestampMicrosecondArray, UnionArray,
            },
            buffer::ScalarBuffer,
            datatypes::{Int8Type, Int16Type, Schema, UnionFields},
            ipc::{reader::FileReader, writer::FileWriter},
            record_batch::RecordBatch,
        };
        let timezone: Arc<str> = Arc::from("UTC");
        let values: ArrayRef =
            Arc::new(TimestampMicrosecondArray::from(vec![1]).with_timezone(Arc::clone(&timezone)));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(
                DictionaryArray::<Int8Type>::try_new(Int8Array::from(vec![0]), Arc::clone(&values))
                    .unwrap(),
            ),
            Arc::new(
                RunArray::<Int16Type>::try_new(&Int16Array::from(vec![1]), values.as_ref())
                    .unwrap(),
            ),
            Arc::new(
                UnionArray::try_new(
                    UnionFields::try_new(
                        vec![7],
                        vec![Arc::new(Field::new(
                            "clock",
                            values.data_type().clone(),
                            true,
                        ))],
                    )
                    .unwrap(),
                    ScalarBuffer::from(vec![7_i8]),
                    Some(ScalarBuffer::from(vec![0_i32])),
                    vec![values],
                )
                .unwrap(),
            ),
        ];
        let schema = Arc::new(Schema::new(
            columns
                .iter()
                .enumerate()
                .map(|(index, array)| {
                    Field::new(index.to_string(), array.data_type().clone(), true)
                })
                .collect::<Vec<_>>(),
        ));
        let record = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
        let mut bytes = Vec::new();
        {
            let mut writer = FileWriter::try_new(&mut bytes, &schema).unwrap();
            writer.write(&record).unwrap();
            writer.finish().unwrap();
        }
        let decoded = FileReader::try_new(std::io::Cursor::new(bytes), None)
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        for column in 0..3 {
            let expected = schema.field(column).data_type();
            let scalar = super::read_scalar_as(
                decoded.column(column).as_ref(),
                0,
                expected,
                "features",
                &mut || Ok(()),
            )
            .unwrap();
            let inner = assert_compiled_wrapper_owners(scalar, expected);
            let ScalarValue::TimestampMicrosecond(Some(1), Some(actual_timezone)) = inner else {
                panic!("expected timestamp scalar");
            };
            assert!(Arc::ptr_eq(&timezone, &actual_timezone));
        }
    }
    fn assert_compiled_wrapper_owners(scalar: ScalarValue, expected: &DataType) -> ScalarValue {
        match (scalar, expected) {
            (ScalarValue::Dictionary(_, value), DataType::Dictionary(..)) => *value,
            (
                ScalarValue::RunEndEncoded(ends, field, value),
                DataType::RunEndEncoded(expected_ends, expected_field),
            ) => {
                assert!(Arc::ptr_eq(&ends, expected_ends));
                assert!(Arc::ptr_eq(&field, expected_field));
                *value
            }
            (
                ScalarValue::Union(Some((7, value)), fields, _),
                DataType::Union(expected_fields, _),
            ) => {
                assert!(Arc::ptr_eq(
                    fields.iter().next().unwrap().1,
                    expected_fields.iter().next().unwrap().1
                ));
                *value
            }
            _ => panic!("unexpected scalar wrapper"),
        }
    }
}

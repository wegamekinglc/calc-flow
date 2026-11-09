use super::*;
use datafusion::arrow::{
    array::{ArrayRef, DictionaryArray, Int8Array, Int32Array, TimestampMillisecondArray},
    buffer::{NullBuffer, OffsetBuffer},
    datatypes::{Field, Int8Type, Schema},
    ipc::{reader::FileReader, writer::FileWriter},
    record_batch::RecordBatch,
};
use std::{collections::HashMap, io::Cursor};

fn field(dtype: DataType) -> Arc<Field> {
    Arc::new(
        Field::new("item", dtype, true)
            .with_metadata(HashMap::from([("unit".into(), "USD".into())])),
    )
}
fn source() -> StructArray {
    let plain = ListArray::new(
        field(DataType::Int32),
        OffsetBuffer::new(vec![0_i32, 2, 3].into()),
        Arc::new(Int32Array::from(vec![1, 2, 3])),
        Some(NullBuffer::from(vec![false, true])),
    );
    let timestamps = TimestampMillisecondArray::from(vec![Some(1000), None]).with_timezone("UTC");
    let dictionary =
        DictionaryArray::<Int8Type>::try_new(Int8Array::from(vec![0, 1, 0]), Arc::new(timestamps))
            .unwrap();
    let encoded = ListArray::new(
        field(dictionary.data_type().clone()),
        OffsetBuffer::new(vec![0_i32, 2, 3].into()),
        Arc::new(dictionary),
        None,
    );
    StructArray::new(
        vec![
            Arc::new(Field::new("plain", plain.data_type().clone(), true)),
            Arc::new(Field::new("encoded", encoded.data_type().clone(), true)),
        ]
        .into(),
        vec![Arc::new(plain), Arc::new(encoded)],
        None,
    )
}
fn decoded(source: ArrayRef) -> ArrayRef {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "input",
        source.data_type().clone(),
        true,
    )]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![source]).unwrap();
    let mut bytes = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut bytes, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }
    let mut reader = FileReader::try_new(Cursor::new(bytes), None).unwrap();
    Arc::clone(reader.next().unwrap().unwrap().column(0))
}
fn struct_fields(dtype: &DataType) -> &datafusion::arrow::datatypes::Fields {
    let DataType::Struct(fields) = dtype else {
        panic!("struct")
    };
    fields
}
fn list_field(dtype: &DataType) -> &Arc<Field> {
    let DataType::List(field) = dtype else {
        panic!("list")
    };
    field
}
fn timestamp_timezone(dtype: &DataType) -> &Arc<str> {
    let DataType::Timestamp(_, Some(tz)) = dtype else {
        panic!("timezone")
    };
    tz
}
fn assert_expected_owners(actual: &StructArray, expected: &DataType) {
    let fields = struct_fields(expected);
    for (actual, expected) in actual.fields().iter().zip(fields.iter()) {
        assert!(Arc::ptr_eq(actual, expected));
    }
    let lists = actual
        .column(1)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let expected_field = list_field(fields[1].data_type());
    assert!(Arc::ptr_eq(list_field(lists.data_type()), expected_field));
    let dictionary = lists
        .values()
        .as_any()
        .downcast_ref::<DictionaryArray<Int8Type>>()
        .unwrap();
    let DataType::Dictionary(_, value) = expected_field.data_type() else {
        panic!("dict")
    };
    assert!(Arc::ptr_eq(
        timestamp_timezone(dictionary.values().data_type()),
        timestamp_timezone(value)
    ));
}
#[test]
fn test_file_reader_nested_scalar_reuses_expected_fields_and_timezone_without_payload_copy() {
    let original = source();
    let expected = original.data_type().clone();
    let input = decoded(Arc::new(original));
    let decoded_fields = struct_fields(input.data_type());
    assert_eq!(input.data_type(), &expected);
    assert!(!Arc::ptr_eq(
        &decoded_fields[0],
        &struct_fields(&expected)[0]
    ));
    let old_count = Arc::strong_count(&decoded_fields[0]);
    let bytes =
        super::super::super::scalar_materialization_plan::plan(&[Arc::clone(&input)], &mut || {
            Ok(())
        })
        .unwrap();
    let mut values = Vec::with_capacity(input.len());
    let allocation = allocation_counter::measure(|| {
        for row in 0..input.len() {
            values.push(
                nested_scalar_as(input.as_ref(), row, &expected, "reader", &mut || Ok(()))
                    .unwrap()
                    .unwrap(),
            );
        }
    });
    assert!(bytes >= allocation.bytes_max);
    assert_eq!(Arc::strong_count(&decoded_fields[0]), old_count);
    for (row, value) in values.iter().enumerate() {
        let ScalarValue::Struct(actual) = value else {
            panic!("struct")
        };
        assert_expected_owners(actual, &expected);
        assert_eq!(actual.to_data(), input.slice(row, 1).to_data());
        let plain = actual
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let original = input
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(
            plain.values().to_data().buffers()[0].data_ptr(),
            original.values().to_data().buffers()[0].data_ptr()
        );
    }
}
#[test]
fn test_expected_metadata_mismatch_is_rejected_without_exposing_metadata() {
    let input = decoded(Arc::new(source()));
    let old = struct_fields(input.data_type());
    let mut fields = old.to_vec();
    fields[0] = Arc::new(
        fields[0]
            .as_ref()
            .clone()
            .with_metadata(HashMap::from([("private".into(), "do-not-expose".into())])),
    );
    let expected = DataType::Struct(fields.into());
    let error = nested_scalar_as(input.as_ref(), 0, &expected, "expected-schema", &mut || {
        Ok(())
    })
    .unwrap_err();
    assert!(!error.to_string().contains("do-not-expose"));
}

fn array_data_owners(actual: &ArrayData, expected: &DataType) {
    assert_eq!(actual.data_type(), expected);
    datatype_owners(actual.data_type(), expected);
    match expected {
        DataType::List(f)
        | DataType::LargeList(f)
        | DataType::ListView(f)
        | DataType::LargeListView(f)
        | DataType::FixedSizeList(f, _)
        | DataType::Map(f, _) => array_data_owners(&actual.child_data()[0], f.data_type()),
        DataType::Struct(fields) => {
            for (child, field) in actual.child_data().iter().zip(fields.iter()) {
                array_data_owners(child, field.data_type());
            }
        }
        _ => encoded_data_owners(actual, expected),
    }
}
fn encoded_data_owners(actual: &ArrayData, expected: &DataType) {
    match expected {
        DataType::Dictionary(_, value) => array_data_owners(&actual.child_data()[0], value),
        DataType::RunEndEncoded(ends, value) => {
            array_data_owners(&actual.child_data()[0], ends.data_type());
            array_data_owners(&actual.child_data()[1], value.data_type());
        }
        DataType::Union(fields, _) => {
            for (child, (_, field)) in actual.child_data().iter().zip(fields.iter()) {
                array_data_owners(child, field.data_type());
            }
        }
        _ => assert!(actual.child_data().is_empty()),
    }
}
fn datatype_owners(actual: &DataType, expected: &DataType) {
    match (actual, expected) {
        (DataType::List(a), DataType::List(e))
        | (DataType::LargeList(a), DataType::LargeList(e))
        | (DataType::ListView(a), DataType::ListView(e))
        | (DataType::LargeListView(a), DataType::LargeListView(e))
        | (DataType::FixedSizeList(a, _), DataType::FixedSizeList(e, _))
        | (DataType::Map(a, _), DataType::Map(e, _)) => assert!(Arc::ptr_eq(a, e)),
        (DataType::Struct(a), DataType::Struct(e)) => {
            for (a, e) in a.iter().zip(e.iter()) {
                assert!(Arc::ptr_eq(a, e));
            }
        }
        _ => encoded_datatype_owners(actual, expected),
    }
}
fn encoded_datatype_owners(actual: &DataType, expected: &DataType) {
    match (actual, expected) {
        (DataType::Dictionary(_, a), DataType::Dictionary(_, e)) => datatype_owners(a, e),
        (DataType::RunEndEncoded(a, b), DataType::RunEndEncoded(e, f)) => {
            assert!(Arc::ptr_eq(a, e));
            assert!(Arc::ptr_eq(b, f));
        }
        (DataType::Union(a, _), DataType::Union(e, _)) => {
            for ((_, a), (_, e)) in a.iter().zip(e.iter()) {
                assert!(Arc::ptr_eq(a, e));
            }
        }
        (DataType::Timestamp(_, Some(a)), DataType::Timestamp(_, Some(e))) => {
            assert!(Arc::ptr_eq(a, e));
        }
        _ => assert_eq!(actual, expected),
    }
}
fn view_arrays() -> Vec<ArrayRef> {
    use datafusion::arrow::buffer::ScalarBuffer;
    let values: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
    vec![
        Arc::new(FixedSizeListArray::new(
            field(DataType::Int32),
            1,
            Arc::clone(&values),
            None,
        )),
        Arc::new(LargeListArray::new(
            field(DataType::Int32),
            OffsetBuffer::new(vec![0_i64, 1, 2].into()),
            Arc::clone(&values),
            None,
        )),
        Arc::new(ListViewArray::new(
            field(DataType::Int32),
            ScalarBuffer::from(vec![0_i32, 1]),
            ScalarBuffer::from(vec![1_i32, 1]),
            Arc::clone(&values),
            None,
        )),
        Arc::new(LargeListViewArray::new(
            field(DataType::Int32),
            ScalarBuffer::from(vec![0_i64, 1]),
            ScalarBuffer::from(vec![1_i64, 1]),
            values,
            None,
        )),
    ]
}
fn map_array() -> ArrayRef {
    let entries = StructArray::new(
        vec![
            Arc::new(Field::new("key", DataType::Int32, false)),
            field(DataType::Int32),
        ]
        .into(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Int32Array::from(vec![3, 4])),
        ],
        None,
    );
    Arc::new(MapArray::new(
        Arc::new(Field::new("entries", entries.data_type().clone(), false)),
        OffsetBuffer::new(vec![0_i32, 1, 2].into()),
        entries,
        None,
        false,
    ))
}
fn run_array() -> ArrayRef {
    use datafusion::arrow::{
        array::{Int16Array, RunArray},
        datatypes::Int16Type,
    };
    let values = StructArray::new(
        vec![field(DataType::Int32)].into(),
        vec![Arc::new(Int32Array::from(vec![4, 5]))],
        None,
    );
    Arc::new(RunArray::<Int16Type>::try_new(&Int16Array::from(vec![1, 2]), &values).unwrap())
}
fn union_array() -> ArrayRef {
    use datafusion::arrow::{array::UnionArray, buffer::ScalarBuffer, datatypes::UnionFields};
    let fields = UnionFields::try_new(
        vec![3, 7],
        vec![field(DataType::Int32), field(DataType::Int32)],
    )
    .unwrap();
    Arc::new(
        UnionArray::try_new(
            fields,
            ScalarBuffer::from(vec![3_i8, 7]),
            None,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(Int32Array::from(vec![3, 4])),
            ],
        )
        .unwrap(),
    )
}
#[test]
fn test_file_reader_rebinds_every_nested_field_family_in_the_single_rebuild() {
    let mut arrays = view_arrays();
    arrays.extend([map_array(), run_array(), union_array()]);
    let fields = arrays
        .iter()
        .enumerate()
        .map(|(i, a)| {
            Arc::new(Field::new(
                format!("column_{i}"),
                a.data_type().clone(),
                true,
            ))
        })
        .collect::<Vec<_>>();
    let original = StructArray::new(fields.into(), arrays, None);
    let expected = original.data_type().clone();
    let input = decoded(Arc::new(original));
    for row in 0..input.len() {
        let ScalarValue::Struct(actual) = nested_scalar_as(
            input.as_ref(),
            row,
            &expected,
            "reader-families",
            &mut || Ok(()),
        )
        .unwrap()
        .unwrap() else {
            panic!("struct")
        };
        assert_eq!(actual.to_data(), input.slice(row, 1).to_data());
        array_data_owners(&actual.to_data(), &expected);
    }
}
#[test]
fn test_expected_field_names_nullability_and_order_remain_strict() {
    let input = decoded(Arc::new(source()));
    let old = struct_fields(input.data_type());
    let changed_name = Arc::new(old[0].as_ref().clone().with_name("different"));
    let changed_nullability =
        Arc::new(old[0].as_ref().clone().with_nullable(!old[0].is_nullable()));
    let cases = [
        vec![changed_name, Arc::clone(&old[1])],
        vec![changed_nullability, Arc::clone(&old[1])],
        vec![Arc::clone(&old[1]), Arc::clone(&old[0])],
    ];
    for fields in cases {
        assert!(
            nested_scalar_as(
                input.as_ref(),
                0,
                &DataType::Struct(fields.into()),
                "expected-schema",
                &mut || Ok(())
            )
            .is_err()
        );
    }
}
#[test]
fn test_expected_metadata_comparison_observes_real_cancel_before_reconstruction() {
    let original = source();
    let expected = original.data_type().clone();
    let input = decoded(Arc::new(original));
    let mut checks = 0;
    let mut result = None;
    let allocation = allocation_counter::measure(|| {
        result = Some(nested_scalar_as(
            input.as_ref(),
            0,
            &expected,
            "cancel-compare",
            &mut || {
                checks += 1;
                if checks == 6 {
                    Err(CalcFlowError::Cancelled {
                        run_id: "cancel".into(),
                    })
                } else {
                    Ok(())
                }
            },
        ));
    });
    assert!(matches!(
        result.unwrap(),
        Err(CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(checks, 6);
    assert_eq!(
        allocation.bytes_max, 6,
        "only the cancellation identity is owned before validation completes"
    );
}

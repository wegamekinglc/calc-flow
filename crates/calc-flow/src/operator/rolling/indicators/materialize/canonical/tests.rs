use super::*;
use datafusion::arrow::{
    array::{Array, FixedSizeListArray, Float64Array, ListArray},
    buffer::NullBuffer,
    datatypes::{DataType, Field, Float64Type},
};

#[test]
fn test_canonical_nested_preserves_null_fixed_list_children_and_metadata() {
    let field = Arc::new(
        Field::new("price", DataType::Float64, true)
            .with_metadata([("unit".into(), "USD".into())].into()),
    );
    let source = FixedSizeListArray::new(
        field,
        3,
        Arc::new(Float64Array::from(vec![1.0; 6])),
        Some(NullBuffer::from(vec![false, true])),
    );
    let scalar = nested_scalar(&source, 0, "canonical", &mut || Ok(()))
        .unwrap()
        .unwrap();
    let ScalarValue::FixedSizeList(array) = scalar else {
        panic!("expected fixed list")
    };
    assert!(array.is_null(0));
    assert_eq!(array.len(), 1);
    assert_eq!(array.data_type(), source.data_type());
    let out = array
        .values()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let input = source
        .values()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(
        out.values().inner().data_ptr(),
        input.values().inner().data_ptr()
    );
}
#[test]
fn test_canonical_list_keeps_original_offsets_and_borrowed_values() {
    let source = ListArray::from_iter_primitive::<Float64Type, _, _>([
        Some(vec![Some(1.0), Some(2.0)]),
        Some(vec![Some(3.0)]),
        None,
    ]);
    for row in 0..source.len() {
        let scalar = nested_scalar(&source, row, "canonical", &mut || Ok(()))
            .unwrap()
            .unwrap();
        let ScalarValue::List(array) = scalar else {
            panic!("expected list")
        };
        assert_eq!(array.to_data(), source.slice(row, 1).to_data());
    }
}

#[test]
fn test_canonical_struct_and_views_match_borrowed_slices_without_copying_payload() {
    use datafusion::arrow::array::{ArrayRef, StringArray, StringViewArray, StructArray};
    let text: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None]));
    let view: ArrayRef = Arc::new(StringViewArray::from(vec![
        Some("a long external string"),
        None,
    ]));
    let source = StructArray::new(
        vec![
            Arc::new(Field::new("text", DataType::Utf8, true)),
            Arc::new(Field::new("view", DataType::Utf8View, true)),
        ]
        .into(),
        vec![text, view],
        Some(NullBuffer::from(vec![true, false])),
    );
    for row in 0..2 {
        let scalar = nested_scalar(&source, row, "canonical", &mut || Ok(()))
            .unwrap()
            .unwrap();
        let ScalarValue::Struct(array) = scalar else {
            panic!("expected struct")
        };
        assert_eq!(array.to_data(), source.slice(row, 1).to_data());
    }
}
#[test]
fn test_canonical_large_list_and_list_views_keep_nullable_rows() {
    use datafusion::arrow::{
        array::{ArrayRef, Int32Array, LargeListArray, LargeListViewArray, ListViewArray},
        buffer::{OffsetBuffer, ScalarBuffer},
    };
    let field = Arc::new(Field::new("item", DataType::Int32, true));
    let values: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
    let nulls = Some(NullBuffer::from(vec![true, false]));
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(LargeListArray::new(
            field.clone(),
            OffsetBuffer::new(ScalarBuffer::from(vec![0_i64, 2, 3])),
            values.clone(),
            nulls.clone(),
        )),
        Arc::new(ListViewArray::new(
            field.clone(),
            ScalarBuffer::from(vec![0, 2]),
            ScalarBuffer::from(vec![2, 1]),
            values.clone(),
            nulls.clone(),
        )),
        Arc::new(LargeListViewArray::new(
            field,
            ScalarBuffer::from(vec![0_i64, 2]),
            ScalarBuffer::from(vec![2_i64, 1]),
            values,
            nulls,
        )),
    ];
    for source in arrays {
        for row in 0..2 {
            let scalar = nested_scalar(source.as_ref(), row, "canonical", &mut || Ok(()))
                .unwrap()
                .unwrap();
            assert_eq!(
                scalar.to_array().unwrap().to_data(),
                source.slice(row, 1).to_data()
            );
        }
    }
}

#[test]
fn test_canonical_leaf_families_preserve_types_through_struct() {
    use DataType::*;
    use datafusion::arrow::{
        array::{StructArray, new_null_array},
        datatypes::{IntervalUnit as I, TimeUnit as T},
    };
    let types = vec![
        Null,
        Boolean,
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Float16,
        Float32,
        Float64,
        Date32,
        Date64,
        Time32(T::Second),
        Time32(T::Millisecond),
        Time64(T::Microsecond),
        Time64(T::Nanosecond),
        Timestamp(T::Second, Some("UTC".into())),
        Timestamp(T::Millisecond, None),
        Timestamp(T::Microsecond, None),
        Timestamp(T::Nanosecond, None),
        Duration(T::Second),
        Duration(T::Millisecond),
        Duration(T::Microsecond),
        Duration(T::Nanosecond),
        Interval(I::YearMonth),
        Interval(I::DayTime),
        Interval(I::MonthDayNano),
        Decimal32(8, 2),
        Decimal64(16, 2),
        Decimal128(28, 2),
        Decimal256(60, 2),
        Binary,
        LargeBinary,
        BinaryView,
        Utf8,
        LargeUtf8,
        Utf8View,
        FixedSizeBinary(3),
    ];
    for dtype in types {
        let values = new_null_array(&dtype, 2);
        let source = StructArray::new(
            vec![Arc::new(Field::new("f", dtype, true))].into(),
            vec![values],
            None,
        );
        let scalar = nested_scalar(&source, 1, "canonical", &mut || Ok(()))
            .unwrap()
            .unwrap();
        assert_eq!(
            scalar.to_array().unwrap().to_data(),
            source.slice(1, 1).to_data()
        );
    }
}

#[test]
fn test_canonical_encoded_children_preserve_selected_ranges() {
    use datafusion::arrow::{
        array::{
            ArrayRef, DictionaryArray, Int8Array, Int16Array, Int32Array, RunArray, StringArray,
            StructArray, UnionArray,
        },
        buffer::ScalarBuffer,
        datatypes::{Int8Type, Int16Type, UnionFields},
    };
    let dictionary: ArrayRef = Arc::new(
        DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(vec![0, 0]),
            Arc::new(StringArray::from(vec!["shared"])),
        )
        .unwrap(),
    );
    let run: ArrayRef = Arc::new(
        RunArray::<Int16Type>::try_new(
            &Int16Array::from(vec![1, 2]),
            &Int32Array::from(vec![5, 7]),
        )
        .unwrap(),
    );
    let fields = UnionFields::try_new(
        [3, 7],
        [
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ],
    )
    .unwrap();
    let union: ArrayRef = Arc::new(
        UnionArray::try_new(
            fields,
            ScalarBuffer::from(vec![3, 7]),
            None,
            vec![
                Arc::new(Int32Array::from(vec![1, 0])),
                Arc::new(Int32Array::from(vec![0, 2])),
            ],
        )
        .unwrap(),
    );
    for child in [dictionary, run, union] {
        let source = StructArray::new(
            vec![Arc::new(Field::new("f", child.data_type().clone(), true))].into(),
            vec![child],
            None,
        );
        let scalar = nested_scalar(&source, 1, "canonical", &mut || Ok(()))
            .unwrap()
            .unwrap();
        assert_eq!(
            scalar.to_array().unwrap().to_data(),
            source.slice(1, 1).to_data()
        );
    }
}

#[test]
fn test_canonical_run_struct_preserves_value_field_metadata_and_sparse_slice() {
    use datafusion::arrow::{
        array::{ArrayData, Int16Array, Int32Array, RunArray},
        datatypes::Int16Type,
    };
    use std::collections::HashMap;
    let child = StructArray::new(
        vec![Arc::new(Field::new("v", DataType::Int32, true))].into(),
        vec![Arc::new(Int32Array::from(vec![4, 5]))],
        None,
    );
    let run = RunArray::<Int16Type>::try_new(&Int16Array::from(vec![2, 4]), &child).unwrap();
    let DataType::RunEndEncoded(ends, value) = run.data_type() else {
        panic!("run")
    };
    let dtype = DataType::RunEndEncoded(
        Arc::clone(ends),
        Arc::new(
            value
                .as_ref()
                .clone()
                .with_metadata(HashMap::from([("unit".into(), "USD".into())])),
        ),
    );
    let run = RunArray::<Int16Type>::from(
        run.to_data()
            .into_builder()
            .data_type(dtype)
            .build()
            .unwrap(),
    )
    .slice(1, 2);
    let source = StructArray::new(
        vec![Arc::new(Field::new("run", run.data_type().clone(), true))].into(),
        vec![Arc::new(run)],
        None,
    );
    let actual = nested_scalar(&source, 1, "run-field", &mut || Ok(()))
        .unwrap()
        .unwrap();
    let ScalarValue::Struct(actual) = actual else {
        panic!("struct")
    };
    assert_eq!(actual.to_data(), source.slice(1, 1).to_data());
    let _: ArrayData = actual.to_data();
}
#[test]
fn test_canonical_map_preserves_entries_and_selected_null_rows() {
    use datafusion::arrow::{
        array::{Int32Array, StringArray},
        buffer::OffsetBuffer,
    };
    let entries = StructArray::new(
        vec![
            Arc::new(Field::new("key", DataType::Int32, false)),
            Arc::new(Field::new("value", DataType::Utf8, true)),
        ]
        .into(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec![Some("first"), None])),
        ],
        None,
    );
    let field = Arc::new(Field::new("entries", entries.data_type().clone(), false));
    let source = MapArray::new(
        field,
        OffsetBuffer::new(vec![0_i32, 1, 2].into()),
        entries,
        Some(NullBuffer::from(vec![false, true])),
        false,
    );
    for row in 0..2 {
        let ScalarValue::Map(actual) = nested_scalar(&source, row, "map", &mut || Ok(()))
            .unwrap()
            .unwrap()
        else {
            panic!("map")
        };
        assert_eq!(actual.to_data(), source.slice(row, 1).to_data());
    }
}

#[test]
fn test_typed_null_empty_union_fallback_preserves_representable_shape_and_fields() {
    use datafusion::arrow::datatypes::{UnionFields, UnionMode};
    let union = DataType::Union(UnionFields::empty(), UnionMode::Sparse);
    let field = Arc::new(Field::new("variant", union.clone(), true));
    let dtype = DataType::List(field);
    let ScalarValue::List(value) = null_scalar(&dtype, "empty-union", &mut || Ok(())).unwrap()
    else {
        panic!("list")
    };
    assert!(value.is_null(0));
    assert_eq!(value.data_type(), &dtype);
    assert_eq!(value.values().len(), 0);
    assert!(matches!(
        null_scalar(&union, "empty-union", &mut || Ok(())).unwrap(),
        ScalarValue::Union(None, _, _)
    ));
    let bad = DataType::FixedSizeList(Arc::new(Field::new("variant", union, true)), 1);
    assert!(null_scalar(&bad, "empty-union", &mut || Ok(())).is_err());
}

#[test]
fn test_guarded_null_array_entry_preserves_shape_and_checks_representability() {
    use datafusion::arrow::datatypes::{Int16Type, UnionFields, UnionMode};
    let empty = DataType::Union(UnionFields::empty(), UnionMode::Sparse);
    let dtype = DataType::List(Arc::new(Field::new("item", empty.clone(), true)));
    let array = null_array(&dtype, 3, "null-array", &mut || Ok(())).unwrap();
    assert_eq!(array.len(), 3);
    assert_eq!(array.null_count(), 3);
    assert_eq!(array.data_type(), &dtype);
    assert_eq!(
        null_array(&empty, 0, "null-array", &mut || Ok(()))
            .unwrap()
            .len(),
        0
    );
    assert!(null_array(&empty, 1, "null-array", &mut || Ok(())).is_err());
    let run = DataType::RunEndEncoded(
        Arc::new(Field::new("ends", DataType::Int16, false)),
        Arc::new(Field::new("value", DataType::Int32, true)),
    );
    assert!(null_array(&run, 32768, "null-array", &mut || Ok(())).is_err());
    let valid = null_array(&run, 2, "null-array", &mut || Ok(())).unwrap();
    assert!(
        valid
            .as_any()
            .is::<datafusion::arrow::array::RunArray<Int16Type>>()
    );
    assert_eq!(valid.len(), 2);
    let error = null_array(&dtype, 1, "null-array", &mut || {
        Err(CalcFlowError::Cancelled {
            run_id: "controlled".into(),
        })
    })
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
}

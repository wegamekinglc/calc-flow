use super::*;
use datafusion::{
    arrow::array::{ArrayRef, Int32Array, StringArray},
    common::ScalarValue,
};
use std::sync::Arc;

fn converted(columns: &[ArrayRef]) -> (Vec<ScalarValue>, allocation_counter::AllocationInfo) {
    let rows = columns.iter().map(Array::len).sum();
    let mut output = Vec::with_capacity(rows);
    let info = allocation_counter::measure(|| {
        for column in columns {
            for row in 0..column.len() {
                let nested = super::super::materialize::canonical::nested_scalar(
                    column.as_ref(),
                    row,
                    "scalar-plan",
                    &mut || Ok(()),
                )
                .unwrap();
                output.push(
                    nested.unwrap_or_else(|| ScalarValue::try_from_array(column, row).unwrap()),
                );
            }
        }
    });
    (output, info)
}
#[test]
fn test_primitives_and_strings_charge_only_owned_payload_per_occurrence() {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])),
        Arc::new(StringArray::from(vec![Some("hello"), None, Some("")])),
    ];
    let bytes = plan(&columns, &mut || Ok(())).unwrap();
    assert_eq!(bytes, 5);
    let (_, actual) = converted(&columns);
    assert!(actual.bytes_max <= bytes);
}

fn assert_bound(columns: &[ArrayRef]) -> u64 {
    let mut bytes = 0;
    let scan = allocation_counter::measure(|| {
        bytes = plan(columns, &mut || Ok(())).unwrap();
    });
    assert_eq!(scan.count_total, 0, "successful scan must not allocate");
    let (_values, actual) = converted(columns);
    assert!(
        bytes >= actual.bytes_max,
        "planned {bytes}, observed {}",
        actual.bytes_max
    );
    bytes
}
#[test]
fn test_binary_and_view_values_copy_each_selected_non_null_occurrence() {
    use datafusion::arrow::array::{
        BinaryArray, BinaryViewArray, FixedSizeBinaryArray, LargeBinaryArray, LargeStringArray,
        StringViewArray,
    };
    let columns: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from(vec![
            Some("repeat"),
            None,
            Some("repeat"),
        ])),
        Arc::new(StringViewArray::from(vec![
            Some("repeat"),
            None,
            Some("repeat"),
        ])),
        Arc::new(BinaryArray::from(vec![
            Some(b"repeat".as_slice()),
            None,
            Some(b"repeat".as_slice()),
        ])),
        Arc::new(LargeBinaryArray::from(vec![
            Some(b"repeat".as_slice()),
            None,
            Some(b"repeat".as_slice()),
        ])),
        Arc::new(BinaryViewArray::from(vec![
            Some(b"repeat".as_slice()),
            None,
            Some(b"repeat".as_slice()),
        ])),
        Arc::new(
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                [Some(b"repeat".as_slice()), None, Some(b"repeat".as_slice())].into_iter(),
                6,
            )
            .unwrap(),
        ),
    ];
    assert_eq!(assert_bound(&columns), 72);
}
#[test]
fn test_null_fixed_list_slices_only_new_metadata_even_when_child_backing_is_large() {
    use datafusion::arrow::{
        array::{FixedSizeListArray, Float64Array},
        buffer::NullBuffer,
        datatypes::{DataType, Field},
    };
    let array = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float64, true)),
        4096,
        Arc::new(Float64Array::from(vec![1.0; 8192])),
        Some(NullBuffer::from(vec![false, true])),
    );
    let columns: Vec<ArrayRef> = vec![Arc::new(array.slice(0, 1))];
    let bytes = assert_bound(&columns);
    assert!(
        bytes < 4096,
        "borrowed physical values are already paid by the caller"
    );
}
#[test]
fn test_struct_slices_charge_child_descriptor_vectors_and_deep_dictionary_types() {
    use datafusion::arrow::{
        array::{DictionaryArray, Int8Array, StringViewArray, StructArray},
        datatypes::{DataType, Field},
    };
    let dictionary: ArrayRef = Arc::new(
        DictionaryArray::<datafusion::arrow::datatypes::Int8Type>::try_new(
            Int8Array::from(vec![0, 0]),
            Arc::new(StringArray::from(vec!["reused"])),
        )
        .unwrap(),
    );
    let view: ArrayRef = Arc::new(StringViewArray::from(vec![
        "a long external view value",
        "another long external value",
    ]));
    let source = StructArray::new(
        vec![
            Arc::new(Field::new("dict", dictionary.data_type().clone(), true)),
            Arc::new(Field::new("view", DataType::Utf8View, true)),
        ]
        .into(),
        vec![dictionary, view],
        None,
    );
    let columns: Vec<ArrayRef> = vec![Arc::new(source.slice(1, 1))];
    assert_bound(&columns);
}

#[test]
fn test_canonical_struct_retained_capacity_is_covered_by_owner_descriptor_bound() {
    use datafusion::arrow::{
        array::StructArray,
        datatypes::{DataType, Field},
    };
    let source = StructArray::new(
        vec![
            Arc::new(Field::new("a", DataType::Int32, true)),
            Arc::new(Field::new("b", DataType::Int32, true)),
        ]
        .into(),
        vec![
            Arc::new(Int32Array::from(vec![1])) as ArrayRef,
            Arc::new(Int32Array::from(vec![2])),
        ],
        None,
    );
    let columns: Vec<ArrayRef> = vec![Arc::new(source)];
    assert_bound(&columns);
    let (values, _) = converted(&columns);
    let ScalarValue::Struct(value) = values.into_iter().next().unwrap() else {
        panic!("struct")
    };
    let charge = descriptor_bytes(value.as_ref(), &mut || Ok(())).unwrap();
    let (_, children, _) = Arc::try_unwrap(value).unwrap().into_parts();
    let actual = size_of::<StructArray>()
        + children.capacity() * size_of::<ArrayRef>()
        + 2 * size_of::<usize>();
    assert!(charge >= u64::try_from(actual).unwrap());
}

#[test]
fn test_plan_does_not_allocate_and_preserves_cancellation() {
    let columns: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec!["abc", "def"]))];
    let info = allocation_counter::measure(|| {
        assert_eq!(plan(&columns, &mut || Ok(())).unwrap(), 6);
    });
    assert_eq!(info.count_total, 0);
    let error = plan(&columns, &mut || {
        Err(CalcFlowError::Cancelled {
            run_id: "probe".into(),
        })
    })
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
}
#[test]
fn test_checked_size_helpers_reject_overflow() {
    assert!(add(u64::MAX, 1).is_err());
    assert!(cost::times::<ScalarValue>(usize::MAX).is_err());
    assert!(cost::product(usize::MAX, 2).is_err());
    assert!(cost::arc_slice::<datafusion::arrow::buffer::Buffer>(usize::MAX).is_err());
}

#[test]
fn test_null_typed_shapes_bound_actual_datafusion_construction() {
    use datafusion::arrow::datatypes::{DataType as D, Field, Fields, UnionFields, UnionMode};
    let field = |dtype| Arc::new(Field::new("item", dtype, true));
    let entry = Arc::new(Field::new(
        "entries",
        D::Struct(Fields::from(vec![
            Arc::new(Field::new("key", D::Int32, false)),
            field(D::Utf8),
        ])),
        false,
    ));
    let types = vec![
        D::Int64,
        D::Utf8,
        D::BinaryView,
        D::List(field(D::Utf8)),
        D::LargeList(field(D::Int32)),
        D::FixedSizeList(field(D::Float64), 4096),
        D::ListView(field(D::Utf8)),
        D::LargeListView(field(D::Int32)),
        D::Struct(vec![field(D::Utf8), field(D::FixedSizeList(field(D::Int32), 3))].into()),
        D::Map(entry, false),
        D::Dictionary(
            Box::new(D::Int8),
            Box::new(D::FixedSizeList(field(D::Utf8), 2)),
        ),
        D::RunEndEncoded(
            Arc::new(Field::new("ends", D::Int16, false)),
            field(D::Utf8),
        ),
        D::Union(
            UnionFields::try_new(vec![2, 7], vec![field(D::Int32), field(D::Utf8)]).unwrap(),
            UnionMode::Dense,
        ),
    ];
    for dtype in types {
        let cost = null::scalar(&dtype, &mut || Ok(())).unwrap();
        let mut scalar = None;
        let actual = allocation_counter::measure(|| {
            scalar = Some(ScalarValue::try_new_null(&dtype).unwrap());
        });
        assert!(
            add(cost.owned, cost.temporary).unwrap() >= actual.bytes_max,
            "{dtype:?}: owned {} temp {} observed {}",
            cost.owned,
            cost.temporary,
            actual.bytes_max
        );
        drop(scalar);
    }
}
#[test]
fn test_dictionary_missing_value_allocates_typed_null_per_occurrence() {
    use datafusion::arrow::{
        array::{DictionaryArray, Int8Array, new_empty_array},
        datatypes::{DataType, Field, Int8Type},
    };
    let dtype =
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 4096);
    let values = new_empty_array(&dtype);
    let dictionary =
        DictionaryArray::<Int8Type>::try_new(Int8Array::from(vec![None, None]), values).unwrap();
    let columns: Vec<ArrayRef> = vec![Arc::new(dictionary)];
    assert!(assert_bound(&columns) > 65_536);
}
#[test]
fn test_encoded_repeated_values_charge_every_logical_scalar_copy() {
    use datafusion::arrow::{
        array::{DictionaryArray, Int8Array, Int16Array, RunArray, UnionArray},
        buffer::ScalarBuffer,
        datatypes::{Field, Int8Type, Int16Type, UnionFields},
    };
    let dictionary: ArrayRef = Arc::new(
        DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(vec![Some(0), None, Some(0)]),
            Arc::new(StringArray::from(vec!["repeat payload"])),
        )
        .unwrap(),
    );
    let run: ArrayRef = Arc::new(
        RunArray::<Int16Type>::try_new(
            &Int16Array::from(vec![3]),
            &StringArray::from(vec!["run payload"]),
        )
        .unwrap()
        .slice(1, 2),
    );
    let strings: ArrayRef = Arc::new(StringArray::from(vec!["union payload"]));
    let fields = UnionFields::try_new(
        vec![7],
        vec![Arc::new(Field::new(
            "value",
            strings.data_type().clone(),
            true,
        ))],
    )
    .unwrap();
    let union: ArrayRef = Arc::new(
        UnionArray::try_new(
            fields,
            ScalarBuffer::from(vec![7_i8, 7]),
            Some(ScalarBuffer::from(vec![0_i32, 0])),
            vec![strings],
        )
        .unwrap(),
    );
    assert_bound(&[dictionary, run, union]);
}

#[test]
fn test_only_selected_unrepresentable_typed_null_run_shape_fails_before_conversion() {
    use datafusion::arrow::{
        array::{DictionaryArray, Int8Array, new_empty_array},
        datatypes::{DataType as D, Field, Int8Type},
    };
    let run = D::RunEndEncoded(
        Arc::new(Field::new("ends", D::Int16, false)),
        Arc::new(Field::new("value", D::Int32, true)),
    );
    let shape = D::FixedSizeList(Arc::new(Field::new("item", run.clone(), true)), 32768);
    let empty = new_empty_array(&shape);
    let unused: ArrayRef = Arc::new(
        DictionaryArray::<Int8Type>::try_new(Int8Array::from(Vec::<i8>::new()), Arc::clone(&empty))
            .unwrap(),
    );
    assert_eq!(plan(&[unused], &mut || Ok(())).unwrap(), 0);
    let selected: ArrayRef =
        Arc::new(DictionaryArray::<Int8Type>::try_new(Int8Array::from(vec![None]), empty).unwrap());
    let error = plan(&[selected], &mut || Ok(()))
        .expect_err("selected typed-null row cannot fit its run-end index");
    assert!(matches!(error, CalcFlowError::Format { .. }));
    let small = D::FixedSizeList(Arc::new(Field::new("item", run.clone(), true)), 32767);
    let cost = null::scalar(&small, &mut || Ok(())).unwrap();
    let mut scalar = None;
    let actual = allocation_counter::measure(|| {
        scalar = Some(ScalarValue::try_new_null(&small).unwrap());
    });
    assert!(add(cost.owned, cost.temporary).unwrap() >= actual.bytes_max);
    let list = D::List(Arc::new(Field::new("item", shape, true)));
    assert!(
        null::scalar(&list, &mut || Ok(())).is_ok(),
        "empty nested child is representable"
    );
}

#[test]
fn test_empty_union_child_descriptor_slot_is_retained_for_each_borrowed_row() {
    use datafusion::arrow::{
        array::{FixedSizeListArray, UnionArray},
        buffer::ScalarBuffer,
        datatypes::{Field, UnionFields},
    };
    let union = UnionArray::try_new(
        UnionFields::empty(),
        ScalarBuffer::from(Vec::<i8>::new()),
        None,
        vec![],
    )
    .unwrap();
    let field = Arc::new(Field::new("item", union.data_type().clone(), true));
    let rows =
        FixedSizeListArray::try_new_with_length(field, 0, Arc::new(union), None, 512).unwrap();
    assert_bound(&[Arc::new(rows)]);
}

#[test]
fn test_empty_union_null_fallback_reserves_actual_nested_constructor_peak() {
    use datafusion::arrow::datatypes::{DataType as D, Field, UnionFields, UnionMode};
    let empty = D::Union(UnionFields::empty(), UnionMode::Sparse);
    let field = |dtype| Arc::new(Field::new("item", dtype, true));
    let list = D::List(field(empty.clone()));
    let entry = Arc::new(Field::new(
        "entries",
        D::Struct(
            vec![
                Arc::new(Field::new("key", D::Int32, false)),
                field(list.clone()),
            ]
            .into(),
        ),
        false,
    ));
    let types = vec![
        list.clone(),
        D::LargeList(field(empty.clone())),
        D::List(field(D::Union(UnionFields::empty(), UnionMode::Dense))),
        D::List(field(D::Union(
            UnionFields::try_new(vec![3, 7], vec![field(D::Int32), field(list.clone())]).unwrap(),
            UnionMode::Dense,
        ))),
        D::ListView(field(empty.clone())),
        D::LargeListView(field(empty.clone())),
        D::FixedSizeList(field(empty.clone()), 0),
        D::FixedSizeList(Arc::new(Field::new("nonnull", empty.clone(), false)), 0),
        D::FixedSizeList(field(list.clone()), 3),
        D::Struct(
            vec![
                field(list.clone()),
                field(D::FixedSizeList(field(empty), 0)),
            ]
            .into(),
        ),
        D::Map(entry, false),
        D::Dictionary(Box::new(D::Int8), Box::new(list.clone())),
        D::List(field(D::RunEndEncoded(
            Arc::new(Field::new("ends", D::Int16, false)),
            field(list),
        ))),
    ];
    for dtype in types {
        let bytes = null_requirement(&dtype, &mut || Ok(())).unwrap();
        let mut scalar = None;
        let actual = allocation_counter::measure(|| {
            scalar = Some(
                super::super::materialize::canonical::null_scalar(&dtype, "null-peak", &mut || {
                    Ok(())
                })
                .unwrap(),
            );
        });
        assert!(
            bytes >= actual.bytes_max,
            "{dtype:?}: planned {bytes} observed {}",
            actual.bytes_max
        );
        drop(scalar);
    }
}

#[test]
fn test_view_metadata_bound_covers_pinned_to_data_vector_growth() {
    use datafusion::arrow::{
        array::StringViewArray,
        buffer::{Buffer, ScalarBuffer},
    };
    for count in [0, 1, 3] {
        let buffers = vec![Buffer::from(Vec::<u8>::new()); count];
        let source =
            StringViewArray::try_new(ScalarBuffer::from(vec![0_u128]), buffers, None).unwrap();
        let (_, _, _, _, buffers, _) = source.to_data().into_parts();
        assert!(descriptor::view_to_data_capacity(count).unwrap() >= buffers.capacity());
    }
}

#[test]
fn test_null_array_requirement_bounds_actual_guarded_array_construction() {
    use datafusion::arrow::datatypes::{DataType as D, Field, UnionFields, UnionMode};
    let types = vec![
        D::Utf8,
        D::FixedSizeList(Arc::new(Field::new("item", D::Int32, true)), 3),
        D::List(Arc::new(Field::new(
            "empty",
            D::Union(UnionFields::empty(), UnionMode::Dense),
            true,
        ))),
    ];
    for dtype in types {
        for len in [0, 3] {
            let mut bytes = 0;
            let scan = allocation_counter::measure(|| {
                bytes = null_array_requirement(&dtype, len, &mut || Ok(())).unwrap();
            });
            assert_eq!(scan.count_total, 0);
            let mut array = None;
            let actual = allocation_counter::measure(|| {
                array = Some(
                    super::super::materialize::canonical::null_array(
                        &dtype,
                        len,
                        "null-array-peak",
                        &mut || Ok(()),
                    )
                    .unwrap(),
                );
            });
            assert!(
                bytes >= actual.bytes_max,
                "{dtype:?}[{len}]: planned {bytes} observed {}",
                actual.bytes_max
            );
        }
    }
}
#[test]
fn test_selected_row_plan_does_not_copy_or_charge_discarded_payload() {
    let values: Vec<datafusion::arrow::array::ArrayRef> = vec![Arc::new(StringArray::from(vec![
        "x".repeat(4 << 20),
        "y".into(),
    ]))];
    let mut planned = None;
    let measured = allocation_counter::measure(|| {
        planned = Some(plan_selected([(values.as_slice(), 1)], &mut || Ok(())).unwrap());
    });
    assert_eq!(planned, Some(1));
    assert_eq!(measured.count_total, 0);
    assert!(plan_selected([(values.as_slice(), 2)], &mut || Ok(())).is_err());
}

#[test]
fn test_null_array_logical_cost_uses_actual_retained_capacities_without_private_headers() {
    use datafusion::arrow::array::{Array, Float64Array};
    let dtype = DataType::Float64;
    let cost = null_array_cost(&dtype, 1, &mut || Ok(())).unwrap();
    let mut array = None;
    let allocation = allocation_counter::measure(|| {
        array = Some(
            super::super::materialize::canonical::null_array(
                &dtype,
                1,
                "null-cost",
                &mut || Ok(()),
            )
            .unwrap(),
        );
    });
    let array = array.unwrap();
    let values = array.as_any().downcast_ref::<Float64Array>().unwrap();
    let payload = values.values().inner().capacity() + values.nulls().unwrap().buffer().capacity();
    let logical =
        descriptor_bytes(array.as_ref(), &mut || Ok(())).unwrap() + u64::try_from(payload).unwrap();
    eprintln!(
        "Float64[1] values_capacity={} mask_capacity={} logical={logical} observed_constructor_peak={}",
        values.values().inner().capacity(),
        values.nulls().unwrap().buffer().capacity(),
        allocation.bytes_max
    );
    assert_eq!(cost.logical_owned, logical);
    assert!(cost.construction_peak >= allocation.bytes_max);
    assert_eq!(
        null_array_requirement(&dtype, 1, &mut || Ok(())).unwrap(),
        cost.construction_peak
    );
}

#[test]
fn test_datatype_clone_requirement_counts_boxes_and_shares_field_metadata() {
    use datafusion::arrow::datatypes::Field;
    use std::collections::HashMap;
    let child = Arc::new(
        Field::new("nested", DataType::Utf8, true)
            .with_metadata(HashMap::from([("payload".into(), "metadata".repeat(512))])),
    );
    let dtype = DataType::Dictionary(
        Box::new(DataType::Int8),
        Box::new(DataType::Dictionary(
            Box::new(DataType::UInt16),
            Box::new(DataType::List(Arc::clone(&child))),
        )),
    );
    let mut bound = 0;
    let scan = allocation_counter::measure(|| {
        bound = datatype_clone_requirement(&dtype, &mut || Ok(())).unwrap();
    });
    assert_eq!(scan.count_total, 0);
    let mut copied = None;
    let allocation = allocation_counter::measure(|| {
        copied = Some(dtype.clone());
    });
    assert_eq!(bound, allocation.bytes_max);
    assert_eq!(bound, u64::try_from(4 * size_of::<DataType>()).unwrap());
    drop(copied.take());
    let parent = DataType::Struct(vec![child].into());
    assert_eq!(
        datatype_clone_requirement(&parent, &mut || Ok(())).unwrap(),
        0
    );
    let allocation = allocation_counter::measure(|| {
        copied = Some(parent.clone());
    });
    assert_eq!(allocation.count_total, 0);
    let error = datatype_clone_requirement(&parent, &mut || {
        Err(CalcFlowError::Cancelled {
            run_id: "dtype".into(),
        })
    })
    .err()
    .unwrap();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
}

use datafusion::{
    arrow::datatypes::{DataType, Field, Schema},
    common::ScalarValue,
};

use super::plan;

#[test]
fn test_plan_precedes_scalar_clones_and_arrow_construction() {
    let schema = Schema::new(vec![Field::new("value", DataType::Utf8, true)]);
    let value = ScalarValue::Utf8(Some("sample".into()));
    let plan = plan(&schema, 1, &mut |_, _| Some(&value), &mut || Ok(())).unwrap();
    let actual = allocation_counter::measure(|| {
        let array = ScalarValue::iter_to_array([value.clone()]).unwrap();
        std::hint::black_box(array);
    });
    assert!(actual.bytes_max <= plan.output_bytes + plan.scratch_bytes);
}

fn assert_column_fits(dtype: &DataType, values: &[Option<ScalarValue>]) {
    let schema = Schema::new(vec![Field::new("value", dtype.clone(), true)]);
    let rows = values.len();
    let workspace = super::planning_workspace(
        &schema,
        rows,
        &mut |_, row| values[row].as_ref(),
        &mut || Ok(()),
    )
    .unwrap();
    let mut planned = None;
    let allocation = allocation_counter::measure(|| {
        planned = Some(
            plan(
                &schema,
                rows,
                &mut |_: usize, row: usize| values[row].as_ref(),
                &mut || Ok(()),
            )
            .unwrap(),
        );
    });
    if workspace == 0 {
        assert_eq!(allocation.count_total, 0, "unfunded planner allocation");
    } else {
        assert!(
            allocation.bytes_max <= workspace,
            "{allocation:?} > planning workspace {workspace}"
        );
    }
    let planned = planned.unwrap();
    let actual = allocation_counter::measure(|| {
        let array = super::encode_column(
            dtype,
            rows,
            &mut |row: usize| values[row].as_ref(),
            &mut || Ok(()),
        )
        .unwrap();
        assert_eq!(array.len(), rows);
        std::hint::black_box(array);
    });
    assert!(
        actual.bytes_max <= planned.output_bytes + planned.scratch_bytes,
        "{dtype:?}: {actual:?}, plan {planned:?}"
    );
}
#[test]
fn test_scalar_and_typed_null_leaf_columns_fit_the_output_peak() {
    assert_column_fits(&DataType::Utf8, &[]);
    assert_column_fits(
        &DataType::Boolean,
        &[
            Some(ScalarValue::Boolean(Some(true))),
            None,
            Some(ScalarValue::Boolean(Some(false))),
        ],
    );
    let integers = (0..9)
        .map(|value| Some(ScalarValue::Int64(Some(value))))
        .collect::<Vec<_>>();
    assert_column_fits(&DataType::Int64, &integers);
    assert_column_fits(
        &DataType::Utf8,
        &[
            Some(ScalarValue::Utf8(Some("long".repeat(8192)))),
            None,
            Some(ScalarValue::Utf8(Some(String::new()))),
        ],
    );
    assert_column_fits(
        &DataType::Binary,
        &[Some(ScalarValue::Binary(Some(vec![1; 17]))), None],
    );
}
#[test]
fn test_scalar_view_blocks_have_their_own_growth_and_descriptor_costs() {
    assert_column_fits(
        &DataType::Utf8View,
        &[
            Some(ScalarValue::Utf8View(Some("outside-inline".into()))),
            None,
        ],
    );
    let values = ["a".repeat(8192), "b".repeat(8192), "c".repeat(32768)]
        .into_iter()
        .map(|value| Some(ScalarValue::Utf8View(Some(value))))
        .collect::<Vec<_>>();
    assert_column_fits(&DataType::Utf8View, &values);
}
#[test]
fn test_wrong_leaf_scalar_type_is_rejected_before_cloning() {
    let schema = Schema::new(vec![Field::new("value", DataType::Float64, true)]);
    let value = ScalarValue::Utf8(Some("not numeric".into()));
    assert!(plan(&schema, 1, &mut |_, _| Some(&value), &mut || Ok(())).is_err());
}

#[test]
fn test_empty_view_retained_descriptors_fit_output_credit() {
    let schema = Schema::new(vec![Field::new("view", DataType::Utf8View, true)]);
    let planned = plan(&schema, 0, &mut |_, _| None, &mut || Ok(())).unwrap();
    let array =
        super::encode_column(&DataType::Utf8View, 0, &mut |_| None, &mut || Ok(())).unwrap();
    let retained =
        super::super::scalar_materialization_plan::descriptor_bytes(array.as_ref(), &mut || Ok(()))
            .unwrap();
    assert!(
        retained <= planned.output_bytes,
        "descriptor {retained} > output {}",
        planned.output_bytes
    );
}

#[test]
fn test_unrepresentable_scalar_vector_fails_before_cell_access() {
    let schema = Schema::new(vec![Field::new("value", DataType::Float64, true)]);
    let rows = usize::MAX / size_of::<ScalarValue>();
    let result = plan(
        &schema,
        rows,
        &mut |_, _| panic!("cell accessed before vector capacity guard"),
        &mut || Ok(()),
    );
    assert!(result.is_err());
}

#[test]
fn test_single_borrowed_list_shares_funded_payload_without_new_output_charge() {
    use datafusion::arrow::{
        array::{Array, ArrayRef, ListArray, StringArray},
        buffer::OffsetBuffer,
    };
    use std::sync::Arc;
    let child = Arc::new(StringArray::from(vec!["already funded".repeat(100_000)]));
    let field = Arc::new(Field::new("item", DataType::Utf8, true));
    let list = ListArray::new(
        field,
        OffsetBuffer::new(vec![0_i32, 1].into()),
        child.clone(),
        None,
    );
    let value = ScalarValue::List(Arc::new(list));
    let dtype = value.data_type();
    let schema = Schema::new(vec![Field::new("list", dtype.clone(), true)]);
    let planned = plan(&schema, 1, &mut |_, _| Some(&value), &mut || Ok(())).unwrap();
    assert!(
        planned.output_bytes < 4096,
        "already-funded child backing was charged again"
    );
    let output = super::encode_column(&dtype, 1, &mut |_| Some(&value), &mut || Ok(())).unwrap();
    let output = output.as_any().downcast_ref::<ListArray>().unwrap();
    let child: ArrayRef = child;
    assert!(Arc::ptr_eq(output.values(), &child));
    assert_column_fits(&dtype, &[Some(value)]);
}

#[test]
fn test_single_nested_array_families_share_source_payloads() {
    use datafusion::arrow::{
        array::{
            Array, ArrayRef, FixedSizeListArray, LargeListArray, LargeListViewArray, ListViewArray,
            MapArray, StringArray, StructArray, UInt64Array,
        },
        buffer::OffsetBuffer,
    };
    use std::sync::Arc;
    let strings = Arc::new(StringArray::from(vec!["prepaid".repeat(100_000)]));
    let field = Arc::new(Field::new("item", DataType::Utf8, true));
    let large = LargeListArray::new(
        field.clone(),
        OffsetBuffer::new(vec![0_i64, 1].into()),
        strings.clone(),
        None,
    );
    let view = ListViewArray::new(
        field.clone(),
        vec![0_i32].into(),
        vec![1_i32].into(),
        strings.clone(),
        None,
    );
    let large_view = LargeListViewArray::new(
        field.clone(),
        vec![0_i64].into(),
        vec![1_i64].into(),
        strings.clone(),
        None,
    );
    let fixed = FixedSizeListArray::new(field, 1, strings.clone(), None);
    let structure = StructArray::from(vec![(
        Arc::new(Field::new("text", DataType::Utf8, true)),
        strings.clone() as ArrayRef,
    )]);
    let entries = StructArray::from(vec![
        (
            Arc::new(Field::new("key", DataType::UInt64, false)),
            Arc::new(UInt64Array::from(vec![1])) as ArrayRef,
        ),
        (
            Arc::new(Field::new("value", DataType::Utf8, true)),
            strings as ArrayRef,
        ),
    ]);
    let map = MapArray::new(
        Arc::new(Field::new("entries", entries.data_type().clone(), false)),
        OffsetBuffer::new(vec![0_i32, 1].into()),
        entries,
        None,
        false,
    );
    let values = [
        ScalarValue::LargeList(Arc::new(large)),
        ScalarValue::ListView(Arc::new(view)),
        ScalarValue::LargeListView(Arc::new(large_view)),
        ScalarValue::FixedSizeList(Arc::new(fixed)),
        ScalarValue::Struct(Arc::new(structure)),
        ScalarValue::Map(Arc::new(map)),
    ];
    for value in values {
        let dtype = value.data_type();
        let schema = Schema::new(vec![Field::new("value", dtype.clone(), true)]);
        let planned = plan(&schema, 1, &mut |_, _| Some(&value), &mut || Ok(())).unwrap();
        assert!(
            planned.output_bytes < 4096,
            "shared payload charged: {dtype:?}"
        );
        assert_column_fits(&dtype, &[Some(value)]);
    }
}

#[test]
fn test_single_typed_null_nested_hole_and_empty_encoded_columns_are_prepaid() {
    use std::sync::Arc;
    let field = Arc::new(Field::new("child", DataType::Utf8, true));
    for dtype in [
        DataType::Struct(vec![field.clone()].into()),
        DataType::FixedSizeList(field, 3),
    ] {
        assert_column_fits(&dtype, &[None]);
    }
    let dtype = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
    assert_column_fits(&dtype, &[]);
}

#[test]
fn test_list_concat_slices_values_but_list_view_concatenates_full_children() {
    use datafusion::arrow::{
        array::{ListArray, ListViewArray, StringArray},
        buffer::OffsetBuffer,
    };
    use std::sync::Arc;
    let child = Arc::new(StringArray::from(vec![
        "unreferenced".repeat(100_000),
        "x".into(),
        "y".into(),
    ]));
    let field = Arc::new(Field::new("item", DataType::Utf8, true));
    let list = ScalarValue::List(Arc::new(ListArray::new(
        field.clone(),
        OffsetBuffer::new(vec![1_i32, 3].into()),
        child.clone(),
        None,
    )));
    let dtype = list.data_type();
    let values = [Some(list.clone()), Some(list)];
    let schema = Schema::new(vec![Field::new("list", dtype.clone(), true)]);
    let output = plan(&schema, 2, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    })
    .unwrap();
    assert!(
        output.output_bytes < 4096,
        "unused list body copied: {output:?}"
    );
    assert_column_fits(&dtype, &values);
    let view = ScalarValue::ListView(Arc::new(ListViewArray::new(
        field,
        vec![1_i32].into(),
        vec![2_i32].into(),
        child,
        None,
    )));
    let dtype = view.data_type();
    let values = [Some(view.clone()), Some(view)];
    let schema = Schema::new(vec![Field::new("view", dtype.clone(), true)]);
    let output = plan(&schema, 2, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    })
    .unwrap();
    assert!(
        output.output_bytes >= 2_400_000,
        "full ListView child not planned: {output:?}"
    );
    assert_column_fits(&dtype, &values);
}

#[test]
fn test_struct_view_children_share_payload_and_map_columns_rebuild_entries() {
    use datafusion::arrow::{
        array::{
            Array, ArrayRef, MapArray, StringArray, StringViewArray, StructArray, UInt64Array,
        },
        buffer::OffsetBuffer,
    };
    use std::sync::Arc;
    let views = Arc::new(StringViewArray::from(vec!["shared".repeat(100_000)]));
    let structure = ScalarValue::Struct(Arc::new(StructArray::from(vec![(
        Arc::new(Field::new("view", DataType::Utf8View, false)),
        views as ArrayRef,
    )])));
    let dtype = structure.data_type();
    let values = [Some(structure.clone()), Some(structure)];
    let schema = Schema::new(vec![Field::new("structure", dtype.clone(), true)]);
    let output = plan(&schema, 2, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    })
    .unwrap();
    assert!(
        output.output_bytes < 4096,
        "shared view payload charged: {output:?}"
    );
    assert_column_fits(&dtype, &values);
    let entries = StructArray::from(vec![
        (
            Arc::new(Field::new("key", DataType::UInt64, false)),
            Arc::new(UInt64Array::from(vec![7])) as ArrayRef,
        ),
        (
            Arc::new(Field::new("value", DataType::Utf8, true)),
            Arc::new(StringArray::from(vec!["copied".repeat(10_000)])) as ArrayRef,
        ),
    ]);
    let map = ScalarValue::Map(Arc::new(MapArray::new(
        Arc::new(Field::new("entries", entries.data_type().clone(), false)),
        OffsetBuffer::new(vec![0_i32, 1].into()),
        entries,
        None,
        false,
    )));
    let dtype = map.data_type();
    assert_column_fits(&dtype, &[Some(map.clone()), Some(map)]);
}

#[test]
fn test_list_concat_prepays_all_sliced_struct_child_descriptors() {
    use datafusion::arrow::{
        array::{Array, ArrayRef, ListArray, StringArray, StructArray},
        buffer::OffsetBuffer,
    };
    use std::sync::Arc;
    let strings = Arc::new(StringArray::from(vec!["", ""]));
    let fields = (0..100)
        .map(|index| {
            (
                Arc::new(Field::new(format!("value{index}"), DataType::Utf8, false)),
                strings.clone() as ArrayRef,
            )
        })
        .collect::<Vec<_>>();
    let children = Arc::new(StructArray::from(fields));
    let list = ScalarValue::List(Arc::new(ListArray::new(
        Arc::new(Field::new("structure", children.data_type().clone(), true)),
        OffsetBuffer::new(vec![1_i32, 2].into()),
        children,
        None,
    )));
    let dtype = list.data_type();
    assert_column_fits(&dtype, &vec![Some(list); 100]);
}

#[test]
fn test_dictionary_column_plans_only_physically_valid_key_positions() {
    let null = ScalarValue::Dictionary(
        Box::new(DataType::Int8),
        Box::new(ScalarValue::Float64(None)),
    );
    let valid = ScalarValue::Dictionary(
        Box::new(DataType::Int8),
        Box::new(ScalarValue::Float64(Some(7.0))),
    );
    let dtype = valid.data_type();
    let mut values = vec![Some(null); 129];
    values[0] = Some(valid.clone());
    assert_column_fits(&dtype, &values);
    values[128] = Some(valid);
    let schema = Schema::new(vec![Field::new("dictionary", dtype, true)]);
    assert!(
        plan(
            &schema,
            129,
            &mut |_, row| values[row].as_ref(),
            &mut || Ok(())
        )
        .is_err()
    );
}

#[test]
fn test_run_comparison_workspace_is_prepaid_without_copying_source_payload() {
    use datafusion::arrow::{
        array::{ListArray, StringArray},
        buffer::OffsetBuffer,
    };
    use std::sync::Arc;
    let list = ScalarValue::List(Arc::new(ListArray::new(
        Arc::new(Field::new("text", DataType::Utf8, true)),
        OffsetBuffer::new(vec![0_i32, 1].into()),
        Arc::new(StringArray::from(vec!["prepaid".repeat(100_000)])),
        None,
    )));
    let value = ScalarValue::RunEndEncoded(
        Arc::new(Field::new("ends", DataType::Int32, false)),
        Arc::new(Field::new("value", list.data_type(), true)),
        Box::new(list.clone()),
    );
    let schema = Schema::new(vec![Field::new("run", value.data_type(), true)]);
    let mut workspace = 0;
    let prefix = allocation_counter::measure(|| {
        workspace = super::planning_workspace(&schema, 2, &mut |_, _| Some(&value), &mut || Ok(()))
            .unwrap();
    });
    assert_eq!(prefix.count_total, 0);
    let same = list.clone();
    let actual = allocation_counter::measure(|| assert!(list == same));
    assert!(
        workspace >= actual.bytes_max,
        "comparison peak {actual:?} > workspace {workspace}"
    );
    assert!(workspace < 4096, "prepaid payload charged: {workspace}");
    let plain = Schema::new(vec![Field::new("plain", DataType::Float64, true)]);
    assert_eq!(
        super::planning_workspace(
            &plain,
            2,
            &mut |_, _| panic!("no scalar comparison for primitive columns"),
            &mut || Ok(())
        )
        .unwrap(),
        0
    );
}

#[test]
fn test_run_plan_uses_real_coalesced_values_and_preserves_null_comparisons() {
    use datafusion::arrow::{
        array::{Array, RunArray},
        datatypes::Int32Type,
    };
    use std::sync::Arc;
    let inner = ScalarValue::Dictionary(
        Box::new(DataType::Int8),
        Box::new(ScalarValue::Float64(Some(1.0))),
    );
    let ends = Arc::new(Field::new("ends", DataType::Int32, false));
    let value = Arc::new(Field::new("values", inner.data_type(), true));
    let run = ScalarValue::RunEndEncoded(ends.clone(), value, Box::new(inner));
    let dtype = run.data_type();
    let values = vec![Some(run); 129];
    assert_column_fits(&dtype, &values);
    let output =
        super::encode_column(&dtype, 129, &mut |row| values[row].as_ref(), &mut || Ok(())).unwrap();
    let array = output
        .as_any()
        .downcast_ref::<RunArray<Int32Type>>()
        .unwrap();
    assert_eq!(array.values().len(), 1);
    let null = ScalarValue::RunEndEncoded(
        ends,
        Arc::new(Field::new("values", DataType::Float64, true)),
        Box::new(ScalarValue::Float64(None)),
    );
    let dtype = null.data_type();
    assert_column_fits(&dtype, &[None, Some(null), None]);
}

#[test]
fn test_one_array_data_tree_workspace_covers_variable_view_buffer_metadata() {
    use datafusion::arrow::array::{Array, StringViewArray};
    let strings = (0..7)
        .map(|index| format!("{index}{}", "body".repeat(8192)))
        .collect::<Vec<_>>();
    let array = StringViewArray::from_iter_values(strings);
    let mut planned = 0;
    let prefix = allocation_counter::measure(|| {
        planned = super::array_data_workspace(&array, &mut || Ok(())).unwrap();
    });
    assert_eq!(prefix.count_total, 0);
    let actual = allocation_counter::measure(|| {
        std::hint::black_box(array.to_data());
    });
    assert!(
        actual.bytes_max <= planned,
        "{actual:?} > metadata {planned}"
    );
    assert!(planned < 8192, "source view payload charged as metadata");
}

#[test]
fn test_nested_typed_holes_are_planned_before_materialization() {
    use datafusion::arrow::{
        array::{ListArray, StringArray},
        buffer::OffsetBuffer,
    };
    use std::sync::Arc;
    let list = ScalarValue::List(Arc::new(ListArray::new(
        Arc::new(Field::new("text", DataType::Utf8, true)),
        OffsetBuffer::new(vec![0_i32, 1].into()),
        Arc::new(StringArray::from(vec!["materialized"])),
        None,
    )));
    let dtype = list.data_type();
    assert_column_fits(&dtype, &[None, Some(list), None]);
    let schema = Schema::new(vec![Field::new("input", dtype.clone(), false)]);
    assert!(plan(&schema, 2, &mut |_, _| None, &mut || Ok(())).is_ok());
}

#[test]
fn test_map_concat_prepays_recursive_entry_slices() {
    use datafusion::arrow::{
        array::{Array, ArrayRef, MapArray, StringArray, StructArray, UInt64Array},
        buffer::OffsetBuffer,
    };
    use std::sync::Arc;
    let text = Arc::new(StringArray::from(vec!["", ""]));
    let columns = (0..100)
        .map(|index| {
            (
                Arc::new(Field::new(format!("field{index}"), DataType::Utf8, false)),
                text.clone() as ArrayRef,
            )
        })
        .collect::<Vec<_>>();
    let fields = Arc::new(StructArray::from(columns));
    let entries = StructArray::from(vec![
        (
            Arc::new(Field::new("key", DataType::UInt64, false)),
            Arc::new(UInt64Array::from(vec![1, 2])) as ArrayRef,
        ),
        (
            Arc::new(Field::new("value", fields.data_type().clone(), false)),
            fields as ArrayRef,
        ),
    ]);
    let map = ScalarValue::Map(Arc::new(MapArray::new(
        Arc::new(Field::new("entries", entries.data_type().clone(), false)),
        OffsetBuffer::new(vec![1_i32, 2].into()),
        entries,
        None,
        false,
    )));
    let dtype = map.data_type();
    assert_column_fits(&dtype, &vec![Some(map); 100]);
}

#[test]
fn test_fixed_list_null_repair_is_prepaid_before_fallback_construction() {
    use datafusion::arrow::array::{FixedSizeListArray, Float64Array};
    use std::sync::Arc;
    let field = Arc::new(Field::new("value", DataType::Float64, true));
    let null =
        ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new_null(field.clone(), 1, 1)));
    let valid = ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new(
        field,
        2,
        Arc::new(Float64Array::from(vec![1.0, 2.0])),
        None,
    )));
    let dtype = valid.data_type();
    assert_column_fits(&dtype, &[Some(null), Some(valid), None]);
}

#[test]
fn test_fixed_list_byte_capacity_uses_logical_offsets_not_source_backing() {
    use datafusion::arrow::array::{FixedSizeListArray, StringArray};
    use std::sync::Arc;
    let strings = StringArray::from(vec!["unused".repeat(100_000), "x".into()]);
    let child = Arc::new(strings.slice(1, 1));
    let field = Arc::new(Field::new("text", DataType::Utf8, true));
    let value =
        ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new(field, 1, child, None)));
    let dtype = value.data_type();
    let values = [Some(value.clone()), None, Some(value)];
    let schema = Schema::new(vec![Field::new("fixed", dtype.clone(), true)]);
    let planned = plan(&schema, 3, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    })
    .unwrap();
    assert!(
        planned.output_bytes < 4096,
        "unused source backing charged: {planned:?}"
    );
    assert_column_fits(&dtype, &values);
}

#[test]
fn test_fixed_boolean_and_null_children_have_distinct_buffer_geometry() {
    use datafusion::arrow::array::{BooleanArray, FixedSizeListArray, NullArray};
    use std::sync::Arc;
    let flags = ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("flag", DataType::Boolean, true)),
        2,
        Arc::new(BooleanArray::from(vec![Some(true), None])),
        None,
    )));
    assert_column_fits(
        &flags.data_type(),
        &[Some(flags.clone()), None, Some(flags)],
    );
    let nulls = ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("null", DataType::Null, true)),
        2,
        Arc::new(NullArray::new(2)),
        None,
    )));
    assert_column_fits(
        &nulls.data_type(),
        &[Some(nulls.clone()), None, Some(nulls)],
    );
}

#[test]
fn test_fixed_list_view_children_keep_source_payload_shared() {
    use datafusion::arrow::array::{FixedSizeListArray, StringViewArray};
    use std::sync::Arc;
    let child = Arc::new(StringViewArray::from(vec!["immutable".repeat(100_000)]));
    let field = Arc::new(Field::new("view", DataType::Utf8View, true));
    let value =
        ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new(field, 1, child, None)));
    let dtype = value.data_type();
    let values = [Some(value.clone()), None, Some(value)];
    let schema = Schema::new(vec![Field::new("fixed", dtype.clone(), true)]);
    let planned = plan(&schema, 3, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    })
    .unwrap();
    assert!(
        planned.output_bytes < 4096,
        "shared view body charged: {planned:?}"
    );
    assert_column_fits(&dtype, &values);
}

#[test]
fn test_fixed_struct_retained_columns_do_not_borrow_output_scratch() {
    use datafusion::arrow::array::{
        Array, ArrayRef, FixedSizeListArray, Float64Array, StructArray,
    };
    use std::sync::Arc;
    let leaf = Arc::new(Float64Array::from(vec![7.0]));
    let fields = (0..64)
        .map(|index| {
            (
                Arc::new(Field::new(
                    format!("value{index}"),
                    DataType::Float64,
                    false,
                )),
                leaf.clone() as ArrayRef,
            )
        })
        .collect::<Vec<_>>();
    let child = Arc::new(StructArray::from(fields));
    let value = ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("row", child.data_type().clone(), true)),
        1,
        child,
        None,
    )));
    let dtype = value.data_type();
    let values = [Some(value.clone()), Some(value)];
    let schema = Schema::new(vec![Field::new("fixed", dtype.clone(), true)]);
    let planned = plan(&schema, 2, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    })
    .unwrap();
    let mut output = None;
    let actual = allocation_counter::measure(|| {
        output = Some(
            super::encode_column(&dtype, 2, &mut |row| values[row].as_ref(), &mut || Ok(()))
                .unwrap(),
        );
    });
    let private_headers = 64 * 7 * u64::try_from(size_of::<usize>()).unwrap();
    let logical = u64::try_from(actual.bytes_current)
        .unwrap()
        .checked_sub(private_headers)
        .unwrap();
    assert!(
        logical <= planned.output_bytes,
        "retained logical {logical} > {planned:?}"
    );
    drop(output);
    assert_column_fits(&dtype, &values);
}

#[test]
fn test_fixed_list_nested_variable_ranges_are_funded_before_mutable_construction() {
    use datafusion::arrow::{
        array::{Array, FixedSizeListArray, ListArray},
        datatypes::Float64Type,
    };
    use std::sync::Arc;
    let child = Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>([Some(
        vec![Some(2.0); 4097],
    )]));
    let value = ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("variable", child.data_type().clone(), true)),
        1,
        child,
        None,
    )));
    let dtype = value.data_type();
    assert_column_fits(&dtype, &[Some(value.clone()), None, Some(value)]);
}

#[test]
fn test_fixed_list_view_copies_selected_child_ranges_without_unused_byte_backing() {
    use datafusion::arrow::array::{Array, FixedSizeListArray, ListViewArray, StringArray};
    use std::sync::Arc;
    let child = Arc::new(ListViewArray::new(
        Arc::new(Field::new("text", DataType::Utf8, false)),
        vec![1_i32, 1].into(),
        vec![1_i32, 1].into(),
        Arc::new(StringArray::from(vec![
            "unused".repeat(100_000),
            "x".into(),
        ])),
        None,
    ));
    let value = ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("views", child.data_type().clone(), true)),
        2,
        child,
        None,
    )));
    let dtype = value.data_type();
    let values = [Some(value.clone()), None, Some(value)];
    let schema = Schema::new(vec![Field::new("fixed", dtype.clone(), true)]);
    let planned = plan(&schema, 3, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    })
    .unwrap();
    assert!(planned.output_bytes < 4096, "{planned:?}");
    assert_column_fits(&dtype, &values);
}

#[test]
fn test_fixed_nested_view_buffers_are_retained_once_per_constructor_input() {
    use datafusion::arrow::array::{Array, FixedSizeListArray, ListViewArray, StringViewArray};
    use std::sync::Arc;
    let child = Arc::new(ListViewArray::new(
        Arc::new(Field::new("text", DataType::Utf8View, false)),
        vec![1_i32, 1].into(),
        vec![1_i32, 1].into(),
        Arc::new(StringViewArray::from(vec![
            "unused".repeat(100_000),
            "a copied long view".into(),
        ])),
        None,
    ));
    let value = ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("views", child.data_type().clone(), true)),
        2,
        child,
        None,
    )));
    let dtype = value.data_type();
    let values = [Some(value.clone()), None, Some(value)];
    let schema = Schema::new(vec![Field::new("fixed", dtype.clone(), true)]);
    let planned = plan(&schema, 3, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    })
    .unwrap();
    assert!(planned.output_bytes < 4096, "{planned:?}");
    assert_column_fits(&dtype, &values);
}

#[test]
fn test_fixed_struct_list_views_do_not_use_normal_concat_full_children() {
    use datafusion::arrow::array::{
        Array, ArrayRef, FixedSizeListArray, ListViewArray, StringArray, StructArray,
    };
    use std::sync::Arc;
    let lists = Arc::new(ListViewArray::new(
        Arc::new(Field::new("text", DataType::Utf8, false)),
        vec![1_i32].into(),
        vec![1_i32].into(),
        Arc::new(StringArray::from(vec![
            "unused".repeat(100_000),
            "x".into(),
        ])),
        None,
    ));
    let child = Arc::new(StructArray::from(vec![(
        Arc::new(Field::new("selected", lists.data_type().clone(), true)),
        lists as ArrayRef,
    )]));
    let value = ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("row", child.data_type().clone(), true)),
        1,
        child,
        None,
    )));
    let dtype = value.data_type();
    let values = [Some(value.clone()), None, Some(value)];
    let schema = Schema::new(vec![Field::new("fixed", dtype.clone(), true)]);
    let planned = plan(&schema, 3, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    })
    .unwrap();
    assert!(planned.output_bytes < 4096, "{planned:?}");
    assert_column_fits(&dtype, &values);
}

#[test]
fn test_fixed_dictionary_shared_values_do_not_become_new_output_payload() {
    use datafusion::arrow::{
        array::{Array, DictionaryArray, FixedSizeListArray, Int8Array, StringArray},
        datatypes::Int8Type,
    };
    use std::sync::Arc;
    let dictionary = Arc::new(DictionaryArray::<Int8Type>::new(
        Int8Array::from(vec![1]),
        Arc::new(StringArray::from(vec![
            "unused".repeat(100_000),
            "x".into(),
        ])),
    ));
    let value = ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new(
            "dictionary",
            dictionary.data_type().clone(),
            true,
        )),
        1,
        dictionary,
        None,
    )));
    let dtype = value.data_type();
    let values = [Some(value.clone()), Some(value)];
    let schema = Schema::new(vec![Field::new("fixed", dtype.clone(), true)]);
    let workspace =
        super::planning_workspace(&schema, 2, &mut |_, row| values[row].as_ref(), &mut || {
            Ok(())
        })
        .unwrap();
    assert!(
        workspace > 0,
        "dictionary ptr_eq metadata requires prior credit"
    );
    let planned = plan(&schema, 2, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    })
    .unwrap();
    assert!(planned.output_bytes < 4096, "{planned:?}");
    assert_column_fits(&dtype, &values);
    assert_retained_column_fits(&dtype, &values, 1);
    assert_column_fits(&dtype, &[values[0].clone(), None]);
    assert_column_fits(&dtype, &[None, None]);
}

#[test]
fn test_fixed_dictionary_constructor_copies_full_values_even_without_copied_keys() {
    use datafusion::arrow::{
        array::{
            Array, DictionaryArray, FixedSizeListArray, Int8Array, ListViewArray, StringArray,
        },
        datatypes::Int8Type,
    };
    use std::sync::Arc;
    let values = ["a", "b"].map(|letter| {
        let dictionary = Arc::new(DictionaryArray::<Int8Type>::new(
            Int8Array::from(vec![0]),
            Arc::new(StringArray::from(vec![letter.repeat(600_000)])),
        ));
        let lists = Arc::new(ListViewArray::new(
            Arc::new(Field::new(
                "dictionary",
                dictionary.data_type().clone(),
                true,
            )),
            vec![0_i32].into(),
            vec![0_i32].into(),
            dictionary,
            None,
        ));
        Some(ScalarValue::FixedSizeList(Arc::new(
            FixedSizeListArray::new(
                Arc::new(Field::new("list", lists.data_type().clone(), true)),
                1,
                lists,
                None,
            ),
        )))
    });
    let dtype = values[0].as_ref().unwrap().data_type();
    let schema = Schema::new(vec![Field::new("fixed", dtype.clone(), true)]);
    let planned = plan(&schema, 2, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    })
    .unwrap();
    assert!(planned.output_bytes >= 1_200_000, "{planned:?}");
    assert_column_fits(&dtype, &values);
}

#[test]
fn test_fixed_dictionary_physical_value_identity_preserves_exact_key_span() {
    use datafusion::arrow::{
        array::{
            Array, ArrayRef, DictionaryArray, FixedSizeListArray, Float64Array, Int8Array,
            NullArray,
        },
        datatypes::Int8Type,
    };
    use std::sync::Arc;
    let fixed = |values: ArrayRef| {
        let dictionary = Arc::new(DictionaryArray::<Int8Type>::new(
            Int8Array::from(vec![0]),
            values,
        ));
        Some(ScalarValue::FixedSizeList(Arc::new(
            FixedSizeListArray::new(
                Arc::new(Field::new(
                    "dictionary",
                    dictionary.data_type().clone(),
                    true,
                )),
                1,
                dictionary,
                None,
            ),
        )))
    };
    let nulls = (0..129)
        .map(|_| fixed(Arc::new(NullArray::new(1))))
        .collect::<Vec<_>>();
    assert_column_fits(&nulls[0].as_ref().unwrap().data_type(), &nulls);
    let floats = (0..129)
        .map(|_| fixed(Arc::new(Float64Array::from(vec![None]))))
        .collect::<Vec<_>>();
    let dtype = floats[0].as_ref().unwrap().data_type();
    assert_column_fits(&dtype, &floats[..128]);
    let schema = Schema::new(vec![Field::new("fixed", dtype.clone(), true)]);
    assert!(
        plan(
            &schema,
            129,
            &mut |_, row| floats[row].as_ref(),
            &mut || Ok(())
        )
        .is_err()
    );
    assert!(
        super::encode_column(&dtype, 129, &mut |row| floats[row].as_ref(), &mut || Ok(())).is_err()
    );
}

#[test]
fn test_fixed_run_children_copy_physical_spans_and_share_view_payload() {
    use datafusion::arrow::{
        array::{Array, FixedSizeListArray, Int16Array, RunArray, StringViewArray},
        datatypes::Int16Type,
    };
    use std::sync::Arc;
    let runs = RunArray::<Int16Type>::try_new(
        &Int16Array::from(vec![2, 4]),
        &StringViewArray::from(vec!["unused".repeat(100_000), "a copied long view".into()]),
    )
    .unwrap();
    let slice = Arc::new(runs.slice(2, 1));
    let value = ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("run", slice.data_type().clone(), true)),
        1,
        slice,
        None,
    )));
    let dtype = value.data_type();
    let values = [Some(value.clone()), None, Some(value)];
    let schema = Schema::new(vec![Field::new("fixed", dtype.clone(), true)]);
    assert_eq!(
        super::planning_workspace(&schema, 3, &mut |_, row| values[row].as_ref(), &mut || Ok(
            ()
        ))
        .unwrap(),
        0
    );
    let planned = plan(&schema, 3, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    })
    .unwrap();
    assert!(planned.output_bytes < 4096, "{planned:?}");
    assert_column_fits(&dtype, &values);
    assert_retained_column_fits(&dtype, &values, 4);
}

fn assert_retained_column_fits(
    dtype: &DataType,
    values: &[Option<ScalarValue>],
    private_headers: u64,
) {
    let schema = Schema::new(vec![Field::new("value", dtype.clone(), true)]);
    let planned = plan(
        &schema,
        values.len(),
        &mut |_, row| values[row].as_ref(),
        &mut || Ok(()),
    )
    .unwrap();
    let mut output = None;
    let actual = allocation_counter::measure(|| {
        output = Some(
            super::encode_column(
                dtype,
                values.len(),
                &mut |row| values[row].as_ref(),
                &mut || Ok(()),
            )
            .unwrap(),
        );
    });
    let private_headers = private_headers * 7 * u64::try_from(size_of::<usize>()).unwrap();
    let logical = u64::try_from(actual.bytes_current)
        .unwrap()
        .checked_sub(private_headers)
        .unwrap();
    assert!(
        logical <= planned.output_bytes,
        "retained {logical} > {planned:?}"
    );
    drop(output);
}

#[test]
fn test_fixed_run_end_limit_is_checked_before_builder_allocation() {
    use datafusion::arrow::{
        array::{Array, FixedSizeListArray, Float64Array, Int16Array, RunArray},
        datatypes::Int16Type,
    };
    use std::sync::Arc;
    let fixed = |span: i16| {
        let runs = Arc::new(
            RunArray::<Int16Type>::try_new(
                &Int16Array::from(vec![span]),
                &Float64Array::from(vec![1.0]),
            )
            .unwrap(),
        );
        ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new(
            Arc::new(Field::new("run", runs.data_type().clone(), true)),
            i32::from(span),
            runs,
            None,
        )))
    };
    let valid = fixed(16_383);
    assert_column_fits(&valid.data_type(), &[Some(valid.clone()), Some(valid)]);
    let invalid = fixed(20_000);
    let dtype = invalid.data_type();
    let schema = Schema::new(vec![Field::new("fixed", dtype.clone(), true)]);
    let mut result = None;
    let observed = allocation_counter::measure(|| {
        result = Some(plan(&schema, 2, &mut |_, _| Some(&invalid), &mut || Ok(())));
    });
    assert!(result.unwrap().is_err());
    assert!(
        observed.bytes_max < 4096,
        "builder allocation preceded range rejection: {observed:?}"
    );
    assert!(super::encode_column(&dtype, 2, &mut |_| Some(&invalid), &mut || Ok(())).is_err());
}

fn fixed_union_value(mode: datafusion::arrow::datatypes::UnionMode) -> ScalarValue {
    use datafusion::arrow::{
        array::{Array, ArrayRef, FixedSizeListArray, StringArray, StringViewArray, UnionArray},
        datatypes::{UnionFields, UnionMode},
    };
    use std::sync::Arc;
    let fields = UnionFields::try_new(
        [0, 1],
        [
            Field::new("selected", DataType::Utf8View, true),
            Field::new("inactive", DataType::Utf8, true),
        ],
    )
    .unwrap();
    let views = StringViewArray::from(vec!["unused".repeat(100_000), "a copied long view".into()]);
    let selected: ArrayRef = if mode == UnionMode::Sparse {
        Arc::new(views.slice(1, 1))
    } else {
        Arc::new(views)
    };
    let offsets = (mode == UnionMode::Dense).then(|| vec![1_i32].into());
    let union = Arc::new(
        UnionArray::try_new(
            fields,
            vec![0_i8].into(),
            offsets,
            vec![
                selected,
                Arc::new(StringArray::from(vec!["inactive".repeat(100_000)])),
            ],
        )
        .unwrap(),
    );
    ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("union", union.data_type().clone(), true)),
        1,
        union,
        None,
    )))
}
#[test]
fn test_fixed_union_dense_does_not_copy_inactive_child_payload() {
    let value = fixed_union_value(datafusion::arrow::datatypes::UnionMode::Dense);
    let dtype = value.data_type();
    let values = [Some(value.clone()), None, Some(value)];
    let schema = Schema::new(vec![Field::new("fixed", dtype.clone(), true)]);
    let planned = plan(&schema, 3, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    })
    .unwrap();
    assert!(planned.output_bytes < 4096, "{planned:?}");
    assert_column_fits(&dtype, &values);
}
#[test]
fn test_fixed_union_sparse_copies_inactive_child_payload() {
    let value = fixed_union_value(datafusion::arrow::datatypes::UnionMode::Sparse);
    let dtype = value.data_type();
    let values = [Some(value.clone()), None, Some(value)];
    let schema = Schema::new(vec![Field::new("fixed", dtype.clone(), true)]);
    let planned = plan(&schema, 3, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    })
    .unwrap();
    assert!(planned.output_bytes >= 1_600_000, "{planned:?}");
    assert_column_fits(&dtype, &values);
}

fn fixed_binary_value(structured: bool) -> ScalarValue {
    use datafusion::arrow::array::{
        Array, ArrayRef, FixedSizeBinaryArray, FixedSizeListArray, StructArray,
    };
    use std::sync::Arc;
    let binary = Arc::new(FixedSizeBinaryArray::try_from_iter([b"ab", b"cd"].into_iter()).unwrap());
    let child: ArrayRef = if structured {
        Arc::new(StructArray::from(vec![(
            Arc::new(Field::new("bytes", binary.data_type().clone(), true)),
            binary as ArrayRef,
        )]))
    } else {
        binary
    };
    ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("value", child.data_type().clone(), true)),
        2,
        child,
        None,
    )))
}
#[test]
fn test_fixed_binary_child_data_has_fixed_width_buffer_capacity() {
    let value = fixed_binary_value(false);
    assert_column_fits(
        &value.data_type(),
        &[Some(value.clone()), None, Some(value)],
    );
}
#[test]
fn test_fixed_binary_struct_child_data_has_fixed_width_buffer_capacity() {
    let value = fixed_binary_value(true);
    assert_column_fits(
        &value.data_type(),
        &[Some(value.clone()), None, Some(value)],
    );
}

fn list_run_scalar(empty: bool) -> ScalarValue {
    use datafusion::arrow::{
        array::{Array, Int16Array, ListArray, RunArray, StringArray},
        buffer::OffsetBuffer,
        datatypes::Int16Type,
    };
    use std::sync::Arc;
    let run = Arc::new(
        RunArray::<Int16Type>::try_new(
            &Int16Array::from(vec![1]),
            &StringArray::from(vec!["payload".repeat(100_000)]),
        )
        .unwrap(),
    );
    ScalarValue::List(Arc::new(ListArray::new(
        Arc::new(Field::new("run", run.data_type().clone(), true)),
        OffsetBuffer::new(vec![0_i32, i32::from(!empty)].into()),
        run,
        None,
    )))
}
#[test]
fn test_normal_run_filters_empty_inputs_before_shared_value_concat() {
    let values = [Some(list_run_scalar(true)), Some(list_run_scalar(false))];
    let dtype = values[0].as_ref().unwrap().data_type();
    let schema = Schema::new(vec![Field::new("list", dtype.clone(), true)]);
    assert!(
        super::planning_workspace(&schema, 2, &mut |_, row| values[row].as_ref(), &mut || Ok(
            ()
        ))
        .unwrap()
            > 0
    );
    let planned = plan(&schema, 2, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    })
    .unwrap();
    assert!(
        planned.output_bytes < 4096,
        "shared run values were copied: {planned:?}"
    );
    assert_column_fits(&dtype, &values);
    assert_retained_column_fits(&dtype, &values, 2);
}
#[test]
fn test_normal_run_two_value_slices_use_normal_byte_concat() {
    let values = [Some(list_run_scalar(false)), Some(list_run_scalar(false))];
    let dtype = values[0].as_ref().unwrap().data_type();
    let schema = Schema::new(vec![Field::new("list", dtype.clone(), true)]);
    let planned = plan(&schema, 2, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    })
    .unwrap();
    assert!(planned.output_bytes >= 1_400_000, "{planned:?}");
    assert_column_fits(&dtype, &values);
}

#[test]
fn test_normal_run_all_filtered_empty_keeps_the_pinned_controlled_error() {
    let values = [Some(list_run_scalar(true)), Some(list_run_scalar(true))];
    let dtype = values[0].as_ref().unwrap().data_type();
    let schema = Schema::new(vec![Field::new("nested_run", dtype.clone(), true)]);
    let result = plan(&schema, 2, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    });
    assert!(
        matches!(result, Err(crate::CalcFlowError::Format {message}) if message.contains("empty normal run concat"))
    );
    assert!(
        super::encode_column(&dtype, 2, &mut |row| values[row].as_ref(), &mut || Ok(())).is_err()
    );
}

fn overflowing_large_list() -> ScalarValue {
    use datafusion::arrow::{
        array::{LargeListArray, NullArray},
        buffer::OffsetBuffer,
    };
    use std::sync::Arc;
    let span =
        (u64::try_from(usize::MAX).unwrap() / 2 + 1).min(u64::try_from(i64::MAX).unwrap() / 2 + 1);
    let len = usize::try_from(span).unwrap();
    ScalarValue::LargeList(Arc::new(LargeListArray::new(
        Arc::new(Field::new("null", DataType::Null, true)),
        OffsetBuffer::new(vec![0_i64, i64::try_from(span).unwrap()].into()),
        Arc::new(NullArray::new(len)),
        None,
    )))
}
#[test]
fn test_normal_list_offset_overflow_is_rejected_without_allocating_child_work() {
    let value = overflowing_large_list();
    let schema = Schema::new(vec![Field::new("large_list", value.data_type(), true)]);
    assert!(plan(&schema, 2, &mut |_, _| Some(&value), &mut || Ok(())).is_err());
}

#[test]
fn test_normal_list_producer_checks_offset_before_large_null_child_work() {
    let value = overflowing_large_list();
    let dtype = value.data_type();
    let mut result = None;
    let observed = allocation_counter::measure(|| {
        result = Some(super::encode_column(
            &dtype,
            2,
            &mut |_| Some(&value),
            &mut || Ok(()),
        ));
    });
    assert!(result.unwrap().is_err());
    assert!(
        observed.bytes_max < 16_384,
        "child work allocated before offset rejection: {observed:?}"
    );
}

#[test]
fn test_controlled_normal_list_view_output_peak_and_retained_buffers_are_funded() {
    use datafusion::arrow::array::{ListViewArray, NullArray};
    use std::sync::Arc;
    let value = ScalarValue::ListView(Arc::new(ListViewArray::new(
        Arc::new(Field::new("item", DataType::Null, true)),
        vec![0].into(),
        vec![1].into(),
        Arc::new(NullArray::new(1)),
        None,
    )));
    let dtype = value.data_type();
    let values = [Some(value.clone()), Some(value)];
    assert_column_fits(&dtype, &values);
    assert_retained_column_fits(&dtype, &values, 2);
}

#[test]
fn test_controlled_normal_list_view_ancestor_columns_and_null_holes_fit_peak() {
    use datafusion::arrow::{
        array::{
            Array, ArrayRef, Float64Array, Int16Array, ListArray, ListViewArray, MapArray,
            RunArray, StructArray, UInt64Array,
        },
        buffer::OffsetBuffer,
        datatypes::Int16Type,
    };
    use std::sync::Arc;
    let view: ArrayRef = Arc::new(ListViewArray::new(
        Arc::new(Field::new("item", DataType::Float64, true)),
        vec![2, 0].into(),
        vec![1, 1].into(),
        Arc::new(Float64Array::from(vec![1., 2., 3., 4.])),
        None,
    ));
    let field = Arc::new(Field::new("view", view.data_type().clone(), true));
    let list = ScalarValue::List(Arc::new(ListArray::new(
        field.clone(),
        OffsetBuffer::new(vec![0, 2].into()),
        view.clone(),
        None,
    )));
    let structure = ScalarValue::Struct(Arc::new(StructArray::new(
        vec![field.clone()].into(),
        vec![view.slice(0, 1)],
        None,
    )));
    let entries = StructArray::new(
        vec![
            Field::new("key", DataType::UInt64, false),
            field.as_ref().clone(),
        ]
        .into(),
        vec![Arc::new(UInt64Array::from(vec![1, 2])), view.clone()],
        None,
    );
    let map = ScalarValue::Map(Arc::new(MapArray::new(
        Arc::new(Field::new("entries", entries.data_type().clone(), false)),
        OffsetBuffer::new(vec![0, 2].into()),
        entries,
        None,
        false,
    )));
    let runs = Arc::new(
        RunArray::<Int16Type>::try_new(&Int16Array::from(vec![2, 4]), view.as_ref()).unwrap(),
    );
    let run = ScalarValue::List(Arc::new(ListArray::new(
        Arc::new(Field::new("run", runs.data_type().clone(), true)),
        OffsetBuffer::new(vec![0, 4].into()),
        runs,
        None,
    )));
    for value in [list, structure, map, run] {
        let dtype = value.data_type();
        assert_column_fits(&dtype, &[Some(value.clone()), None, Some(value)]);
    }
}

#[test]
fn test_normal_copied_views_preserve_unused_blocks_and_reference_indices() {
    use datafusion::arrow::{
        array::{Array, ByteView, ListArray, StringViewArray},
        buffer::OffsetBuffer,
    };
    use std::sync::Arc;
    let source = Arc::new(StringViewArray::from(vec![
        "already-funded-long-value".repeat(1024),
    ]));
    let value = ScalarValue::List(Arc::new(ListArray::new(
        Arc::new(Field::new("item", DataType::Utf8View, true)),
        OffsetBuffer::new(vec![0, 1].into()),
        source.clone(),
        None,
    )));
    let dtype = value.data_type();
    let values = [Some(value.clone()), None, Some(value)];
    assert_column_fits(&dtype, &values);
    assert_retained_column_fits(&dtype, &values, 3);
    let output = super::encode_column(
        &dtype,
        values.len(),
        &mut |row| values[row].as_ref(),
        &mut || Ok(()),
    )
    .unwrap();
    let list = output.as_any().downcast_ref::<ListArray>().unwrap();
    let views = list
        .values()
        .as_any()
        .downcast_ref::<StringViewArray>()
        .unwrap();
    assert_eq!(views.data_buffers().len(), 2);
    assert_eq!(ByteView::from(views.views()[0]).buffer_index, 0);
    assert_eq!(ByteView::from(views.views()[1]).buffer_index, 1);
    assert_eq!(
        views.data_buffers()[0].data_ptr(),
        source.data_buffers()[0].data_ptr()
    );
    assert_eq!(
        views.data_buffers()[1].data_ptr(),
        source.data_buffers()[0].data_ptr()
    );
}

#[test]
fn test_normal_list_selects_fixed_and_union_inputs_before_mutable_concat() {
    use datafusion::arrow::{
        array::{Array, ArrayRef, FixedSizeListArray, Float64Array, ListArray, UnionArray},
        buffer::OffsetBuffer,
        datatypes::UnionFields,
    };
    use std::sync::Arc;
    let values: ArrayRef = Arc::new(Float64Array::from(vec![111., 222., 333.]));
    let fixed: ArrayRef = Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("value", DataType::Float64, true)),
        1,
        values.clone(),
        None,
    ));
    let fields = UnionFields::try_new([7], [Field::new("value", DataType::Float64, true)]).unwrap();
    let dense: ArrayRef = Arc::new(
        UnionArray::try_new(
            fields.clone(),
            vec![7, 7, 7].into(),
            Some(vec![0, 1, 2].into()),
            vec![values.clone()],
        )
        .unwrap(),
    );
    let sparse: ArrayRef =
        Arc::new(UnionArray::try_new(fields, vec![7, 7, 7].into(), None, vec![values]).unwrap());
    for child in [fixed, dense, sparse] {
        let list = ScalarValue::List(Arc::new(ListArray::new(
            Arc::new(Field::new("item", child.data_type().clone(), true)),
            OffsetBuffer::new(vec![1, 2].into()),
            child,
            None,
        )));
        let dtype = list.data_type();
        let values = [Some(list.clone()), Some(list)];
        assert_column_fits(&dtype, &values);
        let output =
            super::encode_column(&dtype, 2, &mut |row| values[row].as_ref(), &mut || Ok(()))
                .unwrap();
        let expected = ScalarValue::iter_to_array(values.into_iter().flatten()).unwrap();
        assert_eq!(output.to_data(), expected.to_data());
    }
}

fn normal_fixed_dictionary_values(rows: usize) -> Vec<Option<ScalarValue>> {
    use datafusion::arrow::{
        array::{
            Array, ArrayRef, DictionaryArray, FixedSizeListArray, Float64Array, Int8Array,
            ListArray, StructArray,
        },
        buffer::OffsetBuffer,
        datatypes::Int8Type,
    };
    use std::sync::Arc;
    (0..rows)
        .map(|_| {
            let values: ArrayRef = Arc::new(StructArray::new(
                vec![Field::new("number", DataType::Float64, true)].into(),
                vec![Arc::new(Float64Array::from(vec![None]))],
                None,
            ));
            let dictionary: ArrayRef = Arc::new(
                DictionaryArray::<Int8Type>::try_new(Int8Array::from(vec![0]), values).unwrap(),
            );
            let fixed: ArrayRef = Arc::new(FixedSizeListArray::new(
                Arc::new(Field::new(
                    "dictionary",
                    dictionary.data_type().clone(),
                    true,
                )),
                1,
                dictionary,
                None,
            ));
            Some(ScalarValue::List(Arc::new(ListArray::new(
                Arc::new(Field::new("fixed", fixed.data_type().clone(), true)),
                OffsetBuffer::new(vec![0, 1].into()),
                fixed,
                None,
            ))))
        })
        .collect()
}
#[test]
fn test_normal_fallback_dictionary_overflow_is_controlled_before_pinned_panic() {
    let values = normal_fixed_dictionary_values(129);
    let dtype = values[0].as_ref().unwrap().data_type();
    let schema = Schema::new(vec![Field::new("value", dtype.clone(), true)]);
    assert!(
        plan(
            &schema,
            values.len(),
            &mut |_, row| values[row].as_ref(),
            &mut || Ok(())
        )
        .is_err()
    );
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::encode_column(
            &dtype,
            values.len(),
            &mut |row| values[row].as_ref(),
            &mut || Ok(()),
        )
    }));
    assert!(
        result
            .expect("controlled producer must not invoke pinned overflow panic")
            .is_err()
    );
}
#[test]
fn test_normal_fallback_dictionary_last_representable_offset_is_preserved() {
    let values = normal_fixed_dictionary_values(128);
    let dtype = values[0].as_ref().unwrap().data_type();
    assert_column_fits(&dtype, &values);
}

#[test]
fn test_normal_mutable_boundary_copies_only_selected_variable_view_spans() {
    use datafusion::arrow::{
        array::{Array, FixedSizeListArray, ListArray, ListViewArray, StringArray},
        buffer::OffsetBuffer,
    };
    use std::sync::Arc;
    let strings = Arc::new(StringArray::from(vec![
        "unused".repeat(100_000),
        "x".into(),
        "y".into(),
    ]));
    let views = Arc::new(ListViewArray::new(
        Arc::new(Field::new("text", DataType::Utf8, true)),
        vec![0, 1, 2].into(),
        vec![1, 1, 1].into(),
        strings,
        None,
    ));
    let fixed = Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("view", views.data_type().clone(), true)),
        1,
        views,
        None,
    ));
    let list = ScalarValue::List(Arc::new(ListArray::new(
        Arc::new(Field::new("fixed", fixed.data_type().clone(), true)),
        OffsetBuffer::new(vec![1, 2].into()),
        fixed,
        None,
    )));
    let dtype = list.data_type();
    let values = [Some(list.clone()), None, Some(list)];
    let schema = Schema::new(vec![Field::new("value", dtype.clone(), true)]);
    let planned = plan(
        &schema,
        values.len(),
        &mut |_, row| values[row].as_ref(),
        &mut || Ok(()),
    )
    .unwrap();
    assert!(
        planned.output_bytes < 4096,
        "unselected source values copied: {planned:?}"
    );
    assert_column_fits(&dtype, &values);
    let output = super::encode_column(
        &dtype,
        values.len(),
        &mut |row| values[row].as_ref(),
        &mut || Ok(()),
    )
    .unwrap();
    let expected = ScalarValue::iter_to_array(
        values
            .into_iter()
            .map(|cell| cell.unwrap_or_else(|| ScalarValue::try_from(&dtype).unwrap())),
    )
    .unwrap();
    assert_eq!(output.to_data(), expected.to_data());
}

fn normal_float_dictionary_values(numbers: &[f64]) -> Vec<Option<ScalarValue>> {
    use datafusion::arrow::{
        array::{Array, ArrayRef, DictionaryArray, Float64Array, Int8Array, ListArray},
        buffer::OffsetBuffer,
        datatypes::Int8Type,
    };
    use std::sync::Arc;
    numbers
        .iter()
        .map(|value| {
            let values: ArrayRef = Arc::new(Float64Array::from(vec![*value]));
            let dictionary: ArrayRef = Arc::new(
                DictionaryArray::<Int8Type>::try_new(Int8Array::from(vec![0]), values).unwrap(),
            );
            Some(ScalarValue::List(Arc::new(ListArray::new(
                Arc::new(Field::new(
                    "dictionary",
                    dictionary.data_type().clone(),
                    true,
                )),
                OffsetBuffer::new(vec![0, 1].into()),
                dictionary,
                None,
            ))))
        })
        .collect()
}
#[test]
fn test_normal_dictionary_merges_repeated_values_without_false_key_span_limit() {
    let values = normal_float_dictionary_values(&vec![1.5; 130]);
    let dtype = values[0].as_ref().unwrap().data_type();
    let schema = Schema::new(vec![Field::new("value", dtype.clone(), true)]);
    let mut workspace = None;
    let prefix = allocation_counter::measure(|| {
        workspace = Some(
            super::planning_workspace(
                &schema,
                values.len(),
                &mut |_, row| values[row].as_ref(),
                &mut || Ok(()),
            )
            .unwrap(),
        );
    });
    assert_eq!(
        prefix.count_total, 0,
        "workspace quote allocated before admission"
    );
    assert!(workspace.unwrap() > 0);
    assert_column_fits(&dtype, &values);
    assert_retained_column_fits(&dtype, &values, 3);
    let output = super::encode_column(
        &dtype,
        values.len(),
        &mut |row| values[row].as_ref(),
        &mut || Ok(()),
    )
    .unwrap();
    let expected = ScalarValue::iter_to_array(values.into_iter().flatten()).unwrap();
    assert_eq!(output.to_data(), expected.to_data());
}

fn normal_struct_dictionary_values(rows: usize) -> Vec<Option<ScalarValue>> {
    use datafusion::arrow::{
        array::{FixedSizeListArray, ListArray},
        buffer::OffsetBuffer,
    };
    use std::sync::Arc;
    normal_fixed_dictionary_values(rows)
        .into_iter()
        .map(|value| {
            let Some(ScalarValue::List(list)) = value else {
                unreachable!()
            };
            let fixed = list
                .values()
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap();
            let dictionary = fixed.values().clone();
            Some(ScalarValue::List(Arc::new(ListArray::new(
                Arc::new(Field::new(
                    "dictionary",
                    dictionary.data_type().clone(),
                    true,
                )),
                OffsetBuffer::new(vec![0, 1].into()),
                dictionary,
                None,
            ))))
        })
        .collect()
}
#[test]
fn test_normal_nonprimitive_dictionary_fallback_reports_error_without_panicking() {
    let values = normal_struct_dictionary_values(129);
    let dtype = values[0].as_ref().unwrap().data_type();
    let schema = Schema::new(vec![Field::new("value", dtype.clone(), true)]);
    assert!(
        plan(
            &schema,
            values.len(),
            &mut |_, row| values[row].as_ref(),
            &mut || Ok(())
        )
        .is_err()
    );
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::encode_column(
            &dtype,
            values.len(),
            &mut |row| values[row].as_ref(),
            &mut || Ok(()),
        )
    }));
    assert!(
        result
            .expect("normal dictionary fallback must not panic")
            .is_err()
    );
}

#[test]
fn test_normal_dictionary_emitted_key_boundary_preserves_128_and_rejects_129() {
    let numbers = (0_u32..129).map(f64::from).collect::<Vec<_>>();
    let values = normal_float_dictionary_values(&numbers);
    let dtype = values[0].as_ref().unwrap().data_type();
    assert_column_fits(&dtype, &values[..128]);
    let schema = Schema::new(vec![Field::new("value", dtype, true)]);
    let error = plan(
        &schema,
        values.len(),
        &mut |_, row| values[row].as_ref(),
        &mut || Ok(()),
    )
    .unwrap_err();
    assert!(
        matches!(error, crate::CalcFlowError::Format {message} if message.contains("normal merged dictionary key"))
    );
}

fn normal_dictionary_list(values: datafusion::arrow::array::ArrayRef) -> ScalarValue {
    use datafusion::arrow::{
        array::{Array, DictionaryArray, Int8Array, ListArray},
        buffer::OffsetBuffer,
        datatypes::Int8Type,
    };
    use std::sync::Arc;
    let dictionary =
        Arc::new(DictionaryArray::<Int8Type>::try_new(Int8Array::from(vec![0]), values).unwrap());
    ScalarValue::List(Arc::new(ListArray::new(
        Arc::new(Field::new(
            "dictionary",
            dictionary.data_type().clone(),
            true,
        )),
        OffsetBuffer::new(vec![0, 1].into()),
        dictionary,
        None,
    )))
}
#[test]
fn test_normal_dictionary_nullable_byte_slot_keeps_pinned_raw_span() {
    use datafusion::arrow::{
        array::{Array, DictionaryArray, ListArray, StringArray},
        buffer::{Buffer, NullBuffer, OffsetBuffer},
        datatypes::Int8Type,
    };
    use std::sync::Arc;
    let payload = "hidden-null-bytes".repeat(40_000);
    let values = [payload.clone(), String::from("different-null-body")]
        .into_iter()
        .map(|text| {
            let len = i32::try_from(text.len()).unwrap();
            Some(normal_dictionary_list(Arc::new(StringArray::new(
                OffsetBuffer::new(vec![0, len].into()),
                Buffer::from_vec(text.into_bytes()),
                Some(NullBuffer::new_null(1)),
            ))))
        })
        .collect::<Vec<_>>();
    let dtype = values[0].as_ref().unwrap().data_type();
    assert_column_fits(&dtype, &values);
    assert_retained_column_fits(&dtype, &values, 5);
    let schema = Schema::new(vec![Field::new("value", dtype.clone(), true)]);
    let planned = plan(&schema, 2, &mut |_, row| values[row].as_ref(), &mut || {
        Ok(())
    })
    .unwrap();
    assert!(planned.output_bytes >= u64::try_from(payload.len()).unwrap());
    let output =
        super::encode_column(&dtype, 2, &mut |row| values[row].as_ref(), &mut || Ok(())).unwrap();
    let lists = output.as_any().downcast_ref::<ListArray>().unwrap();
    let dictionary = lists
        .values()
        .as_any()
        .downcast_ref::<DictionaryArray<Int8Type>>()
        .unwrap();
    let bytes = dictionary
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(bytes.len(), 1);
    assert!(bytes.is_null(0));
    assert_eq!(bytes.values().as_slice(), payload.as_bytes());
}
#[test]
fn test_normal_dictionary_raw_null_presence_matches_pinned_byte_vs_primitive_routing() {
    use datafusion::arrow::{
        array::{Array, ArrayRef, DictionaryArray, Float64Array, ListArray, StringArray},
        buffer::{NullBuffer, ScalarBuffer},
        datatypes::Int8Type,
    };
    use std::sync::Arc;
    let source = StringArray::from(vec!["same bytes"]);
    let masked = StringArray::new(
        source.offsets().clone(),
        source.values().clone(),
        Some(NullBuffer::new_valid(1)),
    );
    assert!(masked.nulls().is_some());
    let values = [
        Some(normal_dictionary_list(Arc::new(source))),
        Some(normal_dictionary_list(Arc::new(masked))),
    ];
    let dtype = values[0].as_ref().unwrap().data_type();
    assert_column_fits(&dtype, &values);
    let output =
        super::encode_column(&dtype, 2, &mut |row| values[row].as_ref(), &mut || Ok(())).unwrap();
    let list = output.as_any().downcast_ref::<ListArray>().unwrap();
    let dictionary = list
        .values()
        .as_any()
        .downcast_ref::<DictionaryArray<Int8Type>>()
        .unwrap();
    assert_eq!(dictionary.values().len(), 1);
    let buffer = ScalarBuffer::from(vec![2.0]);
    let first: ArrayRef = Arc::new(Float64Array::new(buffer.clone(), None));
    let second: ArrayRef = Arc::new(Float64Array::new(
        buffer.clone(),
        Some(NullBuffer::new_valid(1)),
    ));
    let values = [
        Some(normal_dictionary_list(first)),
        Some(normal_dictionary_list(second)),
    ];
    let dtype = values[0].as_ref().unwrap().data_type();
    assert_column_fits(&dtype, &values);
    let output =
        super::encode_column(&dtype, 2, &mut |row| values[row].as_ref(), &mut || Ok(())).unwrap();
    let list = output.as_any().downcast_ref::<ListArray>().unwrap();
    let dictionary = list
        .values()
        .as_any()
        .downcast_ref::<DictionaryArray<Int8Type>>()
        .unwrap();
    let floats = dictionary
        .values()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(
        floats.values().inner().data_ptr(),
        buffer.inner().data_ptr()
    );
}

#[test]
fn test_normal_dictionary_float_interning_keeps_signed_zero_and_nan_payload_bits() {
    use datafusion::arrow::{
        array::{Array, DictionaryArray, Float64Array, ListArray},
        datatypes::Int8Type,
    };
    let numbers = [
        0.0,
        -0.0,
        f64::from_bits(0x7ff8_0000_0000_0001),
        f64::from_bits(0x7ff8_0000_0000_0002),
        f64::from_bits(0x7ff8_0000_0000_0001),
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let values = normal_float_dictionary_values(&numbers);
    let dtype = values[0].as_ref().unwrap().data_type();
    assert_column_fits(&dtype, &values);
    let output = super::encode_column(
        &dtype,
        values.len(),
        &mut |row| values[row].as_ref(),
        &mut || Ok(()),
    )
    .unwrap();
    let expected = ScalarValue::iter_to_array(values.into_iter().flatten()).unwrap();
    let extract = |array: &dyn Array| {
        let list = array.as_any().downcast_ref::<ListArray>().unwrap();
        let dictionary = list
            .values()
            .as_any()
            .downcast_ref::<DictionaryArray<Int8Type>>()
            .unwrap();
        let floats = dictionary
            .values()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        (
            dictionary.keys().values().to_vec(),
            floats
                .values()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        )
    };
    assert_eq!(extract(output.as_ref()), extract(expected.as_ref()));
}

#[test]
fn test_normal_dictionary_empty_and_virtual_null_byte_values_are_funded() {
    use datafusion::arrow::{
        array::{ArrayRef, DictionaryArray, Int8Array, ListArray, StringArray},
        buffer::OffsetBuffer,
        datatypes::Int8Type,
    };
    use std::sync::Arc;
    let value: ArrayRef = Arc::new(StringArray::from(Vec::<&str>::new()));
    let dictionary: ArrayRef =
        Arc::new(DictionaryArray::<Int8Type>::try_new(Int8Array::from(vec![None]), value).unwrap());
    let scalar = ScalarValue::List(Arc::new(ListArray::new(
        Arc::new(Field::new(
            "dictionary",
            dictionary.data_type().clone(),
            true,
        )),
        OffsetBuffer::new(vec![0, 1].into()),
        dictionary,
        None,
    )));
    let dtype = scalar.data_type();
    assert_column_fits(&dtype, &[None, None]);
    assert_column_fits(&dtype, &[Some(scalar.clone()), None, Some(scalar)]);
}

#[test]
fn test_normal_dictionary_workspace_rejects_invalid_root_shape_without_panicking() {
    use datafusion::arrow::array::StructArray;
    use std::sync::Arc;
    let template = normal_float_dictionary_values(&[1.0])
        .pop()
        .unwrap()
        .unwrap();
    let ScalarValue::List(list) = template else {
        unreachable!()
    };
    let empty = ScalarValue::List(Arc::new(list.slice(0, 0)));
    let dictionary = list.values().data_type().clone();
    let expected = DataType::Struct(vec![Field::new("dictionary", dictionary, true)].into());
    let wrong = ScalarValue::Struct(Arc::new(
        StructArray::try_new_with_length(Vec::<Field>::new().into(), vec![], None, 1).unwrap(),
    ));
    for (dtype, value) in [(empty.data_type(), empty), (expected, wrong)] {
        let schema = Schema::new(vec![Field::new("value", dtype, true)]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            super::planning_workspace(&schema, 2, &mut |_, _| Some(&value), &mut || Ok(()))
        }));
        assert!(
            result
                .expect("invalid borrowed root must be a controlled error")
                .is_err()
        );
    }
}

fn scalar_run_dictionary(keys: Vec<i8>, numbers: Vec<f64>) -> ScalarValue {
    use datafusion::arrow::{
        array::{Array, DictionaryArray, Float64Array, Int8Array, ListArray},
        buffer::OffsetBuffer,
        datatypes::Int8Type,
    };
    use std::sync::Arc;
    let count = i32::try_from(keys.len()).unwrap();
    let dictionary = Arc::new(
        DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(keys),
            Arc::new(Float64Array::from(numbers)),
        )
        .unwrap(),
    );
    let list = ScalarValue::List(Arc::new(ListArray::new(
        Arc::new(Field::new(
            "dictionary",
            dictionary.data_type().clone(),
            true,
        )),
        OffsetBuffer::new(vec![0, count].into()),
        dictionary,
        None,
    )));
    ScalarValue::RunEndEncoded(
        Arc::new(Field::new("ends", DataType::Int32, false)),
        Arc::new(Field::new("value", list.data_type(), true)),
        Box::new(list),
    )
}
#[test]
fn test_scalar_run_dictionary_workspace_covers_merge_after_coalescing() {
    let first = scalar_run_dictionary(vec![0, 0], vec![1.0]);
    let second = scalar_run_dictionary(vec![0], vec![2.0; 20]);
    let mut values = vec![Some(first); 100];
    values.push(Some(second));
    let dtype = values[0].as_ref().unwrap().data_type();
    let mut extra = 0;
    let prefix = allocation_counter::measure(|| {
        extra = super::normal::dictionary::planning_workspace(
            &dtype,
            values.len(),
            &mut |row| values[row].as_ref(),
            &mut || Ok(()),
        )
        .unwrap();
    });
    assert_eq!(prefix.count_total, 0);
    let arrays = [
        dictionary_in_scalar_run(values[0].as_ref().unwrap()),
        dictionary_in_scalar_run(values[100].as_ref().unwrap()),
    ];
    let dictionary_only = allocation_counter::measure(|| {
        let mut at = |index: usize| {
            Ok(super::concatenated::Piece {
                array: Some(arrays[index]),
                start: 0,
                len: arrays[index].len(),
            })
        };
        std::hint::black_box(
            super::normal::dictionary::plan(arrays[0].data_type(), 2, 3, &mut at, &mut || Ok(()))
                .unwrap(),
        );
    });
    assert!(
        dictionary_only.bytes_max <= extra,
        "dictionary allocation {dictionary_only:?} > its own P {extra}"
    );
    assert_column_fits(&dtype, &values);
}
fn dictionary_in_scalar_run(value: &ScalarValue) -> &dyn datafusion::arrow::array::Array {
    let ScalarValue::RunEndEncoded(_, _, value) = value else {
        unreachable!()
    };
    let ScalarValue::List(value) = value.as_ref() else {
        unreachable!()
    };
    value.values().as_ref()
}

#[test]
fn test_small_scalar_views_retain_the_pinned_first_slab_capacity() {
    let text = ScalarValue::Utf8View(Some("x".repeat(32)));
    let bytes = ScalarValue::BinaryView(Some(vec![7; 32]));
    for value in [text, bytes] {
        let dtype = value.data_type();
        let values = [Some(value)];
        assert_retained_column_fits(&dtype, &values, 2);
        assert_column_fits(&dtype, &values);
    }
}

#[test]
fn test_scalar_view_slab_growth_is_independent_of_large_literal_capacity() {
    let values = [
        Some(ScalarValue::Utf8View(Some("x".repeat(20_000)))),
        Some(ScalarValue::Utf8View(Some("y".repeat(32)))),
    ];
    assert_retained_column_fits(&DataType::Utf8View, &values, 3);
    assert_column_fits(&DataType::Utf8View, &values);
}

#[test]
fn test_single_constructor_entry_preserves_existing_array_owner_without_allocation() {
    use datafusion::arrow::{
        array::{ArrayRef, Float64Array, ListArray},
        buffer::OffsetBuffer,
    };
    use std::sync::Arc;
    let source = Arc::new(ListArray::new(
        Arc::new(Field::new("value", DataType::Float64, true)),
        OffsetBuffer::new(vec![0, 1].into()),
        Arc::new(Float64Array::from(vec![1.0])),
        None,
    ));
    let value = ScalarValue::List(source.clone());
    let mut output = None;
    let allocation = allocation_counter::measure(|| {
        output = Some(super::producer::single(&value, &mut || Ok(())).unwrap());
    });
    let source: ArrayRef = source;
    assert!(Arc::ptr_eq(&output.unwrap(), &source));
    assert_eq!(allocation.count_total, 0);
}

#[test]
fn test_scalar_union_columns_quote_fresh_children_and_typed_null_holes() {
    use datafusion::arrow::datatypes::{UnionFields, UnionMode};
    let fields = UnionFields::try_new(
        [1, 127],
        [
            Field::new("number", DataType::Float64, true),
            Field::new("text", DataType::Utf8View, true),
        ],
    )
    .unwrap();
    for mode in [UnionMode::Sparse, UnionMode::Dense] {
        let values = [
            Some(ScalarValue::Union(
                Some((1, Box::new(ScalarValue::Float64(Some(1.0))))),
                fields.clone(),
                mode,
            )),
            Some(ScalarValue::Union(
                Some((127, Box::new(ScalarValue::Utf8View(Some("x".repeat(32)))))),
                fields.clone(),
                mode,
            )),
            Some(ScalarValue::Union(None, fields.clone(), mode)),
            None,
        ];
        assert_column_fits(&DataType::Union(fields.clone(), mode), &values);
        assert_union_owned(&DataType::Union(fields.clone(), mode), &values);
    }
}

fn scalar_union(value: ScalarValue, mode: datafusion::arrow::datatypes::UnionMode) -> ScalarValue {
    let fields = datafusion::arrow::datatypes::UnionFields::try_new(
        [127],
        [Field::new("component", value.data_type(), true)],
    )
    .unwrap();
    ScalarValue::Union(Some((127, Box::new(value))), fields, mode)
}
fn assert_union_owned(dtype: &DataType, values: &[Option<ScalarValue>]) {
    use super::super::footprint::Footprint;
    let schema = Schema::new(vec![Field::new("value", dtype.clone(), true)]);
    let cost = plan(
        &schema,
        values.len(),
        &mut |_, row| values[row].as_ref(),
        &mut || Ok(()),
    )
    .unwrap();
    let mut source = Footprint::default();
    for value in values.iter().flatten() {
        source.scalar(value, "union-output").unwrap();
    }
    let array = super::encode_column(
        dtype,
        values.len(),
        &mut |row| values[row].as_ref(),
        &mut || Ok(()),
    )
    .unwrap();
    assert_eq!(array.len(), values.len());
    assert_eq!(array.data_type(), dtype);
    let mut output = Footprint::default();
    output
        .array_checked(&array, "union-output", &mut || Ok(()))
        .unwrap();
    let new = output.incremental_bytes(&source, "union-output").unwrap();
    assert!(new <= cost.output_bytes, "new O {new} > plan {cost:?}");
}
#[test]
fn test_scalar_union_dictionary_null_and_zero_width_values_keep_pointer_identity() {
    use datafusion::arrow::datatypes::UnionMode;
    for value in [
        ScalarValue::Null,
        ScalarValue::FixedSizeBinary(0, Some(Vec::new())),
    ] {
        let value = scalar_union(
            ScalarValue::Dictionary(Box::new(DataType::Int8), Box::new(value)),
            UnionMode::Dense,
        );
        let dtype = value.data_type();
        let values = vec![Some(value); 129];
        assert_column_fits(&dtype, &values);
        assert_union_owned(&dtype, &values);
    }
}
#[test]
fn test_scalar_union_dictionary_constructor_spans_match_the_pinned_fallback_limit() {
    use datafusion::arrow::datatypes::UnionMode;
    let value = scalar_union(
        ScalarValue::Dictionary(
            Box::new(DataType::Int8),
            Box::new(ScalarValue::Float64(None)),
        ),
        UnionMode::Dense,
    );
    let dtype = value.data_type();
    let values = vec![Some(value); 128];
    assert_column_fits(&dtype, &values);
    assert_union_owned(&dtype, &values);
    let schema = Schema::new(vec![Field::new("value", dtype.clone(), true)]);
    assert!(plan(&schema, 129, &mut |_, _| values[0].as_ref(), &mut || Ok(())).is_err());
    assert!(
        super::encode_column(&dtype, 129, &mut |_| values[0].as_ref(), &mut || Ok(())).is_err()
    );
}
#[test]
fn test_scalar_union_run_and_nested_array_children_keep_exact_shape_and_funded_backing() {
    use datafusion::arrow::{
        array::{Array, Float64Array, StructArray},
        datatypes::UnionMode,
    };
    use std::sync::Arc;
    let child = Arc::new(StructArray::new(
        vec![Arc::new(Field::new("x", DataType::Float64, false))].into(),
        vec![Arc::new(Float64Array::from(vec![2.0]))],
        None,
    ));
    let run = ScalarValue::RunEndEncoded(
        Arc::new(Field::new("ends", DataType::Int16, false)),
        Arc::new(Field::new("values", child.data_type().clone(), true)),
        Box::new(ScalarValue::Struct(child)),
    );
    for mode in [UnionMode::Dense, UnionMode::Sparse] {
        let value = scalar_union(run.clone(), mode);
        let dtype = value.data_type();
        for values in [
            vec![Some(value.clone())],
            vec![Some(value.clone()), None, Some(value.clone())],
        ] {
            assert_column_fits(&dtype, &values);
            assert_union_owned(&dtype, &values);
        }
    }
}

#[test]
fn test_scalar_union_output_matches_pinned_values_and_field_metadata() {
    use datafusion::arrow::{
        array::{Array, UnionArray},
        datatypes::{UnionFields, UnionMode},
    };
    let fields = UnionFields::try_new(
        [1, 127],
        [
            Field::new("price", DataType::Float64, true).with_metadata(
                std::collections::HashMap::from([("unit".into(), "USD".into())]),
            ),
            Field::new("label", DataType::Utf8View, true),
        ],
    )
    .unwrap();
    for mode in [UnionMode::Dense, UnionMode::Sparse] {
        let values = [
            ScalarValue::Union(
                Some((1, Box::new(ScalarValue::Float64(Some(8.0))))),
                fields.clone(),
                mode,
            ),
            ScalarValue::Union(
                Some((
                    127,
                    Box::new(ScalarValue::Utf8View(Some("description".repeat(4)))),
                )),
                fields.clone(),
                mode,
            ),
        ];
        let dtype = DataType::Union(fields.clone(), mode);
        let actual = super::encode_column(
            &dtype,
            values.len(),
            &mut |row| Some(&values[row]),
            &mut || Ok(()),
        )
        .unwrap();
        let expected = ScalarValue::iter_to_array(values).unwrap();
        assert_eq!(actual.data_type(), &dtype);
        assert_eq!(actual.to_data(), expected.to_data());
        assert_eq!(
            actual
                .as_any()
                .downcast_ref::<UnionArray>()
                .unwrap()
                .type_id(1),
            127
        );
    }
}
#[test]
fn test_scalar_union_invalid_active_types_and_cancellation_fail_before_returning_output() {
    use datafusion::arrow::datatypes::{UnionFields, UnionMode};
    let fields = UnionFields::try_new([127], [Field::new("x", DataType::Float64, true)]).unwrap();
    let dtype = DataType::Union(fields.clone(), UnionMode::Dense);
    let schema = Schema::new(vec![Field::new("u", dtype.clone(), true)]);
    for value in [
        ScalarValue::Union(
            Some((1, Box::new(ScalarValue::Float64(Some(1.0))))),
            fields.clone(),
            UnionMode::Dense,
        ),
        ScalarValue::Union(
            Some((127, Box::new(ScalarValue::Int32(Some(1))))),
            fields.clone(),
            UnionMode::Dense,
        ),
    ] {
        assert!(plan(&schema, 1, &mut |_, _| Some(&value), &mut || Ok(())).is_err());
        assert!(super::encode_column(&dtype, 1, &mut |_| Some(&value), &mut || Ok(())).is_err());
    }
    let value = ScalarValue::Union(
        Some((127, Box::new(ScalarValue::Float64(Some(1.0))))),
        fields,
        UnionMode::Dense,
    );
    let mut visits = 0;
    let error = plan(&schema, 2, &mut |_, _| Some(&value), &mut || {
        visits += 1;
        if visits == 8 {
            return Err(super::invalid("cancel sentinel"));
        }
        Ok(())
    })
    .unwrap_err();
    assert!(error.to_string().contains("cancel sentinel"));
}

#[test]
fn test_scalar_union_columns_inside_dictionary_and_run_reuse_the_same_plan() {
    use datafusion::arrow::datatypes::UnionMode;
    use std::sync::Arc;
    let union = scalar_union(
        ScalarValue::Utf8View(Some("payload".repeat(5))),
        UnionMode::Dense,
    );
    let values = [
        ScalarValue::Dictionary(Box::new(DataType::Int8), Box::new(union.clone())),
        ScalarValue::RunEndEncoded(
            Arc::new(Field::new("ends", DataType::Int16, false)),
            Arc::new(Field::new("value", union.data_type(), true)),
            Box::new(union),
        ),
    ];
    for value in values {
        let dtype = value.data_type();
        let rows = [Some(value.clone()), None, Some(value)];
        assert_column_fits(&dtype, &rows);
        assert_union_owned(&dtype, &rows);
    }
}

#[test]
fn test_scalar_union_inactive_dictionary_children_prequote_empty_value_metadata() {
    use datafusion::arrow::datatypes::{UnionFields, UnionMode};
    let fields = UnionFields::try_new(
        [1, 127],
        [
            Field::new(
                "dictionary",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Float64)),
                true,
            ),
            Field::new("value", DataType::Float64, true),
        ],
    )
    .unwrap();
    let dtype = DataType::Union(fields.clone(), UnionMode::Dense);
    let inactive = ScalarValue::Union(
        Some((127, Box::new(ScalarValue::Float64(Some(1.0))))),
        fields.clone(),
        UnionMode::Dense,
    );
    let null = ScalarValue::Union(None, fields, UnionMode::Dense);
    for value in [inactive, null] {
        let values = [Some(value.clone()), Some(value)];
        assert_column_fits(&dtype, &values);
        assert_union_owned(&dtype, &values);
    }
}

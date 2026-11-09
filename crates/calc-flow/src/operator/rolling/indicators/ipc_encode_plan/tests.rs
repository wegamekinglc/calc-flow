use std::sync::Arc;

use datafusion::arrow::{
    array::{Array, Float64Array},
    datatypes::{DataType, Field, Schema},
    ipc::writer::FileWriter,
    record_batch::RecordBatch,
};

#[test]
fn actual_default_writer_fits_the_preflight_allocation_plan() {
    let schema = Schema::new(vec![Field::new("value", DataType::Float64, true)]);
    let record = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Float64Array::from(vec![
            Some(1.0),
            None,
            Some(3.0),
        ]))],
    )
    .unwrap();
    let plan = super::plan_fresh(&schema, &record, &mut || Ok(())).unwrap();
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, &schema).unwrap();
    writer.write(&record).unwrap();
    writer.finish().unwrap();
    drop(writer);
    assert!(u64::try_from(bytes.capacity()).unwrap() <= plan.encoded_quota);
    assert!(u64::try_from(bytes.len()).unwrap() <= plan.encoded_quota);
    assert!(plan.scratch_bytes > 0);
}

#[test]
fn cancelled_planning_preserves_the_original_error() {
    let schema = Schema::empty();
    let batch = RecordBatch::new_empty(Arc::new(schema.clone()));
    let error = super::plan_fresh(&schema, &batch, &mut || {
        Err(crate::CalcFlowError::Format {
            message: "cancel sentinel".into(),
        })
    })
    .unwrap_err();
    assert!(error.to_string().contains("cancel sentinel"));
}

fn assert_writer_fits(record: &RecordBatch) -> u64 {
    let schema = record.schema();
    let plan = super::plan_fresh(&schema, record, &mut || Ok(())).unwrap();
    let workspace = super::workspace_fresh(&schema, record, &mut || Ok(())).unwrap();
    let scan = allocation_counter::measure(|| {
        assert_eq!(
            super::workspace_fresh(&schema, record, &mut || Ok(())).unwrap(),
            workspace
        );
    });
    assert_eq!(scan.count_total, 0);
    for column in record.columns() {
        let observed = allocation_counter::measure(|| {
            std::hint::black_box(column.to_data());
        });
        assert!(
            observed.bytes_max <= workspace,
            "planner {} > {workspace}",
            observed.bytes_max
        );
    }
    let observed = allocation_counter::measure(|| {
        let mut writer = FileWriter::try_new(std::io::sink(), &schema).unwrap();
        writer.write(record).unwrap();
        writer.finish().unwrap();
    });
    assert!(
        observed.bytes_max <= plan.scratch_bytes,
        "writer {} > {}",
        observed.bytes_max,
        plan.scratch_bytes
    );
    assert_eq!(observed.bytes_current, 0);
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, &schema).unwrap();
    writer.write(record).unwrap();
    writer.finish().unwrap();
    drop(writer);
    assert!(u64::try_from(bytes.capacity()).unwrap() <= plan.encoded_quota);
    assert!(u64::try_from(bytes.len()).unwrap() <= plan.encoded_quota);
    observed.bytes_max
}

#[test]
fn dictionary_values_are_counted_as_separate_bodies() {
    use datafusion::arrow::{
        array::{DictionaryArray, StringArray},
        datatypes::Int32Type,
    };
    let values = StringArray::from_iter_values(
        (0..512).map(|index| format!("{index:04}{}", "x".repeat(4096))),
    );
    let array = DictionaryArray::<Int32Type>::new(vec![0, 1, 2].into(), Arc::new(values));
    let schema = Arc::new(Schema::new(vec![Field::new(
        "dict",
        array.data_type().clone(),
        false,
    )]));
    let record = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
    assert_writer_fits(&record);
}

#[test]
fn nested_variable_data_and_large_schema_metadata_are_counted() {
    use datafusion::arrow::array::{Array, ListArray, StringArray};
    use datafusion::arrow::buffer::OffsetBuffer;
    let values = Arc::new(StringArray::from(vec!["x".repeat(65_536)]));
    let field = Arc::new(Field::new("item", DataType::Utf8, true));
    let array = ListArray::new(
        field.clone(),
        OffsetBuffer::new(vec![0_i32, 1].into()),
        values,
        None,
    );
    let metadata = [("key".to_string(), "m".repeat(65_536))].into();
    let schema = Arc::new(Schema::new_with_metadata(
        vec![Field::new("list", array.data_type().clone(), true)],
        metadata,
    ));
    let record = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
    assert_writer_fits(&record);
}

#[test]
fn descriptor_workspace_is_admitted_without_charging_source_payloads() {
    use datafusion::arrow::array::StringArray;
    let array = StringArray::from(vec!["x".repeat(1_000_000)]);
    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
    let record = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let workspace = super::workspace_fresh(&schema, &record, &mut || Ok(())).unwrap();
    assert!(workspace > 0);
    assert!(workspace < 4096);
}

fn empty_types() -> Vec<DataType> {
    use DataType::*;
    use datafusion::arrow::datatypes::{IntervalUnit as I, TimeUnit as T, UnionFields, UnionMode};
    let child = Arc::new(Field::new("child", Int32, true));
    let entries = Arc::new(Field::new(
        "entries",
        Struct(
            vec![
                Field::new("key", Utf8, false),
                Field::new("value", Int32, true),
            ]
            .into(),
        ),
        false,
    ));
    let mut types = vec![
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
        Interval(I::YearMonth),
        Interval(I::DayTime),
        Interval(I::MonthDayNano),
        Binary,
        LargeBinary,
        BinaryView,
        Utf8,
        LargeUtf8,
        Utf8View,
        FixedSizeBinary(7),
        List(child.clone()),
        LargeList(child.clone()),
        ListView(child.clone()),
        LargeListView(child.clone()),
        FixedSizeList(child.clone(), 3),
        Struct(vec![child.clone()].into()),
        Map(entries.clone(), false),
        Map(entries, true),
        RunEndEncoded(Arc::new(Field::new("run_ends", Int32, false)), child),
        Decimal32(8, -2),
        Decimal64(17, 3),
        Decimal128(28, 4),
        Decimal256(60, -7),
    ];
    for unit in [T::Second, T::Millisecond, T::Microsecond, T::Nanosecond] {
        types.push(Timestamp(unit, None));
        types.push(Timestamp(unit, Some("UTC".into())));
        types.push(Duration(unit));
    }
    for mode in [UnionMode::Sparse, UnionMode::Dense] {
        types.push(Union(
            UnionFields::try_new(
                [3, 7],
                [Field::new("a", Float32, true), Field::new("b", Utf8, true)],
            )
            .unwrap(),
            mode,
        ));
    }
    for key in [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64] {
        types.push(Dictionary(Box::new(key), Box::new(Utf8)));
    }
    types
}
#[test]
fn all_pinned_datatype_families_fit_for_empty_records() {
    use datafusion::arrow::array::new_empty_array;
    for data_type in empty_types() {
        let array = new_empty_array(&data_type);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            data_type.clone(),
            true,
        )]));
        let record = RecordBatch::try_new(schema, vec![array]).unwrap();
        assert_writer_fits(&record);
    }
}

#[test]
fn empty_view_buffers_still_have_a_real_descriptor_cost() {
    use datafusion::arrow::{
        array::StringViewArray,
        buffer::{Buffer, ScalarBuffer},
    };
    let buffers = vec![Buffer::from_slice_ref(&[] as &[u8]); 1024];
    let array =
        StringViewArray::try_new(ScalarBuffer::from(Vec::<u128>::new()), buffers, None).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "view",
        DataType::Utf8View,
        true,
    )]));
    let record = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let plan = super::plan_fresh(&schema, &record, &mut || Ok(())).unwrap();
    assert!(plan.scratch_bytes > 1024 * u64::try_from(size_of::<Buffer>()).unwrap());
    assert_writer_fits(&record);
}

#[test]
fn checked_arithmetic_rejects_overflow_before_any_writer_exists() {
    assert!(super::math::aligned(u64::MAX).is_err());
    assert!(super::math::vector(u64::MAX, 8).is_err());
    assert!(super::math::growing(u64::MAX, 16).is_err());
    assert!(super::math::hash_bytes(usize::MAX, 16).is_err());
}

#[test]
fn unsupported_direct_dictionary_values_fail_before_encoding() {
    use datafusion::arrow::array::new_empty_array;
    let data_type = DataType::Dictionary(
        Box::new(DataType::Int8),
        Box::new(DataType::Dictionary(
            Box::new(DataType::Int8),
            Box::new(DataType::Utf8),
        )),
    );
    let record = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "dict",
            data_type.clone(),
            true,
        )])),
        vec![new_empty_array(&data_type)],
    )
    .unwrap();
    assert!(FileWriter::try_new(std::io::sink(), &record.schema()).is_err());
    assert!(super::plan_fresh(&record.schema(), &record, &mut || Ok(())).is_err());
}

#[test]
fn nonzero_bitmap_and_variable_offsets_fit_without_copying_source_backing() {
    use datafusion::arrow::array::{ArrayRef, BooleanArray, StringArray, StringViewArray};
    let booleans = BooleanArray::from(vec![Some(true), None, Some(false), Some(true), None]);
    let strings = StringArray::from(vec![
        Some("prefix".repeat(1000)),
        None,
        Some("middle".repeat(1000)),
        Some("last".to_string()),
        None,
    ]);
    let views = StringViewArray::from(vec![
        Some("prefix".repeat(1000)),
        None,
        Some("middle".repeat(1000)),
        Some("last".to_string()),
        None,
    ]);
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(booleans.slice(1, 3)),
        Arc::new(strings.slice(1, 3)),
        Arc::new(views.slice(1, 3)),
    ];
    let fields: Vec<_> = arrays
        .iter()
        .enumerate()
        .map(|(i, array)| Field::new(format!("f{i}"), array.data_type().clone(), true))
        .collect();
    let record = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
    assert_writer_fits(&record);
}

#[test]
fn sliced_run_encoded_values_fit_the_pinned_normalization_path() {
    use datafusion::arrow::{
        array::{Int32Array, RunArray, StringArray},
        datatypes::Int32Type,
    };
    let ends = Int32Array::from(vec![2, 5, 7]);
    let values = StringArray::from(vec![
        Some("a".to_string()),
        None,
        Some("large".repeat(1000)),
    ]);
    let array = RunArray::<Int32Type>::try_new(&ends, &values)
        .unwrap()
        .slice(1, 4);
    let schema = Arc::new(Schema::new(vec![Field::new(
        "run",
        array.data_type().clone(),
        true,
    )]));
    let record = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
    assert_writer_fits(&record);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn logical_counts_must_fit_the_ipc_signed_metadata_fields() {
    use datafusion::arrow::array::NullArray;
    let schema = Arc::new(Schema::new(vec![Field::new("null", DataType::Null, true)]));
    let record = RecordBatch::try_new(schema, vec![Arc::new(NullArray::new(usize::MAX))]).unwrap();
    assert!(super::plan_fresh(&record.schema(), &record, &mut || Ok(())).is_err());
}

#[test]
fn fresh_single_entry_hash_table_uses_its_actual_minimum_bucket_count() {
    use std::collections::HashMap;
    let observed = allocation_counter::measure(|| {
        std::hint::black_box(HashMap::<String, String>::with_capacity(1));
    });
    let bound = super::math::hash_bytes(1, size_of::<(String, String)>()).unwrap();
    assert!(
        observed.bytes_max <= bound,
        "{} > {bound}",
        observed.bytes_max
    );
}

fn tombstoned_metadata() -> std::collections::HashMap<String, String> {
    use std::{collections::HashMap, hash::BuildHasher};
    let mut map = HashMap::<String, String>::with_capacity(896);
    let mask = (map.capacity() + 1).next_power_of_two() - 1;
    let mut inserted = Vec::new();
    for ordinal in 0..5_000_000_u64 {
        let key = format!("key{ordinal}");
        if map.hasher().hash_one(&key) & u64::try_from(mask).unwrap() != 0 {
            continue;
        }
        map.insert(key.clone(), "value".into());
        inserted.push(key);
        if inserted.len() == 896 {
            break;
        }
    }
    assert_eq!(inserted.len(), 896);
    for key in &inserted[..895] {
        map.remove(key);
    }
    map
}

#[test]
fn nested_field_metadata_maps_remain_shared_instead_of_deep_cloned() {
    use datafusion::arrow::array::StructArray;
    let field = Arc::new(
        Field::new("child", DataType::Float64, false).with_metadata(tombstoned_metadata()),
    );
    let array = StructArray::from(vec![(
        field,
        Arc::new(Float64Array::from(vec![1.0])) as datafusion::arrow::array::ArrayRef,
    )]);
    let schema = Arc::new(Schema::new(vec![Field::new(
        "parent",
        array.data_type().clone(),
        false,
    )]));
    let record = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
    assert_writer_fits(&record);
}

fn wide_empty_record() -> RecordBatch {
    use datafusion::arrow::array::new_empty_array;
    let children: Vec<Field> = (0..10)
        .map(|i| Field::new(format!("child{i}"), DataType::Float64, true))
        .collect();
    let data_type = DataType::Struct(children.into());
    let schema = Arc::new(Schema::new(vec![Field::new(
        "root",
        data_type.clone(),
        true,
    )]));
    RecordBatch::try_new(schema, vec![new_empty_array(&data_type)]).unwrap()
}

#[test]
fn deep_prescan_cancellation_never_copies_descriptor_metadata() {
    let record = wide_empty_record();
    let mut error = Some(crate::CalcFlowError::Format {
        message: "deep cancel sentinel".into(),
    });
    let mut calls = 0;
    let mut result = None;
    let observed = allocation_counter::measure(|| {
        result = Some(super::workspace_fresh(
            &record.schema(),
            &record,
            &mut || {
                calls += 1;
                if calls == 8 {
                    Err(error.take().unwrap())
                } else {
                    Ok(())
                }
            },
        ));
    });
    assert_eq!(calls, 8);
    assert_eq!(observed.count_total, 0);
    assert!(
        result
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("deep cancel sentinel")
    );
}

#[test]
fn cancelled_plan_keeps_the_admitted_metadata_peak_and_releases_every_copy() {
    let record = wide_empty_record();
    let schema = record.schema();
    let workspace = super::workspace_fresh(&schema, &record, &mut || Ok(())).unwrap();
    let mut total = 0;
    super::plan_fresh(&schema, &record, &mut || {
        total += 1;
        Ok(())
    })
    .unwrap();
    let mut calls = 0;
    let mut error = Some(crate::CalcFlowError::Format {
        message: "late cancel sentinel".into(),
    });
    let mut result = None;
    let observed = allocation_counter::measure(|| {
        result = Some(super::plan_fresh(&schema, &record, &mut || {
            calls += 1;
            if calls == total {
                Err(error.take().unwrap())
            } else {
                Ok(())
            }
        }));
    });
    assert_eq!(calls, total);
    assert!(observed.bytes_max <= workspace);
    assert_eq!(observed.bytes_current, 0);
    assert!(
        result
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("late cancel sentinel")
    );
}

#[test]
fn all_dictionary_bodies_are_budgeted_while_simultaneously_resident() {
    use datafusion::arrow::{
        array::{ArrayRef, DictionaryArray, StringArray},
        datatypes::Int32Type,
    };
    let values =
        StringArray::from_iter_values((0..512).map(|i| format!("{i:04}{}", "x".repeat(4096))));
    let payload = u64::try_from(values.value_data().len()).unwrap();
    let dictionary: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::new(
        vec![0, 1, 2].into(),
        Arc::new(values),
    ));
    let arrays = vec![dictionary; 4];
    let fields: Vec<_> = arrays
        .iter()
        .enumerate()
        .map(|(i, array)| Field::new(format!("dict{i}"), array.data_type().clone(), true))
        .collect();
    let record = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
    let observed = assert_writer_fits(&record);
    assert!(observed >= payload.checked_mul(4).unwrap());
}

#[test]
fn sparse_union_child_slots_are_admitted_before_to_data_clones_the_header() {
    use datafusion::arrow::{array::UnionArray, buffer::ScalarBuffer, datatypes::UnionFields};
    let fields =
        UnionFields::try_new([127], [Field::new("child", DataType::Float64, true)]).unwrap();
    let array = UnionArray::try_new(
        fields,
        ScalarBuffer::from(vec![127_i8]),
        None,
        vec![Arc::new(Float64Array::from(vec![1.0]))],
    )
    .unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "union",
        array.data_type().clone(),
        true,
    )]));
    let record = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
    assert_writer_fits(&record);
}

#[test]
fn flatbuffer_and_framed_metadata_must_fit_the_pinned_signed_length_limit() {
    assert!(super::math::builder(1_u64 << 31, 1).is_err());
    assert!(super::frame(u64::try_from(i32::MAX).unwrap(), 0).is_err());
}

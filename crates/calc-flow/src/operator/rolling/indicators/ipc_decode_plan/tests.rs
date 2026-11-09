use std::{io::Cursor, sync::Arc};

use datafusion::arrow::{
    array::{ArrayRef, BinaryArray, UInt8Array, UInt64Array},
    datatypes::{DataType, Field, Schema},
    ipc::{reader::FileReader, writer::FileWriter},
    record_batch::RecordBatch,
};

use super::{super::ipc_preflight::preflight, plan};

fn fixture() -> RecordBatch {
    let fields = vec![
        Field::new("_state_kind", DataType::UInt8, false),
        Field::new("_entity_id", DataType::UInt64, false),
        Field::new("_entity_position", DataType::UInt64, true),
        Field::new("_indicator_state", DataType::Binary, true),
    ];
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(UInt8Array::from(vec![0, 2])),
        Arc::new(UInt64Array::from(vec![0, 0])),
        Arc::new(UInt64Array::from(vec![Some(0), None])),
        Arc::new(BinaryArray::from(vec![None, None])),
    ];
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

fn encode(record: &RecordBatch) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, &record.schema()).unwrap();
    writer.write(record).unwrap();
    writer.finish().unwrap();
    drop(writer);
    bytes
}

fn assert_reader_bound(record: &RecordBatch) {
    let bytes = encode(record);
    let schema = record.schema();
    let envelope = preflight(&bytes, &schema, u64::MAX, &mut || Ok(())).unwrap();
    let mut planned = None;
    let allocation = allocation_counter::measure(|| {
        planned = Some(plan(&bytes, &schema, &envelope, &mut || Ok(())).unwrap());
    });
    assert_eq!(allocation.count_total, 0, "planning must only borrow");
    let planned = planned.unwrap();
    let mut retained = None;
    let actual = allocation_counter::measure(|| {
        let mut reader = FileReader::try_new(Cursor::new(bytes.as_slice()), None).unwrap();
        let decoded = reader.next().unwrap().unwrap();
        retained = Some((reader, decoded));
    });
    assert!(
        u64::try_from(actual.bytes_current).unwrap()
            <= planned.reader_bytes + super::construction_headers(&envelope).unwrap(),
        "retained {actual:?}, plan {planned:?}"
    );
    assert!(
        actual.bytes_max <= planned.reader_bytes + planned.scratch_bytes,
        "peak {actual:?}, plan {planned:?}"
    );
    assert_eq!(&retained.as_ref().unwrap().1, record);
    drop(retained);
}

#[test]
fn test_zero_allocation_plan_precedes_default_reader() {
    assert_reader_bound(&fixture());
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

fn nullable_record(dtype: &DataType) -> RecordBatch {
    let mut fields = fixture()
        .schema()
        .fields()
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    fields.push(Arc::new(Field::new("extra", dtype.clone(), true)));
    let mut columns = fixture().columns().to_vec();
    columns.push(datafusion::arrow::array::new_null_array(dtype, 2));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

#[test]
fn test_every_pinned_type_has_a_reader_bound() {
    for dtype in empty_types() {
        assert_reader_bound(&nullable_record(&dtype));
    }
}

#[test]
fn test_empty_variadic_buffers_have_physical_descriptor_cost() {
    use datafusion::arrow::{
        array::StringViewArray,
        buffer::{Buffer, ScalarBuffer},
    };
    let view = StringViewArray::try_new(
        ScalarBuffer::from(vec![0_u128; 2]),
        vec![Buffer::from_slice_ref(&[] as &[u8]); 1024],
        None,
    )
    .unwrap();
    let record = nullable_record(&DataType::Utf8View);
    let mut arrays = record.columns().to_vec();
    arrays[4] = Arc::new(view);
    let record = RecordBatch::try_new(record.schema(), arrays).unwrap();
    assert_reader_bound(&record);
}

fn with_extra(extra: ArrayRef) -> RecordBatch {
    let base = fixture();
    let mut fields = base.schema().fields().to_vec();
    fields.push(Arc::new(Field::new(
        "extra",
        extra.data_type().clone(),
        true,
    )));
    let mut columns = base.columns().to_vec();
    columns.push(extra);
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

#[test]
fn test_nonzero_bitmap_offsets_and_nested_list_slices() {
    use datafusion::arrow::array::{BooleanArray, ListArray, StringArray};
    let booleans = BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]);
    assert_reader_bound(&with_extra(Arc::new(booleans.slice(1, 2))));
    let strings = StringArray::from(vec![
        Some("prefix"),
        Some("long text λ"),
        None,
        Some("tail"),
    ]);
    assert_reader_bound(&with_extra(Arc::new(strings.slice(1, 2))));
    let list = ListArray::from_iter_primitive::<datafusion::arrow::datatypes::Int32Type, _, _>([
        Some(vec![Some(0)]),
        Some(vec![Some(1), None]),
        None,
        Some(vec![Some(3)]),
    ]);
    assert_reader_bound(&with_extra(Arc::new(list.slice(1, 2))));
}

#[test]
fn test_sparse_union_maximum_type_id_and_nested_struct_capacity() {
    use datafusion::arrow::{
        array::{Array, Float64Array, StructArray, UnionArray},
        datatypes::{UnionFields, UnionMode},
    };
    let fields =
        UnionFields::try_new([127], [Field::new("field", DataType::Float64, true)]).unwrap();
    for mode in [UnionMode::Sparse, UnionMode::Dense] {
        let offsets = (mode == UnionMode::Dense).then(|| vec![0_i32, 1].into());
        let union = UnionArray::try_new(
            fields.clone(),
            vec![127_i8; 2].into(),
            offsets,
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0]))],
        )
        .unwrap();
        let structs = StructArray::from(vec![(
            Arc::new(Field::new("union", union.data_type().clone(), true)),
            Arc::new(union) as ArrayRef,
        )]);
        let lists = datafusion::arrow::array::ListArray::new(
            Arc::new(Field::new("items", structs.data_type().clone(), true)),
            datafusion::arrow::buffer::OffsetBuffer::new(vec![0_i32, 1, 2].into()),
            Arc::new(structs),
            None,
        );
        assert_reader_bound(&with_extra(Arc::new(lists)));
    }
}

#[test]
fn test_dictionary_reconstruction_and_schema_metadata_are_bounded() {
    use datafusion::arrow::{
        array::{Array, DictionaryArray, Int8Array, StringArray, StructArray},
        datatypes::Int8Type,
    };
    let values = StringArray::from(vec!["first λ", "second"]);
    let dictionary =
        DictionaryArray::<Int8Type>::try_new(Int8Array::from(vec![0, 1]), Arc::new(values))
            .unwrap();
    let child = Arc::new(Field::new("inner", dictionary.data_type().clone(), true));
    let values = StructArray::from(vec![(child, Arc::new(dictionary) as ArrayRef)]);
    let dictionary =
        DictionaryArray::<Int8Type>::try_new(Int8Array::from(vec![1, 0]), Arc::new(values))
            .unwrap();
    let record = with_extra(Arc::new(dictionary));
    let mut fields = record.schema().fields().to_vec();
    let metadata = (0..100)
        .map(|i| (format!("key{i}"), "value".repeat(20)))
        .collect();
    fields[4] = Arc::new(fields[4].as_ref().clone().with_metadata(metadata));
    let schema =
        Schema::new(fields).with_metadata([(String::from("schema"), "large".repeat(400))].into());
    assert_reader_bound(
        &RecordBatch::try_new(Arc::new(schema), record.columns().to_vec()).unwrap(),
    );
}

#[test]
fn test_reader_validation_run_bitmap_scales_with_logical_child_rows() {
    use datafusion::arrow::{
        array::{Array, Int32Array, ListArray, RunArray, StructArray},
        buffer::{NullBuffer, OffsetBuffer},
        datatypes::Int32Type,
    };
    let ends = Int32Array::from(vec![1_000_000]);
    let values = Int32Array::from(vec![None]);
    let run = RunArray::<Int32Type>::try_new(&ends, &values).unwrap();
    let fields = vec![Arc::new(Field::new("run", run.data_type().clone(), false))].into();
    let structs = StructArray::try_new(
        fields,
        vec![Arc::new(run)],
        Some(NullBuffer::new_null(1_000_000)),
    )
    .unwrap();
    let list = ListArray::new(
        Arc::new(Field::new("item", structs.data_type().clone(), true)),
        OffsetBuffer::new(vec![0_i32, 500_000, 1_000_000].into()),
        Arc::new(structs),
        None,
    );
    assert_reader_bound(&with_extra(Arc::new(list)));
}

#[test]
fn test_envelope_requires_exact_same_encoded_slice_and_expected_schema() {
    let record = fixture();
    let bytes = encode(&record);
    let schema = record.schema();
    let envelope = preflight(&bytes, &schema, 2, &mut || Ok(())).unwrap();
    let same_values = bytes.clone();
    let result = plan(&same_values, &schema, &envelope, &mut || Ok(()));
    assert!(
        matches!(result, Err(crate::CalcFlowError::Format {message}) if message.contains("source identity"))
    );
    assert!(
        plan(
            &bytes[..bytes.len() - 1],
            &schema,
            &envelope,
            &mut || Ok(())
        )
        .is_err()
    );
    assert!(plan(&bytes, &Schema::empty(), &envelope, &mut || Ok(())).is_err());
}

#[test]
fn test_cancellation_is_checked_during_the_borrowed_walk() {
    let record = fixture();
    let bytes = encode(&record);
    let schema = record.schema();
    let envelope = preflight(&bytes, &schema, 2, &mut || Ok(())).unwrap();
    let mut verification_calls = 0;
    preflight(&bytes, &schema, 2, &mut || {
        verification_calls += 1;
        Ok(())
    })
    .unwrap();
    let threshold = verification_calls + 3;
    let mut calls = 0;
    let result = plan(&bytes, &schema, &envelope, &mut || {
        calls += 1;
        if calls == threshold {
            return Err(crate::CalcFlowError::Format {
                message: "stop decode plan".into(),
            });
        }
        Ok(())
    });
    assert_eq!(calls, threshold);
    assert!(
        matches!(result, Err(crate::CalcFlowError::Format {message}) if message == "stop decode plan")
    );
}

#[test]
fn test_checked_allocation_arithmetic_rejects_overflow() {
    use super::math;
    assert!(math::add(u64::MAX, 1).is_err());
    assert!(math::mul(u64::MAX, 2).is_err());
    assert!(math::aligned(u64::MAX).is_err());
    assert!(math::hash::<ArrayRef>(u64::MAX).is_err());
    assert!(math::grow::<ArrayRef>(u64::MAX).is_err());
    assert!(math::bitmap(u64::MAX).is_err());
}

#[test]
fn test_dictionary_view_and_list_view_keep_separate_metadata_workspaces() {
    use datafusion::arrow::{
        array::{Array, DictionaryArray, Int8Array, ListViewArray, StringViewArray},
        datatypes::Int8Type,
    };
    let values = StringViewArray::from(vec![
        Some("a long dictionary value stored outside inline view"),
        None,
    ]);
    let dictionary =
        DictionaryArray::<Int8Type>::try_new(Int8Array::from(vec![0, 1]), Arc::new(values))
            .unwrap();
    let view = ListViewArray::new(
        Arc::new(Field::new("item", dictionary.data_type().clone(), true)),
        vec![1_i32, 0].into(),
        vec![1_i32, 2].into(),
        Arc::new(dictionary),
        None,
    );
    assert_reader_bound(&with_extra(Arc::new(view)));
}

#[test]
fn test_fixed_list_nonnullable_validation_expands_the_parent_mask() {
    use datafusion::arrow::{
        array::{FixedSizeListArray, Int32Array},
        buffer::NullBuffer,
    };
    let array = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Int32, false)),
        16_384,
        Arc::new(Int32Array::from(vec![None; 32_768])),
        Some(NullBuffer::new_null(2)),
    )
    .unwrap();
    assert_reader_bound(&with_extra(Arc::new(array)));
}

#[test]
fn test_nested_union_logical_null_validation_and_map_reconstruction() {
    use datafusion::arrow::{
        array::{Array, Float64Array, ListArray, MapArray, StructArray, UnionArray},
        buffer::{NullBuffer, OffsetBuffer},
        datatypes::{UnionFields, UnionMode},
    };
    let fields = UnionFields::try_new(
        [3, 127],
        [
            Field::new("a", DataType::Float64, true),
            Field::new("b", DataType::Float64, true),
        ],
    )
    .unwrap();
    for mode in [UnionMode::Sparse, UnionMode::Dense] {
        let offsets = (mode == UnionMode::Dense).then(|| vec![0_i32, 1].into());
        let children: Vec<ArrayRef> = vec![
            Arc::new(Float64Array::from(vec![None, Some(1.0)])),
            Arc::new(Float64Array::from(vec![Some(2.0), None])),
        ];
        let union =
            UnionArray::try_new(fields.clone(), vec![3_i8, 127].into(), offsets, children).unwrap();
        let column = Arc::new(Field::new("u", union.data_type().clone(), false));
        let structure = StructArray::try_new(
            vec![column].into(),
            vec![Arc::new(union)],
            Some(NullBuffer::new_null(2)),
        )
        .unwrap();
        let list = ListArray::new(
            Arc::new(Field::new("item", structure.data_type().clone(), true)),
            OffsetBuffer::new(vec![0_i32, 1, 2].into()),
            Arc::new(structure),
            None,
        );
        assert_reader_bound(&with_extra(Arc::new(list)));
    }
    let entries = StructArray::from(vec![
        (
            Arc::new(Field::new("key", DataType::UInt64, false)),
            Arc::new(UInt64Array::from(vec![1, 2])) as ArrayRef,
        ),
        (
            Arc::new(Field::new("value", DataType::Binary, true)),
            Arc::new(BinaryArray::from(vec![Some(b"value".as_slice()), None])) as ArrayRef,
        ),
    ]);
    let map = MapArray::new(
        Arc::new(Field::new("entries", entries.data_type().clone(), false)),
        OffsetBuffer::new(vec![0_i32, 1, 2].into()),
        entries,
        None,
        false,
    );
    assert_reader_bound(&with_extra(Arc::new(map)));
}

#[test]
fn test_wide_struct_reconstruction_retains_array_data_vector_capacity() {
    use datafusion::arrow::array::{Array, Int32Array, ListArray, StructArray};
    let columns = (0..129)
        .map(|i| {
            (
                Arc::new(Field::new(format!("column{i}"), DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![i, i + 1])) as ArrayRef,
            )
        })
        .collect::<Vec<_>>();
    let structure = StructArray::from(columns);
    let lists = ListArray::new(
        Arc::new(Field::new("items", structure.data_type().clone(), true)),
        datafusion::arrow::buffer::OffsetBuffer::new(vec![0_i32, 1, 2].into()),
        Arc::new(structure),
        None,
    );
    assert_reader_bound(&with_extra(Arc::new(lists)));
}

#[test]
fn test_invalid_utf8_reader_error_is_inside_the_admitted_peak() {
    use datafusion::arrow::array::StringArray;
    let record = with_extra(Arc::new(StringArray::from(vec!["valid", "second"])));
    let schema = record.schema();
    let mut bytes = encode(&record);
    let value_offset = {
        let envelope = preflight(&bytes, &schema, 2, &mut || Ok(())).unwrap();
        let block = envelope.footer().recordBatches().unwrap().get(0);
        let message = super::messages::record(&bytes, block).unwrap();
        envelope.record_body.offset
            + usize::try_from(message.buffers().unwrap().get(11).offset()).unwrap()
    };
    bytes[value_offset] = 0xff;
    let envelope = preflight(&bytes, &schema, 2, &mut || Ok(())).unwrap();
    let planned = plan(&bytes, &schema, &envelope, &mut || Ok(())).unwrap();
    let mut error = None;
    let actual = allocation_counter::measure(|| {
        let mut reader = FileReader::try_new(Cursor::new(bytes.as_slice()), None).unwrap();
        error = Some(reader.next().unwrap().unwrap_err());
    });
    assert!(error.unwrap().to_string().contains("UTF8"));
    assert!(
        actual.bytes_max <= planned.reader_bytes + planned.scratch_bytes,
        "error peak {actual:?}, plan {planned:?}"
    );
}

#[test]
fn test_existing_schema_map_capacity_is_not_copied_by_the_reader() {
    let record = fixture();
    let schema = record
        .schema()
        .as_ref()
        .clone()
        .with_metadata([(String::from("key"), String::from("value"))].into());
    let record = RecordBatch::try_new(Arc::new(schema), record.columns().to_vec()).unwrap();
    let bytes = encode(&record);
    let schema = record.schema();
    let envelope = preflight(&bytes, &schema, 2, &mut || Ok(())).unwrap();
    let normal = plan(&bytes, &schema, &envelope, &mut || Ok(())).unwrap();
    let mut spare = std::collections::HashMap::with_capacity(16_384);
    spare.insert(String::from("key"), String::from("value"));
    let expected = schema.as_ref().clone().with_metadata(spare);
    let oversized = plan(&bytes, &expected, &envelope, &mut || Ok(())).unwrap();
    assert_eq!(normal.reader_bytes, oversized.reader_bytes);
    assert_eq!(normal.scratch_bytes, oversized.scratch_bytes);
    assert_reader_bound(&record);
}

#[test]
fn test_cancellation_reaches_the_additional_metadata_inventory() {
    let record = fixture();
    let metadata = (0..1000)
        .map(|i| (format!("key{i}"), String::from("value")))
        .collect();
    let schema = Arc::new(record.schema().as_ref().clone().with_metadata(metadata));
    let record = RecordBatch::try_new(schema.clone(), record.columns().to_vec()).unwrap();
    let bytes = encode(&record);
    let mut verified_calls = 0;
    let envelope = preflight(&bytes, &schema, 2, &mut || {
        verified_calls += 1;
        Ok(())
    })
    .unwrap();
    let mut calls = 0;
    let result = plan(&bytes, &schema, &envelope, &mut || {
        calls += 1;
        if calls > verified_calls + 99 {
            return Err(crate::CalcFlowError::Format {
                message: "metadata cancellation".into(),
            });
        }
        Ok(())
    });
    assert!(
        matches!(result, Err(crate::CalcFlowError::Format {message}) if message == "metadata cancellation")
    );
}

#[test]
fn test_nested_type_error_formatting_has_a_source_derived_workspace() {
    let mut dtype = DataType::Utf8;
    for depth in 0..20 {
        let metadata = [(String::from("escaped"), "\u{7}".repeat(100))].into();
        let child = Field::new(format!("level{depth}"), dtype, true).with_metadata(metadata);
        dtype = DataType::Struct(vec![child].into());
    }
    let record = nullable_record(&dtype);
    let schema = record.schema();
    let bytes = encode(&record);
    let envelope = preflight(&bytes, &schema, 2, &mut || Ok(())).unwrap();
    let bound =
        super::diagnostics::plan(envelope.footer().schema().unwrap(), &mut || Ok(())).unwrap();
    let actual = allocation_counter::measure(|| {
        let message = format!("Failed to read {dtype}: expected {dtype}, actual {dtype}");
        std::hint::black_box(message);
    });
    assert!(
        actual.bytes_max <= bound,
        "formatting {actual:?}, bound {bound}"
    );
}

#[test]
fn test_record_block_inventory_includes_vec_minimum_capacity() {
    let record = fixture();
    let bytes = encode(&record);
    let schema = record.schema();
    let envelope = preflight(&bytes, &schema, 2, &mut || Ok(())).unwrap();
    let blocks = envelope
        .footer()
        .recordBatches()
        .unwrap()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(blocks.len(), 1);
    let storage =
        u64::try_from(blocks.capacity() * size_of::<datafusion::arrow::ipc::Block>()).unwrap();
    assert!(
        storage <= super::record_blocks().unwrap(),
        "capacity bytes {storage} exceed recorded block quota {}",
        super::record_blocks().unwrap()
    );
}

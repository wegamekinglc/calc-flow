use super::*;
use datafusion::arrow::buffer::{Buffer, ScalarBuffer};

#[test]
fn test_timezone_and_foreign_buffer_alias_charge_one_byte_allocation() {
    use datafusion::common::ScalarValue;
    type Bytes = <datafusion::object_store::PutPayload as IntoIterator>::Item;
    struct TimezoneBytes(Arc<str>);
    impl AsRef<[u8]> for TimezoneBytes {
        fn as_ref(&self) -> &[u8] {
            self.0.as_bytes()
        }
    }
    let timezone: Arc<str> = Arc::from("UTC");
    let buffer = Buffer::from(Bytes::from_owner(TimezoneBytes(Arc::clone(&timezone))));
    let mut footprint = Footprint::default();
    footprint
        .scalar(
            &ScalarValue::TimestampMicrosecond(Some(1), Some(timezone)),
            "features",
        )
        .unwrap();
    footprint.buffer(&buffer, "features").unwrap();
    assert_eq!(
        footprint.bytes("features").unwrap(),
        size_of::<ScalarValue>() as u64 + 3
    );
}

#[test]
fn test_sliced_external_buffer_keeps_reported_capacity_and_alias_maximum() {
    let bytes = datafusion::object_store::PutPayload::from_static(&[7; 128])
        .into_iter()
        .next()
        .unwrap();
    let imported = Buffer::from(bytes);
    assert_eq!(imported.capacity(), 128);
    let slice = imported.slice_with_length(96, 8);
    let mut footprint = Footprint::default();
    footprint.buffer(&slice, "features").unwrap();
    assert_eq!(footprint.bytes("features").unwrap(), 128);
    footprint.buffer(&imported, "features").unwrap();
    assert_eq!(footprint.bytes("features").unwrap(), 128);
}

#[test]
fn test_distinct_owned_buffers_count_separately_from_aliases() {
    let first = ScalarBuffer::<f64>::from(vec![1.0; 32]);
    let second = ScalarBuffer::<f64>::from(vec![1.0; 32]);
    let mut left = Footprint::default();
    left.buffer(first.inner(), "features").unwrap();
    let mut right = Footprint::default();
    right.buffer(first.inner(), "features").unwrap();
    right.buffer(second.inner(), "features").unwrap();
    left.merge(&right, "features").unwrap();
    assert_eq!(left.bytes("features").unwrap(), 512);
}

#[test]
fn test_shared_list_slices_retain_one_child_buffer_and_distinct_copies_retain_two() {
    use datafusion::{
        arrow::{
            array::{Array, Float64Array, ListArray},
            buffer::OffsetBuffer,
            datatypes::{DataType, Field},
        },
        common::ScalarValue,
    };
    let list = || {
        ListArray::new(
            Arc::new(Field::new("item", DataType::Float64, true)),
            OffsetBuffer::new(vec![0, 8192, 16384, 24576, 32768].into()),
            Arc::new(Float64Array::from(vec![1.0; 32768])),
            None,
        )
    };
    let source = list();
    let mut footprint = Footprint::default();
    for index in 0..4 {
        footprint
            .scalar(
                &ScalarValue::try_from_array(&source, index).unwrap(),
                "features",
            )
            .unwrap();
    }
    let shared = footprint.bytes("features").unwrap();
    assert!(
        shared < 272_144,
        "one 256KiB backing plus bounded descriptors: {shared}"
    );
    let independent = list();
    footprint
        .scalar(
            &ScalarValue::try_from_array(&independent, 0).unwrap(),
            "features",
        )
        .unwrap();
    assert!(footprint.bytes("features").unwrap() > shared + 262_144);
    assert_eq!(source.len(), 4);
}

#[test]
fn test_equal_owned_strings_have_distinct_capacity_charges() {
    use datafusion::common::ScalarValue;
    let text = || {
        let mut value = String::with_capacity(1024);
        value.push('x');
        ScalarValue::Utf8(Some(value))
    };
    let mut footprint = Footprint::default();
    footprint.scalar(&text(), "features").unwrap();
    footprint.scalar(&text(), "features").unwrap();
    assert_eq!(
        footprint.bytes("features").unwrap(),
        2 * (1024 + size_of::<ScalarValue>() as u64)
    );
}

#[test]
fn test_nested_struct_map_and_dictionary_keep_shared_children_once() {
    use datafusion::{
        arrow::{
            array::{
                ArrayRef, Int32Builder, MapBuilder, StringBuilder, StringDictionaryBuilder,
                StructArray,
            },
            datatypes::{DataType, Field, Int8Type},
        },
        common::ScalarValue,
    };
    let mut map = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
    map.keys().append_value("price");
    map.values().append_value(42);
    map.append(true).unwrap();
    let map: ArrayRef = Arc::new(map.finish());
    let mut dictionary = StringDictionaryBuilder::<Int8Type>::new();
    dictionary.append("immutable context").unwrap();
    let dictionary: ArrayRef = Arc::new(dictionary.finish());
    let fields = vec![
        Arc::new(Field::new("map", map.data_type().clone(), false)),
        Arc::new(Field::new(
            "dictionary",
            dictionary.data_type().clone(),
            false,
        )),
    ];
    let value = ScalarValue::Struct(Arc::new(StructArray::new(
        fields.into(),
        vec![map, dictionary],
        None,
    )));
    let mut footprint = Footprint::default();
    footprint.scalar(&value, "features").unwrap();
    let first = footprint.bytes("features").unwrap();
    footprint.scalar(&value.clone(), "features").unwrap();
    assert_eq!(
        footprint.bytes("features").unwrap(),
        first + size_of::<ScalarValue>() as u64
    );
    assert!(first > 128);
    assert!(matches!(value.data_type(), DataType::Struct(_)));
}

#[test]
fn test_component_merge_deduplicates_shared_struct_descriptors() {
    use datafusion::{
        arrow::{
            array::{Float64Array, StructArray},
            datatypes::{DataType, Field},
        },
        common::ScalarValue,
    };
    let value = ScalarValue::Struct(Arc::new(StructArray::new(
        vec![Arc::new(Field::new("price", DataType::Float64, false))].into(),
        vec![Arc::new(Float64Array::from(vec![1.0]))],
        None,
    )));
    let mut first = Footprint::default();
    first.scalar(&value, "features").unwrap();
    let bytes = first.bytes("features").unwrap();
    let mut second = Footprint::default();
    second.scalar(&value, "features").unwrap();
    first.merge(&second, "features").unwrap();
    assert_eq!(
        first.bytes("features").unwrap(),
        bytes + size_of::<ScalarValue>() as u64
    );
}

#[test]
fn test_dictionary_scalar_charges_its_owned_type_box_separately() {
    use datafusion::{arrow::datatypes::DataType, common::ScalarValue};
    let value = ScalarValue::Dictionary(
        Box::new(DataType::Int8),
        Box::new(ScalarValue::Float64(None)),
    );
    let mut footprint = Footprint::default();
    footprint.scalar(&value, "features").unwrap();
    assert_eq!(
        footprint.bytes("features").unwrap(),
        (2 * size_of::<ScalarValue>() + size_of::<DataType>()) as u64
    );
}

#[test]
fn test_borrowed_backing_preflight_shares_buffers_without_adopting_caller_descriptors() {
    use datafusion::arrow::{
        array::{Array, Float64Array, ListArray},
        buffer::OffsetBuffer,
        datatypes::{DataType, Field},
    };
    let values = Arc::new(Float64Array::from(vec![1.0; 128]));
    let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 64, 128]));
    let expected = values.values().inner().capacity() + offsets.inner().inner().capacity();
    let source = ListArray::new(
        Arc::new(Field::new("item", DataType::Float64, true)),
        offsets,
        values,
        None,
    );
    let left: ArrayRef = Arc::new(source.slice(0, 1));
    let right: ArrayRef = Arc::new(source.slice(1, 1));
    let mut footprint = Footprint::default();
    footprint
        .backing(&left, "features", &mut || Ok(()))
        .unwrap();
    footprint
        .backing(&right, "features", &mut || Ok(()))
        .unwrap();
    assert_eq!(footprint.bytes("features").unwrap(), expected as u64);
    assert!(left.as_any().downcast_ref::<ListArray>().unwrap().len() == 1);
}

#[test]
fn test_incremental_bytes_preserves_new_owned_cost_and_shared_owner_differences() {
    let source = Buffer::from_vec(vec![7_u8; 1024]);
    let distinct = Buffer::from_vec(vec![7_u8; 512]);
    let mut baseline = Footprint::default();
    baseline.buffer(&source, "features").unwrap();
    baseline.add_owned(999, "features").unwrap();
    let mut output = Footprint::default();
    output.buffer(&source.slice(128), "features").unwrap();
    output.add_owned(7, "features").unwrap();
    assert_eq!(output.incremental_bytes(&baseline, "features").unwrap(), 7);
    output.buffer(&distinct, "features").unwrap();
    assert_eq!(
        output.incremental_bytes(&baseline, "features").unwrap(),
        519
    );
    output.owned = u64::MAX;
    assert!(output.incremental_bytes(&baseline, "features").is_err());
}

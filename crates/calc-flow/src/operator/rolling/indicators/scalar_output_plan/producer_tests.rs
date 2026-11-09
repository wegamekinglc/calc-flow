use datafusion::{
    arrow::datatypes::{DataType, Field, UnionFields, UnionMode},
    common::ScalarValue,
};
use std::{collections::HashMap, hash::BuildHasher};

fn tombstoned_metadata() -> HashMap<String, String> {
    let mut map = HashMap::<String, String>::with_capacity(896);
    let mask = (map.capacity() + 1).next_power_of_two() - 1;
    let mut keys = Vec::new();
    for ordinal in 0..5_000_000_u64 {
        let key = format!("key{ordinal}");
        if map.hasher().hash_one(&key) & u64::try_from(mask).unwrap() != 0 {
            continue;
        }
        map.insert(key.clone(), "value".into());
        keys.push(key);
        if keys.len() == 896 {
            break;
        }
    }
    assert_eq!(keys.len(), 896);
    for key in &keys[..895] {
        map.remove(key);
    }
    map
}
fn union_value(metadata: HashMap<String, String>) -> ScalarValue {
    let field = Field::new("payload", DataType::Float64, true).with_metadata(metadata);
    let fields = UnionFields::try_new([0], [field]).unwrap();
    ScalarValue::Union(
        Some((0, Box::new(ScalarValue::Float64(Some(1.0))))),
        fields,
        UnionMode::Sparse,
    )
}
fn convert(value: &ScalarValue) -> datafusion::arrow::array::ArrayRef {
    super::encode_column(&value.data_type(), 1, &mut |_| Some(value), &mut || Ok(())).unwrap()
}
#[test]
fn test_controlled_union_does_not_clone_private_metadata_buckets() {
    let metadata = tombstoned_metadata();
    assert_eq!(metadata.len(), 1);
    let fresh = metadata
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let old = union_value(metadata);
    let fresh = union_value(fresh);
    let first = allocation_counter::measure(|| {
        std::hint::black_box(convert(&old));
    });
    let second = allocation_counter::measure(|| {
        std::hint::black_box(convert(&fresh));
    });
    assert_eq!(first.bytes_max, second.bytes_max);
    let expected = ScalarValue::iter_to_array([old.clone()]).unwrap();
    assert_eq!(convert(&old).to_data(), expected.to_data());
}

fn convert_values(dtype: &DataType, values: &[ScalarValue]) -> datafusion::arrow::array::ArrayRef {
    super::encode_column(dtype, values.len(), &mut |row| values.get(row), &mut || {
        Ok(())
    })
    .unwrap()
}
fn null_union() -> ScalarValue {
    ScalarValue::Union(
        None,
        UnionFields::try_new([0], [Field::new("null-value", DataType::Float64, true)]).unwrap(),
        UnionMode::Sparse,
    )
}
fn run_value(value: ScalarValue) -> ScalarValue {
    use std::sync::Arc;
    let ends = Arc::new(Field::new("ends", DataType::Int32, false));
    let field = Arc::new(
        Field::new("value", value.data_type(), true)
            .with_metadata([(String::from("unit"), String::from("USD"))].into()),
    );
    ScalarValue::RunEndEncoded(ends, field, Box::new(value))
}
#[test]
fn test_union_dictionary_run_union_routes_preserve_values_and_metadata() {
    for leaf in [
        null_union(),
        union_value([(String::from("meta"), String::from("kept"))].into()),
    ] {
        let inner = ScalarValue::Dictionary(Box::new(DataType::Int8), Box::new(run_value(leaf)));
        let fields =
            UnionFields::try_new([127], [Field::new("nested", inner.data_type(), true)]).unwrap();
        let outer = ScalarValue::Union(Some((127, Box::new(inner))), fields, UnionMode::Dense);
        let expected = ScalarValue::iter_to_array([outer.clone()]).unwrap();
        assert_eq!(convert(&outer).to_data(), expected.to_data());
    }
}
#[test]
fn test_dictionary_physical_validity_differs_from_single_child_logical_null() {
    use datafusion::arrow::{
        array::{Array, DictionaryArray, UnionArray},
        datatypes::Int8Type,
    };
    let value =
        ScalarValue::Dictionary(Box::new(DataType::Int8), Box::new(run_value(null_union())));
    assert!(value.is_null());
    let column = convert_values(&value.data_type(), &[value.clone(), value.clone()]);
    let dictionary = column
        .as_any()
        .downcast_ref::<DictionaryArray<Int8Type>>()
        .unwrap();
    assert_eq!(dictionary.keys().null_count(), 0);
    assert_eq!(dictionary.keys().values().as_ref(), [0, 1]);
    let fields = UnionFields::try_new([0], [Field::new("dict", value.data_type(), true)]).unwrap();
    let union = ScalarValue::Union(Some((0, Box::new(value))), fields, UnionMode::Sparse);
    let array = convert(&union);
    let array = array
        .as_any()
        .downcast_ref::<UnionArray>()
        .unwrap()
        .child(0);
    assert_eq!(
        array
            .as_any()
            .downcast_ref::<DictionaryArray<Int8Type>>()
            .unwrap()
            .keys()
            .null_count(),
        1
    );
}
#[test]
fn test_dictionary_only_valid_key_positions_must_fit_the_key_type() {
    let null = ScalarValue::Dictionary(
        Box::new(DataType::Int8),
        Box::new(ScalarValue::Float64(None)),
    );
    let valid = ScalarValue::Dictionary(
        Box::new(DataType::Int8),
        Box::new(ScalarValue::Float64(Some(1.0))),
    );
    let mut values = vec![null; 129];
    values[0] = valid.clone();
    let dtype = valid.data_type();
    assert_eq!(convert_values(&dtype, &values).len(), 129);
    values[128] = valid;
    assert!(
        super::encode_column(
            &dtype,
            values.len(),
            &mut |row| values.get(row),
            &mut || Ok(())
        )
        .is_err()
    );
}
#[test]
fn test_fixed_list_null_repair_keeps_the_first_nonnull_shape() {
    use datafusion::arrow::array::{FixedSizeListArray, Float64Array};
    use std::sync::Arc;
    let field = Arc::new(
        Field::new("item", DataType::Float64, true)
            .with_metadata([(String::from("unit"), String::from("USD"))].into()),
    );
    let null =
        ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new_null(field.clone(), 1, 1)));
    let valid = ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new(
        field,
        2,
        Arc::new(Float64Array::from(vec![1.0, 2.0])),
        None,
    )));
    let dtype = valid.data_type();
    let values = vec![null, valid];
    let expected = ScalarValue::iter_to_array(values.clone()).unwrap();
    assert_eq!(
        convert_values(&dtype, &values).to_data(),
        expected.to_data()
    );
}
#[test]
fn test_empty_columns_and_unrepresentable_null_holes_are_guarded() {
    let dtype = DataType::Union(UnionFields::empty(), UnionMode::Dense);
    let array = super::encode_column(
        &dtype,
        0,
        &mut |_| panic!("no cell for zero rows"),
        &mut || Ok(()),
    )
    .unwrap();
    assert_eq!(array.len(), 0);
    assert!(super::encode_column(&dtype, 1, &mut |_| None, &mut || Ok(())).is_err());
}

#[test]
fn test_fixed_list_first_nonnull_scan_honors_cancellation() {
    use datafusion::arrow::array::FixedSizeListArray;
    use std::sync::Arc;
    let field = Arc::new(Field::new("item", DataType::Float64, true));
    let value = ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new_null(field, 1, 1)));
    let dtype = value.data_type();
    let values = [value.clone(), value];
    let mut calls = 0;
    let result = super::encode_column(&dtype, 2, &mut |row| values.get(row), &mut || {
        calls += 1;
        if calls == 7 {
            return Err(crate::CalcFlowError::Format {
                message: "cancel first-nonnull scan".into(),
            });
        }
        Ok(())
    });
    assert!(
        matches!(result,Err(crate::CalcFlowError::Format{message}) if message=="cancel first-nonnull scan")
    );
}

fn union_dictionary(value: ScalarValue) -> ScalarValue {
    let child = ScalarValue::Dictionary(Box::new(DataType::Int8), Box::new(value));
    let fields =
        UnionFields::try_new([0], [Field::new("dictionary", child.data_type(), true)]).unwrap();
    ScalarValue::Union(Some((0, Box::new(child))), fields, UnionMode::Sparse)
}
#[test]
fn test_nested_dictionary_fallback_reports_its_exact_key_span_limit() {
    let value = union_dictionary(ScalarValue::Float64(None));
    let dtype = value.data_type();
    assert_eq!(convert_values(&dtype, &vec![value.clone(); 128]).len(), 128);
    let values = vec![value; 129];
    let result = super::encode_column(
        &dtype,
        values.len(),
        &mut |row| values.get(row),
        &mut || Ok(()),
    );
    assert!(
        matches!(result, Err(crate::CalcFlowError::Format {message}) if message.contains("dictionary fallback key span"))
    );
}
#[test]
fn test_nested_dictionary_identical_physical_values_do_not_accumulate_key_span() {
    let value = union_dictionary(ScalarValue::Null);
    let dtype = value.data_type();
    assert_eq!(convert_values(&dtype, &vec![value; 129]).len(), 129);
}

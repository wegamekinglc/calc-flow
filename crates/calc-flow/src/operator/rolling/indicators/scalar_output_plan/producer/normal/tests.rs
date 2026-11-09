use datafusion::arrow::{
    array::{
        Array, ArrayRef, Float64Array, ListArray, ListViewArray, MapArray, RunArray, StructArray,
    },
    buffer::{OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field, Int32Type},
};
use std::sync::Arc;

fn view() -> ArrayRef {
    let field = Arc::new(
        Field::new("item", DataType::Float64, true)
            .with_metadata([(String::from("unit"), String::from("USD"))].into()),
    );
    Arc::new(ListViewArray::new(
        field,
        vec![2, 0].into(),
        vec![1, 1].into(),
        Arc::new(Float64Array::from(vec![10., 20., 30., 40.])),
        None,
    ))
}
fn compare(arrays: &[ArrayRef]) {
    let refs = arrays.iter().map(AsRef::as_ref).collect::<Vec<_>>();
    let expected = datafusion::arrow::compute::concat(&refs).unwrap();
    let actual = super::concat(&refs, &mut || Ok(())).unwrap();
    assert_eq!(actual.to_data(), expected.to_data());
    assert_eq!(actual.data_type(), expected.data_type());
}
#[test]
fn test_controlled_list_view_concat_preserves_full_values_metadata_and_slices() {
    let value = view();
    compare(&[value.slice(1, 1), value.slice(0, 1)]);
    compare(&[value.slice(0, 0), value.clone()]);
    compare(std::slice::from_ref(&value));
}
#[test]
fn test_normal_ancestors_route_list_view_without_changing_concatenation() {
    let value = view();
    let field = Arc::new(Field::new("nested", value.data_type().clone(), true));
    let list: ArrayRef = Arc::new(ListArray::new(
        field.clone(),
        OffsetBuffer::new(vec![0, 1, 2].into()),
        value.clone(),
        None,
    ));
    let structure: ArrayRef = Arc::new(StructArray::new(
        vec![field].into(),
        vec![value.clone()],
        None,
    ));
    let keys: ArrayRef = Arc::new(datafusion::arrow::array::Int32Array::from(vec![1, 2]));
    let entries = StructArray::new(
        vec![
            Field::new("key", DataType::Int32, false),
            Field::new("value", value.data_type().clone(), true),
        ]
        .into(),
        vec![keys, value.clone()],
        None,
    );
    let map: ArrayRef = Arc::new(MapArray::new(
        Arc::new(Field::new("entries", entries.data_type().clone(), false)),
        OffsetBuffer::new(vec![0, 1, 2].into()),
        entries,
        None,
        false,
    ));
    let ends = datafusion::arrow::array::Int32Array::new(ScalarBuffer::from(vec![2, 4]), None);
    let run: ArrayRef = Arc::new(RunArray::<Int32Type>::try_new(&ends, value.as_ref()).unwrap());
    for array in [list, structure, map, run] {
        compare(&[array.slice(1, 1), array.clone()]);
    }
}

use super::*;
use datafusion::arrow::{
    array::{Float64Array, ListArray},
    buffer::OffsetBuffer,
    datatypes::{Field, UnionFields, UnionMode},
};
use std::sync::Arc;

#[test]
fn scalar_clone_quote_covers_boxes_and_owned_strings() {
    let fields = UnionFields::try_new([3], [Field::new("selected", DataType::Utf8, true)]).unwrap();
    let scalar = ScalarValue::Union(
        Some((
            3,
            Box::new(ScalarValue::Dictionary(
                Box::new(DataType::Int8),
                Box::new(ScalarValue::Utf8(Some("x".repeat(4097)))),
            )),
        )),
        fields,
        UnionMode::Sparse,
    );
    let quote = scalar_heap(&scalar, &mut || Ok(())).unwrap();
    let mut copied = None;
    let observed = allocation_counter::measure(|| copied = Some(scalar.clone()));
    assert_eq!(u64::try_from(observed.bytes_current).unwrap(), quote);
    assert_eq!(copied.unwrap(), scalar);
}

#[test]
fn nested_array_clone_borrows_large_backing() {
    let values = Arc::new(Float64Array::from(vec![1.0; 200_000]));
    let array = ListArray::new(
        Arc::new(Field::new("item", DataType::Float64, true)),
        OffsetBuffer::new(vec![0_i32, 200_000].into()),
        values,
        None,
    );
    let scalar = ScalarValue::List(Arc::new(array));
    let quote = scalar_heap(&scalar, &mut || Ok(())).unwrap();
    let mut copied = None;
    let observed = allocation_counter::measure(|| copied = Some(scalar.clone()));
    assert_eq!(quote, 0);
    assert_eq!(observed.bytes_max, 0);
    assert!(matches!(copied, Some(ScalarValue::List(_))));
}

#[test]
fn clone_planning_propagates_cancellation() {
    let result = scalar_heap(&ScalarValue::Float64(None), &mut || Err(invalid()));
    assert!(result.is_err());
}

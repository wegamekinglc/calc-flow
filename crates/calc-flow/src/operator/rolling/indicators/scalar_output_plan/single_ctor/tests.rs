use super::super::super::footprint::Footprint;
use datafusion::{
    arrow::{
        array::{Array, Float64Array, StructArray},
        datatypes::{DataType, Field, UnionFields, UnionMode},
    },
    common::ScalarValue,
};
use std::sync::Arc;

fn check_bound(value: &ScalarValue) {
    let expected = value.data_type();
    let mut quoted = None;
    let planning = allocation_counter::measure(|| {
        quoted = Some(super::plan(&expected, value, &mut || Ok(())).unwrap());
    });
    assert_eq!(planning.count_total, 0, "planning must precede admission");
    let quoted = quoted.unwrap();
    let mut source = Footprint::default();
    source.scalar(value, "single-constructor").unwrap();
    let mut result = None;
    let measured = allocation_counter::measure(|| {
        result = Some(super::super::producer::single(value, &mut || Ok(())).unwrap());
    });
    let result = result.unwrap();
    let mut output = Footprint::default();
    output
        .array_checked(&result, "single-constructor", &mut || Ok(()))
        .unwrap();
    let owned = output
        .incremental_bytes(&source, "single-constructor")
        .unwrap();
    assert!(
        quoted.logical_owned >= owned,
        "{expected:?}: owned {owned} exceeds {}",
        quoted.logical_owned
    );
    assert!(
        quoted.construction_peak >= measured.bytes_max,
        "peak {} exceeds {}",
        measured.bytes_max,
        quoted.construction_peak
    );
}

#[test]
fn test_leaf_constructor_bounds_cover_valid_null_and_empty_buffers() {
    for value in [
        ScalarValue::Float64(Some(2.0)),
        ScalarValue::Float64(None),
        ScalarValue::Null,
        ScalarValue::Utf8(Some("a".repeat(80))),
        ScalarValue::Utf8(None),
        ScalarValue::FixedSizeBinary(0, Some(Vec::new())),
        ScalarValue::FixedSizeBinary(0, None),
        ScalarValue::Utf8View(Some("a".repeat(32))),
    ] {
        check_bound(&value);
    }
}

fn structure() -> ScalarValue {
    ScalarValue::Struct(Arc::new(StructArray::new(
        vec![Arc::new(Field::new("price", DataType::Float64, false))].into(),
        vec![Arc::new(Float64Array::from(vec![4.0]))],
        None,
    )))
}

#[test]
fn test_direct_array_backed_single_adds_no_owned_allocation() {
    let value = structure();
    let cost = super::plan(&value.data_type(), &value, &mut || Ok(())).unwrap();
    assert_eq!(cost.logical_owned, 0);
    assert_eq!(cost.construction_peak, 0);
    check_bound(&value);
}

fn run(value: ScalarValue) -> ScalarValue {
    ScalarValue::RunEndEncoded(
        Arc::new(Field::new("ends", DataType::Int16, false)),
        Arc::new(Field::new("value", value.data_type(), true)),
        Box::new(value),
    )
}
fn union(value: ScalarValue, mode: UnionMode) -> ScalarValue {
    let fields = UnionFields::try_new(
        [3, 127],
        [
            Arc::new(Field::new("active", value.data_type(), true)),
            Arc::new(Field::new("inactive", DataType::Utf8, true)),
        ],
    )
    .unwrap();
    ScalarValue::Union(Some((3, Box::new(value))), fields, mode)
}

#[test]
fn test_encoded_single_bounds_include_child_descriptor_reconstruction() {
    for value in [
        run(ScalarValue::Null),
        run(structure()),
        ScalarValue::Dictionary(
            Box::new(DataType::Int8),
            Box::new(ScalarValue::Float64(None)),
        ),
        union(structure(), UnionMode::Dense),
        union(ScalarValue::Float64(None), UnionMode::Sparse),
        union(
            ScalarValue::Dictionary(Box::new(DataType::Int8), Box::new(run(structure()))),
            UnionMode::Dense,
        ),
    ] {
        check_bound(&value);
    }
}

#[test]
fn test_fixed_binary_rejects_inconsistent_value_width_before_construction() {
    let value = ScalarValue::FixedSizeBinary(2, Some(vec![1]));
    assert!(super::plan(&DataType::FixedSizeBinary(2), &value, &mut || Ok(())).is_err());
}

#[test]
fn test_wrapper_bounds_count_all_variadic_view_descriptors_without_copying_body() {
    use datafusion::arrow::{array::StringViewArray, buffer::Buffer};
    let views = StringViewArray::new(
        vec![0_u128].into(),
        (0..64)
            .map(|_| Buffer::from_vec(vec![0_u8; 4096]))
            .collect::<Vec<_>>(),
        None,
    );
    let value = ScalarValue::Struct(Arc::new(StructArray::new(
        vec![Arc::new(Field::new("view", DataType::Utf8View, false))].into(),
        vec![Arc::new(views)],
        None,
    )));
    for value in [run(value.clone()), union(value, UnionMode::Dense)] {
        check_bound(&value);
        let cost = super::plan(&value.data_type(), &value, &mut || Ok(())).unwrap();
        assert!(
            cost.logical_owned < 64 * 4096,
            "funded source body was charged again"
        );
    }
}

#[test]
fn test_null_union_and_unsigned_dictionary_follow_their_real_factory_routes() {
    let fields =
        UnionFields::try_new([5], [Arc::new(Field::new("n", DataType::Null, true))]).unwrap();
    check_bound(&ScalarValue::Union(None, fields, UnionMode::Sparse));
    check_bound(&ScalarValue::Dictionary(
        Box::new(DataType::UInt64),
        Box::new(ScalarValue::Null),
    ));
}

#[test]
fn test_recursive_plan_preserves_cancellation_and_rejects_mismatched_children() {
    let value = union(run(structure()), UnionMode::Sparse);
    let mut calls = 0;
    let error = super::plan(&value.data_type(), &value, &mut || {
        calls += 1;
        if calls > 2 {
            return Err(crate::CalcFlowError::Cancelled {
                run_id: "single-constructor".into(),
            });
        }
        Ok(())
    })
    .unwrap_err();
    assert!(matches!(error, crate::CalcFlowError::Cancelled { .. }));
    assert!(super::plan(&DataType::Float64, &value, &mut || Ok(())).is_err());
}

#[test]
fn test_null_dictionary_values_have_representation_specific_identity() {
    use datafusion::arrow::{array::DictionaryArray, datatypes::Int8Type};
    for (child, identical) in [
        (ScalarValue::Null, true),
        (ScalarValue::Float64(None), false),
    ] {
        let scalar = ScalarValue::Dictionary(Box::new(DataType::Int8), Box::new(child));
        let first = super::super::producer::single(&scalar, &mut || Ok(())).unwrap();
        let second = super::super::producer::single(&scalar, &mut || Ok(())).unwrap();
        let first = first
            .as_any()
            .downcast_ref::<DictionaryArray<Int8Type>>()
            .unwrap();
        let second = second
            .as_any()
            .downcast_ref::<DictionaryArray<Int8Type>>()
            .unwrap();
        assert_eq!(first.values().len(), 1);
        assert_eq!(
            first.values().to_data().ptr_eq(&second.values().to_data()),
            identical
        );
        assert!(!first.keys().to_data().ptr_eq(&second.keys().to_data()));
    }
}

#[test]
fn test_empty_fixed_binary_uses_real_buffer_identity() {
    use datafusion::arrow::{
        array::{ArrayData, make_array},
        buffer::Buffer,
    };
    let scalar = ScalarValue::FixedSizeBinary(0, Some(Vec::new()));
    let first = super::super::producer::single(&scalar, &mut || Ok(())).unwrap();
    let second = super::super::producer::single(&scalar, &mut || Ok(())).unwrap();
    let old = make_array(
        ArrayData::builder(DataType::FixedSizeBinary(0))
            .len(1)
            .add_buffer(Buffer::from_vec(Vec::<u8>::new()))
            .build()
            .unwrap(),
    );
    let first = first.to_data();
    assert_eq!(first.buffers().len(), 1);
    assert_eq!(first.buffers()[0].len(), 0);
    assert!(first.ptr_eq(&second.to_data()));
    assert!(!first.ptr_eq(&old.to_data()));
}

#[test]
fn test_single_union_preserves_shared_field_metadata_without_deep_clone() {
    let mut metadata = (0..4096)
        .map(|index| (index.to_string(), "metadata".into()))
        .collect::<std::collections::HashMap<_, _>>();
    metadata.retain(|key, _| key == "1");
    let field = Arc::new(Field::new("child", DataType::Float64, true).with_metadata(metadata));
    let fields = UnionFields::try_new([4], [Arc::clone(&field)]).unwrap();
    let value = ScalarValue::Union(
        Some((4, Box::new(ScalarValue::Float64(Some(1.0))))),
        fields,
        UnionMode::Dense,
    );
    check_bound(&value);
    let output = super::super::producer::single(&value, &mut || Ok(())).unwrap();
    let DataType::Union(fields, _) = output.data_type() else {
        panic!("union");
    };
    assert!(Arc::ptr_eq(fields.iter().next().unwrap().1, &field));
}

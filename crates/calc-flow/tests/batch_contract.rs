use calc_flow::{Batch, BatchMetadata, CalcFlowError, Result, StaticArrayValues};
use std::{collections::BTreeMap, mem::size_of};

fn sample_metadata() -> BatchMetadata {
    BatchMetadata::new(
        "orders",
        3,
        BTreeMap::from([("region".to_owned(), serde_json::json!({"code": 7}))]),
    )
    .unwrap()
}

fn invalid_argument_field(result: Result<Batch>) -> String {
    match result.unwrap_err() {
        CalcFlowError::InvalidArgument { field, .. } => field,
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

#[test]
fn metadata_clones_share_source_and_attributes() {
    let metadata = sample_metadata();
    let cloned = metadata.clone();

    assert_eq!(metadata.source().as_ptr(), cloned.source().as_ptr());
    assert!(std::ptr::eq(metadata.attributes(), cloned.attributes()));
    assert_eq!(cloned, metadata);
}

#[test]
fn metadata_json_and_debug_representations_are_stable() {
    let metadata = sample_metadata();
    let wire = serde_json::to_string(&metadata).unwrap();

    assert_eq!(
        wire,
        r#"{"source":"orders","sequence":3,"attributes":{"region":{"code":7}}}"#
    );
    assert_eq!(
        serde_json::from_str::<BatchMetadata>(&wire).unwrap(),
        metadata
    );
    assert_eq!(
        serde_json::to_string(&BatchMetadata::default()).unwrap(),
        r#"{"source":"","sequence":0,"attributes":{}}"#
    );
    assert_eq!(
        format!("{metadata:?}"),
        r#"BatchMetadata { source: "orders", sequence: 3, attributes: {"region": Object {"code": Number(7)}} }"#
    );
}

#[test]
fn latched_arrays_reject_invalid_descriptors_shapes_and_counts() {
    assert_eq!(
        invalid_argument_field(Batch::static_array_float(
            "",
            "float64",
            vec![1],
            None,
            vec![1.0],
        )),
        "static_inputs.backend"
    );
    assert_eq!(
        invalid_argument_field(Batch::static_array_int(
            "numpy",
            "float64",
            vec![1],
            None,
            vec![1],
        )),
        "static_inputs.dtype"
    );
    assert_eq!(
        invalid_argument_field(Batch::static_array_bool(
            "numpy",
            vec![u64::MAX, 2],
            None,
            vec![],
        )),
        "static_inputs.shape"
    );
    assert_eq!(
        invalid_argument_field(Batch::static_array_uint(
            "numpy",
            "uint8",
            vec![2],
            Some(vec![false]),
            vec![1, 2],
        )),
        "static_inputs.shape"
    );
    assert_eq!(
        invalid_argument_field(Batch::static_array_uint(
            "numpy",
            "uint8",
            vec![2],
            Some(vec![false, true]),
            vec![1, 2],
        )),
        "static_inputs.shape"
    );
    assert_eq!(
        invalid_argument_field(Batch::static_array_uint(
            "numpy",
            "uint8",
            vec![1],
            None,
            vec![u64::from(u8::MAX) + 1],
        )),
        "static_inputs.dtype"
    );
}

#[test]
fn latched_arrays_report_owned_lengths_backends_and_byte_estimates() {
    let cases = [
        (
            Batch::static_array_bool("numpy", vec![1], None, vec![true]).unwrap(),
            size_of::<bool>() + 5 + 4 + size_of::<u64>(),
        ),
        (
            Batch::static_array_int("numpy", "int8", vec![1], None, vec![1]).unwrap(),
            size_of::<i64>() + 5 + 4 + size_of::<u64>(),
        ),
        (
            Batch::static_array_uint("numpy", "uint8", vec![1], None, vec![1]).unwrap(),
            size_of::<u64>() + 5 + 5 + size_of::<u64>(),
        ),
        (
            Batch::static_array_float("numpy", "float64", vec![1], Some(vec![false]), vec![1.0])
                .unwrap(),
            size_of::<f64>() + 5 + 7 + size_of::<u64>() + size_of::<bool>(),
        ),
    ];

    for (batch, expected_bytes) in cases {
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.external_payload().unwrap().backend(), "numpy");
        assert_eq!(batch.estimated_bytes().unwrap(), expected_bytes);
    }
}

#[test]
fn static_array_snapshots_preserve_every_current_carrier_family_and_width() {
    let signed = [
        ("int8", i64::from(i8::MIN)),
        ("int16", i64::from(i16::MIN)),
        ("int32", i64::from(i32::MIN)),
        ("int64", i64::MIN),
    ];
    for (dtype, value) in signed {
        let snapshot = Batch::static_array_int("numpy", dtype, vec![1], None, vec![value])
            .unwrap()
            .static_array_snapshot()
            .unwrap();
        assert_eq!(snapshot.dtype(), dtype);
        assert!(matches!(snapshot.values(), StaticArrayValues::Int(values) if values == &[value]));
    }

    let unsigned = [
        ("uint8", u64::from(u8::MAX)),
        ("uint16", u64::from(u16::MAX)),
        ("uint32", u64::from(u32::MAX)),
        ("uint64", u64::MAX),
    ];
    for (dtype, value) in unsigned {
        let snapshot = Batch::static_array_uint("jax", dtype, vec![1], None, vec![value])
            .unwrap()
            .static_array_snapshot()
            .unwrap();
        assert_eq!(snapshot.backend(), "jax");
        assert_eq!(snapshot.dtype(), dtype);
        assert!(matches!(snapshot.values(), StaticArrayValues::Uint(values) if values == &[value]));
    }

    for dtype in ["float32", "float64"] {
        let snapshot = Batch::static_array_float("numpy", dtype, vec![1], None, vec![1.25])
            .unwrap()
            .static_array_snapshot()
            .unwrap();
        assert_eq!(snapshot.dtype(), dtype);
        assert!(matches!(snapshot.values(), StaticArrayValues::Float(values) if values == &[1.25]));
    }

    let boolean = Batch::static_array_bool("numpy", vec![1], None, vec![true])
        .unwrap()
        .static_array_snapshot()
        .unwrap();
    assert!(matches!(boolean.values(), StaticArrayValues::Bool(values) if values == &[true]));
}

#[test]
fn static_array_snapshots_preserve_empty_and_compact_null_semantics() {
    let empty =
        Batch::static_array_float("numpy", "float32", vec![0], Some(Vec::new()), Vec::new())
            .unwrap()
            .static_array_snapshot()
            .unwrap();
    assert_eq!(empty.shape(), &[0]);
    assert_eq!(empty.nulls(), Some([].as_slice()));
    assert!(matches!(empty.values(), StaticArrayValues::Float(values) if values.is_empty()));

    let all_false = Batch::static_array_bool(
        "numpy",
        vec![2],
        Some(vec![false, false]),
        vec![true, false],
    )
    .unwrap()
    .static_array_snapshot()
    .unwrap();
    assert_eq!(all_false.nulls(), Some([false, false].as_slice()));

    let compact = Batch::static_array_int(
        "numpy",
        "int64",
        vec![4],
        Some(vec![false, true, false, true]),
        vec![11, 33],
    )
    .unwrap()
    .static_array_snapshot()
    .unwrap();
    assert_eq!(compact.nulls(), Some([false, true, false, true].as_slice()));
    assert!(matches!(compact.values(), StaticArrayValues::Int(values) if values == &[11, 33]));
}

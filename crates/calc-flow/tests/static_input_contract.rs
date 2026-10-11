use calc_flow::{Batch, STATIC_INPUT_DIGEST_VERSION};

#[test]
fn digest_version_string_is_frozen() {
    assert_eq!(
        STATIC_INPUT_DIGEST_VERSION,
        "calc_flow.static_input.digest.v1"
    );
}

#[test]
fn array_constructors_reject_unsupported_dtypes_and_shape_mismatches() {
    assert!(Batch::static_array_float("numpy", "complex128", vec![1], None, vec![1.0]).is_err());
    assert!(Batch::static_array_int("numpy", "int32", vec![2, 2], None, vec![1, 2, 3]).is_err());
    assert!(Batch::static_array_bool("numpy", vec![2], Some(vec![true]), vec![true]).is_err());
    assert!(Batch::static_array_int("numpy", "int8", vec![1], None, vec![300]).is_err());
    let empty = Batch::static_array_float("numpy", "float64", vec![0], None, vec![]).unwrap();
    assert_eq!(empty.num_rows(), 0);
}

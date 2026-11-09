use super::*;
use datafusion::arrow::datatypes::{Field, Int16Type, UnionFields, UnionMode};
use std::sync::Arc;

fn field(dtype: DataType) -> Arc<Field> {
    Arc::new(Field::new("item", dtype, true))
}
fn assert_cost(dtype: &DataType, len: usize, retained_buffer_headers: u64) {
    let mut cost = None;
    let scan = allocation_counter::measure(|| {
        cost = Some(null_array_cost(dtype, len, &mut || Ok(())).unwrap());
    });
    assert_eq!(scan.count_total, 0);
    let cost = cost.unwrap();
    let mut array = None;
    let allocation = allocation_counter::measure(|| {
        array = Some(
            super::super::materialize::canonical::null_array(
                dtype,
                len,
                "null-profile",
                &mut || Ok(()),
            )
            .unwrap(),
        );
    });
    let headers = retained_buffer_headers * 7 * u64::try_from(size_of::<usize>()).unwrap();
    let normalized = u64::try_from(allocation.bytes_current)
        .unwrap()
        .checked_sub(headers)
        .unwrap();
    assert!(
        cost.logical_owned >= normalized,
        "{dtype:?}[{len}] logical {} normalized {normalized}",
        cost.logical_owned
    );
    assert!(
        cost.construction_peak >= allocation.bytes_max,
        "{dtype:?}[{len}] peak {} actual {}",
        cost.construction_peak,
        allocation.bytes_max
    );
}
#[test]
fn test_logical_null_profiles_match_normalized_retained_primitive_and_byte_buffers() {
    for (dtype, len, headers) in [
        (DataType::Null, 0, 0),
        (DataType::Float64, 0, 1),
        (DataType::Float64, 1, 2),
        (DataType::Boolean, 9, 2),
        (DataType::Utf8, 0, 2),
        (DataType::Utf8, 2, 3),
        (DataType::Utf8View, 1, 2),
        (DataType::FixedSizeBinary(3), 2, 2),
    ] {
        assert_cost(&dtype, len, headers);
    }
}
#[test]
fn test_logical_null_profiles_cover_nested_and_encoded_owner_geometries() {
    let run = DataType::RunEndEncoded(
        Arc::new(Field::new("ends", DataType::Int16, false)),
        field(DataType::Float64),
    );
    let fields = UnionFields::try_new(
        vec![3, 7],
        vec![field(DataType::Float64), field(DataType::Float64)],
    )
    .unwrap();
    let entries = DataType::Struct(
        vec![
            Arc::new(Field::new("key", DataType::Int32, false)),
            field(DataType::Float64),
        ]
        .into(),
    );
    let cases = vec![
        (DataType::List(field(DataType::Float64)), 1, 3),
        (DataType::ListView(field(DataType::Float64)), 1, 4),
        (DataType::FixedSizeList(field(DataType::Float64), 3), 2, 3),
        (DataType::FixedSizeList(field(DataType::Float64), 0), 2, 2),
        (
            DataType::Struct(vec![field(DataType::Int64), field(DataType::UInt8)].into()),
            2,
            5,
        ),
        (
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Float64)),
            2,
            3,
        ),
        (run.clone(), 0, 2),
        (run, 1, 3),
        (DataType::Union(fields.clone(), UnionMode::Dense), 1, 5),
        (DataType::Union(fields, UnionMode::Sparse), 1, 5),
        (
            DataType::Map(Arc::new(Field::new("entries", entries, false)), false),
            1,
            4,
        ),
        (
            DataType::Union(UnionFields::empty(), UnionMode::Sparse),
            0,
            1,
        ),
        (
            DataType::List(field(DataType::Union(
                UnionFields::empty(),
                UnionMode::Dense,
            ))),
            1,
            4,
        ),
    ];
    for (dtype, len, headers) in cases {
        assert_cost(&dtype, len, headers);
    }
}
#[test]
fn test_null_profile_keeps_constructor_specific_run_end_and_zero_length_capacities() {
    use datafusion::arrow::array::RunArray;
    let dtype = DataType::RunEndEncoded(
        Arc::new(Field::new("ends", DataType::Int16, false)),
        field(DataType::Float64),
    );
    for (len, capacity) in [(0, 0), (1, 64)] {
        let array = super::super::materialize::canonical::null_array(
            &dtype,
            len,
            "run-capacity",
            &mut || Ok(()),
        )
        .unwrap();
        let run = array
            .as_any()
            .downcast_ref::<RunArray<Int16Type>>()
            .unwrap();
        assert_eq!(run.run_ends().inner().inner().capacity(), capacity);
    }
    assert!(null_array_cost(&dtype, 32768, &mut || Ok(())).is_err());
    let error = null_array_cost(&DataType::Int32, 1, &mut || {
        Err(CalcFlowError::Cancelled {
            run_id: "profile".into(),
        })
    })
    .err()
    .unwrap();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
}

#[test]
fn test_null_array_cost_rejects_unrepresentable_individual_backing_layout() {
    let dtype = DataType::FixedSizeList(
        Arc::new(Field::new(
            "outer",
            DataType::FixedSizeList(
                Arc::new(Field::new("inner", DataType::Float64, true)),
                800_000_000,
            ),
            true,
        )),
        2_147_483_647,
    );
    let raw = 2_147_483_647_u128 * 800_000_000 * 8;
    assert!(raw > u128::try_from(isize::MAX).unwrap());
    let cost = null_array_cost(&dtype, 1, &mut || Ok(()));
    assert!(
        cost.is_err(),
        "an individual allocation larger than Layout permits must be rejected"
    );
    assert!(
        super::super::materialize::canonical::null_array(&dtype, 1, "layout", &mut || Ok(()))
            .is_err()
    );
    let empty = null_array_cost(&dtype, 0, &mut || Ok(())).unwrap();
    assert!(empty.construction_peak > 0);
    let array =
        super::super::materialize::canonical::null_array(&dtype, 0, "layout", &mut || Ok(()))
            .unwrap();
    assert_eq!(array.len(), 0);
}

#[test]
fn test_null_array_cost_keeps_representable_multi_allocation_totals() {
    let rows = usize::try_from(isize::MAX).unwrap() / 16;
    assert!(std::alloc::Layout::from_size_align(rows * 8, 64).is_ok());
    let fields = ["a", "b"].map(|name| Arc::new(Field::new(name, DataType::Float64, true)));
    let dtype = DataType::Struct(fields.into());
    let mut result = None;
    let scan = allocation_counter::measure(|| {
        result = Some(null_array_cost(&dtype, rows, &mut || Ok(())).unwrap());
    });
    assert_eq!(scan.count_total, 0);
    assert!(result.unwrap().construction_peak > u64::try_from(isize::MAX).unwrap());
}

#[test]
fn test_null_array_cost_uses_native_arrow_allocation_alignment() {
    let raw = usize::try_from(isize::MAX).unwrap() / 64 * 64;
    assert!(std::alloc::Layout::from_size_align(raw, 64).is_ok());
    if std::alloc::Layout::from_size_align(raw, datafusion::arrow::alloc::ALIGNMENT).is_err() {
        assert!(null_array_cost(&DataType::Float64, raw / 8, &mut || Ok(())).is_err());
    }
}

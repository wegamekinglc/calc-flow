use super::{GroupId, KeySchema, KeyStore, hash_calls, reset_hash_calls};
use crate::{DataFusionConfig, DataFusionRuntime};
use datafusion::arrow::{
    array::{ArrayRef, StringArray},
    datatypes::{DataType, Field},
};
use std::sync::Arc;

fn store(runtime: &DataFusionRuntime) -> KeyStore {
    KeyStore::new(
        KeySchema::groupable(vec![Arc::new(Field::new("key", DataType::Utf8, false))]).unwrap(),
        runtime.incremental_reservation("key-route-cache-test"),
    )
}

fn strings(values: &[&str]) -> ArrayRef {
    Arc::new(StringArray::from(values.to_vec()))
}

fn cached(keys: &mut KeyStore, array: &ArrayRef) -> Vec<GroupId> {
    keys.route_with_limit_cached(std::slice::from_ref(array), usize::MAX, &mut || Ok(()))
        .unwrap()
}

#[test]
fn default_route_does_not_cache_input_runs() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let mut keys = store(&runtime);
    reset_hash_calls();
    let first = strings(&["A", "B", "A"]);
    keys.route(&[first], &mut || Ok(())).unwrap();
    let repeated = strings(&["A", "B", "A"]);
    keys.route(&[repeated], &mut || Ok(())).unwrap();
    assert_eq!(hash_calls(), 2);
}

#[test]
fn repeated_key_run_reuses_groups_without_rehashing() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let mut keys = store(&runtime);
    reset_hash_calls();
    let first = strings(&["A", "B", "A"]);
    let expected = vec![
        GroupId::new(0).unwrap(),
        GroupId::new(1).unwrap(),
        GroupId::new(0).unwrap(),
    ];
    assert_eq!(cached(&mut keys, &first), expected);
    assert_eq!(hash_calls(), 1);

    let repeated = strings(&["A", "B", "A"]);
    assert_eq!(cached(&mut keys, &repeated), expected);
    assert_eq!(hash_calls(), 1);

    let different = strings(&["B", "A", "B"]);
    assert_eq!(
        cached(&mut keys, &different),
        vec![
            GroupId::new(1).unwrap(),
            GroupId::new(0).unwrap(),
            GroupId::new(1).unwrap(),
        ]
    );
    assert_eq!(hash_calls(), 2);
    assert_eq!(cached(&mut keys, &first), expected);
    assert_eq!(hash_calls(), 3);
}

#[test]
fn rollback_invalidates_repeated_key_run() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let mut keys = store(&runtime);
    let first = strings(&["A"]);
    keys.route(&[first], &mut || Ok(())).unwrap();
    let cut = keys.cut();
    let added = strings(&["B"]);
    cached(&mut keys, &added);
    keys.rollback(cut);
    let error = keys
        .route_with_limit_cached(&[strings(&["B"])], 0, &mut || Ok(()))
        .unwrap_err();
    assert!(error.to_string().contains("budget"));
}

#[test]
fn test_cancelled_key_encoding_allocates_only_the_first_page() {
    use datafusion::arrow::array::LargeStringArray;

    let value = "x".repeat(2 << 20);
    let narrow: ArrayRef = Arc::new(StringArray::from(vec![value.as_str()]));
    let wide: ArrayRef = Arc::new(LargeStringArray::from(vec![value.as_str()]));
    for array in [narrow, wide] {
        let mut stop = || {
            Err(crate::CalcFlowError::Cancelled {
                run_id: "keys".into(),
            })
        };
        let mut work = super::VariableWork::new(&mut stop);
        let allocated = allocation_counter::measure(|| {
            let result = super::canonical_row(std::slice::from_ref(&array), 0, &mut work);
            assert!(matches!(
                result,
                Err(crate::CalcFlowError::Cancelled { .. })
            ));
        });
        eprintln!(
            "cancelled {:?} canonical key: {allocated:?}",
            array.data_type()
        );
        assert!(allocated.bytes_total < 2 * 65_536);
    }
}

#[test]
fn test_sliced_string_keys_keep_their_canonical_encoding() {
    use datafusion::arrow::array::{Int16Array, LargeStringArray};
    let values = vec![Some("unused"), None, Some(""), Some("猫\0")];
    let narrow: ArrayRef = Arc::new(StringArray::from(values.clone()));
    let wide: ArrayRef = Arc::new(LargeStringArray::from(values));
    let other: ArrayRef = Arc::new(Int16Array::from(vec![-2; 3]));
    let arrays = [narrow.slice(1, 3), wide.slice(1, 3), other];
    for (row, string) in [None, Some(""), Some("猫\0")].into_iter().enumerate() {
        let mut expected = Vec::new();
        for tag in [10, 11] {
            expected.extend_from_slice(&[tag, u8::from(string.is_none())]);
            if let Some(string) = string {
                expected.extend_from_slice(&(string.len() as u64).to_le_bytes());
                expected.extend_from_slice(string.as_bytes());
            }
        }
        expected.extend_from_slice(&[3, 0, 0xfe, 0xff]);
        let actual =
            super::canonical_row(&arrays, row, &mut super::VariableWork::new(&mut || Ok(())))
                .unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn test_routing_distinct_fixed_keys_avoids_per_hash_allocations() {
    use datafusion::arrow::array::UInt32Array;

    const ROWS: u32 = 1024;
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let mut keys = KeyStore::new(
        KeySchema::groupable(vec![Arc::new(Field::new("key", DataType::UInt32, false))]).unwrap(),
        runtime.incremental_reservation("key-route-allocation-test"),
    );
    let arrays: [ArrayRef; 1] = [Arc::new(UInt32Array::from_iter_values(0..ROWS))];
    let rows = usize::try_from(ROWS).unwrap();
    let mut ids = Vec::new();
    let allocated = allocation_counter::measure(|| {
        ids = keys
            .route_with_limit(&arrays, rows, &mut || Ok(()))
            .unwrap();
    });
    assert_eq!(
        ids.iter().map(|id| id.index()).collect::<Vec<_>>(),
        (0..rows).collect::<Vec<_>>()
    );
    assert_eq!(
        keys.probe(&arrays, &mut || Ok(())).unwrap(),
        ids.into_iter().map(Some).collect::<Vec<_>>()
    );
    eprintln!("routing {rows} fixed keys: {allocated:?}");
    assert!(
        allocated.count_total < u64::from(ROWS) * 5 / 2,
        "{allocated:?}"
    );
}

#[test]
fn test_equal_hashes_remain_distinct_across_growth_and_rollback() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let mut keys = store(&runtime);
    let values = (0..48).map(|row| format!("key-{row}")).collect::<Vec<_>>();
    let input: ArrayRef = Arc::new(StringArray::from_iter_values(values.iter()));
    let route = |keys: &mut KeyStore, start, length, limit| {
        keys.route_rows(
            &[input.slice(start, length)],
            None,
            limit,
            false,
            vec![7; length],
            &mut || Ok(()),
        )
    };
    let empty = keys.cut();
    let prefix = route(&mut keys, 0, 16, 48).unwrap();
    let cut = keys.cut();
    let added = route(&mut keys, 16, 32, 48).unwrap();
    assert_eq!(
        prefix.iter().map(|id| id.index()).collect::<Vec<_>>(),
        (0..16).collect::<Vec<_>>()
    );
    assert_eq!(
        added.iter().map(|id| id.index()).collect::<Vec<_>>(),
        (16..48).collect::<Vec<_>>()
    );
    keys.rollback(cut);
    assert_eq!(route(&mut keys, 0, 16, 16).unwrap(), prefix);
    assert!(route(&mut keys, 16, 1, 16).is_err());
    assert_eq!(route(&mut keys, 16, 32, 48).unwrap(), added);
    keys.rollback(empty);
    assert_eq!(route(&mut keys, 0, 16, 16).unwrap(), prefix);
}

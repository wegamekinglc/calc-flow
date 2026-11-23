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
}

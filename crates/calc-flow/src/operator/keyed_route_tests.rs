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

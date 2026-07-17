use std::sync::Arc;

use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, DataFusionConfig, DataFusionRuntime, UdfKind,
    UdfReference, UdfRegistry,
};
use datafusion::{
    arrow::{array::Int64Array, datatypes::DataType, record_batch::RecordBatch},
    common::ScalarValue,
    logical_expr::{ColumnarValue, ScalarUDF, Volatility, create_udf},
};
use serde_json::json;

fn constant_udf(name: &str, value: i64) -> Arc<ScalarUDF> {
    Arc::new(create_udf(
        name,
        vec![],
        DataType::Int64,
        Volatility::Immutable,
        Arc::new(move |_| Ok(ColumnarValue::Scalar(ScalarValue::Int64(Some(value))))),
    ))
}

fn constant_udf_with_aliases(
    name: &str,
    value: i64,
    aliases: impl IntoIterator<Item = &'static str>,
) -> Arc<ScalarUDF> {
    Arc::new(
        constant_udf(name, value)
            .as_ref()
            .clone()
            .with_aliases(aliases),
    )
}

fn input() -> Batch {
    let record =
        RecordBatch::try_from_iter(vec![("a", Arc::new(Int64Array::from(vec![1, 2])) as _)])
            .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

#[test]
fn snapshot_is_immutable_and_catalog_is_data_only() {
    let mut registry = UdfRegistry::new();
    registry
        .register_external(
            UdfReference::new("python", "normalize", "1", UdfKind::ExternalScalar).unwrap(),
            1,
        )
        .unwrap();
    let snapshot = registry.snapshot();
    registry
        .register_external(
            UdfReference::new("numpy", "clip", "1", UdfKind::ExternalArray).unwrap(),
            3,
        )
        .unwrap();

    assert_eq!(snapshot.catalog().len(), 1);
    assert_eq!(snapshot.catalog()[0].name, "normalize");
}

#[test]
fn udf_reference_deserialization_enforces_portable_identifiers() {
    for value in [
        json!({
            "provider": "",
            "name": "score",
            "version": "1",
            "kind": "data_fusion_scalar"
        }),
        json!({
            "provider": "rust",
            "name": "score/value",
            "version": "1",
            "kind": "data_fusion_scalar"
        }),
        json!({
            "provider": "rust",
            "name": "score",
            "version": "latest version",
            "kind": "data_fusion_scalar"
        }),
    ] {
        assert!(serde_json::from_value::<UdfReference>(value).is_err());
    }
}

#[test]
fn udf_reference_deserialization_rejects_unknown_and_executable_fields() {
    for field in ["callable", "source", "import_path", "unexpected"] {
        let mut value = json!({
            "provider": "rust",
            "name": "score",
            "version": "1",
            "kind": "data_fusion_scalar"
        });
        value
            .as_object_mut()
            .unwrap()
            .insert(field.into(), json!("not allowed"));

        assert!(serde_json::from_value::<UdfReference>(value).is_err());
    }
}

#[test]
fn udf_reference_serialization_and_accessors_preserve_the_data_shape() {
    let reference = UdfReference::new("rust", "score", "1.2", UdfKind::DataFusionScalar).unwrap();

    assert_eq!(reference.provider(), "rust");
    assert_eq!(reference.name(), "score");
    assert_eq!(reference.version(), "1.2");
    assert_eq!(reference.kind(), UdfKind::DataFusionScalar);
    assert_eq!(
        serde_json::to_value(reference).unwrap(),
        json!({
            "provider": "rust",
            "name": "score",
            "version": "1.2",
            "kind": "data_fusion_scalar"
        })
    );
}

#[test]
fn udf_reference_json_schema_preserves_the_strict_data_shape() {
    let schema = serde_json::to_value(schemars::schema_for!(UdfReference)).unwrap();
    let properties = schema["properties"].as_object().unwrap();

    assert_eq!(properties.len(), 4);
    for field in ["provider", "name", "version", "kind"] {
        assert!(properties.contains_key(field));
    }
    assert_eq!(schema["additionalProperties"], json!(false));
}

#[test]
fn conflicting_versions_of_a_datafusion_name_are_rejected() {
    let references = [
        UdfReference::new("rust", "score", "1", UdfKind::DataFusionScalar).unwrap(),
        UdfReference::new("rust", "score", "2", UdfKind::DataFusionScalar).unwrap(),
    ];

    assert!(calc_flow::validate_selected_udfs(&references).is_err());
}

#[test]
fn distinct_native_references_with_the_same_sql_name_are_rejected() {
    let rust = UdfReference::new("rust", "score", "1", UdfKind::DataFusionScalar).unwrap();
    let plugin = UdfReference::new("plugin", "score", "1", UdfKind::DataFusionScalar).unwrap();

    let error = calc_flow::validate_selected_udfs(&[rust.clone(), plugin.clone()]).unwrap_err();
    let message = error.to_string();

    assert!(message.contains("score"));
    assert!(message.contains("rust:score@1"));
    assert!(message.contains("plugin:score@1"));
}

#[tokio::test]
async fn runtime_registers_and_executes_only_selected_native_references() {
    let selected =
        UdfReference::new("rust", "selected_value", "1", UdfKind::DataFusionScalar).unwrap();
    let unselected =
        UdfReference::new("rust", "unselected_value", "2", UdfKind::DataFusionScalar).unwrap();
    let mut registry = UdfRegistry::new();
    registry
        .register_datafusion(selected.clone(), constant_udf("selected_value", 11), 0)
        .unwrap();
    registry
        .register_datafusion(unselected, constant_udf("unselected_value", 22), 0)
        .unwrap();
    let snapshot = registry.snapshot();
    let mut runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();

    runtime
        .register_udfs(&snapshot, std::slice::from_ref(&selected))
        .unwrap();

    let output = runtime
        .evaluate("selected_value()", &input(), None)
        .await
        .unwrap();
    let result = output.table_payload().unwrap().batches()[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(result.values(), &[11, 11]);
    assert!(
        runtime
            .evaluate("unselected_value()", &input(), None)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn runtime_registers_a_native_udf_after_the_lazy_session_exists() {
    let selected = UdfReference::new("rust", "late_value", "1", UdfKind::DataFusionScalar).unwrap();
    let mut registry = UdfRegistry::new();
    registry
        .register_datafusion(selected.clone(), constant_udf("late_value", 17), 0)
        .unwrap();
    let mut runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();

    runtime.evaluate("a", &input(), None).await.unwrap();
    runtime
        .register_udfs(&registry.snapshot(), &[selected])
        .unwrap();

    let output = runtime
        .evaluate("late_value()", &input(), None)
        .await
        .unwrap();
    let values = output.table_payload().unwrap().batches()[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(values.values(), &[17, 17]);
}

#[tokio::test]
async fn runtime_rejects_alias_to_primary_collisions_before_registration() {
    let first = UdfReference::new("rust", "first", "1", UdfKind::DataFusionScalar).unwrap();
    let second = UdfReference::new("rust", "second", "1", UdfKind::DataFusionScalar).unwrap();
    let mut registry = UdfRegistry::new();
    registry
        .register_datafusion(
            first.clone(),
            constant_udf_with_aliases("first", 11, ["second"]),
            0,
        )
        .unwrap();
    registry
        .register_datafusion(second.clone(), constant_udf("second", 22), 0)
        .unwrap();
    let mut runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();

    let error = runtime
        .register_udfs(&registry.snapshot(), &[first.clone(), second.clone()])
        .unwrap_err();
    let message = error.to_string();

    assert!(message.contains("second"));
    assert!(message.contains("rust:first@1"));
    assert!(message.contains("rust:second@1"));
    assert!(runtime.evaluate("first()", &input(), None).await.is_err());
}

#[tokio::test]
async fn runtime_rejects_alias_to_alias_collisions_before_registration() {
    let first = UdfReference::new("rust", "first", "1", UdfKind::DataFusionScalar).unwrap();
    let second = UdfReference::new("rust", "second", "1", UdfKind::DataFusionScalar).unwrap();
    let mut registry = UdfRegistry::new();
    registry
        .register_datafusion(
            first.clone(),
            constant_udf_with_aliases("first", 11, ["shared"]),
            0,
        )
        .unwrap();
    registry
        .register_datafusion(
            second.clone(),
            constant_udf_with_aliases("second", 22, ["shared"]),
            0,
        )
        .unwrap();
    let mut runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();

    let error = runtime
        .register_udfs(&registry.snapshot(), &[first.clone(), second.clone()])
        .unwrap_err();
    let message = error.to_string();

    assert!(message.contains("shared"));
    assert!(message.contains("rust:first@1"));
    assert!(message.contains("rust:second@1"));
    assert!(runtime.evaluate("first()", &input(), None).await.is_err());
}

#[test]
fn runtime_rejects_an_unknown_native_version_before_registration() {
    let registered = UdfReference::new("rust", "score", "1", UdfKind::DataFusionScalar).unwrap();
    let unknown = UdfReference::new("rust", "score", "2", UdfKind::DataFusionScalar).unwrap();
    let mut registry = UdfRegistry::new();
    registry
        .register_datafusion(registered, constant_udf("score", 11), 0)
        .unwrap();
    let snapshot = registry.snapshot();
    let mut runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();

    assert!(matches!(
        runtime.register_udfs(&snapshot, &[unknown]),
        Err(CalcFlowError::Compile { message })
            if message.contains("unknown UDF rust:score@2")
    ));
}

#[tokio::test]
async fn runtime_resolves_all_selected_udfs_before_registering_any() {
    let known = UdfReference::new("rust", "known", "1", UdfKind::DataFusionScalar).unwrap();
    let unknown = UdfReference::new("rust", "unknown", "1", UdfKind::DataFusionScalar).unwrap();
    let mut registry = UdfRegistry::new();
    registry
        .register_datafusion(known.clone(), constant_udf("known", 11), 0)
        .unwrap();
    let mut runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();

    assert!(
        runtime
            .register_udfs(&registry.snapshot(), &[known, unknown])
            .is_err()
    );
    assert!(runtime.evaluate("known()", &input(), None).await.is_err());
}

#[tokio::test]
async fn runtime_deduplicates_exact_selected_references() {
    let reference =
        UdfReference::new("rust", "selected_value", "1", UdfKind::DataFusionScalar).unwrap();
    let mut registry = UdfRegistry::new();
    registry
        .register_datafusion(reference.clone(), constant_udf("selected_value", 11), 0)
        .unwrap();
    let mut runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();

    runtime
        .register_udfs(&registry.snapshot(), &[reference.clone(), reference])
        .unwrap();

    assert!(
        runtime
            .evaluate("selected_value()", &input(), None)
            .await
            .is_ok()
    );
}

#[test]
fn snapshot_native_map_is_isolated_from_later_registry_mutation() {
    let original = UdfReference::new("rust", "first", "1", UdfKind::DataFusionScalar).unwrap();
    let added_later = UdfReference::new("rust", "second", "1", UdfKind::DataFusionScalar).unwrap();
    let mut registry = UdfRegistry::new();
    registry
        .register_datafusion(original.clone(), constant_udf("first", 11), 0)
        .unwrap();
    let snapshot = registry.snapshot();

    registry
        .register_datafusion(added_later.clone(), constant_udf("second", 22), 0)
        .unwrap();

    assert!(snapshot.resolve_native(&original).is_ok());
    assert!(snapshot.resolve_native(&added_later).is_err());
}

#[test]
fn native_registration_rejects_an_exact_duplicate_without_replacement() {
    let reference = UdfReference::new("rust", "score", "1", UdfKind::DataFusionScalar).unwrap();
    let original = constant_udf("score", 11);
    let mut registry = UdfRegistry::new();
    registry
        .register_datafusion(reference.clone(), Arc::clone(&original), 0)
        .unwrap();

    assert!(
        registry
            .register_datafusion(reference.clone(), constant_udf("score", 22), 0)
            .is_err()
    );
    let resolved = registry.snapshot().resolve_native(&reference).unwrap();
    assert!(Arc::ptr_eq(&resolved, &original));
}

#[test]
fn native_registration_rejects_wrong_kind_without_catalog_mutation() {
    let reference = UdfReference::new("rust", "score", "1", UdfKind::ExternalScalar).unwrap();
    let mut registry = UdfRegistry::new();

    assert!(
        registry
            .register_datafusion(reference.clone(), constant_udf("score", 11), 0)
            .is_err()
    );
    registry.register_external(reference, 0).unwrap();
    assert_eq!(registry.snapshot().catalog().len(), 1);
}

#[test]
fn native_registration_rejects_name_mismatch_without_catalog_mutation() {
    let reference = UdfReference::new("rust", "score", "1", UdfKind::DataFusionScalar).unwrap();
    let mut registry = UdfRegistry::new();

    assert!(
        registry
            .register_datafusion(reference.clone(), constant_udf("other", 11), 0)
            .is_err()
    );
    registry
        .register_datafusion(reference, constant_udf("score", 22), 0)
        .unwrap();
    assert_eq!(registry.snapshot().catalog().len(), 1);
}

#[test]
fn serialized_catalog_contains_metadata_only() {
    let reference = UdfReference::new("rust", "score", "1", UdfKind::DataFusionScalar).unwrap();
    let mut registry = UdfRegistry::new();
    registry
        .register_datafusion(reference, constant_udf("score", 11), 1)
        .unwrap();

    let value = serde_json::to_value(registry.snapshot().catalog()).unwrap();

    assert_eq!(
        value,
        json!([{
            "provider": "rust",
            "name": "score",
            "version": "1",
            "kind": "data_fusion_scalar",
            "argument_count": 1
        }])
    );
}

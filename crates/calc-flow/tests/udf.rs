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
fn conflicting_versions_of_a_datafusion_name_are_rejected() {
    let references = [
        UdfReference::new("rust", "score", "1", UdfKind::DataFusionScalar).unwrap(),
        UdfReference::new("rust", "score", "2", UdfKind::DataFusionScalar).unwrap(),
    ];

    assert!(calc_flow::validate_selected_udfs(&references).is_err());
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

use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{Batch, BatchKind, BatchMetadata, CalcFlowError, ExternalPayload};
use datafusion::arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use serde_json::json;

#[derive(Debug)]
struct TestArray;

impl ExternalPayload for TestArray {
    fn backend(&self) -> &'static str {
        "test"
    }

    fn len(&self) -> usize {
        2
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
struct EmptyBackendArray;

impl ExternalPayload for EmptyBackendArray {
    fn backend(&self) -> &'static str {
        ""
    }

    fn len(&self) -> usize {
        0
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn assert_invalid_argument(error: CalcFlowError, expected_field: &str, expected_message: &str) {
    match error {
        CalcFlowError::InvalidArgument { field, message } => {
            assert_eq!(field, expected_field);
            assert_eq!(message, expected_message);
        }
        other => panic!("expected invalid argument, got {other}"),
    }
}

#[test]
fn table_batch_preserves_metadata_and_rows() {
    let record =
        RecordBatch::try_from_iter(vec![("value", Arc::new(Int64Array::from(vec![1, 2])) as _)])
            .unwrap();
    let metadata = BatchMetadata::new(
        "source",
        7,
        BTreeMap::from([("nested".into(), json!({"ok": true}))]),
    )
    .unwrap();
    let batch = Batch::table(vec![record], metadata.clone()).unwrap();

    assert_eq!(batch.kind(), BatchKind::Table);
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.metadata(), &metadata);
}

#[test]
fn external_batch_is_owned_by_arc() {
    let batch = Batch::external(Arc::new(TestArray), BatchMetadata::default()).unwrap();

    assert_eq!(batch.kind(), BatchKind::Array);
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.external_payload().unwrap().backend(), "test");
}

#[test]
fn zero_row_record_batch_is_an_empty_table_with_its_schema() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int64,
        false,
    )]));
    let record = RecordBatch::new_empty(Arc::clone(&schema));

    let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();

    assert_eq!(batch.kind(), BatchKind::Table);
    assert_eq!(batch.num_rows(), 0);
    assert_eq!(batch.table_payload().unwrap().schema(), &schema);
}

#[test]
fn table_batch_rejects_an_empty_record_batch_collection() {
    let error = Batch::table(Vec::new(), BatchMetadata::default()).unwrap_err();

    assert_invalid_argument(
        error,
        "batches",
        "must contain at least one RecordBatch; represent an empty table with one zero-row batch",
    );
}

#[test]
fn table_batch_rejects_mismatched_schemas() {
    let int_batch = RecordBatch::new_empty(Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int64,
        false,
    )])));
    let string_batch = RecordBatch::new_empty(Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Utf8,
        false,
    )])));

    let error = Batch::table(vec![int_batch, string_batch], BatchMetadata::default()).unwrap_err();

    assert_invalid_argument(error, "batches", "schemas must match");
}

#[test]
fn metadata_rejects_a_source_containing_nul() {
    let error = BatchMetadata::new("source\0name", 0, BTreeMap::new()).unwrap_err();

    assert_invalid_argument(error, "metadata.source", "must not contain NUL");
}

#[test]
fn external_batch_rejects_an_empty_backend_identifier() {
    let error = Batch::external(Arc::new(EmptyBackendArray), BatchMetadata::default()).unwrap_err();

    assert_invalid_argument(error, "backend", "must not be empty");
}

#[test]
fn payload_access_rejects_the_wrong_batch_kind() {
    let table = RecordBatch::new_empty(Arc::new(Schema::empty()));
    let table_batch = Batch::table(vec![table], BatchMetadata::default()).unwrap();
    let external_batch = Batch::external(Arc::new(TestArray), BatchMetadata::default()).unwrap();

    assert_invalid_argument(
        table_batch.external_payload().unwrap_err(),
        "batch",
        "expected array batch",
    );
    assert_invalid_argument(
        external_batch.table_payload().unwrap_err(),
        "batch",
        "expected table batch",
    );
}

#[test]
fn external_payload_supports_downcasting_to_its_concrete_type() {
    let batch = Batch::external(Arc::new(TestArray), BatchMetadata::default()).unwrap();

    assert!(
        batch
            .external_payload()
            .unwrap()
            .as_any()
            .downcast_ref::<TestArray>()
            .is_some()
    );
}

#[test]
fn with_metadata_returns_a_new_envelope_sharing_the_external_payload() {
    let original_metadata =
        BatchMetadata::new("original", 1, BTreeMap::from([("id".into(), json!(1))])).unwrap();
    let replacement_metadata =
        BatchMetadata::new("replacement", 2, BTreeMap::from([("id".into(), json!(2))])).unwrap();
    let batch = Batch::external(Arc::new(TestArray), original_metadata.clone()).unwrap();

    let updated = batch.with_metadata(replacement_metadata.clone());

    assert_eq!(batch.metadata(), &original_metadata);
    assert_eq!(updated.metadata(), &replacement_metadata);
    assert!(Arc::ptr_eq(
        batch.external_payload().unwrap(),
        updated.external_payload().unwrap()
    ));
}

use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{Batch, BatchKind, BatchMetadata, ExternalPayload};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
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

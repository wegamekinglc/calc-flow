#![cfg(feature = "benchmark-internals")]

use calc_flow::benchmark_internals::DispatchTraceOwner;
use calc_flow::{Batch, StreamMessage};
use datafusion::arrow::{array::UInt64Array, record_batch::RecordBatch};
use std::sync::Arc;

fn batch(values: Vec<u64>) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_from_iter([(
                "sequence",
                Arc::new(UInt64Array::from(values)) as datafusion::arrow::array::ArrayRef,
            )])
            .unwrap(),
        ],
        calc_flow::BatchMetadata::default(),
    )
    .unwrap()
}

#[test]
fn test_trace_records_actual_data_geometry_and_completed_callbacks() {
    let owner = DispatchTraceOwner::default();
    let message = StreamMessage::data(batch(vec![5, 6, 7]));
    owner.record("asof", "left", &message).unwrap();
    assert!(owner.take().is_err());
    owner.start(23, "compiled", "asof", false).unwrap();
    owner.record("other", "left", &message).unwrap();
    owner.record("asof", "left", &message).unwrap();
    owner.callback("asof", "data").unwrap();
    let trace = owner.take().unwrap();
    assert_eq!(trace["job_id"], 23);
    assert_eq!(trace["record_count"], 1);
    assert_eq!(trace["capacity"], 8192);
    assert_eq!(
        trace["events"][0],
        serde_json::json!({
            "kind":"data", "ingress":"left", "ordinal":0, "rows":3,
            "first_sequence":5, "last_sequence":7, "record_batch_rows":[3]
        })
    );
    assert_eq!(trace["callback_counts"]["data"], 1);
    assert!(owner.take().is_err());
    assert!(owner.start(23, "compiled", "asof", false).is_err());
}

#[test]
fn test_trace_rejects_overflow_and_unsupported_record_batch_geometry() {
    let message = StreamMessage::data(batch(vec![0]));
    let owner = DispatchTraceOwner::default();
    owner.start(1, "compiled", "asof", false).unwrap();
    for _ in 0..8192 {
        owner.record("asof", "left", &message).unwrap();
    }
    assert!(owner.record("asof", "left", &message).is_err());
    assert!(owner.take().is_err());
    let owner = DispatchTraceOwner::default();
    owner.start(1, "compiled", "asof", false).unwrap();
    let one = batch(vec![0]);
    let records = vec![one.table_payload().unwrap().batches()[0].clone(); 5];
    let message =
        StreamMessage::data(Batch::table(records, calc_flow::BatchMetadata::default()).unwrap());
    assert!(owner.record("asof", "left", &message).is_err());
    assert!(owner.take().is_err());
}

#[test]
fn test_trace_requires_unique_nonempty_completion_nodes() {
    for nodes in [
        vec![],
        vec![String::new()],
        vec!["output".into(), "output".into()],
    ] {
        let owner = DispatchTraceOwner::default();
        owner.start(1, "compiled", "asof", false).unwrap();
        assert!(owner.register_nodes(&nodes).is_err());
    }
}

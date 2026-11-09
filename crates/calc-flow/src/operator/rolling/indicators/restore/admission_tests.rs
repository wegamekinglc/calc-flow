use crate::operator::rolling::{RollingOperator, RollingSpec};
use crate::pipeline::OperatorCheckpointCapability;
use crate::{
    CancellationToken, JsonMap, OperatorManifestEntry, OperatorStateSnapshot, StateBudget,
    StreamJobContext,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use serde_json::json;
use std::{collections::BTreeMap, sync::Arc};
fn operator() -> RollingOperator {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
    ]));
    let spec:RollingSpec=serde_json::from_value(json!({"configuration_version":1,"state_layout_version":5,"partition_by":["seq"],"event_time":"time","sequence_by":["seq"],"outputs":[{"kind":"rsi","primitive_version":1,"input":"price","output":"rsi","period":2}],"allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},"value_policy":"stateful_numeric_v1"})).unwrap();
    let mut operator = RollingOperator::new("features", schema, spec).unwrap();
    operator
        .set_state_budget(StateBudget::new(32, 64 << 10).unwrap())
        .unwrap();
    operator
}
#[test]
fn test_actual_managed_rolling_admission_rejects_large_metadata_before_copy() {
    let operator = operator();
    let graph = "f".repeat(64);
    let job = StreamJobContext::new(41, &graph, JsonMap::new(), None, CancellationToken::new());
    let capability = OperatorCheckpointCapability::CheckpointedStateful { state_version: 5 };
    let record = operator
        .native_checkpoint_record(
            "features",
            graph.clone().into(),
            capability,
            Arc::new(job.native_control()),
        )
        .unwrap();
    let mut snapshot = OperatorStateSnapshot::default();
    snapshot
        .inline_metadata
        .insert("unknown".into(), json!("private-canary".repeat(80_000)));
    let snapshot = capability.encode_snapshot("features", snapshot).unwrap();
    let entry = OperatorManifestEntry {
        progress: BTreeMap::new(),
        inline_metadata: snapshot.inline_metadata,
        segments: vec![],
    };
    let mut error = None;
    let actual = allocation_counter::measure(|| {
        error = record.admit_entry(&entry).err();
    });
    let error = error.unwrap();
    assert!(!error.to_string().contains("private-canary"));
    assert!(
        actual.bytes_max <= 64 << 10,
        "unfunded managed metadata copy: {actual:?}"
    );
}

#[test]
fn test_rolling_admission_rejects_unexpected_large_frontier_before_snapshot_copy() {
    let operator = super::state_tests::operator();
    let mut snapshot = super::state_tests::snapshot(&operator);
    let handle =
        serde_json::from_value(snapshot.inline_metadata["segment_inventory"][0]["handle"].clone())
            .unwrap();
    snapshot.inline_metadata.insert(
        crate::pipeline::OUTPUT_FRONTIER_METADATA_KEY_V1.into(),
        json!("private-canary".repeat(80_000)),
    );
    let capability = OperatorCheckpointCapability::CheckpointedStateful { state_version: 5 };
    let snapshot = capability.encode_snapshot("features", snapshot).unwrap();
    let job = StreamJobContext::new(
        41,
        "f".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let record = operator
        .native_checkpoint_record(
            "features",
            job.fingerprint().into(),
            capability,
            Arc::new(job.native_control()),
        )
        .unwrap();
    let entry = OperatorManifestEntry {
        progress: BTreeMap::new(),
        inline_metadata: snapshot.inline_metadata,
        segments: vec![handle],
    };
    let mut result = None;
    let actual = allocation_counter::measure(|| {
        result = Some(record.admit_entry(&entry));
    });
    let error = result
        .unwrap()
        .expect_err("rolling state cannot carry a runtime output frontier");
    assert!(!error.to_string().contains("private-canary"));
    assert!(
        actual.bytes_max <= 64 << 10,
        "unexpected frontier copied before rejection: {actual:?}"
    );
}

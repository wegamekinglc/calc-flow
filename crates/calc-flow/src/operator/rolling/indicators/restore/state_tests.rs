use crate::operator::rolling::{
    BufferedRow, KeyValue, RollingHistories, RollingOperator, RollingSnapshotMetadata, RollingSpec,
    compute_output_columns, state_v3,
};
use crate::state::{SegmentDescriptor, SegmentKind};
use crate::{Epoch, OperatorStateSnapshot, StateHandle, StateSegment, StreamOperator};
use datafusion::{
    arrow::datatypes::{DataType, Field, Schema, TimeUnit},
    common::ScalarValue,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, sync::Arc};
const GRAPH: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
pub(super) fn operator() -> RollingOperator {
    let schema = Arc::new(Schema::new(vec![
        Field::new("entity", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("seq", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
    ]));
    let spec:RollingSpec=serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":5,"partition_by":["entity"],"event_time":"time","sequence_by":["seq"],
        "outputs":[{"kind":"rsi","primitive_version":1,"input":"price","output":"rsi","period":2},{"kind":"mean","primitive_version":1,"input":"price","output":"mean","frame":{"kind":"rows","size":2},"min_periods":1}],
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},"value_policy":"stateful_numeric_v1"
    })).unwrap();
    RollingOperator::new("features", schema, spec).unwrap()
}
fn row(time: i64) -> BufferedRow {
    BufferedRow::new(
        vec![Some(KeyValue::String("a".into()))],
        vec![KeyValue::Unsigned(u64::try_from(time).unwrap())],
        time,
        vec![
            ScalarValue::Utf8(Some("a".into())),
            ScalarValue::TimestampMicrosecond(Some(time), Some("UTC".into())),
            ScalarValue::UInt64(Some(u64::try_from(time).unwrap())),
            ScalarValue::Float64(Some(f64::from(i32::try_from(time).unwrap()))),
        ],
    )
}
pub(super) fn snapshot(operator: &RollingOperator) -> OperatorStateSnapshot {
    let rows = [row(1), row(2), row(3)];
    let computed = compute_output_columns(
        &rows,
        &RollingHistories::default(),
        &operator.compiled,
        "features",
    )
    .unwrap();
    let mut histories = RollingHistories::default();
    histories.apply(computed.touched);
    let bytes = state_v3::encode(
        &histories,
        &BTreeMap::new(),
        operator.input_ports[0].schema().unwrap(),
        &operator.compiled,
        GRAPH,
        "features",
    )
    .unwrap();
    let segment = StateSegment::new(bytes);
    let epoch = Epoch::new(3).unwrap();
    let id = "base-00000000000000000003-00000000";
    let descriptor = SegmentDescriptor {
        kind: SegmentKind::Base,
        state_layout_version: 5,
        schema_fingerprint: operator.compiled.state_schema_fingerprint.clone(),
        handle: StateHandle::new(
            "features",
            epoch,
            id,
            &format!("committed/features/00000000000000000003-{id}.arrow"),
            u64::try_from(segment.bytes().len()).unwrap(),
            segment.sha256(),
        )
        .unwrap(),
    };
    let metadata = RollingSnapshotMetadata {
        state_layout_version: 5,
        configuration_hash: operator.compiled.configuration_hash.clone(),
        state_schema_fingerprint: operator.compiled.state_schema_fingerprint.clone(),
        kernel_fingerprint: Some(operator.compiled.kernel_plan.fingerprint().into()),
        numerical_profile: Some(operator.compiled.kernel_plan.numerical_profile().into()),
        epoch,
        pipeline_fingerprint: Some(GRAPH.into()),
        operator_id: Some("features".into()),
        last_input_watermark: None,
        next_output_sequence: 1,
        ended: false,
        metrics: crate::operator::rolling::LateMetricDelta::default(),
        segment_inventory: vec![descriptor],
        late_output: None,
    };
    let Value::Object(fields) = serde_json::to_value(metadata).unwrap() else {
        panic!("object")
    };
    OperatorStateSnapshot {
        inline_metadata: fields.into_iter().collect(),
        segments: BTreeMap::from([(id.into(), segment)]),
    }
}
#[test]
fn test_sync_indicator_restore_commits_matching_shared_packet_footprint() {
    let mut operator = operator();
    let snapshot = snapshot(&operator);
    StreamOperator::restore(&mut operator, &snapshot).unwrap();
    assert_eq!(operator.state.histories.by_entity.len(), 1);
    let credit = operator.indicator_resources.as_ref().unwrap().begin_work();
    assert_eq!(
        credit.source().bytes("features").unwrap(),
        operator.state.charge.bytes
    );
    assert_eq!(
        operator.indicator_checkpoint_epoch,
        Some(Epoch::new(3).unwrap())
    );
}
#[tokio::test]
async fn test_async_indicator_restore_publishes_state_only_with_matching_job() {
    let mut operator = operator();
    let snapshot = snapshot(&operator);
    let token = crate::CancellationToken::new();
    let job = crate::StreamJobContext::new(31, GRAPH, crate::JsonMap::new(), None, token);
    let context = crate::StreamOperatorContext::new(&job, "features", None);
    StreamOperator::restore_async(&mut operator, &snapshot, &context)
        .await
        .unwrap();
    let credit = operator.indicator_resources.as_ref().unwrap().begin_work();
    assert_eq!(
        credit.source().bytes("features").unwrap(),
        operator.state.charge.bytes
    );
    assert_eq!(operator.state.operator_id.as_deref(), Some("features"));
}

#[tokio::test]
async fn test_cancelled_indicator_restore_preserves_packet_and_resource_charge() {
    let mut operator = operator();
    let snapshot = snapshot(&operator);
    let original = operator.indicator_state_packet().unwrap();
    let before = operator.indicator_resources.as_ref().unwrap().begin_work();
    let token = crate::CancellationToken::new();
    token.cancel();
    let job = crate::StreamJobContext::new(31, GRAPH, crate::JsonMap::new(), None, token);
    let context = crate::StreamOperatorContext::new(&job, "features", None);
    let result = StreamOperator::restore_async(&mut operator, &snapshot, &context).await;
    assert!(matches!(
        result,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    assert!(Arc::ptr_eq(
        &original,
        &operator.indicator_state_packet().unwrap()
    ));
    let after = operator.indicator_resources.as_ref().unwrap().begin_work();
    assert_eq!(
        before.source().bytes("features").unwrap(),
        after.source().bytes("features").unwrap()
    );
    assert_eq!(operator.indicator_checkpoint_epoch, None);
}

#[test]
fn test_failed_indicator_restore_preserves_existing_packet_and_budget() {
    let mut operator = operator();
    let mut snapshot = snapshot(&operator);
    let original = operator.indicator_state_packet().unwrap();
    let before = operator.indicator_resources.as_ref().unwrap().begin_work();
    snapshot
        .inline_metadata
        .insert("configuration_hash".into(), json!("invalid"));
    assert!(StreamOperator::restore(&mut operator, &snapshot).is_err());
    assert!(Arc::ptr_eq(
        &original,
        &operator.indicator_state_packet().unwrap()
    ));
    let after = operator.indicator_resources.as_ref().unwrap().begin_work();
    assert_eq!(
        before.source().bytes("features").unwrap(),
        after.source().bytes("features").unwrap()
    );
    assert_eq!(operator.indicator_checkpoint_epoch, None);
}

#[test]
fn test_restore_work_identity_stays_owned_and_credited_through_prepared_drop() {
    let operator = operator();
    let snapshot = snapshot(&operator);
    let resources = operator.indicator_resources.as_ref().unwrap();
    let initial = resources.begin_work().source().bytes("features").unwrap();
    let guard = operator
        .try_native_checkpoint_guard("features")
        .unwrap()
        .unwrap();
    let work = operator.restore_work(&snapshot, guard, None).unwrap();
    let identity = Arc::downgrade(&work.node);
    let prepared = work.run(&mut || Ok(())).unwrap();
    assert!(Arc::ptr_eq(&identity.upgrade().unwrap(), &prepared.node));
    assert!(
        resources
            .check(0, crate::StateBudget::new(32, initial).unwrap(), "features")
            .is_err()
    );
    drop(prepared);
    assert!(identity.upgrade().is_none());
    resources
        .check(0, crate::StateBudget::new(32, initial).unwrap(), "features")
        .unwrap();
    assert_eq!(operator.indicator_checkpoint_epoch, None);
}

#[test]
fn test_successful_indicator_restore_clears_effective_late_poison_only_after_commit() {
    let mut operator = operator();
    let mut snapshot = snapshot(&operator);
    operator.indicator_late_failed = true;
    snapshot
        .inline_metadata
        .insert("configuration_hash".into(), json!("invalid"));
    assert!(StreamOperator::restore(&mut operator, &snapshot).is_err());
    assert!(operator.indicator_late_failed);
    let snapshot = self::snapshot(&operator);
    StreamOperator::restore(&mut operator, &snapshot).unwrap();
    assert!(!operator.indicator_late_failed);
}

fn identity_budget_fixture() -> (RollingOperator, OperatorStateSnapshot) {
    let mut operator = operator();
    let mut snapshot = snapshot(&operator);
    snapshot.segments.clear();
    snapshot
        .inline_metadata
        .insert("segment_inventory".into(), json!([]));
    snapshot
        .inline_metadata
        .insert("next_output_sequence".into(), json!(0));
    let quote = super::metadata::quote(&snapshot, &mut || Ok(())).unwrap();
    let bytes = operator.state.charge.bytes.checked_add(quote.peak).unwrap();
    operator
        .set_state_budget(crate::StateBudget::new(32, bytes).unwrap())
        .unwrap();
    (operator, snapshot)
}

#[test]
fn test_sync_identity_admission_failure_is_checkpoint_mismatch_and_keeps_packet() {
    let (mut operator, snapshot) = identity_budget_fixture();
    let original = operator.indicator_state_packet().unwrap();
    let error = StreamOperator::restore(&mut operator, &snapshot).unwrap_err();
    assert!(
        matches!(error, crate::CalcFlowError::CheckpointMismatch { .. }),
        "{error}"
    );
    assert!(Arc::ptr_eq(
        &original,
        &operator.indicator_state_packet().unwrap()
    ));
    assert!(operator.indicator_capture.is_none());
    assert_eq!(operator.indicator_checkpoint_epoch, None);
}

#[tokio::test]
async fn test_async_identity_admission_failure_is_checkpoint_mismatch_and_keeps_packet() {
    let (mut operator, snapshot) = identity_budget_fixture();
    let original = operator.indicator_state_packet().unwrap();
    let job = crate::StreamJobContext::new(
        31,
        GRAPH,
        crate::JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = crate::StreamOperatorContext::new(&job, "features", None);
    let error = StreamOperator::restore_async(&mut operator, &snapshot, &context)
        .await
        .unwrap_err();
    assert!(
        matches!(error, crate::CalcFlowError::CheckpointMismatch { .. }),
        "{error}"
    );
    assert!(Arc::ptr_eq(
        &original,
        &operator.indicator_state_packet().unwrap()
    ));
    assert!(operator.indicator_capture.is_none());
    assert_eq!(operator.indicator_checkpoint_epoch, None);
}

#[test]
fn test_invalid_ipc_with_valid_outer_checksum_is_checkpoint_mismatch() {
    let mut operator = operator();
    let mut snapshot = snapshot(&operator);
    let original = operator.indicator_state_packet().unwrap();
    let (id, segment) = snapshot.segments.first_key_value().unwrap();
    let id = id.clone();
    let mut bytes = segment.bytes().to_vec();
    bytes[0] ^= 1;
    let segment = StateSegment::new(bytes);
    snapshot
        .inline_metadata
        .get_mut("segment_inventory")
        .unwrap()[0]["handle"]["sha256"] = json!(segment.sha256());
    snapshot.segments.insert(id, segment);
    let error = StreamOperator::restore(&mut operator, &snapshot).unwrap_err();
    assert!(
        matches!(error, crate::CalcFlowError::CheckpointMismatch { .. }),
        "{error}"
    );
    assert!(Arc::ptr_eq(
        &original,
        &operator.indicator_state_packet().unwrap()
    ));
}

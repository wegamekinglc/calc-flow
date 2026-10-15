use super::*;
use crate::{Epoch, OperatorStateSnapshot, StateSegment};
use serde_json::{Value, json};

pub(super) async fn projected_snapshot_refusals() {
    let input = input(
        Arc::new(Int64Array::from(vec![10, 20, 30])),
        Arc::new(Int64Array::from(vec![7, 8, 9])),
    );
    let schema = input.table_payload().unwrap().batches()[0].schema();
    let mut source = operator(&schema, true);
    let mut target = operator(&schema, true);
    let job = StreamJobContext::new(
        1,
        "projected-snapshot-refusals",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "selected-state", None);
    let mut output = EdgeCollector::new(source.output_ports().to_vec());
    source
        .process_data("right", input.clone(), &context, &mut output)
        .await
        .unwrap();
    target
        .process_data("left", input, &context, &mut output)
        .await
        .unwrap();
    target.prepare_checkpoint_async(&context).await.unwrap();
    let previous = target.checkpoint(Epoch::INITIAL).unwrap();
    source.prepare_checkpoint_async(&context).await.unwrap();
    let live = source.checkpoint(Epoch::INITIAL).unwrap();
    for (label, metadata) in invalid_metadata(&live) {
        let invalid = OperatorStateSnapshot {
            inline_metadata: metadata,
            ..live.clone()
        };
        reject_snapshot(&mut target, invalid, &job, label).await;
    }
    reject_live_charge(&mut target, &live, &job).await;
    reject_dead_charge(&mut source, &mut target, &context, &mut output).await;
    reject_empty_descriptor(&mut source, &mut target, &context).await;
    target.prepare_checkpoint_async(&context).await.unwrap();
    let continued = target.checkpoint(Epoch::new(2).unwrap()).unwrap();
    cold_checkpoint(&target, continued, &schema, &job)
        .await
        .unwrap();
    let pools = [&mut source, &mut target].map(|operator| {
        operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool()
    });
    drop((source, target, live, previous, context, output));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
}

fn invalid_metadata(snapshot: &OperatorStateSnapshot) -> Vec<(&'static str, JsonMap)> {
    let original = &snapshot.inline_metadata;
    let mut cases = Vec::new();
    for key in ["charge_mode", "retained_payloads"] {
        let mut metadata = original.clone();
        metadata.remove(key);
        cases.push((key, metadata));
    }
    for (key, value) in [("charge_mode", json!("v1_exact")), ("unknown", json!(0))] {
        let mut metadata = original.clone();
        metadata.insert(key.into(), value);
        cases.push((key, metadata));
    }
    descriptor_cases(original, &mut cases);
    cases
}

fn descriptor_cases(original: &JsonMap, cases: &mut Vec<(&'static str, JsonMap)>) {
    for key in ["columns", "logical_schema_digests", "schema_digests"] {
        let mut metadata = original.clone();
        metadata
            .get_mut("retained_payloads")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove(key);
        cases.push((key, metadata));
    }
    let mut unknown = original.clone();
    unknown.get_mut("retained_payloads").unwrap()["unknown"] = json!(0);
    cases.push(("unknown descriptor field", unknown));
    for (label, columns) in [
        ("missing retained side", json!([[1, 2, 3]])),
        ("duplicate retained index", json!([[1, 1, 3], [1, 2, 3]])),
        ("reordered retained indices", json!([[2, 1, 3], [1, 2, 3]])),
        ("broader retained indices", json!([[0, 1, 2, 3], [1, 2, 3]])),
        ("narrower retained indices", json!([[1, 2], [1, 2, 3]])),
        ("out-of-range retained index", json!([[1, 2, 4], [1, 2, 3]])),
    ] {
        let mut metadata = original.clone();
        metadata.get_mut("retained_payloads").unwrap()["columns"] = columns;
        cases.push((label, metadata));
    }
    for key in ["logical_schema_digests", "schema_digests"] {
        let mut metadata = original.clone();
        metadata.get_mut("retained_payloads").unwrap()[key][0] = json!("0".repeat(64));
        cases.push((key, metadata));
    }
}

async fn reject_snapshot(
    target: &mut StreamJoinOperator,
    snapshot: OperatorStateSnapshot,
    job: &StreamJobContext,
    label: &str,
) {
    let status = target.status();
    let epoch = target.state.last_checkpoint_epoch;
    let writer = target.v2_writer.observation_for_test();
    let pool = target.runtime.runtime().unwrap().incremental_memory_pool();
    let baseline = pool.reserved();
    assert!(
        target.restore(&snapshot).is_err(),
        "direct accepted {label}"
    );
    assert_eq!(pool.reserved(), baseline, "direct credit changed: {label}");
    let mut managed = managed_input(&snapshot, target);
    assert!(
        target
            .restore_managed_snapshot(&mut managed, job, None)
            .await
            .is_err(),
        "managed accepted {label}"
    );
    drop(managed);
    assert_eq!(target.status(), status, "state changed: {label}");
    assert_eq!(
        target.state.last_checkpoint_epoch, epoch,
        "epoch changed: {label}"
    );
    assert_eq!(
        target.v2_writer.observation_for_test(),
        writer,
        "writer changed: {label}"
    );
    assert_eq!(pool.reserved(), baseline, "managed credit changed: {label}");
}

async fn reject_live_charge(
    target: &mut StreamJoinOperator,
    live: &OperatorStateSnapshot,
    job: &StreamJobContext,
) {
    let (mut invalid, previous) = change_base_charge(live, 0);
    let retained = live.inline_metadata["metrics"]["right"]["retained_bytes"]
        .as_u64()
        .unwrap();
    invalid.inline_metadata.get_mut("metrics").unwrap()["right"]["retained_bytes"] =
        json!(retained - previous);
    reject_snapshot(
        target,
        invalid,
        job,
        "live charge below retained lower bound",
    )
    .await;
}

async fn reject_dead_charge(
    source: &mut StreamJoinOperator,
    target: &mut StreamJoinOperator,
    context: &StreamOperatorContext<'_>,
    output: &mut EdgeCollector,
) {
    source.on_end(context, output).await.unwrap();
    source.prepare_checkpoint_async(context).await.unwrap();
    let dead = source.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert_eq!(dead.inline_metadata["metrics"]["right"]["retained_rows"], 0);
    assert!(
        !dead.inline_metadata["v2_inventory"]["deltas"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    let limit = source.spec.limits.max_state_bytes_per_side;
    let (invalid, _) = change_base_charge(&dead, limit + 1);
    reject_snapshot(
        target,
        invalid,
        context.job(),
        "dead charge above side limit",
    )
    .await;
}

fn change_base_charge(
    snapshot: &OperatorStateSnapshot,
    charge: u64,
) -> (OperatorStateSnapshot, u64) {
    let (name, segment) = snapshot
        .segments
        .iter()
        .find(|(_, segment)| {
            let bytes = segment.bytes();
            bytes.starts_with(b"CFJIDX2\0") && bytes[12] == 1 && bytes.len() > 56
        })
        .unwrap();
    let mut bytes = segment.bytes().to_vec();
    assert!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()) > 0);
    let previous = u64::from_le_bytes(bytes[48..56].try_into().unwrap());
    bytes[48..56].copy_from_slice(&charge.to_le_bytes());
    let mut invalid = snapshot.clone();
    invalid
        .segments
        .insert(name.clone(), StateSegment::new(bytes));
    (invalid, previous)
}

async fn reject_empty_descriptor(
    source: &mut StreamJoinOperator,
    target: &mut StreamJoinOperator,
    context: &StreamOperatorContext<'_>,
) {
    source.prepare_v2_checkpoint(context).await.unwrap();
    let empty = source.checkpoint(Epoch::new(3).unwrap()).unwrap();
    assert_eq!(empty.inline_metadata["ended"], true);
    assert!(
        empty.inline_metadata["v2_inventory"]["payloads"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    for key in ["charge_mode", "retained_payloads"] {
        let mut invalid = empty.clone();
        invalid.inline_metadata.remove(key);
        reject_snapshot(target, invalid, context.job(), key).await;
    }
    let mut invalid = empty.clone();
    invalid
        .inline_metadata
        .get_mut("retained_payloads")
        .unwrap()["schema_digests"][1] = Value::Null;
    reject_snapshot(target, invalid, context.job(), "empty terminal descriptor").await;
}

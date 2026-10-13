use super::*;
use crate::StateSegment;
use std::collections::BTreeMap;

fn cut(join: &StreamJoinOperator) -> cut::Cut {
    join.v2_writer
        .tracker
        .snapshot(join.state.last_checkpoint_epoch)
}

pub(super) fn image(join: &StreamJoinOperator) -> serde_json::Value {
    let writer = &join.v2_writer;
    let base = writer.base.as_ref().unwrap();
    let rows = [&join.state.left, &join.state.right].map(|rows| {
        (
            Arc::as_ptr(&rows.0) as usize,
            rows.iter()
                .map(|row| {
                    (
                        row.row_id,
                        row.event_time.as_micros(),
                        row.charge,
                        row.encoded_key.as_slice(),
                        row.record.columns().as_ptr() as usize,
                        row.record.offset(),
                    )
                })
                .collect::<Vec<_>>(),
        )
    });
    serde_json::json!({
        "cut": format!("{:?}", cut(join)), "trust": writer.history_trusted,
        "status": join.status(), "base_epoch": writer.base_epoch,
        "deltas": writer.deltas, "payloads": base.payloads,
        "dirty_epochs": writer.dirty_epochs, "migration": writer.migration_required,
        "pending": join.state.deltas.pending.iter().count(),
        "next": [join.state.next_left_row_id, join.state.next_right_row_id, join.state.next_output_sequence],
        "ended": join.state.ended,
        "rows": rows,
        "history": base.segments.iter().map(|(name, segment)| (
            name, segment.sha256(), Arc::as_ptr(&segment.bytes_arc()) as usize, segment.bytes()))
            .collect::<Vec<_>>(),
        "credit": std::ptr::from_ref(base.credit()) as usize,
        "prepared": writer.prepared.is_some(), "prepared_delta": writer.prepared_delta.is_some(),
    })
}

fn replace_index(join: &mut StreamJoinOperator, name: &str, bytes: Vec<u8>) {
    let base = join.v2_writer.base.as_mut().unwrap();
    let original = base.segments.remove(name).unwrap();
    base.segments.insert(
        name.into(),
        StateSegment::new(bytes).with_owner(Arc::new(original)),
    );
}

#[tokio::test]
async fn test_rebase_install_rejects_stale_cut_untrusted_history_and_off_context() {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let disabled = job().with_checkpointing(false);
    let off = StreamOperatorContext::new(&disabled, "prepared-reuse", None);
    let mut pools = Vec::new();
    for invalid in ["cut", "trust", "off"] {
        let (mut join, old) = checkpointed(&context).await;
        let workspace = join.v2_writer_workspace().unwrap();
        let history = encode::rebase::copy(&join, &workspace, &context)
            .await
            .unwrap();
        let base = encode::rebase::encode(&history, &workspace, &|| Ok(())).unwrap();
        let submitted = cut(&join);
        match invalid {
            "cut" => join.v2_writer.changed(false).unwrap(),
            "trust" => join.v2_writer.invalidate_history(),
            "off" => {}
            _ => unreachable!(),
        }
        let before = image(&join);
        join.install_v2_writer_base(
            base,
            submitted,
            Preparation::Rebase,
            if invalid == "off" { &off } else { &context },
        )
        .unwrap();
        assert_eq!(image(&join), before, "{invalid}");
        pools.push(join.runtime.runtime().unwrap().incremental_memory_pool());
        drop((join, old, history, workspace));
    }
    drop((context, off));
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    assert!(disabled.gather_owner().close_and_drain().await.is_empty());
    drop((enabled, disabled));
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
}

fn add_unreferenced_payload(base: &mut encode::Base) {
    let entry = base
        .payloads
        .iter()
        .find(|entry| entry.side == "left")
        .unwrap()
        .clone();
    let original = &base.segments[&format!("left-payload-{}", entry.sha256)];
    let mut payload = original.bytes().to_vec();
    payload[32..40].copy_from_slice(&999_u64.to_le_bytes());
    let segment = StateSegment::new(payload).with_owner(Arc::new(original.clone()));
    let entry = encode::PayloadEntry {
        sha256: segment.sha256().into(),
        ..entry
    };
    base.segments
        .insert(format!("left-payload-{}", entry.sha256), segment);
    base.payloads.push(entry);
    encode::canonical_payloads(&mut base.payloads);
}

#[tokio::test]
async fn test_rebase_rejects_malformed_selected_history_without_mutating_state() {
    check_malformed_history(false).await;
}

#[tokio::test]
async fn test_pending_rebase_rejects_malformed_selected_history_without_mutating_state() {
    check_malformed_history(true).await;
}

async fn check_malformed_history(with_pending: bool) {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let mut pools = Vec::new();
    for invalid in [
        "trailing",
        "truncated_key",
        "unsorted",
        "duplicate",
        "locator",
        "unknown_payload",
        "payload_row",
        "charge",
        "charge_total",
        "count",
        "next_id",
        "header",
        "inventory",
        "missing",
        "overflow",
        "extra_payload",
    ] {
        let (mut join, old) = checkpointed(&context).await;
        if with_pending {
            admit_left(&mut join, &context).await;
            assert!(!join.state.deltas.pending.is_empty());
        }
        let base = join.v2_writer.base.as_mut().unwrap();
        let mut bytes = base.segments["left-delta-4"].bytes().to_vec();
        match invalid {
            "trailing" => bytes.push(0),
            "truncated_key" => {
                bytes.pop();
            }
            "unsorted" => {
                let first_len = 72
                    + usize::try_from(u64::from_le_bytes(bytes[104..112].try_into().unwrap()))
                        .unwrap();
                bytes[40 + first_len..48 + first_len].copy_from_slice(&6_u64.to_le_bytes());
            }
            "duplicate" => {
                let previous = base.segments["left-delta-3"].bytes();
                bytes[40..].copy_from_slice(&previous[40..]);
            }
            "locator" => bytes[96..104].copy_from_slice(&u64::MAX.to_le_bytes()),
            "unknown_payload" => bytes[64..96].fill(0),
            "payload_row" => bytes[96..104].copy_from_slice(&1_u64.to_le_bytes()),
            "charge" => bytes[56..64].copy_from_slice(&u64::MAX.to_le_bytes()),
            "charge_total" => bytes[56] += 1,
            "count" => {
                join.state.metrics.left.retained_rows += 1;
            }
            "next_id" => {
                join.state.next_left_row_id = 7;
            }
            "header" => bytes[0] = 0,
            "inventory" => {
                join.v2_writer.base.as_mut().unwrap().payloads.reverse();
            }
            "missing" => {
                join.v2_writer
                    .base
                    .as_mut()
                    .unwrap()
                    .segments
                    .remove("right-base");
            }
            "overflow" => bytes[24..32].copy_from_slice(&u64::MAX.to_le_bytes()),
            "extra_payload" => add_unreferenced_payload(base),
            _ => unreachable!(),
        }
        if !matches!(
            invalid,
            "count" | "next_id" | "inventory" | "missing" | "extra_payload"
        ) {
            replace_index(&mut join, "left-delta-4", bytes);
        }
        let before = image(&join);
        let bases = count_base(&mut join);
        let failure = join.prepare_checkpoint_async(&context).await;
        assert!(failure.is_err(), "{invalid} must fail selected rebase");
        join.await_compaction_release(&context).await.unwrap();
        assert_eq!(
            bases.load(Ordering::SeqCst),
            0,
            "{invalid} must not fall back"
        );
        assert_eq!(image(&join), before, "{invalid}");
        pools.push(join.runtime.runtime().unwrap().incremental_memory_pool());
        drop((join, old));
    }
    drop(context);
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    drop(enabled);
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
}

#[test]
fn test_rebase_cancellation_cleans_both_passes_and_preserves_history() {
    isolated_writer_test(|service, runtime| {
        runtime.block_on(assert_rebase_cancellation_preserves_history(service));
    });
}

async fn assert_rebase_cancellation_preserves_history(
    service: &crate::runtime::streaming::gather_work::TestService,
) {
    let enabled = job().with_gather_owner(service.owner("rebase-cancel-history".into()));
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let (mut join, old) = checkpointed(&context).await;
    let workspace = join.v2_writer_workspace().unwrap();
    let history = encode::rebase::copy(&join, &workspace, &context)
        .await
        .unwrap();
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    let before = image(&join);
    let checks = std::cell::Cell::new(0_usize);
    let baseline = encode::rebase::encode(&history, &workspace, &|| {
        checks.set(checks.get() + 1);
        Ok(())
    })
    .unwrap();
    let count = checks.get();
    drop(baseline);
    let reserved = pool.reserved();
    for cancel_at in 1..=count {
        checks.set(0);
        let result = encode::rebase::encode(&history, &workspace, &|| {
            checks.set(checks.get() + 1);
            if checks.get() == cancel_at {
                Err(crate::CalcFlowError::Cancelled {
                    run_id: "rebase-check".into(),
                })
            } else {
                Ok(())
            }
        });
        assert!(
            matches!(result, Err(crate::CalcFlowError::Cancelled { .. })),
            "check {cancel_at}"
        );
        assert_eq!(pool.reserved(), reserved, "check {cancel_at}");
        assert_eq!(image(&join), before, "check {cancel_at}");
    }
    drop((history, workspace));
    join.prepare_checkpoint_async(&context).await.unwrap();
    let captured = join.checkpoint(Epoch::new(5).unwrap()).unwrap();
    let mut restored = operator();
    restored.restore(&captured).unwrap();
    assert_same_live(&join, &restored);
    drop((join, old, captured, restored));
    drop(context);
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    drop(enabled);
    assert_eq!(pool.reserved(), 0);
}

pub(super) fn remap_left_ids(
    snapshot: OperatorStateSnapshot,
    remap: impl Fn(u64) -> u64,
) -> OperatorStateSnapshot {
    let mut snapshot = snapshot;
    let mut renamed = BTreeMap::new();
    let mut payloads = Vec::new();
    for payload in snapshot.inline_metadata["v2_inventory"]["payloads"]
        .as_array()
        .unwrap()
    {
        let mut entry = payload.clone();
        if entry["side"] == "left" {
            let old = entry["sha256"].as_str().unwrap().to_owned();
            let original = snapshot
                .segments
                .remove(&format!("left-payload-{old}"))
                .unwrap();
            let mut bytes = original.bytes().to_vec();
            let rows = usize::try_from(entry["rows"].as_u64().unwrap()).unwrap();
            for offset in 0..rows {
                let slot = &mut bytes[32 + offset * 8..40 + offset * 8];
                let id = u64::from_le_bytes(slot.try_into().unwrap());
                slot.copy_from_slice(&remap(id).to_le_bytes());
            }
            let replacement = StateSegment::new(bytes);
            renamed.insert(old, hex::decode(replacement.sha256()).unwrap());
            entry["sha256"] = replacement.sha256().into();
            snapshot.segments.insert(
                format!("left-payload-{}", replacement.sha256()),
                replacement,
            );
        }
        payloads.push(entry);
    }
    payloads.sort_by(|left, right| {
        (left["side"].as_str(), left["sha256"].as_str())
            .cmp(&(right["side"].as_str(), right["sha256"].as_str()))
    });
    snapshot.inline_metadata.get_mut("v2_inventory").unwrap()["payloads"] = payloads.into();
    for (name, segment) in &mut snapshot.segments {
        if !name.starts_with("left-") || name.contains("-payload-") {
            continue;
        }
        *segment = remap_index(segment, name, &renamed, &remap);
    }
    snapshot
}

fn remap_index(
    segment: &StateSegment,
    name: &str,
    renamed: &BTreeMap<String, Vec<u8>>,
    remap: &impl Fn(u64) -> u64,
) -> StateSegment {
    let header = if name == "left-base" { 32 } else { 40 };
    let mut input = &segment.bytes()[header..];
    let mut records = Vec::new();
    while !input.is_empty() {
        let length =
            72 + usize::try_from(u64::from_le_bytes(input[64..72].try_into().unwrap())).unwrap();
        let mut row = input[..length].to_vec();
        let original_id = u64::from_le_bytes(row[..8].try_into().unwrap());
        let id = remap(original_id);
        row[..8].copy_from_slice(&id.to_le_bytes());
        let digest = hex::encode(&row[24..56]);
        row[24..56].copy_from_slice(&renamed[&digest]);
        records.push((id, row));
        input = &input[length..];
    }
    records.sort_by_key(|(id, _)| *id);
    let mut bytes = segment.bytes()[..header].to_vec();
    for (_, row) in records {
        bytes.extend_from_slice(&row);
    }
    StateSegment::new(bytes)
}

#[tokio::test]
async fn test_rebase_merges_validated_restored_interleaved_ids_across_epochs() {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let (original, old) = checkpointed(&context).await;
    let snapshot = remap_left_ids(old, |id| id / 2 + (id % 2) * 4);
    let mut join = operator();
    join.restore(&snapshot).unwrap();
    let mut expected = operator();
    expected.restore(&snapshot).unwrap();
    assert_eq!(
        join.state
            .left
            .iter()
            .map(|row| row.row_id)
            .collect::<Vec<_>>(),
        (0..8).collect::<Vec<_>>()
    );
    let bases = count_base(&mut join);
    join.prepare_checkpoint_async(&context).await.unwrap();
    assert_eq!(bases.load(Ordering::SeqCst), 0);
    assert_same_live(&expected, &join);
    let rebased = join.checkpoint(Epoch::new(5).unwrap()).unwrap();
    let mut restored = operator();
    restored.restore(&rebased).unwrap();
    assert_same_live(&join, &restored);
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    drop((original, snapshot, expected, join, rebased, restored));
    drop(context);
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    drop(enabled);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_rebase_encodes_empty_and_one_sided_history() {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let mut pools = Vec::new();
    for populated in [false, true] {
        let mut join = operator();
        for epoch in 1..=4 {
            if populated {
                admit_left(&mut join, &context).await;
            }
            drop(join.checkpoint(Epoch::new(epoch).unwrap()).unwrap());
        }
        let bases = count_base(&mut join);
        assert!(encode::rebase::eligible(&join, &context).await.unwrap());
        join.prepare_v2_kind(Preparation::Rebase, &context)
            .await
            .unwrap();
        assert_eq!(bases.load(Ordering::SeqCst), 0);
        let snapshot = join.checkpoint(Epoch::new(5).unwrap()).unwrap();
        assert_eq!(snapshot.segments["right-base"].bytes().len(), 32);
        assert_eq!(
            snapshot.inline_metadata["v2_inventory"]["deltas"],
            serde_json::json!([])
        );
        assert_eq!(snapshot.inline_metadata["v2_inventory"]["base_epoch"], 4);
        let mut restored = operator();
        restored.restore(&snapshot).unwrap();
        assert_same_live(&join, &restored);
        pools.push(join.runtime.runtime().unwrap().incremental_memory_pool());
        drop((join, snapshot, restored));
    }
    drop(context);
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    drop(enabled);
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
}

#[tokio::test]
async fn test_rebase_falls_back_for_tombstoned_migrating_and_off_history() {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let disabled = job().with_checkpointing(false);
    let off = StreamOperatorContext::new(&disabled, "prepared-reuse", None);
    let mut pools = Vec::new();
    for boundary in ["pending_tombstone", "tombstone", "migration", "off"] {
        let (mut join, old) = checkpointed(&context).await;
        let bases = count_base(&mut join);
        match boundary {
            "pending_tombstone" | "tombstone" => {
                let mut output = EdgeCollector::new(join.output_ports().to_vec());
                join.on_end(&context, &mut output).await.unwrap();
                if boundary == "tombstone" {
                    drop(join.checkpoint(Epoch::new(5).unwrap()).unwrap());
                }
            }
            "migration" => join.v2_writer.migration_required = true,
            "off" => {}
            _ => unreachable!(),
        }
        join.prepare_checkpoint_async(if boundary == "off" { &off } else { &context })
            .await
            .unwrap();
        assert_eq!(bases.load(Ordering::SeqCst), 1, "{boundary}");
        let snapshot = join.checkpoint(Epoch::new(6).unwrap()).unwrap();
        let mut restored = operator();
        restored.restore(&snapshot).unwrap();
        assert_same_live(&join, &restored);
        pools.push(join.runtime.runtime().unwrap().incremental_memory_pool());
        drop((join, old, snapshot, restored));
    }
    drop((context, off));
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    assert!(disabled.gather_owner().close_and_drain().await.is_empty());
    drop((enabled, disabled));
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
}

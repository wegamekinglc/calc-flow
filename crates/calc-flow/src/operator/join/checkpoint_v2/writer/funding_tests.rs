use super::*;
use crate::{
    OperatorStateSnapshot,
    operator::join::{JoinMetrics, JoinStateLimits, JoinTimeBounds, SideMetrics, StreamJoinSpec},
};
use std::time::Duration;

#[test]
fn test_last_segment_funds_live_allocations_and_drops_bytes_before_credit() {
    let (rows, record) = fixture();
    for payload in [false, true] {
        let tracking = Arc::new(IpcPool::new(16 << 20));
        let pool: Arc<dyn MemoryPool> = tracking.clone();
        let mut retained = None;
        let allocations = allocation_counter::measure(|| {
            let workspace = MemoryConsumer::new("last-segment-owner").register(&pool);
            let mut output = base(
                [&rows, &[]],
                &[record.schema(), record.schema()],
                &workspace,
                &|| Ok(()),
            )
            .unwrap();
            let name = if payload {
                output
                    .segments
                    .keys()
                    .find(|name| name.starts_with("left-payload-"))
                    .unwrap()
                    .clone()
            } else {
                "left-base".into()
            };
            retained = output.segments.remove(&name);
            drop((output, workspace));
        });
        let segment = retained.unwrap();
        assert!(allocations.bytes_max <= u64::try_from(tracking.peak()).unwrap());
        assert!(usize::try_from(allocations.bytes_current).unwrap() <= pool.reserved());
        assert!(pool.reserved() >= segment.bytes_arc().capacity());
        *tracking.retiring.lock().unwrap() = Some(Arc::downgrade(&segment.bytes_arc()));
        drop(segment);
        assert_eq!(
            tracking
                .retirements
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(pool.reserved(), 0);
    }
}

#[test]
fn test_merged_capture_keeps_segment_backing_separate_from_control_owners() {
    use crate::operator::join::checkpoint_v2::writer::{capture_credit, clone_segments};

    let (rows, record) = fixture();
    let (pool, workspace) = reservation(16 << 20);
    let mut output = base(
        [&rows[..1].to_vec(), &[]],
        &[record.schema(), record.schema()],
        &workspace,
        &|| Ok(()),
    )
    .unwrap();
    let incoming = pending_owned(
        &[pending_upsert(JoinSide::Left, &rows[1])],
        &[record.schema(), record.schema()],
        Epoch::new(2).unwrap(),
        &workspace,
        &|| Ok(()),
    )
    .unwrap();
    let controls = [
        Arc::downgrade(&output.funding),
        Arc::downgrade(&incoming.funding),
    ];
    let original = incoming.segments.clone();
    output.merge(incoming).unwrap();
    assert_eq!(output.payloads.len(), 2);
    let capture = capture_credit(output.credit()).unwrap();
    let capture_owner = Arc::downgrade(&capture);
    let segments = clone_segments(&output, None, &capture).unwrap();
    for (name, segment) in &original {
        assert!(Arc::ptr_eq(
            &segment.bytes_arc(),
            &segments[name].bytes_arc()
        ));
        assert_eq!(segment.bytes(), segments[name].bytes());
    }
    let index = segments["left-delta-2"].clone();
    let payload_name = original
        .keys()
        .find(|name| name.starts_with("left-payload-"))
        .unwrap();
    let payload = segments[payload_name].clone();
    let expected = payload.bytes().to_vec();
    let capacity = index.bytes_arc().capacity();
    drop((segments, original, output, workspace, capture));
    assert!(controls.iter().all(|owner| owner.upgrade().is_none()));
    assert!(capture_owner.upgrade().is_some());
    let both = pool.reserved();
    drop(index);
    assert!(both - pool.reserved() >= capacity);
    assert!(pool.reserved() >= payload.bytes_arc().capacity());
    assert_eq!(payload.bytes(), expected);
    assert!(capture_owner.upgrade().is_some());
    drop(payload);
    assert!(capture_owner.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_segment_transfer_rejections_preserve_existing_bytes_and_credit() {
    let (pool, workspace) = reservation(1 << 20);
    let mut encoder = Encoder::new(&workspace).unwrap();
    encoder
        .install("left-delta-1".into(), vec![0x59; 8])
        .unwrap();
    let original = encoder.output.segments["left-delta-1"].clone();
    let before = pool.reserved();
    let pressure = workspace.new_empty();
    pressure.try_grow((1 << 20) - before).unwrap();
    let failure = encoder.install("rejected".into(), vec![0x91; 1 << 16]);
    assert!(failure.is_err());
    assert_eq!(encoder.output.segments.len(), 1);
    assert_eq!(original.bytes(), &[0x59; 8]);
    assert_eq!(pool.reserved(), 1 << 20);
    drop(pressure);
    assert_eq!(pool.reserved(), before);

    let control = Arc::downgrade(&encoder.funding);
    let failure = retarget::at_epoch(encoder.finish(), Epoch::new(7).unwrap());
    assert!(failure.is_err(), "a truncated index cannot be retargeted");
    assert!(control.upgrade().is_none());
    assert_eq!(original.bytes(), &[0x59; 8]);
    drop(workspace);
    assert!(pool.reserved() >= original.bytes_arc().capacity());
    drop(original);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_encoder_cancellation_releases_partial_segment_transfers() {
    use std::cell::Cell;

    let (rows, record) = rich_ipc_fixture();
    let pending = rows
        .iter()
        .map(|row| pending_upsert(JoinSide::Left, row))
        .collect::<Vec<_>>();
    for delta in [false, true] {
        let run = |workspace: &MemoryReservation, check: &dyn Fn() -> Result<()>| {
            if delta {
                pending_owned(
                    &pending,
                    &[record.schema(), record.schema()],
                    Epoch::INITIAL,
                    workspace,
                    check,
                )
            } else {
                base(
                    [&rows, &[]],
                    &[record.schema(), record.schema()],
                    workspace,
                    check,
                )
            }
        };
        let checks = Cell::new(0);
        let (pool, workspace) = reservation(16 << 20);
        let output = run(&workspace, &|| {
            checks.set(checks.get() + 1);
            Ok(())
        })
        .unwrap();
        drop((output, workspace));
        assert_eq!(pool.reserved(), 0);
        for stop in 0..checks.get() {
            let (pool, workspace) = reservation(16 << 20);
            let calls = Cell::new(0);
            let result = run(&workspace, &|| {
                if calls.replace(calls.get() + 1) == stop {
                    Err(CalcFlowError::Cancelled {
                        run_id: "segment-credit".into(),
                    })
                } else {
                    Ok(())
                }
            });
            assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
            drop((result, workspace));
            assert_eq!(pool.reserved(), 0, "delta={delta} check={stop}");
        }
    }
}

#[test]
fn test_restore_cancellation_releases_partial_segment_transfers() {
    use crate::operator::join::checkpoint_v2::inventory::Inventory;
    use std::cell::Cell;

    for owned in [false, true] {
        let (source_pool, source_workspace) = reservation(16 << 20);
        let snapshot = snapshot_fixture(owned, &source_workspace);
        drop(source_workspace);
        let original_credit = source_pool.reserved();
        let inventory = Inventory::decode(&snapshot, &|| Ok(())).unwrap();
        let checks = Cell::new(0);
        let (pool, workspace) = reservation(16 << 20);
        let history = restored::decode(&snapshot, &inventory, &workspace, &|| {
            checks.set(checks.get() + 1);
            Ok(())
        })
        .unwrap();
        drop((history, workspace));
        assert_eq!(pool.reserved(), 0);
        for stop in 0..checks.get() {
            let (pool, workspace) = reservation(16 << 20);
            let calls = Cell::new(0);
            let result = restored::decode(&snapshot, &inventory, &workspace, &|| {
                if calls.replace(calls.get() + 1) == stop {
                    Err(CalcFlowError::Cancelled {
                        run_id: "segment-restore".into(),
                    })
                } else {
                    Ok(())
                }
            });
            assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
            drop((result, workspace));
            assert_eq!(pool.reserved(), 0, "owned={owned} check={stop}");
            assert_eq!(source_pool.reserved(), original_credit);
        }
        drop(snapshot);
        assert_eq!(source_pool.reserved(), 0);
    }
}

fn snapshot_fixture(owned: bool, workspace: &MemoryReservation) -> OperatorStateSnapshot {
    let (rows, record) = flat_ipc_fixture(257);
    let output = base(
        [&rows, &[]],
        &[record.schema(), record.schema()],
        workspace,
        &|| Ok(()),
    )
    .unwrap();
    let spec = StreamJoinSpec::inner(
        ["symbol"],
        ["symbol"],
        "event_time",
        "event_time",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(1_000, 1 << 20, 1_000).unwrap(),
    )
    .unwrap();
    let metrics = JoinMetrics {
        left: SideMetrics {
            retained_rows: u64::try_from(rows.len()).unwrap(),
            retained_bytes: rows.iter().map(|row| row.charge).sum(),
            ..SideMetrics::default()
        },
        ..JoinMetrics::default()
    };
    let metadata = serde_json::json!({
        "layout_version": 2, "spec": spec,
        "next_left_row_id": rows.len(), "next_right_row_id": 0,
        "next_output_sequence": 0, "metrics": metrics, "ended": false, "epoch": 1,
        "v2_inventory": {
            "codec_version": 2, "base_epoch": 1,
            "deltas": [], "payloads": output.payloads,
        },
    });
    let segments = output
        .segments
        .into_iter()
        .map(|(name, segment)| {
            if owned {
                (name, segment)
            } else {
                let mut bytes = Vec::with_capacity(segment.bytes().len() + 73);
                bytes.extend_from_slice(segment.bytes());
                (name, StateSegment::new(bytes))
            }
        })
        .collect();
    OperatorStateSnapshot {
        inline_metadata: metadata.as_object().unwrap().clone().into_iter().collect(),
        segments,
    }
}

#[test]
fn test_restored_segment_aliases_keep_separate_backing_and_pool_ownership() {
    for owned in [false, true] {
        for index_first in [false, true] {
            assert_restored_aliases(owned, index_first);
        }
    }
}

fn assert_restored_aliases(owned: bool, index_first: bool) {
    use crate::operator::join::checkpoint_v2::inventory::Inventory;

    let (source_pool, source_workspace) = reservation(16 << 20);
    let snapshot = snapshot_fixture(owned, &source_workspace);
    drop(source_workspace);
    let (pool, workspace) = reservation(16 << 20);
    let inventory = Inventory::decode(&snapshot, &|| Ok(())).unwrap();
    let history = restored::decode(&snapshot, &inventory, &workspace, &|| Ok(())).unwrap();
    assert_eq!((history.base_epoch, history.dirty_epochs), (1, 0));
    assert!(history.deltas.is_empty());
    assert_eq!(history.base.segments.len(), snapshot.segments.len());
    for (name, original) in &snapshot.segments {
        let restored = &history.base.segments[name];
        assert!(Arc::ptr_eq(&original.bytes_arc(), &restored.bytes_arc()));
        assert_eq!(original.bytes(), restored.bytes());
    }
    let control = Arc::downgrade(&history.base.funding);
    let index = history.base.segments["left-base"].clone();
    let payload = history
        .base
        .segments
        .iter()
        .find(|(name, _)| name.starts_with("left-payload-"))
        .unwrap()
        .1
        .clone();
    let (first, last) = if index_first {
        (index, payload)
    } else {
        (payload, index)
    };
    let first_capacity = first.bytes_arc().capacity();
    let last_capacity = last.bytes_arc().capacity();
    if !owned {
        assert!(first_capacity > first.bytes().len());
        assert!(last_capacity > last.bytes().len());
    }
    let expected = last.bytes().to_vec();
    drop((history, workspace, snapshot));
    let both = pool.reserved();
    let original_both = source_pool.reserved();
    if owned {
        assert!(original_both >= first_capacity + last_capacity);
        assert!(both < first_capacity + last_capacity);
    } else {
        assert_eq!(original_both, 0);
        assert!(both >= first_capacity + last_capacity);
    }
    drop(first);
    if owned {
        assert!(original_both - source_pool.reserved() >= first_capacity);
        assert!(source_pool.reserved() >= last_capacity);
        assert!(both > pool.reserved());
    } else {
        assert!(both - pool.reserved() >= first_capacity);
        assert!(pool.reserved() >= last_capacity);
    }
    assert!(
        control.upgrade().is_none(),
        "segment aliases must not pin the restored Base controls"
    );
    assert_eq!(last.bytes(), expected);
    drop(last);
    assert_eq!((source_pool.reserved(), pool.reserved()), (0, 0));
}

#[test]
fn test_retargeted_index_owns_replacement_credit_with_original_history_held() {
    let (rows, record) = fixture();
    let pending = [
        pending_upsert(JoinSide::Left, &rows[0]),
        pending_upsert(JoinSide::Right, &rows[2]),
    ];
    let (pool, workspace) = reservation(16 << 20);
    let output = pending_owned(
        &pending,
        &[record.schema(), record.schema()],
        Epoch::INITIAL,
        &workspace,
        &|| Ok(()),
    )
    .unwrap();
    let control = Arc::downgrade(&output.funding);
    let original = output.segments.clone();
    let epoch = Epoch::new(7).unwrap();
    let retargeted = retarget::at_epoch(output, epoch).unwrap();
    for side in ["left", "right"] {
        let old = &original[&format!("{side}-delta-1")];
        let next = &retargeted.segments[&format!("{side}-delta-7")];
        let mut expected = old.bytes().to_vec();
        expected[16..24].copy_from_slice(&epoch.as_u64().to_le_bytes());
        assert_eq!(next.bytes(), expected);
        assert!(!Arc::ptr_eq(&old.bytes_arc(), &next.bytes_arc()));
        assert_eq!(&old.bytes()[16..24], &1_u64.to_le_bytes());
    }
    for (name, old) in original
        .iter()
        .filter(|(name, _)| name.contains("-payload-"))
    {
        let next = &retargeted.segments[name];
        assert!(Arc::ptr_eq(&old.bytes_arc(), &next.bytes_arc()));
        assert_eq!(old.bytes(), next.bytes());
    }
    let replacement = retargeted.segments["left-delta-7"].clone();
    let capacity = replacement.bytes_arc().capacity();
    drop((retargeted, workspace));
    assert!(
        control.upgrade().is_none(),
        "replacement indexes must not retain unrelated Base control credit"
    );
    let both = pool.reserved();
    drop(replacement);
    assert!(both - pool.reserved() >= capacity);
    assert!(original.values().all(StateSegment::has_owner));
    drop(original);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_segment_aliases_retire_sibling_backing_and_encoder_credit_independently() {
    let (rows, record) = flat_ipc_fixture(257);
    for index_first in [false, true] {
        let (pool, workspace) = reservation(16 << 20);
        let output = base(
            [&rows, &[]],
            &[record.schema(), record.schema()],
            &workspace,
            &|| Ok(()),
        )
        .unwrap();
        let control = Arc::downgrade(&output.funding);
        let index = output.segments["left-base"].clone();
        let payload = output
            .segments
            .iter()
            .find(|(name, _)| name.starts_with("left-payload-"))
            .unwrap()
            .1
            .clone();
        let (first, last) = if index_first {
            (index, payload)
        } else {
            (payload, index)
        };
        let first_capacity = first.bytes_arc().capacity();
        let last_capacity = last.bytes_arc().capacity();
        let expected = last.bytes().to_vec();
        drop((output, workspace));
        let both = pool.reserved();
        assert!(both >= first_capacity + last_capacity);
        drop(first);
        let remaining = pool.reserved();
        assert!(
            both - remaining >= first_capacity,
            "a surviving segment must not pin sibling backing: {both} -> {remaining}"
        );
        assert!(remaining >= last_capacity);
        assert!(
            control.upgrade().is_none(),
            "a segment alias must not retain Base controls or encoder high-water credit"
        );
        assert_eq!(last.bytes(), expected);
        drop(last);
        assert_eq!(pool.reserved(), 0);
    }
}

#[test]
fn test_finished_base_and_delta_keep_controls_independent_of_row_count() {
    for delta in [false, true] {
        let mut controls = None;
        for length in [1, 257] {
            let (rows, record) = flat_ipc_fixture(length);
            let pending = rows
                .iter()
                .map(|row| pending_upsert(JoinSide::Left, row))
                .collect::<Vec<_>>();
            let tracking = Arc::new(IpcPool::new(16 << 20));
            let pool: Arc<dyn MemoryPool> = tracking.clone();
            let workspace = MemoryConsumer::new("segment-finalize").register(&pool);
            let mut result = None;
            let allocations = allocation_counter::measure(|| {
                result = Some(if delta {
                    pending_owned(
                        &pending,
                        &[record.schema(), record.schema()],
                        Epoch::INITIAL,
                        &workspace,
                        &|| Ok(()),
                    )
                } else {
                    base(
                        [&rows, &[]],
                        &[record.schema(), record.schema()],
                        &workspace,
                        &|| Ok(()),
                    )
                });
                drop(workspace);
            });
            let output = result.unwrap().unwrap();
            assert!(allocations.bytes_max <= u64::try_from(tracking.peak()).unwrap());
            assert!(usize::try_from(allocations.bytes_current).unwrap() <= pool.reserved());
            let backing = output
                .segments
                .values()
                .map(|segment| segment.bytes_arc().capacity())
                .sum::<usize>();
            assert!(pool.reserved() >= backing + output.credit().size());
            if let Some(expected) = controls {
                assert_eq!(
                    output.credit().size(),
                    expected,
                    "fixed inventory controls must not retain row-dependent writer scratch"
                );
            } else {
                controls = Some(output.credit().size());
            }
            drop(output);
            assert_eq!(pool.reserved(), 0);
        }
    }
}

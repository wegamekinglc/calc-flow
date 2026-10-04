use super::*;
use crate::{IngressProgress, IngressState, StateSegment, StreamOperator};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};

fn anchored(operator: &mut StreamAsofJoinOperator) -> (Log, Anchor, OperatorStateSnapshot) {
    let mut log = operator.new_replay_log().unwrap();
    log.credit.try_grow(size_of::<Record>()).unwrap();
    log.records = vec![Record {
        callback: Callback::Progress,
        input_watermark: None,
        progress: [IngressProgress::new(IngressState::Active, None); 2],
        max_rows: 1000,
        max_bytes: 1024 * 1024,
        cursor: None,
        cursor_bytes: 0,
    }];
    let native = operator.capture(Epoch::INITIAL).unwrap();
    let anchor = operator.make_replay_anchor(&native, &log).unwrap();
    let mut snapshot = native;
    snapshot
        .segments
        .insert(anchor.start_id.clone(), anchor.starts.clone());
    (log, anchor, snapshot)
}

#[test]
fn native_anchor_rejects_corruption_and_refunds_credit() {
    let (mut operator, _) = crate::operator::asof::tests::fixture();
    let pool = operator.runtime.pool.clone();
    {
        let (_log, anchor, snapshot) = anchored(&mut operator);
        let baseline = pool.reserved();
        let restored = operator
            .decode_replay_anchor(&snapshot, &anchor.descriptor())
            .unwrap()
            .unwrap();
        assert_eq!(restored.records, anchor.records);
        drop(restored);
        assert_eq!(pool.reserved(), baseline);
        for change in 0..4 {
            let mut descriptor = anchor.descriptor();
            let AnchorControl::Native {
                segments,
                starts,
                credit,
                ..
            } = &mut descriptor
            else {
                unreachable!()
            };
            match change {
                0 => *credit = 0,
                1 => *starts = super::super::CONTROL_ID.into(),
                2 => *starts = "missing".into(),
                3 => segments.push(starts.clone()),
                _ => unreachable!(),
            }
            assert!(
                operator
                    .decode_replay_anchor(&snapshot, &descriptor)
                    .is_err()
            );
            assert_eq!(pool.reserved(), baseline);
        }
        let mut corrupted = snapshot.clone();
        let mut bytes = anchor.starts.bytes().to_vec();
        bytes[8..16].copy_from_slice(&4_u64.to_le_bytes());
        corrupted
            .segments
            .insert(anchor.start_id.clone(), StateSegment::new(bytes));
        assert!(
            operator
                .decode_replay_anchor(&corrupted, &anchor.descriptor())
                .is_err()
        );
        assert_eq!(pool.reserved(), baseline);
    }
    drop(operator);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn native_anchor_decode_refuses_unfunded_workspace() {
    let (mut operator, _) = crate::operator::asof::tests::fixture();
    let (_log, anchor, snapshot) = anchored(&mut operator);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1));
    operator.runtime.pool = pool.clone();
    assert!(
        operator
            .decode_replay_anchor(&snapshot, &anchor.descriptor())
            .is_err()
    );
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn native_anchor_pressure_falls_back_to_a_native_checkpoint() {
    let (mut operator, _) = crate::operator::asof::tests::fixture();
    let pool = operator.runtime.pool.clone();
    let (log, anchor, snapshot) = anchored(&mut operator);
    drop(anchor);
    drop(snapshot);
    operator.replay = Some(Box::new(log));
    operator.status.state_bytes = operator.current_inventory(None).unwrap().bytes;
    let available =
        usize::try_from(operator.spec.limits().max_state_bytes()).unwrap() - pool.reserved();
    let blocker = operator.reserve_workspace((available - 1) as u64).unwrap();
    let snapshot = operator
        .replace_replay_anchor(Epoch::INITIAL)
        .unwrap()
        .expect("unfunded replay anchor falls back to native state");
    assert!(!snapshot.inline_metadata.contains_key("source_replay"));
    assert!(operator.replay.is_none());
    assert_eq!(operator.status.state_bytes, 0);
    let (mut restored, _) = crate::operator::asof::tests::fixture();
    restored.restore(&snapshot).unwrap();
    assert_eq!(restored.status.state_bytes, 0);
    drop(blocker);
    drop(snapshot);
    drop(operator);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn native_anchor_empty_state_preserves_restorable_metrics() {
    let (mut operator, _) = crate::operator::asof::tests::fixture();
    let (log, anchor, snapshot) = anchored(&mut operator);
    drop(anchor);
    drop(snapshot);
    operator.replay = Some(Box::new(log));
    operator.status.state_bytes = operator.current_inventory(None).unwrap().bytes;
    assert!(
        operator
            .replace_replay_anchor(Epoch::INITIAL)
            .unwrap()
            .is_none()
    );
    let native = &operator
        .replay
        .as_ref()
        .unwrap()
        .anchor
        .as_ref()
        .unwrap()
        .snapshot;
    let (mut restored, _) = crate::operator::asof::tests::fixture();
    restored.restore(native).unwrap();
    assert_eq!(restored.status.state_bytes, 0);
}

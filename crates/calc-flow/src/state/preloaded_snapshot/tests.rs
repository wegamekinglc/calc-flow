use std::{
    collections::BTreeMap,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
};

use super::PreloadedSnapshot;
use crate::{CalcFlowError, OperatorStateSnapshot, StateSegment};

struct Receipt {
    bytes: Weak<Vec<u8>>,
    released: Arc<AtomicBool>,
}

impl Drop for Receipt {
    fn drop(&mut self) {
        assert!(
            self.bytes.upgrade().is_none(),
            "receipt refunded while snapshot data is live"
        );
        self.released.store(true, Ordering::SeqCst);
    }
}

fn credited() -> (Arc<PreloadedSnapshot>, Arc<AtomicBool>) {
    let segment = StateSegment::new(vec![7; 4096]);
    let released = Arc::new(AtomicBool::new(false));
    let receipt = Arc::new(Receipt {
        bytes: Arc::downgrade(&segment.bytes_arc()),
        released: Arc::clone(&released),
    });
    let snapshot = OperatorStateSnapshot {
        inline_metadata: BTreeMap::new(),
        segments: BTreeMap::from([("state".into(), segment)]),
    };
    (
        Arc::new(PreloadedSnapshot::new(snapshot, Some(receipt))),
        released,
    )
}

#[test]
fn test_shared_recovery_receipt_outlives_the_last_actual_snapshot_data_owner() {
    let (snapshot, released) = credited();
    let alias = Arc::clone(&snapshot);
    assert!(Arc::ptr_eq(&snapshot, &alias));
    drop(snapshot);
    assert!(!released.load(Ordering::SeqCst));
    assert_eq!(alias.segments["state"].bytes().len(), 4096);
    drop(alias);
    assert!(released.load(Ordering::SeqCst));
}

#[test]
fn test_owned_decode_failure_drops_data_before_refunding_receipt() {
    let (snapshot, released) = credited();
    let snapshot = Arc::try_unwrap(snapshot).ok().unwrap();
    let result = snapshot.map(|_| {
        Err(CalcFlowError::Cancelled {
            run_id: "restore-copy".into(),
        })
    });
    assert!(matches!(result, Err(CalcFlowError::Cancelled { run_id }) if run_id == "restore-copy"));
    assert!(released.load(Ordering::SeqCst));
}

#[test]
fn test_legacy_snapshot_mutation_keeps_shared_original_unchanged() {
    let mut snapshot = Arc::new(PreloadedSnapshot::uncredited(
        OperatorStateSnapshot::default(),
    ));
    let original = Arc::clone(&snapshot);
    PreloadedSnapshot::legacy_mut(&mut snapshot)
        .unwrap()
        .inline_metadata
        .insert("legacy".into(), true.into());
    assert!(original.inline_metadata.is_empty());
    assert_eq!(snapshot.inline_metadata["legacy"], true);
    let mut copy = snapshot.clone_legacy().unwrap();
    copy.inline_metadata.clear();
    assert!(!snapshot.inline_metadata.is_empty());
    let taken = PreloadedSnapshot::take_legacy(&mut snapshot).unwrap();
    assert_eq!(taken.inline_metadata["legacy"], true);
    assert!(snapshot.inline_metadata.is_empty());
    assert!(original.inline_metadata.is_empty());
}

#[test]
fn test_snapshot_descriptor_quote_covers_its_actual_arc_allocation() {
    let mut owned = None;
    let measured = allocation_counter::measure(|| {
        let mut snapshot = PreloadedSnapshot::uncredited(OperatorStateSnapshot::default());
        snapshot.snapshot_mut().inline_metadata.clear();
        owned = Some(Arc::new(snapshot));
    });
    assert!(
        u64::try_from(measured.bytes_current).unwrap() <= PreloadedSnapshot::allocation_bytes()
    );
    drop(owned);
}

#[test]
fn test_credited_snapshots_reject_unfunded_deep_copy_and_legacy_mutation() {
    let (mut snapshot, released) = credited();
    assert!(snapshot.clone_legacy().is_err());
    assert!(PreloadedSnapshot::legacy_mut(&mut snapshot).is_err());
    assert!(!released.load(Ordering::SeqCst));
    drop(snapshot);
    assert!(released.load(Ordering::SeqCst));
}

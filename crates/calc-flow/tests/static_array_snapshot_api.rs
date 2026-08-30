//! External-crate compatibility witnesses for the SCE-13 snapshot surface.

use std::{
    any::Any,
    panic::{RefUnwindSafe, UnwindSafe},
    sync::Arc,
};

use calc_flow::{Batch, BatchMetadata, ExternalPayload, StaticArraySnapshot, StaticArrayValues};

fn assert_auto_traits<T: Send + Sync + Unpin + UnwindSafe + RefUnwindSafe>() {}

fn snapshot(batch: &Batch) -> Option<StaticArraySnapshot> {
    batch.static_array_snapshot()
}

#[test]
fn public_snapshot_is_owned_read_only_and_keeps_the_frozen_auto_traits() {
    assert_auto_traits::<StaticArraySnapshot>();
    assert_auto_traits::<StaticArrayValues>();

    let batch = Batch::static_array_float(
        "numpy",
        "float32",
        vec![2, 1],
        Some(vec![false, false]),
        vec![0.25, 0.75],
    )
    .unwrap();
    let snapshot = snapshot(&batch).expect("engine-latched array");
    drop(batch);

    assert_eq!(snapshot.backend(), "numpy");
    assert_eq!(snapshot.dtype(), "float32");
    assert_eq!(snapshot.shape(), &[2, 1]);
    assert_eq!(snapshot.nulls(), Some([false, false].as_slice()));
    #[allow(
        unreachable_patterns,
        reason = "the public non-exhaustive enum requires a future-proof wildcard"
    )]
    match snapshot.values() {
        StaticArrayValues::Float(values) => assert_eq!(values, &[0.25, 0.75]),
        _ => panic!("unexpected or future static-array carrier"),
    }
}

#[derive(Debug)]
struct GeneralArray;

impl ExternalPayload for GeneralArray {
    fn backend(&self) -> &'static str {
        "numpy"
    }

    fn len(&self) -> usize {
        1
    }

    fn estimated_bytes(&self) -> usize {
        1
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[test]
fn public_snapshot_probe_rejects_non_latched_external_arrays() {
    let batch = Batch::external(Arc::new(GeneralArray), BatchMetadata::default()).unwrap();
    assert!(snapshot(&batch).is_none());
}

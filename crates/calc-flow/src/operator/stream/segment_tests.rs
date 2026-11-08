use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn test_sequence_controlled_segment_hash_preserves_digest_and_checks_each_chunk() {
    let bytes = vec![7_u8; 8193];
    let expected = StateSegment::new(bytes.clone());
    let actual = StateSegment::new_controlled(bytes.clone(), &mut || Ok(())).unwrap();
    assert_eq!(actual, expected);
    let mut checks = 0;
    let result = StateSegment::new_controlled(bytes, &mut || {
        checks += 1;
        if checks == 3 {
            Err(CalcFlowError::Cancelled {
                run_id: "segment-hash".into(),
            })
        } else {
            Ok(())
        }
    });
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(checks, 3);
}

struct Owner(Arc<AtomicUsize>);

impl Drop for Owner {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn test_sequence_segment_allocation_credit_lives_until_final_snapshot_clone() {
    let released = Arc::new(AtomicUsize::new(0));
    let segment = StateSegment::new(vec![1, 2, 3]).with_owner(Arc::new(Owner(released.clone())));
    let clone = segment.clone();
    let last = clone.clone();
    drop(segment);
    drop(clone);
    assert_eq!(released.load(Ordering::SeqCst), 0);
    drop(last);
    assert_eq!(released.load(Ordering::SeqCst), 1);
}

#[test]
fn test_sequence_segment_charge_counts_retained_payload_and_shared_digest_capacity() {
    let mut bytes = Vec::with_capacity(257);
    bytes.extend_from_slice(&[1, 2, 3]);
    let mut checksum = String::with_capacity(128);
    checksum.push_str(StateSegment::new(bytes.clone()).sha256());
    let expected = bytes.capacity() + checksum.capacity();
    let segment = StateSegment::from_validated(bytes, checksum);
    assert_eq!(segment.retained_buffer_bytes(), Some(expected));
    let cloned = segment.clone();
    assert!(std::ptr::eq(
        segment.sha256().as_ptr(),
        cloned.sha256().as_ptr()
    ));
    assert_eq!(cloned.retained_buffer_bytes(), Some(expected));
}

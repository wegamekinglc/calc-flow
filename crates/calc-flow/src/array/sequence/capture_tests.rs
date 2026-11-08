use super::*;
use super::{
    capture::Captures,
    codec::{KernelOptions, SequenceSpec, SequenceState},
};
use crate::{StateSegment, array::EwmaOptions};

fn fixture() -> (SequenceSpec, SequenceState) {
    let spec = SequenceSpec::new(
        ArrayDType::Float64,
        KernelOptions::Ewma(EwmaOptions::new(3).unwrap()),
    )
    .unwrap();
    let state = SequenceState::new(&spec, ArrayStateOptions::default()).unwrap();
    (spec, state)
}

#[test]
fn test_sequence_capture_credits_share_until_final_segment_owner_drops() {
    let (spec, state) = fixture();
    let execution = ArrayExecutionOptions::default();
    let captures = Captures::default();
    let segment = captures
        .capture(&spec, &state, &execution, &mut || Ok(()))
        .unwrap();
    let charged = segment.retained_buffer_bytes().unwrap();
    assert_eq!(captures.bytes(), charged);
    let clone = captures
        .retain(segment.clone(), &execution, state.state_bytes())
        .unwrap();
    assert_eq!(captures.bytes(), charged);
    drop(segment);
    assert_eq!(captures.bytes(), charged);
    drop(clone);
    assert_eq!(captures.bytes(), 0);
}

#[test]
fn test_sequence_capture_peak_counts_old_captures_and_hash_scratch_before_allocating() {
    let (spec, state) = fixture();
    let execution = ArrayExecutionOptions::default().with_max_elements(0);
    let captures = Captures::default();
    let first = captures
        .capture(&spec, &state, &execution, &mut || Ok(()))
        .unwrap();
    let old = captures.bytes();
    let peak = state.state_bytes() + old + 92 + 64 + StateSegment::controlled_scratch_bytes();
    assert!(
        captures
            .capture(
                &spec,
                &state,
                &execution.with_max_allocation_bytes(peak - 1),
                &mut || Ok(())
            )
            .is_err()
    );
    assert_eq!(captures.bytes(), old);
    let second = captures
        .capture(
            &spec,
            &state,
            &execution.with_max_allocation_bytes(peak),
            &mut || Ok(()),
        )
        .unwrap();
    assert_eq!(
        captures.bytes(),
        old + second.retained_buffer_bytes().unwrap()
    );
    drop(first);
    drop(second);
    assert_eq!(captures.bytes(), 0);
}

#[test]
fn test_sequence_capture_cancelled_hash_releases_reservation_for_retry() {
    let (spec, state) = fixture();
    let execution = ArrayExecutionOptions::default();
    let captures = Captures::default();
    let mut count = 0;
    let result = captures.capture(&spec, &state, &execution, &mut || {
        count += 1;
        if count == 5 {
            Err(crate::CalcFlowError::Cancelled {
                run_id: "capture".into(),
            })
        } else {
            Ok(())
        }
    });
    assert!(matches!(
        result,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(captures.bytes(), 0);
    let segment = captures
        .capture(&spec, &state, &execution, &mut || Ok(()))
        .unwrap();
    assert_eq!(captures.bytes(), segment.retained_buffer_bytes().unwrap());
}

#[test]
fn test_sequence_capture_retains_a_preexisting_preload_owner() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct PreviousOwner(std::sync::Arc<AtomicUsize>);
    impl Drop for PreviousOwner {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let dropped = std::sync::Arc::new(AtomicUsize::new(0));
    let segment = StateSegment::new(vec![1, 2, 3])
        .with_owner(std::sync::Arc::new(PreviousOwner(dropped.clone())));
    let captures = Captures::default();
    let retained = captures
        .retain(segment, &ArrayExecutionOptions::default(), 24)
        .unwrap();
    let last = retained.clone();
    drop(retained);
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    assert!(captures.bytes() > 0);
    drop(last);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(captures.bytes(), 0);
}

#[test]
fn test_sequence_capture_deduplicates_clones_created_before_owner_attachment() {
    let raw = StateSegment::new(vec![1, 2, 3]);
    let expected = raw.retained_buffer_bytes().unwrap();
    let captures = Captures::default();
    let execution = ArrayExecutionOptions::default();
    let first = captures.retain(raw.clone(), &execution, 24).unwrap();
    let second = captures.retain(raw, &execution, 24).unwrap();
    assert_eq!(captures.bytes(), expected);
    drop(first);
    assert_eq!(captures.bytes(), expected);
    drop(second);
    assert_eq!(captures.bytes(), 0);
}

#[test]
fn test_sequence_capture_deduplicates_an_owner_hidden_by_another_owner() {
    let captures = Captures::default();
    let execution = ArrayExecutionOptions::default();
    let first = captures
        .retain(StateSegment::new(vec![4, 5]), &execution, 24)
        .unwrap();
    let expected = captures.bytes();
    let wrapped = first
        .clone()
        .with_owner(std::sync::Arc::new("outer preload owner"));
    let second = captures.retain(wrapped, &execution, 24).unwrap();
    assert_eq!(captures.bytes(), expected);
    drop(first);
    assert_eq!(captures.bytes(), expected);
    drop(second);
    assert_eq!(captures.bytes(), 0);
}

use super::codec::{KernelOptions, RestoreLimits, SequenceSpec, SequenceState, decode, encode};
use super::*;
use crate::array::{EwmaOptions, FloatValues, FloatView};

fn view(values: &[f64]) -> ArrayView<'_> {
    ArrayView::from_float(FloatView::from_f64(values))
}

fn step(state: &mut SequenceState, input: ArrayView<'_>) -> Result<NativeArray> {
    let prepared =
        state.prepare_checked(input, &ArrayExecutionOptions::default(), 0, &mut || Ok(()))?;
    if let Some(candidate) = prepared.candidate {
        *state = candidate;
    }
    Ok(prepared.output)
}

#[test]
fn test_sequence_mean_codec_retains_logical_history_and_raw_m2_without_refolding() {
    use crate::{RollingNumericalProfile, array::RollingMeanOptions};
    let spec = SequenceSpec::new(
        ArrayDType::Float32,
        KernelOptions::Mean(
            RollingMeanOptions::new(3)
                .unwrap()
                .with_numerical_profile(RollingNumericalProfile::StableV2Preview),
        ),
    )
    .unwrap();
    let mut state = SequenceState::new(&spec, ArrayStateOptions::default()).unwrap();
    let mask = NullBuffer::from(vec![true, false, true, true]);
    let input = FloatView::try_new(
        FloatValues::Float32(&[1.0, 222.0, -0.0, f32::NAN]),
        Some(&mask),
    )
    .unwrap();
    step(&mut state, ArrayView::from_float(input)).unwrap();
    let execution = ArrayExecutionOptions::default();
    let bytes = encode(&spec, &state, &execution, 0, &mut || Ok(())).unwrap();
    assert_eq!(bytes.len(), 167);
    assert_eq!(&bytes[16..20], &[1, 1, 2, 0]);
    assert_eq!(&bytes[68..76], &4_u64.to_le_bytes());
    assert_eq!(&bytes[76..84], &1_u64.to_le_bytes());
    assert_eq!(&bytes[132..140], &3_u64.to_le_bytes());
    assert_eq!(&bytes[140..149], &[0; 9]);
    assert_eq!(bytes[149], 1);
    assert_eq!(&bytes[150..158], &(-0.0_f64).to_bits().to_le_bytes());
    assert_eq!(&bytes[158..167], &[0; 9]);
    let mut restored = restore(&spec, &bytes, &execution, state.state_bytes()).unwrap();
    assert_eq!(
        bits(&step(&mut restored, view(&[6.0, 8.0])).unwrap()),
        bits(&step(&mut state, view(&[6.0, 8.0])).unwrap())
    );
    let mut raw = bytes.clone();
    raw[100..108].copy_from_slice(&0x7ff8_0000_0000_0055_u64.to_le_bytes());
    let restored = restore(&spec, &raw, &execution, 64).unwrap();
    assert_eq!(
        encode(&spec, &restored, &execution, 0, &mut || Ok(())).unwrap(),
        raw
    );
}

fn bits(output: &NativeArray) -> Vec<Option<u64>> {
    let view = output.float_view().unwrap();
    let FloatValues::Float64(values) = view.values() else {
        panic!("expected f64")
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            view.validity()
                .is_none_or(|mask| mask.is_valid(index))
                .then(|| value.to_bits())
        })
        .collect()
}

#[test]
fn test_sequence_ewma_codec_has_exact_layout_and_preserves_raw_seed_bits() {
    let spec = SequenceSpec::new(
        ArrayDType::Float64,
        KernelOptions::Ewma(EwmaOptions::new(3).unwrap()),
    )
    .unwrap();
    let mut state = SequenceState::new(&spec, ArrayStateOptions::default()).unwrap();
    step(&mut state, view(&[-0.0])).unwrap();
    let execution = ArrayExecutionOptions::default();
    let bytes = encode(&spec, &state, &execution, 0, &mut || Ok(())).unwrap();
    assert_eq!(bytes.len(), 92);
    assert_eq!(&bytes[..8], b"CFASEQ\0\0");
    assert_eq!(&bytes[8..20], &[1, 0, 0, 0, 1, 0, 0, 0, 2, 2, 0, 0]);
    assert_eq!(&bytes[20..52], spec.fingerprint());
    assert_eq!(&bytes[52..60], &3_u64.to_le_bytes());
    assert_eq!(&bytes[60..68], &1_u64.to_le_bytes());
    assert_eq!(&bytes[68..76], &1_u64.to_le_bytes());
    assert_eq!(&bytes[76..84], &1_u64.to_le_bytes());
    assert_eq!(&bytes[84..92], &(-0.0_f64).to_bits().to_le_bytes());
    let mut restored = restore(&spec, &bytes, &execution, 24).unwrap();
    assert_eq!(
        bits(&step(&mut restored, view(&[2.0, 4.0])).unwrap()),
        bits(&step(&mut state, view(&[2.0, 4.0])).unwrap())
    );
}

fn restore(
    spec: &SequenceSpec,
    bytes: &[u8],
    execution: &ArrayExecutionOptions,
    old: usize,
) -> Result<SequenceState> {
    let limits = RestoreLimits {
        state: ArrayStateOptions::default(),
        execution,
        old,
        captures: 0,
        encoded: bytes.len(),
    };
    decode(spec, bytes, &limits, &mut || Ok(()))
}

fn ewma_fixture() -> (SequenceSpec, SequenceState, Vec<u8>) {
    let spec = SequenceSpec::new(
        ArrayDType::Float64,
        KernelOptions::Ewma(EwmaOptions::new(3).unwrap()),
    )
    .unwrap();
    let mut state = SequenceState::new(&spec, ArrayStateOptions::default()).unwrap();
    step(&mut state, view(&[-0.0])).unwrap();
    let bytes = encode(
        &spec,
        &state,
        &ArrayExecutionOptions::default(),
        0,
        &mut || Ok(()),
    )
    .unwrap();
    (spec, state, bytes)
}

#[test]
fn test_sequence_codec_rejects_undercharged_encoded_input_before_restore() {
    let (spec, _, bytes) = ewma_fixture();
    let execution = ArrayExecutionOptions::default();
    let limits = RestoreLimits {
        state: ArrayStateOptions::default(),
        execution: &execution,
        old: 24,
        captures: 0,
        encoded: bytes.len() - 1,
    };
    assert!(matches!(decode(&spec, &bytes, &limits, &mut || Ok(())),
        Err(crate::CalcFlowError::CheckpointMismatch { message }) if message.contains("encoded_bytes")));
}

#[test]
fn test_sequence_codec_budgets_count_overlapping_state_bytes_and_ignore_output_elements() {
    let (spec, state, bytes) = ewma_fixture();
    let execution = ArrayExecutionOptions::default().with_max_elements(0);
    let peak = 24 + 92 + 13 + size_of::<u64>();
    assert!(
        encode(
            &spec,
            &state,
            &execution.with_max_allocation_bytes(peak - 1),
            13,
            &mut || Ok(())
        )
        .is_err()
    );
    assert!(
        encode(
            &spec,
            &state,
            &execution.with_max_allocation_bytes(peak),
            13,
            &mut || Ok(())
        )
        .is_ok()
    );
    let peak = 24 + 24 + bytes.len() + 13 + 32;
    let tight = execution.with_max_allocation_bytes(peak - 1);
    let limits = RestoreLimits {
        state: ArrayStateOptions::default(),
        execution: &tight,
        old: 24,
        captures: 13,
        encoded: bytes.len(),
    };
    assert!(decode(&spec, &bytes, &limits, &mut || Ok(())).is_err());
    let exact = execution.with_max_allocation_bytes(peak);
    let limits = RestoreLimits {
        execution: &exact,
        ..limits
    };
    assert!(decode(&spec, &bytes, &limits, &mut || Ok(())).is_ok());
    let limits = RestoreLimits {
        state: ArrayStateOptions::default().with_max_state_bytes(23),
        ..limits
    };
    assert!(
        matches!(decode(&spec, &bytes, &limits, &mut || Ok(())), Err(crate::CalcFlowError::CheckpointMismatch { message }) if message.contains("max_state_bytes"))
    );
}

#[test]
fn test_sequence_codec_rejects_corrupt_headers_counts_and_trailing_bytes() {
    let (spec, _, bytes) = ewma_fixture();
    let execution = ArrayExecutionOptions::default();
    for index in [0, 8, 12, 16, 17, 18, 19, 20, 52, 60] {
        let mut corrupt = bytes.clone();
        corrupt[index] ^= 0x7f;
        assert!(
            restore(&spec, &corrupt, &execution, 24).is_err(),
            "header byte {index}"
        );
    }
    let mut corrupt = bytes.clone();
    corrupt[76..84].copy_from_slice(&2_u64.to_le_bytes());
    assert!(restore(&spec, &corrupt, &execution, 24).is_err());
    corrupt[76..84].copy_from_slice(&0_u64.to_le_bytes());
    assert!(restore(&spec, &corrupt, &execution, 24).is_err());
    assert!(restore(&spec, &bytes[..91], &execution, 24).is_err());
    let mut extra = bytes;
    extra.push(0);
    assert!(restore(&spec, &extra, &execution, 24).is_err());
}

#[test]
fn test_sequence_mean_codec_rejects_inconsistent_history_and_summary_counts() {
    use crate::array::RollingMeanOptions;
    let spec = SequenceSpec::new(
        ArrayDType::Float64,
        KernelOptions::Mean(RollingMeanOptions::new(3).unwrap()),
    )
    .unwrap();
    let mut state = SequenceState::new(&spec, ArrayStateOptions::default()).unwrap();
    step(&mut state, view(&[1.0, f64::NAN, f64::INFINITY])).unwrap();
    let execution = ArrayExecutionOptions::default();
    let bytes = encode(&spec, &state, &execution, 0, &mut || Ok(())).unwrap();
    for index in [76, 108, 116, 124, 132] {
        let mut corrupt = bytes.clone();
        corrupt[index] ^= 1;
        assert!(
            restore(&spec, &corrupt, &execution, state.state_bytes()).is_err(),
            "mean counter {index}"
        );
    }
    let mut corrupt = bytes.clone();
    corrupt[140] = 2;
    assert!(restore(&spec, &corrupt, &execution, state.state_bytes()).is_err());
    let mut corrupt = bytes.clone();
    corrupt[150] = 1;
    assert!(restore(&spec, &corrupt, &execution, state.state_bytes()).is_err());
    let mut corrupt = bytes;
    corrupt[141..149].copy_from_slice(&f64::NAN.to_bits().to_le_bytes());
    assert!(restore(&spec, &corrupt, &execution, state.state_bytes()).is_err());
}

#[test]
fn test_sequence_codec_checks_cancellation_during_history_encoding_and_decode_scans() {
    use crate::array::RollingMeanOptions;
    let spec = SequenceSpec::new(
        ArrayDType::Float64,
        KernelOptions::Mean(RollingMeanOptions::new(u64::MAX).unwrap()),
    )
    .unwrap();
    let mut state = SequenceState::new(&spec, ArrayStateOptions::default()).unwrap();
    step(&mut state, view(&vec![1.0; 8193])).unwrap();
    let execution = ArrayExecutionOptions::default();
    let error = encode(&spec, &state, &execution, 0, &mut cancel_at(3));
    assert!(matches!(error, Err(crate::CalcFlowError::Cancelled { .. })));
    let bytes = encode(&spec, &state, &execution, 0, &mut || Ok(())).unwrap();
    let limits = RestoreLimits {
        state: ArrayStateOptions::default(),
        execution: &execution,
        old: state.state_bytes(),
        captures: 0,
        encoded: bytes.capacity(),
    };
    for stop in [3, 6] {
        assert!(matches!(
            decode(&spec, &bytes, &limits, &mut cancel_at(stop)),
            Err(crate::CalcFlowError::Cancelled { .. })
        ));
    }
    let mut restored = decode(&spec, &bytes, &limits, &mut || Ok(())).unwrap();
    assert_eq!(
        bits(&step(&mut restored, view(&[3.0])).unwrap()),
        bits(&step(&mut state, view(&[3.0])).unwrap())
    );
}

fn cancel_at(stop: usize) -> impl FnMut() -> Result<()> {
    let mut checks = 0;
    move || {
        checks += 1;
        if checks == stop {
            Err(crate::CalcFlowError::Cancelled {
                run_id: "codec".into(),
            })
        } else {
            Ok(())
        }
    }
}

#[test]
fn test_sequence_fingerprint_matches_independent_canonical_payload_sha256() {
    use sha2::{Digest as _, Sha256};
    let payload = r#"{"identity":"native:array_sequence@1","input_dtype":"float64","kernel":{"function":"ewma","min_periods":1,"span":3},"layout":1,"numeric_semantics":1}"#;
    let expected: [u8; 32] = Sha256::digest(payload.as_bytes()).into();
    let (spec, _, _) = ewma_fixture();
    assert_eq!(spec.fingerprint(), &expected);
    assert_eq!(
        hex::encode(expected),
        "c95a2b00976e4d9c0459e57dad55ced7b0dd7d7a5181bbb3644e4b2522565736"
    );
}

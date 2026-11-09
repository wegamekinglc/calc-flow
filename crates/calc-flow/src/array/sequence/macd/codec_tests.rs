use super::*;
use crate::{
    array::sequence::codec::{self, KernelOptions, RestoreLimits, SequenceSpec, SequenceState},
    array::{ArrayDType, FloatView},
};

fn spec() -> SequenceSpec {
    SequenceSpec::new(
        ArrayDType::Float64,
        KernelOptions::Macd(MacdOptions::new(1, 3, 3).unwrap()),
    )
    .unwrap()
}
fn view(values: &[f64]) -> ArrayView<'_> {
    ArrayView::from_float(FloatView::from_f64(values))
}
fn fixture() -> (SequenceSpec, SequenceState, Vec<u8>) {
    let spec = spec();
    let mut state = SequenceState::new(&spec, ArrayStateOptions::default()).unwrap();
    step(&mut state, view(&[10.0, 14.0])).unwrap();
    let bytes = codec::encode(
        &spec,
        &state,
        &ArrayExecutionOptions::default(),
        0,
        &mut || Ok(()),
    )
    .unwrap();
    (spec, state, bytes)
}
fn step(state: &mut SequenceState, input: ArrayView<'_>) -> Result<crate::array::NativeArray> {
    let prepared =
        state.prepare_checked(input, &ArrayExecutionOptions::default(), 0, &mut || Ok(()))?;
    if let Some(next) = prepared.candidate {
        *state = next;
    }
    Ok(prepared.output)
}
fn decode(spec: &SequenceSpec, bytes: &[u8], maximum: usize) -> Result<SequenceState> {
    let execution = ArrayExecutionOptions::default()
        .with_max_elements(0)
        .with_max_allocation_bytes(maximum);
    codec::decode(
        spec,
        bytes,
        &RestoreLimits {
            state: ArrayStateOptions::default(),
            execution: &execution,
            old: 56,
            captures: 13,
            encoded: bytes.len(),
        },
        &mut || Ok(()),
    )
}

#[test]
fn test_macd_layout_three_header_parameters_and_payload_are_exact() {
    let (spec, _, bytes) = fixture();
    assert_eq!(bytes.len(), 156);
    assert_eq!(&bytes[..8], b"CFAIND\0\0");
    assert_eq!(&bytes[8..20], &[3, 0, 0, 0, 1, 0, 0, 0, 5, 2, 0, 0]);
    assert_eq!(&bytes[20..52], spec.fingerprint());
    assert_eq!(&bytes[52..60], &2_u64.to_le_bytes());
    assert_eq!(&bytes[60..68], &[40, 0, 0, 0, 48, 0, 0, 0]);
    for (offset, value) in [
        (68, 1_u64),
        (76, 3),
        (84, 3),
        (92, 1),
        (100, 1),
        (108, 2),
        (124, 2),
        (140, 2),
    ] {
        assert_eq!(&bytes[offset..offset + 8], &value.to_le_bytes());
    }
    for (offset, value) in [(116, 14.0_f64), (132, 12.0), (148, 1.0)] {
        assert_eq!(&bytes[offset..offset + 8], &value.to_bits().to_le_bytes());
    }
}
#[test]
fn test_macd_codec_preserves_active_raw_nonfinite_and_signed_zero_values() {
    let (spec, _, mut bytes) = fixture();
    for (offset, value) in [
        (116, (-0.0_f64).to_bits()),
        (132, f64::INFINITY.to_bits()),
        (148, 0x7ff8_0000_0000_0042),
    ] {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
    let state = decode(&spec, &bytes, usize::MAX).unwrap();
    assert_eq!(
        codec::encode(
            &spec,
            &state,
            &ArrayExecutionOptions::default(),
            0,
            &mut || Ok(())
        )
        .unwrap(),
        bytes
    );
}
#[test]
fn test_macd_codec_rejects_headers_lengths_zero_count_bits_and_incoherent_counts() {
    let (spec, _, bytes) = fixture();
    for offset in [
        0, 8, 12, 16, 17, 18, 19, 20, 60, 64, 68, 76, 84, 92, 100, 108, 124, 140,
    ] {
        let mut bad = bytes.clone();
        bad[offset] ^= 0x7f;
        assert!(decode(&spec, &bad, usize::MAX).is_err(), "offset {offset}");
    }
    let mut bad = bytes.clone();
    for offset in [108, 124, 140] {
        bad[offset..offset + 8].copy_from_slice(&0_u64.to_le_bytes());
    }
    bad[116..124].copy_from_slice(&(-0.0_f64).to_bits().to_le_bytes());
    assert!(decode(&spec, &bad, usize::MAX).is_err());
    assert!(decode(&spec, &bytes[..155], usize::MAX).is_err());
    let mut trailing = bytes;
    trailing.push(0);
    assert!(decode(&spec, &trailing, usize::MAX).is_err());
}
#[test]
fn test_macd_codec_restores_signal_count_greater_than_price_count() {
    let (spec, _, mut bytes) = fixture();
    bytes[52..60].copy_from_slice(&4_u64.to_le_bytes());
    bytes[140..148].copy_from_slice(&4_u64.to_le_bytes());
    bytes[148..156].copy_from_slice(&1.75_f64.to_bits().to_le_bytes());
    let mut state = decode(&spec, &bytes, usize::MAX).unwrap();
    step(&mut state, view(&[f64::NAN])).unwrap();
    let next = codec::encode(
        &spec,
        &state,
        &ArrayExecutionOptions::default(),
        0,
        &mut || Ok(()),
    )
    .unwrap();
    assert_eq!(&next[108..116], &2_u64.to_le_bytes());
    assert_eq!(&next[140..148], &5_u64.to_le_bytes());
    assert_eq!(&next[148..156], &1.875_f64.to_bits().to_le_bytes());
}
#[test]
fn test_macd_encoding_and_restore_admit_exact_participating_peak() {
    let (spec, state, bytes) = fixture();
    let execution = ArrayExecutionOptions::default().with_max_elements(0);
    let encode_peak = 56 + 156 + 13 + size_of::<u64>();
    assert!(
        codec::encode(
            &spec,
            &state,
            &execution.with_max_allocation_bytes(encode_peak - 1),
            13,
            &mut || Ok(())
        )
        .is_err()
    );
    codec::encode(
        &spec,
        &state,
        &execution.with_max_allocation_bytes(encode_peak),
        13,
        &mut || Ok(()),
    )
    .unwrap();
    let restore_peak = 56 + 56 + 156 + 13 + 32;
    assert!(decode(&spec, &bytes, restore_peak - 1).is_err());
    decode(&spec, &bytes, restore_peak).unwrap();
}

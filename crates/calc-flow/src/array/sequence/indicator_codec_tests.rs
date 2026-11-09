use super::{
    ArrayStateOptions,
    codec::{self, KernelOptions, RestoreLimits, SequenceSpec, SequenceState},
};
use crate::{
    Result,
    array::{
        ArrayDType, ArrayExecutionOptions, ArrayView, AtrOptions, FloatView, NativeArray,
        RsiOptions,
    },
};

fn view(values: &[f64]) -> ArrayView<'_> {
    ArrayView::from_float(FloatView::from_f64(values))
}

fn configured(atr: bool) -> SequenceSpec {
    let kernel = if atr {
        KernelOptions::Atr(AtrOptions::new(1).unwrap())
    } else {
        KernelOptions::Rsi(RsiOptions::new(2).unwrap())
    };
    SequenceSpec::new(ArrayDType::Float64, kernel).unwrap()
}

fn step(state: &mut SequenceState, input: ArrayView<'_>) -> Result<NativeArray> {
    let prepared =
        state.prepare_checked(input, &ArrayExecutionOptions::default(), 0, &mut || Ok(()))?;
    if let Some(candidate) = prepared.candidate {
        *state = candidate;
    }
    Ok(prepared.output)
}

fn encoded(atr: bool) -> (SequenceSpec, SequenceState, Vec<u8>) {
    let spec = configured(atr);
    let mut state = SequenceState::new(&spec, ArrayStateOptions::default()).unwrap();
    let values = if atr {
        vec![10.0, 8.0, 9.0, 12.0, 9.0, 10.0]
    } else {
        vec![-0.0, 1.0, 0.0]
    };
    let input = if atr {
        view(&values).reshape(&[2, 3]).unwrap()
    } else {
        view(&values)
    };
    step(&mut state, input).unwrap();
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

fn restore(
    spec: &SequenceSpec,
    bytes: &[u8],
    execution: &ArrayExecutionOptions,
    captures: usize,
) -> Result<SequenceState> {
    codec::decode(
        spec,
        bytes,
        &RestoreLimits {
            state: ArrayStateOptions::default(),
            execution,
            old: 48,
            captures,
            encoded: bytes.len(),
        },
        &mut || Ok(()),
    )
}

#[test]
fn test_indicator_codec_layout_two_is_exact_and_continues_raw_state() {
    for is_atr in [false, true] {
        let (spec, mut state, bytes) = encoded(is_atr);
        assert_eq!(bytes.len(), if is_atr { 108 } else { 116 });
        assert_eq!(&bytes[..8], b"CFAIND\0\0");
        assert_eq!(&bytes[8..16], &[2, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(&bytes[16..20], &[if is_atr { 4 } else { 3 }, 2, 0, 0]);
        assert_eq!(&bytes[20..52], spec.fingerprint());
        assert_eq!(&bytes[60..64], &8_u32.to_le_bytes());
        assert_eq!(
            &bytes[64..68],
            &(if is_atr { 32_u32 } else { 40 }).to_le_bytes()
        );
        let mut restored = restore(&spec, &bytes, &ArrayExecutionOptions::default(), 0).unwrap();
        assert_eq!(
            codec::encode(
                &spec,
                &restored,
                &ArrayExecutionOptions::default(),
                0,
                &mut || Ok(())
            )
            .unwrap(),
            bytes
        );
        let next = if is_atr {
            view(&[14.0, 11.0, 13.0]).reshape(&[1, 3]).unwrap()
        } else {
            view(&[2.0])
        };
        assert_eq!(
            bits(&step(&mut state, next).unwrap()),
            bits(&step(&mut restored, next).unwrap())
        );
    }
}

fn bits(array: &NativeArray) -> Vec<Option<u64>> {
    let view = array.float_view().unwrap();
    let crate::array::FloatValues::Float64(values) = view.values() else {
        panic!("expected f64");
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            view.validity()
                .unwrap()
                .is_valid(index)
                .then(|| value.to_bits())
        })
        .collect()
}

#[test]
fn test_indicator_codec_rejects_corruption_and_preserves_reset_inventory() {
    let (spec, state, bytes) = encoded(false);
    for offset in [0, 8, 12, 16, 17, 18, 19, 20, 60, 64, 68, 76, 77] {
        let mut bad = bytes.clone();
        bad[offset] ^= 0x7f;
        assert!(
            restore(&spec, &bad, &ArrayExecutionOptions::default(), 0).is_err(),
            "offset {offset}"
        );
    }
    for (offset, value) in [
        (84, 3_u64),
        (100, f64::INFINITY.to_bits()),
        (108, (-1.0_f64).to_bits()),
    ] {
        let mut bad = bytes.clone();
        bad[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        assert!(restore(&spec, &bad, &ArrayExecutionOptions::default(), 0).is_err());
    }
    let mut state = state;
    step(&mut state, view(&[f64::NAN])).unwrap();
    let reset = codec::encode(
        &spec,
        &state,
        &ArrayExecutionOptions::default(),
        0,
        &mut || Ok(()),
    )
    .unwrap();
    assert_eq!(&reset[52..60], &4_u64.to_le_bytes());
    assert_eq!(&reset[76..], &[0_u8; 40]);
    restore(&spec, &reset, &ArrayExecutionOptions::default(), 0).unwrap();
    assert!(restore(&spec, &bytes[..115], &ArrayExecutionOptions::default(), 0).is_err());
    let mut trailing = bytes;
    trailing.push(0);
    assert!(restore(&spec, &trailing, &ArrayExecutionOptions::default(), 0).is_err());
}

#[test]
fn test_indicator_codec_uses_participating_limits_and_no_output_element_charge() {
    for atr in [false, true] {
        let (spec, state, bytes) = encoded(atr);
        let execution = ArrayExecutionOptions::default().with_max_elements(0);
        let peak = state.state_bytes() + bytes.len() + 13 + size_of::<u64>();
        assert!(
            codec::encode(
                &spec,
                &state,
                &execution.with_max_allocation_bytes(peak - 1),
                13,
                &mut || Ok(())
            )
            .is_err()
        );
        codec::encode(
            &spec,
            &state,
            &execution.with_max_allocation_bytes(peak),
            13,
            &mut || Ok(()),
        )
        .unwrap();
        let peak = state.state_bytes() * 2 + bytes.len() + 13 + 32;
        assert!(decode_with_limit(&spec, &state, &bytes, peak - 1).is_err());
        decode_with_limit(&spec, &state, &bytes, peak).unwrap();
    }
}

fn decode_with_limit(
    spec: &SequenceSpec,
    state: &SequenceState,
    bytes: &[u8],
    maximum: usize,
) -> Result<SequenceState> {
    let execution = ArrayExecutionOptions::default()
        .with_max_elements(0)
        .with_max_allocation_bytes(maximum);
    codec::decode(
        spec,
        bytes,
        &RestoreLimits {
            state: ArrayStateOptions::default(),
            execution: &execution,
            old: state.state_bytes(),
            captures: 13,
            encoded: bytes.len(),
        },
        &mut || Ok(()),
    )
}

#[test]
fn test_indicator_restore_keeps_valid_raw_signed_zero_bits() {
    let execution = ArrayExecutionOptions::default();
    for atr in [false, true] {
        let (spec, _, mut bytes) = encoded(atr);
        for offset in [92, 100] {
            bytes[offset..offset + 8].copy_from_slice(&(-0.0_f64).to_bits().to_le_bytes());
        }
        let restored = restore(&spec, &bytes, &execution, 0).unwrap();
        assert_eq!(
            codec::encode(&spec, &restored, &execution, 0, &mut || Ok(())).unwrap(),
            bytes
        );
    }
}

#[test]
fn test_indicator_fingerprints_include_shape_rule_and_exact_normalized_parameters() {
    use sha2::{Digest as _, Sha256};
    let payloads = [
        r#"{"identity":"native:array_sequence@1","input_dtype":"float64","input_shape_rule":"rank_one","kernel":{"function":"rsi","period":2},"layout":2,"numeric_semantics":1}"#,
        r#"{"identity":"native:array_sequence@1","input_dtype":"float64","input_shape_rule":"packed_high_low_close","kernel":{"function":"atr","period":1},"layout":2,"numeric_semantics":1}"#,
    ];
    for (atr, payload) in [false, true].into_iter().zip(payloads) {
        let expected: [u8; 32] = Sha256::digest(payload.as_bytes()).into();
        assert_eq!(configured(atr).fingerprint(), &expected);
    }
}

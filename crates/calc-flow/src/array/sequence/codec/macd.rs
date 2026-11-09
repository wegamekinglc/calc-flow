use super::{KernelOptions, Reader, RestoreLimits, SequenceSpec, SequenceState, mismatch, word};
use crate::{
    CalcFlowError, Result,
    array::{MacdOptions, indicators::MacdKernel, sequence::MacdState},
};

const MAGIC: &[u8; 8] = b"CFAIND\0\0";

pub(super) fn document(options: MacdOptions) -> serde_json::Value {
    serde_json::json!({"function":"macd_components", "fast_span":options.fast_span(),
        "slow_span":options.slow_span(), "signal_span":options.signal_span(),
        "min_periods":options.min_periods(), "signal_min_periods":options.signal_min_periods()})
}
pub(super) fn requirements(length: usize) -> Result<(usize, usize)> {
    if length != 156 {
        return Err(mismatch(
            "encoded_bytes",
            "MACD layout 3 requires exactly 156 encoded bytes",
        ));
    }
    Ok((56, 32))
}
pub(super) fn encode(
    spec: &SequenceSpec,
    state: &SequenceState,
    mut bytes: Vec<u8>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<u8>> {
    let SequenceState::Macd(state) = state else {
        return Err(mismatch("kernel", "layout 3 requires MACD"));
    };
    invariants(state.kernel.validate(state.options))?;
    write_header(spec, state, &mut bytes);
    for value in parameters(state.options) {
        word(&mut bytes, value);
    }
    let kernel = &state.kernel;
    for value in [
        kernel.fast_count,
        kernel.fast_value.to_bits(),
        kernel.slow_count,
        kernel.slow_value.to_bits(),
        kernel.signal_count,
        kernel.signal_value.to_bits(),
    ] {
        word(&mut bytes, value);
    }
    check()?;
    Ok(bytes)
}
fn write_header(spec: &SequenceSpec, state: &MacdState, bytes: &mut Vec<u8>) {
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&spec.markers());
    bytes.extend_from_slice(spec.fingerprint());
    word(bytes, state.kernel.positions);
    bytes.extend_from_slice(&40_u32.to_le_bytes());
    bytes.extend_from_slice(&48_u32.to_le_bytes());
}
fn parameters(options: MacdOptions) -> [u64; 5] {
    [
        options.fast_span(),
        options.slow_span(),
        options.signal_span(),
        options.min_periods(),
        options.signal_min_periods(),
    ]
}
pub(super) fn read_header(spec: &SequenceSpec, reader: &mut Reader<'_>) -> Result<u64> {
    reader.require(MAGIC, "magic")?;
    reader.require(&3_u32.to_le_bytes(), "layout")?;
    reader.require(&1_u32.to_le_bytes(), "numeric_semantics")?;
    reader.require(&spec.markers(), "kernel/dtype/reserved")?;
    read_identity(spec, reader)
}
fn read_identity(spec: &SequenceSpec, reader: &mut Reader<'_>) -> Result<u64> {
    reader.require(spec.fingerprint(), "fingerprint")?;
    let positions = reader.u64()?;
    reader.require(&40_u32.to_le_bytes(), "parameter_bytes")?;
    reader.require(&48_u32.to_le_bytes(), "payload_bytes")?;
    let KernelOptions::Macd(options) = spec.kernel else {
        return Err(mismatch("kernel", "layout 3 requires MACD"));
    };
    read_parameters(options, reader)?;
    Ok(positions)
}
fn read_parameters(options: MacdOptions, reader: &mut Reader<'_>) -> Result<()> {
    for (value, field) in parameters(options).into_iter().zip([
        "fast_span",
        "slow_span",
        "signal_span",
        "min_periods",
        "signal_min_periods",
    ]) {
        reader.require(&value.to_le_bytes(), field)?;
    }
    Ok(())
}
pub(super) fn read_state(
    options: MacdOptions,
    positions: u64,
    mut reader: Reader<'_>,
    limits: &RestoreLimits<'_>,
) -> Result<SequenceState> {
    let (retained, scratch) = requirements(reader.bytes.len())?;
    limits.candidate(retained, scratch)?;
    let kernel = read_kernel(positions, &mut reader)?;
    invariants(kernel.validate(options))?;
    Ok(SequenceState::Macd(MacdState {
        options,
        limits: limits.state,
        kernel,
    }))
}
fn read_kernel(positions: u64, reader: &mut Reader<'_>) -> Result<MacdKernel> {
    Ok(MacdKernel {
        positions,
        fast_count: reader.u64()?,
        fast_value: f64::from_bits(reader.u64()?),
        slow_count: reader.u64()?,
        slow_value: f64::from_bits(reader.u64()?),
        signal_count: reader.u64()?,
        signal_value: f64::from_bits(reader.u64()?),
    })
}
fn invariants(result: Result<()>) -> Result<()> {
    result.map_err(|error| match error {
        CalcFlowError::InvalidArgument { field, message } => {
            mismatch(field.strip_prefix("state.").unwrap_or(&field), &message)
        }
        error => error,
    })
}
